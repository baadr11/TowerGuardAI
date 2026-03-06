"""
TowerGuard — FastAPI Backend
===================================
FastAPI + WebSocket backend for the TowerGuard Saudi Sovereign Digital Twin.
All threshold and configuration values are sourced exclusively from backend.config.
"""

import asyncio
import logging
import os
import time
import traceback
import json
import hashlib
import threading
from contextlib import asynccontextmanager
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from .config import (
    MODEL_CONFIG, EDGE_CONFIG, SEVERITY_CONFIG, ARABIC_LABELS,
    TOWERGUARD_FEATURES, TARGET_COL, get_version_info,
)
from .towerguard_ml import (
    ModelBundle, load_or_train, normalize_input,
    predict_proba, predict_with_confidence, predict_with_confidence_dict,
)
from .edge_inference import edge_predict, ESP32FirmwareGenerator

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("towerguard")

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "towerguard_model.pkl"

# ── Decision thresholds from config ───────────────────────────────────────
THRESHOLD_DANGER = SEVERITY_CONFIG.prob_critical
THRESHOLD_WARN   = SEVERITY_CONFIG.prob_degraded

# ── NOC Phone sourced from environment or config default ──────────────────
NOC_PHONE = EDGE_CONFIG.noc_phone

# ── Development / Production mode ─────────────────────────────────────────
_DEV_MODE = os.getenv("TOWERGUARD_DEV", "1") == "1"
_ALLOWED_ORIGINS = ["*"] if _DEV_MODE else [
    os.getenv("TOWERGUARD_ORIGIN", "http://127.0.0.1:8000"),
]

# ── Rate limiting ──────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["120/minute"])

# ── Model singleton ────────────────────────────────────────────────────────
_bundle: Optional[ModelBundle] = None
_bundle_lock = asyncio.Lock()

# ── Request counters ───────────────────────────────────────────────────────
_counter_lock = threading.Lock()
_request_counter: Dict[str, int] = defaultdict(int)

# ── TTL prediction cache ───────────────────────────────────────────────────
_TTL_CACHE: Dict[str, tuple] = {}
_CACHE_TTL_SECONDS = 30

def _cache_key(row: Dict[str, float]) -> str:
    raw = json.dumps(row, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def _cache_get(key: str):
    entry = _TTL_CACHE.get(key)
    if entry and (time.time() - entry[1]) <= _CACHE_TTL_SECONDS:
        return entry[0]
    _TTL_CACHE.pop(key, None)
    return None

def _cache_set(key: str, value):
    if len(_TTL_CACHE) >= 2000:
        oldest = sorted(_TTL_CACHE, key=lambda k: _TTL_CACHE[k][1])[:500]
        for k in oldest:
            _TTL_CACHE.pop(k, None)
    _TTL_CACHE[key] = (value, time.time())


# ══════════════════════════════════════════════════════════════════════════════
#  WebSocket Connection Manager
# ══════════════════════════════════════════════════════════════════════════════

class WSManager:
    """WebSocket connection manager — pushes real-time updates to all connected clients."""

    def __init__(self):
        self.clients: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.clients.append(ws)
        log.info("WebSocket client connected — total: %d", len(self.clients))

    def disconnect(self, ws: WebSocket):
        if ws in self.clients:
            self.clients.remove(ws)
        log.info("WebSocket client disconnected — remaining: %d", len(self.clients))

    async def broadcast(self, message: dict):
        """Push a message to all connected clients immediately."""
        dead = []
        for ws in self.clients:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def broadcast_alert(self, alert_type: str, tower_id: str, details: dict):
        """Push a connectivity or severity alert to all connected clients."""
        await self.broadcast({
            "type": "alert",
            "alert_type": alert_type,
            "tower_id": tower_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **details,
        })


ws_manager = WSManager()


# ══════════════════════════════════════════════════════════════════════════════
#  Heartbeat Watchdog — Dead Man's Switch
# ══════════════════════════════════════════════════════════════════════════════

# Per-tower heartbeat state registry
_last_heartbeat: Dict[str, Dict[str, Any]] = {}
_HEARTBEAT_TIMEOUT_SEC = 180  # 3-minute silence threshold before connectivity alert

_watchdog_running = False


async def _heartbeat_watchdog():
    """
    Heartbeat watchdog — Dead Man's Switch implementation.

    Polls every 30 seconds. If any tower has not transmitted a heartbeat
    within _HEARTBEAT_TIMEOUT_SEC seconds:
      1. Marks the tower as offline.
      2. Broadcasts a real-time connectivity alert via WebSocket.
      3. Logs a CRITICAL-level event for NOC awareness.
    """
    global _watchdog_running
    _watchdog_running = True
    log.info("Heartbeat watchdog started — timeout: %d seconds", _HEARTBEAT_TIMEOUT_SEC)

    while _watchdog_running:
        try:
            now = time.time()

            for tower_id, hb in list(_last_heartbeat.items()):
                last_ts = hb.get("last_seen", 0)
                gap = now - last_ts

                if gap > _HEARTBEAT_TIMEOUT_SEC and not hb.get("alert_sent", False):
                    # Tower has exceeded silence threshold — raise connectivity alert
                    hb["alert_sent"] = True
                    hb["is_alive"] = False

                    log.critical(
                        "CONNECTIVITY LOSS: Tower %s silent for %d seconds.",
                        tower_id, int(gap)
                    )

                    await ws_manager.broadcast_alert(
                        alert_type="connection_loss",
                        tower_id=tower_id,
                        details={
                            "message": f"Device offline — last heartbeat {int(gap)} seconds ago",
                            "message_ar": f"الجهاز فقد الاتصال — آخر نبضة منذ {int(gap)} ثانية",
                            "severity": SEVERITY_CONFIG.SEVERITY_CRITICAL,
                            "severity_label": SEVERITY_CONFIG.SEVERITY_LABELS_EN[2],
                            "severity_label_ar": SEVERITY_CONFIG.SEVERITY_LABELS_AR[2],
                            "gap_seconds": int(gap),
                            "noc_phone": NOC_PHONE,
                        },
                    )

                    with _counter_lock:
                        _request_counter["dead_mans_switch"] += 1

                elif gap <= _HEARTBEAT_TIMEOUT_SEC and hb.get("alert_sent", False):
                    # Tower has reconnected — broadcast recovery notification
                    hb["alert_sent"] = False
                    hb["is_alive"] = True

                    log.info("CONNECTIVITY RESTORED: Tower %s back online.", tower_id)
                    await ws_manager.broadcast_alert(
                        alert_type="connection_restored",
                        tower_id=tower_id,
                        details={
                            "message": "Device reconnected and online",
                            "message_ar": "الجهاز استعاد الاتصال",
                            "severity": SEVERITY_CONFIG.SEVERITY_OK,
                            "severity_label": SEVERITY_CONFIG.SEVERITY_LABELS_EN[0],
                            "severity_label_ar": SEVERITY_CONFIG.SEVERITY_LABELS_AR[0],
                        },
                    )

        except Exception as exc:
            log.error("Heartbeat watchdog error: %s", exc)

        await asyncio.sleep(30)


# ══════════════════════════════════════════════════════════════════════════════
#  Model Loading
# ══════════════════════════════════════════════════════════════════════════════

async def get_bundle() -> ModelBundle:
    global _bundle
    if _bundle is not None:
        return _bundle
    async with _bundle_lock:
        if _bundle is None:
            log.info("Loading Digital Twin model...")
            loop = asyncio.get_running_loop()
            _bundle = await loop.run_in_executor(None, load_or_train, MODEL_PATH)
            log.info("Digital Twin model loaded successfully.")
    return _bundle


# ── Application lifespan ───────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        await get_bundle()
    except Exception as exc:
        log.critical("Model load failure: %s", exc, exc_info=True)

    # Activate heartbeat watchdog (Dead Man's Switch)
    watchdog_task = asyncio.create_task(_heartbeat_watchdog())
    log.info("Heartbeat watchdog active — silence threshold: %d seconds", _HEARTBEAT_TIMEOUT_SEC)

    yield

    global _watchdog_running
    _watchdog_running = False
    watchdog_task.cancel()
    log.info("TowerGuard server stopping.")


# ── FastAPI application ────────────────────────────────────────────────────
app = FastAPI(
    title="TowerGuard API",
    description="Saudi Sovereign Digital Twin — Cellular Tower Outage Prediction",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS, allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept", "X-Request-ID", "X-API-Key"],
)
app.add_middleware(GZipMiddleware, minimum_size=500)


# ── Request logging middleware ─────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = (time.perf_counter() - start) * 1000
    log.info("← %s %s  %d  %.1fms", request.method, request.url.path, response.status_code, elapsed)
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception: %s %s\n%s", request.method, request.url.path, traceback.format_exc())
    return JSONResponse(status_code=500, content={"ok": False, "error": "Internal server error"})


# ── Prediction helpers ─────────────────────────────────────────────────────
def verdict_from_probability(p: float) -> str:
    if p >= THRESHOLD_DANGER:
        return "danger"
    if p >= THRESHOLD_WARN:
        return "warn"
    return "ok"


def verdict_ar(v: str) -> str:
    return {"danger": "خطر", "warn": "تحذير", "ok": "مستقر"}.get(v, v)


def _model_meta(bundle: ModelBundle) -> Dict[str, Any]:
    return {
        "name": bundle.meta.get("name"),
        "version": bundle.meta.get("version"),
        "n_estimators": MODEL_CONFIG.n_estimators,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Pydantic Request Models
# ══════════════════════════════════════════════════════════════════════════════

class PredictRequest(BaseModel):
    rssi_dbm:        float = Field(..., ge=-140, le=-20)
    snr_db:          float = Field(..., ge=-10, le=50)
    latency_ms:      float = Field(..., ge=0, le=5000)
    packet_loss_pct: float = Field(..., ge=0, le=100)
    tower_load_pct:  float = Field(..., ge=0, le=100)
    temp_celsius:    float = Field(..., ge=-40, le=120)
    rssi_prior:          Optional[float] = None
    signal_variance:     Optional[float] = None
    load_temp_index:     Optional[float] = None
    rssi_rate_of_change: Optional[float] = None
    snr_trend:           Optional[float] = None


class HeartbeatRequest(BaseModel):
    tower_id: str = Field(..., min_length=1, max_length=64)
    rssi_dbm: Optional[float] = None
    verdict: Optional[str] = None


# ══════════════════════════════════════════════════════════════════════════════
#  API Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/health", tags=["meta"])
async def health():
    bundle = await get_bundle()
    info = get_version_info()
    return {"ok": True, **info, "model_name": bundle.meta.get("name")}


@app.get("/api/model-info", tags=["meta"])
async def model_info():
    bundle = await get_bundle()
    return {
        "ok": True,
        "name": bundle.meta.get("name"),
        "version": bundle.meta.get("version"),
        "n_estimators": MODEL_CONFIG.n_estimators,
        "features": bundle.features,
        "feature_importances": bundle.meta.get("feature_importances"),
        "thresholds": {"danger": THRESHOLD_DANGER, "warn": THRESHOLD_WARN},
        "severity_labels": SEVERITY_CONFIG.SEVERITY_LABELS_EN,
    }


# ── Single prediction ──────────────────────────────────────────────────────
@app.post("/api/predict", tags=["prediction"])
@limiter.limit("60/minute")
async def predict(request: Request, req: PredictRequest):
    try:
        row = normalize_input(req.model_dump(exclude_none=True))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    bundle = await get_bundle()
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, predict_with_confidence_dict, bundle, row)

    verdict = verdict_from_probability(result["probability"])

    # Push result to all WebSocket clients immediately
    await ws_manager.broadcast({
        "type": "prediction",
        "tower_id": req.model_dump().get("tower_id", "—"),
        "probability": result["probability"],
        "verdict": verdict,
        "verdict_ar": verdict_ar(verdict),
        "severity": result["الخطورة"],
        "severity_label_ar": result["التصنيف"],
        "confidence_interval": result["فاصل_الثقة"],
        "timestamp": int(time.time() * 1000),
    })

    with _counter_lock:
        _request_counter["total"] += 1
        _request_counter[verdict] += 1

    return {
        "ok": True,
        "probability_outage": result["probability"],
        "risk_percent": int(round(result["probability"] * 100)),
        "verdict": verdict,
        "verdict_ar": verdict_ar(verdict),
        "severity": result["الخطورة"],
        "severity_label_ar": result["التصنيف"],
        "confidence_interval": result["فاصل_الثقة"],
        "class_probabilities": result.get("احتمالات_الفئات", {}),
        "thresholds": {"danger": THRESHOLD_DANGER, "warn": THRESHOLD_WARN},
        "model": _model_meta(bundle),
        "features_used": row,
    }


# ── Batch prediction ───────────────────────────────────────────────────────
@app.post("/api/predict/batch", tags=["prediction"])
@limiter.limit("30/minute")
async def predict_batch(request: Request, req: dict):
    towers = req.get("towers", [])
    if not towers:
        raise HTTPException(status_code=422, detail="No tower data provided")

    rows = []
    for t in towers:
        try:
            rows.append(normalize_input(t))
        except ValueError:
            rows.append(None)

    bundle = await get_bundle()

    def _batch_infer():
        valid_rows = [r for r in rows if r is not None]
        if not valid_rows:
            return []
        X = pd.DataFrame([[r[f] for r in valid_rows] for f in bundle.features]).T
        X.columns = bundle.features
        X_scaled = bundle.scaler.transform(X)
        probas = bundle.model.predict_proba(X_scaled)
        results = []
        for i, (r, p) in enumerate(zip(valid_rows, probas)):
            if len(p) > 2:
                risk = float(p[1] + p[2])
            else:
                risk = float(p[1]) if len(p) > 1 else 0.0
            risk = max(0.0, min(1.0, risk))
            v = verdict_from_probability(risk)
            results.append({
                "index": i,
                "probability_outage": round(risk, 4),
                "risk_percent": int(round(risk * 100)),
                "verdict": v,
                "verdict_ar": verdict_ar(v),
            })
        return results

    loop = asyncio.get_running_loop()
    results = await loop.run_in_executor(None, _batch_infer)

    await ws_manager.broadcast({
        "type": "batch_prediction",
        "count": len(results),
        "results": results,
        "timestamp": int(time.time() * 1000),
    })

    return {"ok": True, "count": len(results), "results": results}


# ══════════════════════════════════════════════════════════════════════════════
#  Heartbeat Endpoints — Dead Man's Switch
# ══════════════════════════════════════════════════════════════════════════════

@app.post("/api/heartbeat", tags=["connectivity"])
async def heartbeat_endpoint(request: Request, req: HeartbeatRequest):
    """
    Device heartbeat endpoint. Each edge device transmits every 30–60 seconds.
    If a tower exceeds the 180-second silence threshold, _heartbeat_watchdog
    raises a connectivity alert automatically.
    """
    now = time.time()
    _last_heartbeat[req.tower_id] = {
        "last_seen": now,
        "rssi": req.rssi_dbm or -70.0,
        "verdict": req.verdict or "ok",
        "is_alive": True,
        "alert_sent": False,
    }
    return {
        "ok": True, "tower_id": req.tower_id,
        "message": "Heartbeat received.",
        "message_ar": "تم استقبال نبض القلب",
    }


@app.get("/api/heartbeat/status", tags=["connectivity"])
async def heartbeat_status():
    now = time.time()
    towers = {}
    for tid, hb in _last_heartbeat.items():
        gap = now - hb.get("last_seen", 0)
        towers[tid] = {
            "is_alive": gap <= _HEARTBEAT_TIMEOUT_SEC,
            "gap_seconds": int(gap),
            "verdict": hb.get("verdict", "—"),
            "alert_sent": hb.get("alert_sent", False),
        }

    alive = sum(1 for t in towers.values() if t["is_alive"])
    dead = len(towers) - alive
    return {
        "ok": True,
        "total": len(towers), "alive": alive, "dead": dead,
        "timeout_sec": _HEARTBEAT_TIMEOUT_SEC,
        "towers": towers,
    }


@app.get("/api/heartbeat/dead", tags=["connectivity"])
async def dead_towers():
    now = time.time()
    dead = [
        {"tower_id": tid, "gap_seconds": int(now - hb["last_seen"])}
        for tid, hb in _last_heartbeat.items()
        if (now - hb.get("last_seen", 0)) > _HEARTBEAT_TIMEOUT_SEC
    ]
    return {"ok": True, "count": len(dead), "dead_towers": dead}


# ══════════════════════════════════════════════════════════════════════════════
#  WebSocket — Real-Time Tower Feed
# ══════════════════════════════════════════════════════════════════════════════

@app.websocket("/ws/towers")
async def ws_tower_feed(websocket: WebSocket):
    """
    Real-time WebSocket feed for all connected dashboard clients.

    Client sends: {"towers": [{rssi_dbm: ..., ...}]}
    Server responds immediately with prediction results and pushes alerts autonomously.
    """
    await ws_manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()

            msg_type = data.get("type", "predict")

            if msg_type == "heartbeat":
                tid = data.get("tower_id", "unknown")
                _last_heartbeat[tid] = {
                    "last_seen": time.time(),
                    "rssi": data.get("rssi_dbm", -70),
                    "verdict": data.get("verdict", "ok"),
                    "is_alive": True, "alert_sent": False,
                }
                await websocket.send_json({
                    "type": "heartbeat_ack",
                    "tower_id": tid,
                    "message": "Heartbeat acknowledged.",
                    "message_ar": "نبض مستلم",
                })
                continue

            if msg_type == "predict" or "towers" in data:
                towers = data.get("towers", [])
                if not towers:
                    await websocket.send_json({"ok": False, "error": "No tower data provided"})
                    continue

                rows = []
                for t in towers:
                    try:
                        rows.append(normalize_input(t))
                    except ValueError:
                        pass

                if not rows:
                    await websocket.send_json({"ok": False, "error": "Invalid input data"})
                    continue

                try:
                    bundle = await get_bundle()

                    def _ws_infer():
                        X = pd.DataFrame(
                            [[r[f] for f in bundle.features] for r in rows],
                            columns=bundle.features
                        )
                        X_scaled = bundle.scaler.transform(X)
                        probas = bundle.model.predict_proba(X_scaled)
                        results = []
                        for i, (r, p) in enumerate(zip(rows, probas)):
                            if len(p) > 2:
                                risk = float(p[1] + p[2])
                            else:
                                risk = float(p[1]) if len(p) > 1 else 0.0
                            risk = max(0.0, min(1.0, risk))

                            # Per-tree confidence interval (sample 20 trees for latency)
                            tree_preds = np.array([
                                (tree.predict_proba(X_scaled[i:i+1])[0][1:].sum()
                                 if len(tree.predict_proba(X_scaled[i:i+1])[0]) > 2
                                 else tree.predict_proba(X_scaled[i:i+1])[0][1]
                                 if len(tree.predict_proba(X_scaled[i:i+1])[0]) > 1
                                 else 0.0)
                                for tree in bundle.model.estimators_[:20]
                            ])
                            std_p = float(np.std(tree_preds))
                            z = MODEL_CONFIG.confidence_z
                            ci_low = max(0.0, risk - z * std_p)
                            ci_high = min(1.0, risk + z * std_p)

                            v = verdict_from_probability(risk)
                            results.append({
                                "index": i,
                                "probability": round(risk, 4),
                                "risk_percent": int(round(risk * 100)),
                                "verdict": v,
                                "verdict_ar": verdict_ar(v),
                                "confidence_interval": {
                                    "low": round(ci_low, 4),
                                    "high": round(ci_high, 4),
                                    "level": MODEL_CONFIG.confidence_level,
                                },
                            })
                        return results

                    loop = asyncio.get_running_loop()
                    results = await loop.run_in_executor(None, _ws_infer)

                    await websocket.send_json({
                        "ok": True,
                        "type": "prediction_results",
                        "count": len(results),
                        "results": results,
                        "timestamp": int(time.time() * 1000),
                    })

                except Exception as exc:
                    await websocket.send_json({"ok": False, "error": f"Prediction error: {exc}"})

    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
    except Exception as exc:
        log.error("WebSocket error: %s", exc)
        ws_manager.disconnect(websocket)


# ══════════════════════════════════════════════════════════════════════════════
#  System Statistics and Edge Model Generation
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/stats", tags=["meta"])
async def stats():
    bundle = await get_bundle()
    now = time.time()
    alive = sum(1 for hb in _last_heartbeat.values() if (now - hb.get("last_seen", 0)) <= _HEARTBEAT_TIMEOUT_SEC)
    dead = len(_last_heartbeat) - alive
    return {
        "ok": True,
        "system": get_version_info(),
        "total_requests": _request_counter.get("total", 0),
        "verdicts": {
            "danger": _request_counter.get("danger", 0),
            "warn": _request_counter.get("warn", 0),
            "ok": _request_counter.get("ok", 0),
        },
        "connectivity": {
            "dead_mans_switch_triggers": _request_counter.get("dead_mans_switch", 0),
            "towers_alive": alive, "towers_dead": dead,
            "ws_clients": len(ws_manager.clients),
        },
    }


@app.get("/api/edge-model", tags=["connectivity"])
async def get_edge_model():
    gen = ESP32FirmwareGenerator(output_dir="/tmp/tg_fw")
    gen.generate_all()
    return {"ok": True, "message": "ESP32 firmware files generated successfully."}


# ── Static file routing ────────────────────────────────────────────────────
def _file_or_404(path: Path) -> FileResponse:
    if not path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(path)


@app.get("/", include_in_schema=False)
async def index():
    return _file_or_404(ROOT_DIR / "towerguard-fow-landing.html")

@app.get("/predictor", include_in_schema=False)
async def predictor():
    return _file_or_404(ROOT_DIR / "tower-predictor.html")

@app.get("/device", include_in_schema=False)
async def device():
    return _file_or_404(ROOT_DIR / "device-simulation.html")

@app.get("/market", include_in_schema=False)
async def market():
    return _file_or_404(ROOT_DIR / "market.html")

@app.get("/map", include_in_schema=False)
async def tower_map():
    return _file_or_404(ROOT_DIR / "tower-map.html")

@app.get("/{path:path}", include_in_schema=False)
async def catch_all(path: str):
    safe = (ROOT_DIR / path).resolve()
    if ROOT_DIR not in safe.parents and safe != ROOT_DIR:
        return JSONResponse(status_code=400, content={"ok": False})
    if safe.exists():
        return FileResponse(safe)
    return JSONResponse(status_code=404, content={"ok": False, "detail": "Not found"})

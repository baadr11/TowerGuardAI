#!/usr/bin/env python3
"""
TowerGuard — Server Entry Point
===================================
Starts the FastAPI + WebSocket server with:
  - Automatic model loading (or training from scratch on first run)
  - Heartbeat watchdog (_heartbeat_watchdog) — 180-second silence threshold
  - Real-time WebSocket feed on /ws/towers
  - Five HTML interfaces: presentation | control panel | device simulation | GIS map | business case

Usage:
    python run.py
"""

import os
import sys
import uvicorn


def main():
    # ── Server configuration (override via environment variables) ──────────
    host        = os.getenv("TG_HOST",   "0.0.0.0")
    port        = int(os.getenv("TG_PORT", "8000"))
    reload_mode = os.getenv("TG_RELOAD", "0") == "1"

    # ── Development mode: allows all origins and shows detailed error traces ──
    os.environ.setdefault("TOWERGUARD_DEV", "1")

    # ── NOC Phone: set via environment to avoid hardcoded values ─────────────
    # os.environ.setdefault("TG_NOC_PHONE", "+966501234567")

    dev_label = "Development 🔧" if os.getenv("TOWERGUARD_DEV") == "1" else "Production 🔒"

    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║   TowerGuard — Saudi Sovereign Digital Twin Prototype   ║")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print(f"  ║  Server:     http://{host}:{port}")
    print(f"  ║  WebSocket:  ws://{host}:{port}/ws/towers")
    print(f"  ║  Mode:       {dev_label}")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  Interfaces:                                             ║")
    print(f"  ║    /            Presentation — system overview")
    print(f"  ║    /predictor   Control panel + 90% confidence interval")
    print(f"  ║    /device      ESP32 edge device simulation")
    print(f"  ║    /map         10,011-tower GIS map (Canvas + Cluster)")
    print(f"  ║    /market      Business case + ROI analysis")
    print("  ╠══════════════════════════════════════════════════════════╣")
    print("  ║  API Endpoints:                                          ║")
    print(f"  ║    POST /api/predict           Single prediction")
    print(f"  ║    POST /api/predict/batch     Batch prediction (≤500)")
    print(f"  ║    POST /api/heartbeat         Device heartbeat")
    print(f"  ║    GET  /api/heartbeat/dead    Silent towers (>180s)")
    print(f"  ║    GET  /api/health            Server health check")
    print(f"  ║    GET  /api/model-info        Model metadata")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload_mode,
        log_level="info",
        ws_ping_interval=30,
        ws_ping_timeout=10,
    )


if __name__ == "__main__":
    main()

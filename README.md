<p align="center">
  <img src="https://img.shields.io/badge/System-TowerGuard-00b4ff?style=for-the-badge" alt="TowerGuard"/>
  <img src="https://img.shields.io/badge/Python-3.10+-3776ab?style=for-the-badge&logo=python&logoColor=white" alt="python"/>
  <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="fastapi"/>
  <img src="https://img.shields.io/badge/License-Proprietary-ff3d5a?style=for-the-badge" alt="license"/>
</p>

<h1 align="center">TowerGuard</h1>
<h3 align="center">Saudi Sovereign Digital Twin — Cellular Tower Outage Prediction</h3>

<p align="center">
  An AI system that monitors 10,011 real cellular towers across the Kingdom of Saudi Arabia<br/>
  and predicts outages <strong>before they occur</strong> using Random Forest + ESP32 Edge Inference.
</p>

---

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [File Structure](#file-structure)
- [Interfaces](#interfaces)
- [API Reference](#api-reference)
- [Configuration](#configuration)
- [ESP32 Firmware](#esp32-firmware)
- [Hardware Specification](#hardware-specification)
- [Compliance](#compliance)

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the server
python run.py

# 3. Open in browser
# http://localhost:8000           → Main presentation
# http://localhost:8000/predictor → Control panel + confidence interval
# http://localhost:8000/device    → ESP32 simulation
# http://localhost:8000/map       → 10,011-tower GIS map
# http://localhost:8000/market    → Business case + ROI
```

> **Note:** On first run, the model trains automatically from the Saudi Digital Twin dataset (10,011 towers). This takes approximately 30–60 seconds.

---

## Architecture

```
┌─────────────┐     WebSocket      ┌──────────────┐     predict_with     ┌──────────────┐
│  Browser    │ ◄═══════════════► │  FastAPI      │ ══════════════════► │ Random Forest│
│  (4 pages)  │    /ws/towers      │  main.py      │    _confidence()    │  200 trees   │
└─────────────┘                    └──────┬───────┘                      └──────────────┘
                                          │
                                   ┌──────┴───────┐
                                   │ _heartbeat_  │
                                   │  watchdog()  │
                                   │ timeout: 180s│
                                   └──────┬───────┘
                                          │ alert
                                          ▼
                               ┌──────────────────┐
                               │  ESP32 + SIM7600  │
                               │  Cloud → SMS →    │
                               │  Edge Fallback    │
                               └──────────────────┘
```

**3-layer connectivity — Connectivity Paradox Solution:**

| Layer | Mechanism | Active When |
|-------|-----------|-------------|
| 1. Cloud API | HTTP/WebSocket → server | LTE Data available |
| 2. SMS Fallback | SIM7600SA-H → SMS | LTE Data down, CS domain active |
| 3. Dead Man's Switch | `_heartbeat_watchdog` | All connectivity down — silence triggers alert |

---

## File Structure

```
towerguard/
│
├── run.py                        ← Single launch entry point
├── requirements.txt              ← Python dependencies
├── README.md                     ← This file
│
├── backend/
│   ├── __init__.py
│   ├── config.py                 ← Single source of truth for all parameters
│   ├── main.py                   ← FastAPI + WebSocket + _heartbeat_watchdog
│   ├── real_data_loader.py       ← Saudi Digital Twin data loader (10,011 towers)
│   ├── towerguard_ml.py          ← Random Forest + predict_with_confidence
│   ├── validation.py             ← Precision/Recall/CM/F1 per class
│   └── edge_inference.py         ← ESP32 firmware generator
│
├── firmware/                     ← C/Arduino files for physical device
│   ├── towerguard_config.h       ← Device settings (from config.py)
│   ├── towerguard_edge_model.h   ← O(1) local prediction model
│   ├── towerguard_main.ino       ← Full Arduino sketch
│   └── towerguard_sms.h          ← SMS fallback handler
│
├── towerguard-prototype.html     ← Main presentation
├── tower-predictor.html          ← Control panel + confidence interval
├── tower-map.html                ← 10,011-tower GIS map (Canvas + Cluster)
├── device-simulation.html        ← ESP32 virtual device simulation
└── market.html                   ← Business case + ROI analysis
```

---

## Interfaces

| Page | Route | Description |
|------|-------|-------------|
| **Presentation** | `/` | System overview + live simulation of 9 towers + architecture |
| **Control Panel** | `/predictor` | Sliders → real-time prediction + **90% ICP confidence interval** |
| **Device Simulation** | `/device` | Virtual ESP32 + LEDs + Serial Monitor + test scenarios |
| **GIS Map** | `/map` | Canvas-rendered map of all 10,011 KSA towers with clustering |
| **Business Case** | `/market` | Market sizing + ROI table + competitor analysis |

All pages use WebSocket for real-time updates and share a unified navigation bar.

---

## API Reference

### Single Prediction

```bash
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "rssi_dbm": -85,
    "snr_db": 8,
    "latency_ms": 120,
    "packet_loss_pct": 12,
    "tower_load_pct": 78,
    "temp_celsius": 55
  }'

# Response:
{
  "ok": true,
  "probability_outage": 0.72,
  "risk_percent": 72,
  "verdict": "danger",
  "verdict_ar": "خطر",
  "severity": 2,
  "severity_label_ar": "حرج",
  "confidence_interval": {
    "low": 0.58,
    "high": 0.86,
    "level": 0.90
  }
}
```

### Heartbeat — Dead Man's Switch

```bash
# Send heartbeat
curl -X POST http://localhost:8000/api/heartbeat \
  -H "Content-Type: application/json" \
  -d '{"tower_id": "T01", "rssi_dbm": -72}'

# Query silent towers (> 180 seconds without heartbeat)
curl http://localhost:8000/api/heartbeat/dead
```

### WebSocket — Real-Time Feed

```javascript
const ws = new WebSocket('ws://localhost:8000/ws/towers');

// Send tower readings
ws.send(JSON.stringify({
  type: 'predict',
  towers: [{ rssi_dbm: -85, snr_db: 8, latency_ms: 120,
             packet_loss_pct: 12, tower_load_pct: 78, temp_celsius: 55 }]
}));

// Receive results immediately
ws.onmessage = (e) => {
  const data = JSON.parse(e.data);
  // data.type === 'prediction_results' | 'alert'
};
```

### All Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/api/health` | Server health check |
| `GET` | `/api/model-info` | Model metadata + feature importances |
| `GET` | `/api/stats` | System statistics |
| `POST` | `/api/predict` | Single prediction + confidence interval |
| `POST` | `/api/predict/batch` | Batch prediction (up to 500 towers) |
| `POST` | `/api/heartbeat` | Device heartbeat |
| `GET` | `/api/heartbeat/status` | All tower connectivity status |
| `GET` | `/api/heartbeat/dead` | Silent towers |
| `WS` | `/ws/towers` | WebSocket — real-time prediction + alerts |

---

## Configuration

All values are defined in `backend/config.py`. No hardcoded constants exist in any other file.

### Environment Variables (optional)

```bash
# Server
export TG_HOST="0.0.0.0"
export TG_PORT="8000"
export TG_RELOAD="1"               # Auto-reload for development

# Security
export TOWERGUARD_DEV="0"          # Production mode
export TOWERGUARD_API_KEY="xxx"    # API key (production only)

# Alerts
export TG_NOC_PHONE="+966501234567"
export TG_SERVER_URL="https://api.towerguard.sa"
```

### Key Parameters

| Parameter | Value | File |
|-----------|-------|------|
| `n_estimators` | 200 | `config.py → MODEL_CONFIG` |
| `max_depth` | 12 | `config.py → MODEL_CONFIG` |
| `confidence_level` | 0.90 | `config.py → MODEL_CONFIG` |
| `heartbeat_timeout` | 180 seconds | `main.py → _HEARTBEAT_TIMEOUT_SEC` |
| `prob_degraded` | 0.35 | `config.py → SEVERITY_CONFIG` |
| `prob_critical` | 0.65 | `config.py → SEVERITY_CONFIG` |

---

## ESP32 Firmware

Files in `firmware/` are ready for deployment on ESP32-S3 + SIM7600SA-H:

```
firmware/
├── towerguard_config.h       ← All device settings (generated from config.py)
├── towerguard_edge_model.h   ← O(1) local prediction model
├── towerguard_main.ino       ← Complete Arduino sketch
└── towerguard_sms.h          ← SMS fallback handler (SIM7600SA-H)
```

**Device execution flow:**

```
Read sensors → Local prediction (Edge) → Attempt Cloud API transmission
                                             ↓ Failed?
                                       SMS via SIM7600SA-H
                                             ↓ Failed?
                                       SPIFFS local buffer
                                             ↓ Connectivity restored
                                       Batch flush to cloud
```

---

## Hardware Specification

| Component | Specification |
|-----------|---------------|
| Microcontroller | ESP32-S3 |
| Cellular Modem | SIM7600SA-H — CITC Type Approved for KSA LTE-FDD/LTE-TDD networks |
| Local Inference | O(1) weighted linear model — no dynamic memory allocation |
| Local Storage | SPIFFS up to 256 KB |
| SMS Standard | 3GPP TS 23.040 — AT+CMGS via SIM7600SA-H |
| Heartbeat | 30-second interval; 180-second watchdog timeout |

---

## Compliance

| Standard | Scope | Status |
|----------|-------|--------|
| **NCA ECC-1:2018 §3.3.5** | Data classification and access controls — hosting on STC Cloud, Riyadh (mandatory architectural requirement) | Compliant |
| **PDPL Article 4** | Personal data rights and residency | Compliant |
| **CITC Type Approval** | SIM7600SA-H certified for KSA LTE networks | Approved |
| **Data Residency** | All data processed within KSA — STC Cloud (Riyadh) primary | Enforced |

---

## Data Sources

| Source | Usage |
|--------|-------|
| **OpenCelliD** (420.csv) | 10,011 real tower locations in Saudi Arabia (MCC=420, LTE/UMTS/NR) |
| **Turkcell KPI Profiles** | Statistical KPI distributions (RSSI, SNR, latency) — distribution patterns only, not values |
| **Saudi Climate Corrections** | Dry desert (+8°C, −2.5dBm) \| Coastal (+3°C) \| Highland (−2°C) — per ITU-R P.618 / GSMA ME 2023 |

---

<p align="center">
  <strong>TowerGuard — Saudi Sovereign Digital Twin Prototype</strong><br/>
  <sub>10,011 KSA Towers · Random Forest 200 Trees · WebSocket Real-Time · ESP32 Edge Inference · STC Cloud (Riyadh)</sub>
</p>

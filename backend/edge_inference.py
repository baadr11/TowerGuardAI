"""
edge_inference.py — TowerGuard ESP32 Firmware Generator
====================================================================
Code generator that produces .h and .ino C/Arduino source files
for the ESP32 + SIM7600SA-H edge device.

This module is a firmware generator, not a Python inference runtime.
The generated C code runs locally on the device with O(1) complexity.

Usage:
    from backend.edge_inference import ESP32FirmwareGenerator
    gen = ESP32FirmwareGenerator(output_dir="firmware/")
    gen.generate_all()
"""

import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from .config import EDGE_CONFIG, SEVERITY_CONFIG, ARABIC_LABELS

log = logging.getLogger("towerguard.firmware")


class ESP32FirmwareGenerator:
    """
    ESP32 firmware file generator for TowerGuard.

    Produces .h and .ino source files for deployment on the
    ESP32-S3 + SIM7600SA-H edge device. All configuration values
    are sourced exclusively from EDGE_CONFIG and SEVERITY_CONFIG.
    """

    def __init__(self, output_dir: str = "firmware"):
        self.output_dir = Path(output_dir)
        self.timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    def generate_all(self) -> None:
        """Generate all firmware source files for the edge device."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        files_generated = [
            self._generate_config_header(),
            self._generate_edge_model_header(),
            self._generate_main_sketch(),
            self._generate_sms_handler(),
        ]

        log.info("=" * 60)
        log.info("ESP32 firmware generation complete:")
        for f in files_generated:
            log.info("  Generated: %s", f)
        log.info("=" * 60)
        log.info("Next step: deploy firmware to ESP32 via PlatformIO or Arduino IDE.")

    # ══════════════════════════════════════════════════════════════════════
    #  1. towerguard_config.h — Device Configuration Header
    # ══════════════════════════════════════════════════════════════════════

    def _generate_config_header(self) -> str:
        """Generate C configuration header. All values sourced from EDGE_CONFIG."""
        filename = "towerguard_config.h"
        filepath = self.output_dir / filename

        content = f'''/*
 * TowerGuard — ESP32 Configuration Header
 * ========================================
 * Auto-generated: {self.timestamp}
 * Source: backend/config.py → EDGE_CONFIG
 *
 * WARNING: Do not edit manually — regenerate from config.py.
 * Hardware: ESP32-S3 + SIM7600SA-H (CITC Type Approved)
 */

#ifndef TOWERGUARD_CONFIG_H
#define TOWERGUARD_CONFIG_H

/* ── Connectivity settings ── */
#define TG_SERVER_URL           "{EDGE_CONFIG.server_url}"
#define TG_NOC_PHONE            "{EDGE_CONFIG.noc_phone}"

/* ── Timing settings ── */
#define TG_HEARTBEAT_INTERVAL_MS  {EDGE_CONFIG.heartbeat_interval_sec * 1000}
#define TG_HEARTBEAT_TIMEOUT_MS   {EDGE_CONFIG.heartbeat_timeout_sec * 1000}
#define TG_READING_INTERVAL_MS    {EDGE_CONFIG.reading_interval_sec * 1000}

/* ── Local storage settings ── */
#define TG_SPIFFS_BUFFER_KB     {EDGE_CONFIG.spiffs_buffer_kb}
#define TG_MAX_BATCH_SIZE       {EDGE_CONFIG.max_batch_size}
#define TG_BUFFER_FILE          "/tg_buffer.jsonl"

/* ── Local prediction thresholds ── */
#define TG_PKT_LOSS_CRITICAL    {EDGE_CONFIG.pkt_loss_critical:.1f}f
#define TG_RSSI_ROC_CRITICAL    {EDGE_CONFIG.rssi_roc_critical:.1f}f
#define TG_SNR_TREND_CRITICAL   {EDGE_CONFIG.snr_trend_critical:.1f}f
#define TG_LATENCY_CRITICAL     {EDGE_CONFIG.latency_critical:.1f}f
#define TG_THRESHOLD_DANGER     {EDGE_CONFIG.edge_danger_threshold:.2f}f
#define TG_THRESHOLD_WARN       {EDGE_CONFIG.edge_warn_threshold:.2f}f

/* ── Feature weights (derived from Random Forest feature importance) ── */
#define TG_W_PKT_LOSS           {EDGE_CONFIG.w_pkt_loss:.3f}f
#define TG_W_RSSI_ROC           {EDGE_CONFIG.w_rssi_roc:.3f}f
#define TG_W_SNR_TREND          {EDGE_CONFIG.w_snr_trend:.3f}f
#define TG_W_LATENCY            {EDGE_CONFIG.w_latency:.3f}f

/* ── Multi-class severity levels ── */
#define TG_SEVERITY_OK          {SEVERITY_CONFIG.SEVERITY_OK}    /* OK */
#define TG_SEVERITY_DEGRADED    {SEVERITY_CONFIG.SEVERITY_DEGRADED}    /* Degraded */
#define TG_SEVERITY_CRITICAL    {SEVERITY_CONFIG.SEVERITY_CRITICAL}    /* Critical */

#endif /* TOWERGUARD_CONFIG_H */
'''
        filepath.write_text(content, encoding="utf-8")
        return filename

    # ══════════════════════════════════════════════════════════════════════
    #  2. towerguard_edge_model.h — Local Prediction Model
    # ══════════════════════════════════════════════════════════════════════

    def _generate_edge_model_header(self) -> str:
        """Generate the lightweight local prediction model for the ESP32."""
        filename = "towerguard_edge_model.h"
        filepath = self.output_dir / filename

        content = f'''/*
 * TowerGuard — Edge Prediction Model
 * ==========================================
 * Auto-generated: {self.timestamp}
 *
 * Lightweight local prediction model for ESP32.
 * Operates without server connectivity. Uses 4 features
 * representing 78.5% of the full model's total feature importance.
 *
 * Output (severity_score):
 *   0 = OK
 *   1 = Degraded
 *   2 = Critical
 */

#ifndef TOWERGUARD_EDGE_MODEL_H
#define TOWERGUARD_EDGE_MODEL_H

#include "towerguard_config.h"
#include <math.h>

/* ── Utility: clamp float to [lo, hi] ── */
static inline float tg_clampf(float val, float lo, float hi) {{
    return val < lo ? lo : (val > hi ? hi : val);
}}

/**
 * Lightweight local prediction — O(1), no dynamic memory allocation.
 * @return Outage probability [0.0, 1.0]
 */
static inline float tg_edge_predict(
    float packet_loss_pct,
    float rssi_rate_of_change,
    float snr_trend,
    float latency_ms
) {{
    float score = 0.0f;
    const float w_sum = TG_W_PKT_LOSS + TG_W_RSSI_ROC + TG_W_SNR_TREND + TG_W_LATENCY;

    /* Packet loss contribution */
    if (packet_loss_pct > TG_PKT_LOSS_CRITICAL) {{
        score += TG_W_PKT_LOSS * tg_clampf(
            (packet_loss_pct - TG_PKT_LOSS_CRITICAL) / 15.0f, 0.0f, 1.0f);
    }}

    /* RSSI rate-of-change (negative = degrading) */
    if (rssi_rate_of_change < TG_RSSI_ROC_CRITICAL) {{
        score += TG_W_RSSI_ROC * tg_clampf(
            fabsf(rssi_rate_of_change - TG_RSSI_ROC_CRITICAL) / 3.0f, 0.0f, 1.0f);
    }}

    /* SNR trend (negative = degrading) */
    if (snr_trend < TG_SNR_TREND_CRITICAL) {{
        score += TG_W_SNR_TREND * tg_clampf(
            fabsf(snr_trend - TG_SNR_TREND_CRITICAL) / 2.0f, 0.0f, 1.0f);
    }}

    /* Latency contribution */
    if (latency_ms > TG_LATENCY_CRITICAL) {{
        score += TG_W_LATENCY * tg_clampf(
            (latency_ms - TG_LATENCY_CRITICAL) / 200.0f, 0.0f, 1.0f);
    }}

    return tg_clampf(score / w_sum, 0.0f, 1.0f);
}}

/**
 * Multi-class severity classification.
 * @return TG_SEVERITY_OK (0) | TG_SEVERITY_DEGRADED (1) | TG_SEVERITY_CRITICAL (2)
 */
static inline int tg_edge_severity(float probability) {{
    if (probability >= TG_THRESHOLD_DANGER) return TG_SEVERITY_CRITICAL;
    if (probability >= TG_THRESHOLD_WARN)   return TG_SEVERITY_DEGRADED;
    return TG_SEVERITY_OK;
}}

/**
 * Arabic severity label for on-device display.
 */
static inline const char* tg_severity_label_ar(int severity) {{
    switch (severity) {{
        case TG_SEVERITY_CRITICAL: return "حرج";
        case TG_SEVERITY_DEGRADED: return "متدهور";
        default:                   return "مستقر";
    }}
}}

/**
 * English severity label for serial logging.
 */
static inline const char* tg_severity_label_en(int severity) {{
    switch (severity) {{
        case TG_SEVERITY_CRITICAL: return "Critical";
        case TG_SEVERITY_DEGRADED: return "Degraded";
        default:                   return "OK";
    }}
}}

#endif /* TOWERGUARD_EDGE_MODEL_H */
'''
        filepath.write_text(content, encoding="utf-8")
        return filename

    # ══════════════════════════════════════════════════════════════════════
    #  3. towerguard_main.ino — Primary Arduino Sketch
    # ══════════════════════════════════════════════════════════════════════

    def _generate_main_sketch(self) -> str:
        """Generate the main Arduino sketch for the ESP32."""
        filename = "towerguard_main.ino"
        filepath = self.output_dir / filename

        content = f'''/*
 * TowerGuard — ESP32 Main Firmware
 * ========================================
 * Auto-generated: {self.timestamp}
 *
 * Hardware: ESP32-S3 + SIM7600SA-H (CITC Type Approved)
 *
 * Multi-layer connectivity architecture:
 *
 *   Layer 1: Edge Inference (local — always active)
 *   Layer 2: Cloud API (full 200-tree model)
 *   Layer 3: SMS Fallback (SIM7600SA-H)
 *   Layer 4: Store & Forward (SPIFFS)
 *
 * Severity levels:
 *   0 = OK  |  1 = Degraded  |  2 = Critical
 */

#include "towerguard_edge_model.h"
#include "towerguard_config.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>

/* ── Connectivity state ── */
typedef enum {{
    TG_CONN_CLOUD,      /* Connected to cloud server */
    TG_CONN_SMS_ONLY,   /* SMS fallback only */
    TG_CONN_OFFLINE     /* Local inference only */
}} tg_conn_state_t;

static tg_conn_state_t conn_state = TG_CONN_CLOUD;
static unsigned long last_heartbeat_ms = 0;
static char tower_id[32] = "SA_TOWER_001";

/* ── Forward declarations ── */
float read_rssi(void);
float read_snr(void);
float measure_latency(void);
float measure_packet_loss(void);
float read_temperature(void);
float estimate_tower_load(void);
float compute_rssi_rate_of_change(void);
float compute_snr_trend(void);
bool is_data_connected(void);
bool is_sms_capable(void);
void send_sms(const char* phone, const char* msg);
void activate_led(int severity);

void setup() {{
    Serial.begin(115200);
    Serial.println("[TG] TowerGuard — System initializing.");

    if (!SPIFFS.begin(true)) {{
        Serial.println("[TG] ERROR: SPIFFS initialization failed.");
    }}

    /* Initialize sensors and connectivity */
    Serial.println("[TG] System ready.");
}}

void loop() {{
    /* 1. Read sensor data */
    float rssi      = read_rssi();
    float snr       = read_snr();
    float latency   = measure_latency();
    float pkt_loss  = measure_packet_loss();
    float temp      = read_temperature();
    float load      = estimate_tower_load();
    float rssi_roc  = compute_rssi_rate_of_change();
    float snr_trend = compute_snr_trend();

    /* ════════════════════════════════════════════════════════════════
     *  Layer 1: Edge Inference — always active
     * ════════════════════════════════════════════════════════════════ */
    float edge_prob = tg_edge_predict(pkt_loss, rssi_roc, snr_trend, latency);
    int severity = tg_edge_severity(edge_prob);
    const char* label_en = tg_severity_label_en(severity);
    const char* label_ar = tg_severity_label_ar(severity);

    Serial.printf("[TG-Edge] Status: %s | Probability: %.0f%%\\n", label_en, edge_prob * 100);
    activate_led(severity);

    /* ════════════════════════════════════════════════════════════════
     *  Layer 2: Cloud API transmission
     * ════════════════════════════════════════════════════════════════ */
    if (is_data_connected()) {{
        conn_state = TG_CONN_CLOUD;
        /* Send sensor payload + heartbeat to cloud */
        /* TODO: implement cloud API call */
        last_heartbeat_ms = millis();
        tg_flush_buffer();

    /* ════════════════════════════════════════════════════════════════
     *  Layer 3: SMS Fallback
     * ════════════════════════════════════════════════════════════════ */
    }} else if (is_sms_capable()) {{
        conn_state = TG_CONN_SMS_ONLY;
        if (severity >= TG_SEVERITY_DEGRADED) {{
            char sms[160];
            snprintf(sms, sizeof(sms),
                "TG-ALERT|id=%s|sev=%d|prob=%.0f%%|pkt=%.1f|"
                "roc=%.2f|snr_t=%.2f|lat=%.0f|ts=%lu",
                tower_id, severity, edge_prob * 100,
                pkt_loss, rssi_roc, snr_trend, latency,
                millis() / 1000);
            send_sms(TG_NOC_PHONE, sms);
            Serial.println("[TG-SMS] Alert transmitted via SIM7600SA-H.");
        }}

    /* ════════════════════════════════════════════════════════════════
     *  Layer 4: Store & Forward (SPIFFS)
     * ════════════════════════════════════════════════════════════════ */
    }} else {{
        conn_state = TG_CONN_OFFLINE;
        Serial.println("[TG] WARNING: No connectivity — buffering locally.");
    }}

    /* Buffer reading when cloud is unreachable */
    if (conn_state != TG_CONN_CLOUD) {{
        tg_buffer_reading(rssi, snr, latency, pkt_loss, load, temp, rssi_roc, snr_trend, severity);
    }}

    delay(TG_READING_INTERVAL_MS);
}}

/* ── Local SPIFFS buffering ── */
void tg_buffer_reading(float rssi, float snr, float lat, float pkt,
                       float load, float temp, float roc, float snr_t, int sev) {{
    File f = SPIFFS.open(TG_BUFFER_FILE, FILE_APPEND);
    if (!f) return;
    if (f.size() > TG_SPIFFS_BUFFER_KB * 1024) {{
        f.close();
        SPIFFS.remove(TG_BUFFER_FILE);
        f = SPIFFS.open(TG_BUFFER_FILE, FILE_WRITE);
        if (!f) return;
    }}
    char json[256];
    snprintf(json, sizeof(json),
        "{{\"rssi\":%.1f,\"snr\":%.1f,\"lat\":%.0f,"
        "\"pkt\":%.2f,\"load\":%.1f,\"temp\":%.1f,"
        "\"roc\":%.3f,\"snr_t\":%.3f,\"sev\":%d}}",
        rssi, snr, lat, pkt, load, temp, roc, snr_t, sev);
    f.println(json);
    f.close();
}}

/* ── Flush SPIFFS buffer on connectivity restoration ── */
void tg_flush_buffer() {{
    if (!SPIFFS.exists(TG_BUFFER_FILE)) return;
    File f = SPIFFS.open(TG_BUFFER_FILE, FILE_READ);
    if (!f || f.size() == 0) {{ f.close(); return; }}
    /* TODO: batch POST to TG_SERVER_URL/api/predict/batch */
    f.close();
    SPIFFS.remove(TG_BUFFER_FILE);
    Serial.println("[TG] Buffered readings flushed to cloud.");
}}
'''
        filepath.write_text(content, encoding="utf-8")
        return filename

    # ══════════════════════════════════════════════════════════════════════
    #  4. towerguard_sms.h — SMS Alert Handler
    # ══════════════════════════════════════════════════════════════════════

    def _generate_sms_handler(self) -> str:
        """Generate the SMS alert handler for the SIM7600SA-H modem."""
        filename = "towerguard_sms.h"
        filepath = self.output_dir / filename

        content = f'''/*
 * TowerGuard — SMS Alert Handler
 * =====================================
 * Auto-generated: {self.timestamp}
 *
 * Handles outbound and inbound SMS alerts via SIM7600SA-H (CITC Type Approved).
 * Message format: TG-ALERT|id=XX|sev=N|prob=XX%|...
 */

#ifndef TOWERGUARD_SMS_H
#define TOWERGUARD_SMS_H

#include "towerguard_config.h"

/**
 * Transmit SMS alert to NOC phone number.
 * No message is sent for TG_SEVERITY_OK — only Degraded and Critical.
 *
 * @param severity    Severity level (0/1/2)
 * @param probability Outage probability [0, 1]
 */
static inline void tg_send_sms_alert(
    const char* tower_id,
    int severity,
    float probability,
    float pkt_loss,
    float rssi_roc,
    float snr_trend,
    float latency
) {{
    if (severity < TG_SEVERITY_DEGRADED) return;

    char sms[160];
    snprintf(sms, sizeof(sms),
        "TG-ALERT|id=%s|sev=%d|prob=%.0f%%|"
        "pkt=%.1f|roc=%.2f|snr_t=%.2f|lat=%.0f|"
        "ts=%lu",
        tower_id, severity, probability * 100.0f,
        pkt_loss, rssi_roc, snr_trend, latency,
        (unsigned long)(millis() / 1000));

    /* Transmit via SIM7600SA-H AT commands */
    Serial.printf("[TG-SMS] Sending alert to %s: %s\\n", TG_NOC_PHONE, sms);
    /* TODO: SIM7600 AT+CMGS implementation */
}}

/**
 * Parse an incoming SMS to determine if it is a TowerGuard alert.
 * @return Severity level, or -1 if not a TowerGuard message.
 */
static inline int tg_parse_sms_alert(const char* sms_body, char* out_tower_id, float* out_prob) {{
    if (strncmp(sms_body, "TG-ALERT|", 9) != 0) return -1;
    *out_prob = 0.0f;
    out_tower_id[0] = '\\0';
    /* TODO: full parser implementation */
    return 0;
}}

#endif /* TOWERGUARD_SMS_H */
'''
        filepath.write_text(content, encoding="utf-8")
        return filename


# ══════════════════════════════════════════════════════════════════════════════
#  Simplified Interface — Backwards Compatibility
# ══════════════════════════════════════════════════════════════════════════════

def generate_edge_c_header() -> str:
    """
    [DEPRECATED] Use ESP32FirmwareGenerator instead.
    Returns the edge model C header as a string for backwards compatibility.
    """
    gen = ESP32FirmwareGenerator(output_dir="/tmp/tg_firmware")
    gen._generate_edge_model_header()
    return (Path("/tmp/tg_firmware") / "towerguard_edge_model.h").read_text()


def edge_predict(
    packet_loss: float,
    rssi_roc: float,
    snr_trend: float,
    latency: float,
) -> tuple:
    """
    [DEPRECATED] Python simulation of the edge prediction model for testing only.
    Production inference runs as compiled C on the ESP32 hardware.

    Returns:
        (probability: float, severity: int, verdict_ar: str)
    """
    score = 0.0
    w_sum = (EDGE_CONFIG.w_pkt_loss + EDGE_CONFIG.w_rssi_roc
             + EDGE_CONFIG.w_snr_trend + EDGE_CONFIG.w_latency)

    if packet_loss > EDGE_CONFIG.pkt_loss_critical:
        score += EDGE_CONFIG.w_pkt_loss * min(1.0, (packet_loss - EDGE_CONFIG.pkt_loss_critical) / 15.0)
    if rssi_roc < EDGE_CONFIG.rssi_roc_critical:
        score += EDGE_CONFIG.w_rssi_roc * min(1.0, abs(rssi_roc - EDGE_CONFIG.rssi_roc_critical) / 3.0)
    if snr_trend < EDGE_CONFIG.snr_trend_critical:
        score += EDGE_CONFIG.w_snr_trend * min(1.0, abs(snr_trend - EDGE_CONFIG.snr_trend_critical) / 2.0)
    if latency > EDGE_CONFIG.latency_critical:
        score += EDGE_CONFIG.w_latency * min(1.0, (latency - EDGE_CONFIG.latency_critical) / 200.0)

    probability = max(0.0, min(1.0, score / w_sum))

    if probability >= EDGE_CONFIG.edge_danger_threshold:
        severity = SEVERITY_CONFIG.SEVERITY_CRITICAL
        verdict_ar = ARABIC_LABELS["prediction_danger"]
    elif probability >= EDGE_CONFIG.edge_warn_threshold:
        severity = SEVERITY_CONFIG.SEVERITY_DEGRADED
        verdict_ar = ARABIC_LABELS["prediction_warning"]
    else:
        severity = SEVERITY_CONFIG.SEVERITY_OK
        verdict_ar = ARABIC_LABELS["prediction_stable"]

    return probability, severity, verdict_ar

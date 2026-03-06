/*
 * towerguard_config.h — ESP32 Configuration
 * =============================================
 * Source: backend/config.py -> EDGE_CONFIG
 * Do NOT edit manually — regenerate from edge_inference.py
 */

#ifndef TOWERGUARD_CONFIG_H
#define TOWERGUARD_CONFIG_H

/* ── إعدادات الاتصال ── */
#define TG_SERVER_URL           "https://api.towerguard.sa"
#define TG_NOC_PHONE            "+966500000000"

/* ── إعدادات التوقيت ── */
#define TG_HEARTBEAT_INTERVAL_MS  30000
#define TG_HEARTBEAT_TIMEOUT_MS   90000
#define TG_READING_INTERVAL_MS    30000

/* ── إعدادات التخزين المحلي ── */
#define TG_SPIFFS_BUFFER_KB     256
#define TG_MAX_BATCH_SIZE       500
#define TG_BUFFER_FILE          "/tg_buffer.jsonl"

/* ── عتبات التنبؤ المحلي ── */
#define TG_PKT_LOSS_CRITICAL    12.0f
#define TG_RSSI_ROC_CRITICAL    -1.5f
#define TG_SNR_TREND_CRITICAL   -0.6f
#define TG_LATENCY_CRITICAL     120.0f
#define TG_THRESHOLD_DANGER     0.60f
#define TG_THRESHOLD_WARN       0.35f

/* ── أوزان المؤشرات (من Feature Importance) ── */
#define TG_W_PKT_LOSS           0.274f
#define TG_W_RSSI_ROC           0.256f
#define TG_W_SNR_TREND          0.250f
#define TG_W_LATENCY            0.220f

/* ── مستويات الخطورة (Multi-Class) ── */
#define TG_SEVERITY_OK          0    /* مستقر */
#define TG_SEVERITY_DEGRADED    1    /* متدهور */
#define TG_SEVERITY_CRITICAL    2    /* حرج */

#endif /* TOWERGUARD_CONFIG_H */

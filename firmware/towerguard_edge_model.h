/*
 * towerguard_edge_model.h — Lightweight Edge Prediction Model
 * =============================================================
 * Runs on ESP32 without server connectivity.
 * Uses 4 features (78.5% of full model importance).
 *
 * Outputs (severity_score):
 *   0 = OK        (stable)
 *   1 = Degraded  (monitor)
 *   2 = Critical  (immediate action)
 */

#ifndef TOWERGUARD_EDGE_MODEL_H
#define TOWERGUARD_EDGE_MODEL_H

#include "towerguard_config.h"
#include <math.h>

/* ── دوال مساعدة ── */
static inline float tg_clampf(float val, float lo, float hi) {
    return val < lo ? lo : (val > hi ? hi : val);
}

/**
 * تنبؤ محلي خفيف — O(1)، بدون تخصيص ذاكرة ديناميكية.
 * @return احتمال الانقطاع [0.0, 1.0]
 */
static inline float tg_edge_predict(
    float packet_loss_pct,
    float rssi_rate_of_change,
    float snr_trend,
    float latency_ms
) {
    float score = 0.0f;
    const float w_sum = TG_W_PKT_LOSS + TG_W_RSSI_ROC + TG_W_SNR_TREND + TG_W_LATENCY;

    /* فقدان الحزم */
    if (packet_loss_pct > TG_PKT_LOSS_CRITICAL) {
        score += TG_W_PKT_LOSS * tg_clampf(
            (packet_loss_pct - TG_PKT_LOSS_CRITICAL) / 15.0f, 0.0f, 1.0f);
    }

    /* معدل تغير RSSI (سالب = أسوأ) */
    if (rssi_rate_of_change < TG_RSSI_ROC_CRITICAL) {
        score += TG_W_RSSI_ROC * tg_clampf(
            fabsf(rssi_rate_of_change - TG_RSSI_ROC_CRITICAL) / 3.0f, 0.0f, 1.0f);
    }

    /* اتجاه SNR (سالب = أسوأ) */
    if (snr_trend < TG_SNR_TREND_CRITICAL) {
        score += TG_W_SNR_TREND * tg_clampf(
            fabsf(snr_trend - TG_SNR_TREND_CRITICAL) / 2.0f, 0.0f, 1.0f);
    }

    /* زمن الاستجابة */
    if (latency_ms > TG_LATENCY_CRITICAL) {
        score += TG_W_LATENCY * tg_clampf(
            (latency_ms - TG_LATENCY_CRITICAL) / 200.0f, 0.0f, 1.0f);
    }

    return tg_clampf(score / w_sum, 0.0f, 1.0f);
}

/**
 * تصنيف متعدد المستويات (Multi-Class Severity).
 * @return TG_SEVERITY_OK (0) | TG_SEVERITY_DEGRADED (1) | TG_SEVERITY_CRITICAL (2)
 */
static inline int tg_edge_severity(float probability) {
    if (probability >= TG_THRESHOLD_DANGER) return TG_SEVERITY_CRITICAL;
    if (probability >= TG_THRESHOLD_WARN)   return TG_SEVERITY_DEGRADED;
    return TG_SEVERITY_OK;
}

/**
 * اسم الحالة بالعربية — للعرض على شاشة الجهاز.
 */
static inline const char* tg_severity_label_ar(int severity) {
    switch (severity) {
        case TG_SEVERITY_CRITICAL: return "حرج";
        case TG_SEVERITY_DEGRADED: return "متدهور";
        default:                   return "مستقر";
    }
}

#endif /* TOWERGUARD_EDGE_MODEL_H */

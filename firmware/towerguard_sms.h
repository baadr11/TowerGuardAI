/*
 * towerguard_sms.h — SMS Alert Handler
 * ========================================
 * Thread-safe sms_stats via FreeRTOS Mutex.
 *
 * Problem: sms_stats written from Core 0 (tg_sms_task) and read from
 *   Core 1 (loop) — torn reads on ESP32 Xtensa LX6 dual-core.
 * Solution: SemaphoreHandle_t + tg_stat_inc() wrapper.
 * Note: xSemaphoreCreateMutex() (not Binary) ensures Priority Inheritance
 *   and prevents Priority Inversion in multi-task environment.
 */

#ifndef TOWERGUARD_SMS_H
#define TOWERGUARD_SMS_H

#include "towerguard_config.h"
#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <freertos/queue.h>
#include <freertos/semphr.h>  /* FreeRTOS Mutex */

/* ── إعدادات SMS ── */
#define TG_SMS_QUEUE_SIZE       8
#define TG_SMS_TASK_STACK       4096
#define TG_SMS_TASK_PRIORITY    1
#define TG_SMS_AT_TIMEOUT_MS    10000
#define TG_SMS_MAX_RETRIES      3
#define TG_SMS_RETRY_BASE_MS    2000
#define TG_SMS_COOLDOWN_MS      30000

/* ── هيكل رسالة SMS ── */
typedef struct {
    char          phone[20];
    char          body[161];
    uint8_t       retry_count;
    unsigned long enqueued_at;
} tg_sms_msg_t;

/* ── إحصائيات SMS ── */
typedef struct {
    volatile uint32_t      sent_ok;
    volatile uint32_t      sent_fail;
    volatile uint32_t      queue_full;
    volatile uint32_t      throttled;
    volatile unsigned long last_sent_ms;
} tg_sms_stats_t;

static tg_sms_stats_t    sms_stats       = {0, 0, 0, 0, 0};
static QueueHandle_t     sms_queue       = NULL;
static TaskHandle_t      sms_task_handle = NULL;
static unsigned long     last_sms_by_severity[3] = {0, 0, 0};

/* Mutex — declared here, created in tg_sms_init().
 * All sms_stats access goes through tg_stat_inc() / tg_stat_snapshot(). */
static SemaphoreHandle_t sms_mutex = NULL;

/* ── Thread-safe increment ── */
static inline void tg_stat_inc(volatile uint32_t* field) {
    if (sms_mutex && xSemaphoreTake(sms_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        (*field)++;
        xSemaphoreGive(sms_mutex);
    }
    /* إذا انتهت المهلة (timeout)، نُسقط التحديث ونُسجّل في Serial */
    /* أفضل من الانتظار اللانهائي الذي قد يجمّد SMS Task */
}

/* ── Thread-safe read snapshot ── */
static inline tg_sms_stats_t tg_stat_snapshot() {
    tg_sms_stats_t snap = {0};
    if (sms_mutex && xSemaphoreTake(sms_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
        snap = sms_stats;
        xSemaphoreGive(sms_mutex);
    }
    return snap;
}

/* ── Thread-safe update last_sent_ms ── */
static inline void tg_stat_set_last_sent(unsigned long ms) {
    if (sms_mutex && xSemaphoreTake(sms_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
        sms_stats.last_sent_ms = ms;
        xSemaphoreGive(sms_mutex);
    }
}

/* ══════════════════════════════════════════════════════════════════════════
 *  AT Command Helper
 * ══════════════════════════════════════════════════════════════════════════ */
static bool tg_at_send_with_timeout(const char* cmd, const char* expect,
                                     unsigned long timeout_ms) {
    Serial2.println(cmd);
    unsigned long start    = millis();
    String        response = "";

    while ((millis() - start) < timeout_ms) {
        if (Serial2.available()) {
            char c = Serial2.read();
            response += c;
            if (response.indexOf(expect) >= 0) return true;
            if (response.indexOf("ERROR") >= 0) return false;
        }
        vTaskDelay(pdMS_TO_TICKS(10));
    }
    Serial.printf("[TG-SMS] ⏰ مهلة AT (%lu ms)\n", timeout_ms);
    return false;
}

/* ══════════════════════════════════════════════════════════════════════════
 *  إرسال SMS عبر SIM7600
 * ══════════════════════════════════════════════════════════════════════════ */
static bool tg_sim7600_send_sms(const char* phone, const char* body) {
    if (!tg_at_send_with_timeout("AT",           "OK",  2000)) return false;
    if (!tg_at_send_with_timeout("AT+CMGF=1",    "OK",  2000)) return false;

    char cmd[48];
    snprintf(cmd, sizeof(cmd), "AT+CMGS=\"%s\"", phone);
    if (!tg_at_send_with_timeout(cmd, ">", TG_SMS_AT_TIMEOUT_MS)) return false;

    Serial2.print(body);
    Serial2.write(0x1A);  /* Ctrl+Z */

    return tg_at_send_with_timeout("", "+CMGS:", TG_SMS_AT_TIMEOUT_MS);
}

/* ══════════════════════════════════════════════════════════════════════════
 *  FreeRTOS SMS Task — يعمل على Core 0
 * ══════════════════════════════════════════════════════════════════════════ */
static void tg_sms_task(void* pv) {
    tg_sms_msg_t msg;
    for (;;) {
        if (xQueueReceive(sms_queue, &msg, portMAX_DELAY) == pdTRUE) {
            bool sent = false;
            for (int attempt = 0; attempt < TG_SMS_MAX_RETRIES && !sent; attempt++) {
                if (attempt > 0) {
                    unsigned long wait_ms = TG_SMS_RETRY_BASE_MS * (1UL << attempt);
                    vTaskDelay(pdMS_TO_TICKS(wait_ms));
                }
                sent = tg_sim7600_send_sms(msg.phone, msg.body);
            }

            /* All sms_stats updates go through Mutex wrapper */
            if (sent) {
                tg_stat_inc(&sms_stats.sent_ok);
                tg_stat_set_last_sent(millis());
                Serial.println("[TG-SMS] ✅ SMS أُرسلت");
            } else {
                tg_stat_inc(&sms_stats.sent_fail);
                Serial.println("[TG-SMS] ❌ فشل الإرسال بعد كل المحاولات");
            }
        }
    }
}

/* SMS Initialization — Mutex created before task launch */
static inline void tg_sms_init() {
    /* Create Mutex first, before any task */
    sms_mutex = xSemaphoreCreateMutex();
    if (!sms_mutex) {
        Serial.println("[TG-SMS] ❌ FATAL: فشل إنشاء sms_mutex!");
        return;  /* لا نُكمل بدون Mutex */
    }

    sms_queue = xQueueCreate(TG_SMS_QUEUE_SIZE, sizeof(tg_sms_msg_t));
    if (!sms_queue) {
        Serial.println("[TG-SMS] ❌ فشل إنشاء طابور SMS");
        return;
    }

    BaseType_t rc = xTaskCreatePinnedToCore(
        tg_sms_task, "tg_sms", TG_SMS_TASK_STACK, NULL,
        TG_SMS_TASK_PRIORITY, &sms_task_handle,
        0  /* Core 0 — بينما loop() على Core 1 */
    );

    if (rc == pdPASS) {
        Serial.println("[TG-SMS] SMS Task + Mutex ready");
    } else {
        Serial.println("[TG-SMS] ❌ فشل إنشاء SMS Task");
    }
}

/* Enqueue an SMS alert (thread-safe) */
static inline bool tg_enqueue_sms_alert(
    const char* tower_id, int severity, float probability,
    float pkt_loss, float rssi_roc, float snr_trend, float latency
) {
    if (severity < TG_SEVERITY_DEGRADED) return false;
    if (!sms_queue)                       return false;

    unsigned long now = millis();
    if ((now - last_sms_by_severity[severity]) < TG_SMS_COOLDOWN_MS) {
        tg_stat_inc(&sms_stats.throttled);  /* thread-safe across cores */
        return false;
    }

    tg_sms_msg_t msg;
    memset(&msg, 0, sizeof(msg));
    strncpy(msg.phone, TG_NOC_PHONE, sizeof(msg.phone) - 1);
    msg.retry_count  = 0;
    msg.enqueued_at  = now;

    snprintf(msg.body, sizeof(msg.body),
        "TG-ALERT|id=%s|sev=%d|prob=%.0f%%|"
        "pkt=%.1f|roc=%.2f|snr_t=%.2f|lat=%.0f|"
        "ts=%lu",
        tower_id, severity, probability * 100.0f,
        pkt_loss, rssi_roc, snr_trend, latency,
        (unsigned long)(now / 1000));

    if (xQueueSend(sms_queue, &msg, 0) == pdTRUE) {
        last_sms_by_severity[severity] = now;
        Serial.printf("[TG-SMS] 📨 تنبيه أُدرج — الخطورة: %d\n", severity);
        return true;
    } else {
        tg_stat_inc(&sms_stats.queue_full);  /* thread-safe */
        Serial.println("[TG-SMS] ⚠ الطابور ممتلئ");
        return false;
    }
}

#endif /* TOWERGUARD_SMS_H */

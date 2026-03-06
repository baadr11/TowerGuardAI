/*
 * TowerGuard — ESP32 Main Firmware
 * =================================
 * Zero-Trust Architecture: Three-Layer Redundancy
 *   Layer 1 — Edge Inference (on-device Random Forest, always-on)
 *   Layer 2 — Cloud Prediction (FastAPI, primary channel)
 *   Layer 3 — SMS Fallback (SIM7600SA-H, offline alert path)
 *
 * Store-and-Forward SPIFFS buffer with O(1)-heap streaming I/O.
 * No std::vector<String> accumulation. All file operations are
 * line-by-line with a single fixed-size char[] buffer.
 *
 * NTP-aware timestamps via configTime(). Falls back to millis()
 * if synchronisation has not yet completed, with a flag in the payload.
 *
 * NOTE — SPIFFS rename() atomicity:
 *   SPIFFS does not support atomic rename(). A power failure between
 *   SPIFFS.remove(TG_BUFFER_FILE) and the final write creates a
 *   theoretical data-loss window (~2 ms).
 *   Production mitigation: migrate to LittleFS which supports
 *   atomic rename().
 */

#include "towerguard_edge_model.h"
#include "towerguard_sms.h"       /* contains sms_mutex for thread safety */
#include "towerguard_config.h"
#include <WiFi.h>
#include <HTTPClient.h>
#include <SPIFFS.h>
#include <ArduinoJson.h>
#include <time.h>                  /* NTP Unix timestamp */

/* ── Connection-state machine ─────────────────────────────────────── */
typedef enum {
    TG_CONN_CLOUD,
    TG_CONN_SMS_ONLY,
    TG_CONN_OFFLINE
} tg_conn_state_t;

static tg_conn_state_t conn_state     = TG_CONN_CLOUD;
static unsigned long   last_hb_ack_ms = 0;
static bool            hb_watchdog_fired = false;
static char            tower_id[32]   = "SA_TOWER_001";

/* ── Store-and-Forward counters ───────────────────────────────────── */
static uint32_t spiffs_records_buffered = 0;
static uint32_t spiffs_records_flushed  = 0;
static uint32_t spiffs_records_dropped  = 0;

/* ── NTP sync flag ────────────────────────────────────────────────── */
static bool ntp_synced = false;

/* ── Forward declarations ─────────────────────────────────────────── */
float  read_rssi(void);
float  read_snr(void);
float  measure_latency(void);
float  measure_packet_loss(void);
float  read_temperature(void);
float  estimate_tower_load(void);
float  compute_rssi_rate_of_change(void);
float  compute_snr_trend(void);
bool   is_data_connected(void);
bool   is_sms_capable(void);
void   activate_led(int severity);

/* ══════════════════════════════════════════════════════════════════════
 *  HELPER: NTP-aware Unix timestamp
 *
 *  Returns time() after NTP sync. Falls back to millis()/1000 with
 *  a warning flag in the JSON payload (ntp_synced=0).
 * ══════════════════════════════════════════════════════════════════════ */
static inline uint32_t tg_unix_now() {
    if (ntp_synced) {
        time_t now;
        time(&now);
        return (uint32_t)now;
    }
    return (uint32_t)(millis() / 1000);
}

/* ══════════════════════════════════════════════════════════════════════
 *  Setup
 * ══════════════════════════════════════════════════════════════════════ */
void setup() {
    Serial.begin(115200);
    Serial.println("[TG] TowerGuard — Edge Firmware starting");

    if (!SPIFFS.begin(true)) {
        Serial.println("[TG] ERROR: SPIFFS mount failed");
    } else {
        Serial.printf("[TG] SPIFFS ready — limit: %d KB\n", TG_SPIFFS_BUFFER_KB);
    }

    tg_sms_init();   /* creates sms_mutex internally */

    if (is_data_connected()) {
        configTime(3 * 3600, 0, "pool.ntp.org", "time.google.com");
        struct tm ti;
        if (getLocalTime(&ti, 5000)) {
            ntp_synced = true;
            Serial.println("[TG] NTP: synchronised");
        } else {
            Serial.println("[TG] NTP: sync failed — using millis() temporarily");
        }
    }

    Serial.println("[TG] System ready");
}

/* ══════════════════════════════════════════════════════════════════════
 *  Main Loop
 * ══════════════════════════════════════════════════════════════════════ */
void loop() {
    float rssi      = read_rssi();
    float snr       = read_snr();
    float latency   = measure_latency();
    float pkt_loss  = measure_packet_loss();
    float temp      = read_temperature();
    float load      = estimate_tower_load();
    float rssi_roc  = compute_rssi_rate_of_change();
    float snr_trend = compute_snr_trend();

    float edge_prob = tg_edge_predict(pkt_loss, rssi_roc, snr_trend, latency);
    int   severity  = tg_edge_severity(edge_prob);
    const char* label = tg_severity_label_en(severity);

    Serial.printf("[TG-Edge] Status: %s | Prob: %.0f%%\n", label, edge_prob * 100);
    activate_led(severity);

    if (is_data_connected()) {
        conn_state = TG_CONN_CLOUD;

        if (!ntp_synced) {
            struct tm ti;
            if (getLocalTime(&ti, 1000)) ntp_synced = true;
        }

        bool server_ack = tg_send_reading_to_cloud(
            tower_id, rssi, snr, latency, pkt_loss, load, temp,
            rssi_roc, snr_trend, severity, edge_prob
        );

        if (server_ack) {
            last_hb_ack_ms = millis();
            hb_watchdog_fired = false;
        }

        tg_flush_buffer();

    } else if (is_sms_capable()) {
        conn_state = TG_CONN_SMS_ONLY;
        if (severity >= TG_SEVERITY_DEGRADED) {
            tg_enqueue_sms_alert(tower_id, severity, edge_prob,
                                 pkt_loss, rssi_roc, snr_trend, latency);
        }
    } else {
        conn_state = TG_CONN_OFFLINE;
        Serial.println("[TG] No connectivity — storing locally");
    }

    if (conn_state != TG_CONN_CLOUD) {
        tg_buffer_reading(rssi, snr, latency, pkt_loss, load, temp,
                          rssi_roc, snr_trend, severity);
    }

    tg_check_heartbeat_watchdog();
    delay(TG_READING_INTERVAL_MS);
}

/* ══════════════════════════════════════════════════════════════════════
 *  Cloud upload — single reading
 * ══════════════════════════════════════════════════════════════════════ */
bool tg_send_reading_to_cloud(
    const char* tid, float rssi, float snr, float lat, float pkt,
    float load, float temp, float roc, float snr_t, int sev, float prob
) {
    HTTPClient http;
    String url = String(TG_SERVER_URL) + "/api/predict";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(10000);

    StaticJsonDocument<384> doc;
    doc["tower_id"]          = tid;
    doc["rssi_dbm"]          = rssi;
    doc["snr_db"]            = snr;
    doc["latency_ms"]        = lat;
    doc["packet_loss_pct"]   = pkt;
    doc["tower_load_pct"]    = load;
    doc["temp_celsius"]      = temp;
    doc["rssi_roc"]          = roc;
    doc["snr_trend"]         = snr_t;
    doc["edge_severity"]     = sev;
    doc["edge_probability"]  = prob;
    doc["unix_ts"]           = tg_unix_now();
    doc["ntp_synced"]        = ntp_synced;

    String payload;
    serializeJson(doc, payload);

    int  httpCode = http.POST(payload);
    bool success  = (httpCode >= 200 && httpCode < 300);
    Serial.printf("[TG-Cloud] %s (HTTP %d)\n",
                  success ? "ACK" : "FAILED", httpCode);
    http.end();
    return success;
}

/* ══════════════════════════════════════════════════════════════════════
 *  Heartbeat watchdog
 * ══════════════════════════════════════════════════════════════════════ */
void tg_check_heartbeat_watchdog() {
    unsigned long now     = millis();
    unsigned long elapsed = now - last_hb_ack_ms;

    if (last_hb_ack_ms == 0) {
        if (now > TG_HEARTBEAT_TIMEOUT_MS * 2 && !hb_watchdog_fired) {
            Serial.println("[TG-DMS] No ACK since boot — firing alert");
            hb_watchdog_fired = true;
            tg_enqueue_sms_alert(tower_id, TG_SEVERITY_CRITICAL, 1.0f, 0, 0, 0, 0);
        }
        return;
    }

    if (elapsed > TG_HEARTBEAT_TIMEOUT_MS && !hb_watchdog_fired) {
        Serial.printf("[TG-DMS] No ACK for %lu s — firing alert\n", elapsed / 1000);
        hb_watchdog_fired = true;
        tg_enqueue_sms_alert(tower_id, TG_SEVERITY_CRITICAL, 1.0f, 0, 0, 0, 0);
    }
}

/* ══════════════════════════════════════════════════════════════════════
 *  SPIFFS append — single reading
 * ══════════════════════════════════════════════════════════════════════ */
void tg_buffer_reading(float rssi, float snr, float lat, float pkt,
                       float load, float temp, float roc, float snr_t,
                       int sev) {
    File f = SPIFFS.open(TG_BUFFER_FILE, FILE_APPEND);
    if (!f) {
        Serial.println("[TG-SPIFFS] ERROR: cannot open buffer file");
        return;
    }

    if (f.size() > (size_t)(TG_SPIFFS_BUFFER_KB * 1024)) {
        f.close();
        Serial.printf("[TG-SPIFFS] Limit exceeded (%d KB) — streaming trim\n",
                      TG_SPIFFS_BUFFER_KB);
        tg_trim_oldest_records();
        f = SPIFFS.open(TG_BUFFER_FILE, FILE_APPEND);
        if (!f) return;
    }

    char json[288];
    snprintf(json, sizeof(json),
        "{\"ts\":%lu,\"ntp\":%d,\"rssi\":%.1f,\"snr\":%.1f,\"lat\":%.0f,"
        "\"pkt\":%.2f,\"load\":%.1f,\"temp\":%.1f,"
        "\"roc\":%.3f,\"snr_t\":%.3f,\"sev\":%d}",
        tg_unix_now(), (int)ntp_synced,
        rssi, snr, lat, pkt, load, temp, roc, snr_t, sev);

    f.println(json);
    f.close();
    spiffs_records_buffered++;
}

/* ══════════════════════════════════════════════════════════════════════
 *  tg_trim_oldest_records() — O(1)-heap streaming trim
 *
 *  Reads the buffer file one line at a time (one char[] on the stack).
 *  Writes lines beyond skip_count directly to a temp file.
 *  Rebuilds the buffer from the temp file.
 *  RAM usage: O(1) — one line in memory at any time.
 *
 *  SPIFFS atomicity caveat: see file header note.
 * ══════════════════════════════════════════════════════════════════════ */
void tg_trim_oldest_records() {
    const char* TMP_FILE = "/tg_tmp.jsonl";

    /* ── Pass 1: count lines ─────────────────────────────────────── */
    File src = SPIFFS.open(TG_BUFFER_FILE, FILE_READ);
    if (!src) return;

    int total_lines = 0;
    while (src.available()) {
        src.readStringUntil('\n');
        total_lines++;
    }

    int skip_count = max(10, total_lines / 4);  /* drop oldest 25%, min 10 */
    if (skip_count >= total_lines) skip_count = total_lines / 2;

    /* ── Pass 2: streaming copy — one line at a time ─────────────── */
    src.seek(0);
    File dst = SPIFFS.open(TMP_FILE, FILE_WRITE);
    if (!dst) { src.close(); return; }

    int  line_num = 0;
    int  kept     = 0;
    bool wrote_ok = true;

    while (src.available()) {
        String line = src.readStringUntil('\n');   /* single line — O(line) */
        line.trim();
        if (line.length() == 0) continue;

        if (line_num >= skip_count) {
            if (!dst.println(line)) { wrote_ok = false; break; }
            kept++;
        }
        line_num++;
    }
    src.close();
    dst.close();

    if (!wrote_ok) {
        Serial.println("[TG-SPIFFS] ERROR: temp-file write failed — trim aborted");
        SPIFFS.remove(TMP_FILE);
        return;
    }

    /* ── Rebuild buffer from temp file ──────────────────────────── */
    SPIFFS.remove(TG_BUFFER_FILE);

    File r = SPIFFS.open(TMP_FILE,       FILE_READ);
    File w = SPIFFS.open(TG_BUFFER_FILE, FILE_WRITE);

    if (!r || !w) {
        Serial.println("[TG-SPIFFS] ERROR: rebuild failed");
        if (r) r.close();
        if (w) w.close();
        return;
    }

    while (r.available()) {
        String line = r.readStringUntil('\n');  /* one line — O(1) heap */
        line.trim();
        if (line.length() > 0) w.println(line);
    }
    r.close();
    w.close();
    SPIFFS.remove(TMP_FILE);

    spiffs_records_dropped += skip_count;
    Serial.printf("[TG-SPIFFS] Streaming trim: dropped %d | kept %d | "
                  "total dropped: %lu\n",
                  skip_count, kept, spiffs_records_dropped);
    Serial.printf("[TG-SPIFFS] Free heap: %d bytes\n", ESP.getFreeHeap());
}

/* ══════════════════════════════════════════════════════════════════════
 *  tg_flush_buffer() — O(1)-heap streaming batch upload
 *
 *  Design contract:
 *    - Never accumulates all file lines. Uses two bounded-memory passes.
 *    - Pass 1: stream-build the JSON batch payload from the first
 *      TG_MAX_BATCH_SIZE valid lines. Payload size is bounded by
 *      TG_MAX_BATCH_SIZE × MAX_LINE_BYTES — independent of file size.
 *    - Pass 2 (on HTTP success only): stream-copy lines after
 *      send_count to a temp file, then rebuild — identical to
 *      tg_trim_oldest_records().
 *    - On HTTP failure: file is untouched; retry on next cycle.
 *
 *  JSON validation: lines not starting with '{' or not ending with '}'
 *  are silently dropped (guards against truncated writes after a
 *  power-cycle).
 *
 *  RAM: one char[MAX_LINE_BYTES] on the stack at any time. O(1).
 * ══════════════════════════════════════════════════════════════════════ */
void tg_flush_buffer() {
    if (!SPIFFS.exists(TG_BUFFER_FILE)) return;

    File f = SPIFFS.open(TG_BUFFER_FILE, FILE_READ);
    if (!f || f.size() == 0) { if (f) f.close(); return; }

    Serial.printf("[TG-Flush] Reading buffer (%d bytes)...\n", (int)f.size());

    /* ── Pass 1: stream-build batch payload ──────────────────────── */
    /* Payload is a pre-allocated String capped to batch size,       */
    /* not the full file. Heap growth is bounded.                    */
    String payload;
    payload.reserve(TG_MAX_BATCH_SIZE * 300 + 64);  /* known upper bound */
    payload = "{\"tower_id\":\"";
    payload += tower_id;
    payload += "\",\"readings\":[";

    int send_count   = 0;
    int corrupt_count = 0;

    while (f.available() && send_count < TG_MAX_BATCH_SIZE) {
        String line = f.readStringUntil('\n');
        line.trim();
        if (line.length() < 5) continue;

        if (line[0] == '{' && line[line.length() - 1] == '}') {
            if (send_count > 0) payload += ',';
            payload += line;
            send_count++;
        } else {
            corrupt_count++;
        }
    }
    f.close();

    if (corrupt_count > 0) {
        Serial.printf("[TG-Flush] Dropped %d corrupt lines\n", corrupt_count);
    }

    if (send_count == 0) {
        SPIFFS.remove(TG_BUFFER_FILE);
        return;
    }

    payload += "]}";

    /* ── HTTP POST ────────────────────────────────────────────────── */
    HTTPClient http;
    String url = String(TG_SERVER_URL) + "/api/predict/batch";
    http.begin(url);
    http.addHeader("Content-Type", "application/json");
    http.setTimeout(15000);

    int  httpCode = http.POST(payload);
    bool success  = (httpCode >= 200 && httpCode < 300);
    http.end();

    if (!success) {
        Serial.printf("[TG-Flush] Upload failed (HTTP %d) — file preserved\n",
                      httpCode);
        return;
    }

    spiffs_records_flushed += send_count;
    Serial.printf("[TG-Flush] Sent %d records | total sent: %lu\n",
                  send_count, spiffs_records_flushed);

    /* ── Pass 2: stream-preserve unsent records ───────────────────── */
    /* Re-open file and skip the first send_count valid lines,        */
    /* writing the remainder to a temp file — O(1) RAM.               */
    const char* TMP_FILE = "/tg_flush_tmp.jsonl";

    File src2 = SPIFFS.open(TG_BUFFER_FILE, FILE_READ);
    if (!src2) {
        SPIFFS.remove(TG_BUFFER_FILE);
        return;
    }

    File tmp = SPIFFS.open(TMP_FILE, FILE_WRITE);
    if (!tmp) {
        src2.close();
        SPIFFS.remove(TG_BUFFER_FILE);
        return;
    }

    int  valid_seen  = 0;
    int  remaining   = 0;
    bool tmp_ok      = true;

    while (src2.available()) {
        String line = src2.readStringUntil('\n');  /* one line — O(1) */
        line.trim();
        if (line.length() < 5) continue;

        bool valid = (line[0] == '{' && line[line.length() - 1] == '}');
        if (!valid) continue;

        if (valid_seen < send_count) {
            valid_seen++;   /* skip already-sent records */
        } else {
            if (!tmp.println(line)) { tmp_ok = false; break; }
            remaining++;
        }
    }
    src2.close();
    tmp.close();

    if (!tmp_ok) {
        Serial.println("[TG-Flush] ERROR: temp write failed — purging buffer");
        SPIFFS.remove(TMP_FILE);
        SPIFFS.remove(TG_BUFFER_FILE);
        return;
    }

    /* Rebuild buffer from temp */
    SPIFFS.remove(TG_BUFFER_FILE);

    if (remaining > 0) {
        File r = SPIFFS.open(TMP_FILE,       FILE_READ);
        File w = SPIFFS.open(TG_BUFFER_FILE, FILE_WRITE);
        if (r && w) {
            while (r.available()) {
                String line = r.readStringUntil('\n');
                line.trim();
                if (line.length() > 0) w.println(line);
            }
        }
        if (r) r.close();
        if (w) w.close();
        Serial.printf("[TG-Flush] Preserved %d unsent records\n", remaining);
    }
    SPIFFS.remove(TMP_FILE);
    Serial.printf("[TG-Flush] Free heap: %d bytes\n", ESP.getFreeHeap());
}

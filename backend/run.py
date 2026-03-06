#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║  TowerGuard — Run Server                                   ║
║  الأمر: python run.py                                            ║
╚══════════════════════════════════════════════════════════════════╝

يبدأ خادم FastAPI + WebSocket مع:
  • تحميل النموذج تلقائياً (أو تدريبه من الصفر)
  • مراقب نبض القلب (_heartbeat_watchdog) — مهلة 180 ثانية
  • WebSocket على /ws/towers للتحديثات الفورية
  • 4 صفحات HTML: العرض التقديمي | لوحة التحكم | محاكاة الجهاز | دراسة الجدوى
"""

import os
import sys
import uvicorn


def main():
    # ── Environment defaults (يمكن تعديلها قبل التشغيل) ──────────
    host = os.getenv("TG_HOST", "0.0.0.0")
    port = int(os.getenv("TG_PORT", "8000"))
    reload_mode = os.getenv("TG_RELOAD", "0") == "1"

    # ── Dev mode (يسمح بكل Origins + يُظهر الأخطاء التفصيلية) ────
    os.environ.setdefault("TOWERGUARD_DEV", "1")

    # ── NOC Phone (يُستخدم في الإنذارات بدلاً من hardcoded) ──────
    # os.environ.setdefault("TG_NOC_PHONE", "+966501234567")

    print()
    print("  ╔══════════════════════════════════════════════╗")
    print("  ║        TowerGuard — بدء التشغيل        ║")
    print("  ╠══════════════════════════════════════════════╣")
    print(f"  ║  الخادم:    http://{host}:{port}             ")
    print(f"  ║  WebSocket:  ws://{host}:{port}/ws/towers     ")
    print(f"  ║  الوضع:     {'تطوير 🔧' if os.getenv('TOWERGUARD_DEV') == '1' else 'إنتاج 🔒'}          ")
    print("  ╠══════════════════════════════════════════════╣")
    print("  ║  الصفحات:                                    ║")
    print(f"  ║    /           العرض التقديمي               ")
    print(f"  ║    /predictor  لوحة التحكم + فاصل الثقة     ")
    print(f"  ║    /device     محاكاة ESP32                 ")
    print(f"  ║    /market     دراسة الجدوى + ROI           ")
    print("  ╠══════════════════════════════════════════════╣")
    print("  ║  API:                                        ║")
    print(f"  ║    POST /api/predict         تنبؤ مفرد      ")
    print(f"  ║    POST /api/predict/batch   تنبؤ جماعي     ")
    print(f"  ║    POST /api/heartbeat       نبض القلب      ")
    print(f"  ║    GET  /api/heartbeat/dead  أبراج مفقودة   ")
    print(f"  ║    GET  /api/health          فحص الخادم     ")
    print(f"  ║    GET  /api/model-info      معلومات النموذج ")
    print("  ╚══════════════════════════════════════════════╝")
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

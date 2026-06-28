"""Планировщик периодических задач для Messengers_Analyze.

Запускает задачи (анализ сообщений и т.д.) по расписанию, настраиваемому
через переменные окружения.

Использует APScheduler 4.x (async-native).

Пример запуска:
    python scheduler.py

Переменные окружения для настройки расписания (см. .env.example):
    SCHEDULE_ANALYSIS_CRON  — cron-выражение (по умолчанию: каждый день в 03:00)
    SCHEDULE_ANALYSIS_INTERVAL_MINUTES — альтернатива: интервал в минутах
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys

from config import PROJECT_ROOT  # noqa: F401  — загружает .env

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [scheduler] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Job: запуск daily_analysis.main()
# ---------------------------------------------------------------------------

async def run_daily_analysis() -> None:
    """Обёртка для запуска batch-анализа как scheduled-задачи."""
    logger.info("⏰ Запуск периодического анализа сообщений...")
    try:
        from daily_analysis import main as analysis_main
        await analysis_main()
        logger.info("✅ Анализ завершён успешно")
    except Exception:
        logger.exception("❌ Ошибка при выполнении анализа")


# ---------------------------------------------------------------------------
# Parsing schedule from environment
# ---------------------------------------------------------------------------

def _parse_cron(expression: str) -> dict:
    """Разобрать cron-выражение '* * * * *' в kwargs для CronTrigger.

    Формат: minute hour day month day_of_week
    Примеры:
        '0 3 * * *'    — каждый день в 03:00
        '*/30 * * * *' — каждые 30 минут
        '0 */6 * * *'  — каждые 6 часов
    """
    parts = expression.strip().split()
    if len(parts) != 5:
        raise ValueError(
            f"Cron-выражение должно содержать 5 частей (minute hour day month day_of_week), "
            f"получено {len(parts)}: '{expression}'"
        )

    fields = ["minute", "hour", "day", "month", "day_of_week"]
    return {field: value for field, value in zip(fields, parts)}


# ---------------------------------------------------------------------------
# Main scheduler loop
# ---------------------------------------------------------------------------

async def main() -> None:
    """Запустить планировщик с задачами по расписанию."""
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from apscheduler.triggers.cron import CronTrigger
        from apscheduler.triggers.interval import IntervalTrigger
    except ImportError:
        logger.error(
            "APScheduler не установлен. Выполните: pip install apscheduler>=3.10"
        )
        sys.exit(1)

    scheduler = AsyncIOScheduler()

    # --- Определяем расписание анализа ---
    cron_expr = os.getenv("SCHEDULE_ANALYSIS_CRON", "").strip()
    interval_minutes = os.getenv("SCHEDULE_ANALYSIS_INTERVAL_MINUTES", "").strip()

    if interval_minutes:
        # Интервальный режим (удобно для разработки / частого запуска)
        minutes = int(interval_minutes)
        trigger = IntervalTrigger(minutes=minutes)
        logger.info("📅 Анализ будет запускаться каждые %d мин.", minutes)
    elif cron_expr:
        # Cron-режим
        cron_kwargs = _parse_cron(cron_expr)
        trigger = CronTrigger(**cron_kwargs)
        logger.info("📅 Анализ по cron-расписанию: %s", cron_expr)
    else:
        # По умолчанию — каждый день в 03:00
        trigger = CronTrigger(hour=3, minute=0)
        logger.info("📅 Анализ по умолчанию: каждый день в 03:00")

    scheduler.add_job(
        run_daily_analysis,
        trigger=trigger,
        id="daily_analysis",
        name="Batch analysis of Telegram messages",
        misfire_grace_time=3600,  # допускаем опоздание до 1 часа
    )

    # --- Запуск первого анализа сразу (если включено) ---
    run_on_start = os.getenv("SCHEDULE_RUN_ON_START", "false").lower() in ("1", "true", "yes")
    if run_on_start:
        logger.info("🚀 Запускаем анализ сразу при старте (SCHEDULE_RUN_ON_START=true)...")
        scheduler.add_job(
            run_daily_analysis,
            id="startup_analysis",
            name="Startup analysis run",
        )

    # --- Graceful shutdown ---
    stop_event = asyncio.Event()

    def _handle_signal(*_):
        logger.info("Получен сигнал остановки, завершаем...")
        stop_event.set()

    if sys.platform != "win32":
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, _handle_signal)
    else:
        signal.signal(signal.SIGINT, _handle_signal)

    scheduler.start()
    logger.info("🟢 Планировщик запущен. Ожидание задач...")

    # Печатаем список запланированных задач
    for job in scheduler.get_jobs():
        logger.info("  → %s | Следующий запуск: %s", job.name, job.next_run_time)

    try:
        await stop_event.wait()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        scheduler.shutdown(wait=False)
        logger.info("🔴 Планировщик остановлен")


if __name__ == "__main__":
    asyncio.run(main())

# -*- coding: utf-8 -*-
"""
ШАГ 5. Полный пайплайн обработки данных.

Запускает все шаги по порядку:
    1. Загрузка данных из Excel и API + валидация
    2. Очистка данных
    3. Анализ данных (статистики, временные ряды, ML)
    4. Генерация отчётов (графики, PDF, Excel)
    5. Сохранение результатов в PostgreSQL

Всё логируется в logs/pipeline.log
Скрипт можно поставить на регулярный запуск через Планировщик задач
(см. README и файл run.bat).

Запуск:  python main.py  (или python 05_automation/pipeline.py)
"""

import logging
import os
import sys
import time

# пути ко всем папкам проекта
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "01_loading"))
sys.path.insert(0, os.path.join(BASE_DIR, "02_cleaning"))
sys.path.insert(0, os.path.join(BASE_DIR, "03_analysis"))
sys.path.insert(0, os.path.join(BASE_DIR, "04_reports"))
sys.path.insert(0, os.path.join(BASE_DIR, "05_automation"))

import pandas as pd  # noqa: E402

import config          # noqa: E402
import load_files      # noqa: E402  (шаг 1)
import validation      # noqa: E402
import load_api        # noqa: E402
import load_database   # noqa: E402
import cleaning        # noqa: E402  (шаг 2)
import analysis        # noqa: E402  (шаг 3)
import reports         # noqa: E402  (шаг 4)
import save_results    # noqa: E402  (шаг 5)


def setup_logging():
    """Настраивает логирование: в файл logs/pipeline.log и в консоль."""
    os.makedirs(config.LOGS_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(config.LOGS_DIR, "pipeline.log"),
                                encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("pipeline")


def run_pipeline():
    """Запускает все шаги пайплайна по порядку."""
    log = setup_logging()
    start = time.time()
    log.info("=== Запуск пайплайна ===")

    # ---------- ШАГ 1. Загрузка данных ----------
    log.info("ШАГ 1. Загрузка данных из файлов")
    deals, zhk = load_files.prepare_raw_data()

    # валидация сырых сделок с записью в лог
    validation.validate(deals, "сделки ДДУ")

    # загрузка из внешнего API (курс валют) - если недоступен, продолжаем
    try:
        usd = load_api.load_usd_rate()
        usd.to_csv(os.path.join(config.DATA_DIR, "usd_rate.csv"), index=False)
        log.info("Курс USD с внешнего API: %s руб.", usd["usd_rub"].iloc[0])
    except Exception as e:
        log.warning("API недоступно, пропускаем: %s", e)

    # ---------- ШАГ 2. Очистка данных ----------
    log.info("ШАГ 2. Очистка данных")
    clean, clean_report = cleaning.clean_deals_data(deals)
    clean.to_csv(os.path.join(config.DATA_DIR, "clean_deals.csv"),
                 index=False, encoding="utf-8-sig")
    log.info("Очистка завершена: %s", clean_report)

    # ---------- ШАГ 3. Анализ данных ----------
    log.info("ШАГ 3. Анализ данных (ML + прогноз 2026)")
    metrics, data = analysis.run_analysis(clean=clean, zhk=zhk)
    log.info("Регрессия (предполагаемая цена): %s",
             metrics["регрессия_предполагаемая_цена"])
    log.info("Классификация (переуступка): %s", metrics["классификация_переуступки"])
    log.info("Прогноз 2026: %s", metrics["прогноз_2026"])

    # ---------- ШАГ 4. Отчёты ----------
    log.info("ШАГ 4. Генерация отчётов")
    reports.make_all_reports(data, metrics)

    # ---------- ШАГ 5. Сохранение в PostgreSQL ----------
    log.info("ШАГ 5. Сохранение результатов в PostgreSQL")
    try:
        save_results.save_all(clean, zhk, metrics)
        # пример чтения из БД сложным запросом (JOIN + агрегаты)
        top = load_database.run_sql(load_database.SQL_AVG_PRICE_BY_DISTRICT)
        log.info("Самые дорогие районы (из БД):\n%s", top.head(5).to_string(index=False))
    except Exception as e:
        log.error("Ошибка при работе с БД: %s", e)

    log.info("=== Пайплайн завершён за %.1f сек ===", time.time() - start)
    return metrics


if __name__ == "__main__":
    run_pipeline()

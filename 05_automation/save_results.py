# -*- coding: utf-8 -*-
"""
ШАГ 5. Сохранение результатов анализа в PostgreSQL.

Создаёт три таблицы:
    deals            - очищенные сделки ДДУ
    zhk              - справочник жилых комплексов
    analysis_results - метрики анализа

Таблицы пересоздаются при каждом запуске (свежие данные).

Запуск отдельно:  python 05_automation/save_results.py
"""

import json
import os
import sys

import pandas as pd
from psycopg2.extras import execute_values

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "01_loading"))

import config          # noqa: E402
from load_database import get_connection  # noqa: E402


def flatten_metrics(metrics):
    """Разворачивает вложенный словарь метрик в список пар (имя, значение)."""
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            for k, v in value.items():
                if isinstance(v, dict):
                    for k2, v2 in v.items():
                        rows.append((f"{key} / {k} / {k2}", str(v2)))
                else:
                    rows.append((f"{key} / {k}", str(v)))
        else:
            rows.append((key, str(value)))
    return rows


def save_deals(cur, clean):
    """Сохраняет очищенные сделки ДДУ."""
    cur.execute("DROP TABLE IF EXISTS deals")
    cur.execute("""
        CREATE TABLE deals (
            id_zhk        INTEGER,
            id_korpus     INTEGER,
            zhk_name      TEXT,
            district      TEXT,
            developer     TEXT,
            housing_class TEXT,
            deal_date     DATE,
            area          NUMERIC,
            rooms         NUMERIC,
            floor         NUMERIC,
            finishing     TEXT,
            mortgage      TEXT,
            assignment    TEXT,
            price         NUMERIC,
            price_sqm     NUMERIC,
            price_source  TEXT
        )
    """)

    df = clean.rename(columns={
        "ID ЖК": "id_zhk",
        "ID Корпуса": "id_korpus",
        "Название ЖК": "zhk_name",
        "Административный округ/ Городской округ/ Район": "district",
        "Девелопер": "developer",
        "Класс ЖК": "housing_class",
        "Дата договора": "deal_date",
        "Площадь": "area",
        "Комнат": "rooms",
        "Этаж": "floor",
        "Отделка (по корпусу)": "finishing",
        "Ипотека": "mortgage",
        "Переуступка": "assignment",
        "цена": "price",
        "цена_квм": "price_sqm",
        "источник_цены": "price_source",
    })
    df = df[["id_zhk", "id_korpus", "zhk_name", "district", "developer",
             "housing_class", "deal_date", "area", "rooms", "floor",
             "finishing", "mortgage", "assignment", "price", "price_sqm",
             "price_source"]]

    df["deal_date"] = pd.to_datetime(df["deal_date"]).dt.date
    # NaN заменяем на NULL, иначе PostgreSQL не примет
    df = df.astype(object).where(pd.notna(df), None)

    rows = list(df.itertuples(index=False, name=None))
    execute_values(cur, "INSERT INTO deals VALUES %s", rows, page_size=1000)
    print(f"Таблица deals: {len(rows)} строк")


def save_zhk(cur, zhk):
    """Сохраняет справочник жилых комплексов."""
    cur.execute("DROP TABLE IF EXISTS zhk")
    cur.execute("""
        CREATE TABLE zhk (
            id_zhk        INTEGER PRIMARY KEY,
            id_korpus     INTEGER,
            name          TEXT,
            build_stage   TEXT,
            vve_planned   TEXT,
            vve_actual    TEXT,
            mkad_km       NUMERIC,
            floors_max    NUMERIC,
            housing_class TEXT
        )
    """)

    df = zhk.rename(columns={
        "ID ЖК": "id_zhk",
        "ID Корпуса": "id_korpus",
        "Название ЖК": "name",
        "Стадия строительства": "build_stage",
        "Планируемый срок ВВЭ (ввод)": "vve_planned",
        "Дата фактического ВВЭ (ввод)": "vve_actual",
        "До МКАД, км": "mkad_km",
        "Этажность макс": "floors_max",
        "Класс": "housing_class",
    })
    df = df[["id_zhk", "id_korpus", "name", "build_stage", "vve_planned",
             "vve_actual", "mkad_km", "floors_max", "housing_class"]]
    df = df.dropna(subset=["id_zhk"]).drop_duplicates(subset=["id_zhk"])
    df = df.astype(object).where(pd.notna(df), None)

    rows = list(df.itertuples(index=False, name=None))
    execute_values(cur, "INSERT INTO zhk VALUES %s", rows, page_size=1000)
    print(f"Таблица zhk: {len(rows)} строк")


def save_metrics(cur, metrics):
    """Сохраняет метрики анализа."""
    cur.execute("DROP TABLE IF EXISTS analysis_results")
    cur.execute("""
        CREATE TABLE analysis_results (
            metric_name  TEXT,
            metric_value TEXT,
            created_at   TIMESTAMP DEFAULT now()
        )
    """)
    rows = flatten_metrics(metrics)
    execute_values(cur,
                   "INSERT INTO analysis_results (metric_name, metric_value) VALUES %s",
                   rows)
    print(f"Таблица analysis_results: {len(rows)} метрик")


def save_all(clean, zhk, metrics):
    """Сохраняет все результаты в базу данных (одна транзакция)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            save_deals(cur, clean)
            save_zhk(cur, zhk)
            save_metrics(cur, metrics)
        conn.commit()
        print("Результаты сохранены в PostgreSQL:", config.DB_NAME)
    finally:
        conn.close()


if __name__ == "__main__":
    clean = pd.read_csv(os.path.join(config.DATA_DIR, "clean_deals.csv"),
                        encoding="utf-8-sig", parse_dates=["Дата договора"])
    zhk = pd.read_csv(os.path.join(config.DATA_DIR, "zhk.csv"), encoding="utf-8-sig")
    with open(os.path.join(config.OUTPUT_DIR, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    save_all(clean, zhk, metrics)

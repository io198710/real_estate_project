# -*- coding: utf-8 -*-
"""
ШАГ 1. Загрузка данных из базы данных PostgreSQL через SQL-запросы.

Показаны сложные запросы: JOIN двух таблиц и агрегатные функции
(AVG, MIN, MAX, COUNT, GROUP BY, HAVING).

Таблицы создаёт скрипт 05_automation/save_results.py,
поэтому сначала нужно запустить полный пайплайн (python main.py).

Запуск отдельно:  python 01_loading/load_database.py
"""

import os
import sys

import pandas as pd
import psycopg2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_connection():
    """Создаёт подключение к PostgreSQL (настройки в config.py)."""
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


def run_sql(query):
    """Выполняет SQL-запрос и возвращает результат в виде DataFrame."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        return pd.DataFrame(rows, columns=columns)
    finally:
        conn.close()


# Запрос с JOIN и агрегатами:
# средняя цена квадратного метра по районам (сделки JOIN справочник ЖК)
SQL_AVG_PRICE_BY_DISTRICT = """
SELECT d.district              AS район,
       COUNT(*)                AS сделок,
       ROUND(AVG(d.price_sqm)) AS средняя_цена_квм,
       ROUND(MIN(d.price_sqm)) AS мин_цена_квм,
       ROUND(MAX(d.price_sqm)) AS макс_цена_квм
FROM deals d
JOIN zhk z ON d.id_zhk = z.id_zhk
GROUP BY d.district
HAVING COUNT(*) > 500
ORDER BY средняя_цена_квм DESC
LIMIT 15;
"""

# Запрос с агрегатами: средняя цена по классу жилья
SQL_AVG_PRICE_BY_CLASS = """
SELECT housing_class        AS класс,
       COUNT(*)             AS сделок,
       ROUND(AVG(price_sqm)) AS средняя_цена_квм
FROM deals
GROUP BY housing_class
ORDER BY средняя_цена_квм DESC;
"""

# Запрос с агрегатами и фильтром: откуда берётся цена сделок
SQL_PRICE_SOURCES = """
SELECT price_source   AS источник_цены,
       COUNT(*)       AS сделок,
       ROUND(AVG(price_sqm)) AS средняя_цена_квм
FROM deals
GROUP BY price_source
ORDER BY сделок DESC;
"""


if __name__ == "__main__":
    print("Средняя цена кв.м по районам (deals JOIN zhk):\n")
    print(run_sql(SQL_AVG_PRICE_BY_DISTRICT).to_string(index=False))
    print("\nСредняя цена кв.м по классам жилья:\n")
    print(run_sql(SQL_AVG_PRICE_BY_CLASS).to_string(index=False))
    print("\nИсточники цены сделок:\n")
    print(run_sql(SQL_PRICE_SOURCES).to_string(index=False))

# -*- coding: utf-8 -*-
"""
ШАГ 1. Загрузка данных из внешнего REST API.

Используем открытый API курсов валют (open.er-api.com, данные
обновляются ежедневно). Загружаем курс доллара к рублю - он влияет
на цены на недвижимость, поэтому его можно учитывать как фактор.

Запуск отдельно:  python 01_loading/load_api.py
"""

import os
import sys

import pandas as pd
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

API_URL = "https://open.er-api.com/v6/latest/USD"


def load_usd_rate():
    """
    Загружает текущий курс доллара к рублю.
    Возвращает DataFrame с датой и курсом.
    """
    response = requests.get(API_URL, timeout=15)
    response.raise_for_status()  # ошибка, если статус не 200
    data = response.json()

    return pd.DataFrame([{
        "date": pd.to_datetime(data["time_last_update_utc"]),
        "usd_rub": data["rates"]["RUB"],
        "eur_rub": data["rates"]["EUR"],
    }])


if __name__ == "__main__":
    try:
        df = load_usd_rate()
        df.to_csv(os.path.join(config.DATA_DIR, "usd_rate.csv"), index=False)
        print("Курс валют загружен из API:")
        print(df.to_string(index=False))
    except Exception as e:
        print("Не удалось загрузить данные из API:", e)

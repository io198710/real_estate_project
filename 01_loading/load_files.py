# -*- coding: utf-8 -*-
"""
ШАГ 1. Загрузка данных из файлов (CSV и Excel).

Основные данные проекта:
    - "Данные по лотам из выписок" - реальные сделки ДДУ (600 тыс. строк,
      история с 2014 года). Цена сделки: ДДУ -> прайс-лист -> статистика.
    - "Справочник по ЖК" - стадии строительства, сроки ВВЭ, МКАД, этажность
      (присоединяется к сделкам по ID Корпуса).

Функции:
    load_csv(path)       - загрузка CSV-файла
    load_excel(path, sheet) - загрузка листа Excel
    prepare_raw_data()   - загружает исходные данные проекта и сохраняет в CSV

Запуск отдельно:  python 01_loading/load_files.py
"""

import os
import sys

import pandas as pd

# добавляем корень проекта в путь, чтобы импортировать config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# колонки сделок, которые нужны проекту (из 40 в файле)
DEALS_COLS = [
    "ID ЖК", "ID Корпуса", "Регион", "Название ЖК", "Девелопер", "Класс ЖК",
    "Административный округ/ Городской округ/ Район",
    "Дата договора", "Регистрационный номер ДДУ",
    "Стоимость по ДДУ, руб.",
    "Стоимость объекта по прайс-листу на момент продажи, руб.",
    "Площадь", "Комнат", "Этаж", "Тип объекта", "Отделка (по корпусу)",
    "Ипотека", "Переуступка",
]

# колонки справочника ЖК для расчёта признаков
# (время до ВВЭ, стадия, МКАД, этажность, потолки, эскроу и т.д.)
ZHK_COLS = [
    "ID ЖК", "ID Корпуса", "Название ЖК", "Стадия строительства",
    "Планируемый срок ВВЭ (ввод)", "Дата фактического ВВЭ (ввод)",
    "До МКАД, км", "Этажность макс", "Этажность мин",
    "Потолки мин, м", "Потолки макс, м",
    "Кол-во жилых объектов, шт", "Кол-во квартир, шт",
    "Эскроу", "Банк по эскроу", "Тип каркаса дома", "Шоссе/направление",
    "Наличие отделки", "Вид отделки", "Cредняя площадь лота, м2.", "Класс",
]


def load_csv(path):
    """Загружает CSV-файл и возвращает DataFrame."""
    return pd.read_csv(path, encoding="utf-8-sig", low_memory=False)


def load_excel(path, sheet, usecols=None):
    """Загружает один лист Excel-файла. path - файл, sheet - название листа."""
    return pd.read_excel(path, sheet_name=sheet, usecols=usecols)


def prepare_raw_data(use_cache=False):
    """
    Загружает исходные данные из Excel-файла и сохраняет их в CSV.
    Дальнейшие шаги работают с CSV - это быстрее.

    Создаёт файлы:
        data/raw_deals.csv  - сделки ДДУ из выписок
        data/zhk.csv        - справочник по жилым комплексам

    use_cache=True - если CSV уже созданы, читаем их (не трогаем Excel).
    """
    if not os.path.exists(config.EXCEL_FILE):
        raise FileNotFoundError(
            "Не найден Excel-файл. Положите его в папку data/ (см. config.py)"
        )

    os.makedirs(config.DATA_DIR, exist_ok=True)
    deals_csv = os.path.join(config.DATA_DIR, "raw_deals.csv")
    zhk_csv = os.path.join(config.DATA_DIR, "zhk.csv")

    if use_cache and os.path.exists(deals_csv) and os.path.exists(zhk_csv):
        print("Читаю ранее сохранённые CSV...")
        return load_csv(deals_csv), load_csv(zhk_csv)

    print("Читаю Excel-файл, это занимает пару минут...")

    # 1. Сделки ДДУ из выписок - основные данные проекта.
    #    Это реальные продажи: в них есть цена ДДУ (если раскрыта в выписке)
    #    и цена по прайс-листу на момент продажи.
    deals = load_excel(config.EXCEL_FILE, config.SHEET_DEALS, usecols=DEALS_COLS)
    deals.to_csv(deals_csv, index=False, encoding="utf-8-sig")
    print("raw_deals.csv:", deals.shape)

    # 2. Справочник по ЖК: стадия строительства, сроки ВВЭ, МКАД, этажность.
    #    Присоединяется к сделкам по ID Корпуса на шаге анализа.
    zhk = load_excel(config.EXCEL_FILE, config.SHEET_ZHK, usecols=ZHK_COLS)
    zhk.to_csv(zhk_csv, index=False, encoding="utf-8-sig")
    print("zhk.csv:", zhk.shape)

    return deals, zhk


if __name__ == "__main__":
    deals, zhk = prepare_raw_data()
    print("\nИтог:")
    print("Сделки:", deals.shape)
    print("Справочник ЖК:", zhk.shape)

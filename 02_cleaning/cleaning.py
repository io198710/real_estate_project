# -*- coding: utf-8 -*-
"""
ШАГ 2. Очистка данных (сделки ДДУ из выписок).

Определение цены каждой сделки (порядок приоритета):
    1. Цена по ДДУ (самая точная, из выписки)
    2. Если ДДУ нет - цена по прайс-листу на момент продажи
    3. Если нет ни той, ни другой - заполняем МЕДИАНОЙ
       (задание: обработка пропущенных значений замена медианой)

Функции:
    fix_rooms          - приводит "Комнат" к числу (студия = 0)
    remove_duplicates  - удаляет полные дубликаты строк
    handle_missing     - обрабатывает пропуски (среднее / медиана / мода / удаление)
    remove_outliers    - удаляет выбросы методом IQR
    encode_one_hot     - One-Hot кодирование категорий
    encode_label       - Label Encoding категорий
    scale_numeric      - масштабирование чисел (minmax / standard)
    parse_dates        - приводит колонку к типу datetime
    fill_missing_price - цена: ДДУ -> прайс-лист -> медиана
    clean_deals_data   - полная очистка сделок

Запуск отдельно:  python 02_cleaning/cleaning.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def fix_rooms(df, column="Комнат"):
    """
    Приводит колонку комнатности к числовому виду.
    "студия" заменяется на 0, нечисловые значения становятся NaN.
    """
    df = df.copy()
    df[column] = df[column].replace("студия", 0)
    df[column] = pd.to_numeric(df[column], errors="coerce")
    return df


def remove_duplicates(df):
    """Удаляет полные дубликаты строк."""
    return df.drop_duplicates().reset_index(drop=True)


def handle_missing(df, column, method="median"):
    """
    Заполняет пропуски в одной колонке.
    method:
        "mean"   - средним значением (для чисел)
        "median" - медианой (для чисел)
        "mode"   - самым частым значением (для категорий)
        "drop"   - удалить строки с пропуском
    """
    df = df.copy()
    if method == "drop":
        return df.dropna(subset=[column]).reset_index(drop=True)
    if method == "mode" or not pd.api.types.is_numeric_dtype(df[column]):
        df[column] = df[column].fillna(df[column].mode()[0])
    elif method == "mean":
        df[column] = df[column].fillna(df[column].mean())
    else:
        df[column] = df[column].fillna(df[column].median())
    return df


def remove_outliers(df, column, k=1.5):
    """
    Удаляет выбросы по методу IQR из указанной колонки.
    k - коэффициент размаха (1.5 - стандартное значение).
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    mask = (df[column] >= q1 - k * iqr) & (df[column] <= q3 + k * iqr)
    return df[mask].reset_index(drop=True)


def encode_one_hot(df, columns):
    """One-Hot Encoding: каждая категория становится отдельной колонкой 0/1."""
    return pd.get_dummies(df, columns=columns, dtype=int)


def encode_label(df, column):
    """
    Label Encoding: каждой категории присваивается число.
    Добавляет колонку "<column>_код" и возвращает (df, словарь кодов).
    """
    df = df.copy()
    codes, uniques = pd.factorize(df[column])
    df[column + "_код"] = codes
    mapping = {category: i for i, category in enumerate(uniques)}
    return df, mapping


def scale_numeric(df, columns, method="minmax"):
    """
    Масштабирует числовые колонки.
    method:
        "minmax"   - приводит к диапазону [0, 1]
        "standard" - вычитает среднее и делит на стандартное отклонение
    """
    df = df.copy()
    for col in columns:
        if method == "standard":
            df[col] = (df[col] - df[col].mean()) / df[col].std()
        else:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    return df


def parse_dates(df, column):
    """Приводит колонку к типу datetime (даты-строки становятся датами)."""
    df = df.copy()
    df[column] = pd.to_datetime(df[column], errors="coerce")
    return df


def fill_missing_price(df):
    """
    Определяет цену каждой сделки (порядок приоритета):
      1. Цена по ДДУ (самая точная, из выписки)
      2. Цена по прайс-листу на момент продажи. Прайс-лист - это цена
         без скидки, а ДДУ - реальная цена сделки. По сделкам, где
         известны обе цены, медианное отношение ДДУ/прайс = 0.96
         (скидка ~4%, почти все сделки укладываются в ±20%).
         Поэтому прайс умножаем на медианный коэффициент скидки.
      3. Медиана: пропуски заполняем медианной ценой кв.м по рынку
         (задание: обработка пропущенных значений замена медианой)

    В колонке "источник_цены" видно, откуда взята цена каждой сделки.
    Возвращает (df, коэффициент_скидки).
    """
    df = df.copy()
    ddu = df["Стоимость по ДДУ, руб."]
    plist = df["Стоимость объекта по прайс-листу на момент продажи, руб."]

    # медианный коэффициент скидки ДДУ к прайсу (по сделкам с обеими ценами)
    both = df[ddu.notna() & plist.notna() & (plist > 0)]
    discount = float((both["Стоимость по ДДУ, руб."] /
                      both["Стоимость объекта по прайс-листу на момент продажи, руб."]).median())
    if not np.isfinite(discount):
        discount = 1.0

    # 1-2. известная цена: ДДУ в приоритете, иначе прайс-лист со скидкой
    df["источник_цены"] = "пропуск"
    df.loc[plist.notna(), "источник_цены"] = "прайс-лист"
    df.loc[ddu.notna(), "источник_цены"] = "ДДУ"
    df["цена"] = ddu.fillna(plist * discount)
    df["цена_квм"] = df["цена"] / df["Площадь"]

    # 3. оставшиеся пропуски заполняем медианой (как требует задание)
    med = df["цена_квм"].median()
    miss = df["цена"].isna()
    df.loc[miss, "цена_квм"] = med
    df.loc[miss, "цена"] = med * df.loc[miss, "Площадь"]
    df.loc[miss, "источник_цены"] = "медиана"
    return df, discount


def clean_deals_data(df):
    """
    Полная очистка сделок ДДУ:
      1. оставляем нужные колонки
      2. убираем дубликаты по регистрационному номеру ДДУ
      3. приводим даты и фильтруем период анализа (2014-2025)
      4. комнатность: студия = 0, пропуски - модой по ЖК
      5. цена: ДДУ -> прайс-лист -> медиана (см. fill_missing_price)
      6. убираем выбросы цены кв.м (30-2000 тыс. руб.)
    Возвращает (очищенный DataFrame, словарь-отчёт об очистке).
    """
    report = {}

    # 1. Оставляем только колонки, которые нужны дальше
    cols = ["ID ЖК", "ID Корпуса", "Название ЖК", "Девелопер", "Класс ЖК",
            "Административный округ/ Городской округ/ Район",
            "Дата договора", "Регистрационный номер ДДУ",
            "Стоимость по ДДУ, руб.",
            "Стоимость объекта по прайс-листу на момент продажи, руб.",
            "Площадь", "Комнат", "Этаж", "Отделка (по корпусу)",
            "Ипотека", "Переуступка"]
    df = df[cols].copy()

    # 2. Дубликаты по регистрационному номеру ДДУ
    # (сделки без номера не трогаем - это не дубликаты)
    reg = "Регистрационный номер ДДУ"
    dup_mask = df[reg].notna() & df[reg].duplicated(keep="first")
    before = len(df)
    df = df[~dup_mask].reset_index(drop=True)
    report["удалено_дубликатов_дду"] = before - len(df)

    # 3. Даты: приводим к datetime, берём период 2014-2025
    df = parse_dates(df, "Дата договора")
    df = df.dropna(subset=["Дата договора", "Площадь"])
    df = df[(df["Дата договора"] >= "2014-01-01") &
            (df["Дата договора"] <= "2025-12-31")]
    df = df[df["Площадь"] > 0]

    # 4. Комнатность: студия = 0, пропуски заполняем модой по ЖК,
    #    если в ЖК нет данных - модой по классу жилья
    df = fix_rooms(df)
    mode_jk = df.dropna(subset=["Комнат"]).groupby("Название ЖК")["Комнат"].agg(
        lambda s: s.mode().iloc[0] if len(s) else np.nan)
    miss_rooms = df["Комнат"].isna()
    df.loc[miss_rooms, "Комнат"] = df.loc[miss_rooms, "Название ЖК"].map(mode_jk)
    df = handle_missing(df, "Комнат", method="mode")
    report["заполнено_комнатности"] = int(miss_rooms.sum())

    # 5. Цена: ДДУ -> прайс-лист со скидкой -> медиана
    df, discount = fill_missing_price(df)
    report["коэффициент_скидки_дду"] = round(discount, 3)
    report["источники_цены"] = df["источник_цены"].value_counts().to_dict()

    # 6. Выбросы цены кв.м: реальный диапазон рынка 30-2000 тыс. руб.
    before = len(df)
    df = df[(df["цена_квм"] >= 30000) & (df["цена_квм"] <= 2000000)]
    report["удалено_выбросов_цены"] = before - len(df)

    # Категории: пусто = "нет"
    df["Этаж"] = pd.to_numeric(df["Этаж"], errors="coerce")
    df = handle_missing(df, "Этаж", method="median")
    df["Ипотека"] = df["Ипотека"].fillna("НЕТ")
    df["Переуступка"] = df["Переуступка"].fillna("Первый владелец")
    df["Отделка (по корпусу)"] = df["Отделка (по корпусу)"].fillna("нет")

    # Итоговые колонки
    df = df[["ID ЖК", "ID Корпуса", "Название ЖК", "Девелопер", "Класс ЖК",
             "Административный округ/ Городской округ/ Район",
             "Дата договора", "Площадь", "Комнат", "Этаж",
             "Отделка (по корпусу)", "Ипотека", "Переуступка",
             "цена", "цена_квм", "источник_цены"]]

    report["строк_после_очистки"] = len(df)
    return df, report


if __name__ == "__main__":
    # папка шага 1 нужна для импорта load_files
    sys.path.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "01_loading"))
    from load_files import load_csv

    raw = load_csv(os.path.join(config.DATA_DIR, "raw_deals.csv"))
    print("До очистки:", raw.shape)

    clean, report = clean_deals_data(raw)
    print("Отчёт об очистке:")
    for key, value in report.items():
        print(f"  {key}: {value}")
    print("После очистки:", clean.shape)

    # --- демонстрация кодирования и масштабирования ---
    encoded, class_mapping = encode_label(clean, "Класс ЖК")
    print("\nКоды классов жилья:", class_mapping)

    one_hot = encode_one_hot(clean[["Ипотека"]].head(), ["Ипотека"])
    print("\nOne-Hot по ипотеке (пример):")
    print(one_hot.to_string(index=False))

    scaled = scale_numeric(clean, ["Площадь", "Этаж"])
    print("\nПосле масштабирования площадь от",
          round(scaled["Площадь"].min(), 3), "до", round(scaled["Площадь"].max(), 3))

    # Сохраняем очищенные данные
    clean.to_csv(os.path.join(config.DATA_DIR, "clean_deals.csv"),
                 index=False, encoding="utf-8-sig")
    print("\nОчищенные данные сохранены: data/clean_deals.csv")

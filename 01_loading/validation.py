# -*- coding: utf-8 -*-
"""
ШАГ 1. Валидация данных при загрузке.

Проверяем:
    - дубликаты
    - пропуски
    - типы данных
    - выбросы (методы IQR и Z-score)

Все результаты проверки записываются в лог-файл logs/validation.log

Запуск отдельно:  python 01_loading/validation.py
"""

import logging
import os
import sys

import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def get_logger(log_file):
    """Создаёт логгер, который пишет в лог-файл."""
    os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
    logger = logging.getLogger("validation")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    return logger


def check_duplicates(df):
    """Возвращает количество полных дубликатов строк."""
    return int(df.duplicated().sum())


def check_missing(df):
    """Возвращает количество пропусков по колонкам (только где они есть)."""
    missing = df.isna().sum()
    return missing[missing > 0].sort_values(ascending=False)


def check_types(df):
    """Возвращает типы данных всех колонок."""
    return df.dtypes


def find_outliers_iqr(df, column):
    """
    Находит выбросы методом межквартильного размаха (IQR).
    Возвращает булеву маску: True - значение является выбросом.
    """
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    low = q1 - 1.5 * iqr
    high = q3 + 1.5 * iqr
    return (df[column] < low) | (df[column] > high)


def find_outliers_zscore(df, column, threshold=3.0):
    """
    Находит выбросы методом Z-score.
    Значение считается выбросом, если |z| > threshold (по умолчанию 3).
    """
    values = pd.to_numeric(df[column], errors="coerce")
    std = values.std()
    # если разброса нет - выбросов нет
    if pd.isna(std) or std == 0:
        return pd.Series(False, index=df.index)
    z = (values - values.mean()) / std
    return z.abs() > threshold


def validate(df, name="данные", log_file=None):
    """
    Полная проверка данных при загрузке.
    Пишет отчёт в лог-файл и возвращает словарь с результатами.
    """
    if log_file is None:
        log_file = os.path.join(config.LOGS_DIR, "validation.log")
    logger = get_logger(log_file)

    logger.info("=" * 60)
    logger.info("Валидация: %s | строк: %d | колонок: %d", name, df.shape[0], df.shape[1])

    result = {"rows": df.shape[0], "cols": df.shape[1]}

    # 1. Дубликаты
    result["duplicates"] = check_duplicates(df)
    logger.info("Дубликаты: %d", result["duplicates"])

    # 2. Пропуски
    missing = check_missing(df)
    result["missing"] = missing.to_dict()
    if len(missing) > 0:
        logger.info("Пропуски (колонок: %d, первые 10): %s", len(missing), missing.head(10).to_dict())
    else:
        logger.info("Пропуски: нет")

    # 3. Типы данных
    result["types"] = {col: str(dtype) for col, dtype in check_types(df).items()}
    logger.info("Типы данных определены для %d колонок", len(result["types"]))

    # 4. Выбросы в числовых колонках (проверяем первые 15)
    outliers = {}
    for col in list(df.select_dtypes(include="number").columns)[:15]:
        mask = find_outliers_iqr(df, col)
        found = int(mask.sum())
        if found > 0:
            outliers[col] = found
    result["outliers_iqr"] = outliers
    if outliers:
        logger.info("Выбросы (IQR): %s", outliers)
    else:
        logger.info("Выбросы (IQR): не найдены")

    return result


if __name__ == "__main__":
    from load_files import load_csv

    df = load_csv(os.path.join(config.DATA_DIR, "raw_deals.csv"))
    result = validate(df, "сделки ДДУ")
    print("\nИтог валидации:")
    print("Дубликатов рег.номера ДДУ:", result["duplicates"])
    print("Колонок с пропусками:", len(result["missing"]))
    print("Числовых колонок с выбросами:", len(result["outliers_iqr"]))

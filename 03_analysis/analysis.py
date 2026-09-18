# -*- coding: utf-8 -*-
"""
ШАГ 3. Анализ данных.

Главная задача - ПРЕДПОЛАГАЕМАЯ ЦЕНА ДДУ и ПРОГНОЗ ЦЕН НА 2026 ГОД.

Функции:
    basic_stats(df, column)       - среднее, медиана, мода, станд. отклонение
    parse_vve(val)                - срок ВВЭ ('26/IV' или дата) -> datetime
    prepare_ml_data(clean, zhk)   - JOIN сделок со справочником, расчёт признаков
    train_regression(data)        - XGBoost: прогноз цены кв.м (предполагаемая
                                    цена ДДУ) + сравнение с простой медианой
    train_classification(data)    - XGBoost: класс жилья по характеристикам
    time_series_analysis(df)      - ряд цен по месяцам: тренд, сезонность
    forecast_2026(df)             - прогноз средней цены кв.м на 2026 год
    detect_anomalies(df, column)  - аномалии методом Z-score
    run_analysis()               - весь анализ, метрики -> output/metrics.json

Запуск отдельно:  python 03_analysis/analysis.py
"""

import json
import os
import sys

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# числовые и категориальные признаки для моделей (как в большом BI-проекте)
NUM_FEATURES = [
    "Площадь", "months_to_vve", "deal_year", "deal_month",
    "До МКАД, км", "Этажность макс", "Этажность мин",
    "Потолки мин, м", "Потолки макс, м",
    "Кол-во жилых объектов, шт", "Кол-во квартир, шт",
    "Cредняя площадь лота, м2.",
]
CAT_FEATURES = [
    "Регион", "Район", "Название ЖК", "Девелопер", "Класс ЖК", "Комнат",
    "Этаж", "Переуступка", "Стадия строительства", "Ипотека", "Отделка",
    "Тип объекта", "Банк по эскроу", "Тип каркаса дома", "Шоссе/направление",
    "Наличие отделки", "Вид отделки", "Эскроу",
]


# ---------- 1. Базовые статистики ----------

def basic_stats(df, column):
    """Считает базовые статистики по одной числовой колонке."""
    s = df[column]
    return {
        "колонка": column,
        "среднее": round(float(s.mean()), 1),
        "медиана": round(float(s.median()), 1),
        "мода": round(float(s.mode()[0]), 1),
        "станд_отклонение": round(float(s.std()), 1),
        "минимум": round(float(s.min()), 1),
        "максимум": round(float(s.max()), 1),
    }


# ---------- 2. Подготовка данных для ML ----------

def parse_vve(val):
    """
    Срок ВВЭ из справочника -> datetime.
    Формат '26/IV' означает 4 квартал 2026 года -> 2026-04-28.
    Обычные даты ('2023-01-12') парсятся как есть.
    """
    roman = {"I": 1, "II": 2, "III": 3, "IV": 4}
    s = str(val).strip() if pd.notna(val) else ""
    if not s or s.lower() == "nan":
        return pd.NaT
    if "/" in s:
        try:
            parts = s.split("/")
            yy = int(parts[0])
            year = 2000 + yy if yy < 100 else yy
            q = roman.get(parts[1].strip().upper())
            if q:
                return pd.Timestamp(year=year, month=q * 3, day=28)
        except (ValueError, IndexError, KeyError):
            pass
    return pd.to_datetime(s, errors="coerce")


def prepare_ml_data(clean, zhk):
    """
    Присоединяет к сделкам справочник ЖК (по ID Корпуса) и считает признаки:

      months_to_vve - сколько месяцев оставалось до ВВЭ на момент сделки.
        Это главный фактор стадии строительства: на этапе котлована
        (до ВВЭ ещё 2-3 года) метр заметно дешевле, чем после ВВЭ.
      стадия_сдачи - то же самое бакетами: после ВВЭ, <1 года, 1-2 года...
      deal_year / deal_month - когда прошла сделка
      + факторы справочника: МКАД, этажность, потолки, эскроу,
        тип каркаса, шоссе и т.д. (как в большом BI-проекте)

    Этаж и Комнат - КАТЕГОРИИ (Label Encoding): первый и последний этаж
    дешевле средних, а 4-комнатная не в 4 раза дороже 1-комнатной.
    """
    # признаки из справочника ЖК
    zhk_features = ["ID Корпуса", "Стадия строительства",
                    "Планируемый срок ВВЭ (ввод)", "Дата фактического ВВЭ (ввод)",
                    "До МКАД, км", "Этажность макс", "Этажность мин",
                    "Потолки мин, м", "Потолки макс, м",
                    "Кол-во жилых объектов, шт", "Кол-во квартир, шт",
                    "Эскроу", "Банк по эскроу", "Тип каркаса дома",
                    "Шоссе/направление", "Наличие отделки", "Вид отделки",
                    "Cредняя площадь лота, м2."]
    # берём только те колонки справочника, которые есть в данных
    zhk_cols = [c for c in zhk_features if c in zhk.columns]
    data = clean.merge(zhk[zhk_cols], on="ID Корпуса", how="left")

    # короткие имена для признаков
    data = data.rename(columns={
        "Административный округ/ Городской округ/ Район": "Район",
        "Отделка (по корпусу)": "Отделка"})

    # ВВЭ: фактическая дата надёжнее, иначе планируемый срок
    data["ввэ_факт"] = data["Дата фактического ВВЭ (ввод)"].apply(parse_vve)
    data["ввэ_план"] = data["Планируемый срок ВВЭ (ввод)"].apply(parse_vve)
    data["ввэ"] = data["ввэ_факт"].fillna(data["ввэ_план"])

    # месяцы до ВВЭ на момент сделки (999 = срок неизвестен)
    delta = (data["ввэ"] - data["Дата договора"]).dt.days
    data["months_to_vve"] = (delta / 30.44).round()
    data["months_to_vve"] = data["months_to_vve"].fillna(999)
    data.loc[data["months_to_vve"] < -120, "months_to_vve"] = 999

    # стадия сдачи бакетами (для анализа "котлован vs ВВЭ")
    bins = [-np.inf, 0, 12, 24, 36, 998.9, np.inf]
    labels = ["после ВВЭ", "до ВВЭ <1 года", "1-2 года до ВВЭ",
              "2-3 года до ВВЭ", ">3 лет до ВВЭ (котлован)", "нет данных"]
    data["стадия_сдачи"] = pd.cut(data["months_to_vve"], bins=bins, labels=labels)

    # дата сделки -> год и месяц
    data["deal_year"] = data["Дата договора"].dt.year
    data["deal_month"] = data["Дата договора"].dt.month

    # пропуски справочника заполняем типовыми значениями
    # (у части сделок корпус не найден в справочнике)
    num_fills = {"До МКАД, км": 30, "Этажность макс": 25, "Этажность мин": 25,
                 "Потолки мин, м": 2.7, "Потолки макс, м": 3.0,
                 "Кол-во жилых объектов, шт": 500, "Кол-во квартир, шт": 500,
                 "Cредняя площадь лота, м2.": 50}
    for col, value in num_fills.items():
        if col in data.columns:
            data[col] = data[col].fillna(value)

    cat_fills = ["Стадия строительства", "Банк по эскроу", "Тип каркаса дома",
                 "Шоссе/направление", "Наличие отделки", "Вид отделки",
                 "Регион", "Тип объекта", "Эскроу"]
    for col in cat_fills:
        if col in data.columns:
            data[col] = data[col].fillna("неизвестно")

    return data


def encode_features(data, cat_features):
    """Кодирует категориальные признаки Label Encoding'ом (номер категории)."""
    data = data.copy()
    for col in cat_features:
        # Этаж и Комнат - числа, но это категории: переводим в строки
        data[col] = data[col].astype(str).str.strip()
        data[col + "_код"] = LabelEncoder().fit_transform(data[col])
    return data


# ---------- 3. ML: регрессия (предполагаемая цена ДДУ) ----------

def train_regression(data):
    """
    Обучает XGBoost: прогноз цены кв.м по характеристикам лота.
    Это и есть модель предполагаемой цены ДДУ: для лота, у которого
    цена неизвестна, модель даёт оценку по похожим лотам.

    Обучаемся только на сделках с реальной ценой (ДДУ или прайс-лист).
    Для сравнения считается базовая линия - простая медиана цены.

    Возвращает (модель, метрики).
    """
    # только реальные цены (ДДУ и прайс-лист, не заполненные медианой)
    real = data[data["источник_цены"] != "медиана"].copy()

    # берём только те признаки, которые есть в данных
    num_features = [c for c in NUM_FEATURES if c in real.columns]
    cat_features = [c for c in CAT_FEATURES if c in real.columns]
    real = encode_features(real, cat_features)
    feature_cols = num_features + [c + "_код" for c in cat_features]

    X = real[feature_cols]
    y = real["цена_квм"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # --- модель: XGBoost (градиентный бустинг) ---
    model = xgb.XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.1,
        min_child_weight=10, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    # --- базовая линия: простая медиана цены (как если бы заполняли
    #     все пропуски одной медианой без учёта характеристик) ---
    med = y_train.median()
    base_pred = np.full(len(y_test), med)

    metrics = {
        "модель": "XGBoost (градиентный бустинг)",
        "MAE_руб": round(float(mean_absolute_error(y_test, pred)), 0),
        "RMSE_руб": round(float(np.sqrt(mean_squared_error(y_test, pred))), 0),
        "R2": round(float(r2_score(y_test, pred)), 3),
        "baseline_медиана_MAE_руб": round(float(mean_absolute_error(y_test, base_pred)), 0),
        "baseline_медиана_R2": round(float(r2_score(y_test, base_pred)), 3),
    }

    # важность признаков
    importance = sorted(zip(feature_cols, model.feature_importances_),
                        key=lambda x: -x[1])
    metrics["важность_признаков"] = {f: round(float(i), 3) for f, i in importance[:10]}

    return model, metrics


# ---------- 4. ML: классификация (класс жилья) ----------

def train_classification(data):
    """
    Обучает бинарную классификацию: предсказываем переуступку
    (владелец "Второй"/"Последующий" = перепродажа ДДУ до регистрации)
    по параметрам лота и сделки. Бизнес-смысл: переуступки - индикатор
    спекулятивного спроса, застройщику полезно знать, какие лоты
    перепродают. Возвращает (модель, словарь метрик).
    """
    df = data.copy()
    # целевой признак: первый владелец -> 0, второй/последующий -> 1
    y = (df["Переуступка"] != "Первый владелец").astype(int)

    num_features = [c for c in ["Площадь", "months_to_vve", "deal_year",
                                "deal_month", "цена_квм"] if c in df.columns]
    cat_features = [c for c in ["Район", "Девелопер", "Класс ЖК", "Комнат",
                                "Этаж", "Ипотека", "Отделка",
                                "Стадия строительства"] if c in df.columns]
    df = encode_features(df, cat_features)
    feature_cols = num_features + [c + "_код" for c in cat_features]

    X = df[feature_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=8, learning_rate=0.1,
        tree_method="hist", n_jobs=-1, random_state=42)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)

    # ROC-AUC: для двух классов берем вероятность положительного класса,
    # для многих классов считаем по схеме "один против всех" (ovr)
    if len(model.classes_) == 2:
        roc_auc = roc_auc_score(y_test, proba[:, 1])
    else:
        roc_auc = roc_auc_score(y_test, proba, multi_class="ovr", average="weighted")

    metrics = {
        "модель": "XGBoost (градиентный бустинг)",
        "accuracy": round(float(accuracy_score(y_test, pred)), 3),
        "precision": round(float(precision_score(y_test, pred, average="weighted")), 3),
        "recall": round(float(recall_score(y_test, pred, average="weighted")), 3),
        "F1": round(float(f1_score(y_test, pred, average="weighted")), 3),
        "ROC_AUC": round(float(roc_auc), 3),
    }
    return model, metrics


# ---------- 5. Временной ряд и прогноз 2026 ----------

def get_monthly_prices(clean):
    """Помесячная средняя цена кв.м по реальным сделкам (ДДУ + прайс-лист)."""
    real = clean[clean["источник_цены"] != "медиана"]
    ts = (real.groupby(pd.Grouper(key="Дата договора", freq="MS"))["цена_квм"]
          .mean())
    # месяцы без сделок заполняем интерполяцией, задаём частоту ряда
    ts = ts.asfreq("MS").interpolate()
    return ts


def time_series_analysis(ts, window=12):
    """
    Анализ временного ряда цен:
      - тренд через скользящее среднее
      - декомпозиция на тренд / сезонность / остатки
    """
    trend = ts.rolling(window=window, min_periods=1).mean()
    decomposition = seasonal_decompose(ts, model="additive", period=12)

    return {
        "ряд": ts,
        "тренд": trend,
        "декомпозиция": decomposition,
        "итог": {
            "начало_ряда": str(ts.index[0].date()),
            "конец_ряда": str(ts.index[-1].date()),
            "первое_значение": round(float(ts.iloc[0]), 0),
            "последнее_значение": round(float(ts.iloc[-1]), 0),
            "изменение_%": round(float((ts.iloc[-1] / ts.iloc[0] - 1) * 100), 1),
        },
    }


def forecast_2026(ts):
    """
    Прогноз средней цены кв.м на 2026 год.
    Метод: сглаживание Хольта-Уинтерса (тренд + сезонность,
    затухающий тренд - прогноз не улетает в бесконечность).
    """
    model = ExponentialSmoothing(
        ts, trend="add", damped_trend=True, seasonal="add",
        seasonal_periods=12, initialization_method="estimated").fit()

    # сколько месяцев вперёд прогнозируем (до конца 2026 года)
    last = ts.index[-1]
    steps = (2026 - last.year) * 12 + (12 - last.month)
    forecast = model.forecast(steps)

    # оставляем только 2026 год
    forecast_2026 = forecast[forecast.index.year == 2026]

    return {
        "прогноз": forecast,
        "итог": {
            "средняя_цена_квм_2025": round(float(ts[ts.index.year == 2025].mean()), 0),
            "прогноз_средней_цены_2026": round(float(forecast_2026.mean()), 0),
            "ожидаемый_рост_%": round(float(
                (forecast_2026.mean() / ts[ts.index.year == 2025].mean() - 1) * 100), 1),
            "прогноз_по_месяцам": {str(d.date()): round(float(v), 0)
                                   for d, v in forecast_2026.items()},
        },
    }


# ---------- 6. Аномалии ----------

def detect_anomalies(df, column, threshold=3.0):
    """
    Находит аномальные значения методом Z-score.
    Возвращает строки, где |z| > threshold.
    """
    z = (df[column] - df[column].mean()) / df[column].std()
    return df[z.abs() > threshold]


# ---------- Запуск всего анализа ----------

def run_analysis(clean=None, zhk=None):
    """
    Полный анализ: статистики, цена по стадиям, аномалии, ML-модели,
    временной ряд и прогноз на 2026 год.
    Метрики сохраняются в output/metrics.json
    Возвращает (метрики, данные с признаками для отчётов).
    """
    if clean is None:
        clean = pd.read_csv(os.path.join(config.DATA_DIR, "clean_deals.csv"),
                            encoding="utf-8-sig", parse_dates=["Дата договора"])
    if zhk is None:
        zhk = pd.read_csv(os.path.join(config.DATA_DIR, "zhk.csv"),
                          encoding="utf-8-sig")

    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    metrics = {}

    # 0. Данные с признаками (JOIN со справочником)
    data = prepare_ml_data(clean, zhk)

    # 1. Базовые статистики
    metrics["статистики_цены"] = basic_stats(data, "цена_квм")
    metrics["статистики_площади"] = basic_stats(data, "Площадь")

    # 2. Цена по стадии сдачи (главный фактор: котлован дешевле ВВЭ)
    by_stage = (data.groupby("стадия_сдачи", observed=True)["цена_квм"]
                .agg(["count", "mean"]).round(0))
    metrics["цена_по_стадии_сдачи"] = {
        str(stage): {"сделок": int(row["count"]),
                     "средняя_цена_квм": float(row["mean"])}
        for stage, row in by_stage.iterrows()
    }

    # 3. Аномалии по цене
    anomalies = detect_anomalies(data[data["источник_цены"] != "медиана"],
                                  "цена_квм")
    metrics["аномалии_цены"] = {
        "количество": int(len(anomalies)),
        "доля_%": round(len(anomalies) / len(data) * 100, 2),
    }

    # 4. ML-модели (обучение на реальных ценах)
    _, reg_metrics = train_regression(data)
    metrics["регрессия_предполагаемая_цена"] = reg_metrics

    _, clf_metrics = train_classification(data)
    metrics["классификация_переуступки"] = clf_metrics

    # 5. Временной ряд и прогноз 2026
    ts = get_monthly_prices(clean)
    ts_result = time_series_analysis(ts)
    metrics["временной_ряд"] = ts_result["итог"]

    fc = forecast_2026(ts)
    metrics["прогноз_2026"] = fc["итог"]

    # сохраняем ряд и прогноз для отчётов
    monthly = pd.DataFrame({
        "Месяц": ts.index,
        "цена_квм": ts.values,
        "тренд": ts_result["тренд"].values,
    })
    fc_series = fc["прогноз"]
    monthly = pd.concat([monthly, pd.DataFrame({
        "Месяц": fc_series.index,
        "прогноз_2026": fc_series.values,
    })], ignore_index=True)
    monthly.to_csv(os.path.join(config.OUTPUT_DIR, "monthly_prices.csv"),
                   index=False, encoding="utf-8-sig")

    # таблица "цена по стадии и году" для Excel-отчёта
    stage_year = (data.groupby(["deal_year", "стадия_сдачи"], observed=True)
                  ["цена_квм"].agg(["count", "mean"]).round(0).reset_index())
    stage_year.to_csv(os.path.join(config.OUTPUT_DIR, "stage_prices.csv"),
                      index=False, encoding="utf-8-sig")

    # сохраняем все метрики в JSON
    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print("Метрики сохранены:", metrics_path)

    return metrics, data


if __name__ == "__main__":
    result, _ = run_analysis()
    print(json.dumps(result, ensure_ascii=False, indent=2))

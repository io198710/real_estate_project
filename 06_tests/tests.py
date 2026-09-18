# -*- coding: utf-8 -*-
"""
ШАГ 6. Юнит-тесты основных функций проекта.

Проверяются функции загрузки, валидации, очистки и анализа.

Запуск:  python 06_tests/tests.py
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "01_loading"))
sys.path.insert(0, os.path.join(BASE_DIR, "02_cleaning"))
sys.path.insert(0, os.path.join(BASE_DIR, "03_analysis"))

import analysis    # noqa: E402
import cleaning    # noqa: E402
import load_files  # noqa: E402
import validation  # noqa: E402


class TestLoading(unittest.TestCase):
    """Тесты загрузки и валидации данных."""

    def test_load_csv(self):
        # создаём временный CSV и загружаем его
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        path = os.path.join(os.path.dirname(__file__), "tmp_test.csv")
        df.to_csv(path, index=False, encoding="utf-8-sig")
        loaded = load_files.load_csv(path)
        self.assertEqual(len(loaded), 2)
        self.assertEqual(list(loaded.columns), ["a", "b"])
        os.remove(path)

    def test_check_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
        self.assertEqual(validation.check_duplicates(df), 1)

    def test_check_missing(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": [1, 2, 3]})
        missing = validation.check_missing(df)
        self.assertEqual(missing["a"], 1)   # один пропуск в колонке a
        self.assertNotIn("b", missing)      # в колонке b пропусков нет

    def test_outliers_iqr(self):
        df = pd.DataFrame({"price": [10, 11, 12, 11, 10, 1000]})
        mask = validation.find_outliers_iqr(df, "price")
        self.assertEqual(mask.sum(), 1)     # одно значение - выброс
        self.assertTrue(mask.iloc[-1])     # и это 1000

    def test_outliers_zscore(self):
        # 20 обычных значений и одно сильное отклонение
        df = pd.DataFrame({"price": [10] * 20 + [1000]})
        mask = validation.find_outliers_zscore(df, "price")
        self.assertEqual(mask.sum(), 1)

    def test_zscore_without_spread(self):
        # если все значения одинаковые - выбросов быть не должно
        df = pd.DataFrame({"price": [5, 5, 5]})
        mask = validation.find_outliers_zscore(df, "price")
        self.assertEqual(mask.sum(), 0)


def make_deals_fixture():
    """Мини-набор сделок для тестов очистки."""
    return pd.DataFrame({
        "ID ЖК": [1, 1, 2, 2],
        "ID Корпуса": [10, 10, 20, 20],
        "Название ЖК": ["А", "А", "Б", "Б"],
        "Девелопер": ["Д1", "Д1", "Д2", "Д2"],
        "Класс ЖК": ["Комфорт", "Комфорт", "Бизнес", "Бизнес"],
        "Административный округ/ Городской округ/ Район": ["С", "С", "Ю", "Ю"],
        "Дата договора": ["2024-01-15", "2024-02-20", "2024-03-10", "2024-04-05"],
        "Регистрационный номер ДДУ": ["R1", "R1", None, None],
        "Стоимость по ДДУ, руб.": [10_000_000, None, None, None],
        "Стоимость объекта по прайс-листу на момент продажи, руб.": [None, 8_000_000, None, 12_000_000],
        "Площадь": [50, 40, 45, 60],
        "Комнат": [1, 1, None, 2],
        "Этаж": [3, 5, None, 7],
        "Отделка (по корпусу)": [None, None, None, None],
        "Ипотека": [None, "ДА", None, None],
        "Переуступка": [None, None, None, None],
    })


class TestCleaning(unittest.TestCase):
    """Тесты очистки данных."""

    def test_remove_duplicates(self):
        df = pd.DataFrame({"a": [1, 1, 2]})
        out = cleaning.remove_duplicates(df)
        self.assertEqual(len(out), 2)

    def test_handle_missing_median(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        out = cleaning.handle_missing(df, "a", method="median")
        self.assertEqual(out["a"].iloc[1], 2.0)   # медиана [1, 3] = 2

    def test_handle_missing_mean(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        out = cleaning.handle_missing(df, "a", method="mean")
        self.assertEqual(out["a"].iloc[1], 2.0)   # среднее [1, 3] = 2

    def test_handle_missing_mode(self):
        df = pd.DataFrame({"a": ["x", None, "x", "y"]})
        out = cleaning.handle_missing(df, "a", method="mode")
        self.assertEqual(out["a"].iloc[1], "x")   # самое частое значение

    def test_handle_missing_drop(self):
        df = pd.DataFrame({"a": [1.0, None, 3.0]})
        out = cleaning.handle_missing(df, "a", method="drop")
        self.assertEqual(len(out), 2)

    def test_fix_rooms(self):
        # студия считается как 0 комнат
        df = pd.DataFrame({"Комнат": ["студия", "2", "1"]})
        out = cleaning.fix_rooms(df)
        self.assertEqual(list(out["Комнат"]), [0, 2, 1])

    def test_remove_outliers(self):
        df = pd.DataFrame({"цена": [10, 11, 12, 13, 1000]})
        out = cleaning.remove_outliers(df, "цена")
        self.assertNotIn(1000, list(out["цена"]))
        self.assertEqual(len(out), 4)

    def test_encode_one_hot(self):
        df = pd.DataFrame({"цвет": ["красный", "синий", "красный"]})
        out = cleaning.encode_one_hot(df, ["цвет"])
        self.assertIn("цвет_красный", out.columns)
        self.assertNotIn("цвет", out.columns)     # исходная колонка заменена
        self.assertEqual(out["цвет_красный"].sum(), 2)

    def test_encode_label(self):
        df = pd.DataFrame({"город": ["Москва", "Казань", "Москва"]})
        out, mapping = cleaning.encode_label(df, "город")
        self.assertEqual(mapping, {"Москва": 0, "Казань": 1})
        self.assertEqual(list(out["город_код"]), [0, 1, 0])

    def test_scale_numeric_minmax(self):
        df = pd.DataFrame({"a": [0, 5, 10]})
        out = cleaning.scale_numeric(df, ["a"])
        self.assertAlmostEqual(out["a"].min(), 0.0)
        self.assertAlmostEqual(out["a"].max(), 1.0)

    def test_parse_dates(self):
        df = pd.DataFrame({"d": ["2025-11-01", "2025-12-05"]})
        out = cleaning.parse_dates(df, "d")
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(out["d"]))

    def test_price_ddu_priority(self):
        # цена ДДУ важнее цены по прайс-листу
        df = pd.DataFrame({
            "Стоимость по ДДУ, руб.": [100.0],
            "Стоимость объекта по прайс-листу на момент продажи, руб.": [999.0],
            "Площадь": [10.0],
            "Название ЖК": ["А"],
            "Класс ЖК": ["Комфорт"],
            "Комнат": [1],
        })
        out, discount = cleaning.fill_missing_price(df)
        self.assertEqual(out["цена"].iloc[0], 100.0)
        self.assertEqual(out["источник_цены"].iloc[0], "ДДУ")
        # по паре с обеими ценами считается коэффициент скидки
        self.assertAlmostEqual(discount, 100.0 / 999.0, places=3)

    def test_fill_price_by_median(self):
        # пропущенная цена заполняется медианной ценой кв.м по рынку
        df = pd.DataFrame({
            "Стоимость по ДДУ, руб.": [10_000_000, None, None],
            "Стоимость объекта по прайс-листу на момент продажи, руб.": [None, 8_000_000, None],
            "Площадь": [50, 40, 45],
        })
        out, discount = cleaning.fill_missing_price(df)
        # нет пар с обеими ценами -> скидка 1.0, медиана кв.м = 200000
        self.assertEqual(discount, 1.0)
        self.assertEqual(out["цена"].iloc[2], 200000 * 45)
        self.assertEqual(out["источник_цены"].iloc[2], "медиана")

    def test_clean_deals_data(self):
        df = make_deals_fixture()
        clean, report = cleaning.clean_deals_data(df)
        # дубликат по рег.номеру R1 удалён: 4 -> 3
        self.assertEqual(len(clean), 3)
        # источники цены: ДДУ, прайс-лист и медиана
        self.assertEqual(set(clean["источник_цены"]),
                         {"ДДУ", "прайс-лист", "медиана"})
        # комнатность и этаж заполнены везде
        self.assertFalse(clean["Комнат"].isna().any())
        self.assertFalse(clean["Этаж"].isna().any())
        # цена кв.м в разумных границах
        self.assertTrue((clean["цена_квм"] > 0).all())


class TestAnalysis(unittest.TestCase):
    """Тесты анализа данных."""

    def _ml_fixture(self):
        """Мини-набор сделок + справочник для тестов ML."""
        n = 300
        rng = np.random.RandomState(42)
        clean = pd.DataFrame({
            "ID ЖК": 1,
            "ID Корпуса": 10,
            "Название ЖК": "Тест",
            "Девелопер": "Дев",
            "Класс ЖК": "Комфорт",
            "Административный округ/ Городской округ/ Район": "Северный",
            "Дата договора": pd.date_range("2024-01-01", periods=n, freq="D"),
            "Площадь": rng.uniform(30, 100, n).round(1),
            "Комнат": rng.randint(1, 4, n),
            "Этаж": rng.randint(1, 20, n),
            "Отделка (по корпусу)": "нет",
            "Ипотека": "НЕТ",
            "Переуступка": "Первый владелец",
            "источник_цены": "ДДУ",
        })
        zhk = pd.DataFrame({
            "ID Корпуса": [10],
            "Стадия строительства": ["котлован"],
            "Планируемый срок ВВЭ (ввод)": ["27/IV"],
            "Дата фактического ВВЭ (ввод)": [None],
            "До МКАД, км": [5.0],
            "Этажность макс": [25],
        })
        return clean, zhk

    def test_basic_stats(self):
        df = pd.DataFrame({"цена": [10, 20, 30, 40]})
        stats = analysis.basic_stats(df, "цена")
        self.assertEqual(stats["среднее"], 25.0)
        self.assertEqual(stats["медиана"], 25.0)
        self.assertEqual(stats["мода"], 10.0)
        self.assertEqual(stats["минимум"], 10.0)
        self.assertEqual(stats["максимум"], 40.0)

    def test_detect_anomalies(self):
        # много обычных цен и одна аномальная
        df = pd.DataFrame({"цена": [100, 102, 101, 99, 100] * 4 + [5000]})
        anomalies = analysis.detect_anomalies(df, "цена")
        self.assertEqual(len(anomalies), 1)

    def test_parse_vve(self):
        # формат '26/IV' = 4 квартал 2026 года -> конец квартала (декабрь)
        self.assertEqual(analysis.parse_vve("26/IV"), pd.Timestamp("2026-12-28"))
        # обычная дата
        self.assertEqual(analysis.parse_vve("2023-01-12"), pd.Timestamp("2023-01-12"))
        # пустые значения
        self.assertTrue(pd.isna(analysis.parse_vve(None)))
        self.assertTrue(pd.isna(analysis.parse_vve("")))

    def test_prepare_ml_data(self):
        clean, zhk = self._ml_fixture()
        clean = clean.copy()
        clean["цена_квм"] = 200000.0
        data = analysis.prepare_ml_data(clean, zhk)
        # месяцы до ВВЭ посчитаны (сделки 2024, ВВЭ 2027)
        self.assertTrue((data["months_to_vve"] > 0).all())
        self.assertTrue((data["months_to_vve"] < 999).all())
        # стадия сдачи определена бакетом
        self.assertTrue(data["стадия_сдачи"].notna().all())

    def test_train_regression(self):
        # синтетические данные: цена = 2 * площадь (тыс. руб.)
        clean, zhk = self._ml_fixture()
        clean["цена_квм"] = 2 * clean["Площадь"] * 1000
        data = analysis.prepare_ml_data(clean, zhk)
        model, metrics = analysis.train_regression(data)
        self.assertGreater(metrics["R2"], 0.8)
        # модель должна быть заметно точнее базовой медианы
        self.assertLess(metrics["MAE_руб"], metrics["baseline_медиана_MAE_руб"])

    def test_train_classification(self):
        # переуступка зависит от цены - модель должна это уловить
        clean, zhk = self._ml_fixture()
        half = len(clean) // 2
        clean["Переуступка"] = ["Первый владелец"] * half + \
            ["Второй владелец"] * (len(clean) - half)
        clean["цена_квм"] = [200000.0] * half + [400000.0] * (len(clean) - half)
        data = analysis.prepare_ml_data(clean, zhk)
        model, metrics = analysis.train_classification(data)
        self.assertGreaterEqual(metrics["accuracy"], 0.9)
        self.assertGreater(metrics["F1"], 0.9)

    def test_forecast_2026(self):
        # синтетический ряд на 3 года, заканчивается в декабре 2025
        idx = pd.date_range("2023-01-01", periods=36, freq="MS")
        ts = pd.Series(200000 + np.arange(36) * 1000.0, index=idx)
        fc = analysis.forecast_2026(ts)
        self.assertIn("прогноз_средней_цены_2026", fc["итог"])
        self.assertEqual(len(fc["итог"]["прогноз_по_месяцам"]), 12)
        # прогноз продолжается вверх (в ряду ровный рост)
        self.assertGreater(fc["итог"]["прогноз_средней_цены_2026"],
                           fc["итог"]["средняя_цена_квм_2025"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

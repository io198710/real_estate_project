# -*- coding: utf-8 -*-
"""
Итоговый проект: автоматизация обработки данных.
Рынок новостроек Москвы.

Точка входа - запускает полный пайплайн:
    загрузка -> валидация -> очистка -> анализ -> отчёты -> сохранение в БД

Запуск:  python main.py
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "05_automation"))

from pipeline import run_pipeline

if __name__ == "__main__":
    run_pipeline()

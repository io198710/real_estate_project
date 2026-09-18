# -*- coding: utf-8 -*-
"""
Общие настройки проекта: пути к файлам, база данных, почта.
Все скрипты проекта импортируют настройки отсюда.
"""

import os

# Корень проекта
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Папки проекта
DATA_DIR = os.path.join(BASE_DIR, "data")      # исходные и промежуточные данные
OUTPUT_DIR = os.path.join(BASE_DIR, "output")  # отчёты, графики, метрики
LOGS_DIR = os.path.join(BASE_DIR, "logs")      # лог-файлы

# Исходный Excel-файл (нужно положить в папку data)
EXCEL_FILE = os.path.join(DATA_DIR, "Отчёт_Пульс_Продаж_Новостроек_Москва_2025.11.xlsx")

# Листы Excel, которые используются в проекте
SHEET_DEALS = "Данные по лотам из выписок"  # реальные сделки ДДУ (основные данные)
SHEET_ZHK = "Справочник по ЖК"             # справочник ЖК (стадии, сроки ВВЭ, МКАД)

# Параметры подключения к PostgreSQL
DB_HOST = "localhost"
DB_PORT = 5432
DB_NAME = "real_estate_db"
DB_USER = "postgres"
DB_PASSWORD = "postgres"

# Настройки почты для отправки отчётов (SMTP).
# Если SMTP_USER пустой - письмо не отправляется, остальное работает.
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = ""        # ваша почта
SMTP_PASSWORD = ""    # пароль приложения
EMAIL_FROM = ""
EMAIL_TO = ""

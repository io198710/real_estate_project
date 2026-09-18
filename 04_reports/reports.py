# -*- coding: utf-8 -*-
"""
ШАГ 4. Отчётность.

Генерирует:
    - статичные графики (Matplotlib + Seaborn)  -> output/plots/*.png
    - интерактивный график (Plotly)            -> output/plots/*.html
    - PDF-отчёт с метриками и графиками        -> output/report.pdf
    - Excel-отчёт с несколькими листами        -> output/report.xlsx
    - отправку отчёта по email (SMTP)          -> если настроен config.py

Запуск отдельно:  python 04_reports/reports.py
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")  # сохраняем графики в файлы, ничего не открываем на экране
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer,
                                Table, TableStyle)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

sns.set_theme(style="whitegrid")

# логичный порядок стадий для графиков
STAGE_ORDER = ["после ВВЭ", "до ВВЭ <1 года", "1-2 года до ВВЭ",
               "2-3 года до ВВЭ", ">3 лет до ВВЭ (котлован)", "нет данных"]


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


# ---------- Статичные графики ----------

def make_plots(data, plots_dir):
    """Строит графики Matplotlib и Seaborn, сохраняет в PNG."""
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Распределение цены квадратного метра
    plt.figure(figsize=(10, 6))
    prices = data[data["источник_цены"] != "медиана"]["цена_квм"] / 1000
    plt.hist(prices, bins=60, color="steelblue", edgecolor="white")
    plt.title("Распределение цены квадратного метра по сделкам")
    plt.xlabel("Цена, тыс. руб./кв.м")
    plt.ylabel("Количество сделок")
    plt.savefig(os.path.join(plots_dir, "01_распределение_цены.png"), dpi=100)
    plt.close()

    # 2. Цена по классам жилья (boxplot)
    plt.figure(figsize=(10, 6))
    order = (data.groupby("Класс ЖК")["цена_квм"].median()
             .sort_values().index.tolist())
    sns.boxplot(data=data, x="Класс ЖК", y="цена_квм", order=order)
    plt.title("Цена кв.м по классам жилья")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "02_цена_по_классам.png"), dpi=100)
    plt.close()

    # 3. Динамика цены по месяцам + тренд + ПРОГНОЗ 2026
    monthly = pd.read_csv(os.path.join(config.OUTPUT_DIR, "monthly_prices.csv"),
                          encoding="utf-8-sig", parse_dates=["Месяц"])
    plt.figure(figsize=(12, 6))
    plt.plot(monthly["Месяц"], monthly["цена_квм"] / 1000,
             color="steelblue", alpha=0.6, label="средняя цена (факт)")
    plt.plot(monthly["Месяц"], monthly["тренд"] / 1000,
             color="red", linewidth=2, label="тренд (скользящее среднее)")
    fc = monthly[monthly["прогноз_2026"].notna()]
    plt.plot(fc["Месяц"], fc["прогноз_2026"] / 1000,
             color="green", linewidth=3, linestyle="--", label="ПРОГНОЗ 2026")
    plt.title("Средняя цена кв.м по месяцам и прогноз на 2026 год")
    plt.ylabel("Цена, тыс. руб./кв.м")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "03_динамика_и_прогноз_2026.png"), dpi=100)
    plt.close()

    # 4. Цена по стадии сдачи (котлован дешевле ВВЭ)
    plt.figure(figsize=(12, 6))
    stages = [s for s in STAGE_ORDER if s in set(data["стадия_сдачи"])]
    sns.boxplot(data=data, x="стадия_сдачи", y="цена_квм", order=stages)
    plt.title("Цена кв.м по стадии строительства на момент сделки")
    plt.xticks(rotation=20, fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "04_цена_по_стадии_сдачи.png"), dpi=100)
    plt.close()

    # 5. Корреляции числовых признаков
    num_cols = ["Площадь", "Этаж", "months_to_vve", "До МКАД, км",
                "Этажность макс", "цена_квм"]
    corr = data[num_cols].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Корреляции числовых признаков")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "05_корреляции.png"), dpi=100)
    plt.close()

    # 6. Источники цены сделки - итог шага очистки данных
    src = data["источник_цены"].value_counts()
    plt.figure(figsize=(8, 5))
    plt.bar(src.index, src.values, color="steelblue")
    plt.title("Откуда взята цена сделки (результат очистки)")
    plt.ylabel("Сделок")
    for i, v in enumerate(src.values):
        plt.text(i, v, f"{v:,}".replace(",", " "), ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "06_источники_цены.png"), dpi=100)
    plt.close()

    print("Графики сохранены в", plots_dir)


# ---------- Интерактивный график ----------

def plot_interactive(data, plots_dir):
    """Строит интерактивный график Plotly и сохраняет в HTML."""
    by_class = (data.groupby("Класс ЖК")
                .agg(sdelok=("цена_квм", "size"),
                     avg_price=("цена_квм", "mean"),
                     avg_area=("Площадь", "mean"))
                .round(0).reset_index())
    by_class.columns = ["Класс жилья", "Сделок", "Средняя цена кв.м",
                        "Средняя площадь"]

    fig = px.bar(by_class, x="Класс жилья", y="Средняя цена кв.м",
                 color="Класс жилья", text="Сделок",
                 title="Средняя цена кв.м по классам жилья (по сделкам)")
    fig.write_html(os.path.join(plots_dir, "07_интерактивный_график.html"))
    print("Интерактивный график сохранён")


# ---------- PDF-отчёт ----------

def make_pdf_report(metrics, plots_dir, out_file):
    """Собирает PDF-отчёт: таблица метрик + все графики."""
    # шрифт с поддержкой русского языка (идёт вместе с matplotlib)
    font_path = os.path.join(os.path.dirname(matplotlib.__file__),
                             "mpl-data", "fonts", "ttf", "DejaVuSans.ttf")
    pdfmetrics.registerFont(TTFont("DejaVu", font_path))

    title_style = ParagraphStyle("title", fontName="DejaVu", fontSize=16, spaceAfter=12)
    head_style = ParagraphStyle("head", fontName="DejaVu", fontSize=13,
                                spaceBefore=14, spaceAfter=8)
    text_style = ParagraphStyle("text", fontName="DejaVu", fontSize=10)

    story = [
        Paragraph("Отчёт: рынок новостроек Москвы", title_style),
        Paragraph("Сделки ДДУ из выписок | предполагаемая цена ДДУ "
                  "и прогноз на 2026 год", text_style),
        Paragraph("Ключевые метрики", head_style),
    ]

    # таблица с метриками
    rows = [["Показатель", "Значение"]] + flatten_metrics(metrics)
    table = Table(rows, colWidths=[11 * cm, 6 * cm])
    table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), "DejaVu"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4472C4")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EDF2FA")]),
    ]))
    story.append(table)

    # графики
    story.append(Paragraph("Графики", head_style))
    for name in sorted(os.listdir(plots_dir)):
        if name.endswith(".png"):
            story.append(Image(os.path.join(plots_dir, name), width=16 * cm, height=9 * cm))
            story.append(Spacer(1, 10))

    doc = SimpleDocTemplate(out_file, pagesize=A4,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="Отчёт по новостройкам Москвы")
    doc.build(story)
    print("PDF-отчёт:", out_file)


# ---------- Excel-отчёт ----------

def make_excel_report(data, metrics, out_file):
    """Собирает Excel-отчёт с несколькими листами (openpyxl)."""
    # лист 1: все метрики
    stats_df = pd.DataFrame(flatten_metrics(metrics), columns=["Показатель", "Значение"])

    # лист 1а: отчёт об очистке данных (что удалено/заполнено)
    clean_rows = flatten_metrics({"очистка_данных":
                                  metrics.get("очистка_данных", {})})
    clean_df = pd.DataFrame(clean_rows, columns=["Показатель", "Значение"])

    # лист 1б: результаты ML-моделей (регрессия + классификация)
    reg = metrics.get("регрессия_предполагаемая_цена", {})
    clf = metrics.get("классификация_переуступки", {})
    ml_rows = [("РЕГРЕССИЯ - предполагаемая цена ДДУ", "")]
    ml_rows += [(k, str(v)) for k, v in reg.items()
                if k != "важность_признаков"]
    ml_rows += [("важность признаков:", "")]
    ml_rows += [(k, str(v))
                for k, v in reg.get("важность_признаков", {}).items()]
    ml_rows += [("", ""), ("КЛАССИФИКАЦИЯ - переуступка", "")]
    ml_rows += [(k, str(v)) for k, v in clf.items()]
    ml_df = pd.DataFrame(ml_rows, columns=["Показатель", "Значение"])

    # лист 2: средние по классам жилья
    by_class = (data.groupby("Класс ЖК")
                .agg(sdelok=("цена_квм", "size"),
                     avg_price=("цена_квм", "mean"),
                     median_price=("цена_квм", "median"),
                     avg_area=("Площадь", "mean"))
                .round(0).reset_index())
    by_class.columns = ["Класс жилья", "Сделок", "Средняя цена кв.м",
                        "Медианная цена кв.м", "Средняя площадь"]

    # лист 3: динамика по месяцам + прогноз 2026
    monthly = pd.read_csv(os.path.join(config.OUTPUT_DIR, "monthly_prices.csv"),
                          encoding="utf-8-sig", parse_dates=["Месяц"])
    monthly[["цена_квм", "тренд", "прогноз_2026"]] = \
        monthly[["цена_квм", "тренд", "прогноз_2026"]].round(0)

    # лист 4: цена по стадии сдачи и году сделки
    stage_year = pd.read_csv(os.path.join(config.OUTPUT_DIR, "stage_prices.csv"),
                             encoding="utf-8-sig")
    pivot = (stage_year.pivot_table(index="deal_year", columns="стадия_сдачи",
                                    values="mean", observed=True).round(0))
    counts = (stage_year.pivot_table(index="deal_year", columns="стадия_сдачи",
                                     values="count", observed=True))

    with pd.ExcelWriter(out_file, engine="openpyxl") as writer:
        stats_df.to_excel(writer, sheet_name="Метрики", index=False)
        clean_df.to_excel(writer, sheet_name="Очистка данных", index=False)
        ml_df.to_excel(writer, sheet_name="ML-модели", index=False)
        by_class.to_excel(writer, sheet_name="По классам жилья", index=False)
        monthly.to_excel(writer, sheet_name="Динамика и прогноз 2026", index=False)
        pivot.to_excel(writer, sheet_name="Цена по стадии и году")
        counts.to_excel(writer, sheet_name="Сделок по стадии и году")
    print("Excel-отчёт:", out_file)


# ---------- Отправка по email ----------

def send_email(subject, body, attachments=None):
    """
    Отправляет письмо с отчётами через SMTP.
    Настройки задаются в config.py. Если они не заполнены - пропускаем.
    """
    if not config.SMTP_USER or not config.SMTP_PASSWORD:
        print("Отправка email отключена: не заполнены настройки SMTP в config.py")
        return False

    import smtplib
    from email.message import EmailMessage

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = config.EMAIL_FROM or config.SMTP_USER
    msg["To"] = config.EMAIL_TO
    msg.set_content(body)

    # вкладываем файлы
    for path in attachments or []:
        ext = "pdf" if path.endswith(".pdf") else \
            "vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        with open(path, "rb") as f:
            msg.add_attachment(f.read(), maintype="application",
                               subtype=ext, filename=os.path.basename(path))

    with smtplib.SMTP(config.SMTP_SERVER, config.SMTP_PORT) as server:
        server.starttls()
        server.login(config.SMTP_USER, config.SMTP_PASSWORD)
        server.send_message(msg)
    print("Письмо отправлено:", config.EMAIL_TO)
    return True


# ---------- Генерация всех отчётов ----------

def make_all_reports(data, metrics):
    """Строит все отчёты сразу: графики, HTML, PDF, Excel, письмо."""
    plots_dir = os.path.join(config.OUTPUT_DIR, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # чистим старые графики, чтобы не было мусора от прошлых запусков
    for old in os.listdir(plots_dir):
        os.remove(os.path.join(plots_dir, old))

    # 1-2. статичные и интерактивные графики
    make_plots(data, plots_dir)
    plot_interactive(data, plots_dir)

    # 3. PDF и Excel
    pdf_path = os.path.join(config.OUTPUT_DIR, "report.pdf")
    excel_path = os.path.join(config.OUTPUT_DIR, "report.xlsx")
    make_pdf_report(metrics, plots_dir, pdf_path)
    make_excel_report(data, metrics, excel_path)

    # 4. письмо с отчётами (если настроен SMTP)
    try:
        send_email("Отчёт по рынку новостроек Москвы",
                   "Здравствуйте! Во вложении отчёты по рынку новостроек.",
                   [pdf_path, excel_path])
    except Exception as e:
        print("Не удалось отправить письмо:", e)


if __name__ == "__main__":
    data = pd.read_csv(os.path.join(config.DATA_DIR, "clean_deals.csv"),
                       encoding="utf-8-sig", parse_dates=["Дата договора"])
    # признаки для графиков считаем так же, как в анализе
    sys.path.append(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "03_analysis"))
    from analysis import prepare_ml_data
    zhk = pd.read_csv(os.path.join(config.DATA_DIR, "zhk.csv"), encoding="utf-8-sig")
    data = prepare_ml_data(data, zhk)

    with open(os.path.join(config.OUTPUT_DIR, "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)
    make_all_reports(data, metrics)

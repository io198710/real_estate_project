# -*- coding: utf-8 -*-
"""
ШАГ 5. Простой REST API для доступа к результатам анализа.

Через этот API результаты может забрать любая система
бизнес-анализа (например, Power BI или дашборд).

Запуск:       python 05_automation/api_server.py
Адреса:
    http://localhost:8000/metrics   - метрики анализа (JSON)
    http://localhost:8000/health    - проверка, что сервис работает
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

import config  # noqa: E402


class ApiHandler(BaseHTTPRequestHandler):
    """Обработчик запросов к API."""

    def do_GET(self):
        if self.path == "/metrics":
            metrics_file = os.path.join(config.OUTPUT_DIR, "metrics.json")
            if not os.path.exists(metrics_file):
                self._send(404, {"error": "metrics.json не найден, запустите main.py"})
                return
            with open(metrics_file, encoding="utf-8") as f:
                metrics = json.load(f)
            self._send(200, metrics)

        elif self.path == "/health":
            self._send(200, {"status": "ok"})

        else:
            self._send(404, {"error": "неизвестный адрес"})

    def _send(self, code, data):
        """Отправляет JSON-ответ."""
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass  # не пишем служебные сообщения в консоль


if __name__ == "__main__":
    server = HTTPServer(("localhost", 8000), ApiHandler)
    print("API запущен:")
    print("  http://localhost:8000/metrics")
    print("  http://localhost:8000/health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nAPI остановлен")

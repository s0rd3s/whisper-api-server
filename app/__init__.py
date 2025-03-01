import json
from typing import Dict
from flask import Flask
import waitress

# Импорт классов и функций из других модулей
from .transcriber import WhisperTranscriber
from .routes import Routes
from .logger import logger

class WhisperServiceAPI:
    """Класс для API сервиса распознавания речи."""

    def __init__(self, config_path: str):
        """
        Инициализация API сервиса.

        Args:
            config_path: Путь к конфигурационному файлу.
        """
        # Загрузка конфигурации
        self.config = self._load_config(config_path)

        # Порт для сервиса
        self.port = self.config["service_port"]

        # Создание экземпляра транскрайбера
        self.transcriber = WhisperTranscriber(self.config)

        # Создание Flask-приложения
        self.app = Flask("whisper-service")

        # Регистрация маршрутов
        Routes(self.app, self.transcriber, self.config)
        # self._register_routes()

        logger.info(f"API сервис инициализирован, порт: {self.port}")
    
    def _load_config(self, config_path: str) -> Dict:
        """
        Загрузка конфигурации из JSON-файла.

        Args:
            config_path: Путь к файлу конфигурации.

        Returns:
            Словарь с параметрами конфигурации.

        Raises:
            FileNotFoundError: Если файл конфигурации не найден.
            json.JSONDecodeError: Если файл конфигурации содержит некорректный JSON.
        """
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            return config
        except FileNotFoundError as e:
            logger.error(f"Файл конфигурации не найден: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise

    def run(self):
        """Запуск сервиса."""
        logger.info(f"Запуск сервиса на порту {self.port}")
        
        # Использовать waitress для production-ready сервера
        waitress.serve(self.app, host='0.0.0.0', port=self.port)

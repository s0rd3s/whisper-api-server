import json
import os
from typing import Dict
from flask import Flask
from flask_cors import CORS
import waitress

# Импорт классов и функций из других модулей
from .transcriber import WhisperTranscriber
from .routes import Routes
from .utils import logger

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

        # Определение пути к директории static
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_folder_path = os.path.join(current_dir, 'static')
        
        # Создание Flask-приложения с явным указанием пути к static
        self.app = Flask("whisper-service", static_folder=static_folder_path)

        # Настройка CORS с явным разрешением всех методов, заголовков и источников
        CORS(self.app, resources={r"/*": {"origins": "*", "supports_credentials": True}})

        # Регистрация маршрутов
        Routes(self.app, self.transcriber, self.config)

        logger.info(f"API сервис инициализирован, порт: {self.port}")
        logger.info(f"Статические файлы будут обслуживаться из: {static_folder_path}")
    
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
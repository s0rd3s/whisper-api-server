"""
Модуль routes.py содержит классы для регистрации маршрутов API
для сервиса распознавания речи.
"""

import os
from flask import request, jsonify
from typing import Dict

from .transcriber_service import TranscriptionService
from .audio_sources import (
    UploadedFileSource,
    URLSource,
    Base64Source,
    LocalFileSource
)


class Routes:
    """Класс для регистрации всех эндпоинтов API."""

    def __init__(self, app, transcriber, config: Dict):
        """
        Инициализация маршрутов.

        Args:
            app: Flask-приложение.
            transcriber: Экземпляр транскрайбера.
            config: Словарь с конфигурацией.
        """
        self.app = app
        self.config = config
        self.transcription_service = TranscriptionService(transcriber, config)

        # Регистрация маршрутов
        self._register_routes()

    def _register_routes(self):
        """Регистрация всех эндпоинтов."""

        @self.app.route('/', methods=['GET'])
        def index():
            """Корень. Отдаёт HTML клиент."""
            return self.app.send_static_file('index.html')

        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Эндпоинт для проверки статуса сервиса."""
            return jsonify({
                "status": "ok",
                "version": self.config.get("version", "1.0.0")
            }), 200

        @self.app.route('/config', methods=['GET'])
        def get_config():
            """Эндпоинт для получения конфигурации сервиса."""
            return jsonify(self.config), 200

        @self.app.route('/local/transcriptions', methods=['POST'])
        def local_transcribe():
            """Эндпоинт для локальной транскрибации файла по пути на сервере."""
            data = request.json

            if not data or "file_path" not in data:
                return jsonify({"error": "No file_path provided"}), 400

            file_path = data["file_path"]
            source = LocalFileSource(file_path, self.config.get("max_file_size", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, data)
            return jsonify(response), status_code

        @self.app.route('/v1/models', methods=['GET'])
        def list_models():
            """Эндпоинт для получения списка доступных моделей."""
            return jsonify({
                "data": [
                    {
                        "id": os.path.basename(self.config["model_path"]),
                        "object": "model",
                        "owned_by": "openai",
                        "permissions": []
                    }
                ],
                "object": "list"
            }), 200

        @self.app.route('/v1/models/<model_id>', methods=['GET'])
        def retrieve_model(model_id):
            """Эндпоинт для получения информации о конкретной модели."""
            if model_id == os.path.basename(self.config["model_path"]):
                return jsonify({
                    "id": model_id,
                    "object": "model",
                    "owned_by": "openai",
                    "permissions": []
                }), 200
            else:
                return jsonify({
                    "error": "Model not found",
                    "details": f"Model '{model_id}' does not exist"
                }), 404

        @self.app.route('/v1/audio/transcriptions', methods=['POST'])
        def openai_transcribe_endpoint():
            """Эндпоинт для транскрибации аудиофайла (multipart-форма)."""
            source = UploadedFileSource(request.files, self.config.get("max_file_size", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, request.form)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/url', methods=['POST'])
        def transcribe_from_url():
            """Эндпоинт для транскрибации аудиофайла по URL."""
            data = request.json

            if not data or "url" not in data:
                return jsonify({
                    "error": "No URL provided",
                    "details": "Please provide 'url' in the JSON request"
                }), 400

            url = data["url"]
            # Извлекаем параметры транскрибации, если они есть
            params = {k: v for k, v in data.items() if k != "url"}

            source = URLSource(url, self.config.get("max_file_size", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, params)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/base64', methods=['POST'])
        def transcribe_from_base64():
            """Эндпоинт для транскрибации аудио, закодированного в base64."""
            data = request.json

            if not data or "file" not in data:
                return jsonify({
                    "error": "No base64 file provided",
                    "details": "Please provide 'file' in the JSON request"
                }), 400

            base64_data = data["file"]
            # Извлекаем параметры транскрибации, если они есть
            params = {k: v for k, v in data.items() if k != "file"}

            source = Base64Source(base64_data, self.config.get("max_file_size", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, params)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/multipart', methods=['POST'])
        def transcribe_multipart():
            """Эндпоинт для транскрибации аудиофайла, загруженного через форму."""
            source = UploadedFileSource(request.files, self.config.get("max_file_size", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, request.form)
            return jsonify(response), status_code

"""
Модуль routes.py содержит классы для обработки транскрибации аудиофайлов
и регистрации маршрутов API для сервиса распознавания речи.
"""

import os
import uuid
import tempfile
import time
import librosa
from flask import request, jsonify
from typing import Dict, Tuple

from .utils import logger
from .audio_sources import (
    AudioSource, 
    UploadedFileSource, 
    URLSource, 
    Base64Source, 
    LocalFileSource
)


class TranscriptionService:
    """Сервис для обработки и транскрибации аудиофайлов."""
    
    def __init__(self, transcriber, config: Dict):
        """
        Инициализация сервиса транскрибации.
        
        Args:
            transcriber: Экземпляр транскрайбера.
            config: Словарь с конфигурацией.
        """
        self.transcriber = transcriber
        self.config = config
        self.max_file_size_mb = self.config.get("max_file_size", 100)  # Default 100MB
        
    def get_audio_duration(self, file_path: str) -> float:
        """
        Определяет длительность аудиофайла в секундах.
        
        Args:
            file_path: Путь к аудиофайлу.
            
        Returns:
            Длительность в секундах.
        """
        try:
            y, sr = librosa.load(file_path, sr=None)
            duration = librosa.get_duration(y=y, sr=sr)
            return duration
        except Exception as e:
            logger.error(f"Ошибка при определении длительности файла: {e}")
            return 0.0
        
    def transcribe_from_source(self, source: AudioSource, params: Dict = None) -> Tuple[Dict, int]:
        """
        Транскрибирует аудиофайл из указанного источника.
        
        Args:
            source: Источник аудиофайла.
            params: Дополнительные параметры для транскрибации.
            
        Returns:
            Кортеж (JSON-ответ, HTTP-код).
        """
        # Получаем файл из источника
        file, filename, error = source.get_audio_file()
        
        # Обрабатываем ошибки получения файла
        if error:
            return jsonify({"error": error}), 400
            
        if not file:
            return jsonify({"error": "Failed to get audio file"}), 400
            
        # Извлекаем параметры из запроса, если они есть
        params = params or {}
        language = params.get('language', self.config.get('language', 'en'))
        temperature = float(params.get('temperature', 0.0))
        prompt = params.get('prompt', '')
        
        # Проверяем, запрошены ли временные метки
        return_timestamps = params.get('return_timestamps', self.config.get('return_timestamps', False))
        # Преобразуем строковое значение в булево, если необходимо
        if isinstance(return_timestamps, str):
            return_timestamps = return_timestamps.lower() in ('true', 't', 'yes', 'y', '1')
        
        # Временно изменяем настройку return_timestamps в транскрайбере
        original_return_timestamps = self.transcriber.return_timestamps
        self.transcriber.return_timestamps = return_timestamps
        
        # Сохраняем файл во временный файл
        temp_dir = tempfile.mkdtemp()
        temp_file_path = os.path.join(temp_dir, str(uuid.uuid4()) + "_" + os.path.basename(filename))
        file.save(temp_file_path)
        
        # Определяем длительность аудиофайла
        duration = self.get_audio_duration(temp_file_path)
        
        # Для файлов из внешних источников (URL, base64), закрываем их и выполняем очистку
        if hasattr(source, 'cleanup'):
            file.file.close()  # Закрываем файловый объект
            source.cleanup()  # Очищаем временные файлы источника
        
        try:
            start_time = time.time()
            result = self.transcriber.process_file(temp_file_path)
            processing_time = time.time() - start_time
            
            # Формируем ответ в зависимости от return_timestamps
            if return_timestamps:
                response = {
                    "segments": result.get("segments", []),
                    "text": result.get("text", ""),
                    "processing_time": processing_time,
                    "response_size_bytes": len(str(result).encode('utf-8')),
                    "duration_seconds": duration,
                    "model": os.path.basename(self.config["model_path"])
                }
            else:
                # Если не запрашивались временные метки, result - это строка
                response = {
                    "text": result,
                    "processing_time": processing_time,
                    "response_size_bytes": len(str(result).encode('utf-8')),
                    "duration_seconds": duration,
                    "model": os.path.basename(self.config["model_path"])
                }
            
            return jsonify(response), 200
            
        except Exception as e:
            logger.error(f"Ошибка при транскрибации: {e}")
            return jsonify({"error": str(e)}), 500
            
        finally:
            # Восстанавливаем оригинальное значение return_timestamps
            self.transcriber.return_timestamps = original_return_timestamps
            
            # Очистка временных файлов
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)


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
            
            return self.transcription_service.transcribe_from_source(source, data)
            
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
            return self.transcription_service.transcribe_from_source(source, request.form)
            
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
            return self.transcription_service.transcribe_from_source(source, params)
            
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
            return self.transcription_service.transcribe_from_source(source, params)
            
        @self.app.route('/v1/audio/transcriptions/multipart', methods=['POST'])
        def transcribe_multipart():
            """Эндпоинт для транскрибации аудиофайла, загруженного через форму."""
            source = UploadedFileSource(request.files, self.config.get("max_file_size", 100))
            return self.transcription_service.transcribe_from_source(source, request.form)

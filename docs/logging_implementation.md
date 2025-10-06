# Реализация логирования для обращений к API с невалидными типами файлов

## 1. Обновление app/utils.py

Добавление декоратора для логирования запросов с невалидными файлами:

```python
"""
Модуль utils.py содержит утилиты и вспомогательные функции.
"""

import logging
import functools
from flask import request
from .validators import ValidationError

# Настройка логирования
logger = logging.getLogger(__name__)


def log_invalid_file_request(func):
    """
    Декоратор для логирования запросов с невалидными файлами.
    
    Args:
        func: Декорируемая функция.
        
    Returns:
        Обернутая функция с логированием ошибок валидации файлов.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            # Получение информации о запросе
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
            endpoint = request.endpoint or 'unknown'
            method = request.method or 'unknown'
            
            # Получение имени файла из запроса
            filename = 'unknown'
            if 'file' in request.files:
                filename = request.files['file'].filename
            elif request.is_json:
                data = request.get_json()
                if data and 'file' in data:
                    filename = data.get('filename', 'base64_data')
            
            # Логирование обращения к API с невалидным файлом
            logger.warning(f"Обращение к эндпоинту {method} {endpoint} с невалидным файлом '{filename}' "
                          f"от клиента {client_ip}. Ошибка: {str(e)}")
            
            # Пробрасываем исключение дальше
            raise
    return wrapper
```

## 2. Обновление app/validators.py

Добавление логирования в методы валидации:

```python
"""
Модуль validators.py содержит классы и функции для валидации входных данных.
"""

import os
import magic
from typing import Dict, List, BinaryIO, Optional
from .utils import logger


class ValidationError(Exception):
    """Исключение для ошибок валидации."""
    pass


class FileValidator:
    """
    Класс для валидации файлов.
    
    Проверяет тип файла, размер и другие параметры на основе конфигурации.
    """
    
    def __init__(self, config: Dict):
        """
        Инициализация валидатора файлов.
        
        Args:
            config: Словарь с параметрами конфигурации.
        """
        self.validation_config = config.get("file_validation", {})
        self.max_file_size_mb = self.validation_config.get("max_file_size_mb", 100)
        self.allowed_extensions = self.validation_config.get("allowed_extensions", 
                                                             [".wav", ".mp3", ".ogg", ".flac", ".m4a"])
        self.allowed_mime_types = self.validation_config.get("allowed_mime_types", 
                                                            ["audio/wav", "audio/mpeg", "audio/ogg", 
                                                             "audio/flac", "audio/mp4"])
    
    def validate_file(self, file: BinaryIO, filename: str) -> bool:
        """
        Валидирует файл на основе конфигурации.
        
        Args:
            file: Файловый объект.
            filename: Имя файла.
            
        Returns:
            True, если файл прошел валидацию.
            
        Raises:
            ValidationError: Если файл не прошел валидацию.
        """
        try:
            # Проверка размера файла
            self._validate_file_size(file)
            
            # Проверка расширения файла
            self._validate_file_extension(filename)
            
            # Проверка MIME-типа файла
            self._validate_file_mime_type(file)
            
            return True
        except ValidationError as e:
            # Логирование общей ошибки валидации
            logger.warning(f"Ошибка валидации файла '{filename}': {str(e)}")
            raise
    
    def _validate_file_size(self, file: BinaryIO) -> None:
        """
        Валидирует размер файла.
        
        Args:
            file: Файловый объект.
            
        Raises:
            ValidationError: Если размер файла превышает максимально допустимый.
        """
        # Сохранение текущей позиции
        current_position = file.tell()
        
        # Переход в конец файла для определения размера
        file.seek(0, os.SEEK_END)
        file_length = file.tell()
        
        # Возврат к исходной позиции
        file.seek(current_position)
        
        max_size_bytes = self.max_file_size_mb * 1024 * 1024
        if file_length > max_size_bytes:
            logger.warning(f"Попытка загрузки файла размером {file_length / (1024*1024):.2f} МБ, "
                          f"что превышает максимально допустимый размер {self.max_file_size_mb} МБ")
            
            raise ValidationError(f"Размер файла ({file_length / (1024*1024):.2f} МБ) "
                                 f"превышает максимально допустимый ({self.max_file_size_mb} МБ)")
    
    def _validate_file_extension(self, filename: str) -> None:
        """
        Валидирует расширение файла.
        
        Args:
            filename: Имя файла.
            
        Raises:
            ValidationError: Если расширение файла не входит в список разрешенных.
        """
        if not any(filename.lower().endswith(ext.lower()) for ext in self.allowed_extensions):
            # Логирование попытки загрузки файла с неразрешенным расширением
            file_extension = os.path.splitext(filename)[1]
            logger.warning(f"Попытка загрузки файла с неразрешенным расширением '{file_extension}'. "
                          f"Имя файла: {filename}. Разрешенные расширения: {', '.join(self.allowed_extensions)}")
            
            raise ValidationError(f"Расширение файла не разрешено. "
                                 f"Разрешенные расширения: {', '.join(self.allowed_extensions)}")
    
    def _validate_file_mime_type(self, file: BinaryIO) -> None:
        """
        Валидирует MIME-тип файла.
        
        Args:
            file: Файловый объект.
            
        Raises:
            ValidationError: Если MIME-тип файла не входит в список разрешенных.
        """
        # Сохранение текущей позиции
        current_position = file.tell()
        
        try:
            # Чтение первых байтов для определения MIME-типа
            header = file.read(1024)
            mime_type = magic.from_buffer(header, mime=True)
            
            # Возврат к исходной позиции
            file.seek(current_position)
            
            if mime_type not in self.allowed_mime_types:
                # Логирование попытки загрузки файла с неразрешенным MIME-типом
                logger.warning(f"Попытка загрузки файла с неразрешенным MIME-типом '{mime_type}'. "
                              f"Разрешенные MIME-типы: {', '.join(self.allowed_mime_types)}")
                
                raise ValidationError(f"MIME-тип файла ({mime_type}) не разрешен. "
                                     f"Разрешенные MIME-типы: {', '.join(self.allowed_mime_types)}")
        except Exception as e:
            # Возврат к исходной позиции в случае ошибки
            file.seek(current_position)
            logger.warning(f"Не удалось определить MIME-тип файла: {e}")
            # Не прерываем валидацию, если не удалось определить MIME-тип
    
    @staticmethod
    def validate_local_file_path(file_path: str, allowed_directories: Optional[List[str]] = None) -> str:
        """
        Валидирует путь к локальному файлу для предотвращения атак обхода пути.
        
        Args:
            file_path: Путь к файлу.
            allowed_directories: Список разрешенных директорий.
            
        Returns:
            Нормализованный и проверенный путь к файлу.
            
        Raises:
            ValidationError: Если путь к файлу небезопасен.
        """
        # Нормализация пути
        normalized_path = os.path.normpath(file_path)
        
        # Если указаны разрешенные директории, проверяем, что путь находится в одной из них
        if allowed_directories:
            for allowed_dir in allowed_directories:
                full_allowed_path = os.path.abspath(allowed_dir)
                full_file_path = os.path.abspath(os.path.join(full_allowed_path, normalized_path))
                
                if full_file_path.startswith(full_allowed_path):
                    return full_file_path
            
            logger.warning(f"Попытка доступа к файлу вне разрешенных директорий: {file_path}")
            raise ValidationError("Путь к файлу не находится в разрешенных директориях")
        
        # Если разрешенные директории не указаны, просто возвращаем нормализованный путь
        return normalized_path
```

## 3. Обновление app/routes.py

Применение декоратора к эндпоинтам, обрабатывающим файлы:

```python
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
from .validators import ValidationError
from .async_tasks import transcribe_audio_async, task_manager
from .cache import model_cache
from .utils import logger, log_invalid_file_request


class Routes:
    """
    Класс для регистрации всех эндпоинтов API.
    
    Attributes:
        app (Flask): Flask-приложение.
        config (Dict): Словарь с конфигурацией.
        transcription_service (TranscriptionService): Сервис транскрибации.
        file_validator (FileValidator): Валидатор файлов.
    """

    def __init__(self, app, transcriber, config: Dict, file_validator):
        """
        Инициализация маршрутов.

        Args:
            app: Flask-приложение.
            transcriber: Экземпляр транскрайбера.
            config: Словарь с конфигурацией.
            file_validator: Валидатор файлов.
        """
        self.app = app
        self.config = config
        self.transcription_service = TranscriptionService(transcriber, config)
        self.file_validator = file_validator

        # Регистрация маршрутов
        self._register_routes()

    def _register_routes(self) -> None:
        """
        Регистрация всех эндпоинтов.
        """
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
            
            # Валидация пути к файлу
            try:
                validated_path = self.file_validator.validate_local_file_path(
                    file_path, 
                    allowed_directories=self.config.get("allowed_directories", [])
                )
            except ValidationError as e:
                # Логирование обращения к API с невалидным путем к файлу
                client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
                logger.warning(f"Обращение к эндпоинту /local/transcriptions с невалидным путем к файлу '{file_path}' "
                              f"от клиента {client_ip}. Ошибка: {str(e)}")
                return jsonify({"error": str(e)}), 400
            
            source = LocalFileSource(validated_path, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
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
        @log_invalid_file_request
        def openai_transcribe_endpoint():
            """Эндпоинт для транскрибации аудиофайла (multipart-форма)."""
            source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, request.form, self.file_validator)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/url', methods=['POST'])
        @log_invalid_file_request
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

            source = URLSource(url, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, params, self.file_validator)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/base64', methods=['POST'])
        @log_invalid_file_request
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

            source = Base64Source(base64_data, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, params, self.file_validator)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/multipart', methods=['POST'])
        @log_invalid_file_request
        def transcribe_multipart():
            """Эндпоинт для транскрибации аудиофайла, загруженного через форму."""
            source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, request.form, self.file_validator)
            return jsonify(response), status_code
        
        @self.app.route('/v1/audio/transcriptions/async', methods=['POST'])
        @log_invalid_file_request
        def transcribe_async():
            """Эндпоинт для асинхронной транскрибации аудиофайла."""
            source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            
            # Получаем файл
            file, filename, error = source.get_audio_file()
            
            if error:
                return jsonify({"error": error}), 400
            
            if not file:
                return jsonify({"error": "Failed to get audio file"}), 400
            
            # Валидация файла
            try:
                self.file_validator.validate_file(file, filename)
            except ValidationError as e:
                return jsonify({"error": str(e)}), 400
            
            # Сохраняем файл во временный файл
            from .file_manager import temp_file_manager
            with temp_file_manager.temp_file() as temp_path:
                file.save(temp_path)
                
                # Запускаем асинхронную транскрибацию
                task_id = transcribe_audio_async(temp_path, self.transcription_service.transcriber)
                
                return jsonify({"task_id": task_id}), 202
        
        @self.app.route('/v1/tasks/<task_id>', methods=['GET'])
        def get_task_status(task_id):
            """Эндпоинт для получения статуса асинхронной задачи."""
            task_info = task_manager.get_task_status(task_id)
            
            if not task_info:
                return jsonify({"error": "Task not found"}), 404
            
            response = {
                "task_id": task_id,
                "status": task_info["status"]
            }
            
            if task_info["status"] == "completed":
                response["result"] = task_info["result"]
            elif task_info["status"] == "failed":
                response["error"] = task_info["error"]
            
            return jsonify(response)
```

## 4. Обновление app/transcriber_service.py

Добавление логирования в метод transcribe_from_source:

```python
"""
Модуль transcriber_service.py содержит класс TranscriptionService,
который отвечает за обработку и транскрибацию аудиофайлов.
"""

import os
import uuid
import tempfile
import time
import librosa
from typing import Dict, Tuple

from .utils import logger
from .history_logger import HistoryLogger
from .audio_sources import AudioSource
from .validators import FileValidator, ValidationError


class TranscriptionService:
    """
    Сервис для обработки и транскрибации аудиофайлов.
    
    Attributes:
        transcriber: Экземпляр транскрайбера.
        config (Dict): Словарь с конфигурацией.
        max_file_size_mb (int): Максимальный размер файла в МБ.
        history (HistoryLogger): Объект журналирования.
    """

    def __init__(self, transcriber, config: Dict):
        """
        Инициализация сервиса транскрибации.

        Args:
            transcriber: Экземпляр транскрайбера.
            config: Словарь с конфигурацией.
        """
        self.transcriber = transcriber
        self.config = config
        self.max_file_size_mb = self.config.get("file_validation", {}).get("max_file_size_mb", 100)

        # Объект журналирования
        self.history = HistoryLogger(config)

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

    def transcribe_from_source(self, source: AudioSource, params: Dict = None, file_validator: FileValidator = None) -> Tuple[Dict, int]:
        """
        Транскрибирует аудиофайл из указанного источника.

        Args:
            source: Источник аудиофайла.
            params: Дополнительные параметры для транскрибации.
            file_validator: Валидатор файлов.

        Returns:
            Кортеж (JSON-ответ, HTTP-код).
        """
        # Получаем файл из источника
        file, filename, error = source.get_audio_file()

        # Обрабатываем ошибки получения файла
        if error:
            logger.warning(f"Ошибка получения файла из источника: {error}")
            return {"error": error}, 400

        if not file:
            logger.warning("Не удалось получить аудиофайл из источника")
            return {"error": "Failed to get audio file"}, 400
        
        # Валидация файла, если предоставлен валидатор
        if file_validator:
            try:
                file_validator.validate_file(file, filename)
            except ValidationError as e:
                # Логирование ошибки валидации
                logger.warning(f"Ошибка валидации файла '{filename}': {str(e)}")
                return {"error": str(e)}, 400

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
        from .file_manager import temp_file_manager
        with temp_file_manager.temp_file() as temp_file_path:
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

                # Журналирование результата
                self.history.save(response, filename)

                return response, 200

            except Exception as e:
                logger.error(f"Ошибка при транскрибации: {e}")
                return {"error": str(e)}, 500

            finally:
                # Восстанавливаем оригинальное значение return_timestamps
                self.transcriber.return_timestamps = original_return_timestamps
```

## 5. Примеры сообщений в логах

Примеры сообщений, которые будут появляться в логах при попытках загрузки невалидных файлов:

```
WARNING: Попытка загрузки файла с неразрешенным расширением '.pdf'. Имя файла: document.pdf. Разрешенные расширения: .wav, .mp3, .ogg, .flac, .m4a
WARNING: Ошибка валидации файла 'document.pdf': Расширение файла не разрешено. Разрешенные расширения: .wav, .mp3, .ogg, .flac, .m4a
WARNING: Обращение к эндпоинту POST /v1/audio/transcriptions с невалидным файлом 'document.pdf' от клиента 192.168.1.100. Ошибка: Расширение файла не разрешено. Разрешенные расширения: .wav, .mp3, .ogg, .flac, .m4a
```

```
WARNING: Попытка загрузки файла с неразрешенным MIME-типом 'application/pdf'. Разрешенные MIME-типы: audio/wav, audio/mpeg, audio/ogg, audio/flac, audio/mp4
WARNING: Ошибка валидации файла 'document.pdf': MIME-тип файла (application/pdf) не разрешен. Разрешенные MIME-типы: audio/wav, audio/mpeg, audio/ogg, audio/flac, audio/mp4
WARNING: Обращение к эндпоинту POST /v1/audio/transcriptions с невалидным файлом 'document.pdf' от клиента 192.168.1.100. Ошибка: MIME-тип файла (application/pdf) не разрешен. Разрешенные MIME-типы: audio/wav, audio/mpeg, audio/ogg, audio/flac, audio/mp4
```

```
WARNING: Попытка загрузки файла размером 150.00 МБ, что превышает максимально допустимый размер 100 МБ
WARNING: Ошибка валидации файла 'large_audio.wav': Размер файла (150.00 МБ) превышает максимально допустимый (100 МБ)
WARNING: Обращение к эндпоинту POST /v1/audio/transcriptions с невалидным файлом 'large_audio.wav' от клиента 192.168.1.100. Ошибка: Размер файла (150.00 МБ) превышает максимально допустимый (100 МБ)
```

Эти сообщения позволят легко отслеживать и анализировать попытки загрузки невалидных файлов, что поможет в дальнейшем улучшении сервиса и выявлении потенциальных проблем.
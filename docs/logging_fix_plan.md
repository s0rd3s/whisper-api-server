# План исправления проблемы с логированием

## Описание проблемы

После исправления циклического импорта логирование перестало работать - сообщения не отображаются ни на экране, ни в файле. Это связано с тем, что логгеры в разных модулях не настроены должным образом.

## Причины проблемы

1. Каждый модуль создает свой собственный логгер через `logging.getLogger(__name__)`, но ни один из них не настроен для вывода сообщений.
2. Отсутствует централизованная настройка логирования для всего приложения.
3. Возможно, отсутствует настройка обработчиков (handlers) для вывода логов в консоль и файл.

## Решение

### 1. Создание централизованного модуля логирования

Создадим новый модуль `app/logging_config.py`, который будет отвечать за настройку логирования для всего приложения:

```python
"""
Модуль logging_config.py содержит централизованную настройку логирования.
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logging(log_level=logging.INFO, log_file=None):
    """
    Настройка логирования для всего приложения.
    
    Args:
        log_level: Уровень логирования (по умолчанию INFO).
        log_file: Путь к файлу для записи логов (опционально).
    """
    # Создаем корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Очищаем существующие обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Добавляем обработчик для вывода в консоль
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # Добавляем обработчик для записи в файл, если указан путь
    if log_file:
        # Создаем директорию для файла логов, если она не существует
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10 МБ
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Устанавливаем уровень логирования для логгеров в других модулях
    logging.getLogger('app').setLevel(log_level)
    
    return root_logger
```

### 2. Обновление app/utils.py

```python
"""
Модуль utils.py содержит утилиты и вспомогательные функции.
"""

import logging
import functools
from flask import request

# Получаем логгер из централизованной настройки
logger = logging.getLogger('app.utils')


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
        except Exception as e:
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

### 3. Обновление app/validators.py

```python
"""
Модуль validators.py содержит классы и функции для валидации входных данных.
"""

import os
import magic
from typing import Dict, List, BinaryIO, Optional
import logging

# Получаем логгер из централизованной настройки
logger = logging.getLogger('app.validators')


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

### 4. Обновление app/__init__.py

```python
import json
import os
import logging
from typing import Dict
from flask import Flask
from flask_cors import CORS
import waitress

# Импорт классов и функций из других модулей
from .transcriber import WhisperTranscriber
from .routes import Routes
from .validators import FileValidator
from .file_manager import temp_file_manager
from .logging_config import setup_logging


class WhisperServiceAPI:
    """
    Класс для API сервиса распознавания речи.
    
    Attributes:
        config (Dict): Словарь с параметрами конфигурации.
        port (int): Порт для сервиса.
        transcriber (WhisperTranscriber): Экземпляр транскрайбера.
        app (Flask): Flask-приложение.
        file_validator (FileValidator): Валидатор файлов.
    """

    def __init__(self, config_path: str):
        """
        Инициализация API сервиса.

        Args:
            config_path: Путь к конфигурационному файлу.
        """
        # Загрузка конфигурации
        self.config = self._load_config(config_path)

        # Настройка логирования
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        log_file = self.config.get("log_file", "logs/whisper_api.log")
        setup_logging(log_level=log_level, log_file=log_file)
        
        # Получаем логгер
        self.logger = logging.getLogger('app')
        self.logger.info("Инициализация API сервиса")

        # Порт для сервиса
        self.port = self.config["service_port"]

        # Создание экземпляра транскрайбера
        self.transcriber = WhisperTranscriber(self.config)
        
        # Создание валидатора файлов
        self.file_validator = FileValidator(self.config)

        # Определение пути к директории static
        current_dir = os.path.dirname(os.path.abspath(__file__))
        static_folder_path = os.path.join(current_dir, 'static')
        
        # Создание Flask-приложения с явным указанием пути к static
        self.app = Flask("whisper-service", static_folder=static_folder_path)

        # Настройка CORS с явным разрешением всех методов, заголовков и источников
        CORS(self.app)

        # Регистрация маршрутов
        Routes(self.app, self.transcriber, self.config, self.file_validator)

        self.logger.info(f"API сервис инициализирован, порт: {self.port}")
        self.logger.info(f"Статические файлы будут обслуживаться из: {static_folder_path}")
    
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
            self.logger.error(f"Файл конфигурации не найден: {e}")
            raise
        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка при загрузке конфигурации: {e}")
            raise

    def run(self) -> None:
        """
        Запуск сервиса.
        """
        self.logger.info(f"Запуск сервиса на порту {self.port}")
        
        # Использовать waitress для production-ready сервера
        waitress.serve(self.app, host='0.0.0.0', port=self.port)
    
    def cleanup(self) -> None:
        """
        Очистка ресурсов перед завершением работы.
        """
        self.logger.info("Очистка ресурсов перед завершением работы")
        temp_file_manager.cleanup_all()
```

### 5. Обновление config.json

```json
{
  "service_port": 5000,
  "model_path": "models/whisper-medium",
  "language": "ru",
  "chunk_length_s": 30,
  "batch_size": 24,
  "max_new_tokens": 128,
  "return_timestamps": false,
  "temperature": 0.0,
  "audio_rate": 16000,
  "norm_level": "-0.5",
  "compand_params": "0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2",
  "file_validation": {
    "max_file_size_mb": 100,
    "allowed_extensions": [".wav", ".mp3", ".ogg", ".flac", ".m4a"],
    "allowed_mime_types": ["audio/wav", "audio/mpeg", "audio/ogg", "audio/flac", "audio/mp4"]
  },
  "allowed_directories": [],
  "version": "1.0.0",
  "log_level": "INFO",
  "log_file": "logs/whisper_api.log"
}
```

## Преимущества предложенного решения

1. **Централизованное управление**: Вся настройка логирования находится в одном модуле.
2. **Гибкость**: Возможность настройки уровня логирования и файла для записи логов через конфигурацию.
3. **Избегание циклических импортов**: Модуль логирования не зависит от других модулей приложения.
4. **Полное покрытие**: Все модули используют настроенные логгеры.
5. **Информативность**: Логи содержат временную метку, имя модуля, уровень и сообщение.

## Реализация

1. Создать файл `app/logging_config.py`
2. Обновить файлы `app/utils.py`, `app/validators.py`, `app/__init__.py`
3. Обновить `config.json` с добавлением настроек логирования
4. Создать директорию `logs` в корне проекта, если она не существует

Это решение должно полностью восстановить функциональность логирования и при этом избежать циклических импортов.
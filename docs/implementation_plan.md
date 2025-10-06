# План реализации улучшений для Whisper API Server

## 1. Централизованное управление временными файлами

### Новые файлы для создания

#### app/file_manager.py
```python
"""
Модуль file_manager.py содержит классы для централизованного управления временными файлами.
Предоставляет унифицированный интерфейс для создания, отслеживания и очистки временных файлов.
"""

import os
import uuid
import tempfile
import contextlib
from typing import List, Tuple, Optional, Generator
from .utils import logger


class TempFileManager:
    """
    Класс для централизованного управления временными файлами.
    
    Предоставляет методы для создания временных файлов и их последующей очистки.
    Использует контекстные менеджеры для автоматической очистки ресурсов.
    """
    
    def __init__(self):
        """
        Инициализация менеджера временных файлов.
        """
        self.temp_files = []
        self.temp_dirs = []
    
    def create_temp_file(self, suffix: str = ".wav") -> Tuple[str, str]:
        """
        Создает временный файл с уникальным именем.
        
        Args:
            suffix: Расширение временного файла.
            
        Returns:
            Кортеж (путь к файлу, путь к временной директории).
        """
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}{suffix}")
        
        self.temp_files.append(temp_file)
        self.temp_dirs.append(temp_dir)
        
        logger.debug(f"Создан временный файл: {temp_file}")
        return temp_file, temp_dir
    
    def cleanup_temp_files(self, file_paths: Optional[List[str]] = None) -> None:
        """
        Очищает временные файлы и директории.
        
        Args:
            file_paths: Список путей к файлам для очистки. Если None, очищает все отслеживаемые файлы.
        """
        paths_to_clean = file_paths if file_paths is not None else self.temp_files
        
        for path in paths_to_clean:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Удален временный файл: {path}")
                    
                    # Попытка удалить директорию, если она пуста
                    temp_dir = os.path.dirname(path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                        logger.debug(f"Удалена временная директория: {temp_dir}")
                        
                        # Удаление из списка отслеживаемых директорий
                        if temp_dir in self.temp_dirs:
                            self.temp_dirs.remove(temp_dir)
            except Exception as e:
                logger.warning(f"Не удалось очистить временный файл {path}: {e}")
        
        # Удаление файлов из списка отслеживаемых
        if file_paths is None:
            self.temp_files.clear()
        else:
            for path in file_paths:
                if path in self.temp_files:
                    self.temp_files.remove(path)
    
    @contextlib.contextmanager
    def temp_file(self, suffix: str = ".wav") -> Generator[str, None, None]:
        """
        Контекстный менеджер для создания и автоматической очистки временного файла.
        
        Args:
            suffix: Расширение временного файла.
            
        Yields:
            Путь к временному файлу.
        """
        temp_file, _ = self.create_temp_file(suffix)
        try:
            yield temp_file
        finally:
            self.cleanup_temp_files([temp_file])
    
    def cleanup_all(self) -> None:
        """
        Очищает все отслеживаемые временные файлы и директории.
        """
        self.cleanup_temp_files()
        for temp_dir in self.temp_dirs:
            try:
                if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                    os.rmdir(temp_dir)
                    logger.debug(f"Удалена временная директория: {temp_dir}")
            except Exception as e:
                logger.warning(f"Не удалось очистить временную директорию {temp_dir}: {e}")
        self.temp_dirs.clear()


# Глобальный экземпляр менеджера временных файлов
temp_file_manager = TempFileManager()
```

#### app/context_managers.py
```python
"""
Модуль context_managers.py содержит контекстные менеджеры для управления ресурсами.
"""

import os
import contextlib
from typing import Generator, BinaryIO
from .utils import logger


@contextlib.contextmanager
def open_file(file_path: str, mode: str = 'rb') -> Generator[BinaryIO, None, None]:
    """
    Контекстный менеджер для безопасного открытия и закрытия файлов.
    
    Args:
        file_path: Путь к файлу.
        mode: Режим открытия файла.
        
    Yields:
        Файловый объект.
    """
    file_obj = None
    try:
        file_obj = open(file_path, mode)
        yield file_obj
    except Exception as e:
        logger.error(f"Ошибка при работе с файлом {file_path}: {e}")
        raise
    finally:
        if file_obj:
            file_obj.close()


@contextlib.contextmanager
def audio_file(file_path: str) -> Generator[BinaryIO, None, None]:
    """
    Контекстный менеджер для работы с аудиофайлами.
    
    Args:
        file_path: Путь к аудиофайлу.
        
    Yields:
        Файловый объект в бинарном режиме.
    """
    with open_file(file_path, 'rb') as f:
        yield f
```

## 2. Валидация входных данных с настраиваемыми ограничениями размера файла

### Обновления в config.json
```json
{
    "service_port": 5042,
    "model_path": "/home/text-generation/models.whisper/antony-ties-podlodka-v1.2",
    "language": "russian",
    "enable_history": true,
    "chunk_length_s": 28,
    "batch_size": 8,
    "max_new_tokens": 384,
    "temperature": 0.01,
    "return_timestamps": false,
    "audio_rate": 8000,
    "norm_level": "-0.55",
    "compand_params": "0.3,1 -90,-90,-70,-50,-40,-15,0,0 -7 0 0.15",
    "file_validation": {
        "max_file_size_mb": 100,
        "allowed_extensions": [".wav", ".mp3", ".ogg", ".flac", ".m4a"],
        "allowed_mime_types": ["audio/wav", "audio/mpeg", "audio/ogg", "audio/flac", "audio/mp4"]
    }
}
```

### Новый файл app/validators.py
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
        # Проверка размера файла
        self._validate_file_size(file)
        
        # Проверка расширения файла
        self._validate_file_extension(filename)
        
        # Проверка MIME-типа файла
        self._validate_file_mime_type(file)
        
        return True
    
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
            
            raise ValidationError("Путь к файлу не находится в разрешенных директориях")
        
        # Если разрешенные директории не указаны, просто возвращаем нормализованный путь
        return normalized_path
```

## 3. Документация - добавление докстрингов

### Обновленный app/transcriber.py
```python
"""
Модуль transcriber.py содержит класс WhisperTranscriber, который использует модель Whisper от 
OpenAI для транскрибации аудиофайлов в текст. Класс включает в себя методы для загрузки модели, 
обработки аудио (с использованием класса AudioProcessor), и выполнения транскрибации. 
Обрабатывает выбор устройства (CPU, CUDA, MPS) для выполнения вычислений и обеспечивает 
возможность использования Flash Attention 2 для ускорения работы модели на поддерживаемых GPU.
"""

import time
from typing import Dict, Tuple, Union

import librosa
import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

from .audio_processor import AudioProcessor
from .file_manager import temp_file_manager
from .utils import logger


class WhisperTranscriber:
    """
    Класс для распознавания речи с помощью модели Whisper.
    
    Attributes:
        config (Dict): Словарь с параметрами конфигурации.
        model_path (str): Путь к модели Whisper.
        language (str): Язык распознавания.
        chunk_length_s (int): Длина аудиочанка в секундах.
        batch_size (int): Размер пакета для обработки.
        max_new_tokens (int): Максимальное количество новых токенов для генерации.
        return_timestamps (bool): Флаг возврата временных меток.
        temperature (float): Параметр температуры для генерации.
        torch_dtype (torch.dtype): Оптимальный тип данных для тензоров.
        audio_processor (AudioProcessor): Объект для обработки аудио.
        device (torch.device): Устройство для вычислений.
        model (WhisperForConditionalGeneration): Загруженная модель Whisper.
        processor (WhisperProcessor): Процессор для модели Whisper.
        asr_pipeline (pipeline): Пайплайн для автоматического распознавания речи.
    """
    
    def __init__(self, config: Dict):
        """
        Инициализация транскрайбера.

        Args:
            config: Словарь с параметрами конфигурации.
        """
        self.config = config
        self.model_path = config["model_path"]
        self.language = config["language"]
        self.chunk_length_s = config["chunk_length_s"]
        self.batch_size = config["batch_size"]
        self.max_new_tokens = config["max_new_tokens"]
        self.return_timestamps = config["return_timestamps"]
        self.temperature = config["temperature"]

        # Оптимальный тип для тензоров
        self.torch_dtype = torch.bfloat16

        # Создаем объект для обработки аудио
        self.audio_processor = AudioProcessor(config)

        # Определяем устройство для вычислений
        self.device = self._get_device()

        # Загружаем модель при инициализации
        self._load_model()

    def _get_device(self) -> torch.device:
        """
        Определение доступного устройства для вычислений.
        
        Returns:
            Объект устройства PyTorch.
        """
        if torch.cuda.is_available():
            # Проверяем, доступна ли GPU с индексом 1
            if torch.cuda.device_count() > 1:
                logger.info("Используется CUDA GPU с индексом 1 для вычислений")
                return torch.device("cuda:1")
            else:
                logger.info("Доступна только одна CUDA GPU, используется GPU с индексом 0")
                return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("Используется MPS (Apple Silicon) для вычислений")
            # Обходное решение для MPS
            setattr(torch.distributed, "is_initialized", lambda: False)
            return torch.device("mps")
        else:
            logger.info("Используется CPU для вычислений")
            return torch.device("cpu")

    def _load_model(self) -> None:
        """
        Загрузка модели и процессора.
        
        Raises:
            Exception: Если не удалось загрузить модель.
        """
        logger.info(f"Загрузка модели из {self.model_path}")

        try:
            # Проверка возможности использования Flash Attention 2
            if self.device.type == "cuda":
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True,
                    attn_implementation="flash_attention_2"
                ).to(self.device)
                logger.info("Используется Flash Attention 2")
            else:
                self.model = WhisperForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    low_cpu_mem_usage=True,
                    use_safetensors=True
                ).to(self.device)
        except Exception as e:
            logger.warning(f"Не удалось загрузить модель с Flash Attention: {e}")
            # Fallback к обычной версии
            self.model = WhisperForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
                use_safetensors=True
            ).to(self.device)

        self.processor = WhisperProcessor.from_pretrained(self.model_path)

        self.asr_pipeline = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=self.chunk_length_s,
            batch_size=self.batch_size,
            return_timestamps=self.return_timestamps,
            torch_dtype=self.torch_dtype,
            device=self.device,
        )

        logger.info("Модель успешно загружена и готова к использованию")

    def _load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Загрузка аудиофайла с использованием librosa.

        Args:
            file_path: Путь к аудиофайлу.

        Returns:
            Кортеж (массив numpy, частота дискретизации).
            
        Raises:
            Exception: Если не удалось загрузить аудиофайл.
        """
        try:
            audio_array, sampling_rate = librosa.load(file_path, sr=16000)
            return audio_array, sampling_rate
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио {file_path}: {e}")
            raise

    def transcribe(self, audio_path: str) -> Union[str, Dict]:
        """
        Транскрибация аудиофайла.
        
        Args:
            audio_path: Путь к обработанному аудиофайлу.

        Returns:
            В зависимости от параметра return_timestamps:
            - Если return_timestamps=False: строка с распознанным текстом
            - Если return_timestamps=True: словарь с ключами "segments" (список словарей с ключами start_time_ms, end_time_ms, text) и "text" (полный текст)
        """
        logger.info(f"Начало транскрибации файла: {audio_path}")

        # Загрузка аудио в формате numpy array
        audio_array, sampling_rate = self._load_audio(audio_path)

        # Транскрибация с корректным форматом данных
        result = self.asr_pipeline(
            {"raw": audio_array, "sampling_rate": sampling_rate}, 
            generate_kwargs={"language": self.language, "max_new_tokens": self.max_new_tokens, "temperature": self.temperature},
            return_timestamps=self.return_timestamps
        )

        # Если временные метки не запрошены, возвращаем только текст
        if not self.return_timestamps:
            transcribed_text = result.get("text", "")
            logger.info(f"Транскрибация завершена: получено {len(transcribed_text)} символов текста")
            return transcribed_text
        
        # Если временные метки запрошены, обрабатываем и форматируем результат
        segments = []
        full_text = result.get("text", "")
        
        if "chunks" in result:
            # Для новых версий модели Whisper
            for chunk in result["chunks"]:
                start_time = chunk.get("timestamp", [0, 0])[0]
                end_time = chunk.get("timestamp", [0, 0])[1]
                text = chunk.get("text", "").strip()
                
                segments.append({
                    "start_time_ms": int(start_time * 1000),
                    "end_time_ms": int(end_time * 1000),
                    "text": text
                })
        elif hasattr(result, "get") and "segments" in result:
            # Для старых версий модели Whisper
            for segment in result["segments"]:
                start_time = segment.get("start", 0)
                end_time = segment.get("end", 0)
                text = segment.get("text", "").strip()
                
                segments.append({
                    "start_time_ms": int(start_time * 1000),
                    "end_time_ms": int(end_time * 1000),
                    "text": text
                })
        else:
            logger.warning("Временные метки запрошены, но не найдены в результате транскрибации")
        
        logger.info(f"Транскрибация с временными метками завершена: получено {len(segments)} сегментов")
        
        # Возвращаем словарь с сегментами и полным текстом
        return {
            "segments": segments,
            "text": full_text
        }

    def process_file(self, input_path: str) -> Union[str, Dict]:
        """
        Полный процесс обработки и транскрибации аудиофайла.
        
        Args:
            input_path: Путь к исходному аудиофайлу.
            
        Returns:
            В зависимости от параметра return_timestamps:
            - Если return_timestamps=False: строка с распознанным текстом
            - Если return_timestamps=True: словарь с ключами "segments" и "text"
        """
        start_time = time.time()
        logger.info(f"Начало обработки файла: {input_path}")

        temp_files = []

        try:
            # Обработка аудио (конвертация, нормализация, добавление тишины)
            processed_path, temp_files = self.audio_processor.process_audio(input_path)

            # Транскрибация
            result = self.transcribe(processed_path)

            elapsed_time = time.time() - start_time
            logger.info(f"Обработка и транскрибация завершены за {elapsed_time:.2f} секунд")

            return result

        finally:
            # Очистка временных файлов
            temp_file_manager.cleanup_temp_files(temp_files)
```

## 4. Улучшения управления ресурсами

### Обновленный app/audio_processor.py
```python
"""
Модуль audio_processor.py содержит класс AudioProcessor, предназначенный для предобработки аудиофайлов 
перед их использованием в системах распознавания речи. Класс предоставляет методы для конвертации 
аудио в формат WAV с частотой дискретизации 16 кГц, нормализации уровня громкости, 
добавления тишины в начало записи, а также для удаления временных файлов, созданных в процессе обработки. 
"""

import os
import subprocess
import uuid
from typing import Dict, Tuple

from .file_manager import temp_file_manager
from .context_managers import open_file
from .utils import logger


class AudioProcessor:
    """
    Класс для предобработки аудиофайлов перед распознаванием.
    
    Attributes:
        config (Dict): Словарь с параметрами конфигурации.
        norm_level (str): Уровень нормализации аудио.
        compand_params (str): Параметры компрессора аудио.
    """
    
    def __init__(self, config: Dict):
        """
        Инициализация обработчика аудио.
        
        Args:
            config: Словарь с параметрами конфигурации.
        """
        self.config = config
        self.norm_level = config.get("norm_level", "-0.5")
        self.compand_params = config.get("compand_params", "0.3,1 -90,-90,-70,-70,-60,-20,0,0 -5 0 0.2")
    
    def convert_to_wav(self, input_path: str) -> str:
        """
        Конвертация входного аудиофайла в WAV формат с частотой дискретизации 16 кГц.
        
        Args:
            input_path: Путь к исходному аудиофайлу.
            
        Returns:
            Путь к сконвертированному WAV-файлу.
            
        Raises:
            subprocess.CalledProcessError: Если произошла ошибка при конвертации.
        """
        audio_rate = self.config["audio_rate"]

        # Проверка расширения файла
        if input_path.lower().endswith('.wav'):
            # Проверяем, нужно ли преобразовывать WAV-файл (например, если частота не 16 кГц)
            try:
                info = subprocess.check_output(['soxi', input_path]).decode()
                if f'{audio_rate} Hz' in info:
                    logger.info(f"Файл {input_path} уже в формате WAV с частотой {audio_rate} Гц")
                    return input_path
            except subprocess.CalledProcessError:
                logger.warning(f"Не удалось получить информацию о WAV-файле {input_path}")
                # Продолжаем конвертацию, чтобы быть уверенными в формате

        # Создаем временный файл для WAV
        output_path, _ = temp_file_manager.create_temp_file(".wav")
        
        # Команда для конвертации
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", input_path,
            "-ar", f"{audio_rate}",
            "-ac", "1",  # Монофонический звук
            output_path
        ]
        
        logger.info(f"Конвертация в WAV: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Файл конвертирован в WAV: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при конвертации в WAV: {e.stderr.decode()}")
            raise
    
    def normalize_audio(self, input_path: str) -> str:
        """
        Нормализация аудиофайла с использованием sox.
        
        Args:
            input_path: Путь к WAV-файлу.
            
        Returns:
            Путь к нормализованному WAV-файлу.
            
        Raises:
            subprocess.CalledProcessError: Если произошла ошибка при нормализации.
        """
        # Создаем временный файл для нормализованного аудио
        output_path, _ = temp_file_manager.create_temp_file("_normalized.wav")
        
        # Команда для нормализации аудио с помощью sox
        cmd = [
            "sox", 
            input_path, 
            output_path, 
            "norm", self.norm_level,
            "compand"
        ] + self.compand_params.split()
        
        logger.info(f"Нормализация аудио: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Аудио нормализовано: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при нормализации аудио: {e.stderr.decode()}")
            raise
    
    def add_silence(self, input_path: str) -> str:
        """
        Добавляет тишину в начало аудиофайла.
        
        Args:
            input_path: Путь к аудиофайлу.
            
        Returns:
            Путь к аудиофайлу с добавленной тишиной.
            
        Raises:
            subprocess.CalledProcessError: Если произошла ошибка при добавлении тишины.
        """
        # Создаем временный файл
        output_path, _ = temp_file_manager.create_temp_file("_silence.wav")
        
        # Команда для добавления тишины в начало файла
        cmd = [
            "sox",
            input_path,
            output_path,
            "pad", "2.0", "1.0"  # Добавление тишины в начале и в конце (секунды)
        ]
        
        logger.info(f"Добавление тишины: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Тишина добавлена: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при добавлении тишины: {e.stderr.decode()}")
            raise
    
    def process_audio(self, input_path: str) -> Tuple[str, list]:
        """
        Полная обработка аудиофайла: конвертация, нормализация и добавление тишины.
        
        Args:
            input_path: Путь к исходному аудиофайлу.
            
        Returns:
            Кортеж: (путь к обработанному файлу, список временных файлов для удаления)
            
        Raises:
            Exception: Если произошла ошибка при обработке аудио.
        """
        temp_files = []
        
        try:
            # Конвертация в WAV
            wav_path = self.convert_to_wav(input_path)
            if wav_path != input_path:  # Если был создан временный файл
                temp_files.append(wav_path)
            
            # Нормализация
            normalized_path = self.normalize_audio(wav_path)
            temp_files.append(normalized_path)
            
            # Добавление тишины
            silence_path = self.add_silence(normalized_path)
            temp_files.append(silence_path)
            
            return silence_path, temp_files
        
        except Exception as e:
            logger.error(f"Ошибка при обработке аудио {input_path}: {e}")
            temp_file_manager.cleanup_temp_files(temp_files)
            raise
```

## 5. Улучшения производительности

### Новый файл app/cache.py
```python
"""
Модуль cache.py содержит функции для кэширования данных.
"""

import time
from typing import Dict, Any, Optional, Callable
from functools import wraps
from .utils import logger


class SimpleCache:
    """
    Простой кэш на основе словаря с поддержкой TTL (Time To Live).
    
    Attributes:
        cache (Dict): Словарь для хранения кэшированных данных.
        ttl (int): Время жизни кэша в секундах.
    """
    
    def __init__(self, ttl: int = 300):
        """
        Инициализация кэша.
        
        Args:
            ttl: Время жизни кэша в секундах (по умолчанию 5 минут).
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
    
    def get(self, key: str) -> Optional[Any]:
        """
        Получение значения из кэша.
        
        Args:
            key: Ключ для получения значения.
            
        Returns:
            Кэшированное значение или None, если ключ не найден или срок действия истек.
        """
        if key in self.cache:
            item = self.cache[key]
            if time.time() - item["timestamp"] < self.ttl:
                logger.debug(f"Кэш hit для ключа: {key}")
                return item["value"]
            else:
                # Удаление просроченного элемента
                del self.cache[key]
                logger.debug(f"Кэш expired для ключа: {key}")
        
        logger.debug(f"Кэш miss для ключа: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """
        Установка значения в кэш.
        
        Args:
            key: Ключ для хранения значения.
            value: Значение для кэширования.
        """
        self.cache[key] = {
            "value": value,
            "timestamp": time.time()
        }
        logger.debug(f"Значение кэшировано для ключа: {key}")
    
    def clear(self) -> None:
        """
        Очистка кэша.
        """
        self.cache.clear()
        logger.debug("Кэш очищен")
    
    def delete(self, key: str) -> bool:
        """
        Удаление значения из кэша.
        
        Args:
            key: Ключ для удаления.
            
        Returns:
            True, если ключ был удален, иначе False.
        """
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Значение удалено из кэша для ключа: {key}")
            return True
        return False


# Глобальные экземпляры кэша
model_cache = SimpleCache(ttl=3600)  # Кэш для метаданных модели (1 час)
config_cache = SimpleCache(ttl=300)   # Кэш для конфигурации (5 минут)


def cache_result(cache_instance: SimpleCache, key_prefix: str = ""):
    """
    Декоратор для кэширования результатов функции.
    
    Args:
        cache_instance: Экземпляр кэша.
        key_prefix: Префикс для ключа кэша.
        
    Returns:
        Декорированная функция.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Генерация ключа кэша на основе имени функции и аргументов
            cache_key = f"{key_prefix}{func.__name__}_{str(args)}_{str(kwargs)}"
            
            # Попытка получить результат из кэша
            cached_result = cache_instance.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Если результат не в кэше, вызываем функцию
            result = func(*args, **kwargs)
            
            # Сохраняем результат в кэш
            cache_instance.set(cache_key, result)
            
            return result
        return wrapper
    return decorator
```

### Новый файл app/async_tasks.py
```python
"""
Модуль async_tasks.py содержит функции для асинхронной обработки задач.
"""

import uuid
import time
from typing import Dict, Any, Callable, Optional
from threading import Thread
from .utils import logger


class AsyncTaskManager:
    """
    Менеджер асинхронных задач на основе потоков.
    
    Attributes:
        tasks (Dict): Словарь для хранения информации о задачах.
    """
    
    def __init__(self):
        """
        Инициализация менеджера асинхронных задач.
        """
        self.tasks: Dict[str, Dict[str, Any]] = {}
    
    def run_task(self, func: Callable, *args, **kwargs) -> str:
        """
        Запуск задачи в отдельном потоке.
        
        Args:
            func: Функция для выполнения.
            *args: Позиционные аргументы для функции.
            **kwargs: Именованные аргументы для функции.
            
        Returns:
            ID задачи.
        """
        task_id = str(uuid.uuid4())
        
        # Создание информации о задаче
        self.tasks[task_id] = {
            "status": "pending",
            "result": None,
            "error": None,
            "created_at": time.time(),
            "started_at": None,
            "completed_at": None
        }
        
        # Создание и запуск потока
        thread = Thread(target=self._run_task_thread, args=(task_id, func, args, kwargs))
        thread.daemon = True
        thread.start()
        
        return task_id
    
    def _run_task_thread(self, task_id: str, func: Callable, args: tuple, kwargs: dict) -> None:
        """
        Функция для выполнения задачи в потоке.
        
        Args:
            task_id: ID задачи.
            func: Функция для выполнения.
            args: Позиционные аргументы для функции.
            kwargs: Именованные аргументы для функции.
        """
        try:
            # Обновление статуса задачи
            self.tasks[task_id]["status"] = "running"
            self.tasks[task_id]["started_at"] = time.time()
            
            # Выполнение функции
            result = func(*args, **kwargs)
            
            # Сохранение результата
            self.tasks[task_id]["status"] = "completed"
            self.tasks[task_id]["result"] = result
            self.tasks[task_id]["completed_at"] = time.time()
            
            logger.info(f"Задача {task_id} завершена успешно")
        except Exception as e:
            # Обработка ошибки
            self.tasks[task_id]["status"] = "failed"
            self.tasks[task_id]["error"] = str(e)
            self.tasks[task_id]["completed_at"] = time.time()
            
            logger.error(f"Задача {task_id} завершилась с ошибкой: {e}")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Получение статуса задачи.
        
        Args:
            task_id: ID задачи.
            
        Returns:
            Информация о задаче или None, если задача не найдена.
        """
        return self.tasks.get(task_id)
    
    def cleanup_completed_tasks(self, max_age_seconds: int = 3600) -> None:
        """
        Очистка завершенных задач старше указанного возраста.
        
        Args:
            max_age_seconds: Максимальный возраст задачи в секундах.
        """
        current_time = time.time()
        tasks_to_remove = []
        
        for task_id, task_info in self.tasks.items():
            if (task_info["status"] in ["completed", "failed"] and 
                "completed_at" in task_info and 
                current_time - task_info["completed_at"] > max_age_seconds):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
            logger.debug(f"Задача {task_id} удалена из-за устаревания")


# Глобальный экземпляр менеджера асинхронных задач
task_manager = AsyncTaskManager()


def transcribe_audio_async(file_path: str, transcriber) -> str:
    """
    Асинхронная транскрибация аудиофайла.
    
    Args:
        file_path: Путь к аудиофайлу.
        transcriber: Экземпляр транскрайбера.
        
    Returns:
        ID задачи.
    """
    return task_manager.run_task(transcriber.process_file, file_path)
```

## 6. Обновления существующих файлов

### Обновленный app/__init__.py
```python
import json
import os
from typing import Dict
from flask import Flask
from flask_cors import CORS
import waitress

# Импорт классов и функций из других модулей
from .transcriber import WhisperTranscriber
from .routes import Routes
from .validators import FileValidator
from .file_manager import temp_file_manager
from .utils import logger


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

    def run(self) -> None:
        """
        Запуск сервиса.
        """
        logger.info(f"Запуск сервиса на порту {self.port}")
        
        # Использовать waitress для production-ready сервера
        waitress.serve(self.app, host='0.0.0.0', port=self.port)
    
    def cleanup(self) -> None:
        """
        Очистка ресурсов перед завершением работы.
        """
        logger.info("Очистка ресурсов перед завершением работы")
        temp_file_manager.cleanup_all()
```

### Обновленный app/routes.py
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
from .utils import logger


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
        def openai_transcribe_endpoint():
            """Эндпоинт для транскрибации аудиофайла (multipart-форма)."""
            source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, request.form, self.file_validator)
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

            source = URLSource(url, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, params, self.file_validator)
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

            source = Base64Source(base64_data, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, params, self.file_validator)
            return jsonify(response), status_code

        @self.app.route('/v1/audio/transcriptions/multipart', methods=['POST'])
        def transcribe_multipart():
            """Эндпоинт для транскрибации аудиофайла, загруженного через форму."""
            source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
            response, status_code = self.transcription_service.transcribe_from_source(source, request.form, self.file_validator)
            return jsonify(response), status_code
        
        @self.app.route('/v1/audio/transcriptions/async', methods=['POST'])
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

### Обновленный app/transcriber_service.py
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
            return {"error": error}, 400

        if not file:
            return {"error": "Failed to get audio file"}, 400
        
        # Валидация файла, если предоставлен валидатор
        if file_validator:
            try:
                file_validator.validate_file(file, filename)
            except ValidationError as e:
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

## 7. Обновление зависимостей

### Обновленный requirements.txt
```txt
# Conda environment dependencies

# Tourch
torch @ https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp313-cp313-manylinux_2_28_x86_64.whl ; platform_system == 'Linux' and platform_machine == 'x86_64' and python_version == '3.13'
torch @ https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64.whl ; platform_system == 'Linux' and platform_machine == 'x86_64' and python_version == '3.12'
torch @ https://download.pytorch.org/whl/cu128/torch-2.7.0%2Bcu128-cp311-cp311-manylinux_2_28_x86_64.whl ; platform_system == 'Linux' and platform_machine == 'x86_64' and python_version == '3.11'

# Linux FA2 from https://github.com/Dao-AILab/flash-attention/releases
flash_attn @ https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp313-cp313-linux_x86_64.whl ; platform_system == 'Linux' and platform_machine == 'x86_64' and python_version == '3.13'
flash_attn @ https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp312-cp312-linux_x86_64.whl ; platform_system == 'Linux' and platform_machine == 'x86_64' and python_version == '3.12'
flash_attn @ https://github.com/kingbri1/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu128torch2.7.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl ; platform_system == 'Linux' and platform_machine == 'x86_64' and python_version == '3.11'

Flask==3.1.0
flask-cors==4.0.0
waitress==3.0.0
librosa==0.10.1
transformers==4.49.0
accelerate==1.4.0
python-magic==0.4.27
flask-limiter==3.5.0
```

## 8. Инструкции по реализации

1. Создайте новые файлы согласно приведенному выше коду:
   - app/file_manager.py
   - app/context_managers.py
   - app/validators.py
   - app/cache.py
   - app/async_tasks.py

2. Обновите существующие файлы:
   - app/__init__.py
   - app/routes.py
   - app/transcriber.py
   - app/transcriber_service.py
   - app/audio_processor.py

3. Обновите config.json, добавив секцию file_validation.

4. Обновите requirements.txt, добавив новые зависимости.

5. Установите python-magic для вашей операционной системы:
   - Ubuntu/Debian: `sudo apt-get install libmagic1`
   - macOS: `brew install libmagic`
   - Windows: Скачайте binaries с официального сайта и добавьте в PATH

6. Перезапустите сервис для применения изменений.

Эти улучшения повысят безопасность, производительность и поддерживаемость вашего сервиса распознавания речи.
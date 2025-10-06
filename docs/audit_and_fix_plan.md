# Whisper API Server - План аудита и исправлений

## Краткое резюме

Этот документ представляет собой комплексный план аудита и исправлений для проекта Whisper API Server с фокусом на избыточность кода, проблемы поддержки и конкретные предупреждения в логах. План структурирован по приоритетам и включает детальные шаги для реализации каждого исправления.

## Проблемы, выявленные в процессе аудита

### 1. Предупреждения в логах (Высокий приоритет)

#### 1.1 Устаревший параметр `inputs` в Whisper
**Проблема**: 
```
/home/serge/.miniconda/envs/whisper-api/lib/python3.11/site-packages/transformers/models/whisper/generation_whisper.py:573: FutureWarning: The input name `inputs` is deprecated. Please make sure to use `input_features` instead.
```

**Причина**: Использование устаревшего API в методе `transcribe()` класса `WhisperTranscriber`.

**Решение**: Обновить метод для использования `input_features` вместо `inputs`.

#### 1.2 Конфликт между `language` и `forced_decoder_ids`
**Проблема**:
```
You have passed language=ru, but also have set `forced_decoder_ids` to [[1, None], [2, 50360]] which creates a conflict. `forced_decoder_ids` will be ignored in favor of language=ru.
```

**Причина**: Одновременное использование параметров `language` и `forced_decoder_ids` в методе `transcribe()`.

**Решение**: Убрать `forced_decoder_ids` при использовании параметра `language`.

#### 1.3 Отсутствие attention mask
**Проблема**:
```
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
```

**Причина**: Не передается attention mask при обработке аудио в transformers.

**Решение**: Добавить генерацию и передачу attention mask в методе `transcribe()`.

#### 1.4 Устаревший метод в librosa
**Проблема**:
```
/home/serge/.miniconda/envs/whisper-api/lib/python3.11/site-packages/librosa/core/audio.py:184: FutureWarning: librosa.core.audio.__audioread_load
	Deprecated as of librosa version 0.10.0.
	It will be removed in librosa version 1.0.
```

**Причина**: Использование устаревшего метода загрузки аудио в librosa.

**Решение**: Обновить метод загрузки аудио для использования нового API.

### 2. Избыточность кода (Средний приоритет)

#### 2.1 Дублирующиеся эндпоинты API
**Проблема**: Эндпоинты `/v1/audio/transcriptions` и `/v1/audio/transcriptions/multipart` имеют идентичные реализации.

**Решение**: Создать общую функцию и сделать оба эндпоинта алиасами к ней, чтобы сохранить совместимость с разными клиентами.

#### 2.2 Дублирование загрузки аудио
**Проблема**: Функции загрузки аудио дублируются в `WhisperTranscriber._load_audio()` и `TranscriptionService.get_audio_duration()`.

**Решение**: Создать общий утилитарный класс для загрузки аудио.

#### 2.3 Управление временными файлами
**Проблема**: Множественные классы используют похожие паттерны создания и очистки временных файлов.

**Решение**: Создать централизованный класс `TempFileManager`.

## План реализации

### Этап 1: Исправление предупреждений в логах (Высокий приоритет)

#### 1.1 Обновление метода transcribe() в WhisperTranscriber

**Файл**: `app/transcriber.py`

**Изменения**:
1. Заменить `inputs` на `input_features` в методе `transcribe()`
2. Убрать конфликтующие `forced_decoder_ids` при использовании `language`
3. Добавить генерацию attention_mask

```python
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
    
    # Подготовка входных данных для модели
    inputs = self.processor(
        audio_array, 
        sampling_rate=sampling_rate, 
        return_tensors="pt"
    ).to(self.device)
    
    # Генерация attention mask
    attention_mask = inputs.attention_mask
    
    # Транскрибация с корректным форматом данных
    result = self.asr_pipeline(
        inputs.input_features,  # Используем input_features вместо inputs
        attention_mask=attention_mask,  # Добавляем attention_mask
        generate_kwargs={
            "language": self.language, 
            "max_new_tokens": self.max_new_tokens, 
            "temperature": self.temperature
            # Убираем forced_decoder_ids при использовании language
        },
        return_timestamps=self.return_timestamps
    )
    
    # Остальная часть метода без изменений
    # ...
```

#### 1.2 Обновление метода _load_audio()

**Файл**: `app/transcriber.py`

**Изменения**: Обновить метод для использования нового API librosa:

```python
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
        # Используем новый API librosa
        audio_array, sampling_rate = librosa.load(
            file_path, 
            sr=16000,
            mono=True  # Явно указываем моно
        )
        return audio_array, sampling_rate
    except Exception as e:
        logger.error(f"Ошибка при загрузке аудио {file_path}: {e}")
        raise
```

#### 1.3 Обновление метода get_audio_duration()

**Файл**: `app/transcriber_service.py`

**Изменения**: Обновить метод для использования нового API librosa:

```python
def get_audio_duration(self, file_path: str) -> float:
    """
    Определяет длительность аудиофайла в секундах.

    Args:
        file_path: Путь к аудиофайлу.

    Returns:
        Длительность в секундах.
    """
    try:
        # Используем новый API librosa
        y, sr = librosa.load(
            file_path, 
            sr=None,
            mono=True  # Явно указываем моно
        )
        duration = librosa.get_duration(y=y, sr=sr)
        return duration
    except Exception as e:
        logger.error(f"Ошибка при определении длительности файла: {e}")
        return 0.0
```

### Этап 2: Устранение избыточности кода (Средний приоритет)

#### 2.1 Создание утилитарного класса AudioUtils

**Новый файл**: `app/audio_utils.py`

```python
"""
Модуль audio_utils.py содержит утилитарные функции для работы с аудио.
"""

import librosa
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger('app.audio_utils')


class AudioUtils:
    """Утилитарный класс для работы с аудио."""
    
    @staticmethod
    def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Загрузка аудиофайла с использованием librosa.

        Args:
            file_path: Путь к аудиофайлу.
            sr: Целевая частота дискретизации.

        Returns:
            Кортеж (массив numpy, частота дискретизации).
            
        Raises:
            Exception: Если не удалось загрузить аудиофайл.
        """
        try:
            # Используем новый API librosa
            audio_array, sampling_rate = librosa.load(
                file_path, 
                sr=sr,
                mono=True  # Явно указываем моно
            )
            return audio_array, sampling_rate
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио {file_path}: {e}")
            raise
    
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """
        Определяет длительность аудиофайла в секундах.

        Args:
            file_path: Путь к аудиофайлу.

        Returns:
            Длительность в секундах.
        """
        try:
            # Используем новый API librosa
            y, sr = librosa.load(
                file_path, 
                sr=None,
                mono=True  # Явно указываем моно
            )
            duration = librosa.get_duration(y=y, sr=sr)
            return duration
        except Exception as e:
            logger.error(f"Ошибка при определении длительности файла: {e}")
            return 0.0
```

#### 2.2 Обновление WhisperTranscriber для использования AudioUtils

**Файл**: `app/transcriber.py`

**Изменения**:
1. Добавить импорт `from .audio_utils import AudioUtils`
2. Заменить метод `_load_audio()` на вызов `AudioUtils.load_audio()`

```python
from .audio_utils import AudioUtils

# Удалить метод _load_audio() и заменить его использование на AudioUtils.load_audio()

def transcribe(self, audio_path: str) -> Union[str, Dict]:
    # ...
    # Загрузка аудио в формате numpy array
    audio_array, sampling_rate = AudioUtils.load_audio(audio_path, sr=16000)
    # ...
```

#### 2.3 Обновление TranscriptionService для использования AudioUtils

**Файл**: `app/transcriber_service.py`

**Изменения**:
1. Добавить импорт `from .audio_utils import AudioUtils`
2. Заменить метод `get_audio_duration()` на вызов `AudioUtils.get_audio_duration()`

```python
from .audio_utils import AudioUtils

# Удалить метод get_audio_duration() и заменить его использование на AudioUtils.get_audio_duration()

def transcribe_from_source(self, source: AudioSource, params: Dict = None, file_validator: FileValidator = None) -> Tuple[Dict, int]:
    # ...
    # Определяем длительность аудиофайла
    duration = AudioUtils.get_audio_duration(temp_file_path)
    # ...
```

#### 2.4 Оптимизация дублирующихся эндпоинтов

**Файл**: `app/routes.py`

**Изменения**: Создать общую функцию и сделать оба эндпоинта алиасами к ней:

```python
def _handle_transcription_request():
    """Общая функция для обработки запросов транскрибации."""
    source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
    response, status_code = self.transcription_service.transcribe_from_source(source, request.form, self.file_validator)
    return jsonify(response), status_code

@app.route('/v1/audio/transcriptions', methods=['POST'])
@log_invalid_file_request
def openai_transcribe_endpoint():
    """Эндпоинт для транскрибации аудиофайла (multipart-форма)."""
    return _handle_transcription_request()

@app.route('/v1/audio/transcriptions/multipart', methods=['POST'])
@log_invalid_file_request
def transcribe_multipart():
    """Эндпоинт для транскрибации аудиофайла, загруженного через форму."""
    return _handle_transcription_request()
```

### Этап 3: Улучшение управления временными файлами (Низкий приоритет)

#### 3.1 Создание централизованного TempFileManager

**Файл**: `app/file_manager.py` (обновление существующего)

```python
"""
Модуль file_manager.py содержит классы для управления файлами.
"""

import os
import uuid
import tempfile
import shutil
import logging
from typing import Tuple, List

logger = logging.getLogger('app.file_manager')


class TempFileManager:
    """Централизованный менеджер временных файлов."""
    
    def __init__(self):
        self.temp_files = []
    
    def create_temp_file(self, suffix: str = ".wav") -> Tuple[str, str]:
        """
        Создает временный файл.
        
        Args:
            suffix: Расширение файла.
            
        Returns:
            Кортеж (путь к файлу, путь к временной директории).
        """
        temp_dir = tempfile.mkdtemp()
        temp_file = os.path.join(temp_dir, f"{uuid.uuid4()}{suffix}")
        self.temp_files.append(temp_file)
        return temp_file, temp_dir
    
    def cleanup_temp_files(self, file_paths: List[str] = None) -> None:
        """
        Очищает временные файлы.
        
        Args:
            file_paths: Список путей к файлам для очистки. Если None, очищает все созданные файлы.
        """
        files_to_cleanup = file_paths or self.temp_files
        
        for path in files_to_cleanup:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    temp_dir = os.path.dirname(path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                        logger.debug(f"Удалена временная директория: {temp_dir}")
                    logger.debug(f"Удален временный файл: {path}")
            except Exception as e:
                logger.warning(f"Не удалось очистить временный файл {path}: {e}")
    
    def temp_file(self, suffix: str = ".wav"):
        """
        Контекстный менеджер для временного файла.
        
        Args:
            suffix: Расширение файла.
            
        Returns:
            Контекстный менеджер.
        """
        class TempFileContext:
            def __init__(self, manager, suffix):
                self.manager = manager
                self.suffix = suffix
                self.temp_path = None
            
            def __enter__(self):
                self.temp_path, _ = self.manager.create_temp_file(self.suffix)
                return self.temp_path
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                if self.temp_path:
                    self.manager.cleanup_temp_files([self.temp_path])
        
        return TempFileContext(self, suffix)


# Глобальный экземпляр менеджера временных файлов
temp_file_manager = TempFileManager()
```

#### 3.2 Обновление AudioProcessor для использования TempFileManager

**Файл**: `app/audio_processor.py`

**Изменения**: Обновить методы для использования централизованного TempFileManager.

### Этап 4: Обновление зависимостей (Средний приоритет)

#### 4.1 Обновление requirements.txt

**Файл**: `requirements.txt`

**Изменения**: Обновить версии зависимостей:

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
librosa==0.11.0  # Обновлено с 0.10.1 до 0.11.0
transformers==4.49.0
accelerate==1.4.0
python-magic==0.4.27
flask-limiter==3.5.0
```

## Приоритеты реализации

### Высокий приоритет (Критические проблемы)
1. Исправление предупреждений в логах:
   - Обновление метода `transcribe()` для использования `input_features`
   - Устранение конфликта между `language` и `forced_decoder_ids`
   - Добавление attention_mask
   - Обновление методов загрузки аудио

### Средний приоритет (Улучшение поддерживаемости)
1. Устранение избыточности кода:
   - Создание утилитарного класса `AudioUtils`
   - Обновление классов для использования `AudioUtils`
   - Оптимизация дублирующихся эндпоинтов
2. Обновление зависимостей для повышения безопасности

### Низкий приоритет (Оптимизация)
1. Улучшение управления временными файлами:
   - Создание централизованного `TempFileManager`
   - Обновление классов для использования `TempFileManager`

## Инструкции по реализации

1. Создайте новый файл `app/audio_utils.py` с классом `AudioUtils`.
2. Обновите методы в `app/transcriber.py`:
   - Измените метод `transcribe()` для использования `input_features` и `attention_mask`
   - Уберите `forced_decoder_ids` при использовании `language`
   - Замените `_load_audio()` на `AudioUtils.load_audio()`
3. Обновите `app/transcriber_service.py`:
   - Замените `get_audio_duration()` на `AudioUtils.get_audio_duration()`
4. Обновите `app/routes.py`:
   - Создайте общую функцию `_handle_transcription_request()`
   - Сделайте оба эндпоинта (`/v1/audio/transcriptions` и `/v1/audio/transcriptions/multipart`) алиасами к этой функции
5. Обновите `requirements.txt` с новыми версиями зависимостей.
6. Перезапустите сервис для применения изменений.

Эти изменения устранят предупреждения в логах, уменьшат избыточность кода и улучшат поддерживаемость проекта.
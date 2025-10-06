# План добавления логирования для обращений к API с невалидными типами файлов

## Описание задачи

Необходимо добавить логирование для всех обращений к API, когда пользователь пытается загрузить файл с невалидным типом или расширением. Это поможет отслеживать попытки использования неподдерживаемых форматов файлов и анализировать частоту таких ошибок.

## Места для добавления логирования

### 1. app/routes.py

В следующих эндпоинтах необходимо добавить логирование при обнаружении невалидных типов файлов:

1. `openai_transcribe_endpoint()` - `/v1/audio/transcriptions`
2. `transcribe_from_url()` - `/v1/audio/transcriptions/url`
3. `transcribe_from_base64()` - `/v1/audio/transcriptions/base64`
4. `transcribe_multipart()` - `/v1/audio/transcriptions/multipart`
5. `transcribe_async()` - `/v1/audio/transcriptions/async`

### 2. app/validators.py

В методах класса `FileValidator` необходимо добавить логирование:

1. `_validate_file_extension()` - при обнаружении неразрешенного расширения
2. `_validate_file_mime_type()` - при обнаружении неразрешенного MIME-типа
3. `validate_file()` - при общей ошибке валидации

## Реализация

### 1. Обновление app/validators.py

```python
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
```

### 2. Обновление app/routes.py

В каждом эндпоинте, который обрабатывает файлы, необходимо добавить логирование при ошибках валидации:

```python
@app.route('/v1/audio/transcriptions', methods=['POST'])
def openai_transcribe_endpoint():
    """Эндпоинт для транскрибации аудиофайла (multipart-форма)."""
    source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
    
    # Получаем файл для логирования
    file, filename, error = source.get_audio_file()
    
    try:
        response, status_code = self.transcription_service.transcribe_from_source(source, request.form, self.file_validator)
        return jsonify(response), status_code
    except ValidationError as e:
        # Логирование обращения к API с невалидным типом файла
        client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
        logger.warning(f"Обращение к эндпоинту /v1/audio/transcriptions с невалидным файлом '{filename}' "
                      f"от клиента {client_ip}. Ошибка: {str(e)}")
        return jsonify({"error": str(e)}), 400
```

Аналогичные изменения необходимо внести в другие эндпоинты.

### 3. Создание декоратора для логирования

Для уменьшения дублирования кода можно создать декоратор в app/utils.py:

```python
def log_invalid_file_request(func):
    """
    Декоратор для логирования запросов с невалидными файлами.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValidationError as e:
            # Получение информации о запросе
            from flask import request
            client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.environ.get('REMOTE_ADDR', 'unknown'))
            endpoint = request.endpoint
            method = request.method
            
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

Затем применить этот декоратор к соответствующим эндпоинтам:

```python
@app.route('/v1/audio/transcriptions', methods=['POST'])
@log_invalid_file_request
def openai_transcribe_endpoint():
    """Эндпоинт для транскрибации аудиофайла (multipart-форма)."""
    source = UploadedFileSource(request.files, self.config.get("file_validation", {}).get("max_file_size_mb", 100))
    response, status_code = self.transcription_service.transcribe_from_source(source, request.form, self.file_validator)
    return jsonify(response), status_code
```

## Преимущества предложенного решения

1. **Полное покрытие**: Логирование добавляется на всех уровнях, где происходит валидация файлов.
2. **Информативность**: В логи включается IP-адрес клиента, имя файла, тип ошибки и эндпоинт.
3. **Уменьшение дублирования**: Использование декоратора позволяет избежать повторения кода.
4. **Анализируемость**: Структурированные логи позволят легко анализировать частоту ошибок и типы невалидных файлов.

## Формат логов

Пример сообщения в логе:
```
WARNING: Обращение к эндпоинту POST /v1/audio/transcriptions с невалидным файлом 'document.pdf' от клиента 192.168.1.100. Ошибка: Расширение файла не разрешено. Разрешенные расширения: .wav, .mp3, .ogg, .flac, .m4a
```

Это позволит легко фильтровать и анализировать запросы с невалидными файлами.
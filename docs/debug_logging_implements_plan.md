# План реализации отладочного логирования запросов

## Требования
Добавить отладочный режим логирования, который:
- Включается через конфиг `"log_debug": true/false` в секции `request_logging`
- Логирует все данные запроса в оригинальном виде без обработки
- Использует `json.dumps` для преобразования данных в строку JSON
- По умолчанию отключен для безопасности
- Логирует как запросы, так и ответы в полном объеме

## Текущее состояние
- ✅ Базовая система логирования запросов реализована
- ✅ Конфигурация `log_debug` добавлена в `config.json`
- ✅ Middleware логирования работает для всех эндпоинтов
- ❌ Отладочный режим еще не реализован

## Архитектура решения

### 1. Модификация RequestLogger
**Файл:** `app/request_logger.py`

#### Добавление параметра debug в `_extract_request_info()`
```python
def _extract_request_info(self, debug: bool = False) -> Dict[str, Any]:
    # В отладочном режиме не фильтруем заголовки
    if debug:
        # Включаем все заголовки без фильтрации
        headers = dict(request.headers)
    else:
        # Стандартная фильтрация чувствительных заголовков
        headers = {key: value for key, value in request.headers 
                  if key.lower() not in self.sensitive_headers}
```

#### Новые методы для отладочного режима
```python
def _log_debug_request(self, request_info: Dict[str, Any]):
    """Логирование полных данных запроса в отладочном режиме."""
    debug_data = {
        "timestamp": time.time(),
        "type": "request",
        "data": request_info
    }
    self.logger.info(
        "DEBUG REQUEST: %s", 
        json.dumps(debug_data, ensure_ascii=False, default=str)
    )

def _log_debug_response(self, response, processing_time: float):
    """Логирование полных данных ответа в отладочном режиме."""
    response_info = {
        "status_code": response.status_code,
        "headers": dict(response.headers),
        "content_length": response.content_length,
        "processing_time": round(processing_time, 3)
    }
    debug_data = {
        "timestamp": time.time(),
        "type": "response", 
        "data": response_info
    }
    self.logger.info(
        "DEBUG RESPONSE: %s",
        json.dumps(debug_data, ensure_ascii=False, default=str)
    )
```

#### Обновление основных методов
```python
def _before_request(self):
    if not self._should_log_request():
        return
    
    g.start_time = time.time()
    request_info = self._extract_request_info(debug=self.config.get('log_debug', False))
    
    if self.config.get('log_debug', False):
        self._log_debug_request(request_info)
    else:
        message = self._format_request_message(request_info)
        self.logger.info(message, extra={"type": "request"})

def _after_request(self, response):
    if not self._should_log_request():
        return response
    
    processing_time = time.time() - getattr(g, 'start_time', time.time())
    
    if self.config.get('log_debug', False):
        self._log_debug_response(response, processing_time)
    else:
        message = self._format_response_message(response, processing_time)
        self.logger.info(message, extra={"type": "response"})
    
    return response
```

### 2. Особенности отладочного режима

#### Данные запроса включают:
- Все заголовки (включая чувствительные)
- Полные значения параметров (query, form, JSON)
- Метаданные файлов (имя, тип, размер)
- IP клиента и User Agent
- Метод и путь запроса

#### Данные ответа включают:
- Статус код
- Все заголовки ответа
- Размер контента
- Время обработки

### 3. Примеры вывода

**Обычный режим:**
```
POST /v1/audio/transcriptions от 192.168.1.100 (curl/7.68.0) файлы: audio.wav (1048576 байт) параметры: language, return_timestamps
200 за 1.051 сек, 245 байт
```

**Отладочный режим:**
```
DEBUG REQUEST: {"timestamp": 1696626701.619, "type": "request", "data": {"method": "POST", "path": "/v1/audio/transcriptions", "client_ip": "192.168.1.100", "user_agent": "curl/7.68.0", "query_params": {}, "form_data": {"language": "ru", "return_timestamps": "true"}, "files": {"file": {"filename": "audio.wav", "content_type": "audio/wav", "content_length": 1048576}}, "headers": {"Authorization": "Bearer xxx", "Cookie": "session=yyy", "Content-Type": "multipart/form-data", "Accept": "*/*"}}}
DEBUG RESPONSE: {"timestamp": 1696626702.670, "type": "response", "data": {"status_code": 200, "headers": {"Content-Type": "application/json", "Content-Length": "245"}, "content_length": 245, "processing_time": 1.051}}
```

### 4. Безопасность
- Отладочный режим по умолчанию отключен (`log_debug: false`)
- Четкое разделение между обычным и отладочным режимами
- Рекомендация использовать только для отладки в development среде

### 5. Преимущества
- **Полнота данных**: Все исходные данные запроса доступны для анализа
- **Простота использования**: Включение/выключение через конфиг
- **Совместимость**: Не нарушает работу существующего логирования
- **Гибкость**: Можно быстро включить для диагностики проблем

## Файлы для изменения
- `app/request_logger.py` - основная реализация отладочного режима
- `config.json` - уже обновлен с параметром `log_debug`

## Тестирование
1. Включить `log_debug: true` в конфигурации
2. Отправить тестовые запросы к различным эндпоинтам
3. Проверить, что в логах появляются полные JSON данные запросов и ответов
4. Убедиться, что обычный режим логирования продолжает работать при `log_debug: false`

## Следующие шаги
Перейти к реализации в режиме Code.
# План исправления проблем с логированием

## Проблема

Лог обрывается после ошибки транскрибации, и сообщение об ошибке пустое:
```
2025-10-06 14:32:21,263 - app.utils - ERROR - Ошибка при транскрибации: 
```

## Причины

1. Метод `transcribe()` в `WhisperTranscriber` не обрабатывает исключения, которые могут возникнуть при вызове `self.asr_pipeline()`
2. В методе `process_file()` исключение не правильно обрабатывается, что приводит к пустому сообщению об ошибке
3. Отсутствует детальная информация об исключении в логах

## Решение

### 1. Улучшение обработки исключений в WhisperTranscriber.transcribe()

Нужно добавить try-except блок вокруг вызова `self.asr_pipeline()`:

```python
def transcribe(self, audio_path: str) -> Union[str, Dict]:
    logger.info(f"Начало транскрибации файла: {audio_path}")
    
    try:
        # Загрузка аудио в формате numpy array
        audio_array, sampling_rate = AudioUtils.load_audio(audio_path, sr=16000)
        
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
            },
            return_timestamps=self.return_timestamps
        )
        
        # Остальная часть метода без изменений
        # ...
        
    except Exception as e:
        logger.error(f"Ошибка в процессе транскрибации аудиофайла '{audio_path}': {str(e)}")
        logger.error(f"Тип исключения: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
```

### 2. Улучшение обработки исключений в WhisperTranscriber.process_file()

Нужно добавить детальное логирование исключений:

```python
def process_file(self, input_path: str) -> Union[str, Dict]:
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
        
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Ошибка при обработке файла '{input_path}' через {elapsed_time:.2f} секунд: {str(e)}")
        logger.error(f"Тип исключения: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
        
    finally:
        # Очистка временных файлов
        temp_file_manager.cleanup_temp_files(temp_files)
```

### 3. Улучшение обработки исключений в TranscriptionService.transcribe_from_source()

Нужно добавить детальное логирование исключений:

```python
try:
    start_time = time.time()
    result = self.transcriber.process_file(temp_file_path)
    processing_time = time.time() - start_time
    
    # Формируем ответ в зависимости от return_timestamps
    # ...
    
    return response, 200

except Exception as e:
    logger.error(f"Ошибка при транскрибации файла '{filename}': {str(e)}")
    logger.error(f"Тип исключения: {type(e).__name__}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    return {"error": str(e)}, 500

finally:
    # Восстанавливаем оригинальное значение return_timestamps
    self.transcriber.return_timestamps = original_return_timestamps
```

## Приоритет

Высокий - это критично для отладки и диагностики проблем в сервисе.

## Ожидаемый результат

1. Детальные логи ошибок с информацией о типе исключения и traceback
2. Корректная обработка исключений без обрыва логов
3. Улучшенная диагностика проблем при транскрибации
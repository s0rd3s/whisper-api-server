# Инструкции по реализации исправлений логирования

## Обзор

Этот документ содержит детальные инструкции по реализации исправлений для улучшения логирования в Whisper API Server.

## Шаги реализации

### Шаг 1: Обновление WhisperTranscriber.transcribe()

**Файл**: `app/transcriber.py`

**Задача**: Добавить try-except блок вокруг вызова `self.asr_pipeline()` для детального логирования ошибок.

**Изменения**:
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
        
    except Exception as e:
        logger.error(f"Ошибка в процессе транскрибации аудиофайла '{audio_path}': {str(e)}")
        logger.error(f"Тип исключения: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
```

### Шаг 2: Обновление WhisperTranscriber.process_file()

**Файл**: `app/transcriber.py`

**Задача**: Добавить детальное логирование исключений.

**Изменения**:
```python
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

### Шаг 3: Обновление TranscriptionService.transcribe_from_source()

**Файл**: `app/transcriber_service.py`

**Задача**: Добавить детальное логирование исключений.

**Изменения**:
```python
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
    logger.error(f"Ошибка при транскрибации файла '{filename}': {str(e)}")
    logger.error(f"Тип исключения: {type(e).__name__}")
    import traceback
    logger.error(f"Traceback: {traceback.format_exc()}")
    return {"error": str(e)}, 500

finally:
    # Восстанавливаем оригинальное значение return_timestamps
    self.transcriber.return_timestamps = original_return_timestamps
```

## Важные замечания

1. **Импорт traceback**: Не забудьте добавить `import traceback` в начало файлов, где он используется.
2. **Детальное логирование**: Новые логи будут содержать детальную информацию об исключениях, включая тип и полный traceback.
3. **Сохранение исключений**: Мы сохраняем исходное исключение с помощью `raise`, чтобы не нарушать существующую логику обработки ошибок.

## Тестирование

После внесения изменений:

1. Перезапустите сервис
2. Отправьте тестовый запрос на транскрибацию
3. Проверьте логи на наличие детальной информации об ошибках (если они возникнут)
4. Убедитесь, что логи не обрываются при возникновении исключений

## Ожидаемый результат

При возникновении ошибки в процессе транскрибации логи будут содержать:
- Сообщение об ошибке
- Тип исключения
- Полный traceback для отладки

Это значительно упростит диагностику проблем в работе сервиса.
# Исправление проблемы с форматом данных в AutomaticSpeechRecognitionPipeline

## Проблема

При попытке передать данные в pipeline возникает ошибка:
```
ValueError: When passing a dictionary to AutomaticSpeechRecognitionPipeline, the dict needs to contain a "raw" key containing the numpy array representing the audio and a "sampling_rate" key, containing the sampling_rate associated with that array
```

## Причина

AutomaticSpeechRecognitionPipeline ожидает данные в формате словаря с ключами:
- "raw" - numpy массив аудио
- "sampling_rate" - частота дискретизации

Мы пытались передать словарь с ключом "input_features", что не соответствует ожидаемому формату.

## Решение

Нужно вернуться к исходному формату данных с "raw" и "sampling_rate":

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
        
        # Транскрибация с корректным форматом данных
        result = self.asr_pipeline(
            {"raw": audio_array, "sampling_rate": sampling_rate}, 
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
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
```

## Объяснение

Хотя мы и вернулись к исходному формату данных, мы сохранили все улучшения:
1. Использование AudioUtils для загрузки аудио
2. Детальное логирование ошибок
3. Правильную обработку исключений
4. Устранение конфликта между `language` и `forced_decoder_ids`

## Приоритет

Высокий - это критично для работы сервиса.

## Ожидаемый результат

После внесения изменений:
1. Ошибка `ValueError` будет устранена
2. Сервис будет корректно транскрибировать аудиофайлы
3. Логи будут содержать детальную информацию о процессе транскрибации
4. Предупреждения в логах будут устранены
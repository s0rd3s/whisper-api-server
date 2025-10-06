# Исправление проблемы с attention_mask в WhisperTranscriber

## Проблема

При попытке получить `attention_mask` из `inputs` возникает ошибка:
```
AttributeError: attention_mask
```

## Причина

WhisperProcessor не создает `attention_mask` по умолчанию. Этот атрибут нужно генерировать вручную или использовать альтернативный подход.

## Решение

Есть несколько вариантов решения:

### Вариант 1: Генерация attention_mask вручную

```python
# Подготовка входных данных для модели
inputs = self.processor(
    audio_array, 
    sampling_rate=sampling_rate, 
    return_tensors="pt"
).to(self.device)

# Генерация attention_mask вручную
if hasattr(inputs, 'attention_mask'):
    attention_mask = inputs.attention_mask
else:
    # Создаем attention_mask вручную (все единицы, так как нет паддинга)
    attention_mask = torch.ones(inputs.input_features.shape[:2], dtype=torch.long, device=self.device)

# Транскрибация с корректным форматом данных
result = self.asr_pipeline(
    inputs.input_features,
    attention_mask=attention_mask,
    generate_kwargs={
        "language": self.language, 
        "max_new_tokens": self.max_new_tokens, 
        "temperature": self.temperature
    },
    return_timestamps=self.return_timestamps
)
```

### Вариант 2: Использование формата словаря (рекомендуемый)

```python
# Подготовка входных данных для модели
inputs = self.processor(
    audio_array, 
    sampling_rate=sampling_rate, 
    return_tensors="pt"
).to(self.device)

# Транскрибация с корректным форматом данных
result = self.asr_pipeline(
    {"input_features": inputs.input_features},
    generate_kwargs={
        "language": self.language, 
        "max_new_tokens": self.max_new_tokens, 
        "temperature": self.temperature
    },
    return_timestamps=self.return_timestamps
)
```

### Вариант 3: Возврат к старому формату с улучшенной обработкой

```python
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
```

## Рекомендация

Я рекомендую использовать **Вариант 2**, так как он:
1. Соответствует новому API с использованием `input_features`
2. Не требует ручной генерации `attention_mask`
3. Является наиболее стабильным подходом

## Изменения в WhisperTranscriber.transcribe()

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
        
        # Транскрибация с корректным форматом данных
        result = self.asr_pipeline(
            {"input_features": inputs.input_features},
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

## Приоритет

Высокий - это критично для работы сервиса.

## Ожидаемый результат

После внесения изменений:
1. Ошибка `AttributeError: attention_mask` будет устранена
2. Сервис будет корректно транскрибировать аудиофайлы
3. Логи будут содержать детальную информацию о процессе транскрибации
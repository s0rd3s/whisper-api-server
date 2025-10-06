# Whisper API Server - Архитектура аудита и исправлений

## Обзор архитектуры текущего проекта

```mermaid
graph TB
    subgraph "Текущая архитектура"
        Client[Клиент] --> API[Flask API]
        API --> Routes[Routes]
        Routes --> TS[TranscriptionService]
        TS --> WT[WhisperTranscriber]
        WT --> AP[AudioProcessor]
        AP --> External[FFmpeg/SoX]
        
        Routes --> AS[AudioSources]
        AS --> FM[FileManager]
        AS --> V[Validators]
        
        TS --> HL[HistoryLogger]
        
        WT --> Model[Whisper Model]
        Model --> Transformers[Transformers]
        Transformers --> Librosa[Librosa]
    end
    
    subgraph "Проблемы"
        Problem1[Устаревший API transformers]
        Problem2[Дублирование кода]
        Problem3[Избыточные эндпоинты]
        Problem4[Устаревший librosa]
    end
    
    Problem1 -.-> Transformers
    Problem2 -.-> WT
    Problem2 -.-> TS
    Problem3 -.-> Routes
    Problem4 -.-> Librosa
```

## Целевая архитектура после исправлений

```mermaid
graph TB
    subgraph "Улучшенная архитектура"
        Client[Клиент] --> API[Flask API]
        API --> Routes[Routes]
        Routes --> TS[TranscriptionService]
        TS --> WT[WhisperTranscriber]
        WT --> AU[AudioUtils]
        AU --> Librosa[Librosa v0.11+]
        WT --> AP[AudioProcessor]
        AP --> TFM[TempFileManager]
        AP --> External[FFmpeg/SoX]
        
        Routes --> AS[AudioSources]
        AS --> TFM
        AS --> V[Validators]
        
        TS --> HL[HistoryLogger]
        
        WT --> Model[Whisper Model]
        Model --> Transformers[Transformers v4.49+]
        
        subgraph "Новые компоненты"
            AU
            TFM
        end
    end
    
    subgraph "Устраненные проблемы"
        Fixed1[✓ Обновленный API transformers]
        Fixed2[✓ Устранено дублирование]
        Fixed3[✓ Созданы алиасы для эндпоинтов]
        Fixed4[✓ Обновлен librosa]
    end
```

## План реализации по этапам

```mermaid
gantt
    title План аудита и исправлений Whisper API Server
    dateFormat  YYYY-MM-DD
    section Этап 1: Критические исправления
    Обновление transcribe()      :a1, 2024-01-01, 2d
    Исправление attention mask   :a2, after a1, 1d
    Обновление librosa API       :a3, after a2, 1d
    
    section Этап 2: Устранение избыточности
    Создание AudioUtils          :b1, after a3, 2d
    Рефакторинг WhisperTranscriber :b2, after b1, 2d
    Рефакторинг TranscriptionService :b3, after b2, 2d
    Оптимизация эндпоинтов       :b4, after b3, 1d
    
    section Этап 3: Оптимизация
    Улучшение TempFileManager    :c1, after b4, 2d
    Обновление AudioProcessor    :c2, after c1, 2d
    Обновление зависимостей      :c3, after c2, 1d
    
    section Тестирование
    Тестирование изменений       :d1, after c3, 3d
    Финальное развертывание      :d2, after d1, 1d
```

## Детализация проблем и решений

### 1. Проблемы с transformers и Whisper

```mermaid
graph LR
    subgraph "Текущая реализация"
        A1[transcribe()] --> A2[inputs]
        A2 --> A3[forced_decoder_ids + language]
        A3 --> A4[Нет attention_mask]
    end
    
    subgraph "Проблемы"
        B1[Устаревший параметр inputs]
        B2[Конфликт параметров]
        B3[Отсутствие mask]
    end
    
    subgraph "Исправленная реализация"
        C1[transcribe()] --> C2[input_features]
        C2 --> C3[attention_mask]
        C3 --> C4[language только]
    end
    
    A1 -.-> B1
    A2 -.-> B2
    A3 -.-> B3
    
    C1 --> Fixed1[✓ Решено]
    C2 --> Fixed2[✓ Решено]
    C3 --> Fixed3[✓ Решено]
```

### 2. Дублирование кода загрузки аудио

```mermaid
graph LR
    subgraph "Текущая реализация"
        A1[WhisperTranscriber._load_audio]
        A2[TranscriptionService.get_audio_duration]
        A1 --> A3[librosa.load]
        A2 --> A4[librosa.load]
    end
    
    subgraph "Проблема"
        B1[Дублирование кода]
    end
    
    subgraph "Исправленная реализация"
        C1[AudioUtils.load_audio]
        C2[AudioUtils.get_audio_duration]
        C1 --> C3[librosa.load v0.11+]
        C2 --> C3
    end
    
    A1 -.-> B1
    A2 -.-> B1
    
    C1 --> Fixed[✓ Устранено дублирование]
    C2 --> Fixed
```

### 3. Дублирующиеся эндпоинты API

```mermaid
graph LR
    subgraph "Текущая реализация"
        A1[/v1/audio/transcriptions]
        A2[/v1/audio/transcriptions/multipart]
        A1 --> A3[UploadedFileSource]
        A2 --> A4[UploadedFileSource]
        A3 --> A5[transcribe_from_source]
        A4 --> A6[transcribe_from_source]
    end
    
    subgraph "Проблема"
        B1[Дублирование эндпоинтов]
    end
    
    subgraph "Исправленная реализация"
        C1[_handle_transcription_request]
        C2[/v1/audio/transcriptions]
        C3[/v1/audio/transcriptions/multipart]
        C1 --> C4[UploadedFileSource]
        C4 --> C5[transcribe_from_source]
        C2 --> C1
        C3 --> C1
    end
    
    A1 -.-> B1
    A2 -.-> B1
    
    C1 --> Fixed[✓ Алиасы к общей функции]
```

## Влияние изменений на систему

```mermaid
graph TB
    subgraph "Влияние на производительность"
        P1[Уменьшение предупреждений в логах]
        P2[Более стабильная работа]
        P3[Оптимизированное использование памяти]
    end
    
    subgraph "Влияние на поддерживаемость"
        M1[Уменьшение дублирования кода]
        M2[Централизованное управление файлами]
        M3[Четкое разделение ответственности]
    end
    
    subgraph "Влияние на безопасность"
        S1[Обновленные зависимости]
        S2[Меньше уязвимостей]
        S3[Более предсказуемое поведение]
    end
    
    Changes[Изменения] --> P1
    Changes --> P2
    Changes --> P3
    Changes --> M1
    Changes --> M2
    Changes --> M3
    Changes --> S1
    Changes --> S2
    Changes --> S3
```

## Риски и митигация

```mermaid
graph LR
    subgraph "Потенциальные риски"
        R1[Несовместимость версий]
        R2[Изменение поведения API]
        R3[Регрессионные ошибки]
    end
    
    subgraph "Стратегии митигации"
        M1[Тестирование в изолированной среде]
        M2[Постепенное развертывание]
        M3[Комплексное тестирование]
        M4[Резервное копирование]
    end
    
    R1 --> M1
    R2 --> M2
    R3 --> M3
    
    AllRisks[Все риски] --> M4
```

## Заключение

План аудита и исправлений адресует ключевые проблемы проекта:

1. **Критические проблемы**: Предупреждения в логах, которые могут влиять на функциональность
2. **Проблемы поддерживаемости**: Дублирование кода и избыточные компоненты
3. **Проблемы безопасности**: Устаревшие зависимости

Реализация этого плана улучшит стабильность, поддерживаемость и безопасность проекта без значительного изменения архитектуры.
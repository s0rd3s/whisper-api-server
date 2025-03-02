"""
Модуль transcribe.py содержит класс WhisperTranscriber, который использует модель Whisper от 
OpenAI для транскрибации аудиофайлов в текст. Класс включает в себя методы для загрузки модели, 
обработки аудио (с использованием класса AudioProcessor), и выполнения транскрибации. 
Обрабатывает выбор устройства (CPU, CUDA, MPS) для выполнения вычислений и обеспечивает 
возможность использования Flash Attention 2 для ускорения работы модели на поддерживаемых GPU.
"""

import time
from typing import Dict, Tuple

import librosa
import numpy as np
import torch
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    pipeline,
)

from .audio_processor import AudioProcessor
from .logger import logger

class WhisperTranscriber:
    """Класс для распознавания речи с помощью модели Whisper."""

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

        # Оптимальный тип для тензоров
        self.torch_dtype = torch.bfloat16

        # Создаем объект для обработки аудио
        self.audio_processor = AudioProcessor(config)

        # Определяем устройство для вычислений
        self.device = self._get_device()

        # Загружаем модель при инициализации
        self._load_model()

    def _get_device(self) -> torch.device:
        """Определение доступного устройства для вычислений."""
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

    def _load_model(self):
        """Загрузка модели и процессора."""
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
            Tuple с массивом numpy и частотой дискретизации.
        """
        try:
            audio_array, sampling_rate = librosa.load(file_path, sr=16000)
            return audio_array, sampling_rate
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио {file_path}: {e}")
            raise

    def transcribe(self, audio_path: str) -> str:
        """
        Транскрибация аудиофайла.
        
        Args:
            audio_path: Путь к обработанному аудиофайлу.

        Returns:
            Распознанный текст.
        """
        logger.info(f"Начало транскрибации файла: {audio_path}")

        # Загрузка аудио в формате numpy array
        audio_array, sampling_rate = self._load_audio(audio_path)

        # Транскрибация с корректным форматом данных
        result = self.asr_pipeline(
            {"raw": audio_array, "sampling_rate": sampling_rate}, 
            generate_kwargs={"language": self.language, "max_new_tokens": self.max_new_tokens},
            return_timestamps=self.return_timestamps
        )

        transcribed_text = result.get("text", "")
        logger.info(f"Транскрибация завершена: получено {len(transcribed_text)} символов текста")

        return transcribed_text

    def process_file(self, input_path: str) -> str:
        """
        Полный процесс обработки и транскрибации аудиофайла.
        
        Args:
            input_path: Путь к исходному аудиофайлу.
            
        Returns:
            Распознанный текст.
        """
        start_time = time.time()
        logger.info(f"Начало обработки файла: {input_path}")

        temp_files = []

        try:
            # Обработка аудио (конвертация, нормализация, добавление тишины)
            processed_path, temp_files = self.audio_processor.process_audio(input_path)

            # Транскрибация
            text = self.transcribe(processed_path)

            elapsed_time = time.time() - start_time
            logger.info(f"Обработка и транскрибация завершены за {elapsed_time:.2f} секунд")

            return text

        finally:
            # Очистка временных файлов
            self.audio_processor.cleanup_temp_files(temp_files)

#!/usr/bin/env python3
"""
Локальный сервис распознавания речи с использованием модели Whisper.
Запускается как системный сервис, загружает модель в память один раз и обрабатывает 
запросы через REST API.
"""

import os
import json
import time
import uuid
import tempfile
import logging
import subprocess
import argparse
from typing import Dict, Optional, Union, Tuple
from pathlib import Path

# Flask для REST API
from flask import Flask, request, jsonify
from routes import Routes
import waitress

# Импортируем компоненты из существующего кода
import torch
import librosa
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline


# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/transcribe.log")
    ]
)
logger = logging.getLogger("whisper-service")


class AudioProcessor:
    """Класс для предобработки аудиофайлов перед распознаванием."""
    
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
        """
        # Проверка расширения файла
        if input_path.lower().endswith('.wav'):
            # Проверяем, нужно ли преобразовывать WAV-файл (например, если частота не 16 кГц)
            try:
                info = subprocess.check_output(['soxi', input_path]).decode()
                if '16000 Hz' in info:
                    logger.info(f"Файл {input_path} уже в формате WAV с частотой 16 кГц")
                    return input_path
            except subprocess.CalledProcessError:
                logger.warning(f"Не удалось получить информацию о WAV-файле {input_path}")
                # Продолжаем конвертацию, чтобы быть уверенными в формате
        
        # Создаем временный файл для WAV
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{uuid.uuid4()}.wav")
        
        # Команда для конвертации
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "warning",
            "-i", input_path,
            "-ar", "16000",
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
        """
        # Создаем временный файл для нормализованного аудио
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{uuid.uuid4()}_normalized.wav")
        
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
        """
        # Создаем временный файл
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{uuid.uuid4()}_silence.wav")
        
        # Команда для добавления тишины в начало файла
        cmd = [
            "sox",
            input_path,
            output_path,
            "pad", "2.0"  # 2 секунды тишины в начале
        ]
        
        logger.info(f"Добавление тишины: {' '.join(cmd)}")
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            logger.info(f"Тишина добавлена: {output_path}")
            return output_path
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка при добавлении тишины: {e.stderr.decode()}")
            raise
    
    def cleanup_temp_files(self, file_paths: list):
        """
        Удаление временных файлов и директорий.
        
        Args:
            file_paths: Список путей к временным файлам.
        """
        for path in file_paths:
            try:
                if os.path.exists(path):
                    os.remove(path)
                    logger.debug(f"Удален временный файл: {path}")
                    
                    # Попытка удалить директорию, если она пуста
                    temp_dir = os.path.dirname(path)
                    if os.path.exists(temp_dir) and not os.listdir(temp_dir):
                        os.rmdir(temp_dir)
                        logger.debug(f"Удалена временная директория: {temp_dir}")
            except Exception as e:
                logger.warning(f"Не удалось очистить временный файл {path}: {e}")
    
    def process_audio(self, input_path: str) -> Tuple[str, list]:
        """
        Полная обработка аудиофайла: конвертация, нормализация и добавление тишины.
        
        Args:
            input_path: Путь к исходному аудиофайлу.
            
        Returns:
            Кортеж: (путь к обработанному файлу, список временных файлов для удаления)
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
            self.cleanup_temp_files(temp_files)
            raise


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


class WhisperServiceAPI:
    """Класс для API сервиса распознавания речи."""

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

        # Создание Flask-приложения
        self.app = Flask("whisper-service")

        # Регистрация маршрутов
        Routes(self.app, self.transcriber, self.config)
        # self._register_routes()

        logger.info(f"API сервис инициализирован, порт: {self.port}")
    
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

    def run(self):
        """Запуск сервиса."""
        logger.info(f"Запуск сервиса на порту {self.port}")
        
        # Использовать waitress для production-ready сервера
        waitress.serve(self.app, host='0.0.0.0', port=self.port)


def main():
    """Основная функция для запуска сервиса."""
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description="Сервис распознавания речи с использованием модели Whisper")
    parser.add_argument("--config", help="Путь к файлу конфигурации", default="config.json")
    
    args = parser.parse_args()
    
    # Запуск сервиса
    service = WhisperServiceAPI(args.config)
    service.run()


if __name__ == "__main__":
    main()

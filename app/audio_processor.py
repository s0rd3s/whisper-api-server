import os
import subprocess
import tempfile
import uuid
from typing import Dict, Tuple

# Импорт классов и функций из других модулей
from .logger import logger

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

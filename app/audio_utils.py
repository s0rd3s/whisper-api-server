"""
Модуль audio_utils.py содержит утилитарные функции для работы с аудио.
"""

import librosa
import numpy as np
import logging
from typing import Tuple

logger = logging.getLogger('app.audio_utils')


class AudioUtils:
    """Утилитарный класс для работы с аудио."""
    
    @staticmethod
    def load_audio(file_path: str, sr: int = 16000) -> Tuple[np.ndarray, int]:
        """
        Загрузка аудиофайла с использованием librosa.

        Args:
            file_path: Путь к аудиофайлу.
            sr: Целевая частота дискретизации.

        Returns:
            Кортеж (массив numpy, частота дискретизации).
            
        Raises:
            Exception: Если не удалось загрузить аудиофайл.
        """
        try:
            # Используем новый API librosa
            audio_array, sampling_rate = librosa.load(
                file_path, 
                sr=sr,
                mono=True  # Явно указываем моно
            )
            return audio_array, sampling_rate
        except Exception as e:
            logger.error(f"Ошибка при загрузке аудио {file_path}: {e}")
            raise
    
    @staticmethod
    def get_audio_duration(file_path: str) -> float:
        """
        Определяет длительность аудиофайла в секундах.

        Args:
            file_path: Путь к аудиофайлу.

        Returns:
            Длительность в секундах.
        """
        try:
            # Используем новый API librosa
            y, sr = librosa.load(
                file_path, 
                sr=None,
                mono=True  # Явно указываем моно
            )
            duration = librosa.get_duration(y=y, sr=sr)
            return duration
        except Exception as e:
            logger.error(f"Ошибка при определении длительности файла: {e}")
            return 0.0
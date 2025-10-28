"""
Модуль transcriber_service.py содержит класс TranscriptionService,
который отвечает за обработку и транскрибацию аудиофайлов.
"""

import os
import uuid
import tempfile
import time
import traceback
from typing import Dict, Tuple

from .utils import logger
from .audio_utils import AudioUtils
from .history_logger import HistoryLogger
from .audio_sources import AudioSource
from .validators import FileValidator, ValidationError
from .diarizer import Diarizer
from .audio_processor import AudioProcessor

class TranscriptionService:
    """
    Сервис для обработки и транскрибации аудиофайлов.
    
    Attributes:
        transcriber: Экземпляр транскрайбера.
        config (Dict): Словарь с конфигурацией.
        max_file_size_mb (int): Максимальный размер файла в МБ.
        history (HistoryLogger): Объект журналирования.
        diarizer (Diarizer): Объект для diarization.
        audio_processor (AudioProcessor): Объект для обработки аудио.
    """

    def __init__(self, transcriber, config: Dict):
        """
        Инициализация сервиса транскрибации.

        Args:
            transcriber: Экземпляр транскрайбера.
            config: Словарь с конфигурацией.
        """
        self.transcriber = transcriber
        self.config = config
        self.max_file_size_mb = self.config.get("file_validation", {}).get("max_file_size_mb", 100)
        self.history = HistoryLogger(config)
        self.diarizer = Diarizer(config)
        self.audio_processor = AudioProcessor(config)

    def transcribe_from_source(self, source: AudioSource, params: Dict = None, file_validator: FileValidator = None) -> Tuple[Dict, int]:
        """
        Транскрибирует аудиофайл из указанного источника.

        Args:
            source: Источник аудиофайла.
            params: Дополнительные параметры для транскрибации.
            file_validator: Валидатор файлов.

        Returns:
            Кортеж (JSON-ответ, HTTP-код).
        """
        file, filename, error = source.get_audio_file()
        if error:
            logger.warning(f"Ошибка получения файла из источника: {error}")
            return {"error": error}, 400

        if not file:
            logger.warning("Не удалось получить аудиофайл из источника")
            return {"error": "Failed to get audio file"}, 400
        
        if file_validator:
            try:
                file_validator.validate_file(file, filename)
            except ValidationError as e:
                logger.warning(f"Ошибка валидации файла '{filename}': {str(e)}")
                return {"error": str(e)}, 400

        params = params or {}
        language = params.get('language', self.config.get('language', 'en'))
        temperature = float(params.get('temperature', 0.0))
        prompt = params.get('prompt', '')
        return_timestamps = params.get('return_timestamps', self.config.get('return_timestamps', False))
        if isinstance(return_timestamps, str):
            return_timestamps = return_timestamps.lower() in ('true', 't', 'yes', 'y', '1')

        original_return_timestamps = self.transcriber.return_timestamps
        self.transcriber.return_timestamps = return_timestamps

        from .file_manager import temp_file_manager
        with temp_file_manager.temp_file(suffix='.mp3') as temp_file_path:
            file.save(temp_file_path)

            try:
                duration = AudioUtils.get_audio_duration(temp_file_path)
            except Exception as e:
                logger.error(f"Ошибка при определении длительности файла: {e}")
                return {"error": f"Не удалось определить длительность аудиофайла: {e}"}, 500

            if hasattr(source, 'cleanup'):
                file.file.close()
                source.cleanup()

            try:
                start_time = time.time()
                result = self.transcriber.process_file(temp_file_path)
                processing_time = time.time() - start_time

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
                    response = {
                        "text": result,
                        "processing_time": processing_time,
                        "response_size_bytes": len(str(result).encode('utf-8')),
                        "duration_seconds": duration,
                        "model": os.path.basename(self.config["model_path"])
                    }

                self.history.save(response, filename)
                return response, 200

            except Exception as e:
                logger.error(f"Ошибка при транскрибации файла '{filename}': {str(e)}")
                logger.error(f"Тип исключения: {type(e).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {"error": str(e)}, 500

            finally:
                self.transcriber.return_timestamps = original_return_timestamps

    def transcribe_and_diarize_from_source(self, source: AudioSource, params: Dict = None, file_validator: FileValidator = None) -> Tuple[Dict, int]:
        """
        Транскрибирует и диаризует аудиофайл из указанного источника.

        Args:
            source: Источник аудиофайла.
            params: Дополнительные параметры.
            file_validator: Валидатор файлов.

        Returns:
            Кортеж (JSON-ответ с текстом по спикерам, HTTP-код).
        """
        file, filename, error = source.get_audio_file()
        if error:
            logger.warning(f"Ошибка получения файла из источника: {error}")
            return {"error": error}, 400

        if not file:
            logger.warning("Не удалось получить аудиофайл из источника")
            return {"error": "Failed to get audio file"}, 400
        
        if file_validator:
            try:
                file_validator.validate_file(file, filename)
            except ValidationError as e:
                logger.warning(f"Ошибка валидации файла '{filename}': {str(e)}")
                return {"error": str(e)}, 400

        params = params or {}
        language = params.get('language', self.config.get('language', 'en'))
        temperature = float(params.get('temperature', 0.0))
        prompt = params.get('prompt', '')

        original_return_timestamps = self.transcriber.return_timestamps
        self.transcriber.return_timestamps = True

        from .file_manager import temp_file_manager
        temp_files = []
        with temp_file_manager.temp_file(suffix='.mp3') as temp_file_path:
            file.save(temp_file_path)
            temp_files.append(temp_file_path)

            # Use AudioProcessor for WAV conversion and processing
            try:
                wav_path, processing_temp_files = self.audio_processor.process_audio(temp_file_path)
                temp_files.extend(processing_temp_files)
            except Exception as e:
                logger.error(f"Ошибка при обработке аудио '{filename}': {str(e)}")
                return {"error": f"Failed to process audio: {str(e)}"}, 400

            try:
                duration = AudioUtils.get_audio_duration(wav_path)
            except Exception as e:
                logger.error(f"Ошибка при определении длительности файла: {e}")
                return {"error": f"Не удалось определить длительность аудиофайла: {e}"}, 500

            if hasattr(source, 'cleanup'):
                file.file.close()
                source.cleanup()

            try:
                start_time = time.time()
                result = self.transcriber.process_file(wav_path)
                segments = sorted(result.get("segments", []), key=lambda x: x["start_time_ms"])  # Sort segments chronologically
                full_text = result.get("text", "")

                speaker_segments = self.diarizer.diarize(wav_path)

                speaker_texts: Dict[str, list] = {str(seg['spk']): [] for seg in speaker_segments}
                unassigned_segments = []
                overlap_threshold = self.config.get("diarizer", {}).get("overlap_threshold", 0.3)

                # Assign segments based on overlap
                for seg in segments:
                    seg_start = seg["start_time_ms"] / 1000.0
                    seg_end = seg["end_time_ms"] / 1000.0
                    text = seg["text"]
                    best_overlap = 0
                    best_speaker = None
                    for speaker_seg in speaker_segments:
                        overlap = max(0, min(seg_end, speaker_seg["e"]) - max(seg_start, speaker_seg["s"]))
                        overlap_ratio = overlap / (seg_end - seg_start) if (seg_end - seg_start) > 0 else 0
                        if overlap_ratio >= overlap_threshold and overlap_ratio > best_overlap:
                            best_overlap = overlap_ratio
                            best_speaker = speaker_seg['spk']
                    if best_speaker is not None:
                        speaker_texts[str(best_speaker)].append({"start": seg_start, "end": seg_end, "text": text})
                        logger.debug(f"Assigned segment: {seg_start:.2f}-{seg_end:.2f}, text='{text}', to spk={best_speaker}, overlap={best_overlap:.2f}")
                    else:
                        unassigned_segments.append((seg_start, seg_end, text))
                        logger.debug(f"Unassigned segment: {seg_start:.2f}-{seg_end:.2f}, text='{text}'")

                # Fallback: Assign unassigned segments to nearest speaker
                for seg_start, seg_end, text in unassigned_segments:
                    min_distance = float('inf')
                    closest_speaker = None
                    for speaker_seg in speaker_segments:
                        distance = min(abs(seg_start - speaker_seg["e"]), abs(seg_end - speaker_seg["s"]))
                        if distance < min_distance and distance <= self.config.get("diarizer", {}).get("max_distance", 3.0):
                            min_distance = distance
                            closest_speaker = speaker_seg["spk"]
                    if closest_speaker is not None:
                        speaker_texts[str(closest_speaker)].append({"start": seg_start, "end": seg_end, "text": text})
                        logger.debug(f"Assigned via proximity: {seg_start:.2f}-{seg_end:.2f}, text='{text}', to spk={closest_speaker}, distance={min_distance:.2f}")
                    else:
                        if "Unknown" not in speaker_texts:
                            speaker_texts["Unknown"] = []
                        speaker_texts["Unknown"].append({"start": seg_start, "end": seg_end, "text": text})
                        logger.debug(f"Assigned to Unknown: {seg_start:.2f}-{seg_end:.2f}, text='{text}'")

                # Format speaker texts chronologically
                formatted_speakers = {}
                for speaker, texts in speaker_texts.items():
                    # Sort texts by start time to ensure chronological order
                    sorted_texts = sorted(texts, key=lambda x: x["start"])
                    formatted_speakers[speaker] = {
                        "text": " ".join(item["text"] for item in sorted_texts),
                        "segments": sorted_texts
                    }

                processing_time = time.time() - start_time
                response = {
                    "speakers": {k: v["text"] for k, v in formatted_speakers.items()},
                    "full_transcript": full_text,
                    "segments": segments,
                    "num_speakers": len(speaker_texts) - (1 if "Unknown" in speaker_texts else 0),
                    "processing_time": processing_time,
                    "duration_seconds": duration,
                    "model": os.path.basename(self.config["model_path"]),
                    "detailed_segments": formatted_speakers
                }

                self.history.save(response, filename)
                return response, 200

            except Exception as e:
                logger.error(f"Ошибка при транскрибации/диаризации файла '{filename}': {str(e)}")
                logger.error(f"Тип исключения: {type(e).__name__}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                return {"error": str(e)}, 500

            finally:
                self.transcriber.return_timestamps = original_return_timestamps
                temp_file_manager.cleanup_temp_files(temp_files)


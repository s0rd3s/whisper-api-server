"""
Модуль diarizer.py содержит класс Diarizer для выполнения speaker diarization с использованием pyannote.audio.
"""

import os
import traceback
from typing import Dict, List

import torch
from pyannote.audio import Pipeline

from .utils import logger

class Diarizer:
    """
    Класс для speaker diarization с использованием pyannote.audio.

    Attributes:
        config (Dict): Конфигурация diarizer из config.json.
        device (torch.device): Устройство для вычислений (GPU/CPU).
        pipeline (Pipeline): Загруженная модель для diarization.
    """

    def __init__(self, config: Dict):
        """
        Инициализация diarizer.

        Args:
            config: Словарь с конфигурацией (из config.json).
        """
        # Explicitly disable TF32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        logger.info("TF32 disabled for CUDA operations")

        self.config = config.get('diarizer', {})
        if not self.config:
            raise ValueError("Diarizer configuration not found in config.json")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for diarization: {self.device}")
        
        # Загрузка модели
        self.pipeline = self._load_pipeline()

    def _load_pipeline(self) -> Pipeline:
        """
        Загрузка pipeline pyannote для diarization.
        
        Returns:
            Pipeline: Загруженная модель.
        """
        repo = self.config.get("speaker_diarization_model_name", "pyannote/speaker-diarization-3.1")
        token = os.getenv("HF_TOKEN")
        logger.info(f"Loading pyannote speaker diarization pipeline: {repo}")
        pipeline = Pipeline.from_pretrained(repo, use_auth_token=token)
        pipeline.to(self.device)
        
        # Tune pipeline parameters
        pipeline_params = self.config.get("pipeline_params", {
            "segmentation": {
                "min_duration_off": 0.7  # Balanced to reduce over-segmentation
            },
            "clustering": {
                "method": "centroid",
                "min_cluster_size": 5,   # Balanced to avoid small clusters
                "threshold": 0.98        # Stricter to reduce speaker swapping
            }
        })
        pipeline.instantiate(pipeline_params)
        logger.info(f"Pipeline parameters: {pipeline_params}")
        
        return pipeline

    def diarize(self, audio_path: str) -> List[Dict]:
        """
        Выполнение diarization на аудиофайле.

        Args:
            audio_path: Путь к аудиофайлу.

        Returns:
            Список сегментов {'spk': int, 's': float, 'e': float}.
        """
        try:
            if not os.path.exists(audio_path):
                raise RuntimeError(f"Audio file not found: {audio_path}")

            logger.info(f"Diarizing audio: {audio_path}")
            
            diarization = self.pipeline(audio_path)
            
            speaker_map = {}
            next_speaker_id = 0
            segments = []
            
            for turn, _, speaker in diarization.itertracks(yield_label=True):
                duration = turn.end - turn.start
                if duration < self.config.get("min_segment_duration", 0.5):
                    logger.debug(f"Skipping short segment: {turn.start:.2f}-{turn.end:.2f} ({duration:.2f}s, raw_speaker={speaker})")
                    continue
                
                if speaker not in speaker_map:
                    speaker_map[speaker] = next_speaker_id
                    next_speaker_id += 1
                
                segments.append({"spk": speaker_map[speaker], "s": turn.start, "e": turn.end})
                logger.debug(f"Segment: spk={speaker_map[speaker]}, start={turn.start:.2f}, end={turn.end:.2f}, raw_speaker={speaker}")
            
            # Merge segments to reduce fragmentation
            gap = self.config.get("gap", 0.7)
            segments = self.merge_segments(segments, gap=gap)
            
            spk_cnt = len(speaker_map)
            logger.info(f"Diarization completed for {audio_path}, found {spk_cnt} speakers")
            logger.debug(f"Speaker map: {speaker_map}")
            return segments

        except Exception as e:
            logger.error(f"Failed to diarize {audio_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def merge_segments(self, segments: List[Dict], gap: float = 0.7) -> List[Dict]:
        """
        Слияние сегментов с дополнительной логикой для сглаживания.

        Args:
            segments: Список сегментов {'spk': int, 's': float, 'e': float}.
            gap: Макс. пауза для слияния (с).

        Returns:
            Список слитых сегментов.
        """
        if not segments:
            logger.info("No segments to merge")
            return []
        
        merged = []
        cur = segments[0].copy()
        for seg in segments[1:]:
            if seg["spk"] == cur["spk"] and seg["s"] <= cur["e"] + gap:
                cur["e"] = max(cur["e"], seg["e"])
                logger.debug(f"Merged segment: spk={cur['spk']}, {cur['s']:.2f}-{cur['e']:.2f}")
            else:
                if (cur["e"] - cur["s"]) >= self.config.get("min_segment_duration", 0.5):
                    merged.append(cur)
                    logger.debug(f"Appended segment: spk={cur['spk']}, {cur['s']:.2f}-{cur['e']:.2f}")
                else:
                    logger.debug(f"Skipped short merged segment: spk={cur['spk']}, {cur['s']:.2f}-{cur['e']:.2f}")
                cur = seg.copy()
        
        if (cur["e"] - cur["s"]) >= self.config.get("min_segment_duration", 0.5):
            merged.append(cur)
            logger.debug(f"Appended final segment: spk={cur['spk']}, {cur['s']:.2f}-{cur['e']:.2f}")
        
        # Post-merge: Smooth short segments
        i = 0
        while i < len(merged) - 1:
            if (merged[i]["e"] - merged[i]["s"]) < 1.0 and merged[i]["spk"] == merged[i+1]["spk"]:
                merged[i]["e"] = merged[i+1]["e"]
                logger.debug(f"Smoothed segment: spk={merged[i]['spk']}, {merged[i]['s']:.2f}-{merged[i]['e']:.2f}")
                del merged[i+1]
            else:
                i += 1
        
        logger.info(f"Merged into {len(merged)} segments")
        return merged


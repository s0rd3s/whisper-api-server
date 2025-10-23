"""
Модуль diarizer.py содержит класс Diarizer для выполнения speaker diarization с использованием NeMo.
"""

import json
import os
import yaml
import traceback
from typing import Dict, List

import numpy as np
import librosa
import torch
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from nemo.collections.asr.models import EncDecSpeakerLabelModel
import tempfile
import soundfile as sf

from .utils import logger

class Diarizer:
    """
    Класс для speaker diarization с использованием NeMo embeddings + clustering.

    Attributes:
        config (Dict): Конфигурация diarizer из config.json.
        device (torch.device): Устройство для вычислений (GPU/CPU).
        model (EncDecSpeakerLabelModel): Загруженная модель для эмбеддингов.
    """

    def __init__(self, config: Dict):
        """
        Инициализация diarizer.

        Args:
            config: Словарь с конфигурацией (из config.json).
        """
        self.config = config.get('diarizer', {})
        if not self.config:
            raise ValueError("Diarizer configuration not found in config.json")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device for diarization: {self.device}")
        
        # Загрузка модели
        self.model = self._load_model()

    def _load_model(self) -> EncDecSpeakerLabelModel:
        """
        Загрузка модели NeMo для эмбеддингов.
        
        Returns:
            EncDecSpeakerLabelModel: Загруженная модель.
        """
        repo = self.config.get("speaker_embeddings_model_name", "nvidia/speakerverification_en_titanet_large")
        token = os.getenv("HF_TOKEN")
        logger.info(f"Loading NeMo speaker embedding model: {repo}")
        try:
            if token:
                return EncDecSpeakerLabelModel.from_pretrained(repo, token=token).to(self.device).eval()
            raise TypeError
        except TypeError:
            return EncDecSpeakerLabelModel.from_pretrained(repo).to(self.device).eval()

    def extract_embeddings(self, wav: np.ndarray, sr: int, win_s: float = 3.0, step_s: float = 1.5) -> tuple[np.ndarray, List[tuple]]:
        """
        Извлечение эмбеддингов из аудио.

        Args:
            wav: Аудио данные.
            sr: Частота дискретизации.
            win_s: Длина окна (с).
            step_s: Шаг окна (с).

        Returns:
            (эмбеддинги, timestamps).
        """
        embs, stamps = [], []
        t = 0.0
        total_dur = len(wav) / sr
        logger.info(f"Extracting embeddings for audio of duration {total_dur:.2f}s with win_s={win_s}, step_s={step_s}")
        while t + win_s <= total_dur:
            segment = wav[int(t * sr): int((t + win_s) * sr)]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, segment, sr)
                tmp_path = tmp.name
            try:
                with torch.no_grad():
                    emb = self.model.get_embedding(tmp_path).cpu().numpy().squeeze()
                embs.append(emb / np.linalg.norm(emb))
                stamps.append((t, t + win_s))
            finally:
                os.remove(tmp_path)
            t += step_s
        if t < total_dur and (total_dur - t) >= 1.0:
            segment = wav[int(t * sr):]
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, segment, sr)
                tmp_path = tmp.name
            try:
                with torch.no_grad():
                    emb = self.model.get_embedding(tmp_path).cpu().numpy().squeeze()
                embs.append(emb / np.linalg.norm(emb))
                stamps.append((t, total_dur))
            finally:
                os.remove(tmp_path)
        if not embs:
            logger.warning("No embeddings extracted")
            return np.array([]), []
        logger.info(f"Extracted {len(embs)} embeddings")
        return np.stack(embs), stamps

    def auto_cluster(self, embs: np.ndarray, max_k: int = 10) -> np.ndarray:
        """
        Авто-кластеризация эмбеддингов.

        Args:
            embs: Эмбеддинги.
            max_k: Макс. число спикеров.

        Returns:
            Labels спикеров.
        """
        n_samples = len(embs)
        if n_samples < 2:
            logger.info("Too few embeddings; assuming 1 speaker")
            return np.zeros(n_samples, dtype=int)
        
        logger.info(f"Clustering {n_samples} samples, max_k={max_k}")
        
        min_k = 2
        best_lbl = np.zeros(n_samples, dtype=int)
        best_sc = -1
        for k in range(min_k, min(max_k + 1, n_samples + 1)):
            try:
                clustering = AgglomerativeClustering(n_clusters=k, metric='cosine', linkage='average')
                lbl = clustering.fit_predict(embs)
                sc = silhouette_score(embs, lbl, metric='cosine')
                logger.info(f"Clustering k={k}, silhouette_score={sc:.4f}")
                if sc > best_sc:
                    best_lbl, best_sc = lbl, sc
            except ValueError as e:
                logger.warning(f"Clustering failed for k={k}: {str(e)}")
                continue
        

        if best_sc < 0.1:
            logger.warning("Low silhouette scores; forcing 2 speakers for dialog")
            clustering = AgglomerativeClustering(n_clusters=2, metric='cosine', linkage='average')
            best_lbl = clustering.fit_predict(embs) if n_samples >= 2 else np.zeros(n_samples, dtype=int)
        
        return best_lbl

    def merge_segments(self, stamps: List[tuple], labels: np.ndarray, gap: float = 0.5) -> List[Dict]:
        """
        Слияние сегментов с дополнительной логикой для сглаживания.

        Args:
            stamps: Timestamps.
            labels: Labels спикеров.
            gap: Макс. пауза для слияния (с).

        Returns:
            Список сегментов {'spk': int, 's': float, 'e': float}.
        """
        if len(stamps) == 0:
            logger.info("No segments to merge")
            return []
        
        merged = []
        cur = {"spk": int(labels[0]), "s": stamps[0][0], "e": stamps[0][1]}
        for (s, e), lab in zip(stamps[1:], labels[1:]):
            lab = int(lab)
            if lab == cur["spk"] and s <= cur["e"] + gap:
                cur["e"] = e
            else:
                # Optional: If short isolated segment (<1s), assign to prev/next if close
                if (e - s) < 1.0 and merged and s <= merged[-1]["e"] + gap * 2:
                    merged[-1]["e"] = e
                else:
                    merged.append(cur)
                    cur = {"spk": lab, "s": s, "e": e}
        merged.append(cur)
        
        # Post-merge: Smooth short segments by merging to neighbors if same speaker
        i = 0
        while i < len(merged) - 1:
            if (merged[i]["e"] - merged[i]["s"]) < 1.0 and merged[i]["spk"] == merged[i+1]["spk"]:
                merged[i]["e"] = merged[i+1]["e"]
                del merged[i+1]
            else:
                i += 1
        
        logger.info(f"Merged into {len(merged)} segments")
        return merged

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

            logger.info(f"Reading audio: {audio_path}")
            wav, sr = librosa.load(audio_path, sr=16000, mono=True)

            win_s = self.config.get("win_s", 3.0)
            step_s = self.config.get("step_s", 1.5)
            max_k = self.config.get("max_speakers", 10)
            gap = self.config.get("gap", 0.5)

            logger.info("Extracting embeddings …")
            embs, stamps = self.extract_embeddings(wav, sr, win_s=win_s, step_s=step_s)

            logger.info(f"Auto-clustering 2..{max_k} speakers …")
            labels = self.auto_cluster(embs, max_k=max_k)
            spk_cnt = len(set(labels))
            logger.info(f"Selected {spk_cnt} speakers.")

            diar = self.merge_segments(stamps, labels, gap=gap)

            logger.info(f"Diarization completed for {audio_path}, found {spk_cnt} speakers")
            return diar

        except Exception as e:
            logger.error(f"Failed to diarize {audio_path}: {str(e)}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

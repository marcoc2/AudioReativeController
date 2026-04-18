"""Server-side stem separation service using audio_separator.

Models (lazy-downloaded on first use):
  - BS-RoFormer         for vocals/instrumental  (SDR 12.97)
  - BS-Roformer-SW      for 6-stem separation    (vocals, drums, bass, guitar, piano, other)

Available modes:
  - ``vocals``     : BS-RoFormer → 2 stems (vocals + instrumental)
  - ``every-stem`` : BS-Roformer-SW → 6 stems (vocals, drums, bass, guitar, piano, other)
"""

from __future__ import annotations

import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Dict, List, Optional
from uuid import uuid4

try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger("StemService")

@contextmanager
def _float32_default_dtype():
    """Temporarily force torch default dtype to float32."""
    import torch
    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)

class StemInfo:
    """Lightweight stem result descriptor."""
    __slots__ = ("id", "stem_type", "file_path", "file_name", "duration")

    def __init__(self, stem_type: str, file_path: str, file_name: str, duration: float = 0.0, id: Optional[str] = None):
        self.id = id or str(uuid4())
        self.stem_type = stem_type
        self.file_path = file_path
        self.file_name = file_name
        self.duration = duration

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "stem_type": self.stem_type,
            "file_path": self.file_path,
            "file_name": self.file_name,
            "duration": self.duration,
        }

def _get_audio_duration(path: str) -> float:
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.duration
    except Exception:
        pass
    try:
        import librosa
        dur = librosa.get_duration(path=path)
        return float(dur)
    except Exception:
        return 0.0

class StemService:
    """Service for audio stem separation."""
    ROFORMER_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
    ROFORMER_SW_MODEL = "BS-Roformer-SW.ckpt"
    DEMUCS_6S_MODEL = "htdemucs_6s.yaml"

    def __init__(self, output_root: Optional[str] = None, device: str = "auto"):
        self._separator = None
        self._lock = threading.Lock()
        self._device = device
        if output_root is None:
            # Assumes being in AudioReativeController/core/
            project_root = Path(__file__).resolve().parent.parent
            output_root = str(project_root / "stems_output")
        self._output_root = output_root
        os.makedirs(self._output_root, exist_ok=True)

    def _get_separator(self):
        if self._separator is None:
            with self._lock:
                if self._separator is None:
                    from audio_separator.separator import Separator
                    self._separator = Separator()
        return self._separator

    def is_available(self) -> bool:
        try:
            from audio_separator.separator import Separator
            return True
        except ImportError:
            return False

    def separate(self, audio_path: str, mode: str = "demucs", progress_callback: Optional[Callable[[str, float], None]] = None) -> List[StemInfo]:
        job_id = str(uuid4())
        output_dir = Path(self._output_root) / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        separator = self._get_separator()
        dispatch = {
            "vocals": self._separate_vocals,
            "demucs": self._separate_demucs_6s,
            "roformer": self._separate_roformer_sw,
            "every-stem": self._separate_demucs_6s,
        }
        handler = dispatch.get(mode)
        if handler is None:
            raise ValueError(f"Unknown stem mode: {mode!r}")
        with _float32_default_dtype():
            return handler(separator, audio_path, output_dir, progress_callback)

    @staticmethod
    def _resolve(fpath: str, output_dir: Path) -> Path:
        p = Path(fpath)
        return p if p.is_absolute() else output_dir / p

    @staticmethod
    def _classify_stem_type(fname_lower: str, candidates: tuple) -> str:
        for c in candidates:
            if c in fname_lower:
                return c
        return "other"

    def _separate_vocals(self, sep, audio_path, output_dir, cb) -> List[StemInfo]:
        sep.output_dir = str(output_dir)
        sep.output_format = "flac"
        sep.load_model(model_filename=self.ROFORMER_MODEL)
        files = sep.separate(audio_path)
        stems: List[StemInfo] = []
        for fp in files:
            fp = self._resolve(str(fp), output_dir)
            stem_type = "vocals" if "vocal" in fp.stem.lower() else "instrumental"
            stems.append(StemInfo(stem_type=stem_type, file_path=str(fp), file_name=fp.name, duration=_get_audio_duration(str(fp))))
        return stems

    def _separate_demucs_6s(self, sep, audio_path, output_dir, cb) -> List[StemInfo]:
        sep.output_dir = str(output_dir)
        sep.output_format = "flac"
        sep.load_model(model_filename=self.DEMUCS_6S_MODEL)
        files = sep.separate(audio_path)
        return self._wrap_6stems(files, output_dir, cb)

    def _separate_roformer_sw(self, sep, audio_path, output_dir, cb) -> List[StemInfo]:
        sep.output_dir = str(output_dir)
        sep.output_format = "flac"
        sep.load_model(model_filename=self.ROFORMER_SW_MODEL)
        files = sep.separate(audio_path)
        return self._wrap_6stems(files, output_dir, cb)

    def _wrap_6stems(self, files, output_dir, cb) -> List[StemInfo]:
        stems: List[StemInfo] = []
        for fp in files:
            fp = self._resolve(str(fp), output_dir)
            stem_type = self._classify_stem_type(fp.stem.lower(), ("vocals", "drums", "bass", "guitar", "piano", "other"))
            stems.append(StemInfo(stem_type=stem_type, file_path=str(fp), file_name=fp.name, duration=_get_audio_duration(str(fp))))
        return stems

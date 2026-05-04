"""Rhythm analyzer: derive a RhythmGrid from an audio signal.

Uses `librosa.beat.beat_track` to find beat timestamps and estimate BPM.
Downbeats are inferred by assuming the meter in `time_signature` and
anchoring the first detected beat as the first downbeat. Results are
cached on disk keyed by file hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np

from core.rhythm.grid import RhythmGrid

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = PROJECT_ROOT / "rhythm_cache"


def _file_hash(path: Path) -> str:
    st = path.stat()
    key = f"{path}|{st.st_size}|{st.st_mtime}"
    return hashlib.md5(key.encode()).hexdigest()


def _cache_path(audio_path: Path, time_signature: Tuple[int, int]) -> Path:
    h = _file_hash(audio_path)
    tag = f"{time_signature[0]}-{time_signature[1]}"
    return CACHE_DIR / f"{h}_{tag}.json"


def _load_cache(cache_file: Path, fps: int) -> Optional[RhythmGrid]:
    if not cache_file.exists():
        return None
    try:
        data = json.loads(cache_file.read_text())
        return RhythmGrid(
            bpm=float(data["bpm"]),
            time_signature=tuple(data["time_signature"]),
            fps=fps,
            beats=np.asarray(data["beats"], dtype=float),
            downbeats=np.asarray(data["downbeats"], dtype=float),
            start_offset=float(data.get("start_offset", 0.0)),
        )
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def _save_cache(cache_file: Path, grid: RhythmGrid) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "bpm": float(grid.bpm),
        "time_signature": list(grid.time_signature),
        "beats": grid.beats.tolist() if grid.beats is not None else [],
        "downbeats": grid.downbeats.tolist() if grid.downbeats is not None else [],
        "start_offset": float(grid.start_offset),
    }
    cache_file.write_text(json.dumps(payload))


def analyze(y: np.ndarray, sr: int, time_signature: Tuple[int, int] = (4, 4),
            fps: int = 24) -> RhythmGrid:
    """Run beat tracking on raw audio and return a populated RhythmGrid."""
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="frames")
    beats = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beats) < 2:
        bpm = float(tempo) if np.ndim(tempo) == 0 else float(np.atleast_1d(tempo)[0])
        return RhythmGrid(bpm=bpm or 120.0, time_signature=time_signature, fps=fps)
    downbeats = beats[:: time_signature[0]]
    return RhythmGrid.from_beats(
        beats, time_signature=time_signature, fps=fps, downbeats=downbeats
    )


def analyze_file(audio_path: str, time_signature: Tuple[int, int] = (4, 4),
                 fps: int = 24, use_cache: bool = True) -> RhythmGrid:
    """Load audio and analyze, with optional disk caching by file hash."""
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(audio_path)
    cache_file = _cache_path(audio_path, time_signature)
    if use_cache:
        cached = _load_cache(cache_file, fps)
        if cached is not None:
            return cached
    y, sr = librosa.load(str(audio_path), sr=None)
    grid = analyze(y, sr, time_signature=time_signature, fps=fps)
    if use_cache:
        _save_cache(cache_file, grid)
    return grid

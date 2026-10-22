"""The audio features of any audio file (a stem, a part, the song), read as the song's are:
``features_of(path, fps)`` -> ``t -> dict`` (AudioFeatureExtractor.get_features_at_time on
that file: its level, sub-bands, texture, brightness, change). A knob says which file with
``audio:`` (``{feature: texture.noisiness, audio: input/song/piano.wav}``), as a trigger says
whose attacks. Each file is analysed once a process and kept (a few seconds a song).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable, Optional

_kept: dict = {}                 # (file, mtime, fps) -> the extractor
_lock = threading.Lock()
_busy: dict = {}                 # the same key -> a lock (two threads asking at once analyse it once)


def resolve(path) -> Path:
    """A path as written in a scene: as it is, else from the repository's root."""
    from core.feature_extractor import PROJECT_ROOT
    p = Path(str(path))
    return p if p.is_absolute() or p.exists() else PROJECT_ROOT / p


def extractor_of(path, fps: int):
    """The file's extractor (analysed now, or kept from before)."""
    from core.feature_extractor import AudioFeatureExtractor
    p = resolve(path)
    if not p.is_file():
        raise FileNotFoundError(f"no audio at {path}")
    key = (str(p.resolve()).lower(), p.stat().st_mtime, int(fps))
    with _lock:
        got = _kept.get(key)
        if got is not None:
            return got
        busy = _busy.setdefault(key, threading.Lock())
    with busy:
        with _lock:
            got = _kept.get(key)
        if got is None:
            got = AudioFeatureExtractor(str(p), fps=int(fps), skip_separation=True)
            with _lock:
                _kept[key] = got
    return got


def features_of(path, fps: int) -> Callable[[float], Optional[dict]]:
    ex = extractor_of(path, fps)
    return lambda t: ex.get_features_at_time(t, apply_gate=False)


def known(path, fps: int) -> bool:
    """Whether the file is analysed already (a caller may say it is reading it meanwhile)."""
    p = resolve(path)
    if not p.is_file():
        return False
    key = (str(p.resolve()).lower(), p.stat().st_mtime, int(fps))
    with _lock:
        return key in _kept

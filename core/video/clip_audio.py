"""AudioLibrary — decode the audio track of video clips into numpy arrays.

Sibling of :mod:`core.video.clip_library`: same idea (system ffmpeg through a
rawvideo-style pipe, no extra dependency), same folder of clips, but pulling
the *sound* out instead of the pictures. Built for the cube TV room, where a
clip is not only something you look at but something you can hear.

Clips are addressed by path, not by index into a re-scanned folder, so a
caller can hand over ``ClipLibrary.paths`` and be certain that clip *i*'s
picture and clip *i*'s sound are the same file.

A clip with no audio track decodes to an empty array — silence, not an error;
a folder of AI-generated footage is allowed to be partly mute.
"""
from __future__ import annotations

import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

import numpy as np


def decode_audio(path: str | Path, sample_rate: int = 48000) -> np.ndarray:
    """Decode ``path``'s audio to float32 stereo ``(n, 2)`` at ``sample_rate``.

    Returns ``(0, 2)`` when the file carries no audio stream.
    """
    cmd = [
        "ffmpeg", "-v", "error",
        "-i", str(path),
        "-vn",
        "-f", "f32le", "-acodec", "pcm_f32le",
        "-ac", "2", "-ar", str(int(sample_rate)),
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros((0, 2), dtype=np.float32)
    n = len(proc.stdout) // 8               # 2 channels x float32
    if n == 0:
        return np.zeros((0, 2), dtype=np.float32)
    arr = np.frombuffer(proc.stdout[: n * 8], dtype=np.float32)
    return arr.reshape(n, 2)


class AudioLibrary:
    """Audio tracks of a fixed list of clips, decoded on demand.

    LRU-bounded like :class:`~core.video.clip_library.ClipLibrary`, because a
    run can stream through thousands of clips: audio is cheap per clip (~1.6 MB
    for 8s of stereo float32, against ~47 MB for the same clip as 256x256
    frames) but not free in the thousands.
    """

    def __init__(self, paths: Sequence[str | Path], sample_rate: int = 48000,
                 cache_size: int = 64):
        self.paths = [Path(p) for p in paths]
        self.sample_rate = int(sample_rate)
        self.cache_size = max(1, cache_size)
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> np.ndarray:
        idx %= len(self.paths)
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        self._cache[idx] = decode_audio(self.paths[idx], self.sample_rate)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return self._cache[idx]

    def release(self, idx: int) -> None:
        """Drop a clip the caller knows it will not ask for again."""
        self._cache.pop(idx % len(self.paths), None)

    def silent(self, idx: int) -> bool:
        return len(self.get(idx)) == 0

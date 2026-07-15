"""ClipLibrary — decode pre-rendered video clips into numpy frame arrays.

Uses the system ffmpeg (already a project requirement) through a rawvideo
pipe, so no extra Python dependency is needed. Each clip is decoded at the
output resolution and fps, meaning 1 output frame == 1 clip frame and
reverse playback is just index arithmetic. Decoded clips are kept in a
small LRU cache since only the current (and next) clip is needed at a time.
"""
from __future__ import annotations

import subprocess
from collections import OrderedDict
from pathlib import Path
from typing import List

import numpy as np

VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


class ClipFrames:
    """Decoded frames of a single clip as a (n, H, W, 3) uint8 array."""

    def __init__(self, path: Path, frames: np.ndarray):
        self.path = path
        self.frames = frames

    def __len__(self) -> int:
        return len(self.frames)

    def frame(self, idx: int) -> np.ndarray:
        idx = max(0, min(len(self.frames) - 1, idx))
        return self.frames[idx]


class ClipLibrary:
    """Folder of video clips, decoded on demand at a fixed size/fps."""

    def __init__(
        self,
        folder: str | Path,
        width: int,
        height: int,
        fps: int,
        cache_size: int = 4,
    ):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(self.folder)
        self.width = width
        self.height = height
        self.fps = fps
        self.cache_size = max(1, cache_size)
        self.paths: List[Path] = sorted(
            p for p in self.folder.iterdir() if p.suffix.lower() in VIDEO_EXTS
        )
        if not self.paths:
            raise ValueError(f"no video clips found in {self.folder}")
        self._cache: OrderedDict[int, ClipFrames] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> ClipFrames:
        idx %= len(self.paths)
        if idx in self._cache:
            self._cache.move_to_end(idx)
            return self._cache[idx]
        clip = self._decode(self.paths[idx])
        self._cache[idx] = clip
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return clip

    def _decode(self, path: Path) -> ClipFrames:
        W, H = self.width, self.height
        vf = (
            f"scale={W}:{H}:force_original_aspect_ratio=decrease,"
            f"pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:black,setsar=1,fps={self.fps}"
        )
        cmd = [
            "ffmpeg", "-v", "error",
            "-i", str(path),
            "-vf", vf,
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-",
        ]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg failed decoding {path}: {proc.stderr.decode(errors='replace')}"
            )
        frame_bytes = W * H * 3
        n = len(proc.stdout) // frame_bytes
        if n == 0:
            raise RuntimeError(f"no frames decoded from {path}")
        arr = np.frombuffer(proc.stdout[: n * frame_bytes], dtype=np.uint8)
        return ClipFrames(path, arr.reshape(n, H, W, 3))

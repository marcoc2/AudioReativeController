"""Dry-run resolver — compute the output arrangement without decoding video.

``MetaLibrary`` satisfies ClipComposer's library interface using only clip
durations (ffprobe, parallel, no frame decoding), so a whole song resolves
in milliseconds: which clip plays at every frame, in which direction, at
what speed. Feeds the ARC Studio output lane and future arrangement tools.
"""
from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np

from core.video.clip_library import VIDEO_EXTS
from core.video.composer import ClipComposer


class MetaClip:
    """A clip's frame count, nothing else."""

    def __init__(self, n_frames: int):
        self._n = max(1, int(n_frames))

    def __len__(self) -> int:
        return self._n

    def frame(self, idx: int):
        return None   # dry-run: no pixels


class MetaLibrary:
    """Clip lengths via ffprobe; same file ordering as ClipLibrary."""

    def __init__(self, folder, fps: int, workers: int = 8):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise NotADirectoryError(self.folder)
        self.fps = fps
        self.paths = sorted(
            p for p in self.folder.iterdir() if p.suffix.lower() in VIDEO_EXTS
        )
        if not self.paths:
            raise ValueError(f"no video clips found in {self.folder}")
        with ThreadPoolExecutor(max_workers=workers) as ex:
            durs = list(ex.map(self._probe, self.paths))
        self._clips = [MetaClip(round(d * fps)) for d in durs]

    @staticmethod
    def _probe(path: Path) -> float:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True,
        )
        try:
            return float(out.stdout.strip())
        except ValueError:
            return 1.0

    def __len__(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> MetaClip:
        return self._clips[idx % len(self._clips)]


@dataclass
class Segment:
    t0: float
    t1: float
    clip_idx: int
    clip_name: str
    direction: int


def resolve_song(library, grid, notes: Sequence, video_cfg: dict, fps: int,
                 start: float, end: float):
    """Replay the composer's transport over [start, end).

    Returns (segments, times, speeds): segments merge consecutive frames
    with the same clip+direction; times/speeds sample the playback speed
    per output frame (the gravity curve, ready to draw).
    """
    comp = ClipComposer(library, grid, notes, video_cfg)
    comp.seek(start)
    n = max(1, int((end - start) * fps))
    times = np.empty(n, dtype=np.float64)
    speeds = np.empty(n, dtype=np.float32)
    segments: List[Segment] = []
    cur = None
    for fi in range(n):
        t = start + fi / fps
        comp.frame_at(t)
        tp = comp.transport
        times[fi] = t
        speeds[fi] = tp.speed
        if cur is None or (tp.clip_idx, tp.direction) != (cur.clip_idx, cur.direction):
            if cur is not None:
                cur.t1 = t
            name = (Path(library.paths[tp.clip_idx]).stem
                    if hasattr(library, "paths") else str(tp.clip_idx))
            cur = Segment(t, end, tp.clip_idx, name, tp.direction)
            segments.append(cur)
    return segments, times, speeds

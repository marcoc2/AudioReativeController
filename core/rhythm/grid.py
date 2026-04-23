"""RhythmGrid: time representation in musical units (bars, beats, subdivisions).

Supports fixed-BPM grids and variable grids populated from beat/downbeat arrays
(e.g. from librosa beat tracking or MIDI).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

SUBDIVISIONS = {
    "whole": 1,
    "half": 2,
    "quarter": 4,
    "eighth": 8,
    "16th": 16,
    "32nd": 32,
    "64th": 64,
}


@dataclass
class RhythmGrid:
    bpm: float
    time_signature: Tuple[int, int] = (4, 4)
    fps: int = 24
    beats: Optional[np.ndarray] = None
    downbeats: Optional[np.ndarray] = None
    start_offset: float = 0.0

    def __post_init__(self):
        if self.bpm <= 0:
            raise ValueError(f"bpm must be > 0, got {self.bpm}")
        if self.beats is not None:
            self.beats = np.asarray(self.beats, dtype=float)
        if self.downbeats is not None:
            self.downbeats = np.asarray(self.downbeats, dtype=float)

    @classmethod
    def from_beats(cls, beats, time_signature=(4, 4), fps=24, downbeats=None) -> "RhythmGrid":
        beats = np.asarray(beats, dtype=float)
        if len(beats) < 2:
            raise ValueError("need at least 2 beats to infer BPM")
        bpm = 60.0 / float(np.median(np.diff(beats)))
        if downbeats is None:
            downbeats = beats[:: time_signature[0]]
        return cls(bpm=bpm, time_signature=time_signature, fps=fps,
                   beats=beats, downbeats=np.asarray(downbeats, dtype=float),
                   start_offset=float(beats[0]))

    @property
    def beat_duration(self) -> float:
        return 60.0 / self.bpm

    @property
    def bar_duration(self) -> float:
        return self.beat_duration * self.time_signature[0]

    @property
    def whole_duration(self) -> float:
        return self.beat_duration * self.time_signature[1]

    def _default_tol(self) -> float:
        return 0.5 / self.fps

    def _has_markers(self, arr: Optional[np.ndarray]) -> bool:
        return arr is not None and len(arr) > 0

    def phase(self, t: float) -> float:
        """Position within the current bar in [0, 1)."""
        if self._has_markers(self.downbeats):
            idx = np.searchsorted(self.downbeats, t, side="right") - 1
            if idx < 0:
                rel = (t - float(self.downbeats[0])) % self.bar_duration
                return rel / self.bar_duration
            start = float(self.downbeats[idx])
            end = float(self.downbeats[idx + 1]) if idx + 1 < len(self.downbeats) else start + self.bar_duration
            length = end - start
            if length <= 0:
                return 0.0
            return ((t - start) / length) % 1.0
        rel = (t - self.start_offset) % self.bar_duration
        return rel / self.bar_duration

    def beat_phase(self, t: float) -> float:
        """Position within the current beat in [0, 1)."""
        if self._has_markers(self.beats):
            idx = np.searchsorted(self.beats, t, side="right") - 1
            if idx < 0:
                rel = (t - float(self.beats[0])) % self.beat_duration
                return rel / self.beat_duration
            start = float(self.beats[idx])
            end = float(self.beats[idx + 1]) if idx + 1 < len(self.beats) else start + self.beat_duration
            length = end - start
            if length <= 0:
                return 0.0
            return ((t - start) / length) % 1.0
        rel = (t - self.start_offset) % self.beat_duration
        return rel / self.beat_duration

    def _near_marker(self, t: float, markers: np.ndarray, tol: float) -> bool:
        idx = np.searchsorted(markers, t)
        if idx > 0 and abs(t - float(markers[idx - 1])) <= tol:
            return True
        if idx < len(markers) and abs(t - float(markers[idx])) <= tol:
            return True
        return False

    def is_beat(self, t: float, tol: Optional[float] = None) -> bool:
        tol = self._default_tol() if tol is None else tol
        if self._has_markers(self.beats):
            return self._near_marker(t, self.beats, tol)
        rel = (t - self.start_offset) % self.beat_duration
        return rel <= tol or (self.beat_duration - rel) <= tol

    def is_downbeat(self, t: float, tol: Optional[float] = None) -> bool:
        tol = self._default_tol() if tol is None else tol
        if self._has_markers(self.downbeats):
            return self._near_marker(t, self.downbeats, tol)
        rel = (t - self.start_offset) % self.bar_duration
        return rel <= tol or (self.bar_duration - rel) <= tol

    def subdivision_duration(self, name: str) -> float:
        if name not in SUBDIVISIONS:
            raise ValueError(f"unknown subdivision {name!r}; expected one of {list(SUBDIVISIONS)}")
        return self.whole_duration / SUBDIVISIONS[name]

    def subdivision_index(self, t: float, name: str) -> int:
        step = self.subdivision_duration(name)
        anchor = float(self.downbeats[0]) if self._has_markers(self.downbeats) else self.start_offset
        return int((t - anchor) // step)

    def is_subdivision(self, t: float, name: str, tol: Optional[float] = None) -> bool:
        tol = self._default_tol() if tol is None else tol
        step = self.subdivision_duration(name)
        anchor = float(self.downbeats[0]) if self._has_markers(self.downbeats) else self.start_offset
        rel = (t - anchor) % step
        return rel <= tol or (step - rel) <= tol

    def frames_per_bar(self) -> float:
        return self.bar_duration * self.fps

    def frames_per_beat(self) -> float:
        return self.beat_duration * self.fps

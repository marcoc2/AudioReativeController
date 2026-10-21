"""Each pixel from its own moment: a map picks how far back in time every pixel looks.

The slitscan delays by row; here a map says, pixel by pixel, how long ago to look
(TouchDesigner's Time Machine TOP), and the delay is counted in note values on the
song's grid, so the past it shows is a sixteenth, an eighth, a beat ago:

    - source: time_map
      map: figures          # figures: the figures now, the world lagging more the farther
                            #          from them (needs the mask track: --masks or track:)
                            # luma:    bright pixels now, dark ones late | noise: slow blotches
      depth: 1/2            # the longest delay: seconds or a note value (default: half a bar)
      step: 1/16            # delays come in whole steps of this (0: smooth)
      amount: 1.0           # 0..1 of the depth, or any knob that follows the music
      rgb: false            # true: red now, green one step back, blue two (the map ignored)

CPU (a ring of past frames); a GPU stack hands it the frame and takes it back.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

MAPS = ("figures", "luma", "noise")


class TimeMap:
    """Post-op: see the module docstring."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: float,
                 features_at=None, onset_loader=None, grid=None,
                 default_track: Optional[str] = None, tracks: Optional[Dict] = None):
        from core.video.cast import _Role
        from core.video.layers import _music_knob
        self.map = spec.get("map", "figures")
        if self.map not in MAPS:
            raise ValueError(f"time_map: map {self.map!r} (use {', '.join(MAPS)})")
        self.rgb = bool(spec.get("rgb", False))
        self.fps = float(fps)
        depth = _Role._period(spec.get("depth", "1/2" if grid is not None else 1.0), grid, 0)
        step = spec.get("step", "1/16" if grid is not None else 0)
        self.step = _Role._period(step, grid, 0) if step else 0.0
        self.depth = depth
        self.n = max(2, int(round(depth * self.fps)) + 1)            # frames kept
        if self.rgb:
            self.n = max(self.n, int(round(2 * max(self.step, 1.0 / self.fps) * self.fps)) + 1)
        self._amount = _music_knob("amount", spec.get("amount", 1.0), notes, features_at, onset_loader, grid)
        self.W, self.H = width, height
        self._ring: Optional[np.ndarray] = None
        self._head = -1
        self._filled = 0
        self.track = None
        if self.map == "figures" and not self.rgb:
            folder = spec.get("track") or default_track
            if not folder:
                raise ValueError("time_map: map: figures needs a mask track (track: <folder>) or the render's --masks")
            from core.segment.mask_track import MaskTrack
            tracks = tracks if tracks is not None else {}
            if folder not in tracks:
                tracks[folder] = MaskTrack(folder)
            self.track = tracks[folder]
        rng = np.random.default_rng(int(spec.get("seed", 0)))
        self._noise = cv2.GaussianBlur(cv2.resize(rng.uniform(0, 1, (6, 4)).astype(np.float32), (width, height),
                                                  interpolation=cv2.INTER_CUBIC), (0, 0), max(width, height) / 30)
        self._noise = (self._noise - self._noise.min()) / max(1e-6, float(self._noise.max() - self._noise.min()))
        self._rows = np.arange(height)[:, None]
        self._cols = np.arange(width)[None, :]

    def _field(self, frame: np.ndarray, t: float) -> np.ndarray:
        """0 (now) .. 1 (the whole depth back), per pixel."""
        if self.map == "luma":
            return 1.0 - frame.astype(np.float32).dot(np.array([0.299, 0.587, 0.114], np.float32)) / 255.0
        if self.map == "noise":
            return self._noise
        lab = self.track.labels_at(t)
        if lab is None:
            return self._noise
        inside = cv2.resize((lab > 0).astype(np.uint8), (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        d = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)
        return np.clip(d / (0.5 * min(self.W, self.H)), 0.0, 1.0)

    def _frames_back(self, seconds: np.ndarray) -> np.ndarray:
        if self.step > 0:
            seconds = np.round(seconds / self.step) * self.step
        k = np.round(seconds * self.fps).astype(np.int32)
        return np.clip(k, 0, min(self._filled - 1, self.n - 1))

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self._ring is None or self._ring.shape[1:3] != frame.shape[:2]:
            self._ring = np.zeros((self.n,) + frame.shape, np.uint8)
            self._head, self._filled = -1, 0
        self._head = (self._head + 1) % self.n
        self._ring[self._head] = frame
        self._filled = min(self._filled + 1, self.n)
        amount = float(np.clip(self._amount(t), 0.0, 1.0))
        if amount <= 0.0 or self._filled < 2:
            return frame
        if self.rgb:
            s = max(self.step, 1.0 / self.fps) * amount
            out = frame.copy()
            for c, back in ((1, 1), (2, 2)):
                k = int(self._frames_back(np.array(back * s)))
                out[..., c] = self._ring[(self._head - k) % self.n][..., c]
            return out
        k = self._frames_back(self._field(frame, t) * np.float32(self.depth * amount))
        idx = (self._head - k) % self.n
        return self._ring[idx, self._rows, self._cols]

    def observe(self, frame: np.ndarray, t: float) -> None:
        """Gated by a ``bars:`` window: keep the ring going so it opens with a past."""
        self.process(frame, t)

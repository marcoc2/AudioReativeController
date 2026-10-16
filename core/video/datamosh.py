"""Datamosh on the cut: the old shot's pixels dragged by the new shot's motion.

A video codec sends a full picture now and then (the I-frame) and, between them, only
how blocks of the picture move. Drop the I-frame at a cut and the decoder keeps moving
the old picture's blocks with the new shot's motion: the classic glitch of music videos.
Here the composer knows its cuts (``ClipComposer.cut_times``); at each one this post-op
keeps the last picture of the shot that ends and, frame by frame, pushes it along the
optical flow of the shot that begins (OpenCV DIS, on a small copy), the flow evened out
over macroblocks. The real picture comes back on the next hit of ``until`` (the snare as
the I-frame) or after ``length``.

    - source: datamosh
      length: 1/1         # longest a mosh lasts: seconds or a note value (default: a whole note)
      until: {track: snare}   # optional: the hit that brings the real picture back
      block: 16           # macroblock size in pixels at 720p (0: a smooth smear)
      push: 1.5           # how far the motion carries the pixels (1: as the new shot moves)
      leak: 0.02          # how much of the true picture seeps in each frame
      every: 1            # mosh every cut (2: every other one, ...)

Put it right above the clips: the motion it follows is the picture below it. CPU only
(the flow is OpenCV's); a GPU stack hands it the frame and takes it back.
"""
from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

FLOW_SIDE = 200                   # the flow is worked out on a copy this many pixels wide


class Datamosh:
    """Post-op: see the module docstring."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: float,
                 onset_loader=None, grid=None, cuts=None):
        from core.video.cast import _Role
        from core.video.layers import _layer_hits
        if cuts is None:
            raise ValueError("datamosh: needs the clips' cuts (a scene with - source: clips)")
        self._cuts = cuts
        self._known = (-np.inf, np.zeros(0))
        length = spec.get("length", "1/1" if grid is not None else 2.0)
        self.length = _Role._period(length, grid, 0)
        u = spec.get("until")
        self._until = np.array(sorted(_layer_hits(u, notes, onset_loader, grid)), dtype=float) if u else None
        self.block = max(0, int(round(float(spec.get("block", 16)) * height / 720.0)))
        self.push = float(spec.get("push", 1.5))
        self.leak = float(spec.get("leak", 0.02))
        self.every = max(1, int(spec.get("every", 1)))
        self.W, self.H, self.fps = width, height, float(fps)
        self._dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._prev_small: Optional[np.ndarray] = None   # the picture below, last frame (grey, small)
        self._last: Optional[np.ndarray] = None         # the picture below, last frame
        self._mosh: Optional[np.ndarray] = None         # the moshed picture (float32)
        self._end = -np.inf
        self._k = None                                  # the cut being moshed
        h = max(8, int(round(FLOW_SIDE * height / width)))
        self._small = (FLOW_SIDE, h)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        self._xx, self._yy = xx, yy

    def _times(self, t: float) -> np.ndarray:
        if t + 600.0 > self._known[0]:
            self._known = (t + 3600.0, np.asarray(self._cuts(t + 3600.0), dtype=float))
        return self._known[1]

    def _end_of(self, cut: float) -> float:
        end = cut + self.length
        if self._until is not None:
            i = int(np.searchsorted(self._until, cut + 1e-6, side="left"))
            if i < len(self._until):
                end = min(end, float(self._until[i]))
        return end

    def _flow(self, small: np.ndarray) -> np.ndarray:
        """The motion from the last frame to this one, at full size (H×W×2, pixels)."""
        f = self._dis.calc(self._prev_small, small, None)
        if self.block > 1:                                # the same push for a whole macroblock
            bw, bh = max(1, self.W // self.block), max(1, self.H // self.block)
            f = cv2.resize(f, (bw, bh), interpolation=cv2.INTER_AREA)
            f = cv2.resize(f, (self.W, self.H), interpolation=cv2.INTER_NEAREST)
        else:
            f = cv2.resize(f, (self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return f * np.float32(self.W / self._small[0])

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), self._small, interpolation=cv2.INTER_AREA)
        ts = self._times(t)
        i = int(np.searchsorted(ts, t + 1e-6, side="right"))      # cuts at or before t
        out = frame
        if i > 0 and (i - 1) % self.every == 0 and self._k != i - 1 and t - ts[i - 1] < 1.5 / self.fps \
                and self._last is not None:
            self._k = i - 1                                        # a cut just came: start moshing
            self._mosh = self._last.astype(np.float32)
            self._end = self._end_of(float(ts[i - 1]))
        if self._mosh is not None and t < self._end and self._prev_small is not None:
            f = self._flow(small) * np.float32(self.push)
            # each pixel takes what was where the motion came from
            moved = cv2.remap(self._mosh, self._xx - f[..., 0], self._yy - f[..., 1],
                              interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            self._mosh = moved * np.float32(1 - self.leak) + frame.astype(np.float32) * np.float32(self.leak)
            out = np.clip(self._mosh + 0.5, 0, 255).astype(np.uint8)
        elif self._mosh is not None and t >= self._end:
            self._mosh = None                                      # the I-frame: the real picture
        self._prev_small = small
        self._last = frame
        return out

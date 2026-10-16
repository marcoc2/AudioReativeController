"""Pixel sorting with musical spans: runs of bright pixels sorted into streaks.

Along each row (or column), the pixels brighter than a threshold form runs, and each run
is sorted by brightness, so the picture streaks (Kim Asendorf's ASDFPixelSort idea,
written afresh). Here the threshold follows the music: ``amount`` 0 sorts nothing, 1
sorts nearly everything, so held to a note's envelope the streaks grow while it sounds
and melt back once it ends.

    - source: pixel_sort
      amount: {adsr: {track: pad}, release: 0.6}   # how much of the picture sorts (0..1; a knob)
      direction: [horizontal, vertical]   # one, or a list taken in turn at each clip cut
      reverse: false        # true: dark to bright the other way

CPU (numpy); a GPU stack hands it the frame and takes it back.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def sort_rows(img: np.ndarray, lo: float, reverse: bool = False) -> np.ndarray:
    """Each row's runs of pixels with luma above ``lo`` (0..1), sorted by luma in place."""
    luma = img.astype(np.float32).dot(LUMA) / 255.0
    on = luma > lo
    if not on.any():
        return img
    # a run id per pixel that only grows along the row: a new id wherever "on" starts or a
    # pixel is off, so sorting by (id, luma) sorts inside the runs and leaves the rest
    start = on & ~np.concatenate([np.zeros((on.shape[0], 1), bool), on[:, :-1]], axis=1)
    run = np.cumsum(start | ~on, axis=1).astype(np.float64)
    key = run * 4.0 + np.where(on, (1.0 - luma if reverse else luma), 0.0)
    order = np.argsort(key, axis=1, kind="stable")
    return np.take_along_axis(img, order[..., None], axis=1)


class PixelSort:
    """Post-op: see the module docstring."""

    def __init__(self, spec: dict, notes: Sequence, fps: float, features_at=None, onset_loader=None,
                 grid=None, cuts=None):
        from core.video.layers import _music_knob
        self._amount = _music_knob("amount", spec.get("amount", 0.5), notes, features_at, onset_loader, grid)
        d = spec.get("direction", "horizontal")
        self.directions = [d] if isinstance(d, str) else list(d)
        if not self.directions or any(x not in ("horizontal", "vertical") for x in self.directions):
            raise ValueError(f"pixel_sort: direction {d!r} (horizontal, vertical, or a list of them)")
        if len(self.directions) > 1 and cuts is None:
            raise ValueError("pixel_sort: a list of directions turns at the clip cuts (a scene with - source: clips)")
        self.reverse = bool(spec.get("reverse", False))
        self._cuts = cuts
        self._known = (-np.inf, np.zeros(0))

    def _direction(self, t: float) -> str:
        if len(self.directions) == 1:
            return self.directions[0]
        if t + 600.0 > self._known[0]:
            self._known = (t + 3600.0, np.asarray(self._cuts(t + 3600.0), dtype=float))
        k = int(np.searchsorted(self._known[1], t + 1e-6, side="right"))
        return self.directions[k % len(self.directions)]

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        a = float(np.clip(self._amount(t), 0.0, 1.0))
        if a <= 0.0:
            return frame
        lo = 1.0 - a
        if self._direction(t) == "vertical":
            return np.ascontiguousarray(sort_rows(frame.transpose(1, 0, 2), lo, self.reverse).transpose(1, 0, 2))
        return sort_rows(frame, lo, self.reverse)

"""CellSystem — 8-bit microscope cells driven by audio features.

Renders on a tiny grid (default 160x160) with a 12-hue posterized palette,
meant to be nearest-neighbor upscaled: authentically 8-bit and cheap.

Audio mapping (via the controls dict from AudioFeatureExtractor):
  chroma (12-vec) -> each cell is bound to a pitch class; its color energy
                     follows that bin (harmony literally paints the dish)
  flux            -> membrane wobble amplitude + phase speed
  bass/sub energy -> drift speed pulse
Mitosis (cell division) is triggered externally (e.g. snare hits).
"""
from __future__ import annotations

from typing import Optional

import numpy as np


def _hsv(h: float, s: float, v: float):
    i = int(h * 6) % 6
    f = h * 6 - int(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r, g, b = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)][i]
    return [r * 255, g * 255, b * 255]


# 12 pitch classes -> 12 hues around the wheel
PALETTE = np.array([_hsv(i / 12.0, 0.75, 0.95) for i in range(12)], dtype=np.float32)


class CellSystem:
    def __init__(self, n_base: int = 12, n_max: int = 48, grid: int = 160,
                 seed: Optional[int] = None):
        self.grid = int(grid)
        self.n_max = int(n_max)
        self.rng = np.random.default_rng(seed)
        n = int(n_base)
        g = self.grid
        self.pos = self.rng.uniform(0.12, 0.88, (n, 2)).astype(np.float32) * g
        self.vel = self.rng.normal(0.0, 2.5, (n, 2)).astype(np.float32)
        self.rad = (self.rng.uniform(0.05, 0.11, n) * g).astype(np.float32)
        self.bin = self.rng.integers(0, 12, n)
        self.phase = self.rng.uniform(0, 6.283, n).astype(np.float32)
        self._flux = 0.0
        self._chroma = np.full(12, 0.5, dtype=np.float32)

    @property
    def n(self) -> int:
        return len(self.rad)

    def mitosis(self) -> None:
        """Split a random cell: parent shrinks, child buds off beside it."""
        if self.n >= self.n_max:
            return
        i = int(self.rng.integers(self.n))
        self.rad[i] *= 0.78
        off = self.rng.normal(0, self.rad[i] * 0.9, 2).astype(np.float32)
        self.pos = np.vstack([self.pos, self.pos[i] + off])
        self.vel = np.vstack([self.vel, -self.vel[i] * 0.8])
        self.rad = np.append(self.rad, self.rad[i] * 0.9)
        self.bin = np.append(self.bin, self.bin[i])
        self.phase = np.append(self.phase, self.rng.uniform(0, 6.283))

    def step(self, dt: float, controls: dict) -> None:
        self._flux = float(controls.get("flux", 0.0) or 0.0)
        ch = controls.get("chroma")
        if ch is not None and len(ch) == 12:
            self._chroma = np.asarray(ch, dtype=np.float32)
        sub = controls.get("subbands") or {}
        bass = float(sub.get("bass", controls.get("bass_energy", 0.0) or 0.0))
        g = self.grid
        self.pos += self.vel * dt * (1.0 + 3.0 * bass)
        # bounce on the dish borders
        for ax in (0, 1):
            low, high = self.pos[:, ax] < 2, self.pos[:, ax] > g - 2
            self.vel[low | high, ax] *= -1.0
        np.clip(self.pos, 2, g - 2, out=self.pos)
        self.phase += dt * (1.0 + 6.0 * self._flux)

    def render(self) -> np.ndarray:
        """(grid, grid, 3) uint8, posterized."""
        g = self.grid
        img = np.zeros((g, g, 3), dtype=np.float32)
        yy, xx = np.mgrid[0:g, 0:g].astype(np.float32)
        wob_amp = 0.06 + 0.22 * self._flux
        for i in range(self.n):
            cx, cy = self.pos[i]
            dx, dy = xx - cx, yy - cy
            d = np.sqrt(dx * dx + dy * dy)
            th = np.arctan2(dy, dx)
            rr = self.rad[i] * (1.0 + wob_amp * np.sin(5.0 * th + self.phase[i]))
            energy = float(self._chroma[self.bin[i]])
            col = PALETTE[self.bin[i]] * (0.30 + 0.70 * energy)
            body = d < rr
            img[body] = np.maximum(img[body], col * 0.55)
            ring = np.abs(d - rr) < 1.4          # membrane
            img[ring] = np.maximum(img[ring], col)
            nuc = d < self.rad[i] * 0.32          # nucleus
            img[nuc] = np.maximum(img[nuc], col * (0.85 + 0.4 * self._flux))
        return ((img // 43) * 43).astype(np.uint8)   # 6 levels/channel = 8-bit feel

"""JuliaSystem — Julia set navigating the Mandelbrot boundary, audio-driven.

The whole fractal is defined by one complex parameter ``c``. We keep ``c``
orbiting the main cardioid of the Mandelbrot set — the edge of chaos:

  chroma (dominant pitch class) -> target angle on the cardioid (harmony
                                   picks the fractal "family"; the angle
                                   glides, so chord changes morph smoothly)
  bass energy                   -> radial factor across the boundary
                                   (calm = connected shapes, loud = dust)
  flux                          -> color-cycling speed
  centroid                      -> palette base hue
  zoom pulse / invert           -> external envelopes (kick / snare hits)

Rendered by vectorized escape-time on a small grid, posterized and meant
for nearest-neighbor upscale — same 8-bit identity as core.cells.
"""
from __future__ import annotations

import numpy as np

TWO_PI = 2.0 * np.pi


def _cardioid(theta: float) -> complex:
    """Point on the boundary of the Mandelbrot main cardioid."""
    mu = np.exp(1j * theta)
    return mu / 2.0 - mu * mu / 4.0


class JuliaSystem:
    def __init__(self, grid: int = 256, iters: int = 48):
        self.grid = int(grid)
        self.iters = int(iters)
        self.theta = 0.0          # current angle on the cardioid
        self.cycle = 0.0          # palette cycling phase
        self._radius = 0.96
        self._hue = 0.6
        ax = np.linspace(-1.6, 1.6, self.grid, dtype=np.float32)
        self._plane = (ax[None, :] + 1j * ax[:, None]).astype(np.complex64)

    def step(self, dt: float, controls: dict) -> None:
        ch = controls.get("chroma")
        if ch is not None and len(ch) == 12:
            target = (int(np.argmax(ch)) / 12.0) * TWO_PI
            # glide along the shortest arc (chord changes morph, not jump)
            diff = (target - self.theta + np.pi) % TWO_PI - np.pi
            self.theta += diff * min(1.0, dt * 2.5)
        sub = controls.get("subbands") or {}
        bass = float(sub.get("bass", controls.get("bass_energy", 0.0) or 0.0))
        self._radius = 0.92 + 0.22 * bass          # crosses the boundary ~1.0
        flux = float(controls.get("flux", 0.0) or 0.0)
        self.cycle += dt * (0.4 + 4.0 * flux)
        cen = controls.get("centroid")
        if cen is not None:
            self._hue = 0.55 + 0.35 * float(cen)   # dark blue -> amber

    def render(self, zoom: float = 0.0, invert: float = 0.0) -> np.ndarray:
        """(grid, grid, 3) uint8. ``zoom``/``invert`` are 0..1 envelopes."""
        c = np.complex64(_cardioid(self.theta) * self._radius)
        z = (self._plane / np.float32(1.0 + 0.8 * zoom)).copy()
        counts = np.zeros(z.shape, dtype=np.int32)
        alive = np.ones(z.shape, dtype=bool)
        for i in range(self.iters):
            z[alive] = z[alive] * z[alive] + c
            escaped = alive & (z.real * z.real + z.imag * z.imag > 4.0)
            counts[escaped] = i + 1
            alive &= ~escaped
        # palette: iteration count -> hue around the base, cycling
        t = counts.astype(np.float32) / self.iters
        h = (self._hue + 0.25 * t + 0.15 * np.sin(self.cycle)) % 1.0
        v = np.where(alive, 0.06, 0.25 + 0.75 * t)      # interior near-black
        if invert > 0.0:
            v = v * (1.0 - invert) + (1.0 - v) * invert
        # cheap HSV->RGB (s fixed at 0.85), vectorized
        i6 = (h * 6.0).astype(np.int32) % 6
        f = h * 6.0 - np.floor(h * 6.0)
        p, q, tt = v * 0.15, v * (1 - 0.85 * f), v * (1 - 0.85 * (1 - f))
        r = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4], [v, q, p, p, tt], v)
        g = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4], [tt, v, v, q, p], p)
        b = np.select([i6 == 0, i6 == 1, i6 == 2, i6 == 3, i6 == 4], [p, p, tt, v, v], q)
        img = (np.stack([r, g, b], axis=-1) * 255.0)
        return ((img // 43) * 43).astype(np.uint8)   # posterize: 8-bit feel

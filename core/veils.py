"""Veils — a luminous mist that breathes. A layer source for music without attacks.

Most generators here answer to hits: something strikes, something jumps. A
drone or a pad gives them nothing to do. This one is driven by the *texture*
of the sound (see ``core/texture``) and by its harmony:

    loudness        -> how bright the mist is (dB scale, so a quiet drone still glows)
    swell           -> breathing: the field dilates while the sound grows and
                       contracts while it fades
    tremolo         -> shimmer, at the rate the volume really pulsates
    noisiness       -> sparks, when the top end of the mix opens up
    percussive      -> how fast the sheets drift
    hue             -> the chord: each root has its colour, and a chord change
                       is a slow glide across the wheel plus a soft bloom
    ripples         -> rings spreading from the centre, one per hit (kick)

The image is two interfering sheets of domain-warped sine bands — no noise
tables, no state beyond a few phases — computed at a fraction of the output
size and upscaled: mist has no fine detail to lose, and it keeps the CPU cost
near a quarter. Everything advances by ``dt``; nothing reads the wall clock.
"""
from __future__ import annotations

import colorsys
from typing import Sequence

import numpy as np

RIPPLE_LIFE = 1.6        # seconds a ring stays visible
RIPPLE_SPEED = 0.85      # field units per second (the frame is about 2 units wide)
RIPPLE_WIDTH = 0.07
BREATH_GAIN = 0.55       # how far a full-scale swell pushes the zoom per second
BREATH_LEAK = 6.0        # seconds for the breath to settle back
HUE_GLIDE = 1.2          # seconds for the colour to reach a new chord
SPARK_FLOOR = 0.25       # noisiness below this makes no sparks


def _hsv(h: float, s: float, v: float) -> np.ndarray:
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v), dtype=np.float32)


try:                      # optional: ~4 ms per 1080p frame instead of ~120 ms
    import cv2 as _cv2
except ImportError:       # not a declared dependency of the project
    _cv2 = None


class _Upscaler:
    """Bilinear upscale of a uint8 image: OpenCV when it is around, otherwise a
    separable numpy pass with the index maths done once (same picture, slower)."""

    def __init__(self, src_h: int, src_w: int, dst_h: int, dst_w: int):
        self.dst = (dst_w, dst_h)
        self.same = (src_h, src_w) == (dst_h, dst_w)
        ys = np.linspace(0, src_h - 1, dst_h, dtype=np.float32)
        xs = np.linspace(0, src_w - 1, dst_w, dtype=np.float32)
        self.y0 = np.floor(ys).astype(np.intp)
        self.x0 = np.floor(xs).astype(np.intp)
        self.y1 = np.minimum(self.y0 + 1, src_h - 1)
        self.x1 = np.minimum(self.x0 + 1, src_w - 1)
        self.wy = (ys - self.y0)[:, None, None]
        self.wx = (xs - self.x0)[None, :, None]

    def __call__(self, img: np.ndarray, use_cv2: bool = True) -> np.ndarray:
        if self.same:
            return img
        if _cv2 is not None and use_cv2:
            return _cv2.resize(img, self.dst, interpolation=_cv2.INTER_LINEAR)
        small = img.astype(np.float32)
        rows = small[:, self.x0] * (1 - self.wx) + small[:, self.x1] * self.wx
        return (rows[self.y0] * (1 - self.wy) + rows[self.y1] * self.wy + 0.5).astype(np.uint8)


class VeilsSystem:
    def __init__(self, width: int, height: int, seed: int = 0, detail: float = 0.5,
                 hue: float = 0.8, brightness: float = 1.0):
        self.width, self.height = int(width), int(height)
        self.seed = int(seed)
        self.brightness = float(brightness)
        detail = float(np.clip(detail, 0.1, 1.0))
        h = max(8, int(round(self.height * detail)))
        w = max(8, int(round(self.width * detail)))
        self._up = _Upscaler(h, w, self.height, self.width)

        aspect = self.width / self.height
        ys, xs = np.meshgrid(np.linspace(-1, 1, h, dtype=np.float32),
                             np.linspace(-aspect, aspect, w, dtype=np.float32), indexing="ij")
        self._x, self._y = xs, ys
        self._r = np.sqrt(xs * xs + ys * ys)
        self._glow = np.exp(-(self._r ** 2) * 1.1).astype(np.float32)

        rng = np.random.default_rng(self.seed)
        self._offsets = rng.uniform(0, 2 * np.pi, size=(3, 2)).astype(np.float32)
        self._spins = rng.uniform(0.6, 1.4, size=(3, 2)).astype(np.float32)

        self.phase = 0.0          # drift of the sheets
        self.breath = 0.0         # log-zoom: > 0 dilated, < 0 contracted
        self.shimmer = 0.0        # phase of the tremolo
        self.hue = float(hue) % 1.0
        self._frame = 0

    def render(self, dt: float, loudness: float = 0.5, swell: float = 0.0,
               tremolo_depth: float = 0.0, tremolo_rate: float = 0.0,
               noisiness: float = 0.0, percussive: float = 0.0,
               target_hue: float | None = None, bloom: float = 0.0,
               ripple_ages: Sequence[float] = ()) -> np.ndarray:
        """Advance by ``dt`` seconds and draw one frame (uint8, H x W x 3)."""
        dt = max(0.0, float(dt))
        # loudness arrives on a dB scale; the useful part of it is the top two thirds
        level = float(np.clip((loudness - 0.25) / 0.65, 0.0, 1.0)) ** 1.3

        self.phase += dt * (0.12 + 0.5 * level + 0.8 * float(percussive))
        self.breath += dt * (BREATH_GAIN * float(swell) - self.breath / BREATH_LEAK)
        self.breath = float(np.clip(self.breath, -0.6, 0.6))
        self.shimmer += dt * 2 * np.pi * float(tremolo_rate)
        if target_hue is not None:                       # glide the short way round the wheel
            gap = ((target_hue - self.hue + 0.5) % 1.0) - 0.5
            self.hue = (self.hue + gap * (1 - np.exp(-dt / HUE_GLIDE))) % 1.0

        zoom = np.float32(np.exp(-self.breath))
        px, py = self._x * zoom, self._y * zoom
        ph = np.float32(self.phase)
        for i, (freq, amp) in enumerate(((1.7, 0.55), (2.9, 0.33), (4.7, 0.21))):
            (o1, o2), (s1, s2) = self._offsets[i], self._spins[i]
            px, py = (px + amp * np.sin(freq * py + ph * s1 + o1),
                      py + amp * np.sin(freq * px - ph * s2 + o2))
        sheet_a = 0.5 + 0.5 * np.sin(2.2 * px + 1.3 * py + 0.7 * ph)
        sheet_b = 0.5 + 0.5 * np.sin(1.1 * (px - py) - 0.4 * ph + 1.7)
        veil = (sheet_a * sheet_b) ** 1.6

        for age in ripple_ages:
            if 0.0 <= age < RIPPLE_LIFE:
                ring = np.exp(-(((self._r - age * RIPPLE_SPEED) / RIPPLE_WIDTH) ** 2))
                veil = veil + ring * np.float32(0.8 * (1 - age / RIPPLE_LIFE) ** 2)

        if noisiness > SPARK_FLOOR:                      # sparks where the mist already is
            rng = np.random.default_rng([self.seed, self._frame])
            density = 0.02 * (float(noisiness) - SPARK_FLOOR) / (1 - SPARK_FLOOR)
            sparks = rng.random(veil.shape, dtype=np.float32) > 1 - density
            veil = veil + sparks * (0.4 + veil) * 1.5

        deep, base, accent = _hsv(self.hue - 0.08, 0.9, 0.22), _hsv(self.hue, 0.75, 1.0), _hsv(self.hue + 0.42, 0.55, 1.0)
        glow = self._glow[..., None]
        v = veil[..., None]
        rgb = (deep * (0.35 + 0.65 * glow)
               + base * v
               + accent * (v * (sheet_b ** 3)[..., None] * 0.6 + glow * np.float32(0.5 * bloom)))

        gain = (0.10 + 0.90 * level) * (1 + 0.6 * float(tremolo_depth) * np.sin(self.shimmer))
        self._frame += 1
        # to 8 bits while still small, then upscale: both steps are far cheaper that way round
        small = (np.clip(rgb * np.float32(gain * self.brightness), 0.0, 1.0) * 255).astype(np.uint8)
        return np.ascontiguousarray(self._up(small))

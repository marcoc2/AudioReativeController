"""Chladni — sand on a vibrating plate. A layer source where the sound draws its own figure.

Sprinkle sand on a metal plate and bow its edge: the grains dance away from
wherever the plate moves and come to rest on the lines that stand still, and
every pitch has its own figure. That is the whole model here:

    chord           -> which figure: each root has a pair of mode numbers (n, m);
                       a chord change moves the still lines and the sand migrates
    brightness      -> how fine the figure is (a higher sound has more lines, as
                       on a real plate)
    harmonic change -> the figure wanders while the timbre moves, rests while it holds
    loudness        -> how hard the plate shakes: how fast the sand finds its lines,
                       and how bright the picture is
    swell           -> the plate breathes: the figure dilates as the sound grows
    jolt            -> the sand is really thrown off its lines and settles again (slow:
                       right for a lone hit, a blur under a run of sixteenths)
    gestures        -> what a drum voice does to the *picture*, on the very frame of the
                       hit, leaving the sand where it is: ``punch`` (the figure swells and
                       flashes), ``twist`` (it turns a few degrees, alternating sides),
                       ``sizzle`` (the grains leap and glow), ``ring`` (a shock wave
                       travels out through the sand)
    noisiness       -> grains catch the light
    pour            -> how much of the sand is on the plate yet

A grain moves by a random walk whose step is the local amplitude of the plate
``|f|`` (zero on a nodal line, so it stops there), helped by a slide down
``f**2``. ``f = cos(n pi x) cos(m pi y) - cos(m pi x) cos(n pi y)``, the classic
square-plate approximation; ``n`` and ``m`` need not be whole, so one figure
melts into the next. Everything advances by ``dt`` and every random draw is
seeded by ``(seed, frame)``: same inputs, same film.
"""
from __future__ import annotations

import colorsys

import numpy as np

try:                      # optional, as in core/veils: only makes the blurs faster
    import cv2 as _cv2
except ImportError:
    _cv2 = None

# one figure per root, indexed in circle-of-fifths order (ChladniLayer: MODES[(root * 7) % 12])
MODES = ((1, 2), (2, 3), (1, 4), (3, 4), (2, 5), (1, 6),
         (3, 5), (4, 5), (2, 7), (3, 7), (5, 6), (4, 7))

MODE_GLIDE = 0.9         # seconds for the figure to reach a new chord's modes
HUE_GLIDE = 1.2          # seconds for the colour to do the same
WALK = 0.55              # random-walk step at full amplitude, field units per sqrt(second)
SLIDE = 0.9              # pull down the slope of f**2
REST = 0.012             # the walk never quite stops: grains on a line still shiver
JOLT = 0.10              # how far a full hit throws the sand (field units per sqrt(second))
WANDER = 0.45            # how far the modes stray from the chord's pair
PUNCH = 0.05             # how much a punch swells the figure
TWIST = 0.07             # radians a twist turns it
SIZZLE = 0.018           # how far a sizzle makes the grains leap (field units)
RING_LIFE = 0.6          # seconds a shock wave lasts
RING_SPEED = 3.2         # field units per second
RING_WIDTH = 0.12
RING_PUSH = 0.06
BREATH_GAIN = 0.35
BREATH_LEAK = 6.0


def _hsv(h: float, s: float, v: float) -> np.ndarray:
    return np.array(colorsys.hsv_to_rgb(h % 1.0, s, v), dtype=np.float32)


def _blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05:
        return img
    if _cv2 is not None:
        return _cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, sigma=sigma, mode="nearest")


class ChladniSystem:
    def __init__(self, width: int, height: int, seed: int = 0, grains: int = 120_000,
                 modes=(2, 3), hue: float = 0.8, brightness: float = 1.0):
        self.width, self.height = int(width), int(height)
        self.seed = int(seed)
        self.brightness = float(brightness)
        self.aspect = self.width / self.height
        rng = np.random.default_rng(self.seed)
        self.pos = rng.uniform(-1, 1, size=(int(grains), 2)).astype(np.float32)
        self.pos[:, 0] *= self.aspect
        self._xs = np.linspace(-self.aspect, self.aspect, self.width, dtype=np.float32)
        self._ys = np.linspace(-1, 1, self.height, dtype=np.float32)
        self._px = self.height / 720.0            # blur radii are given for a 720-line frame

        self.n, self.m = float(modes[0]), float(modes[1])
        self.path = 0.0           # how far the timbre has travelled (drives the wander)
        self.breath = 0.0
        self.shimmer = 0.0
        self.hue = float(hue) % 1.0
        self._frame = 0

    # -- the plate ---------------------------------------------------------
    def current_modes(self, fineness: float = 1.0) -> tuple:
        n = (self.n + WANDER * np.sin(self.path)) * fineness
        m = (self.m + WANDER * np.cos(0.63 * self.path + 1.0)) * fineness
        return float(n), float(m)

    @staticmethod
    def field(x, y, n: float, m: float):
        """Amplitude of the plate and its gradient at (x, y)."""
        a, b = np.float32(n * np.pi), np.float32(m * np.pi)
        cax, cbx, cay, cby = np.cos(a * x), np.cos(b * x), np.cos(a * y), np.cos(b * y)
        f = cax * cby - cbx * cay
        fx = -a * np.sin(a * x) * cby + b * np.sin(b * x) * cay
        fy = -b * cax * np.sin(b * y) + a * cbx * np.sin(a * y)
        return f, fx, fy

    # -- one frame -----------------------------------------------------------
    def render(self, dt: float, level: float = 0.5, modes=None, fineness: float = 1.0,
               harmonic_change: float = 0.0, swell: float = 0.0, jolt: float = 0.0,
               noisiness: float = 0.0, tremolo_depth: float = 0.0, tremolo_rate: float = 0.0,
               target_hue: float | None = None, bloom: float = 0.0,
               pour: float = 1.0, punch: float = 0.0, twist: float = 0.0,
               sizzle: float = 0.0, ring_ages=()) -> np.ndarray:
        """Advance by ``dt`` seconds and draw one frame (uint8, H x W x 3)."""
        dt = max(0.0, float(dt))
        level = float(np.clip(level, 0.0, 1.0))
        glide = 1 - np.exp(-dt / MODE_GLIDE)
        if modes is not None:
            self.n += (float(modes[0]) - self.n) * glide
            self.m += (float(modes[1]) - self.m) * glide
        if target_hue is not None:
            gap = ((target_hue - self.hue + 0.5) % 1.0) - 0.5
            self.hue = (self.hue + gap * (1 - np.exp(-dt / HUE_GLIDE))) % 1.0
        self.path += dt * 1.6 * float(harmonic_change)
        self.breath += dt * (BREATH_GAIN * float(swell) - self.breath / BREATH_LEAK)
        self.breath = float(np.clip(self.breath, -0.4, 0.4))
        self.shimmer += dt * 2 * np.pi * float(tremolo_rate)
        zoom = np.float32(np.exp(-self.breath))
        n, m = self.current_modes(fineness)

        # the sand
        active = int(round(len(self.pos) * float(np.clip(pour, 0.0, 1.0))))
        pos = self.pos[:active]
        heat = np.zeros(0, dtype=np.float32)
        if active:
            rng = np.random.default_rng([self.seed, self._frame])
            f, fx, fy = self.field(pos[:, 0] * zoom, pos[:, 1] * zoom, n, m)
            amp = np.abs(f)
            slope = np.float32(SLIDE * dt * level / (n * n + m * m)) * f
            sigma = (np.float32(WALK * level) * amp + np.float32(REST * level + JOLT * float(jolt))) \
                * np.float32(np.sqrt(dt))
            pos[:, 0] += -slope * fx + sigma * rng.standard_normal(active, dtype=np.float32)
            pos[:, 1] += -slope * fy + sigma * rng.standard_normal(active, dtype=np.float32)
            a = np.float32(self.aspect)                      # grains bounce off the rim
            x, y = pos[:, 0], pos[:, 1]
            np.copyto(x, np.where(x > a, 2 * a - x, np.where(x < -a, -2 * a - x, x)))
            np.copyto(y, np.where(y > 1, 2 - y, np.where(y < -1, -2 - y, y)))
            np.clip(x, -a, a, out=x)
            np.clip(y, -1, 1, out=y)
            heat = np.minimum(amp, 1.0)                      # grains still on the move glow warmer
            if noisiness > 0:
                heat = heat + (rng.random(active, dtype=np.float32) < 0.04 * float(noisiness)) * np.float32(6.0)

        frame = self._draw(pos, heat, n, m, zoom, level, float(bloom),
                           float(tremolo_depth) * np.sin(self.shimmer),
                           float(punch), float(twist), float(sizzle), ring_ages)
        self._frame += 1
        return frame

    def _gesture(self, x, y, heat, punch, twist, sizzle, ring_ages):
        """Where the grains are *drawn* under the drum gestures (the sand itself stays put)."""
        r = np.sqrt(x * x + y * y) + np.float32(1e-4)
        push = np.float32(PUNCH * punch) * r
        for age in ring_ages:
            if 0.0 <= age < RING_LIFE:
                band = (np.exp(-(((r - np.float32(age * RING_SPEED)) / np.float32(RING_WIDTH)) ** 2))
                        * np.float32(1 - age / RING_LIFE))
                push = push + np.float32(RING_PUSH) * band
                heat = heat + band
        k = 1 + push / r
        c, s = np.float32(np.cos(TWIST * twist)), np.float32(np.sin(TWIST * twist))
        x, y = (x * c - y * s) * k, (x * s + y * c) * k
        if sizzle > 0:
            rng = np.random.default_rng([self.seed, self._frame, 1])
            x = x + rng.standard_normal(len(x), dtype=np.float32) * np.float32(SIZZLE * sizzle)
            y = y + rng.standard_normal(len(y), dtype=np.float32) * np.float32(SIZZLE * sizzle)
            heat = heat + np.float32(sizzle)
        return x, y, heat

    def _draw(self, pos, heat, n, m, zoom, level, bloom, shimmer,
              punch=0.0, twist=0.0, sizzle=0.0, ring_ages=()) -> np.ndarray:
        h, w = self.height, self.width
        if len(pos):
            x, y = pos[:, 0], pos[:, 1]
            if punch or twist or sizzle or len(ring_ages):
                x, y, heat = self._gesture(x, y, heat, punch, twist, sizzle, ring_ages)
                inside = (np.abs(x) <= self.aspect) & (np.abs(y) <= 1)
                x, y, heat = x[inside], y[inside], heat[inside]
            ix = ((x / self.aspect + 1) * 0.5 * (w - 1) + 0.5).astype(np.intp)
            iy = ((y + 1) * 0.5 * (h - 1) + 0.5).astype(np.intp)
            idx = iy * w + ix
            norm = np.float32(h * w / len(self.pos))          # 1.0 = all the sand spread evenly
            rest = 1 - np.float32(0.85) * np.minimum(heat, 1.0)   # a grain is drawn pale at rest, warm on the move
            still = np.bincount(idx, weights=rest, minlength=h * w).astype(np.float32).reshape(h, w) * norm
            hot = np.bincount(idx, weights=heat, minlength=h * w).astype(np.float32).reshape(h, w) * norm
        else:
            still = hot = np.zeros((h, w), dtype=np.float32)
        sand = 1 - np.exp(-0.22 * _blur(still, 0.9 * self._px))
        warm = 1 - np.exp(-0.45 * _blur(hot, 1.2 * self._px))
        halo = 1 - np.exp(-0.10 * _blur(still, 7.0 * self._px))

        # the plate itself: where it moves most it glows faintly (separable, so cheap)
        swell = zoom / np.float32(1 + PUNCH * punch)
        a, b = np.float32(n * np.pi) * swell, np.float32(m * np.pi) * swell
        plate = np.abs(np.outer(np.cos(b * self._ys), np.cos(a * self._xs))
                       - np.outer(np.cos(a * self._ys), np.cos(b * self._xs))) * np.float32(0.5)

        deep, base = _hsv(self.hue - 0.06, 0.9, 0.30), _hsv(self.hue, 0.8, 1.0)
        pale, accent = _hsv(self.hue + 0.03, 0.22, 1.0), _hsv(self.hue + 0.30, 0.85, 1.0)
        rgb = (deep * (plate * np.float32(0.15 + 0.55 * level + 0.8 * bloom))[..., None]
               + base * (halo * np.float32(0.55))[..., None]
               + pale * sand[..., None]
               + accent * (warm * np.float32(0.9))[..., None])
        gain = (0.30 + 0.70 * level) * (1 + 0.5 * shimmer) * (1 + 0.4 * punch) * self.brightness
        return (np.clip(rgb * np.float32(gain), 0.0, 1.0) * 255).astype(np.uint8)

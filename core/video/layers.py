"""Layer compositor — stack visual sources with blend modes.

A composition is a bottom-up list of layers. The base layer supplies the
canvas (normally the clip compositor); each further layer renders its own
RGB frame and is blended in with an opacity that may vary per frame
(trigger envelopes now; audio-driven modulation next).

Scene YAML:

    video:
      layers:
        - source: clips            # base — uses the video: section itself
        - source: solid
          color: [255, 255, 255]
          blend: add               # normal | add | screen | multiply
          triggers:
            snare: {notes: [38, 40], envelope: 0.1}   # flash decay (s)
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence
import math
import collections
from scipy.ndimage import affine_transform

import numpy as np

BLENDS = {"normal", "add", "screen", "multiply"}


def draw_line_numpy(img: np.ndarray, x0: float, y0: float, x1: float, y1: float, color: tuple):
    steps = int(max(abs(x1 - x0), abs(y1 - y0)) * 1.5)
    steps = max(2, steps)
    t_vals = np.linspace(0, 1, steps)
    xs = x0 + t_vals * (x1 - x0)
    ys = y0 + t_vals * (y1 - y0)
    H, W = img.shape[:2]
    for x, y in zip(xs, ys):
        ix, iy = int(round(x)), int(round(y))
        if 0 <= ix < W and 0 <= iy < H:
            img[iy, ix] = color


def draw_circle_numpy(img: np.ndarray, cx: float, cy: float, r: float, color: tuple):
    H, W = img.shape[:2]
    x_min = max(0, int(math.floor(cx - r)))
    x_max = min(W - 1, int(math.ceil(cx + r)))
    y_min = max(0, int(math.floor(cy - r)))
    y_max = min(H - 1, int(math.ceil(cy + r)))
    
    if x_max < x_min or y_max < y_min:
        return
        
    yy, xx = np.ogrid[y_min : y_max + 1, x_min : x_max + 1]
    mask = (xx - cx)**2 + (yy - cy)**2 <= r*r
    img[y_min : y_max + 1, x_min : x_max + 1][mask] = color


def scale_rotate_image(img: np.ndarray, scale: float, angle: float) -> np.ndarray:
    H, W, C = img.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    
    inv_s = 1.0 / max(1e-4, scale)
    M = np.array([
        [inv_s * cos_a, inv_s * sin_a],
        [-inv_s * sin_a, inv_s * cos_a]
    ])
    
    center = np.array([cy, cx])
    offset = center - M.dot(center)
    
    out = np.empty_like(img)
    for c in range(C):
        out[:, :, c] = affine_transform(
            img[:, :, c],
            matrix=M,
            offset=offset,
            order=1,
            mode='constant',
            cval=0
        )
    return out


def hue_shift_channels(img: np.ndarray, shift: float) -> np.ndarray:
    if shift <= 0.0:
        return img
    r = img[:, :, 0].astype(np.float32)
    g = img[:, :, 1].astype(np.float32)
    b = img[:, :, 2].astype(np.float32)
    
    k = shift % 1.0
    nr = (1.0 - k) * r + k * g
    ng = (1.0 - k) * g + k * b
    nb = (1.0 - k) * b + k * r
    
    out = np.stack([nr, ng, nb], axis=2)
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def blend_frames(base: np.ndarray, top: np.ndarray, mode: str, opacity: float) -> np.ndarray:
    """Blend ``top`` over ``base`` (both HxWx3 uint8) with 0..1 opacity."""
    if opacity <= 0.0:
        return base
    a = base.astype(np.float32)
    b = top.astype(np.float32) * float(min(1.0, opacity))
    if mode == "add":
        out = a + b
    elif mode == "screen":
        out = 255.0 - (255.0 - a) * (255.0 - b) / 255.0
    elif mode == "multiply":
        k = float(min(1.0, opacity))
        out = a * ((1.0 - k) + k * top.astype(np.float32) / 255.0)
    else:  # normal
        k = float(min(1.0, opacity))
        out = a * (1.0 - k) + top.astype(np.float32) * k
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


class EnvelopeOpacity:
    """Opacity from trigger hits: 1.0 at each hit, linear decay over ``dur``."""

    def __init__(self, times: Sequence[float], dur: float = 0.1):
        self.times = np.asarray(sorted(times), dtype=float)
        self.dur = max(1e-3, float(dur))

    def __call__(self, t: float) -> float:
        idx = int(np.searchsorted(self.times, t, side="right")) - 1
        if idx < 0:
            return 0.0
        dt = t - float(self.times[idx])
        return max(0.0, 1.0 - dt / self.dur)


class SolidLayer:
    """Constant-color frame; pair with EnvelopeOpacity for drum flashes."""

    def __init__(self, width: int, height: int, color=(255, 255, 255)):
        self._frame = np.empty((height, width, 3), dtype=np.uint8)
        self._frame[:] = np.asarray(color, dtype=np.uint8)

    def frame_at(self, t: float) -> np.ndarray:
        return self._frame


class Compositor:
    """Bottom-up layer stack. First layer is the base canvas."""

    def __init__(self):
        self._layers: List[tuple] = []

    def add(self, source, blend: str = "normal",
            opacity: Optional[Callable[[float], float]] = None) -> None:
        if blend not in BLENDS:
            raise ValueError(f"unknown blend {blend!r}; expected {sorted(BLENDS)}")
        if not self._layers and hasattr(source, "process"):
            raise ValueError("a post-op cannot be the base layer")
        self._layers.append((source, blend, opacity))

    def __len__(self) -> int:
        return len(self._layers)

    def frame_at(self, t: float) -> np.ndarray:
        if not self._layers:
            raise RuntimeError("compositor has no layers")
        src0, _, _ = self._layers[0]
        out = src0.frame_at(t)
        for src, blend, opacity in self._layers[1:]:
            if hasattr(src, "process"):
                # post-op: transforms the composite built so far (a ``bars:`` window gates it)
                op = 1.0 if opacity is None else float(opacity(t))
                if op <= 0.0:
                    continue
                done = src.process(out, t)
                out = done if op >= 1.0 else blend_frames(out, done, "normal", op)
                continue
            op = 1.0 if opacity is None else float(opacity(t))
            if op <= 0.0:
                continue
            # a source that paints itself with what lies below (``frame_at_over``) sees the composite so far
            top = src.frame_at_over(out, t) if hasattr(src, "frame_at_over") else src.frame_at(t)
            if top is not None:
                out = blend_frames(out, top, blend, op)
        return out


def _layer_events(spec: dict, notes: Sequence, onset_loader=None, grid=None) -> list:
    """The events behind a layer trigger, sorted by time: MIDI ``notes``, audio
    ``audio`` onsets or a transcription's ``score`` (same spec shape as composer
    triggers; score events are snapped onto ``grid`` the same way)."""
    if "audio" in spec:
        if onset_loader is None:
            from core.video.composer import _default_onset_loader
            onset_loader = _default_onset_loader
        src = onset_loader(spec)
    elif "score" in spec:
        from core.video.composer import _default_score_loader
        src = _default_score_loader(spec, grid)
        if "notes" in spec:
            wanted = set(spec["notes"])
            src = [n for n in src if n.pitch in wanted]
    else:
        from core.rhythm import select_notes
        src = select_notes(notes, spec)
    min_vel = int(spec.get("min_velocity", 0))
    return sorted((n for n in src if n.velocity >= min_vel), key=lambda n: n.time)


class BarWindow:
    """Opacity from the song's form: 1 inside the listed bars, 0 outside.

        bars: [[2, 21], [36, 51]]     # DAW bar numbers, inclusive
        fade_in: 1                    # bars the layer takes to come up (inside the window)
        fade_out: 2                   # bars it takes to go down (inside the window)

    Works on every layer, post-ops included; the grid answers where the bars are,
    so meter changes are honoured.
    """

    def __init__(self, spec: dict, grid):
        if grid is None:
            raise ValueError("bars: needs a rhythm grid (render with --midi)")
        bars = spec["bars"]
        bars = [bars] if bars and not isinstance(bars[0], (list, tuple)) else bars
        self.spans = []
        for first, last in bars:
            t0, t1 = grid.bar_start(int(first) - 1), grid.bar_start(int(last))
            up = float(spec.get("fade_in", 0)) * (grid.bar_start(int(first)) - t0)
            down = float(spec.get("fade_out", 0)) * (t1 - grid.bar_start(int(last) - 1))
            self.spans.append((t0, t1, up, down))

    def __call__(self, t: float) -> float:
        for t0, t1, up, down in self.spans:
            if t0 <= t < t1:
                k = 1.0
                if up > 0:
                    k = min(k, (t - t0) / up)
                if down > 0:
                    k = min(k, (t1 - t) / down)
                return max(0.0, min(1.0, k))
        return 0.0


def _feature(feats: dict, path: str, default: float = 0.0) -> float:
    """``"stems.piano"`` -> ``feats["stems"]["piano"]`` (a number, or ``default``)."""
    node = feats
    for key in path.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return default


class _PitchFollower:
    """The pitch class of the latest note of a trigger spec (``None`` before the first)."""

    def __init__(self, spec, notes, onset_loader=None, grid=None):
        events = _layer_events(spec, notes, onset_loader, grid) if spec else []
        self.times = np.array([e.time for e in events], dtype=float)
        self.pitches = [int(e.pitch) % 12 for e in events]

    def __call__(self, t: float):
        i = int(np.searchsorted(self.times, t, side="right")) - 1
        return self.pitches[i] if i >= 0 else None


def _layer_hits(spec: dict, notes: Sequence, onset_loader=None, grid=None) -> list:
    """Hit times for a layer trigger (see ``_layer_events``)."""
    return [n.time for n in _layer_events(spec, notes, onset_loader, grid)]


class CellsLayer:
    """8-bit cells (core.cells) rendered per frame and nearest-upscaled.

    ``features_at(t) -> dict`` supplies audio controls (chroma/flux/...);
    ``mitosis`` trigger spec (notes or audio) splits cells on hits.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None):
        from core.cells import CellSystem
        self.sys = CellSystem(
            n_base=int(spec.get("n_base", 12)),
            n_max=int(spec.get("n_max", 48)),
            grid=int(spec.get("resolution", 160)),
            seed=spec.get("seed"),
        )
        self.features_at = features_at
        self.fps = fps
        self.W, self.H = width, height
        mit = spec.get("mitosis") or {}
        self.hits = sorted(_layer_hits(mit, notes, onset_loader)) if mit else []
        self._ptr = 0

    def frame_at(self, t: float) -> np.ndarray:
        while self._ptr < len(self.hits) and self.hits[self._ptr] <= t:
            self.sys.mitosis()
            self._ptr += 1
        controls = (self.features_at(t) if self.features_at else None) or {}
        self.sys.step(1.0 / self.fps, controls)
        img = self.sys.render()
        g = img.shape[0]
        iy = np.arange(self.H) * g // self.H
        ix = np.arange(self.W) * g // self.W
        return img[iy][:, ix]


class JuliaLayer:
    """Audio-driven Julia set (core.fractal), nearest-upscaled.

    ``zoom_pulse`` and ``invert`` trigger specs (notes or audio) drive
    envelopes: kick sucks the camera in, snare flips the palette.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None):
        from core.fractal import JuliaSystem
        self.sys = JuliaSystem(grid=int(spec.get("resolution", 256)),
                               iters=int(spec.get("iters", 48)),
                               aspect=height / width)
        self.features_at = features_at
        self.fps = fps
        self.W, self.H = width, height
        def env(key, default_dur):
            tspec = spec.get(key) or {}
            if not tspec:
                return None
            hits = _layer_hits(tspec, notes, onset_loader)
            return EnvelopeOpacity(hits, tspec.get("envelope", default_dur))
        self._zoom = env("zoom_pulse", 0.25)
        self._invert = env("invert", 0.1)

    def frame_at(self, t: float) -> np.ndarray:
        controls = (self.features_at(t) if self.features_at else None) or {}
        self.sys.step(1.0 / self.fps, controls)
        img = self.sys.render(
            zoom=self._zoom(t) if self._zoom else 0.0,
            invert=self._invert(t) if self._invert else 0.0,
        )
        gh, gw = img.shape[:2]
        iy = np.arange(self.H) * gh // self.H
        ix = np.arange(self.W) * gw // self.W
        return img[iy][:, ix]


class MandelbulbLayer:
    """GPU ray-marched Mandelbulb (core.mandelbulb) at native resolution."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None):
        from core.mandelbulb import MandelbulbSystem
        self.sys = MandelbulbSystem(width, height,
                                    supersample=int(spec.get("supersample", 2)))
        self.features_at = features_at
        self.fps = fps
        tspec = spec.get("zoom_pulse") or {}
        self._zoom = (EnvelopeOpacity(_layer_hits(tspec, notes, onset_loader),
                                      tspec.get("envelope", 0.3))
                      if tspec else None)

    def frame_at(self, t: float) -> np.ndarray:
        controls = (self.features_at(t) if self.features_at else None) or {}
        self.sys.step(1.0 / self.fps, controls)
        return self.sys.render(zoom=self._zoom(t) if self._zoom else 0.0)


class MandelboxLayer:
    """Infinite-zoom KIFS (core.mandelbox); loop period locked to bars."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        from core.mandelbox import MandelboxSystem
        self.sys = MandelboxSystem(width, height,
                                   supersample=int(spec.get("supersample", 2)))
        self.sys.hue_spread = float(spec.get("hue_spread", 0.35))
        self.features_at = features_at
        self.fps = fps
        if grid is not None:
            self.loop_s = float(spec.get("loop_bars", 4)) * grid.bar_duration
        else:
            self.loop_s = float(spec.get("loop_seconds", 8.0))
        tspec = spec.get("zoom_pulse") or {}
        self._pulse = (EnvelopeOpacity(_layer_hits(tspec, notes, onset_loader),
                                       tspec.get("envelope", 0.3))
                       if tspec else None)

    def frame_at(self, t: float) -> np.ndarray:
        controls = (self.features_at(t) if self.features_at else None) or {}
        self.sys.step(1.0 / self.fps, controls)
        return self.sys.render(phase=(t / self.loop_s) % 1.0,
                               pulse=self._pulse(t) if self._pulse else 0.0)


class VeilsLayer:
    """A luminous mist that breathes with the sound (``core/veils``), for music
    with no attacks to follow. CPU only.

        - source: veils
          detail: 0.5            # fraction of the output size the mist is computed at
          seed: 7
          hue: 0.8               # colour when there is no harmony to follow (0..1)
          brightness: 1.0
          harmony:               # optional: the chord sets the colour
            score: input/song/sheetsage
            snap: auto           # as in composer triggers
            bloom: 1.5           # seconds the glow of a chord change lasts
          ripples:               # optional: a ring per hit
            notes: [36]

    Loudness, swell, tremolo, noisiness and percussive come from
    ``features['texture']``; without it the mist just drifts at half light.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        from core.veils import VeilsSystem
        self.fps = fps
        self.features_at = features_at
        self.sys = VeilsSystem(width, height, seed=int(spec.get("seed", 0)),
                               detail=float(spec.get("detail", 0.5)),
                               hue=float(spec.get("hue", 0.8)),
                               brightness=float(spec.get("brightness", 1.0)))
        hspec = spec.get("harmony")
        self._chords = _layer_events(hspec, notes, onset_loader, grid) if hspec else []
        self._chord_times = np.array([e.time for e in self._chords], dtype=float)
        self._bloom = (EnvelopeOpacity(list(self._chord_times), float(hspec.get("bloom", 1.5)))
                       if hspec else None)
        rspec = spec.get("ripples")
        self._ripples = np.array(_layer_hits(rspec, notes, onset_loader, grid), dtype=float) if rspec else np.zeros(0)
        self._smooth: dict = {}
        self._last_t: Optional[float] = None

    @staticmethod
    def hue_of_root(root: int) -> float:
        """Roots a fifth apart sit next to each other on the wheel, so the usual
        chord moves are short glides rather than jumps across the spectrum."""
        return ((int(root) * 7) % 12) / 12.0

    def _lowpass(self, name: str, value: float, dt: float, tau: float) -> float:
        prev = self._smooth.get(name, value)
        prev += (value - prev) * (1 - math.exp(-dt / tau)) if dt > 0 else 0.0
        self._smooth[name] = prev
        return prev

    def frame_at(self, t: float) -> np.ndarray:
        dt = 1.0 / self.fps if self._last_t is None else min(0.1, max(0.0, t - self._last_t))
        self._last_t = t
        tex = (self.features_at(t) or {}).get("texture", {}) if self.features_at else {}
        controls = {
            "loudness":      self._lowpass("loudness", tex.get("loudness", 0.5), dt, 0.20),
            "swell":         self._lowpass("swell", tex.get("swell", 0.0), dt, 0.30),
            "tremolo_depth": self._lowpass("tremolo_depth", tex.get("tremolo_depth", 0.0), dt, 0.50),
            "tremolo_rate":  self._lowpass("tremolo_rate", tex.get("tremolo_rate", 0.0), dt, 0.50),
            "noisiness":     self._lowpass("noisiness", tex.get("noisiness", 0.0), dt, 0.30),
            "percussive":    self._lowpass("percussive", tex.get("percussive", 0.0), dt, 0.30),
        }
        target_hue = None
        if len(self._chords):
            i = int(np.searchsorted(self._chord_times, t, side="right")) - 1
            if i >= 0:
                target_hue = self.hue_of_root(self._chords[i].pitch)
        ages = t - self._ripples[(self._ripples <= t) & (self._ripples > t - 2.0)] if len(self._ripples) else ()
        return self.sys.render(dt, target_hue=target_hue,
                               bloom=self._bloom(t) if self._bloom else 0.0,
                               ripple_ages=ages, **controls)


GESTURES = ("punch", "twist", "sizzle", "ring")


def _parse_gestures(specs, notes, onset_loader=None, grid=None, who="layer") -> list:
    """``hits:`` of a layer -> [(gesture, hit times, envelope seconds)]."""
    out = []
    for g in specs or []:
        if g.get("gesture") not in GESTURES:
            raise ValueError(f"{who}: unknown gesture {g.get('gesture')!r} (use one of {GESTURES})")
        out.append((g["gesture"], np.array(_layer_hits(g, notes, onset_loader, grid), dtype=float),
                    max(1e-3, float(g.get("envelope", 0.1)))))
    return out


def _gestures_at(gestures, t: float) -> dict:
    """Full strength on the frame of the hit, linear decay; a twist alternates sides hit by hit."""
    out = {"punch": 0.0, "twist": 0.0, "sizzle": 0.0, "ring_ages": []}
    for gesture, times, dur in gestures:
        i = int(np.searchsorted(times, t, side="right")) - 1
        if i < 0:
            continue
        if gesture == "ring":
            out["ring_ages"] += [t - h for h in times[max(0, i - 3):i + 1] if t - h < 1.0]
            # a wave is born small at the centre: a short punch gives it an attack to see
            out["punch"] = max(out["punch"], 0.6 * max(0.0, 1.0 - (t - times[i]) / 0.08))
            continue
        env = max(0.0, 1.0 - (t - times[i]) / dur)
        if gesture == "twist":
            env = env if i % 2 == 0 else -env
            out["twist"] = env if abs(env) > abs(out["twist"]) else out["twist"]
        else:
            out[gesture] = max(out[gesture], env)
    return out


class ChladniLayer:
    """Sand on a vibrating plate (``core/chladni``): the chord picks the figure,
    the loudness shakes the sand onto it, a hit throws it off again. CPU only.

        - source: chladni
          grains: 120000
          seed: 3
          root: 10               # chord root before the first change (0 = C … 11 = B)
          modes: {10: [1, 2]}    # optional: your own figure (n, m) for a root
          fineness: [0.8, 1.5]   # figure scale from the darkest to the brightest sound
          pour: 25               # seconds of full-level sound until all the sand is on the plate
          harmony:               # optional: chord changes (as in composer triggers)
            score: input/song/sheetsage
            snap: auto
            bloom: 1.5
          jolt: {notes: [36], envelope: 0.25}    # optional: hits that really scatter the sand (slow)
          hits:                  # optional: one instant gesture per drum voice
            - {notes: [36], gesture: punch, envelope: 0.10}     # punch | twist | sizzle | ring
            - {notes: [40], gesture: sizzle, envelope: 0.09}

    Reads ``features['texture']`` (loudness, swell, harmonic_change, noisiness,
    tremolo, percussive) and ``features['centroid']``.
    """

    CENTROID_RANGE = (0.02, 0.25)     # fraction of Nyquist mapped onto ``fineness`` (log scale)

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        from core.chladni import MODES, ChladniSystem
        self.fps = fps
        self.features_at = features_at
        self._table = {r: MODES[(r * 7) % 12] for r in range(12)}
        self._table.update({int(k) % 12: (float(v[0]), float(v[1]))
                            for k, v in (spec.get("modes") or {}).items()})
        self._root = int(spec.get("root", 0)) % 12
        self._follow_hue = "hue" not in spec
        self.sys = ChladniSystem(width, height, seed=int(spec.get("seed", 0)),
                                 grains=int(spec.get("grains", 120_000)),
                                 modes=self._table[self._root],
                                 hue=float(spec.get("hue", VeilsLayer.hue_of_root(self._root))),
                                 brightness=float(spec.get("brightness", 1.0)))
        lo, hi = spec.get("fineness", (1.0, 1.0))
        self._fineness = (float(lo), float(hi))
        self._pour_s = float(spec.get("pour", 0.0))
        self._poured = 0.0 if self._pour_s > 0 else 1.0
        hspec = spec.get("harmony")
        self._chords = _layer_events(hspec, notes, onset_loader, grid) if hspec else []
        self._chord_times = np.array([e.time for e in self._chords], dtype=float)
        self._bloom = (EnvelopeOpacity(list(self._chord_times), float(hspec.get("bloom", 1.5)))
                       if hspec else None)
        jspec = spec.get("jolt")
        self._jolt = (EnvelopeOpacity(_layer_hits(jspec, notes, onset_loader, grid),
                                      float(jspec.get("envelope", 0.25))) if jspec else None)
        self._gestures = _parse_gestures(spec.get("hits"), notes, onset_loader, grid, who="chladni")
        self._level_from = spec.get("level_from")          # e.g. "stems.piano" instead of the mix loudness
        self._level_gain = float(spec.get("level_gain", 1.0))
        self._smooth: dict = {}
        self._last_t: Optional[float] = None

    def _lowpass(self, name: str, value: float, dt: float, tau: float) -> float:
        prev = self._smooth.get(name, value)
        prev += (value - prev) * (1 - math.exp(-dt / tau)) if dt > 0 else 0.0
        self._smooth[name] = prev
        return prev

    def frame_at(self, t: float) -> np.ndarray:
        dt = 1.0 / self.fps if self._last_t is None else min(0.1, max(0.0, t - self._last_t))
        self._last_t = t
        feats = (self.features_at(t) or {}) if self.features_at else {}
        tex = feats.get("texture", {})
        loud = self._lowpass("loudness", tex.get("loudness", 0.5), dt, 0.20)
        perc = self._lowpass("percussive", tex.get("percussive", 0.0), dt, 0.15)
        # loudness arrives on a dB scale; the useful part of it is the top two thirds
        level = float(np.clip((loud - 0.25) / 0.65, 0.0, 1.0)) ** 1.3
        if self._level_from:
            level = float(np.clip(self._level_gain * self._lowpass(
                "level_from", _feature(feats, self._level_from), dt, 0.25), 0.0, 1.0))
        self._poured = min(1.0, self._poured + dt * level / self._pour_s) if self._pour_s > 0 else 1.0

        c_lo, c_hi = self.CENTROID_RANGE
        cen = float(np.clip(feats.get("centroid", c_lo) or c_lo, c_lo, c_hi))
        bright = self._lowpass("bright", math.log(cen / c_lo) / math.log(c_hi / c_lo), dt, 2.5)
        fineness = self._fineness[0] + (self._fineness[1] - self._fineness[0]) * bright

        root = self._root
        if len(self._chords):
            i = int(np.searchsorted(self._chord_times, t, side="right")) - 1
            if i >= 0:
                root = int(self._chords[i].pitch) % 12
        return self.sys.render(
            dt, level=min(1.0, level + 0.3 * perc), modes=self._table[root], fineness=fineness,
            harmonic_change=self._lowpass("harmonic_change", tex.get("harmonic_change", 0.0), dt, 0.5),
            swell=self._lowpass("swell", tex.get("swell", 0.0), dt, 0.30),
            jolt=self._jolt(t) if self._jolt else 0.0,
            noisiness=self._lowpass("noisiness", tex.get("noisiness", 0.0), dt, 0.30),
            tremolo_depth=self._lowpass("tremolo_depth", tex.get("tremolo_depth", 0.0), dt, 0.50),
            tremolo_rate=self._lowpass("tremolo_rate", tex.get("tremolo_rate", 0.0), dt, 0.50),
            target_hue=VeilsLayer.hue_of_root(root) if self._follow_hue else None,
            bloom=self._bloom(t) if self._bloom else 0.0, pour=self._poured,
            **_gestures_at(self._gestures, t))


class FlameLayer:
    """A fractal flame (``core/flame``) turned by the music. GPU (moderngl compute);
    falls back to a grainier CPU render when no 4.3 context can be had.

        - source: flame
          genome: {random: 5}    # or {xforms: [{affine: [a,b,c,d,e,f], variations: {swirl: 1}, ...}]}
          symmetry: 1            # rotational symmetry order
          samples: 6             # millions of points per frame
          supersample: 2
          backend: auto          # auto | gpu | cpu
          spin: 0.02             # turns per second the xforms rotate at full level
          pose: 0.25             # how far a chord change re-poses the figure (0 = colour only)
          root: 10               # chord root before the first change
          harmony: {score: input/song/sheetsage, snap: auto, bloom: 1.5}
          hits:                  # instant gestures, as in ``chladni``
            - {notes: [36], gesture: punch, envelope: 0.10}
          paint: under           # colour every point with the layers below it (a video):
          paint_mix: 0.9         # the figure becomes a kaleidoscope of that picture
          paint_gain: 5          # undo a dimming solid below: the mandala shows the video at full colour

    Loudness drives the light and the speed, swell the zoom, harmonic change makes
    the figure wander; the chord sets the palette and the pose.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        from core.flame import FlameSystem, Genome
        self.fps = fps
        self.features_at = features_at
        genome = Genome.from_spec(spec.get("genome") or {"random": 5}).with_symmetry(int(spec.get("symmetry", 1)))
        self.sys = FlameSystem(width, height, genome, seed=int(spec.get("seed", 0)),
                               samples=float(spec.get("samples", 6.0)),
                               supersample=int(spec.get("supersample", 2)),
                               backend=str(spec.get("backend", "auto")),
                               exposure=float(spec.get("exposure", 3.0)),
                               white=float(spec.get("white", 40.0)), gamma=float(spec.get("gamma", 2.4)),
                               centered=int(spec.get("symmetry", 1)) > 1)
        self._spin = float(spec.get("spin", 0.02))
        self._pose_amount = float(spec.get("pose", 0.25))
        self._root = int(spec.get("root", 0)) % 12
        hspec = spec.get("harmony")
        self._chords = _layer_events(hspec, notes, onset_loader, grid) if hspec else []
        self._chord_times = np.array([e.time for e in self._chords], dtype=float)
        self._bloom = (EnvelopeOpacity(list(self._chord_times), float(hspec.get("bloom", 1.5)))
                       if hspec else None)
        self._gestures = _parse_gestures(spec.get("hits"), notes, onset_loader, grid, who="flame")
        self._hue_notes = _PitchFollower(spec.get("hue_notes"), notes, onset_loader, grid)
        if spec.get("paint") not in (None, "under"):
            raise ValueError(f"flame: paint must be 'under', got {spec.get('paint')!r}")
        self._paint_mix = float(spec.get("paint_mix", 1.0)) if spec.get("paint") == "under" else 0.0
        self._paint_gain = float(spec.get("paint_gain", 1.0))
        self._under = None
        self._pose_notes = _PitchFollower(spec.get("pose_notes"), notes, onset_loader, grid)
        self._glide = float(spec.get("glide", 1.2))
        self._level_from = spec.get("level_from")
        self._level_gain = float(spec.get("level_gain", 1.0))
        self._smooth: dict = {}
        self._last_t: Optional[float] = None
        self.phase = 0.0
        self.breath = 0.0
        self.hue = float(spec.get("hue", VeilsLayer.hue_of_root(self._root)))
        self._follow_hue = "hue" not in spec
        self.pose = self._pose_of(self._root)

    def _pose_of(self, root: int) -> float:
        return 2 * math.pi * self._pose_amount * VeilsLayer.hue_of_root(root)

    def _lowpass(self, name: str, value: float, dt: float, tau: float) -> float:
        prev = self._smooth.get(name, value)
        prev += (value - prev) * (1 - math.exp(-dt / tau)) if dt > 0 else 0.0
        self._smooth[name] = prev
        return prev

    def frame_at_over(self, under: np.ndarray, t: float) -> np.ndarray:
        self._under = under if self._paint_mix > 0 else None
        return self.frame_at(t)

    def frame_at(self, t: float) -> np.ndarray:
        dt = 1.0 / self.fps if self._last_t is None else min(0.1, max(0.0, t - self._last_t))
        self._last_t = t
        feats = (self.features_at(t) or {}) if self.features_at else {}
        tex = feats.get("texture", {})
        loud = self._lowpass("loudness", tex.get("loudness", 0.5), dt, 0.20)
        level = float(np.clip((loud - 0.25) / 0.65, 0.0, 1.0)) ** 1.3
        if self._level_from:
            level = float(np.clip(self._level_gain * self._lowpass(
                "level_from", _feature(feats, self._level_from), dt, 0.25), 0.0, 1.0))
        change = self._lowpass("harmonic_change", tex.get("harmonic_change", 0.0), dt, 0.5)
        swell = self._lowpass("swell", tex.get("swell", 0.0), dt, 0.30)
        self.phase += dt * (2 * math.pi * self._spin * (0.25 + 0.75 * level) + 0.35 * change)
        self.breath += dt * (0.35 * swell - self.breath / 6.0)
        self.breath = float(np.clip(self.breath, -0.4, 0.4))

        root = self._root
        if len(self._chords):
            i = int(np.searchsorted(self._chord_times, t, side="right")) - 1
            if i >= 0:
                root = int(self._chords[i].pitch) % 12
        glide = 1 - math.exp(-dt / self._glide)
        pose_root = self._pose_notes(t)                    # a track's notes beat the chord
        hue_root = self._hue_notes(t)
        self.pose += (self._pose_of(root if pose_root is None else pose_root) - self.pose) * glide
        if self._follow_hue or hue_root is not None:
            root = root if hue_root is None else hue_root
            gap = ((VeilsLayer.hue_of_root(root) - self.hue + 0.5) % 1.0) - 0.5
            self.hue = (self.hue + gap * glide) % 1.0
        return self.sys.render(dt, phase=self.phase, pose=self.pose, hue=self.hue, level=level,
                               zoom=math.exp(self.breath), bloom=self._bloom(t) if self._bloom else 0.0,
                               paint=self._under, paint_mix=self._paint_mix, paint_gain=self._paint_gain,
                               **_gestures_at(self._gestures, t))


class EyesLayer:
    """A Voronoi wall of eyeballs (``core/eyes``, GPU) that follows a gaze.

        - source: eyes
          rows: 5                # eyes down the height (about 9 across at 16:9)
          seed: 3
          supersample: 2
          depth: 0.9             # how far in front of the wall the gaze point floats (bigger = eyes look straighter)
          hue: 0.55              # iris colour; a ``harmony:`` block lets the chord choose it,
          iris_hues: [0.08, 0.6] # ... mapped onto real iris colours: amber -> green -> blue
          pupil_from: texture.loudness   # feature that dilates the pupils (0..1)
          wander: 0.15           # how far the gaze drifts on its own (frame height units)
          hits:                  # gestures: saccade (gaze jumps) | dilate | blink
            - {track: snare, gesture: saccade}
            - {track: kick, gesture: dilate, envelope: 0.25}
            - {track: lead, gesture: blink, envelope: 0.20}
    """

    GESTURES = ("saccade", "dilate", "blink")

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        from core.eyes import EyesSystem
        self.fps = fps
        self.features_at = features_at
        self.sys = EyesSystem(width, height, rows=float(spec.get("rows", 5)), seed=int(spec.get("seed", 0)),
                              supersample=int(spec.get("supersample", 2)), depth=float(spec.get("depth", 0.9)))
        self.aspect = width / height
        self._pupil_from = spec.get("pupil_from", "texture.loudness")
        lo, hi = spec.get("iris_hues", (0.08, 0.6))
        self._iris_hues = (float(lo), float(hi))
        self._wander = float(spec.get("wander", 0.15))
        self._rng = np.random.default_rng(int(spec.get("seed", 0)))
        hspec = spec.get("harmony")
        self._chords = _layer_events(hspec, notes, onset_loader, grid) if hspec else []
        self._chord_times = np.array([e.time for e in self._chords], dtype=float)
        self._gestures = []
        for g in spec.get("hits") or []:
            if g.get("gesture") not in self.GESTURES:
                raise ValueError(f"eyes: unknown gesture {g.get('gesture')!r} (use one of {self.GESTURES})")
            self._gestures.append((g["gesture"], np.array(_layer_hits(g, notes, onset_loader, grid), dtype=float),
                                   max(1e-3, float(g.get("envelope", 0.2)))))
        sacc = [t for g, t, _ in self._gestures if g == "saccade"]
        self._saccades = np.sort(np.concatenate(sacc)) if sacc else np.zeros(0)
        self._n_sacc = 0
        self._target = np.zeros(2)
        self._gaze = np.zeros(2)
        self._smooth: dict = {}
        self._last_t: Optional[float] = None
        self.hue = float(spec.get("hue", 0.55))

    def _lowpass(self, name, value, dt, tau):
        prev = self._smooth.get(name, value)
        prev += (value - prev) * (1 - math.exp(-dt / tau)) if dt > 0 else 0.0
        self._smooth[name] = prev
        return prev

    def frame_at(self, t: float) -> np.ndarray:
        dt = 1.0 / self.fps if self._last_t is None else min(0.1, max(0.0, t - self._last_t))
        self._last_t = t
        feats = (self.features_at(t) or {}) if self.features_at else {}
        pupil = self._lowpass("pupil", _feature(feats, self._pupil_from, 0.3), dt, 0.15)
        dilate = blink = 0.0
        for gesture, times, dur in self._gestures:
            i = int(np.searchsorted(times, t, side="right")) - 1
            if i < 0:
                continue
            age = t - times[i]
            if gesture == "dilate":
                dilate = max(dilate, max(0.0, 1.0 - age / dur))
            elif gesture == "blink" and age < dur:
                blink = max(blink, math.sin(math.pi * age / dur))
        n_sacc = int(np.searchsorted(self._saccades, t, side="right"))
        if n_sacc != self._n_sacc:                                        # the gaze jumps somewhere new
            self._n_sacc = n_sacc
            self._target = self._rng.uniform(-1, 1, 2) * np.array([self.aspect * 0.8, 0.7])
        # slow drift around the target
        drift = self._wander * np.array([math.sin(0.37 * t), math.sin(0.23 * t + 1.0)])
        self._gaze += (self._target + drift - self._gaze) * (1 - math.exp(-dt / 0.06))
        if len(self._chords):
            i = int(np.searchsorted(self._chord_times, t, side="right")) - 1
            if i >= 0:
                lo, hi = self._iris_hues
                target_hue = lo + (hi - lo) * VeilsLayer.hue_of_root(self._chords[i].pitch)
                self.hue += (target_hue - self.hue) * (1 - math.exp(-dt / 1.2))
        return self.sys.render(t, gaze=tuple(self._gaze), saccade=n_sacc,
                               pupil=min(1.0, 0.15 + 0.5 * pupil + 0.6 * dilate), blink=blink, hue=self.hue)


class MouthsLayer:
    """A Voronoi wall of singing mouths (``core/mouths``, GPU), the companion of ``eyes``.

        - source: mouths
          rows: 5
          seed: 3
          supersample: 2
          voice: stems_output/<hash>/demucs/<song>_(Vocals)_htdemucs_6s.flac
                                 # the vocal stem: its level and F1 drop the jaw, its F2
                                 # spreads the lips ("i") or rounds them ("u"); see core/voice
          jaw_from: stems.vocals # without ``voice:``, a feature that opens the mouths (0..1)
          harmony: {score: input/song/sheetsage, snap: auto}   # the chord bruises the lips
          hits:                  # gestures: clench (teeth bared) | tremble | gasp (jaw drops)
            - {track: snare, gesture: clench, envelope: 0.2}
            - {track: kick, gesture: tremble, envelope: 0.15}
    """

    GESTURES = ("clench", "tremble", "gasp")

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        from core.mouths import MouthsSystem
        self.fps = fps
        self.features_at = features_at
        self.sys = MouthsSystem(width, height, rows=float(spec.get("rows", 5)), seed=int(spec.get("seed", 0)),
                                supersample=int(spec.get("supersample", 2)))
        self._voice = None
        if spec.get("voice"):
            from core.voice import read_voice
            self._voice = read_voice(spec["voice"])
        self._jaw_from = spec.get("jaw_from", "stems.vocals")
        hspec = spec.get("harmony")
        self._chords = _layer_events(hspec, notes, onset_loader, grid) if hspec else []
        self._chord_times = np.array([e.time for e in self._chords], dtype=float)
        self._gestures = []
        for g in spec.get("hits") or []:
            if g.get("gesture") not in self.GESTURES:
                raise ValueError(f"mouths: unknown gesture {g.get('gesture')!r} (use one of {self.GESTURES})")
            self._gestures.append((g["gesture"], np.array(_layer_hits(g, notes, onset_loader, grid), dtype=float),
                                   max(1e-3, float(g.get("envelope", 0.2)))))
        self._smooth: dict = {}
        self._last_t: Optional[float] = None
        self.tint = float(spec.get("tint", 0.0))

    def _lowpass(self, name, value, dt, tau):
        prev = self._smooth.get(name, value)
        prev += (value - prev) * (1 - math.exp(-dt / tau)) if dt > 0 else 0.0
        self._smooth[name] = prev
        return prev

    def frame_at(self, t: float) -> np.ndarray:
        dt = 1.0 / self.fps if self._last_t is None else min(0.1, max(0.0, t - self._last_t))
        self._last_t = t
        if self._voice is not None:
            jaw, spread, _ = self._voice.at(t)
        else:
            feats = (self.features_at(t) or {}) if self.features_at else {}
            jaw, spread = _feature(feats, self._jaw_from, 0.0), 0.0
        # the jaw drops fast and closes a little slower, as a real one does
        tau = 0.025 if jaw > self._smooth.get("jaw", 0.0) else 0.05
        jaw = self._lowpass("jaw", jaw, dt, tau)
        spread = self._lowpass("spread", spread, dt, 0.06)
        clench = tremble = gasp = 0.0
        for gesture, times, dur in self._gestures:
            i = int(np.searchsorted(times, t, side="right")) - 1
            if i < 0:
                continue
            k = max(0.0, 1.0 - (t - times[i]) / dur)
            if gesture == "clench":
                clench = max(clench, k)
            elif gesture == "tremble":
                tremble = max(tremble, k)
            else:
                gasp = max(gasp, math.sin(math.pi * min(1.0, (t - times[i]) / dur)))
        if len(self._chords):
            i = int(np.searchsorted(self._chord_times, t, side="right")) - 1
            if i >= 0:
                target = VeilsLayer.hue_of_root(self._chords[i].pitch)
                self.tint += (target - self.tint) * (1 - math.exp(-dt / 1.2))
        return self.sys.render(t, jaw=max(jaw, 0.9 * gasp), spread=spread * (1.0 - gasp), clench=clench,
                               tremble=tremble, tint=self.tint)


class OrbitersLayer:
    """Nested orbiting mandalas (layer source).

    Orbit speed of parents tied to bar_phase + bass, satellites tied to beat_phase + flux.
    Radii are modulated by centroid and flux.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None, grid=None):
        self.W, self.H = width, height
        self.fps = fps
        self.features_at = features_at
        self.grid = grid
        self.seed = spec.get("seed", 7)
        self.rng = np.random.default_rng(self.seed)
        self.n_parents = int(spec.get("n_parents", 6))
        self.n_satellites = int(spec.get("n_satellites", 4))
        self.parent_radius_spec = spec.get("parent_radius_spec", "centroid")
        self.satellite_radius_spec = spec.get("satellite_radius_spec", "flux")
        self.grid_res = int(spec.get("resolution", 160))
        
        # Base scale values (as fractions of grid size)
        self.parent_r_base = float(spec.get("parent_r_base", 0.22))
        self.sat_r_base = float(spec.get("sat_r_base", 0.08))
        
        # Angle accumulators for audio-reactive speed
        self.p_angle_accum = 0.0
        self.s_angle_accum = 0.0

    def frame_at(self, t: float) -> np.ndarray:
        g = self.grid_res
        img = np.zeros((g, g, 3), dtype=np.uint8)
        controls = (self.features_at(t) if self.features_at else None) or {}
        
        # Feature inputs
        p_mod = float(controls.get(self.parent_radius_spec, 0.5) or 0.5)
        s_mod = float(controls.get(self.satellite_radius_spec, 0.5) or 0.5)
        
        # Dynamic rotation speed based on bass & flux
        dt = 1.0 / self.fps
        bass = float(controls.get("bass", controls.get("bass_energy", 0.5)) or 0.5)
        flux = float(controls.get("flux", 0.5) or 0.5)
        
        self.p_angle_accum += dt * (1.0 + 8.0 * bass)
        self.s_angle_accum += dt * (2.0 + 12.0 * flux)
        
        # Chroma feature for colors (fallback to all 0.5 if not present)
        chroma = controls.get("chroma", [0.5] * 12)
        if len(chroma) < 12:
            chroma = [0.5] * 12
            
        cx_c, cy_c = g / 2.0, g / 2.0
        
        from core.cells import PALETTE
        
        # Draw connections and nodes
        for i in range(self.n_parents):
            # Map parent to a chroma hue
            p_bin = (i * 12 // self.n_parents) % 12
            p_energy = float(chroma[p_bin])
            
            # Parent color based on chroma
            col_float = PALETTE[p_bin] * (0.35 + 0.65 * p_energy)
            color = tuple(int(c) for c in col_float)
            
            # Parent angle (accumulated + offset) and radius (scaled up modulation)
            p_angle = self.p_angle_accum + (2.0 * math.pi * i / self.n_parents)
            p_r = g * (self.parent_r_base + 0.35 * p_mod)
            
            px = cx_c + p_r * math.cos(p_angle)
            py = cy_c + p_r * math.sin(p_angle)
            
            # Draw line from center to parent (dimmed color)
            draw_line_numpy(img, cx_c, cy_c, px, py, tuple(int(c * 0.25) for c in color))
            
            # Draw satellites
            for j in range(self.n_satellites):
                s_angle = -self.s_angle_accum + (2.0 * math.pi * j / self.n_satellites)
                s_r = g * (self.sat_r_base + 0.15 * s_mod)
                
                sx = px + s_r * math.cos(s_angle)
                sy = py + s_r * math.sin(s_angle)
                
                # Draw line from parent to satellite
                draw_line_numpy(img, px, py, sx, sy, tuple(int(c * 0.45) for c in color))
                
                # Draw satellite circle
                draw_circle_numpy(img, sx, sy, g * 0.02, color)
                
            # Draw parent circle
            draw_circle_numpy(img, px, py, g * 0.035, tuple(int(c * 0.8) for c in color))
            
        # Draw central hub circle
        draw_circle_numpy(img, cx_c, cy_c, g * 0.05, (255, 255, 255))
        
        # Nearest-neighbor upscale to H x W
        iy = np.arange(self.H) * g // self.H
        ix = np.arange(self.W) * g // self.W
        return img[iy][:, ix]


class ParticlesLayer:
    """Unifies ParticleSystem and emits particles from a central rotating mutant polygon.

    Emissions are triggered on MIDI or audio onset triggers. Physics forces (bass, flux)
    continuously deform the simulation.
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, features_at=None, onset_loader=None):
        self.W, self.H = width, height
        self.fps = fps
        self.features_at = features_at
        self.seed = spec.get("seed", 42)
        self.rng = np.random.default_rng(self.seed)
        
        self.grid_res = int(spec.get("resolution", 200))
        self.n_particles = int(spec.get("n_particles", 10000))
        
        # Core particle arrays
        self.pos = self.rng.random((self.n_particles, 2), dtype=np.float32)
        self.vel = (self.rng.random((self.n_particles, 2), dtype=np.float32) - 0.5) * 0.02
        self.life = self.rng.random(self.n_particles, dtype=np.float32)
        self.life_max = self.rng.random(self.n_particles, dtype=np.float32) * 1.5 + 0.5
        self.hue_off = self.rng.random(self.n_particles, dtype=np.float32)
        
        # Central emitter polygon
        self.n_vertices = int(spec.get("n_vertices", 5))
        self.poly_r = float(spec.get("emitter_radius", 0.2)) # fraction of grid
        self.rot_speed = float(spec.get("rotation_speed", 1.5)) # rad/s
        
        # Trigger config
        tspec = spec.get("triggers", {}).get("burst") or spec.get("mitosis") or {}
        self.hits = sorted(_layer_hits(tspec, notes, onset_loader)) if tspec else []
        self.ptr_trigger = 0
        self.burst_qty = int(tspec.get("quantity", 250))
        
        # Current index to cycle through for respawning particles
        self.respawn_idx = 0

    def frame_at(self, t: float) -> np.ndarray:
        g = self.grid_res
        dt = 1.0 / self.fps
        
        # Process trigger hits
        bursts_to_trigger = 0
        while self.ptr_trigger < len(self.hits) and self.hits[self.ptr_trigger] <= t:
            bursts_to_trigger += 1
            self.ptr_trigger += 1
            
        controls = (self.features_at(t) if self.features_at else None) or {}
        
        # Rotate the emitter polygon
        theta = t * self.rot_speed
        
        # Calculate polygon vertices (normalized 0..1 coordinates)
        vertices = []
        for i in range(self.n_vertices):
            angle = theta + 2.0 * math.pi * i / self.n_vertices
            vx = 0.5 + self.poly_r * math.cos(angle)
            vy = 0.5 + self.poly_r * math.sin(angle)
            vertices.append((vx, vy, angle))
            
        # If triggers occurred, respawn particles at vertices and shoot them out
        if bursts_to_trigger > 0 and self.n_vertices > 0:
            qty_per_vertex = max(1, self.burst_qty // self.n_vertices)
            for vx, vy, angle in vertices:
                for _ in range(qty_per_vertex):
                    idx = self.respawn_idx
                    self.pos[idx] = [vx, vy]
                    # Velocity pointing outwards + random spray
                    spray = self.rng.normal(0, 0.05, 2)
                    vel_dir = np.array([math.cos(angle), math.sin(angle)], dtype=np.float32)
                    self.vel[idx] = vel_dir * self.rng.uniform(0.12, 0.28) + spray
                    self.life[idx] = 0.0
                    self.life_max[idx] = self.rng.uniform(0.8, 2.0)
                    self.respawn_idx = (self.respawn_idx + 1) % self.n_particles
                    
        # Update physics
        center = np.array([0.5, 0.5], dtype=np.float32)
        diff = self.pos - center
        dist = np.linalg.norm(diff, axis=1, keepdims=True).clip(1e-6)
        radial = diff / dist
        
        # Audio-reactive continuous forces
        bass = float(controls.get("bass", controls.get("bass_energy", 0.0)) or 0.0)
        flux = float(controls.get("flux", 0.0) or 0.0)
        
        # Bass -> radial explosion from center
        self.vel += radial * (bass * 0.02)
        
        # Flux -> turbulence (noise)
        if flux > 0.05:
            noise = self.rng.random((self.n_particles, 2), dtype=np.float32) - 0.5
            self.vel += noise * (flux * 0.008)
            
        # Attract to center attractor
        to_center = center - self.pos
        self.vel += to_center * 0.02 * dt
        
        # Drag
        self.vel *= 0.90
        
        # Move
        self.pos += self.vel * (dt * 60.0)
        self.pos %= 1.0  # wrap at edges
        
        # Age
        self.life += dt / self.life_max
        
        # Respawn naturally dead particles randomly
        dead = self.life >= 1.0
        n_dead = int(dead.sum())
        if n_dead > 0:
            self.pos[dead] = self.rng.random((n_dead, 2), dtype=np.float32)
            self.vel[dead] = (self.rng.random((n_dead, 2), dtype=np.float32) - 0.5) * 0.01
            self.life[dead] = 0.0
            
        # Draw particles additive buffer
        px = (self.pos[:, 0] * (g - 1)).astype(np.int32)
        py = (self.pos[:, 1] * (g - 1)).astype(np.int32)
        brightness = np.sin(self.life * np.pi).astype(np.float32)
        
        centroid = float(controls.get("centroid", 0.5) or 0.5)
        bar_phase = float(t / 4.0) % 1.0
        
        base_hue = centroid * 0.67
        hue = (base_hue + self.hue_off * 0.25 + bar_phase * 0.08) % 1.0
        
        from core.particles import _hsv_to_rgb_vec
        r_f, g_f, b_f = _hsv_to_rgb_vec(hue, 0.85, brightness)
        
        buf = np.zeros((g, g, 3), dtype=np.float32)
        np.add.at(buf[:, :, 0], (py, px), r_f)
        np.add.at(buf[:, :, 1], (py, px), g_f)
        np.add.at(buf[:, :, 2], (py, px), b_f)
        
        np.clip(buf, 0.0, 1.0, out=buf)
        buf_u8 = (buf * 255).astype(np.uint8)
        
        # Draw the central polygon outline on the buffer
        poly_pts = [(int(vx * g), int(vy * g)) for vx, vy, _ in vertices]
        for i in range(len(poly_pts)):
            x0, y0 = poly_pts[i]
            x1, y1 = poly_pts[(i + 1) % len(poly_pts)]
            draw_line_numpy(buf_u8, x0, y0, x1, y1, (100, 200, 255))
            
        # Nearest-neighbor upscale to output resolution
        iy = np.arange(self.H) * g // self.H
        ix = np.arange(self.W) * g // self.W
        return buf_u8[iy][:, ix]


class FeedbackLayer:
    """Post-op: accumulates feedback by blending the previous frame,

    slightly scaled and rotated, under the current frame.
    """

    def __init__(self, spec: dict, notes: Sequence, fps: int,
                 features_at=None, onset_loader=None):
        self.fps = fps
        self.features_at = features_at
        self.decay = float(spec.get("decay", 0.85))
        self.max_scale = float(spec.get("max_scale", 1.05))
        self.base_scale = float(spec.get("base_scale", 0.96))
        self.max_rotate = float(spec.get("max_rotate", 0.1)) # rad
        self.hue_shift = float(spec.get("hue_shift", 0.0))   # 0..1 per pass: the trail walks round the colour wheel
        
        # Trigger for zoom (e.g. kick)
        zspec = spec.get("zoom_pulse")
        self._zoom_env = (EnvelopeOpacity(_layer_hits(zspec, notes, onset_loader),
                                          zspec.get("envelope", 0.25))
                          if zspec else None)
                          
        # Smooth state for rotation (centroid lowpass)
        self.rot_smooth = 0.0
        self._prev = None

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self.decay <= 0.0:
            return frame
            
        if self._prev is None or self._prev.shape != frame.shape:
            self._prev = frame.copy()
            return frame
            
        # Compute scale
        if self._zoom_env is not None:
            env_val = self._zoom_env(t)
            scale = self.base_scale + (self.max_scale - self.base_scale) * env_val
        else:
            scale = self.base_scale
            
        # Compute rotation (angle)
        dt = 1.0 / self.fps
        x = 0.5
        if self.features_at is not None:
            feats = self.features_at(t) or {}
            x = float(feats.get("centroid", 0.5) or 0.5)
            
        tau = 0.15
        self.rot_smooth += (x - self.rot_smooth) * (1.0 - math.exp(-dt / tau))
        angle = self.max_rotate * (self.rot_smooth - 0.5)
        
        # Scale and rotate self._prev
        warped = scale_rotate_image(self._prev, scale, angle)
        if self.hue_shift > 0.0:
            warped = hue_shift_channels(warped, self.hue_shift)
        
        # Blend: current frame composed over warped history using screen mode
        out = blend_frames(frame, warped, "screen", self.decay)
        
        # Save output for next iteration
        self._prev = out.copy()
        return out


class GrainLayer:
    """Post-op: film grain, to take the synthetic edge off a render.

        - source: grain
          amount: 0.06           # strength (0..1): standard deviation of the noise, in units of full white
          size: 1.2              # grain radius in output pixels (soft-edged, never square)
          oversample: 2          # the grain is made at this multiple of the output size and averaged down
          color: 0.25            # 0 = monochrome grain, 1 = independent per channel
          seed: 1
          trigger: {track: kick, envelope: 0.2}   # optional: an extra burst on hits
          burst: 0.15            # how much a hit adds
          bars: [[6, 14]]        # optional: only inside these bars (works on every layer)

    The noise is drawn at ``oversample`` times the frame, blurred to the grain size
    and area-averaged down, so each grain is a soft blob rather than a block. It is
    strongest in the mid-tones and fades in the blacks and the whites, as on film.
    Each frame's noise is seeded by (seed, frame), so a render replays.
    """

    def __init__(self, spec: dict, notes: Sequence, fps: int, features_at=None, onset_loader=None, grid=None):
        self.fps = fps
        self.amount = float(spec.get("amount", 0.06))
        self.size = max(0.3, float(spec.get("size", 1.2)))
        self.over = max(1, int(spec.get("oversample", 2)))
        self.color = float(np.clip(spec.get("color", 0.25), 0.0, 1.0))
        self.seed = int(spec.get("seed", 1))
        self.burst = float(spec.get("burst", 0.15))
        tspec = spec.get("trigger")
        self._burst = (EnvelopeOpacity(_layer_hits(tspec, notes, onset_loader, grid), float(tspec.get("envelope", 0.2)))
                       if tspec else None)

    BANK = 12                # grain planes made once; a frame draws one at a random offset

    def _plane(self, H: int, W: int, rng) -> np.ndarray:
        """One unit-variance grain plane, (H, W, 3) float32: oversampled, softened, averaged down."""
        h, w = H * self.over, W * self.over
        planes = 3 if self.color > 0 else 1
        raw = rng.standard_normal((h, w, planes), dtype=np.float32)
        sigma = 0.5 * self.size * self.over
        try:
            import cv2
            soft = cv2.GaussianBlur(raw, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
            soft = soft.reshape(h, w, planes)
        except ImportError:
            from scipy.ndimage import gaussian_filter
            soft = np.stack([gaussian_filter(raw[..., i], sigma, mode="wrap") for i in range(planes)], axis=2)
        if self.over > 1:                                                   # area average down
            soft = soft.reshape(H, self.over, W, self.over, planes).mean(axis=(1, 3))
        soft /= max(1e-6, float(soft.std()))
        if planes == 3:
            mono = soft.mean(axis=2, keepdims=True)
            soft = mono * (1 - self.color) + soft * self.color
            soft /= max(1e-6, float(soft.std()))
            return np.ascontiguousarray(soft)
        return np.ascontiguousarray(np.repeat(soft, 3, axis=2))

    def _noise(self, H: int, W: int, t: float) -> np.ndarray:
        if getattr(self, "_bank_shape", None) != (H, W):
            rng = np.random.default_rng([self.seed, 0xB4A8])
            self._bank = [self._plane(H, W, rng) for _ in range(self.BANK)]
            self._bank_shape = (H, W)
        rng = np.random.default_rng([self.seed, int(round(t * self.fps))])
        plane = self._bank[int(rng.integers(self.BANK))]
        return np.roll(plane, (int(rng.integers(H)), int(rng.integers(W))), axis=(0, 1))

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        amount = self.amount + (self.burst * self._burst(t) if self._burst else 0.0)
        if amount <= 0.0:
            return frame
        H, W = frame.shape[:2]
        f = frame.astype(np.float32)
        lum = f.mean(axis=2, keepdims=True) / 255.0
        weight = 1.0 - (2.0 * lum - 1.0) ** 2                              # mid-tones grain most
        out = f + self._noise(H, W, t) * (amount * 255.0) * (0.25 + 0.75 * weight)
        return np.clip(out, 0.0, 255.0).astype(np.uint8)


class RgbNoiseLayer:
    """Post-op: RGB noise the way GIMP/GEGL's ``noise-rgb`` does it — per pixel, no blur.

        - source: rgb_noise
          amount: 0.2            # 0..1; or per channel: red / green / blue
          correlated: true       # true: multiplicative (out = in * (1 + amount * n)) — noise scales
                                 #       with the pixel's light, blacks stay clean; false: additive
          independent: true      # a different draw per channel (colour noise) or one for all three
          linear: true           # work in linear light (sRGB decoded), as GEGL does
          gaussian: true         # gaussian or uniform draws
          seed: 0
          trigger: {track: kick, envelope: 0.2}   # optional: an extra burst on hits
          burst: 0.2
          bars: [[6, 14]]        # optional window

    Each frame's noise is seeded by (seed, frame), so a render replays.
    """

    def __init__(self, spec: dict, notes: Sequence, fps: int, features_at=None, onset_loader=None, grid=None):
        self.fps = fps
        amount = float(spec.get("amount", 0.2))
        self.amount = np.array([float(spec.get(k, amount)) for k in ("red", "green", "blue")], dtype=np.float32)
        self.correlated = bool(spec.get("correlated", True))
        self.independent = bool(spec.get("independent", True))
        self.linear = bool(spec.get("linear", True))
        self.gaussian = bool(spec.get("gaussian", True))
        self.seed = int(spec.get("seed", 0))
        self.burst = float(spec.get("burst", 0.2))
        tspec = spec.get("trigger")
        self._burst = (EnvelopeOpacity(_layer_hits(tspec, notes, onset_loader, grid), float(tspec.get("envelope", 0.2)))
                       if tspec else None)
        x = np.arange(256, dtype=np.float32) / 255.0                        # sRGB -> linear, as a table
        self._to_lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)

    @staticmethod
    def _to_srgb(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, 0.0, 1.0)
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        gain = 1.0 + (self.burst / max(1e-6, float(self.amount.max())) * self._burst(t) if self._burst else 0.0)
        amt = self.amount * gain
        if amt.max() <= 0.0:
            return frame
        H, W = frame.shape[:2]
        rng = np.random.default_rng([self.seed, int(round(t * self.fps))])
        shape = (H, W, 3) if self.independent else (H, W, 1)
        n = rng.standard_normal(shape, dtype=np.float32) if self.gaussian             else rng.random(shape, dtype=np.float32) * 2.0 - 1.0
        work = self._to_lin[frame] if self.linear else frame.astype(np.float32) / 255.0
        if self.correlated:
            out = work * (1.0 + amt * n)
        else:
            out = work + 0.5 * amt * n                                      # the 0.5 matches GEGL
        out = np.clip(out, 0.0, 1.0)
        if self.linear:
            out = self._to_srgb(out)
        return (out * 255.0 + 0.5).astype(np.uint8)


class EchoesLayer:
    """Post-op: dynamically blends previous frames to create temporal echoes

    with Doppler scaling and hue shifting.
    """

    def __init__(self, spec: dict, notes: Sequence, fps: int,
                 features_at=None, onset_loader=None):
        self.fps = fps
        self.features_at = features_at
        self.depth = max(0, min(16, int(spec.get("depth", 5)))) # cap at 16 frames
        self.opacity_decay = float(spec.get("opacity_decay", 0.75))
        self.scale_decay = float(spec.get("scale_decay", 0.95))
        self.hue_shift = float(spec.get("hue_shift", 0.08))
        self.history = collections.deque(maxlen=self.depth)

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self.depth <= 0 or self.opacity_decay <= 0.0:
            return frame
            
        # Add copy of current frame to history
        self.history.append(frame.copy())
        
        if len(self.history) < 2:
            return frame
            
        out = frame.copy()
        n = len(self.history)
        
        # Get current flux to modulate hue shift
        flux = 0.5
        if self.features_at is not None:
            flux = float((self.features_at(t) or {}).get("flux", 0.5) or 0.5)
            
        # Draw history from oldest to newest (index 0 to n-2)
        for i in range(n - 1):
            past_img = self.history[i]
            age = n - 1 - i
            
            scale = self.scale_decay ** age
            opacity = self.opacity_decay ** age
            
            # Scale past frame
            scaled = scale_rotate_image(past_img, scale, angle=0.0)
            
            # Apply hue shift
            shift_val = self.hue_shift * age * (0.3 + 0.7 * flux)
            shifted = hue_shift_channels(scaled, shift_val)
            
            # Additive/screen blend over current composite
            out = blend_frames(out, shifted, "screen", opacity)
            
        return out


class RgbSplit:
    """Post-op reference implementation: chromatic aberration on impact.

    Shifts the R and B channels apart by up to ``amount`` pixels, driven
    by a trigger envelope (e.g. snare hits) — the classic "camera hit"
    glitch that resolves with the drum's decay. Stateless; identity when
    the envelope is at zero, so scenes without hits are untouched.
    """

    def __init__(self, spec: dict, notes: Sequence, fps: int,
                 features_at=None, onset_loader=None):
        tspec = spec.get("trigger") or {}
        self._env = (EnvelopeOpacity(_layer_hits(tspec, notes, onset_loader),
                                     tspec.get("envelope", 0.12))
                     if tspec else None)
        self.amount = float(spec.get("amount", 8.0))
        self.features_at = features_at

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        if self._env is not None:
            e = self._env(t)
        elif self.features_at is not None:
            e = float((self.features_at(t) or {}).get("flux", 0.0) or 0.0)
        else:
            e = 0.0
        px = int(round(self.amount * e))
        if px <= 0:
            return frame
        out = frame.copy()
        out[:, :, 0] = np.roll(frame[:, :, 0], px, axis=1)
        out[:, :, 2] = np.roll(frame[:, :, 2], -px, axis=1)
        return out


class CubesLayer:
    """Stage/scenography layer: reactive clips painted on a lattice of cubes
    that a camera flies through, over an otherwise empty solid-color space.

    This is a *stage*, not a generated world: it takes clip footage — the
    content — and shows it on the faces of cubes floating in space. It owns
    its own clip playback (``ClipLibrary`` + ``ClipComposer``), so it is
    self-contained and never touches the global clip base.

    Scene YAML (``source: cubes``)::

        - source: cubes
          clips_dir: C:/path/to/clips     # own clip pool (required)
          bg_color: [0, 0, 0]             # empty-space color
          n_cubes: 40                     # cubes in the periodic corridor
          faces_with_clips: 3             # 0..6 faces showing the clip screen
          face_resolution: 256            # per-face texture size (RAM/CPU)
          crop_faces: true                # center-crop 16:9 clips to fill (no bars)
          one_clip_per_face: false        # false: same screen on all faces;
                                          # true: an independent clip per face-index
          loop_bars: 2                    # camera advances 1 corridor / N bars
          zoom_pulse: {notes: [36], envelope: 0.25}   # kick = forward lurch
          clip_reactivity:                # a ClipComposer config (the "screen")
            clip_per_bar: true
            clip_order: shuffle           # per-face variety comes from seeded streams
            triggers:
              kick: {notes: [36], actions: [reverse]}

    ``one_clip_per_face`` toggles the depth of the recursion: false = a single
    internal ``ClipComposer`` shown on every clip face (Stage 0); true = one
    ``ClipComposer`` per face-index (0..``faces_with_clips``-1), each an
    independent reactive stream (Stage 1). All composers share one decode
    library, so per-face variety costs O(faces) RAM, not O(cubes). Distinct
    faces come from per-face seeds; use ``clip_order: shuffle`` or ``random``
    (``sequential`` keeps faces in lock-step by design).
    """

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int,
                 fps: int, grid=None, features_at=None, onset_loader=None):
        from core.video.clip_library import ClipLibrary
        from core.video.composer import ClipComposer
        self.W, self.H = width, height
        self.fps = fps
        clips_dir = spec.get("clips_dir")
        if not clips_dir:
            raise ValueError("cubes layer requires 'clips_dir'")
        face_res = int(spec.get("face_resolution", 256))

        # one reactive screen shared by all faces (Stage 0), or one per
        # face-index when one_clip_per_face is set (Stage 1) — bounded to the
        # 0..6 face count, never to the cube count.
        k = max(0, min(6, int(spec.get("faces_with_clips", 3))))
        n_screens = k if (spec.get("one_clip_per_face") and k > 0) else 1

        # a single shared library (a decode cache): size it to hold every
        # screen's current clip at once, so per-face variety costs O(faces),
        # not O(cubes), in RAM.
        cache = max(int(spec.get("cache_size", 4)), n_screens + 2)
        # cover = center-crop 16:9 footage to fill the square faces (no bars)
        fit = "cover" if spec.get("crop_faces") else "contain"
        self.lib = ClipLibrary(clips_dir, face_res, face_res, fps,
                               cache_size=cache, fit=fit)

        base_react = dict(spec.get("clip_reactivity") or {})
        base_seed = spec.get("seed")
        self.composers = []
        for i in range(n_screens):
            react = dict(base_react)
            # distinct stream per face (shuffle/random spread apart by seed);
            # keep Stage 0's single-composer seed exactly as before.
            react["seed"] = base_react.get("seed", base_seed) if n_screens == 1 else (
                None if base_seed is None else int(base_seed) + i * 9973)
            comp = ClipComposer(self.lib, grid, notes, react, onset_loader=onset_loader)
            if n_screens > 1 and len(self.lib) > 1:
                comp._first_selection = False           # don't force clip 0 on bar 0
                comp.transport.set_clip(i % len(self.lib))   # distinct start per face
            self.composers.append(comp)

        self.bg = spec.get("bg_color", [0, 0, 0])
        if grid is not None:
            self.loop_s = float(spec.get("loop_bars", 2)) * grid.bar_duration
        else:
            self.loop_s = float(spec.get("loop_seconds", 8.0))
        zspec = spec.get("zoom_pulse") or {}
        self._pulse = (EnvelopeOpacity(_layer_hits(zspec, notes, onset_loader),
                                       zspec.get("envelope", 0.25))
                       if zspec else None)
        self._pulse_gain = float(spec.get("pulse_gain", 0.12))
        from core.cubes import CubeField
        self.field = CubeField(width, height, spec, seed=base_seed, n_screens=n_screens)
        self._seeked = False

    def frame_at(self, t: float) -> np.ndarray:
        if not self._seeked:
            for comp in self.composers:
                comp.seek(t)          # don't dump pre-start events on frame 0
            self._seeked = True
        screens = [comp.frame_at(t) for comp in self.composers]   # one per face-index
        phase = (t / self.loop_s) % 1.0
        if self._pulse is not None:                     # kick = transient lurch
            phase = (phase + self._pulse_gain * self._pulse(t)) % 1.0
        return self.field.render(phase, screens, self.bg)


def build_compositor(base, video_cfg: dict, notes: Sequence,
                     width: int, height: int, onset_loader=None,
                     fps: int = 24, features_at=None, grid=None) -> "Compositor":
    """Compose ``base`` (ClipComposer) with the scene's extra layers.

    Layers with ``source: clips`` map to the base; unknown sources raise.
    Without a ``layers:`` section, the result is just the base (legacy).
    Layer triggers accept MIDI ``notes`` or ``audio`` onset sources.
    """
    comp = Compositor()
    # no layers: section -> legacy single clips base; a layers: list is
    # used as-is (first layer = canvas), so pure-generative scenes work
    layers_cfg = video_cfg.get("layers") or [{"source": "clips"}]
    windows: list = []
    for spec in layers_cfg:
        src_name = spec.get("source", "clips")
        blend = spec.get("blend", "normal")
        static_op = spec.get("opacity")
        op_fn = (lambda t, v=float(static_op): v) if isinstance(static_op, (int, float)) else None
        window = BarWindow(spec, grid) if "bars" in spec else None
        windows.append(window)
        if window is not None:
            op_fn = (lambda t, w=window, v=float(static_op) if isinstance(static_op, (int, float)) else 1.0:
                     w(t) * v)
        if src_name == "clips":
            comp.add(base, blend if len(comp) else "normal", op_fn)
            continue
        if src_name == "solid":
            src = SolidLayer(width, height, spec.get("color", [255, 255, 255]))
            opacity = op_fn
            trig = spec.get("triggers") or {}
            for name, tspec in trig.items():
                hits = _layer_hits(tspec, notes, onset_loader)
                opacity = EnvelopeOpacity(hits, tspec.get("envelope", 0.1))
            comp.add(src, blend, opacity)
            continue
        if src_name == "cells":
            comp.add(CellsLayer(spec, notes, width, height, fps,
                                features_at=features_at,
                                onset_loader=onset_loader),
                     blend, op_fn)
            continue
        if src_name == "veils":
            comp.add(VeilsLayer(spec, notes, width, height, fps,
                                features_at=features_at,
                                onset_loader=onset_loader, grid=grid),
                     blend if len(comp) else "normal", op_fn)
            continue
        if src_name == "flame":
            comp.add(FlameLayer(spec, notes, width, height, fps,
                                features_at=features_at,
                                onset_loader=onset_loader, grid=grid),
                     blend if len(comp) else "normal", op_fn)
            continue
        if src_name == "eyes":
            comp.add(EyesLayer(spec, notes, width, height, fps,
                               features_at=features_at,
                               onset_loader=onset_loader, grid=grid),
                     blend if len(comp) else "normal", op_fn)
            continue
        if src_name == "mouths":
            comp.add(MouthsLayer(spec, notes, width, height, fps,
                                 features_at=features_at,
                                 onset_loader=onset_loader, grid=grid),
                     blend if len(comp) else "normal", op_fn)
            continue
        if src_name == "chladni":
            comp.add(ChladniLayer(spec, notes, width, height, fps,
                                  features_at=features_at,
                                  onset_loader=onset_loader, grid=grid),
                     blend if len(comp) else "normal", op_fn)
            continue
        if src_name == "orbiters":
            comp.add(OrbitersLayer(spec, notes, width, height, fps,
                                   features_at=features_at,
                                   onset_loader=onset_loader, grid=grid),
                     blend, op_fn)
            continue
        if src_name == "particles":
            comp.add(ParticlesLayer(spec, notes, width, height, fps,
                                    features_at=features_at,
                                    onset_loader=onset_loader),
                     blend, op_fn)
            continue
        if src_name == "feedback":
            comp.add(FeedbackLayer(spec, notes, fps,
                                   features_at=features_at,
                                   onset_loader=onset_loader),
                     "normal", None)
            continue
        if src_name == "rgb_noise":
            comp.add(RgbNoiseLayer(spec, notes, fps, features_at=features_at,
                                   onset_loader=onset_loader, grid=grid), "normal", None)
            continue
        if src_name == "grain":
            comp.add(GrainLayer(spec, notes, fps, features_at=features_at,
                                onset_loader=onset_loader, grid=grid), "normal", None)
            continue
        if src_name == "echoes":
            comp.add(EchoesLayer(spec, notes, fps,
                                 features_at=features_at,
                                 onset_loader=onset_loader),
                     "normal", None)
            continue
        if src_name == "rgb_split":
            comp.add(RgbSplit(spec, notes, fps, features_at=features_at,
                              onset_loader=onset_loader), "normal", None)
            continue
        if src_name == "mandelbox":
            comp.add(MandelboxLayer(spec, notes, width, height, fps,
                                    features_at=features_at,
                                    onset_loader=onset_loader, grid=grid),
                     blend, op_fn)
            continue
        if src_name == "mandelbulb":
            comp.add(MandelbulbLayer(spec, notes, width, height, fps,
                                     features_at=features_at,
                                     onset_loader=onset_loader),
                     blend, op_fn)
            continue
        if src_name == "julia":
            comp.add(JuliaLayer(spec, notes, width, height, fps,
                                features_at=features_at,
                                onset_loader=onset_loader),
                     blend, op_fn)
            continue
        if src_name == "cubes":
            comp.add(CubesLayer(spec, notes, width, height, fps,
                                grid=grid, features_at=features_at,
                                onset_loader=onset_loader),
                     blend if len(comp) else "normal", op_fn)
            continue
        raise ValueError(f"unknown layer source {src_name!r}")
    for i, window in enumerate(windows):               # post-ops are added without opacity: gate them here
        src, blend, opacity = comp._layers[i]
        if window is not None and hasattr(src, "process"):
            comp._layers[i] = (src, blend, window)
    return comp

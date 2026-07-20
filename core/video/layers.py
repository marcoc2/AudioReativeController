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

import numpy as np

BLENDS = {"normal", "add", "screen", "multiply"}


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
                # post-op: transforms the composite built so far
                out = src.process(out, t)
                continue
            op = 1.0 if opacity is None else float(opacity(t))
            if op <= 0.0:
                continue
            top = src.frame_at(t)
            if top is not None:
                out = blend_frames(out, top, blend, op)
        return out


def _layer_hits(spec: dict, notes: Sequence, onset_loader=None) -> list:
    """Hit times for a layer trigger: MIDI ``notes`` or audio ``audio``
    (same spec shape as composer triggers)."""
    if "audio" in spec:
        if onset_loader is None:
            from core.video.composer import _default_onset_loader
            onset_loader = _default_onset_loader
        src = onset_loader(spec)
    else:
        pitches = set(spec.get("notes", []))
        src = [n for n in notes if n.pitch in pitches]
    min_vel = int(spec.get("min_velocity", 0))
    return [n.time for n in src if n.velocity >= min_vel]


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
    for spec in layers_cfg:
        src_name = spec.get("source", "clips")
        blend = spec.get("blend", "normal")
        static_op = spec.get("opacity")
        op_fn = (lambda t, v=float(static_op): v) if isinstance(static_op, (int, float)) else None
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
        raise ValueError(f"unknown layer source {src_name!r}")
    return comp

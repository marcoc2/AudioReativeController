"""A track's own FX, and the lanes (groups) of the Arrange window.

An FX in the stack works on everything under it. One in a layer's ``fx:`` list works on
that layer's picture alone, as an insert on a REAPER track:

    - source: footage                   # a character
      file: lion.mp4
      cutout: {track: render_output/footage/lion_segments}
      fx:                               # in order; each one an FX layer, as in the stack
        - {source: pixel_sort, bars: [[9, 12]]}
        - {source: rgb_split, mix: 0.6}
      spill: true                       # let what the FX spread out show (trails, glow)

The chain is a small stack of its own (``build_compositor``: the FX keep their ``bars:``,
``mix:`` and knobs), its base being the layer's picture. A frame-filling layer (a
Generator) is just that picture. A placed one (a ``footage`` character) is drawn on black
with its own alpha, and its FX show where it is. With ``spill: true`` they also show where
they spread it (a trail, a split edge, a glow), as far as that is bright: on black, dark is
empty. Without it, a grain or a noise stays on the figure instead of filling the frame.

A ``group`` layer is a lane: its layers, then its FX over all of them. The background
(``canvas: below``) is drawn right onto the stack, so it is spliced into it (its layers,
then its FX: ``expand_groups``). The foreground (``canvas: clear``, the default) with FX of
its own is drawn on its own, on black with an alpha, its FX on it alone, and then put over
the stack (without FX it is spliced in too):

    - source: group
      name: FG
      layers:                            # the characters, each with its bars and own fx
        - {source: footage, file: lion.mp4, cutout: {...}, bars: [[5, 17]]}
      fx: [{source: rgb_split}]          # on all of them, not on the background
      spill: false
"""
from __future__ import annotations

from typing import Optional

import numpy as np

KEY = 0.12               # how bright an FX's spill must be (0..1) to show fully over the stack


class _Given:
    """The chain's base: the track's picture of this frame."""

    def __init__(self, cut_times=None):
        self.frame: Optional[np.ndarray] = None
        self.cut_times = cut_times

    def frame_at(self, t: float) -> np.ndarray:
        return self.frame


def _through(chain, given: _Given, rgb: np.ndarray, alpha: np.ndarray, spill: bool, t: float):
    """A picture on black (premultiplied) with its alpha, through a chain of FX: the result
    (float, premultiplied) and its alpha. What the FX left outside the alpha goes, unless
    ``spill`` (then what is bright there shows)."""
    given.frame = rgb if rgb.dtype == np.uint8 else np.clip(rgb + 0.5, 0, 255).astype(np.uint8)
    done = chain.frame_at(t).astype(np.float32)
    if spill:
        alpha = np.maximum(alpha, np.clip(done.max(axis=2) / (255.0 * KEY), 0.0, 1.0))
    else:
        done *= (alpha > 1e-3)[..., None]
    return done, alpha


def _over(rgb: np.ndarray, alpha: np.ndarray, below: np.ndarray) -> np.ndarray:
    """A premultiplied picture over ``below``."""
    keep = 1.0 - alpha[..., None]
    return np.clip(rgb.astype(np.float32) + below.astype(np.float32) * keep + 0.5, 0, 255).astype(np.uint8)


class TrackFX:
    """A frame-filling layer (``frame_at``) through its own FX."""

    def __init__(self, src, chain, given: _Given, spill: bool = False):
        self.src, self.chain, self.given, self.spill = src, chain, given, spill

    def frame_at(self, t: float) -> np.ndarray:
        self.given.frame = self.src.frame_at(t)
        return self.chain.frame_at(t)

    def frame_rgba(self, t: float):
        return self.frame_at(t), None                    # None: it fills the frame


class TrackFXOver(TrackFX):
    """A layer drawn over the stack (``frame_at_over``) through its own FX: a placed one
    (``frame_rgba``) on black with its alpha, else what it draws over the stack."""

    def frame_rgba(self, t: float):
        if not hasattr(self.src, "frame_rgba"):
            return self.frame_at(t), None
        got = self.src.frame_rgba(t)
        if got is None:
            return None
        return _through(self.chain, self.given, *got, self.spill, t)

    def frame_at_over(self, below: np.ndarray, t: float) -> Optional[np.ndarray]:
        if not hasattr(self.src, "frame_rgba"):
            over = self.src.frame_at_over(below, t)
            if over is None:
                return None
            self.given.frame = over
            return self.chain.frame_at(t)
        got = self.frame_rgba(t)
        return None if got is None else _over(*got, below)


def with_fx(src, fx_specs: list, build, base=None, spill: bool = False):
    """``src`` wrapped by its FX chain; ``build(layers, given)`` makes the chain's
    compositor (``build_compositor`` with the render's notes, size and knobs)."""
    given = _Given(getattr(base, "cut_times", None))
    chain = build([{"source": "clips"}] + list(fx_specs), given)
    chain.use_gpu = False                # the chain's frames are numpy; the stack moves them
    if hasattr(src, "frame_at_over"):
        return TrackFXOver(src, chain, given, spill)
    return TrackFX(src, chain, given, spill)


def expand_groups(layers: list) -> list:
    """The layer list with each background group (``canvas: below``) spliced in: its
    layers, then its FX; a foreground one with no FX of its own too (drawn on its own or in
    the stack, it looks the same: spliced, it costs nothing). A group switched off goes whole."""
    out = []
    for spec in layers:
        if isinstance(spec, dict) and spec.get("source") == "group" and (
                spec.get("canvas") == "below" or not any(isinstance(f, dict) for f in spec.get("fx") or [])):
            if spec.get("enabled", True) is not False:
                out += [m for m in spec.get("layers") or [] if isinstance(m, dict)]
                out += [f for f in spec.get("fx") or [] if isinstance(f, dict)]
            continue
        out.append(spec)
    return out


class Group:
    """A lane drawn on its own (the foreground): its layers on black with an alpha, each
    over the ones before it; its FX over all of them; then the lot over the stack."""

    def __init__(self, spec: dict, build, width: int, height: int, base=None):
        members = [m for m in spec.get("layers") or [] if isinstance(m, dict)]
        inner = build([{"source": "solid", "color": [0, 0, 0]}] + members,
                      _Given(getattr(base, "cut_times", None)))
        inner.use_gpu = False
        self.members = inner._layers[1:]
        fx = [f for f in spec.get("fx") or [] if isinstance(f, dict)]
        self.given = _Given(getattr(base, "cut_times", None))
        self.chain = build([{"source": "clips"}] + fx, self.given) if fx else None
        if self.chain is not None:
            self.chain.use_gpu = False
        self.spill = bool(spec.get("spill", False))
        self.W, self.H = width, height

    def frame_rgba(self, t: float):
        acc = np.zeros((self.H, self.W, 3), np.float32)
        alpha = np.zeros((self.H, self.W), np.float32)
        for src, _blend, opacity, _matte in self.members:
            op = 1.0 if opacity is None else float(opacity(t))
            if op <= 0.0:
                if hasattr(src, "observe"):
                    src.observe(np.clip(acc + 0.5, 0, 255).astype(np.uint8), t)
                continue
            if hasattr(src, "process"):                  # an FX among the layers: on what is drawn so far
                done = src.process(np.clip(acc + 0.5, 0, 255).astype(np.uint8), t).astype(np.float32)
                acc += (done - acc) * op
                continue
            got = src.frame_rgba(t) if hasattr(src, "frame_rgba") else (src.frame_at(t), None)
            if got is None or got[0] is None:
                continue
            rgb, a = got                                 # premultiplied by a; a None: it fills the frame
            a = np.full((self.H, self.W), op, np.float32) if a is None else a.astype(np.float32) * op
            acc = rgb.astype(np.float32) * op + acc * (1.0 - a)[..., None]
            alpha = a + alpha * (1.0 - a)
        if self.chain is not None and alpha.any():
            acc, alpha = _through(self.chain, self.given, acc, alpha, self.spill, t)
        return acc, alpha

    def frame_at_over(self, below: np.ndarray, t: float) -> Optional[np.ndarray]:
        rgb, alpha = self.frame_rgba(t)
        return None if not alpha.any() else _over(rgb, alpha, below)

    def frame_at(self, t: float) -> np.ndarray:
        """As the base layer: on black."""
        black = np.zeros((self.H, self.W, 3), np.uint8)
        out = self.frame_at_over(black, t)
        return black if out is None else out

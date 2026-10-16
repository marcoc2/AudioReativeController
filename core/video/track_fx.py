"""A track's own FX: its picture goes through them before it joins the stack.

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


class TrackFX:
    """A frame-filling layer (``frame_at``) through its own FX."""

    def __init__(self, src, chain, given: _Given, spill: bool = False):
        self.src, self.chain, self.given, self.spill = src, chain, given, spill

    def frame_at(self, t: float) -> np.ndarray:
        self.given.frame = self.src.frame_at(t)
        return self.chain.frame_at(t)


class TrackFXOver(TrackFX):
    """A layer drawn over the stack (``frame_at_over``) through its own FX: a placed one
    (``frame_rgba``) on black with its alpha, else what it draws over the stack."""

    def frame_at_over(self, below: np.ndarray, t: float) -> Optional[np.ndarray]:
        if not hasattr(self.src, "frame_rgba"):
            over = self.src.frame_at_over(below, t)
            if over is None:
                return None
            self.given.frame = over
            return self.chain.frame_at(t)
        got = self.src.frame_rgba(t)
        if got is None:
            return None
        rgb, alpha = got
        self.given.frame = rgb
        done = self.chain.frame_at(t)
        done = done.astype(np.float32)
        if self.spill:
            alpha = np.maximum(alpha, np.clip(done.max(axis=2) / (255.0 * KEY), 0.0, 1.0))
        else:                                            # what the FX left outside the figure goes
            done *= (alpha > 1e-3)[..., None]
        keep = 1.0 - alpha[..., None]                    # the clip on black is premultiplied: over
        return np.clip(done + below.astype(np.float32) * keep + 0.5, 0, 255).astype(np.uint8)


def with_fx(src, fx_specs: list, build, base=None, spill: bool = False):
    """``src`` wrapped by its FX chain; ``build(layers, given)`` makes the chain's
    compositor (``build_compositor`` with the render's notes, size and knobs)."""
    given = _Given(getattr(base, "cut_times", None))
    chain = build([{"source": "clips"}] + list(fx_specs), given)
    chain.use_gpu = False                # the chain's frames are numpy; the stack moves them
    if hasattr(src, "frame_at_over"):
        return TrackFXOver(src, chain, given, spill)
    return TrackFX(src, chain, given, spill)

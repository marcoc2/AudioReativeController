"""Drops that follow what they fell on — the ``drops`` post-op with SAM2 holding the region.

The GPU ``drops`` layer (``core/drops``) keeps its paint where it is on the screen: as
the drawing moves, the stain keeps filling whatever region is now under it. Here the
stain belongs to the thing it fell on: when a drop lands, the region it fell into
(bounded by the drawing's lines, the same ink test as ``core/frame_read``) is handed to
SAM2 as a mask, and SAM2 follows that region through the next frames; the paint spreads
from the drop — which rides along with the region — until it fills it, then dries.

Same scene spec as ``DropsLayer`` (``hits``, ``at``, ``color``, ``boost``, ``harmony``,
``speed``, ``grow``, ``life``, ``fade``, ``radius``, ``opacity``, ``stain``, ``rim``,
``rough``, ``cut``, ``ink``, ``wall``, ``seed``), ``shape: blob`` only, plus:

    region: [0.0005, 0.35]     # share of the frame a region may cover to be used as the
                               # prompt; outside it (a leak into the background, a speck)
                               # SAM2 gets the drop's point and picks the region itself

Runs on the CPU (numpy) apart from SAM2; a prototype of the look, see follow_drops.py.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import cv2
import numpy as np

from core.video.layers import VeilsLayer, _layer_events, drop_colour, drop_spot

CUT_DRY = 0.3               # seconds for the stains of a cut-away picture to dry


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def ink_walls(frame: np.ndarray, gain: float = 1.0, wall: float = 0.5) -> np.ndarray:
    """The drawing's lines as walls (bool H×W): ``INK_GLSL``'s test — much darker than the
    ~12 px (at 720p) around it, or simply very dark — thickened by a pixel all round."""
    H = frame.shape[0]
    L = frame.astype(np.float32).dot(np.array([0.299, 0.587, 0.114], np.float32)) / 255.0
    k = max(3, int(round(12.0 * H / 720.0)) | 1)
    Lm = cv2.blur(L, (k, k))
    ink = np.maximum(_smoothstep(0.03, 0.03 + 0.25 / gain, Lm - L), _smoothstep(0.30, 0.10, L))
    return cv2.dilate((ink > wall).astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


def region_at(walls: np.ndarray, x: float, y: float) -> Optional[np.ndarray]:
    """The region enclosed by the walls around (x, y), or None when the point is on a wall."""
    xi, yi = int(x), int(y)
    if walls[yi, xi]:
        return None
    _, labels = cv2.connectedComponents((~walls).astype(np.uint8), connectivity=4)
    return labels == labels[yi, xi]


@dataclass
class _Drop:
    id: int
    born: float
    x: float
    y: float
    colour: np.ndarray
    radius: float
    roll: tuple
    c0: Optional[np.ndarray] = None
    mask: Optional[np.ndarray] = None
    cut_at: Optional[float] = None
    prompt: str = "mask"


class FollowedDrops:
    """``process(frame, t)`` paints the drops onto one frame; frames must come in order."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: float,
                 tracker, grid=None, onset_loader=None):
        if spec.get("shape", "blob") != "blob":
            raise ValueError("drops that follow (SAM2) only do shape: blob for now")
        self.W, self.H, self.fps = width, height, float(fps)
        self.tracker = tracker
        scale = height / 720.0
        self.scale = scale
        self.speed = float(spec.get("speed", 6)) * scale          # pixels per frame
        self.grow = float(spec.get("grow", 2.0))
        self.life = float(spec.get("life", 4.0))
        self.fade = max(1e-3, float(spec.get("fade", 1.0)))
        self.radius = float(spec.get("radius", 6.0)) * scale
        self.opacity = float(spec.get("opacity", 0.9))
        self.stain = float(spec.get("stain", 0.0))
        self.rim = float(spec.get("rim", 0.35))
        self.rough = float(spec.get("rough", 0.35))
        self.cut = float(spec.get("cut", 40))
        self.ink = float(spec.get("ink", 1.0))
        self.wall = float(spec.get("wall", 0.5))
        self.at = spec.get("at", "random")
        self.color = spec.get("color", "under")
        self.boost = float(spec.get("boost", 1.8))
        self.region_share = tuple(spec.get("region", (0.0005, 0.35)))

        hits = _layer_events(spec["hits"], notes, onset_loader, grid) if spec.get("hits") else []
        self._times = np.array([e.time for e in hits], dtype=float)
        self._vel = np.array([e.velocity / 127.0 for e in hits], dtype=float)
        hspec = spec.get("harmony")
        self._chords = _layer_events(hspec, notes, onset_loader, grid) if hspec else []
        self._chord_times = np.array([e.time for e in self._chords], dtype=float)

        self._rng = np.random.default_rng(int(spec.get("seed", 0)))
        # one smooth noise field for every ragged front; each drop reads it at its own offset
        small = self._rng.random((max(2, height // 24), max(2, width // 24))).astype(np.float32)
        self._noise = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC) - 0.5
        self._drops: List[_Drop] = []
        self._next_id = 0
        self._frame_no = -1
        self._prev: Optional[np.ndarray] = None
        self._last_t: Optional[float] = None
        self.stats = {"drops": 0, "mask_prompts": 0, "point_prompts": 0}

    # ------------------------------------------------------------------ drops
    def _chord_hue(self, t: float) -> Optional[float]:
        i = int(np.searchsorted(self._chord_times, t, side="right")) - 1
        return VeilsLayer.hue_of_root(self._chords[i].pitch) if i >= 0 else None

    def _land(self, frame: np.ndarray, walls: np.ndarray, t: float, vel: float) -> None:
        x, y = drop_spot(frame, self._rng, self.at)
        colour = np.array(drop_colour(frame, x, y, self.color, self.boost, self._chord_hue(t)),
                          np.float32)
        d = _Drop(self._next_id, t, x, y, colour, self.radius * (0.5 + vel),
                  (int(self._rng.integers(self.H)), int(self._rng.integers(self.W))))
        self._next_id += 1
        frames = int(math.ceil(self.life * self.fps)) + 1
        region = region_at(walls, x, y)
        share = region.mean() if region is not None else 0.0
        if region is not None and self.region_share[0] <= share <= self.region_share[1]:
            self.tracker.add(d.id, frames=frames, mask=region)
            self.stats["mask_prompts"] += 1
        else:
            self.tracker.add(d.id, frames=frames, point=(x, y))
            d.prompt = "point"
            self.stats["point_prompts"] += 1
        self._drops.append(d)
        self.stats["drops"] += 1

    # ------------------------------------------------------------------ paint
    def _paint(self, out: np.ndarray, frame: np.ndarray, walls: np.ndarray, d: _Drop, t: float):
        M = d.mask
        age = t - d.born
        alpha = self.opacity * min(1.0, max(0.0, (self.life - age) / self.fade))
        if d.cut_at is not None:
            alpha *= max(0.0, 1.0 - (t - d.cut_at) / CUT_DRY)
        if alpha <= 0.0 or M is None or not M.any():
            return
        ys, xs = np.nonzero(M)
        centroid = np.array([xs.mean(), ys.mean()])
        if d.c0 is None:
            d.c0 = centroid
        sx, sy = np.array([d.x, d.y]) + (centroid - d.c0)          # the drop rides on its region
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        r = d.radius + self.speed * self.fps * min(max(age, 0.0), self.grow)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - sx, yy - sy)
        if self.rough > 0:
            noise = np.roll(self._noise, d.roll, axis=(0, 1))[y0:y1, x0:x1]
            dist = dist + noise * (2.0 * self.rough * 0.35 * r)
        P = M[y0:y1, x0:x1] & ~walls[y0:y1, x0:x1] & (dist < r)
        if not P.any():
            return
        # the darker wet rim, like watercolour
        inside = cv2.distanceTransform(P.astype(np.uint8), cv2.DIST_L2, 3)
        rim_px = max(2.0, 6.0 * self.scale)
        shade = 1.0 - self.rim * np.clip(1.0 - inside / rim_px, 0.0, 1.0)
        src = frame[y0:y1, x0:x1].astype(np.float32) / 255.0
        flood = d.colour[None, None, :] * shade[..., None]
        if self.stain > 0:
            L = src.dot(np.array([0.299, 0.587, 0.114], np.float32))[..., None]
            tint = np.clip(d.colour[None, None, :] * (0.25 + 1.1 * L), 0.0, 1.0) * shade[..., None]
            flood = flood * (1.0 - self.stain) + tint * self.stain
        dst = out[y0:y1, x0:x1].astype(np.float32) / 255.0
        a = alpha * P[..., None]
        out[y0:y1, x0:x1] = np.clip((dst * (1.0 - a) + flood * a) * 255.0 + 0.5, 0, 255).astype(np.uint8)

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        self._frame_no += 1
        lo = t - 1.0 / self.fps if self._last_t is None else self._last_t
        self._last_t = t
        small = frame[::16, ::16].astype(np.int16)
        if self._prev is not None and np.abs(small - self._prev).mean() > self.cut:
            for d in self._drops:                       # the picture changed: old stains dry
                if d.cut_at is None:
                    d.cut_at = t
                    self.tracker.forget(d.id)
        self._prev = small

        self.tracker.set_frame(self._frame_no, frame)
        walls = ink_walls(frame, self.ink, self.wall)
        for i in range(int(np.searchsorted(self._times, lo, side="right")),
                       int(np.searchsorted(self._times, t, side="right"))):
            self._land(frame, walls, t, float(self._vel[i]))
        masks = self.tracker.step()

        out = frame.copy()
        keep = []
        for d in self._drops:
            if d.cut_at is None:
                if d.id not in masks:                   # its window is over
                    continue
                d.mask = masks[d.id]
            elif t - d.cut_at >= CUT_DRY:
                continue
            if t - d.born >= self.life:
                self.tracker.forget(d.id)
                continue
            self._paint(out, frame, walls, d, t)
            keep.append(d)
        self._drops = keep
        return out

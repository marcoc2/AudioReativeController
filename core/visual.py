"""
Core visual primitives for ARC.

VisualObject  — pure data: what to draw (position, size, colour).
Frame         — one rendered moment: continuous temporal context + object list.
render_frame  — translates a Frame to pixels on a pygame Surface.

Design rule: Frame holds only continuous quantities (t, bar_phase, beat_phase).
Binary rendering decisions (flash on downbeat, etc.) belong in build_frame or
the renderer, never as boolean fields on Frame.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import pygame


@dataclass
class VisualObject:
    """Data-only description of one drawable entity.

    x, y          — normalised 0..1 fractions of surface width / height.
    radius        — fraction of the renderer's max_r (caller defines the scale).
    shape         — "circle" or "polygon".
    n_vertices    — number of vertices when shape="polygon".
    rotation      — rotation angle in radians (polygon only).
    vertex_jitter — 0..1 radial perturbation per vertex (polygon only).
    in_trail      — whether this object participates in trail ghost rendering.
    """
    id: str
    x: float
    y: float
    radius: float
    color: tuple           # (R, G, B)
    alpha: int = 255
    filled: bool = True    # False → outline only
    ring_width: int = 4
    in_trail: bool = True
    shape: str = "circle"  # "circle" | "polygon"
    n_vertices: int = 5
    rotation: float = 0.0
    vertex_jitter: float = 0.0


@dataclass
class Frame:
    """One rendered moment.

    Only continuous temporal quantities live here.  Renderers and build_frame
    derive any discrete decisions (is_downbeat, flash, etc.) from bar_phase /
    beat_phase themselves.
    """
    frame_idx: int
    t: float            # absolute seconds
    bar_phase: float    # 0..1 — position within current bar
    beat_phase: float   # 0..1 — position within current beat
    objects: List[VisualObject] = field(default_factory=list)


def render_frame(
    surface: pygame.Surface,
    frame: Frame,
    max_r: int,
    trail_frames: Optional[List[List[VisualObject]]] = None,
) -> None:
    """Draw frame onto surface.

    max_r        — pixel radius that VisualObject.radius == 1.0 maps to.
    trail_frames — list of past object snapshots, ordered oldest → newest.
                   Each snapshot is rendered with increasing alpha.
    """
    W, H = surface.get_size()

    if trail_frames:
        n = len(trail_frames)
        for i, past_objects in enumerate(trail_frames):
            trail_alpha = int(255 * (i + 1) / (n + 1))
            for obj in past_objects:
                if obj.in_trail:
                    _draw_obj(surface, obj, W, H, max_r, alpha_override=trail_alpha)

    for obj in frame.objects:
        _draw_obj(surface, obj, W, H, max_r)


def _polygon_points(cx: int, cy: int, r: int, n: int,
                    rotation: float, jitter: float) -> list:
    """Compute pixel-space vertices for a mutant polygon.

    Each vertex radius is perturbed by a smooth sinusoidal function so the
    shape morphs organically as rotation changes over time.
    """
    pts = []
    for i in range(n):
        angle = rotation + 2 * math.pi * i / n
        # Irrational multipliers give each vertex a distinct phase
        r_i = r * (1.0 + jitter * math.sin(angle * 2.3 + rotation * 1.7))
        pts.append((cx + r_i * math.cos(angle), cy + r_i * math.sin(angle)))
    return pts


def _draw_obj(
    surface: pygame.Surface,
    obj: VisualObject,
    W: int,
    H: int,
    max_r: int,
    alpha_override: Optional[int] = None,
) -> None:
    cx = int(obj.x * W)
    cy = int(obj.y * H)
    r  = int(obj.radius * max_r)
    if r < 1:
        return

    alpha = alpha_override if alpha_override is not None else obj.alpha
    color = obj.color
    width = 0 if obj.filled else obj.ring_width

    if obj.shape == "polygon":
        pts = _polygon_points(cx, cy, r, obj.n_vertices, obj.rotation, obj.vertex_jitter)
        if alpha < 255:
            # bounding box for the temp surface
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            bx = int(min(xs)) - 2
            by = int(min(ys)) - 2
            bw = int(max(xs)) - bx + 4
            bh = int(max(ys)) - by + 4
            if bw < 1 or bh < 1:
                return
            tmp = pygame.Surface((bw, bh), pygame.SRCALPHA)
            local_pts = [(p[0] - bx, p[1] - by) for p in pts]
            pygame.draw.polygon(tmp, (*color, alpha), local_pts, width)
            surface.blit(tmp, (bx, by))
        else:
            pygame.draw.polygon(surface, color, pts, width)
        return

    # circle (default)
    if alpha < 255:
        tmp = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        pygame.draw.circle(tmp, (*color, alpha), (r, r), r, width)
        surface.blit(tmp, (cx - r, cy - r))
    else:
        pygame.draw.circle(surface, color, (cx, cy), r, width)

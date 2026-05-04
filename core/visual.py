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

from dataclasses import dataclass, field
from typing import List, Optional

import pygame


@dataclass
class VisualObject:
    """Data-only description of one drawable entity.

    x, y     — normalised 0..1 fractions of surface width / height.
    radius   — fraction of the renderer's max_r (caller defines the scale).
    in_trail — whether this object participates in trail ghost rendering.
    """
    id: str
    x: float
    y: float
    radius: float
    color: tuple          # (R, G, B)
    alpha: int = 255
    filled: bool = True   # False → ring outline
    ring_width: int = 4
    in_trail: bool = True


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

    if alpha < 255:
        tmp = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
        if obj.filled:
            pygame.draw.circle(tmp, (*color, alpha), (r, r), r)
        else:
            pygame.draw.circle(tmp, (*color, alpha), (r, r), r, obj.ring_width)
        surface.blit(tmp, (cx - r, cy - r))
    else:
        if obj.filled:
            pygame.draw.circle(surface, color, (cx, cy), r)
        else:
            pygame.draw.circle(surface, color, (cx, cy), r, obj.ring_width)

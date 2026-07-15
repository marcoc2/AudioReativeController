"""ClipComposer — turns musical time + MIDI drum events into video frames.

Configured by the ``video:`` section of the scene YAML:

    video:
      clip_per_bar: true          # pick a new clip at every bar start
      clip_order: sequential      # or "random" (uses seed)
      seed: 42
      triggers:
        kick:
          notes: [36]
          actions: [reverse]      # reverse | next_clip | random_clip | restart
          min_velocity: 0         # ignore hits softer than this (0..127)
        snare:
          notes: [38, 40]
          actions: [next_clip]

``frame_ops`` is the seam for future pixel-level effects: a list of
callables ``(frame, t, transport) -> frame`` applied in order after the
clip frame is fetched, so system-rendered layers (particles, overlays)
can composite on top of pre-rendered footage.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from core.video.transport import ClipTransport

ACTIONS = {"reverse", "next_clip", "random_clip", "restart"}


@dataclass
class TriggerEvent:
    time: float
    name: str
    velocity: int
    actions: tuple


class ClipComposer:
    def __init__(self, library, grid, notes: Sequence, video_cfg: Optional[dict] = None):
        """``library`` needs ``__len__`` and ``get(idx) -> ClipFrames``-like.

        ``notes`` is the MidiNote list from ``read_midi`` (may be empty).
        """
        cfg = video_cfg or {}
        self.library = library
        self.grid = grid
        self.transport = ClipTransport(len(library))
        self.clip_per_bar: bool = bool(cfg.get("clip_per_bar", True))
        self.clip_order: str = cfg.get("clip_order", "sequential")
        self._rng = random.Random(cfg.get("seed"))
        self.frame_ops: List[Callable] = []

        self.events = self._build_events(notes, cfg.get("triggers", {}))
        self._ev_ptr = 0
        self._last_bar: Optional[int] = None
        self._first_selection = True

    # ------------------------------------------------------------------

    @staticmethod
    def _build_events(notes: Sequence, triggers: dict) -> List[TriggerEvent]:
        events: List[TriggerEvent] = []
        for name, spec in triggers.items():
            pitches = set(spec.get("notes", []))
            actions = tuple(spec.get("actions", []))
            min_vel = int(spec.get("min_velocity", 0))
            unknown = [a for a in actions if a not in ACTIONS]
            if unknown:
                raise ValueError(
                    f"trigger {name!r}: unknown actions {unknown}; expected {sorted(ACTIONS)}"
                )
            if not pitches or not actions:
                continue
            for n in notes:
                if n.pitch in pitches and n.velocity >= min_vel:
                    events.append(TriggerEvent(n.time, name, n.velocity, actions))
        events.sort(key=lambda e: e.time)
        return events

    def seek(self, t: float) -> None:
        """Skip events strictly before t (call once before the render loop)."""
        self._ev_ptr = 0
        while self._ev_ptr < len(self.events) and self.events[self._ev_ptr].time < t:
            self._ev_ptr += 1

    def _bar_index(self, t: float) -> int:
        db = self.grid.downbeats
        if db is not None and len(db) > 0:
            idx = int(np.searchsorted(db, t, side="right")) - 1
            return max(0, idx)
        return int((t - self.grid.start_offset) // self.grid.bar_duration)

    def _select_clip(self) -> None:
        n = len(self.library)
        if self.clip_order == "random":
            if n > 1:
                choices = [i for i in range(n) if i != self.transport.clip_idx]
                self.transport.set_clip(self._rng.choice(choices))
        elif self._first_selection:
            self.transport.set_clip(0)
        else:
            self.transport.next_clip()
        self._first_selection = False

    def _apply_action(self, action: str) -> None:
        if action == "random_clip":
            n = len(self.library)
            if n > 1:
                choices = [i for i in range(n) if i != self.transport.clip_idx]
                self.transport.set_clip(self._rng.choice(choices))
        else:
            getattr(self.transport, action)()

    # ------------------------------------------------------------------

    def frame_at(self, t: float) -> np.ndarray:
        """Return the composed RGB frame (H, W, 3 uint8) for time t.

        Must be called with monotonically increasing t (one call per
        output frame): the transport advances as a side effect.
        """
        bar = self._bar_index(t)
        if bar != self._last_bar:
            if self.clip_per_bar or self._first_selection:
                self._select_clip()
            self._last_bar = bar

        while self._ev_ptr < len(self.events) and self.events[self._ev_ptr].time <= t:
            for action in self.events[self._ev_ptr].actions:
                self._apply_action(action)
            self._ev_ptr += 1

        clip = self.library.get(self.transport.clip_idx)
        frame = clip.frame(self.transport.frame_index)
        self.transport.advance(len(clip))

        for op in self.frame_ops:
            frame = op(frame, t, self.transport)
        return frame

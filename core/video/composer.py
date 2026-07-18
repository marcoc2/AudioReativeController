"""ClipComposer — turns musical time + MIDI drum events into video frames.

Configured by the ``video:`` section of the scene YAML:

    video:
      clip_per_bar: true          # pick a new clip at every bar start
      clip_order: sequential      # sequential | random | shuffle
      seed: 42

    clip_order "shuffle" is a shuffle bag: every clip is used exactly once
    per cycle in random order, and the bag is reshuffled when it empties
    (never repeating the previous cycle's order, nor the same clip twice
    in a row across the boundary). "random" draws independently per switch.
      triggers:
        kick:
          notes: [36]             # hits from the MIDI note list…
          actions: [reverse]      # reverse | next_clip | random_clip | restart
          min_velocity: 0         # ignore hits softer than this (0..127)
          until: snare            # stop firing once 'snare' lands its first hit
          gravity:                # hits warp playback speed like gravity wells:
            peak: 3.0             #   speed at the exact hit
            floor: 0.3            #   speed far away from any hit
            radius: 0.45          #   influence radius in seconds
            curve: 2.0            #   >1 concentrates the pull near the hit
        snare:
          audio: stems/snare.wav  # …or transients detected in an audio stem
          threshold: 0.3          # onset sensitivity (higher = fewer hits)
          min_gap: 0.05           # debounce between hits, seconds
          exclude:                # drop onsets near another trigger's hits
            trigger: kick         #   (kills mic bleed: kick thump in the
            window: 0.04          #    snare mic lands exactly on MIDI kicks)
          actions: [next_clip]

    A trigger sources its hits from MIDI (``notes``) or from audio
    transients (``audio``); everything downstream (actions, min_velocity,
    gravity, ``until``) treats both identically. ``until: <other>``
    deactivates the trigger — events AND gravity — from the first hit of
    ``<other>`` onward (hand-over, e.g. kick drives clip changes only
    until the snare comes in).

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


@dataclass
class GravityWarp:
    """Playback-speed warp around trigger hits.

    Speed peaks at each hit time and falls off to ``floor`` at ``radius``
    seconds away, so playback accelerates toward a hit and decelerates
    leaving it — symmetric in time, like falling into and out of a well.
    """
    times: np.ndarray     # sorted hit times in seconds
    peak: float = 3.0
    floor: float = 0.3
    radius: float = 0.45
    curve: float = 2.0

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError("gravity radius must be > 0")
        self.floor = max(0.05, float(self.floor))  # never stall the playhead
        self.times = np.asarray(self.times, dtype=float)

    def speed_at(self, t: float) -> float:
        if len(self.times) == 0:
            return self.floor
        idx = int(np.searchsorted(self.times, t))
        d = np.inf
        if idx > 0:
            d = min(d, abs(t - float(self.times[idx - 1])))
        if idx < len(self.times):
            d = min(d, abs(float(self.times[idx]) - t))
        w = max(0.0, 1.0 - d / self.radius)
        w = w ** self.curve
        return self.floor + (self.peak - self.floor) * w


def _default_onset_loader(spec: dict):
    from core.rhythm.onset_reader import read_onsets  # lazy: pulls in librosa
    return read_onsets(
        spec["audio"],
        threshold=float(spec.get("threshold", 0.3)),
        min_gap=float(spec.get("min_gap", 0.05)),
    )


class ClipComposer:
    def __init__(
        self,
        library,
        grid,
        notes: Sequence,
        video_cfg: Optional[dict] = None,
        onset_loader: Optional[Callable] = None,
    ):
        """``library`` needs ``__len__`` and ``get(idx) -> ClipFrames``-like.

        ``notes`` is the MidiNote list from ``read_midi`` (may be empty).
        ``onset_loader(spec) -> List[MidiNote]`` resolves ``audio:`` trigger
        sources; injectable for tests.
        """
        cfg = video_cfg or {}
        self.library = library
        self.grid = grid
        self.transport = ClipTransport(len(library))
        self.clip_per_bar: bool = bool(cfg.get("clip_per_bar", True))
        self.clip_order: str = cfg.get("clip_order", "sequential")
        self._rng = random.Random(cfg.get("seed"))
        self.frame_ops: List[Callable] = []
        self._onset_loader = onset_loader or _default_onset_loader

        # manual pins: {bar_index: clip name (path stem) or clip index}
        self.overrides = {int(k): v for k, v in (cfg.get("overrides") or {}).items()}
        self._name_to_idx = None

        triggers = cfg.get("triggers", {})
        hits = self._collect_hits(notes, triggers)
        self.events = self._build_events(hits, triggers)
        self.gravity = self._build_gravity(hits, triggers)
        self._ev_ptr = 0
        self._last_bar: Optional[int] = None
        self._first_selection = True
        self._bag: List[int] = []

    # ------------------------------------------------------------------

    def _collect_hits(self, notes: Sequence, triggers: dict) -> dict:
        """Resolve each trigger's hit list (MIDI notes or audio onsets).

        Pipeline per trigger: collect raw hits (min_velocity applied) →
        ``exclude`` cross-trigger suppression (e.g. mic bleed) → ``until``
        hand-overs (keep only hits strictly before the other trigger's
        first hit). Exclusion compares against the raw snapshot, so the
        result does not depend on trigger declaration order.
        """
        hits: dict = {}
        for name, spec in triggers.items():
            if "audio" in spec:
                src = self._onset_loader(spec)
            else:
                pitches = set(spec.get("notes", []))
                src = [n for n in notes if n.pitch in pitches]
            min_vel = int(spec.get("min_velocity", 0))
            hits[name] = sorted(
                (n for n in src if n.velocity >= min_vel), key=lambda n: n.time
            )

        raw = dict(hits)
        for name, spec in triggers.items():
            ex = spec.get("exclude")
            if not ex:
                continue
            other = ex.get("trigger")
            window = float(ex.get("window", 0.04))
            if other not in raw:
                raise ValueError(
                    f"trigger {name!r}: 'exclude' references unknown trigger {other!r}"
                )
            other_t = np.array([h.time for h in raw[other]], dtype=float)
            if len(other_t):
                hits[name] = [
                    h for h in hits[name]
                    if float(np.abs(other_t - h.time).min()) > window
                ]

        for name, spec in triggers.items():
            until = spec.get("until")
            if until is None:
                continue
            if until not in hits:
                raise ValueError(
                    f"trigger {name!r}: 'until' references unknown trigger {until!r}"
                )
            if hits[until]:
                cutoff = hits[until][0].time
                hits[name] = [h for h in hits[name] if h.time < cutoff]
        return hits

    @staticmethod
    def _build_events(hits: dict, triggers: dict) -> List[TriggerEvent]:
        events: List[TriggerEvent] = []
        for name, spec in triggers.items():
            actions = tuple(spec.get("actions", []))
            unknown = [a for a in actions if a not in ACTIONS]
            if unknown:
                raise ValueError(
                    f"trigger {name!r}: unknown actions {unknown}; expected {sorted(ACTIONS)}"
                )
            if not actions:
                continue
            for h in hits[name]:
                events.append(TriggerEvent(h.time, name, h.velocity, actions))
        events.sort(key=lambda e: e.time)
        return events

    @staticmethod
    def _build_gravity(hits: dict, triggers: dict) -> List[GravityWarp]:
        """Collect speed warps from triggers that declare a ``gravity`` block.

        Gravity is independent of ``actions``: a trigger may warp speed
        without firing any transport action (and vice versa). It respects
        the same ``until`` cutoff as the trigger's events.
        """
        warps: List[GravityWarp] = []
        for name, spec in triggers.items():
            g = spec.get("gravity")
            if not g:
                continue
            times = [h.time for h in hits[name]]
            warps.append(GravityWarp(
                times=np.asarray(times, dtype=float),
                peak=float(g.get("peak", 3.0)),
                floor=float(g.get("floor", 0.3)),
                radius=float(g.get("radius", 0.45)),
                curve=float(g.get("curve", 2.0)),
            ))
        return warps

    def speed_at(self, t: float) -> float:
        """Playback speed at time t: strongest gravity pull wins, else 1.0."""
        if not self.gravity:
            return 1.0
        return max(g.speed_at(t) for g in self.gravity)

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

    def _next_from_bag(self, n: int) -> int:
        if not self._bag:
            bag = list(range(n))
            self._rng.shuffle(bag)
            # don't show the same clip twice in a row across the cycle boundary
            if n > 1 and not self._first_selection and bag[0] == self.transport.clip_idx:
                bag[0], bag[-1] = bag[-1], bag[0]
            self._bag = bag
        return self._bag.pop(0)

    def _select_clip(self) -> None:
        n = len(self.library)
        if self.clip_order == "shuffle":
            self.transport.set_clip(self._next_from_bag(n))
        elif self.clip_order == "random":
            if n > 1:
                choices = [i for i in range(n) if i != self.transport.clip_idx]
                self.transport.set_clip(self._rng.choice(choices))
        elif self._first_selection:
            self.transport.set_clip(0)
        else:
            self.transport.next_clip()
        self._first_selection = False

    def _resolve_clip_ref(self, ref) -> Optional[int]:
        if isinstance(ref, int):
            return ref % len(self.library)
        if self._name_to_idx is None:
            paths = getattr(self.library, "paths", None)
            self._name_to_idx = (
                {p.stem: i for i, p in enumerate(paths)} if paths else {})
        return self._name_to_idx.get(str(ref))

    def _apply_override(self) -> None:
        """A pinned bar wins over any rule/action, without restarting the
        clip on every frame (only switches when needed)."""
        ref = self.overrides.get(self._last_bar)
        if ref is None:
            return
        idx = self._resolve_clip_ref(ref)
        if idx is not None and idx != self.transport.clip_idx:
            self.transport.set_clip(idx)

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

        self._apply_override()

        clip = self.library.get(self.transport.clip_idx)
        frame = clip.frame(self.transport.frame_index)
        self.transport.speed = self.speed_at(t)
        self.transport.advance(len(clip))

        for op in self.frame_ops:
            frame = op(frame, t, self.transport)
        return frame

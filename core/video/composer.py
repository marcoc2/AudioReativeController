"""ClipComposer — turns musical time + MIDI drum events into video frames.

Configured by the ``video:`` section of the scene YAML:

    video:
      clip_per_bar: true          # pick a new clip at every bar start
      bars_per_clip: 1            # ... or every N bars (counted from the song's bar 1),
                                  #     e.g. 2 to cut on two-bar phrases
      clip_end: bounce            # a clip shorter than its turn: bounce (play it back
                                  #     and forth) | loop (start it over)
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
        harmony:
          score: input/song/sheetsage  # …or what a transcription says is written
          events: chord_changes   # chord_changes | section_changes | melody
          snap: auto              # auto | bar | beat | off  (see below)
          notes: [8]              # optional: only changes INTO these roots (8 = G#)
          actions: [next_clip]

    A trigger sources its hits from MIDI (``notes``), from audio transients
    (``audio``) or from a SheetSage2 transcription (``score``, see core/score):
    for music with no attacks to follow, "the chord changed" is the event.
    Everything downstream (actions, min_velocity, gravity, ``until``,
    ``exclude``) treats all three identically.

    A transcription knows *what* happens far better than *when*, so score
    events are snapped onto this composer's grid. ``snap: auto`` (default)
    snaps to bars when the grid carries real markers — it came from a MIDI
    file — and leaves times alone otherwise, since a placeholder grid would
    only make them worse. With ``score``, ``notes`` filters by the event's
    pitch: the new root's pitch class for chord changes (0 = C … 11 = B),
    the section's index for section changes, the MIDI pitch for melody.

    ``until: <other>``
    deactivates the trigger — events AND gravity — from the first hit of
    ``<other>`` onward (hand-over, e.g. kick drives clip changes only
    until the snare comes in).

Pins: a bar can be given its clip, by name (the file's stem) or ``none`` (nothing there:
black, whatever is over it shows). The clips layer carries them by bar number (1: the
song's first bar), as the Arrange window writes them; they win over ``overrides:``, the
older form (bar indexes from 0, the studio's timeline pins):

    layers:
      - source: clips
        pins: {3: lofi_2, 4: lofi_2, 9: none}

A pinned bar changes nothing around it: the montage draws the same clips for the other
bars, and a clip pinned to bars in a row plays on across them.

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


def pins_of(layers) -> dict:
    """The clips layer's ``pins:`` (bar number -> clip name or "none") as overrides (bar
    index from 0 -> the same)."""
    for layer in layers or []:
        if isinstance(layer, dict) and layer.get("source", "clips") == "clips":
            return {int(k) - 1: v for k, v in (layer.get("pins") or {}).items()}
    return {}


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


SCORE_EVENTS = {"chord_changes", "section_changes", "melody"}
SCORE_SNAPS = {"auto", "bar", "beat", "off"}


def _grid_is_real(grid) -> bool:
    """True when the grid carries bar markers, i.e. it was read from something
    (a MIDI file) rather than being a bare placeholder tempo."""
    downbeats = getattr(grid, "downbeats", None)
    return downbeats is not None and len(downbeats) >= 2


def _default_score_loader(spec: dict, grid):
    events = spec.get("events", "chord_changes")
    if events not in SCORE_EVENTS:
        raise ValueError(f"score trigger: unknown events {events!r}; expected {sorted(SCORE_EVENTS)}")
    score = load_score(spec, grid, melody=events == "melody")
    if events == "melody":
        return score.melody(spec.get("voice", "instrumental"))
    return getattr(score, events)()


def load_score(spec: dict, grid, melody: bool = False):
    """The transcription a ``score:`` spec names, snapped onto ``grid`` as ``snap`` says
    (auto: to the bars when the grid is real)."""
    from core.score import read_score  # lazy: keeps the composer import light

    snap = spec.get("snap", "auto")
    if snap not in SCORE_SNAPS:
        raise ValueError(f"score trigger: unknown snap {snap!r}; expected {sorted(SCORE_SNAPS)}")
    if snap == "auto":
        snap = "bar" if _grid_is_real(grid) else "off"

    score = read_score(spec["score"])
    if snap != "off":
        # melody is rhythm: snapping it to bars would flatten it, so it follows
        # sixteenths of the grid while chords and sections follow ``snap``
        score = score.snap_to(grid, unit=snap, melody_division=4 if melody else None)
    return score


class ClipComposer:
    def __init__(
        self,
        library,
        grid,
        notes: Sequence,
        video_cfg: Optional[dict] = None,
        onset_loader: Optional[Callable] = None,
        score_loader: Optional[Callable] = None,
    ):
        """``library`` needs ``__len__`` and ``get(idx) -> ClipFrames``-like.

        ``notes`` is the MidiNote list from ``read_midi`` (may be empty).
        ``onset_loader(spec) -> List[MidiNote]`` resolves ``audio:`` trigger
        sources and ``score_loader(spec, grid) -> List[MidiNote]`` resolves
        ``score:`` ones; both injectable for tests.
        """
        cfg = video_cfg or {}
        self.library = library
        self.grid = grid
        self.transport = ClipTransport(len(library))
        clip_end = cfg.get("clip_end", "bounce")
        if clip_end not in ("bounce", "loop"):
            raise ValueError(f"video: clip_end {clip_end!r} (use bounce or loop)")
        self.transport.loop = clip_end == "loop"
        self.clip_per_bar: bool = bool(cfg.get("clip_per_bar", True))
        self.bars_per_clip: int = max(1, int(cfg.get("bars_per_clip", 1)))
        self.clip_order: str = cfg.get("clip_order", "sequential")
        self._rng = random.Random(cfg.get("seed"))
        self.frame_ops: List[Callable] = []
        self._onset_loader = onset_loader or _default_onset_loader
        self._score_loader = score_loader or _default_score_loader

        # manual pins: {bar_index: clip name (path stem) or clip index}
        self.overrides = {int(k): v for k, v in (cfg.get("overrides") or {}).items()}
        self.overrides.update(pins_of(cfg.get("layers")))
        self.blank = False             # this bar is pinned to nothing
        self._drawn = None             # (clip, pos, direction) a pin took the place of, this bar
        self._held = None              # the pinned clip as the last bar left it
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
            elif "score" in spec:
                src = self._score_loader(spec, self.grid)
                if "notes" in spec:          # e.g. only changes into G#
                    wanted = set(spec["notes"])
                    src = [n for n in src if n.pitch in wanted]
            else:
                from core.rhythm.midi_reader import select_notes
                src = select_notes(notes, spec)
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

    def take_state(self, other: "ClipComposer") -> None:
        """Go on from where ``other`` stands: the same scene, played (cheaply, on a library
        of the same clips that only knows their lengths) up to a point. A preview of bar 20
        then shows the clips the whole song would: the montage's draws depend on what came
        before."""
        own = {"library": self.library, "frame_ops": self.frame_ops, "_name_to_idx": None}
        self.__dict__.update({k: v for k, v in other.__dict__.items() if k not in own})
        self.__dict__.update(own)

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
        clip on every frame (only switches when needed); what the montage drew
        is kept, to go on with after the pinned bar."""
        ref = self.overrides.get(self._last_bar)
        self.blank = isinstance(ref, str) and ref.strip().lower() == "none"
        if ref is None or self.blank:
            return
        idx = self._resolve_clip_ref(ref)
        tp = self.transport
        if idx is None or idx == tp.clip_idx:
            return
        if self._drawn is None:
            self._drawn = (tp.clip_idx, tp.pos, tp.direction)
        tp.set_clip(idx)
        if self._held is not None and self._held[0] == idx:   # pinned the bar before too: on it goes
            tp.pos, tp.direction = self._held[1], self._held[2]

    def _unpin(self) -> None:
        """A new bar: the clip the montage drew comes back (it chooses on from there)."""
        tp = self.transport
        if self._drawn is None:
            self._held = None
            return
        self._held = (tp.clip_idx, tp.pos, tp.direction)
        tp.clip_idx, tp.pos, tp.direction = self._drawn
        self._drawn = None

    def _apply_action(self, action: str) -> None:
        if action == "random_clip":
            n = len(self.library)
            if n > 1:
                choices = [i for i in range(n) if i != self.transport.clip_idx]
                self.transport.set_clip(self._rng.choice(choices))
        else:
            getattr(self.transport, action)()

    def cut_times(self, until: float) -> np.ndarray:
        """When the clip changes, up to ``until``, known before rendering: the bars that pick
        a new clip, the triggers' clip actions and the pinned bars (what ``_cuts.json``
        records afterwards). Effects use it to see a cut coming."""
        if len(self.library) < 2:
            return np.zeros(0)
        cuts = []
        bar = 1
        while True:
            t = float(self.grid.bar_start(bar))
            if t > until:
                break
            if self._bar_index(t) == bar and ((self.clip_per_bar and bar % self.bars_per_clip == 0)
                                              or bar in self.overrides):
                cuts.append(t)
            bar += 1
        cuts += [e.time for e in self.events
                 if e.time <= until and any(a in ("next_clip", "random_clip") for a in e.actions)]
        return np.unique(np.round(cuts, 6))

    # ------------------------------------------------------------------

    def frame_at(self, t: float) -> np.ndarray:
        """Return the composed RGB frame (H, W, 3 uint8) for time t.

        Must be called with monotonically increasing t (one call per
        output frame): the transport advances as a side effect.
        """
        bar = self._bar_index(t)
        if bar != self._last_bar:
            self._unpin()
            if (self.clip_per_bar and bar % self.bars_per_clip == 0) or self._first_selection:
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
        if self.blank:
            frame = np.zeros_like(frame)

        for op in self.frame_ops:
            frame = op(frame, t, self.transport)
        return frame

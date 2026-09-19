"""Read a SheetSage2 transcription folder into a Score.

Where ``feature_extractor`` answers "what does it sound like right now" and
``rhythm`` answers "where in the bar are we", a Score answers "what is
written in the music": which chord, which key, which section, which melody
note. For material without transients — pads, drones, anything ethereal —
these are the events a scene can hang on when there is no kick to follow.

SheetSage2 itself is *not* a dependency of this project. It runs elsewhere
(its own venv, weights in the shared Hugging Face cache) and leaves a folder
of plain text files behind; this module only reads that folder:

    chord.lab       start  end  label        Harte labels: "A:maj", "D:7", "N"
    key.lab         start  end  label        "A:major"
    structure.lab   start  end  label        "intro", "verse", "chorus", ...
    beat.lab        time  beat_in_bar  num  den
    downbeat.lab    time
    melody_vocal.lab / melody_instrumental.lab     start  end  midi_pitch

Times are seconds in the transcribed audio file. Every file is optional — a
missing one simply leaves that part of the Score empty.

Changes (chord, section) and melody notes are also offered as ``MidiNote``
events, the same shape ``read_midi`` and ``read_onsets`` produce, so the
trigger system can treat "the chord changed" exactly like "the kick hit".
"""
from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import MidiNote

NO_CHORD = "N"
SCORE_CHANNEL = 0
CHANGE_VELOCITY = 100

_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Harte chord qualities -> semitones above the root. Anything not listed
# falls back to a triad guessed from the name, so an unknown quality still
# lights up sensible keys instead of raising.
_QUALITY_INTERVALS = {
    "maj": (0, 4, 7), "min": (0, 3, 7), "dim": (0, 3, 6), "aug": (0, 4, 8),
    "sus2": (0, 2, 7), "sus4": (0, 5, 7), "5": (0, 7), "1": (0,),
    "7": (0, 4, 7, 10), "maj7": (0, 4, 7, 11), "min7": (0, 3, 7, 10),
    "minmaj7": (0, 3, 7, 11), "dim7": (0, 3, 6, 9), "hdim7": (0, 3, 6, 10),
    "maj6": (0, 4, 7, 9), "min6": (0, 3, 7, 9),
    "9": (0, 4, 7, 10, 14), "maj9": (0, 4, 7, 11, 14), "min9": (0, 3, 7, 10, 14),
    "11": (0, 4, 7, 10, 14, 17), "13": (0, 4, 7, 10, 14, 21),
}


@dataclass(frozen=True)
class Span:
    """A labelled stretch of time: one chord, one key, one section."""
    start: float
    end: float
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def parse_chord(label: str) -> Tuple[Optional[int], str, Tuple[int, ...]]:
    """Split a Harte label into (root pitch class, quality, pitch classes).

    ``"A:min7"`` -> ``(9, "min7", (9, 0, 4, 7))``. No-chord (``"N"``) and
    anything unparseable give ``(None, "", ())``.
    """
    label = label.strip()
    if not label or label == NO_CHORD or label[0] not in _PITCH_CLASSES:
        return None, "", ()
    root_name, _, rest = label.partition(":")
    root = _PITCH_CLASSES[root_name[0]]
    for accidental in root_name[1:]:
        root += 1 if accidental == "#" else -1 if accidental == "b" else 0
    root %= 12

    quality = rest.split("/")[0].split("(")[0] or "maj"
    intervals = _QUALITY_INTERVALS.get(quality)
    if intervals is None:
        intervals = _QUALITY_INTERVALS["min" if quality.startswith("min") else "maj"]
    return root, quality, tuple((root + i) % 12 for i in intervals)


def _merge(spans: Sequence[Span]) -> List[Span]:
    """Join neighbours that carry the same label.

    SheetSage2 writes one chord line per bar, so four bars of A major arrive
    as four spans; musically that is one chord, and one change event.
    """
    merged: List[Span] = []
    for s in spans:
        if merged and merged[-1].label == s.label and abs(merged[-1].end - s.start) < 1e-3:
            merged[-1] = Span(merged[-1].start, s.end, s.label)
        else:
            merged.append(s)
    return merged


def _grid_lines(grid: RhythmGrid, unit: str, until: float) -> np.ndarray:
    """Every beat (``unit="beat"``) or bar line (``"bar"``) of ``grid`` up to ``until`` seconds.

    Uses the grid's own markers when it has them and carries on metronomically
    past the last one — a MIDI file often ends before its audio does.
    """
    if unit not in ("beat", "bar"):
        raise ValueError(f"unit must be 'beat' or 'bar', got {unit!r}")
    step = grid.beat_duration if unit == "beat" else grid.bar_duration
    marks = grid.beats if unit == "beat" else grid.downbeats
    if marks is not None and len(marks) >= 2:
        lines = np.asarray(marks, dtype=float)
    else:
        lines = np.array([float(grid.start_offset)])
    if lines[-1] < until:
        extra = lines[-1] + step * np.arange(1, int(np.ceil((until - lines[-1]) / step)) + 1)
        lines = np.concatenate([lines, extra])
    return lines


def _snap(times, lines: np.ndarray) -> np.ndarray:
    """Move each time to the nearest of ``lines`` (sorted)."""
    times = np.asarray(times, dtype=float)
    right = np.clip(np.searchsorted(lines, times), 1, len(lines) - 1)
    left = right - 1
    return np.where(times - lines[left] <= lines[right] - times, lines[left], lines[right])


def _snap_spans(spans: Sequence[Span], lines: np.ndarray) -> List[Span]:
    if not spans or len(lines) < 2:
        return list(spans)
    starts = _snap([s.start for s in spans], lines)
    ends = _snap([s.end for s in spans], lines)
    # a span squeezed to nothing between two lines disappears; equal neighbours rejoin
    return _merge([Span(float(a), float(b), s.label) for s, a, b in zip(spans, starts, ends) if b > a])


def _span_at(spans: Sequence[Span], starts: Sequence[float], t: float) -> Optional[Span]:
    i = bisect_right(starts, t) - 1
    if i < 0:
        return None
    return spans[i] if t < spans[i].end else None


@dataclass
class Score:
    duration: float = 0.0
    chords: List[Span] = field(default_factory=list)
    keys: List[Span] = field(default_factory=list)
    sections: List[Span] = field(default_factory=list)
    beats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    beat_numbers: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    downbeats: np.ndarray = field(default_factory=lambda: np.zeros(0))
    time_signature: Tuple[int, int] = (4, 4)
    melody_vocal: List[MidiNote] = field(default_factory=list)
    melody_instrumental: List[MidiNote] = field(default_factory=list)

    def __post_init__(self):
        self._chord_starts = [s.start for s in self.chords]
        self._key_starts = [s.start for s in self.keys]
        self._section_starts = [s.start for s in self.sections]

    # ── what is written at time t ─────────────────────────────────────────────

    def chord_at(self, t: float) -> Optional[Span]:
        return _span_at(self.chords, self._chord_starts, t)

    def key_at(self, t: float) -> Optional[Span]:
        return _span_at(self.keys, self._key_starts, t)

    def section_at(self, t: float) -> Optional[Span]:
        return _span_at(self.sections, self._section_starts, t)

    def next_chord(self, t: float) -> Optional[Span]:
        i = bisect_right(self._chord_starts, t)
        return self.chords[i] if i < len(self.chords) else None

    def melody_at(self, t: float, voice: str = "instrumental") -> Optional[MidiNote]:
        """The melody note sounding at ``t`` (``voice``: vocal | instrumental)."""
        for n in self._voice(voice):
            if n.time > t:
                break
            if t < n.time + n.duration:
                return n
        return None

    def bar_beat_at(self, t: float) -> Tuple[int, int]:
        """(bar, beat), both 1-based, from the transcribed beats; (0, 0) before the first."""
        if len(self.beats) == 0 or t < self.beats[0]:
            return 0, 0
        bar = int(np.searchsorted(self.downbeats, t, side="right"))
        beat = int(self.beat_numbers[np.searchsorted(self.beats, t, side="right") - 1])
        return bar, beat

    # ── the same information as trigger events ────────────────────────────────

    def chord_changes(self) -> List[MidiNote]:
        """One event per chord change; ``pitch`` is the new root's pitch class.

        Going into a no-chord stretch is not an event — nothing new arrives.
        """
        events = []
        for s in self.chords:
            root, _, _ = parse_chord(s.label)
            if root is not None:
                events.append(MidiNote(time=s.start, pitch=root, velocity=CHANGE_VELOCITY,
                                       channel=SCORE_CHANNEL, duration=s.duration))
        return events

    def section_changes(self) -> List[MidiNote]:
        """One event per section start; ``pitch`` indexes the section in song order."""
        return [MidiNote(time=s.start, pitch=i, velocity=CHANGE_VELOCITY,
                         channel=SCORE_CHANNEL, duration=s.duration)
                for i, s in enumerate(self.sections)]

    def melody(self, voice: str = "instrumental") -> List[MidiNote]:
        return list(self._voice(voice))

    def grid(self, fps: int = 24) -> Optional[RhythmGrid]:
        """The transcribed beats as a RhythmGrid, or None with fewer than 2 beats."""
        if len(self.beats) < 2:
            return None
        # Tempo from a line fitted through every beat, not from the median gap:
        # SheetSage2 rounds times to 10 ms, so a 461.5 ms beat reads 460 ms and
        # the median reports 130.43 BPM for any song at 130.
        seconds_per_beat = float(np.polyfit(np.arange(len(self.beats)), self.beats, 1)[0])
        return RhythmGrid(bpm=60.0 / seconds_per_beat, time_signature=self.time_signature, fps=fps,
                          beats=self.beats, downbeats=self.downbeats,
                          start_offset=float(self.beats[0]))

    def snap_to(self, grid: RhythmGrid, unit: str = "bar",
                melody_division: Optional[int] = None) -> "Score":
        """This score re-timed onto a grid you trust (typically the song's MIDI).

        SheetSage2 hears harmony well and rhythm badly: it assumes 4/4 whatever
        the song is, its bar lines wander, and its beats can sit ~0.2 s off for
        long stretches. So when a trusted grid exists, the score keeps *what*
        happens and the grid decides *when*:

        - chord, key and section boundaries move to the nearest ``unit`` of the
          grid: ``"bar"`` (default) or ``"beat"``. Prefer bars. Measured against
          Reaper MIDI on two songs at 130 BPM, SheetSage2's chord changes sat
          0.25-0.65 s away from the real bar line. That is more than half a beat
          (0.23 s), so snapping to the nearest *beat* is a coin flip — on a 5/4
          song it put 32 of 45 changes on beat 5 instead of beat 1 — while the
          nearest *bar* was right every time. Use ``"beat"`` only for music whose
          harmony really moves inside the bar and whose transcribed beats you
          have checked to be tight;
        - beats, downbeats and time signature become the grid's, so
          ``bar_beat_at`` agrees with everything else that follows that grid;
        - melody keeps its own timing unless ``melody_division`` is given
          (grid lines per beat: 2 = eighths, 4 = sixteenths) — a melody is
          rhythm, and coarse snapping would flatten it.

        Returns a new Score; this one is left as transcribed.
        """
        beats = _grid_lines(grid, "beat", self.duration)
        downbeats = _grid_lines(grid, "bar", self.duration)
        lines = beats if unit == "beat" else _grid_lines(grid, unit, self.duration)

        # number each beat within its bar: 1 on a downbeat, 0 before the first bar
        bar_of = np.searchsorted(downbeats, beats + 1e-6, side="right") - 1
        since_bar = beats - downbeats[np.maximum(bar_of, 0)]
        beat_numbers = np.where(bar_of >= 0, np.round(since_bar / grid.beat_duration).astype(int) + 1, 0)

        def snap_melody(notes: List[MidiNote]) -> List[MidiNote]:
            if not melody_division or not notes:
                return list(notes)
            step = grid.beat_duration / melody_division
            fine = beats[0] + step * np.arange(int(np.ceil((self.duration - beats[0]) / step)) + 2)
            times = _snap([n.time for n in notes], fine)
            return [MidiNote(time=float(t), pitch=n.pitch, velocity=n.velocity,
                             channel=n.channel, duration=max(n.duration, step))
                    for n, t in zip(notes, times)]

        return Score(
            duration=self.duration,
            chords=_snap_spans(self.chords, lines),
            keys=_snap_spans(self.keys, lines),
            sections=_snap_spans(self.sections, lines),
            beats=beats, beat_numbers=beat_numbers, downbeats=downbeats,
            time_signature=grid.time_signature,
            melody_vocal=snap_melody(self.melody_vocal),
            melody_instrumental=snap_melody(self.melody_instrumental),
        )

    def _voice(self, voice: str) -> List[MidiNote]:
        if voice == "vocal":
            return self.melody_vocal
        if voice == "instrumental":
            return self.melody_instrumental
        raise ValueError(f"voice must be 'vocal' or 'instrumental', got {voice!r}")


def _rows(path: Path) -> List[List[str]]:
    if not path.is_file():
        return []
    return [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_spans(path: Path) -> List[Span]:
    return _merge([Span(float(r[0]), float(r[1]), r[2].strip()) for r in _rows(path) if len(r) >= 3])


def _read_melody(path: Path) -> List[MidiNote]:
    return [MidiNote(time=float(r[0]), pitch=int(float(r[2])), velocity=CHANGE_VELOCITY,
                     channel=SCORE_CHANNEL, duration=float(r[1]) - float(r[0]))
            for r in _rows(path) if len(r) >= 3]


def has_score(folder: str | Path) -> bool:
    """True when ``folder`` looks like a SheetSage2 output (any of its .lab files)."""
    p = Path(folder)
    return p.is_dir() and any((p / n).is_file() for n in ("chord.lab", "structure.lab", "beat.lab"))


def read_score(folder: str | Path) -> Score:
    """Load the SheetSage2 output in ``folder``."""
    p = Path(folder)
    if not p.is_dir():
        raise FileNotFoundError(p)

    chords = _read_spans(p / "chord.lab")
    keys = _read_spans(p / "key.lab")
    sections = _read_spans(p / "structure.lab")

    beat_rows = [r for r in _rows(p / "beat.lab") if len(r) >= 2]
    beats = np.array([float(r[0]) for r in beat_rows])
    beat_numbers = np.array([int(r[1]) for r in beat_rows], dtype=int)
    time_signature = (4, 4)
    if beat_rows and len(beat_rows[0]) >= 4:
        time_signature = (int(beat_rows[0][2]), int(beat_rows[0][3]))

    downbeats = np.array([float(r[0]) for r in _rows(p / "downbeat.lab")])
    if len(downbeats) == 0 and len(beats):
        downbeats = beats[beat_numbers == 1]

    melody_vocal = _read_melody(p / "melody_vocal.lab")
    melody_instrumental = _read_melody(p / "melody_instrumental.lab")

    ends = [s[-1].end for s in (chords, keys, sections) if s]
    ends += [n.time + n.duration for m in (melody_vocal, melody_instrumental) for n in m[-1:]]
    if len(beats):
        ends.append(float(beats[-1]))

    return Score(
        duration=max(ends, default=0.0),
        chords=chords, keys=keys, sections=sections,
        beats=beats, beat_numbers=beat_numbers, downbeats=downbeats,
        time_signature=time_signature,
        melody_vocal=melody_vocal, melody_instrumental=melody_instrumental,
    )

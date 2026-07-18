"""Read MIDI files into a RhythmGrid + note list.

Conventions
-----------
- ``set_tempo`` events drive BPM (last value wins) and ``time_signature``
  meta events drive the bar formula (first value wins). Both assume a
  static value per file, typical for sequenced MIDI.
- With tempo meta present, the grid is metronomic: beats/downbeats laid
  from t=0 (DAW bar 1) at the declared BPM and time signature, so bars
  match the session's bar lines exactly.
- Without tempo meta, kick hits (``KICK_NOTE`` = 36) anchor the grid as a
  fallback; without kicks, the first note does.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
import numpy as np

from core.rhythm.grid import RhythmGrid

KICK_NOTE = 36  # General MIDI Bass Drum 1 ("C2" in Yamaha pitch notation)


@dataclass
class MidiNote:
    time: float       # seconds from start
    pitch: int        # 0..127
    velocity: int     # 1..127
    channel: int      # 0..15
    duration: float   # seconds; 0.0 if note_off not seen


def read_midi(
    path: str | Path,
    time_signature: Optional[Tuple[int, int]] = None,
    fps: int = 24,
) -> Tuple[RhythmGrid, List[MidiNote]]:
    """Parse ``path`` and return (grid, notes).

    ``time_signature=None`` (default) uses the file's ``time_signature``
    meta event, falling back to 4/4; pass a tuple to override.

    With ``set_tempo`` present the grid is metronomic from t=0 (see module
    docstring); otherwise it is anchored on kick hits when present
    (>=2 kicks), or on the first note.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    mid = mido.MidiFile(str(p))

    notes: List[MidiNote] = []
    starts: Dict[Tuple[int, int], Tuple[float, int]] = {}
    bpm: Optional[float] = None
    ts_meta: Optional[Tuple[int, int]] = None

    t = 0.0
    for msg in mid:
        t += msg.time  # mido cumulative iteration yields seconds
        if msg.type == "set_tempo":
            bpm = float(mido.tempo2bpm(msg.tempo))
        elif msg.type == "time_signature" and ts_meta is None:
            ts_meta = (int(msg.numerator), int(msg.denominator))
        elif msg.type == "note_on" and msg.velocity > 0:
            starts[(msg.channel, msg.note)] = (t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (msg.channel, msg.note)
            if key in starts:
                start_t, vel = starts.pop(key)
                notes.append(MidiNote(time=start_t, pitch=msg.note, velocity=vel,
                                      channel=msg.channel, duration=t - start_t))

    # flush unterminated notes (file truncated before note_off)
    for (ch, pitch), (start_t, vel) in starts.items():
        notes.append(MidiNote(time=start_t, pitch=pitch, velocity=vel,
                              channel=ch, duration=0.0))
    notes.sort(key=lambda n: n.time)

    if time_signature is None:
        time_signature = ts_meta if ts_meta else (4, 4)

    kicks = np.array([n.time for n in notes if n.pitch == KICK_NOTE], dtype=float)

    # Choose BPM: prefer set_tempo; fall back to kick spacing only if missing.
    tempo_meta = bpm is not None
    if bpm is None:
        if len(kicks) >= 2:
            kick_gap = float(np.median(np.diff(kicks)))
            # assume kick lands on every beat in the absence of better info
            bpm = 60.0 / kick_gap if kick_gap > 0 else 120.0
        else:
            bpm = 120.0

    beat_dur = 60.0 / bpm
    beats_per_bar = time_signature[0]

    if tempo_meta:
        # Metronomic grid anchored at t=0 (DAW bar 1): bars follow the
        # session's bar lines, independent of where the drums actually hit.
        bar_dur = beat_dur * beats_per_bar
        last_t  = (notes[-1].time if notes else 0.0) + bar_dur
        return (
            RhythmGrid(
                bpm=bpm,
                time_signature=time_signature,
                fps=fps,
                beats=np.arange(0.0, last_t, beat_dur),
                downbeats=np.arange(0.0, last_t, bar_dur),
                start_offset=0.0,
            ),
            notes,
        )

    if len(kicks) >= 2:
        # Auto-classify kick role from its spacing relative to the beat duration.
        # ``beats_per_kick`` is how many beats elapse between consecutive kicks.
        avg_gap = float(np.median(np.diff(kicks)))
        beats_per_kick = max(1, int(round(avg_gap / beat_dur)))
        # Every Nth kick is a downbeat (N == kicks per bar).
        kicks_per_bar = max(1, beats_per_bar // beats_per_kick) if beats_per_kick <= beats_per_bar else 1
        downbeats = kicks[::kicks_per_bar]

        anchor = float(downbeats[0])
        last_t = float(kicks[-1]) + beat_dur
        n_beats = max(2, int((last_t - anchor) / beat_dur) + 1)
        beats_arr = np.array([anchor + i * beat_dur for i in range(n_beats)], dtype=float)

        return (
            RhythmGrid(
                bpm=bpm,
                time_signature=time_signature,
                fps=fps,
                beats=beats_arr,
                downbeats=downbeats,
                start_offset=anchor,
            ),
            notes,
        )

    anchor = float(notes[0].time) if notes else 0.0
    return (
        RhythmGrid(
            bpm=bpm,
            time_signature=time_signature,
            fps=fps,
            start_offset=anchor,
        ),
        notes,
    )

"""Read MIDI files into a RhythmGrid + note list.

Conventions
-----------
- ``set_tempo`` events drive BPM (last value wins; one tempo per file).
- ``time_signature`` meta events drive the bar formula, and every one of
  them counts: a change takes effect at the bar line it sits on, so a tune
  in 5/4 with five bars of 6/4 in the middle gets bar lines where the DAW
  has them. When the file does not carry the changes (exported without
  them), declare them by bar with ``meter_changes``.
- With tempo meta present, the grid is metronomic: beats/downbeats laid
  from t=0 (DAW bar 1) at the declared BPM and time signature, so bars
  match the session's bar lines exactly.
- Without tempo meta, kick hits (``KICK_NOTE`` = 36) anchor the grid as a
  fallback; without kicks, the first note does.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

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
    track: str = ""   # the track's name in the file ("kick", "bass", ...)
    track_index: int = -1


def select_notes(notes: Sequence["MidiNote"], spec: dict) -> List["MidiNote"]:
    """The notes a trigger spec asks for: ``notes: [36]`` (pitches), ``track: kick``
    (a name, an index or a list of them) or both.

    A DAW export puts every instrument on its own track, often all on channel 0
    and with the same pitches (three tracks playing note 36), so the pitch alone
    does not say who played. Tracks that share a name are taken together; a note
    doubled on two tracks (one part feeding two synths) counts once.
    """
    out = list(notes)
    if "track" in spec:
        wanted = spec["track"] if isinstance(spec["track"], (list, tuple)) else [spec["track"]]
        names = {str(w).strip().lower() for w in wanted if not isinstance(w, int)}
        indexes = {w for w in wanted if isinstance(w, int)}
        out = [n for n in out if n.track.lower() in names or n.track_index in indexes]
        if not out:
            known = sorted({f"{n.track_index}:{n.track}" for n in notes})
            raise ValueError(f"track {spec['track']!r} matches no note; tracks in the file: {known}")
        seen, unique = set(), []
        for n in out:
            key = (round(n.time, 4), n.pitch)
            if key not in seen:
                seen.add(key)
                unique.append(n)
        out = unique
    if "notes" in spec or "track" not in spec:
        pitches = set(spec.get("notes", []))
        out = [n for n in out if n.pitch in pitches]
    return out


MeterChanges = Sequence[Tuple[int, Tuple[int, int]]]   # (bar number, 1-based) -> (num, den)


def parse_meter_changes(text: str) -> List[Tuple[int, Tuple[int, int]]]:
    """``"22:6/4,27:5/4"`` -> ``[(22, (6, 4)), (27, (5, 4))]`` (bar numbers as the DAW shows them)."""
    changes = []
    for item in filter(None, (part.strip() for part in text.split(","))):
        try:
            bar, sig = item.split(":")
            num, den = sig.split("/")
            changes.append((int(bar), (int(num), int(den))))
        except ValueError:
            raise ValueError(f"meter change {item!r}: expected BAR:NUM/DEN, e.g. 22:6/4") from None
        if changes[-1][0] < 1 or min(changes[-1][1]) < 1:
            raise ValueError(f"meter change {item!r}: bar and signature must be positive")
    return sorted(changes)


def _lay_bars(total_beats: float, first: Tuple[int, int],
              at_beat: Sequence[Tuple[float, Tuple[int, int]]], at_bar: MeterChanges):
    """Walk the song bar by bar and return (downbeat positions in quarter-note beats, beats per bar).

    ``at_beat`` are the file's own meter events (they take effect at the bar
    line they sit on); ``at_bar`` are declared by 1-based bar number and win.
    """
    by_bar = dict(at_bar)
    pending = sorted(at_beat)
    sig, pos, bar = first, 0.0, 1
    downbeats, bar_beats = [], []
    while pos < total_beats or not downbeats:
        while pending and pending[0][0] <= pos + 1e-6:
            sig = pending.pop(0)[1]
        sig = by_bar.get(bar, sig)
        downbeats.append(pos)
        bar_beats.append(sig[0])
        pos += sig[0] * 4.0 / sig[1]          # bar length in quarter notes
        bar += 1
    return np.array(downbeats), np.array(bar_beats, dtype=int)


def read_midi(
    path: str | Path,
    time_signature: Optional[Tuple[int, int]] = None,
    fps: int = 24,
    meter_changes: Optional[MeterChanges] = None,
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
    starts: Dict[Tuple[int, int, int], Tuple[float, int]] = {}
    bpm: Optional[float] = None
    ts_meta: Optional[Tuple[int, int]] = None
    ts_events: List[Tuple[float, Tuple[int, int]]] = []    # (seconds, signature), every one

    # Merge the tracks ourselves (as mido does: by tick, stable) so every note
    # remembers the track it came from.
    events = []
    names = []
    for ti, track in enumerate(mid.tracks):
        names.append(next((m.name.strip() for m in track if m.type == "track_name"), ""))
        tick = 0
        for msg in track:
            tick += msg.time
            events.append((tick, ti, msg))
    events.sort(key=lambda e: e[0])

    t, last_tick, tempo = 0.0, 0, 500000
    for tick, ti, msg in events:
        t += mido.tick2second(tick - last_tick, mid.ticks_per_beat, tempo)
        last_tick = tick
        if msg.type == "set_tempo":
            tempo = msg.tempo
            bpm = float(mido.tempo2bpm(msg.tempo))
        elif msg.type == "time_signature":
            ts_events.append((t, (int(msg.numerator), int(msg.denominator))))
            if ts_meta is None:
                ts_meta = ts_events[-1][1]
        elif msg.type == "note_on" and msg.velocity > 0:
            starts[(ti, msg.channel, msg.note)] = (t, msg.velocity)
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (ti, msg.channel, msg.note)
            if key in starts:
                start_t, vel = starts.pop(key)
                notes.append(MidiNote(time=start_t, pitch=msg.note, velocity=vel,
                                      channel=msg.channel, duration=t - start_t,
                                      track=names[ti], track_index=ti))

    # flush unterminated notes (file truncated before note_off)
    for (ti, ch, pitch), (start_t, vel) in starts.items():
        notes.append(MidiNote(time=start_t, pitch=pitch, velocity=vel,
                              channel=ch, duration=0.0, track=names[ti], track_index=ti))
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
        later   = [(sec / beat_dur, sig) for sec, sig in ts_events if sec > 1e-6]
        if later or meter_changes:
            # The meter moves along the song: lay the bars one by one.
            downbeats, bar_beats = _lay_bars(last_t / beat_dur, time_signature, later, meter_changes or ())
            return (
                RhythmGrid(
                    bpm=bpm, time_signature=time_signature, fps=fps,
                    beats=np.arange(0.0, float(downbeats[-1]) * beat_dur + bar_dur, beat_dur),
                    downbeats=downbeats * beat_dur, bar_beats=bar_beats, start_offset=0.0,
                ),
                notes,
            )
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


def shift_in_time(grid: RhythmGrid, notes: List[MidiNote], seconds: float) -> None:
    """Move a MIDI reading ``seconds`` later so it lines up with the audio.

    A MIDI file counts from DAW bar 1; the audio it belongs to may start a
    little later (encoder padding, a trimmed export). Everything that carries
    a time moves together — beats, downbeats, the grid's anchor and the notes —
    so ``t_audio = t_midi + seconds`` holds for all of them. In place.
    """
    if seconds == 0.0:
        return
    if grid.beats is not None:
        grid.beats = grid.beats + seconds
    if grid.downbeats is not None:
        grid.downbeats = grid.downbeats + seconds
    grid.start_offset += seconds
    for n in notes:
        n.time += seconds

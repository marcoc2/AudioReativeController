"""MIDI automation reader — CC lanes and note-velocity tracks as per-frame floats.

Converts discrete MIDI events into interpolated float arrays that ARCPipeline
can reference as named sources, identical in interface to audio features.

Source path syntax (in scene YAML):
    midi.ch<N>.cc.<num>      CC lane, channel N, controller number
    midi.ch<N>.note.<pitch>  note-velocity lane, channel N, MIDI pitch
    midi.cc.<num>            shorthand: first channel found with this CC
    midi.note.<pitch>        shorthand: first channel found with this pitch

Interpolation:
    CC lanes      — linear interpolation between consecutive events;
                    0.0 before the first event, hold last value after.
    Note lanes    — value equals velocity/127 while the note is held (note_on
                    to note_off); 0.0 between notes.  Draw velocity curves in
                    Reaper by stacking held notes with the desired velocities.

All values are normalised to 0.0..1.0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mido
import numpy as np


class MidiAutomationReader:
    """Parse a MIDI file and expose CC / note-velocity lanes as per-frame arrays."""

    def __init__(
        self,
        path: str | Path,
        fps: int = 24,
        duration: Optional[float] = None,
    ) -> None:
        self._fps = fps
        self._lanes: Dict[str, np.ndarray] = {}
        self._duration = duration or 0.0
        self._parse(Path(path))

    # ------------------------------------------------------------------
    # public API

    def get(self, lane: str, frame_idx: int) -> float:
        """Return the normalised 0..1 value for *lane* at *frame_idx*.

        Returns 0.0 for unknown lanes or out-of-range indices.
        """
        arr = self._lanes.get(lane)
        if arr is None:
            return 0.0
        return float(arr[min(frame_idx, len(arr) - 1)])

    @property
    def available_lanes(self) -> List[str]:
        return sorted(self._lanes.keys())

    @property
    def duration(self) -> float:
        return self._duration

    # ------------------------------------------------------------------
    # parsing

    def _parse(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(path)

        mid = mido.MidiFile(str(path))

        cc_raw:   Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
        note_raw: Dict[Tuple[int, int], List[Tuple[float, int, str]]] = {}
        active:   Dict[Tuple[int, int], Tuple[float, int]] = {}

        total_t = 0.0
        t = 0.0
        for msg in mid:
            t += msg.time
            total_t = max(total_t, t)

            if msg.type == "control_change":
                key = (msg.channel, msg.control)
                cc_raw.setdefault(key, []).append((t, msg.value))

            elif msg.type == "note_on" and msg.velocity > 0:
                key = (msg.channel, msg.note)
                active[key] = (t, msg.velocity)
                note_raw.setdefault(key, []).append((t, msg.velocity, "on"))

            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                key = (msg.channel, msg.note)
                if key in active:
                    _, vel = active.pop(key)
                    note_raw.setdefault(key, []).append((t, vel, "off"))

        # flush notes still open at file end
        for key, (_, vel) in active.items():
            note_raw.setdefault(key, []).append((total_t, vel, "off"))

        if self._duration == 0.0:
            self._duration = total_t

        n = max(1, int(self._duration * self._fps) + 1)

        for (ch, cc_num), events in cc_raw.items():
            arr = self._interp_cc(events, n)
            full  = f"ch{ch}.cc.{cc_num}"
            short = f"cc.{cc_num}"
            self._lanes[full] = arr
            if short not in self._lanes:
                self._lanes[short] = arr

        for (ch, pitch), events in note_raw.items():
            arr = self._build_note_lane(events, n)
            full  = f"ch{ch}.note.{pitch}"
            short = f"note.{pitch}"
            self._lanes[full] = arr
            if short not in self._lanes:
                self._lanes[short] = arr

    # ------------------------------------------------------------------
    # interpolation helpers

    def _interp_cc(
        self, events: List[Tuple[float, int]], n_frames: int
    ) -> np.ndarray:
        times = np.array([e[0] for e in events], dtype=np.float64)
        vals  = np.array([e[1] / 127.0 for e in events], dtype=np.float32)
        frame_times = np.arange(n_frames, dtype=np.float64) / self._fps
        return np.interp(
            frame_times, times, vals,
            left=0.0, right=float(vals[-1]),
        ).astype(np.float32)

    def _build_note_lane(
        self, events: List[Tuple[float, int, str]], n_frames: int
    ) -> np.ndarray:
        arr   = np.zeros(n_frames, dtype=np.float32)
        on_t  = None
        on_vel = 0.0
        for t, vel, etype in sorted(events, key=lambda e: e[0]):
            if etype == "on":
                on_t  = t
                on_vel = vel / 127.0
            elif etype == "off" and on_t is not None:
                f0 = max(0, min(int(on_t  * self._fps), n_frames))
                f1 = max(0, min(int(t     * self._fps), n_frames))
                arr[f0:f1] = on_vel
                on_t = None
        return arr

"""Extract drum hits from an audio stem as MidiNote events.

Bridge between recorded audio tracks and the trigger system: transients
detected in a (preferably single-instrument) stem become events with the
same shape as MIDI notes, so triggers, min_velocity and gravity behave
identically whichever source they come from.

Onset strength maps to velocity 1..127 (normalised to the loudest hit in
the file), so soft ghost notes can be filtered with ``min_velocity``.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import librosa
import numpy as np

from core.rhythm.midi_reader import MidiNote

AUDIO_PITCH = -1   # synthetic pitch marking audio-derived hits
_HOP = 512


def read_onsets(
    path: str | Path,
    threshold: float = 0.3,
    min_gap: float = 0.05,
) -> List[MidiNote]:
    """Detect transients in ``path`` and return them as MidiNote events.

    threshold — peak-picking delta over the local envelope average
                (higher = fewer, stronger hits).
    min_gap   — minimum spacing between hits in seconds (debounce).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    y, sr = librosa.load(str(p), sr=None, mono=True)
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=_HOP)
    frames = librosa.onset.onset_detect(
        onset_envelope=env,
        sr=sr,
        hop_length=_HOP,
        delta=threshold,
        wait=max(1, int(round(min_gap * sr / _HOP))),
        backtrack=False,
        normalize=True,
    )
    if len(frames) == 0:
        return []

    times = librosa.frames_to_time(frames, sr=sr, hop_length=_HOP)
    strengths = env[frames]
    smax = float(strengths.max())

    notes = []
    for t, s in zip(times, strengths):
        vel = int(round(1 + 126.0 * (float(s) / smax))) if smax > 0 else 100
        notes.append(MidiNote(time=float(t), pitch=AUDIO_PITCH,
                              velocity=vel, channel=0, duration=0.0))
    return notes

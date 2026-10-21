"""Bar.beat, as the DAW and the timeline count them, and song seconds."""
from __future__ import annotations

import numpy as np


def bar_beat_to_time(text: str, grid) -> float:
    """"18.3" (bar 18, beat 3, as the DAW and the timeline count them) -> song seconds."""
    parts = [p for p in str(text).strip().split(".") if p != ""]
    if not parts or len(parts) > 2 or not all(p.isdigit() for p in parts):
        raise ValueError(f"{text!r}: write bar.beat, e.g. 18.3")
    bar, beat = int(parts[0]), int(parts[1]) if len(parts) > 1 else 1
    db = getattr(grid, "downbeats", None)
    if db is None or not len(db):
        raise ValueError("load the project first (no bars yet)")
    if not 1 <= bar <= len(db):
        raise ValueError(f"bar {bar}: the song has bars 1..{len(db)}")
    t0 = float(db[bar - 1])
    t1 = float(db[bar]) if bar < len(db) else t0 + float(grid.bar_duration)
    beats = getattr(grid, "beats", None)
    inside = [float(b) for b in (beats if beats is not None else []) if t0 - 1e-3 <= b < t1 - 1e-3]
    if not inside:                                   # no beat markers: split the bar evenly
        n = max(1, int(round((t1 - t0) / max(1e-6, float(grid.beat_duration)))))
        inside = [t0 + k * (t1 - t0) / n for k in range(n)]
    if not 1 <= beat <= len(inside):
        raise ValueError(f"bar {bar} has beats 1..{len(inside)}")
    return inside[beat - 1]


def time_to_bar_beat(t: float, grid) -> str:
    """Song seconds -> "bar.beat" (the beat it falls in)."""
    db = getattr(grid, "downbeats", None)
    if db is None or not len(db):
        return ""
    i = max(0, int(np.searchsorted(db, t + 1e-3, side="right")) - 1)
    beats = getattr(grid, "beats", None)
    t0 = float(db[i])
    if beats is not None and len(beats):
        k = int(np.sum((beats >= t0 - 1e-3) & (beats <= t + 1e-3)))
    else:
        k = int((t - t0) // max(1e-6, float(grid.beat_duration))) + 1
    return f"{i + 1}.{max(1, k)}"

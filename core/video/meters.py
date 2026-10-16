"""A read-out of what drives a scene, drawn over the picture: for judging an effect.

A band at the top names the effect; a strip at the bottom scrolls the music past a
playhead, a few seconds of the past and a little of what comes: one lane per knob (its
value as a trace, the same knob spec the effect is given, so a YAML anchor can share it),
lanes of notes (a piano roll of a track or of the transcription's melody) and of chords
(the transcription's chord names), over the bar lines, the beats and the clip cuts.

    - source: meters
      title: pixel sort
      subtitle: streaks grow while the organ holds a chord
      past: 2.5             # seconds shown before the playhead
      future: 0.8           # and after it
      lanes:
        - {label: organ, roll: {track: organ}}
        - {label: chord, chords: {score: input/me_diz/sheetsage, snap: beat}}
        - {label: organ adsr -> amount, knob: *amount}      # any knob form
        - {label: kick, knob: {hits: {track: kick}, envelope: 0.2}, range: [0, 1]}

A knob's lane is scaled to what it does over the next minute; ``range: [lo, hi]`` fixes
it, ``range: window`` scales it to what is on screen (a count that only grows). Quote a
label with a comma in it (a YAML flow mapping splits there).
Put it last: it draws over everything below it. CPU (PIL).
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

PALETTE = [(255, 196, 64), (90, 200, 255), (255, 110, 150), (140, 235, 120),
           (200, 150, 255), (255, 150, 80), (120, 255, 220)]
RATE = 60.0          # knob samples per second
AHEAD = 60.0         # seconds sampled at the start to scale a knob's lane


class _KnobLane:
    def __init__(self, f, scale):
        self.f = f
        self.times: list = []
        self.values: list = []
        self.windowed = scale == "window"
        self.lo, self.hi = (None, None) if scale in (None, "window") else (float(scale[0]), float(scale[1]))

    def fill(self, until: float, start: float) -> None:
        """Sample the knob up to ``until``, in time order (knobs expect it)."""
        t = self.times[-1] + 1.0 / RATE if self.times else start
        while t <= until:
            self.times.append(t)
            self.values.append(float(self.f(t)))
            t += 1.0 / RATE
        if self.lo is None or self.hi is None:
            v = np.asarray(self.values)
            lo, hi = float(v.min()), float(v.max())
            if hi - lo < 1e-9:
                lo, hi = lo - 0.5, hi + 0.5
            self.lo = lo if self.lo is None else self.lo
            self.hi = hi if self.hi is None else self.hi

    def window(self, a: float, b: float):
        times = np.asarray(self.times)
        i, j = np.searchsorted(times, a), np.searchsorted(times, b, side="right")
        return times[i:j], np.asarray(self.values[i:j])

    def at(self, t: float) -> float:
        i = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        return self.values[i]


class Meters:
    """Post-op: see the module docstring."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: float,
                 features_at=None, onset_loader=None, grid=None, cuts=None):
        from core.video.layers import _layer_events, _music_knob
        self.W, self.H = int(width), int(height)
        self.grid, self._cuts = grid, cuts
        self.title = str(spec.get("title", ""))
        self.subtitle = str(spec.get("subtitle", ""))
        self.past, self.future = float(spec.get("past", 2.5)), float(spec.get("future", 0.8))
        k = self.H / 854.0
        self.lane_h = int(round(float(spec.get("lane", 38)) * k))
        self.font = self._font(max(9, int(round(12 * k))))
        self.big = self._font(max(12, int(round(19 * k))))
        self.lanes = []
        for i, lane in enumerate(spec.get("lanes") or []):
            color = tuple(int(c) for c in lane.get("color", PALETTE[i % len(PALETTE)]))
            label = str(lane.get("label", ""))
            if "knob" in lane:
                f = _music_knob(label or f"lane {i}", lane["knob"], notes, features_at, onset_loader, grid, cuts)
                self.lanes.append(("knob", label, color, _KnobLane(f, lane.get("range"))))
            elif "roll" in lane:
                evs = _layer_events(lane["roll"], notes, onset_loader, grid)
                if not evs:
                    raise ValueError(f"meters: lane {label!r} has no notes")
                self.lanes.append(("roll", label, color, evs))
            elif "chords" in lane:
                from core.video.composer import load_score
                self.lanes.append(("chords", label, color, load_score(lane["chords"], grid).chords))
            else:
                raise ValueError(f"meters: lane {label!r} needs knob:, roll: or chords:")
        self._known = (-np.inf, np.zeros(0))
        self._started: Optional[float] = None

    @staticmethod
    def _font(size: int):
        from PIL import ImageFont
        for name in ("arialbd.ttf", "DejaVuSans-Bold.ttf", "Arial Bold.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except OSError:
                continue
        return ImageFont.load_default(size=size)

    def _cut_times(self, t: float) -> np.ndarray:
        if self._cuts is None:
            return np.zeros(0)
        if t + 600.0 > self._known[0]:
            self._known = (t + 3600.0, np.asarray(self._cuts(t + 3600.0), dtype=float))
        return self._known[1]

    def _x(self, s: float, t: float) -> float:
        return (s - (t - self.past)) / (self.past + self.future) * self.W

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        from PIL import Image, ImageDraw
        if self._started is None:
            self._started = t
            for kind, _, _, lane in self.lanes:
                if kind == "knob":
                    lane.fill(t + AHEAD, t - self.past)
        for kind, _, _, lane in self.lanes:
            if kind == "knob":
                lane.fill(t + self.future + 1.0, self._started - self.past)
        img = Image.fromarray(frame).convert("RGBA")
        over = Image.new("RGBA", img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(over)
        W, H, lh = self.W, self.H, self.lane_h
        pad = max(3, lh // 8)
        if self.title or self.subtitle:
            th = int(lh * (1.9 if self.subtitle else 1.2))
            d.rectangle([0, 0, W, th], fill=(0, 0, 0, 165))
            d.text((pad * 2, pad), self.title, font=self.big, fill=(255, 255, 255, 255))
            if self.subtitle:
                d.text((pad * 2, pad + int(lh * 0.85)), self.subtitle, font=self.font, fill=(220, 220, 220, 255))
        if self.lanes:
            top = H - lh * len(self.lanes) - pad
            d.rectangle([0, top - pad, W, H], fill=(0, 0, 0, 195))
            a, b = t - self.past, t + self.future
            if self.grid is not None:                                  # beats dim, bars bright
                i = self.grid.bar_index(a)
                while self.grid.bar_start(i) <= b:
                    s0, s1 = self.grid.bar_start(i), self.grid.bar_start(i + 1)
                    n = max(1, self.grid.beats_in_bar(i))
                    for j in range(n):
                        x = self._x(s0 + (s1 - s0) * j / n, t)
                        if 0 <= x < W:
                            d.line([x, top - pad, x, H], fill=(255, 255, 255, 110 if j == 0 else 40),
                                   width=2 if j == 0 else 1)
                    i += 1
            for c in self._cut_times(t):                               # the clip cuts
                if a <= c <= b:
                    x = self._x(c, t)
                    d.line([x, top - pad, x, H], fill=(255, 60, 60, 230), width=2)
                    d.text((x + 3, top - pad - int(lh * 0.5)), "cut", font=self.font, fill=(255, 90, 90, 255))
            for k, (kind, label, color, lane) in enumerate(self.lanes):
                y0 = top + k * lh
                y1 = y0 + lh - 2
                getattr(self, "_" + kind)(d, lane, color, t, a, b, y0, y1)
                d.text((pad * 2, y0 + 1), label, font=self.font, fill=color + (255,))
            px = self._x(t, t)                                         # the playhead
            d.line([px, top - pad, px, H], fill=(255, 255, 255, 255), width=2)
        return np.asarray(Image.alpha_composite(img, over).convert("RGB"))

    def _knob(self, d, lane: _KnobLane, color, t, a, b, y0, y1) -> None:
        times, values = lane.window(a, b)
        if len(times) < 2:
            return
        lo, hi = lane.lo, lane.hi
        if lane.windowed:
            lo, hi = float(values.min()), float(values.max())
            if hi - lo < 1e-9:
                lo, hi = lo - 0.5, hi + 0.5
        v = np.clip((values - lo) / (hi - lo), 0.0, 1.0)
        top = y0 + (y1 - y0) * 0.42                                     # leave room for the label
        ys = y1 - v * (y1 - top)
        xs = self._x(times, t)
        pts = list(zip(xs.tolist(), ys.tolist()))
        d.polygon(pts + [(xs[-1], y1), (xs[0], y1)], fill=color + (70,))
        d.line(pts, fill=color + (255,), width=2)
        now = lane.at(t)
        nv = float(np.clip((now - lo) / (hi - lo), 0.0, 1.0))
        cy, cx, r = y1 - nv * (y1 - top), self._x(t, t), 4
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=color + (255,))
        d.text((self.W - 6, y0 + 1), f"{now:.2f}", font=self.font, fill=color + (255,), anchor="ra")

    def _roll(self, d, events, color, t, a, b, y0, y1) -> None:
        pitches = [e.pitch for e in events]
        lo, hi = min(pitches), max(pitches)
        span = max(1, hi - lo)
        top = y0 + (y1 - y0) * 0.42
        h = max(2.0, (y1 - top) / (span + 1))
        for e in events:
            end = e.time + max(e.duration, 0.06)
            if end < a or e.time > b:
                continue
            y = y1 - (e.pitch - lo + 1) * (y1 - top) / (span + 1)
            on = e.time <= t <= end
            d.rectangle([self._x(e.time, t), y, max(self._x(e.time, t) + 2, self._x(end, t)), y + h],
                        fill=color + (255 if on else 120,))

    def _chords(self, d, spans, color, t, a, b, y0, y1) -> None:
        for s in spans:
            if s.end < a or s.start > b or s.label == "N":
                continue
            on = s.start <= t < s.end
            x0, x1 = max(0.0, self._x(s.start, t)), min(float(self.W), self._x(s.end, t))
            if x1 - x0 < 3:
                continue
            top = y0 + (y1 - y0) * 0.42
            d.rectangle([x0 + 1, top, x1 - 1, y1], outline=color + (255 if on else 110,),
                        fill=color + (90 if on else 25,))
            name = s.label.replace(":maj", "").replace(":min", "m").replace(":", "")
            d.text((x0 + 5, (top + y1) / 2), name, font=self.font, anchor="lm",
                   fill=(255, 255, 255, 255 if on else 150))

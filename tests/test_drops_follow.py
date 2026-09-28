"""Drops that follow their region (SAM2 stand-in: a fake tracker that slides the mask)."""
import numpy as np
import pytest

from core.rhythm.midi_reader import MidiNote
from core.video.drops_follow import FollowedDrops, ink_walls, region_at


def _box_frame(x0=100, y0=100, size=100, W=320, H=240):
    """White paper with a black square outline drawn on it."""
    f = np.full((H, W, 3), 235, np.uint8)
    f[y0:y0 + size, x0:x0 + 3] = 0
    f[y0:y0 + size, x0 + size - 3:x0 + size] = 0
    f[y0:y0 + 3, x0:x0 + size] = 0
    f[y0 + size - 3:y0 + size, x0:x0 + size] = 0
    return f


class _SlidingTracker:
    """Follows a region by sliding its first mask ``dx`` pixels a frame."""

    def __init__(self, dx=2):
        self.dx, self.added, self.regions, self.frame = dx, [], {}, -1

    def set_frame(self, i, frame):
        self.frame = i

    def add(self, rid, frames, mask=None, point=None):
        self.added.append((rid, "mask" if mask is not None else "point"))
        if mask is None:
            mask = np.zeros((240, 320), bool)
            mask[int(point[1]) - 5:int(point[1]) + 5, int(point[0]) - 5:int(point[0]) + 5] = True
        self.regions[rid] = (mask, self.frame, frames)

    def step(self):
        out = {}
        for rid, (mask, born, frames) in list(self.regions.items()):
            k = self.frame - born
            if k >= frames:
                del self.regions[rid]
                continue
            out[rid] = np.roll(mask, self.dx * k, axis=1)
        return out

    def forget(self, rid):
        self.regions.pop(rid, None)

    @property
    def live(self):
        return len(self.regions)


def test_walls_and_the_region_inside_them():
    f = _box_frame()
    walls = ink_walls(f)
    assert walls[100, 150] and walls[150, 101] and not walls[150, 150] and not walls[20, 20]
    inside = region_at(walls, 150, 150)
    assert inside[150, 150] and inside[110, 110] and not inside[20, 20]
    assert 0.08 < inside.mean() < 0.13                      # ~92×92 of 320×240
    assert region_at(walls, 100, 150) is None                # on the line itself


def _drops(tracker, **spec):
    base = {"hits": {"notes": [38]}, "at": "center", "speed": 40, "grow": 0.5,
            "life": 2.0, "fade": 0.5, "rough": 0.0, "opacity": 1.0, "color": "complement"}
    base.update(spec)
    notes = [MidiNote(time=0.0, pitch=38, velocity=100, channel=9, duration=0.1)]
    return FollowedDrops(base, notes, 320, 240, 24, tracker)


def test_the_drop_fills_its_region_and_rides_with_it():
    tracker = _SlidingTracker(dx=2)
    drops = _drops(tracker)
    frames = [drops.process(np.roll(_box_frame(), 2 * i, axis=1), i / 24) for i in range(12)]
    assert tracker.added == [(0, "mask")]                    # the square's inside was the prompt
    last = frames[-1].astype(int)
    painted = np.abs(last - np.roll(_box_frame(), 22, axis=1).astype(int)).sum(axis=2) > 30
    ys, xs = np.nonzero(painted)
    # the paint filled the square and moved 22 px with it; nothing leaked outside the lines
    assert painted[150, 150 + 22] and painted[110, 110 + 22]
    assert xs.min() >= 100 + 22 and xs.max() <= 200 + 22 and ys.min() >= 100 and ys.max() <= 200
    assert painted.mean() > 0.08


def test_the_stain_dries_after_its_life():
    tracker = _SlidingTracker(dx=0)
    drops = _drops(tracker, life=0.5, fade=0.2)
    for i in range(20):
        out = drops.process(_box_frame(), i / 24)
    assert np.array_equal(out, _box_frame())
    assert tracker.live == 0


def test_a_leaking_region_falls_back_to_the_point():
    tracker = _SlidingTracker()
    drops = _drops(tracker)                                  # blank paper: the "region" is everything
    drops.process(np.full((240, 320, 3), 235, np.uint8), 0.0)
    assert tracker.added == [(0, "point")]


def test_a_cut_dries_the_stains():
    tracker = _SlidingTracker(dx=0)
    drops = _drops(tracker, life=4.0)
    for i in range(6):
        drops.process(_box_frame(), i / 24)
    dark = np.full((240, 320, 3), 20, np.uint8)             # another picture altogether
    for i in range(6, 6 + 12):
        out = drops.process(dark, i / 24)
    assert np.array_equal(out, dark) and tracker.live == 0


def test_spiral_is_refused():
    with pytest.raises(ValueError, match="blob"):
        _drops(_SlidingTracker(), shape="spiral")

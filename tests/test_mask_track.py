"""The mask track: shots, objects found and followed, labels and stats on disk, read back."""
import numpy as np
import pytest

from core.segment.mask_track import MaskTrack, find_cuts, pick_objects, write_mask_track

W, H, FPS = 160, 120, 10
RED, BLUE, YELLOW, GREEN = (220, 30, 30), (30, 40, 220), (240, 220, 20), (30, 200, 60)


def _frame(i):
    """Shot A (0-9): a red square sliding right, a blue one still, a yellow one from frame 5.
    Shot B (10-19): a dark picture with a green square."""
    if i < 10:
        f = np.full((H, W, 3), 240, np.uint8)
        f[40:80, 10 + 4 * i:50 + 4 * i] = RED
        f[10:30, 110:150] = BLUE
        if i >= 5:
            f[90:115, 100:130] = YELLOW
    else:
        f = np.full((H, W, 3), 15, np.uint8)
        f[30:90, 50:110] = GREEN
    return f


class _ColourSegmenter:
    """Finds each saturated colour as an object (plus the paper, and the red square's
    top half as a 'part'), and follows an object by its colour — a perfect tracker."""

    def __init__(self):
        self.frame, self.regions, self.auto_calls = None, {}, []

    def set_frame(self, i, frame):
        self.i, self.frame = i, frame

    def _by(self, frame, colour):
        return np.all(np.abs(frame.astype(int) - colour) < 10, axis=2)

    def auto_masks(self, frame, points_per_side=32, min_iou=0.6, min_stability=0.7):
        self.auto_calls.append(self.i)
        out = []
        for c in (RED, BLUE, YELLOW, GREEN):
            m = self._by(frame, c)
            if m.any():
                out.append(m)
                if c == RED:
                    part = m.copy()
                    part[60:] = False
                    out.append(part)
        out.append(np.ones((H, W), bool))            # "the whole picture": too big to follow
        return out

    def add(self, rid, frames, mask=None, point=None):
        colour = self.frame[mask].mean(axis=0)
        self.regions[rid] = (colour, self.i, frames)

    def step(self):
        out = {}
        for rid, (colour, born, frames) in list(self.regions.items()):
            if self.i - born >= frames:
                del self.regions[rid]
                continue
            out[rid] = self._by(self.frame, colour)
        return out

    def forget(self, rid):
        self.regions.pop(rid, None)


def test_cuts():
    starts, n = find_cuts(_frame(i) for i in range(20))
    assert starts == [0, 10] and n == 20


def test_pick_whole_skips_parts_and_the_background():
    seg = _ColourSegmenter()
    seg.set_frame(0, _frame(0))
    masks = seg.auto_masks(_frame(0))
    whole = pick_objects(masks, max_share=0.6)
    assert len(whole) == 2                                  # red and blue; not red's half, not the paper
    parts = pick_objects(masks, granularity="parts", max_share=0.6)
    assert parts[0].sum() == masks[1].sum()                 # the half comes first
    taken = masks[0]
    assert all(not (m & taken).any() for m in pick_objects(masks, taken=taken))


@pytest.fixture
def track(tmp_path):
    seg = _ColourSegmenter()
    frames = lambda: (_frame(i) for i in range(20))
    meta = write_mask_track("fake.mp4", frames, seg, str(tmp_path / "seg"), FPS,
                            start_time=13.0, reseed=0.3, log=lambda m: None)
    return MaskTrack(str(tmp_path / "seg")), meta, seg


def test_objects_found_per_shot_and_reseed(track):
    mt, meta, seg = track
    assert meta["shots"] == [{"first": 0, "last": 9}, {"first": 10, "last": 19}]
    by = {tuple(o["colour"]): o for o in meta["objects"]}
    red, blue, yellow, green = by[RED], by[BLUE], by[YELLOW], by[GREEN]
    assert (red["shot"], red["first"], red["last"], red["found"]) == (0, 0, 9, "shot")
    assert (yellow["first"], yellow["found"]) == (6, "reseed")      # appeared on 5, next look at 6
    assert (green["shot"], green["first"], green["last"]) == (1, 10, 19)
    assert len(meta["objects"]) == 4                                   # red and blue never added twice
    assert seg.auto_calls == [0, 3, 6, 9, 10, 13, 16, 19]


def test_labels_and_stats_follow_the_objects(track):
    mt, meta, _ = track
    red = next(o["id"] for o in meta["objects"] if tuple(o["colour"]) == RED)
    for i in (0, 9):
        m = mt.mask(i, red)
        assert m[60, 30 + 4 * i] and m.sum() == 40 * 40
    s0, s9 = mt.stat(0, red), mt.stat(9, red)
    assert s9["cx"] - s0["cx"] == pytest.approx(36 / W, abs=1e-4)      # slid 36 px
    assert s0["share"] == pytest.approx(1600 / (W * H), abs=1e-4)
    assert mt.stat(12, red)["share"] == 0.0 and red not in mt.in_view(12)
    assert (mt.labels(12) > 0).sum() == 60 * 60                        # only the green square


def test_song_time(track):
    mt, _, _ = track
    assert mt.frame_at(13.0) == 0 and mt.frame_at(13.5) == 5 and mt.frame_at(12.9) is None
    assert mt.frame_at(13.0 + 2.0) is None
    assert np.array_equal(mt.labels_at(13.5), mt.labels(5))

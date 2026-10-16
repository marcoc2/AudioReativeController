"""meters: the read-out strip that shows what drives a scene."""
import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import MidiNote
from core.video.meters import Meters

W, H, FPS = 240, 427, 30


def _kick(times):
    return [MidiNote(time=t, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=0)
            for t in times]


def test_draws_a_strip_at_the_bottom_and_a_title_band_on_top():
    notes = _kick([0.5, 1.0, 1.5, 2.0])
    m = Meters({"title": "test", "lanes": [{"label": "kick", "knob": {"hits": {"track": "kick"}, "envelope": 0.3}},
                                            {"label": "roll", "roll": {"track": "kick"}}]},
               notes, W, H, FPS, grid=RhythmGrid(bpm=120.0))
    frame = np.full((H, W, 3), 200, np.uint8)
    out = m.process(frame, 1.0)
    assert out.shape == frame.shape and out.dtype == np.uint8
    assert out[: H // 20].mean() < 150                              # the title band darkens the top
    assert out[-H // 10:].mean() < 150                              # the strip darkens the bottom
    assert (out[H // 3: H // 2] == 200).all()                       # the middle is left alone


def test_knobs_are_sampled_in_time_order():
    seen = []
    m = Meters({"lanes": [{"label": "t", "knob": 0.0}]}, [], W, H, FPS)
    lane = m.lanes[0][3]
    lane.f = lambda t: seen.append(t) or t
    for i in range(5):
        m.process(np.zeros((H, W, 3), np.uint8), 1.0 + i / FPS)
    assert seen == sorted(seen)
    assert lane.at(1.1) == pytest.approx(1.1, abs=1 / 60)


def test_the_playhead_is_where_the_past_ends():
    m = Meters({"past": 3.0, "future": 1.0, "lanes": [{"label": "", "knob": 0.5, "range": [0, 1]}]},
               [], W, H, FPS)
    out = m.process(np.zeros((H, W, 3), np.uint8), 5.0)
    x = int(round(W * 3.0 / 4.0))
    assert out[-5, x - 1: x + 2].max() > 200                        # the white playhead


def test_a_lane_needs_a_kind_and_a_roll_needs_notes():
    with pytest.raises(ValueError, match="knob:, roll: or chords:"):
        Meters({"lanes": [{"label": "x"}]}, [], W, H, FPS)
    with pytest.raises(ValueError, match="no notes"):
        Meters({"lanes": [{"label": "x", "roll": {"notes": [99]}}]}, _kick([1.0]), W, H, FPS)

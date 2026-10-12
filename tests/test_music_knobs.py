"""Knobs that follow the harmony and the clip cuts: ``chord:`` and ``cut:`` (no GPU needed)."""
import numpy as np
import pytest

from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import MidiNote
from core.video.composer import ClipComposer
from core.video.layers import _music_knob


def _knob(how, grid=None, cuts=None):
    return _music_knob("u", how, [], grid=grid, cuts=cuts)


@pytest.fixture
def score(tmp_path):
    rows = [(0.0, 2.0, "N"), (2.0, 4.0, "A:min7"), (4.0, 6.0, "N"), (6.0, 8.0, "C:maj"),
            (8.0, 10.0, "B:dim"), (10.0, 12.0, "D:sus4")]
    (tmp_path / "chord.lab").write_text("\n".join("\t".join(map(str, r)) for r in rows) + "\n", encoding="utf-8")
    return {"score": str(tmp_path), "snap": "off"}


def test_chord_parts(score):
    at = lambda how, t: _knob({"chord": score, **how})(t)
    assert at({}, 1.0) == 9.0                                     # before the first chord: that chord
    assert at({}, 5.0) == 9.0                                     # a no-chord stretch keeps the last one
    assert [at({"part": "tone", "tone": k}, 3.0) for k in range(5)] == [9, 0, 4, 7, 9]   # A C E G, wraps
    assert at({"part": "size"}, 3.0) == 4.0 and at({"part": "size"}, 7.0) == 3.0
    assert [at({"part": "mode"}, t) for t in (3.0, 7.0, 9.0, 11.0)] == [0.0, 1.0, 0.0, 0.5]
    assert [at({"part": "tension"}, t) for t in (3.0, 7.0, 9.0, 11.0)] == [0.25, 0.0, 0.5, 0.25]
    with pytest.raises(ValueError):
        _knob({"chord": score, "part": "colour"})


def test_chord_root_glides_the_short_way_round(score):
    root = _knob({"chord": score, "glide": 0.3})
    vals = [root(t) for t in np.arange(5.0, 8.0, 1 / 30)]          # A (9) -> C (0): up through 10, 11
    assert vals[0] == 9.0 and min(vals[-1], 12 - vals[-1]) < 0.05
    steps = [((b - a + 6) % 12) - 6 for a, b in zip(vals, vals[1:])]
    assert all(0 <= s < 0.5 for s in steps)                        # never back down through 8, 7..


def test_cut_parts():
    cuts = lambda until: np.array([t for t in (2.0, 4.0, 7.0) if t <= until])
    at = lambda part, t, **kw: _knob({"cut": part, **kw}, cuts=cuts)(t)
    assert [at("count", t) for t in (1.0, 2.0, 3.0, 5.0, 8.0)] == [0, 1, 1, 2, 3]
    assert at("since", 1.0) == 1000.0 and at("since", 3.5) == 1.5
    assert at("until", 3.5) == 0.5 and at("until", 8.0) == 1000.0
    assert at("phase", 5.5) == 0.5 and at("phase", 1.0) == 0.0
    assert 1.0 < at("count", 4.1, morph=0.4) < 2.0
    with pytest.raises(ValueError):
        _knob({"cut": "count"})                                    # no clips, no cuts
    with pytest.raises(ValueError):
        _knob({"cut": "soon"}, cuts=cuts)


class _Library:
    def __init__(self, n=3):
        self.n = n

    def __len__(self):
        return self.n


def test_the_composer_knows_its_cuts_ahead():
    beats = np.arange(0.0, 20.0, 0.5)                              # 120 BPM, bars every 2 s
    grid = RhythmGrid(bpm=120.0, time_signature=(4, 4), fps=24, beats=beats, downbeats=beats[::4])
    snare = [MidiNote(time=5.0, pitch=38, velocity=100, channel=0, duration=0.1, track="snare", track_index=1),
             MidiNote(time=9.0, pitch=38, velocity=100, channel=0, duration=0.1, track="snare", track_index=1)]
    cfg = {"bars_per_clip": 2, "overrides": {3: 0},
           "triggers": {"snare": {"track": "snare", "actions": ["next_clip"]}}}
    comp = ClipComposer(_Library(), grid, snare, cfg)
    assert list(comp.cut_times(12.0)) == [4.0, 5.0, 6.0, 8.0, 9.0, 12.0]   # every 2 bars, the pin, the snare
    assert list(ClipComposer(_Library(1), grid, snare, cfg).cut_times(12.0)) == []

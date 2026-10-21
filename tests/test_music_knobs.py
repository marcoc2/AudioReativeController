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


def test_groove_is_the_bars_pattern_of_hits():
    grid = RhythmGrid.from_beats(np.arange(0, 8.0, 0.5), time_signature=(4, 4))   # bars of 2 s, 16ths of 0.125
    kick = [MidiNote(time=t, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=0)
            for t in (0.0, 0.74, 1.01, 2.5, 3.94)]                 # steps 0, 6, 8 | 4 and the next bar's 0
    g = lambda **kw: _music_knob("u", {"groove": {"track": "kick"}, **kw}, kick, grid=grid)
    assert g()(1.5) == float(1 | 1 << 6 | 1 << 8)
    assert g()(3.0) == float(1 << 4)                               # 3.94 is a hair early for bar 2's one
    assert g()(4.2) == 1.0
    assert g(ago=1)(3.0) == float(1 | 1 << 6 | 1 << 8)
    assert g(part="steps")(1.0) == 16.0 and g(division=8, part="steps")(1.0) == 8.0
    with pytest.raises(ValueError):
        _music_knob("u", {"groove": {"track": "kick"}}, kick)


def test_every_knob_takes_gain_and_bias():
    cuts = lambda until: np.array([t for t in (2.0, 4.0) if t <= until])
    depth = _knob({"cut": "phase", "gain": 40, "bias": 8}, cuts=cuts)
    assert depth(3.0) == 28.0 and depth(1.0) == 8.0


def _note(t, d, v=127, track="pad"):
    return MidiNote(time=t, pitch=60, velocity=v, channel=0, duration=d, track=track, track_index=0)


def test_adsr_holds_while_the_note_sounds_then_lets_go():
    env = _music_knob("u", {"adsr": {"track": "pad"}, "attack": 0.1, "decay": 0.1, "sustain": 0.5,
                            "release": 0.3}, [_note(1.0, 2.0)])
    assert env(0.5) == 0.0 and abs(env(1.05) - 0.5) < 1e-6 and abs(env(1.1) - 1.0) < 1e-6
    assert abs(env(2.5) - 0.5) < 0.01                               # held: the sustain
    assert 0.0 < env(3.1) < 0.4 and env(4.5) < 0.01                 # released after the note ends
    soft = _music_knob("u", {"adsr": {"track": "pad"}}, [_note(1.0, 1.0, v=64)])
    assert soft(1.02) < 0.55                                         # velocity scales it


def test_before_ramps_up_ahead_of_the_hit():
    pre = _music_knob("u", {"before": {"track": "pad"}, "window": 0.5}, [_note(2.0, 0.1)])
    assert pre(1.0) == 0.0 and abs(pre(1.75) - 0.5) < 1e-9 and pre(1.99) > 0.95 and pre(2.01) == 0.0


def test_relative_follows_the_recent_average():
    level = {"v": 1.0}
    rel = _music_knob("u", {"relative": "x", "window": 2.0, "attack": 0.01, "release": 0.01}, [],
                      features_at=lambda t: {"x": level["v"]})
    for t in np.arange(0, 10, 0.1):
        quiet = rel(t)
    assert abs(quiet - 1.0) < 1e-6                                   # a steady level reads as 1
    level["v"] = 3.0
    burst = rel(10.1)
    assert burst > 2.0                                               # three times louder than lately
    for t in np.arange(10.2, 30, 0.1):
        settled = rel(t)
    assert abs(settled - 1.0) < 0.05                                 # and that becomes the new usual


def test_lfos_keep_step_with_the_grid():
    grid = RhythmGrid(bpm=120.0)                                     # a beat is 0.5 s, a whole note 2 s
    tri = _music_knob("u", {"lfo": "tri", "period": "1/1"}, [], grid=grid)
    assert tri(0.0) == 0.0 and tri(1.0) == 1.0 and abs(tri(1.5) - 0.5) < 1e-9
    sq = _music_knob("u", {"lfo": "pulse", "period": 1.0, "width": 0.25}, [])
    assert sq(0.1) == 1.0 and sq(0.3) == 0.0
    sine = _music_knob("u", {"lfo": "sine", "period": "1/4", "gain": 2.0}, [], grid=grid)
    assert abs(sine(0.25) - 2.0) < 1e-9                              # gain applies to every knob
    with pytest.raises(ValueError):
        _music_knob("u", {"lfo": "wobble"}, [])


def test_random_on_each_hit_eases_there():
    hits = [_note(1.0, 0.1), _note(2.0, 0.1)]
    r = _music_knob("u", {"random_on": {"track": "pad"}, "ease": 0.2, "seed": 4}, hits)
    a, b, c = r(0.5), r(1.5), r(2.5)
    assert len({a, b, c}) == 3 and all(0 <= v <= 1 for v in (a, b, c))
    assert min(a, b) <= r(1.1) <= max(a, b)                          # sliding between them


class _Track:
    """Two figures in view at frames 0..9, none after; the bigger one at (0.2, 0.7)."""
    def frame_at(self, t):
        i = int(round(t * 10))
        return i if 0 <= i < 20 else None

    def in_view(self, i):
        return [3, 5] if i < 10 else []

    def stat(self, i, k):
        return {3: {"share": 0.3, "cx": 0.2, "cy": 0.7}, 5: {"share": 0.1, "cx": 0.9, "cy": 0.1}}[k]


def test_figure_follows_a_figure_of_the_mask_track():
    tr = _Track()
    x = _music_knob("u", {"figure": "largest", "part": "x", "glide": 0}, [], track=lambda: tr)
    y2 = _music_knob("u", {"figure": "second", "part": "y", "glide": 0}, [], track=lambda: tr)
    assert x(0.3) == 0.2 and y2(0.3) == 0.1
    assert x(1.5) == 0.2                                            # nobody in view: held
    with pytest.raises(ValueError):
        _music_knob("u", {"figure": "largest"}, [])                 # no track


def test_voice_takes_the_k_th_note_sounding():
    chord = [_note(1.0, 1.0), MidiNote(time=1.0, pitch=67, velocity=100, channel=0, duration=1.0, track="pad",
                                       track_index=0)]
    low = _music_knob("u", {"voice": {"track": "pad"}, "k": 0}, chord)
    high = _music_knob("u", {"voice": {"track": "pad"}, "k": 1}, chord)
    lev = _music_knob("u", {"voice": {"track": "pad"}, "k": 1, "part": "level", "release": 0.5}, chord)
    assert low(1.5) == 60.0 and high(1.5) == 67.0 and lev(1.5) == 1.0
    assert abs(lev(2.25) - 0.5) < 1e-9 and lev(3.0) == 0.0 and high(3.0) == 67.0   # released; pitch held

"""The knob editor's forms read and write every knob the example scenes use."""
from pathlib import Path

import numpy as np
import pytest
import yaml

from core.knob_forms import FORMS, choices, form_of, new_state, sample, spec_of, state_of
from core.rhythm.grid import RhythmGrid
from core.rhythm.midi_reader import MidiNote


def _example_knobs():
    out = []
    for path in sorted(Path("examples").glob("*.yaml")):
        video = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("video") or {}
        for layer in video.get("layers") or []:
            out += list((layer.get("uniforms") or {}).values())
            out += [lane["knob"] for lane in layer.get("lanes") or [] if "knob" in lane]
    return out


def test_every_example_knob_comes_back_as_it_was():
    knobs = _example_knobs()
    assert len(knobs) > 300
    for spec in knobs:
        assert form_of(spec) is not None, spec
        assert spec_of(state_of(spec)) == spec


def test_every_form_is_listed_once_by_its_meaning():
    listed = [k for k, _ in choices()]
    assert sorted(listed) == sorted(FORMS)
    assert listed[0] == "hits"                       # beat first


def test_a_new_knob_writes_its_usual_settings_and_keeps_the_source():
    kick = state_of({"hits": {"track": "drums (midi)", "notes": [36]}, "envelope": 0.3})
    adsr = new_state("adsr", like=kick)
    assert spec_of(adsr) == {"adsr": {"track": "drums (midi)", "notes": [36]}, "attack": 0.02,
                             "decay": 0.2, "sustain": 0.6, "release": 0.4}
    assert spec_of(new_state("bar")) == {"bar": {"step": 1.0, "morph": 0.35}}
    assert spec_of(new_state("harmony", like=kick)) == {"harmony": {"score": "", "snap": "beat"},
                                                        "glide": 1.2}
    st = new_state("rate")
    st["from"]["source"]["notes"] = "38"
    assert spec_of(st) == {"rate": 2.0, "from": {"hits": {"notes": [38]}, "envelope": 0.25},
                           "base": 0.0}
    st["from"] = None
    assert spec_of(st) == {"rate": 2.0, "base": 0.0}


def test_gain_and_a_plain_feature():
    st = state_of("subbands.bass")
    assert spec_of(st) == "subbands.bass"
    st["gain"] = 2.0
    assert spec_of(st) == {"feature": "subbands.bass", "gain": 2.0}


def test_a_new_feature_glides_and_an_old_one_stays_as_it_was():
    assert spec_of(new_state("feature")) == {"feature": "subbands.bass", "attack": 0.1, "release": 0.5}
    st = state_of("subbands.bass")
    assert st["params"] == {"attack": 0.0, "release": 0.0}
    st["params"]["release"] = 0.4
    assert spec_of(st) == {"feature": "subbands.bass", "release": 0.4}


def test_a_gliding_feature_rises_and_falls_the_same_whatever_came_before():
    from core.video.layers import _knob
    loud = lambda t: {"subbands": {"bass": 1.0 if 1.0 <= t < 2.0 else 0.0}}   # noqa: E731
    spec = {"feature": "subbands.bass", "attack": 0.1, "release": 0.5}
    def in_order(t):                                                 # a render from the start to t
        f = _knob("k", spec, [], features_at=loud)
        for s in np.arange(0.0, t, 1 / 24):
            f(s)
        return f(t)
    assert in_order(0.9) == 0.0
    assert 0.3 < in_order(1.05) < 0.6                                # rising, not a jump
    assert in_order(1.9) > 0.99
    assert 0.5 < in_order(2.2) < 0.75                                # falling slower than it rose
    f = _knob("k", spec, [], features_at=loud)                       # a seek: forward, back, ...
    for t in (3.0, 0.5, 2.2, 1.05):
        assert f(t) == pytest.approx(in_order(t), abs=1e-3)
    raw = _knob("k", {"feature": "subbands.bass"}, [], features_at=loud)
    assert raw(1.0) == 1.0 and raw(2.0) == 0.0                       # without them: as it is


def test_the_curve_is_the_knob_a_render_gets():
    grid = RhythmGrid(bpm=120.0, fps=30, beats=np.arange(16) * 0.5, downbeats=np.arange(4) * 2.0)
    kicks = [MidiNote(time=t, pitch=36, velocity=100, channel=0, duration=0.1, track="kick")
             for t in (1.0, 3.0)]
    ts, vs = sample({"hits": {"track": "kick"}, "envelope": 0.5}, 0.0, 4.0, 401,
                    notes=kicks, grid=grid)
    assert vs[np.searchsorted(ts, 1.0)] == pytest.approx(1.0, abs=0.02)
    assert vs[np.searchsorted(ts, 0.9)] == 0.0
    assert vs[np.searchsorted(ts, 2.0)] < 0.1
    _, bar = sample({"bar": {"step": 1, "morph": 0.01}}, 0.5, 7.5, 8, grid=grid)
    assert list(np.round(bar, 3)) == [0, 0, 1, 1, 2, 2, 3, 3]          # a step a bar


def test_a_track_name_with_commas_stays_whole():
    from core.knob_forms import split_tracks
    assert split_tracks("kick, kick-2") == ["kick", "kick-2"]
    assert split_tracks("FM5 MIDI (patches 2, 5, 10), kick") == ["FM5 MIDI (patches 2, 5, 10)", "kick"]


def test_clip_holds_any_link_inside_its_range():
    from core.video.layers import _music_knob
    loud = lambda t: {"subbands": {"bass": 3.0 if t > 1 else -2.0}}        # noqa: E731
    for spec in ({"feature": "subbands.bass", "clip": [-0.5, 0.5]},
                 {"feature": "subbands.bass", "gain": 2, "clip": [0.5, -0.5]},
                 {"lfo": "sine", "period": 1.0, "gain": 9, "clip": [-0.5, 0.5]}):
        f = _music_knob("k", spec, [], features_at=loud)
        assert all(-0.5 <= f(t) <= 0.5 for t in np.linspace(0, 3, 40)), spec
    st = state_of({"relative": "texture.loudness", "gain": 0.4, "bias": -0.4, "clip": [-0.5, 0.5]})
    assert st["clip"] == [-0.5, 0.5] and spec_of(st)["clip"] == [-0.5, 0.5]
    st["clip"] = None
    assert "clip" not in spec_of(st)
    plain = state_of("subbands.bass")
    plain["clip"] = [0, 1]
    assert spec_of(plain) == {"feature": "subbands.bass", "clip": [0.0, 1.0]}

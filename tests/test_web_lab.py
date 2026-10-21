"""The Feature Lab's engine side: a link's steps as the render runs them, what a feature
listens to, the parameter's range, and a link tried on the Live stack before Apply."""
import numpy as np
import pytest
import soundfile as sf
import yaml

from core import feature_info
from core.feature_extractor import AudioFeatureExtractor
from core.knob_forms import new_state, state_of
from tests.test_web_session import _session, project  # noqa: F401  (the fixture)


def _loud(t):                                   # a bass that is on in 1..2 s and 5..6 s
    on = 1.0 if (1.0 <= t < 2.0 or 5.0 <= t < 6.0) else 0.0
    return {"subbands": {"bass": on}, "stems": {"kick": on * 0.5}, "texture": {"loudness": 0.2 + 0.6 * on}}


def test_every_feature_is_said_with_its_range_and_what_it_hears():
    assert set(feature_info.BANDS) == set(AudioFeatureExtractor.SUBBAND_RANGES)
    for name, (lo, hi, _) in feature_info.BANDS.items():
        assert AudioFeatureExtractor.SUBBAND_RANGES[name] == (lo, hi)
    for name in feature_info.names(["kick"]):
        d = feature_info.describe(name, ["kick"])
        assert d["about"] and d["group"] in feature_info.GROUPS and d["range"][0] < d["range"][1], name
    assert feature_info.describe("stems.kick", ["kick"])["heard"] == {"kind": "stem", "stem": "kick"}
    assert feature_info.describe("subbands.bass")["heard"] == {"kind": "band", "lo": 60, "hi": 250}


def test_the_steps_of_a_feature_link_are_the_render_s(project):  # noqa: F811
    s = _session(project)
    s.set_features(_loud)
    st = state_of({"feature": "subbands.bass", "attack": 0.1, "release": 0.5, "gain": 2.0, "bias": -0.5,
                   "clip": [-0.5, 1.0]})
    got = s.knob_stages(st, 0.0, 8.0, 160)
    assert [x["key"] for x in got["stages"]] == ["feature", "glided", "final"]
    feat, glided, final = got["stages"]
    assert feat["max"] == 1.0 and feat["min"] == 0.0 and got["cols"] == 160
    assert 0.9 < glided["max"] < 1.0                     # it rises, it does not jump
    assert final["max"] == 1.0 and final["min"] == -0.5  # clipped: 2 * 0.96 - 0.5 held at 1
    i = int(1.5 / 8 * 160)
    assert final["hi"][i] == pytest.approx(min(1.0, 2 * glided["hi"][i] - 0.5), abs=1e-3)
    assert got["song"]["p95"] == 1.0 and got["song"]["p5"] == 0.0
    assert got["yaml"].startswith("{feature: subbands.bass")


def test_a_relative_link_shows_its_average_and_ratio(project):  # noqa: F811
    s = _session(project)
    s.set_features(_loud)
    st = state_of({"relative": "texture.loudness", "window": 6, "gain": 0.4, "bias": -0.4})
    keys = [x["key"] for x in s.knob_stages(st, 0.0, 8.0, 80)["stages"]]
    assert keys == ["feature", "average", "ratio", "final"]
    stages = {x["key"]: x for x in s.knob_stages(st, 0.0, 8.0, 80)["stages"]}
    assert stages["ratio"]["max"] > 1.5 and stages["average"]["max"] < 0.6
    assert stages["final"]["max"] == pytest.approx(0.4 * stages["ratio"]["max"] - 0.4, abs=1e-3)


def test_a_parameter_s_range_from_its_about_text_or_the_user(project, monkeypatch):  # noqa: F811
    import core.web.api as api
    s = _session(project)
    ref = ["BG", "layers", 2]                            # fases_arc: u_level "(0..5, e volta)"
    assert s.param_range(ref, ["uniforms", "u_level"]) == {"range": [0.0, 5.0], "from": "about",
                                                            "key": "shadertoy:fases_arc:u_level"}
    monkeypatch.setattr(api, "prefs", lambda: {"ui": {"ranges": {"shadertoy:fases_arc:u_level": [1, 3]}}})
    assert s.param_range(ref, ["uniforms", "u_level"])["range"] == [1.0, 3.0]
    assert s.param_range(ref, ["opacity"])["range"] == [0.0, 1.0]
    assert s.knob_open(ref, ["uniforms", "u_level"])["param"]["from"] == "yours"


def test_a_link_tried_on_live_only_changes_the_previews_stack(project):  # noqa: F811
    s = _session(project)
    ref = ["BG", "layers", 0]
    s.knob_draft(ref, ["uniforms", "u_s"], new_state("feature"))
    layer = lambda cfg: cfg["layers"][1]["layers"][0]     # noqa: E731
    assert layer(s.video_cfg(draft=True))["uniforms"]["u_s"] == {"feature": "subbands.bass", "attack": 0.1, "release": 0.5}
    assert layer(s.video_cfg())["uniforms"]["u_s"] == 0.4  # what is saved and rendered: as it was
    s.save()
    assert yaml.safe_load(s.scene.read_text(encoding="utf-8"))["video"]["layers"][1]["layers"][0]["uniforms"]["u_s"] == 0.4
    assert s.knob_draft_clear()["had"] and layer(s.video_cfg(draft=True))["uniforms"]["u_s"] == 0.4


def test_the_lab_hears_a_stem_s_file_and_a_band_of_the_mix(project, tmp_path):  # noqa: F811
    s = _session(project)
    sr = 22050
    t = np.arange(4 * sr) / sr
    kick = (np.sin(2 * np.pi * 50 * t) * (t % 0.5 < 0.1)).astype(np.float32)
    sf.write(tmp_path / "kick.wav", kick, sr)
    sf.write(tmp_path / "song.wav", kick + 0.2 * np.sin(2 * np.pi * 3000 * t).astype(np.float32), sr)
    s.audio, s.stem_names, s.stem_files = tmp_path / "song.wav", ["kick"], {"kick": str(tmp_path / "kick.wav")}
    heard = s.lab_heard("stems.kick")
    assert heard["heard"]["url"] == "/media/kick.wav" and heard["label"].startswith("kick")
    wave = s.lab_wave("stems.kick")
    assert wave["hz"] == 200 and len(wave["peaks"]) == 800 and max(wave["peaks"]) == 1.0
    assert wave["peaks"][10] > 0.5 and wave["peaks"][40] < 0.05          # on at 0.05 s, off at 0.2 s
    band = s.lab_wave("subbands.sub_bass")                                # 20-60 Hz: the kick, not the 3 kHz tone
    assert band["kind"] == "band" and band["peaks"][40] < 0.1 < band["peaks"][10]
    assert s.lab_wave("texture.loudness") == {"kind": "mix"}
    s.stem_files = {}
    assert s.lab_heard("stems.kick")["heard"]["kind"] == "mix"           # no file: the mix, said why

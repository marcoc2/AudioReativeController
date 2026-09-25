"""Tests for ARC Studio's render command construction (no GUI context needed)."""
import sys

from arc_studio import S, _build_render_cmd, _rows_to_video_cfg, _video_cfg_to_rows


def _reset_state():
    S.audio_path = "song.mp3"
    S.midi_path  = "song.mid"
    S.scene_path = "scenes/clips_enxame.yaml"
    S.mode = "clips"
    S.bars = 8
    S.fps  = 24
    S.width, S.height = 480, 480
    S.clips_dir  = "clips_folder"
    S.clip_order = "(scene)"
    S.full_song  = False
    S.cache_size = 8
    S.grav_enable = False
    S.midi_offset = 0.0
    S.meter = ""


def test_clips_mode_dispatches_to_clip_generator():
    _reset_state()
    cmd = _build_render_cmd()
    assert cmd[0] == sys.executable
    assert cmd[1] == "clip_generator.py"
    assert cmd[cmd.index("--clips") + 1] == "clips_folder"
    assert cmd[cmd.index("--bars") + 1] == "8"
    assert cmd[cmd.index("--midi") + 1] == "song.mid"
    assert "--clip-order" not in cmd       # "(scene)" keeps the YAML's order
    assert "--gravity-peak" not in cmd


def test_clips_full_song_renders_bars_zero():
    _reset_state()
    S.full_song = True
    cmd = _build_render_cmd()
    assert cmd[cmd.index("--bars") + 1] == "0"


def test_clips_order_and_gravity_overrides():
    _reset_state()
    S.clip_order = "shuffle"
    S.grav_enable = True
    S.grav_peak, S.grav_floor = 7.0, 0.7
    S.grav_radius, S.grav_curve = 0.45, 3.0
    cmd = _build_render_cmd()
    assert cmd[cmd.index("--clip-order") + 1] == "shuffle"
    assert cmd[cmd.index("--gravity-peak") + 1] == "7.0"
    assert cmd[cmd.index("--gravity-floor") + 1] == "0.7"
    assert cmd[cmd.index("--gravity-curve") + 1] == "3.0"


def test_particles_mode_unchanged():
    _reset_state()
    S.mode = "particles"
    cmd = _build_render_cmd()
    assert cmd[1] == "generators/particle_generator.py"
    assert "--clips" not in cmd
    assert cmd[cmd.index("--midi") + 1] == "song.mid"


def test_no_midi_omits_flag():
    _reset_state()
    S.midi_path = None
    cmd = _build_render_cmd()
    assert "--midi" not in cmd


def test_mode_script_map():
    _reset_state()
    for mode, script in (("particles", "generators/particle_generator.py"),
                         ("particles_v2", "generators/particle_generator_new.py"),
                         ("particles_v3", "generators/particle_generator_v3.py"),
                         ("geometry", "generators/animation_generator.py")):
        S.mode = mode
        assert _build_render_cmd()[1] == script


# ---------------------------------------------------------------------------
# scene trigger editor round-trip

ENXAME_VIDEO = {
    "clip_per_bar": False,
    "clip_order": "shuffle",
    "triggers": {
        "kick": {"notes": [36], "actions": ["next_clip"], "until": "snare"},
        "snare": {
            "audio": "stems/caixa.mp3",
            "threshold": 0.3,
            "min_gap": 0.08,
            "exclude": {"trigger": "kick", "window": 0.04},
            "actions": ["next_clip"],
        },
    },
}


def test_editor_round_trip_preserves_scene():
    rows = _video_cfg_to_rows(ENXAME_VIDEO)
    out = _rows_to_video_cfg(rows, ENXAME_VIDEO["clip_per_bar"],
                             ENXAME_VIDEO["clip_order"])
    assert out["clip_per_bar"] is False
    assert out["clip_order"] == "shuffle"
    assert out["triggers"]["kick"]["notes"] == [36]
    assert out["triggers"]["kick"]["until"] == "snare"
    snare = out["triggers"]["snare"]
    assert snare["audio"] == "stems/caixa.mp3"
    assert snare["min_gap"] == 0.08
    # unknown key kept through the editor round-trip
    assert snare["exclude"] == {"trigger": "kick", "window": 0.04}


def test_editor_rows_gravity_and_parsing():
    rows = [{"name": "kick", "source": "notes", "notes": "36; 35",
             "actions": " reverse , next_clip ", "until": "", "min_vel": 40,
             "grav_on": True, "peak": 7.0, "floor": 0.7, "radius": 0.45,
             "curve": 3.0, "_extra": {}}]
    cfg = _rows_to_video_cfg(rows, True, "sequential")
    spec = cfg["triggers"]["kick"]
    assert spec["notes"] == [36, 35]
    assert spec["actions"] == ["reverse", "next_clip"]
    assert spec["min_velocity"] == 40
    assert spec["gravity"] == {"peak": 7.0, "floor": 0.7, "radius": 0.45, "curve": 3.0}
    assert "until" not in spec


def test_toggle_action_builds_canonical_list():
    from arc_studio import _toggle_action
    row = {"actions": "reverse"}
    _toggle_action(row, "next_clip", True)
    assert row["actions"] == "next_clip,reverse"
    _toggle_action(row, "reverse", False)
    assert row["actions"] == "next_clip"
    _toggle_action(row, "next_clip", False)
    assert row["actions"] == ""


def test_editor_skips_unnamed_rows():
    rows = [{"name": "  ", "source": "notes", "notes": "36", "actions": "next_clip"}]
    cfg = _rows_to_video_cfg(rows, True, "shuffle")
    assert cfg["triggers"] == {}


def test_generative_scene_renders_without_clips_folder():
    _reset_state()
    S.clips_dir = None
    cmd = _build_render_cmd()
    assert "--clips" not in cmd and None not in cmd


def test_midi_offset_and_meter_reach_the_render():
    _reset_state()
    assert "--midi-offset" not in _build_render_cmd() and "--meter" not in _build_render_cmd()
    S.midi_offset, S.meter = 0.018, "22:6/4,27:5/4"
    cmd = _build_render_cmd()
    assert cmd[cmd.index("--midi-offset") + 1] == "0.018"
    assert cmd[cmd.index("--meter") + 1] == "22:6/4,27:5/4"


def test_track_and_score_triggers_survive_the_editor():
    video = {"triggers": {
        "kick":  {"track": ["kick", "kick-2"], "actions": ["reverse"]},
        "chord": {"score": "input/x/sheetsage", "actions": ["next_clip"]},
        "snare": {"track": "snare", "notes": [38], "actions": ["restart"]},
    }}
    back = _rows_to_video_cfg(_video_cfg_to_rows(video), True, "sequential")["triggers"]
    assert back["kick"] == {"track": ["kick", "kick-2"], "actions": ["reverse"]}
    assert back["chord"] == {"score": "input/x/sheetsage", "actions": ["next_clip"]}
    assert back["snare"]["notes"] == [38] and back["snare"]["track"] == "snare"


def _norm_layer(spec):
    """What a layer means, without its defaults spelled out."""
    out = dict(spec)
    if out.get("enabled", True) is True:
        out.pop("enabled", None)
    if out.get("blend") == "normal":
        out.pop("blend")
    if out.get("opacity") == 1 or out.get("opacity") == 1.0:
        out.pop("opacity")
    return out


def test_every_example_scene_survives_the_layer_editor():
    from pathlib import Path

    import yaml

    from arc_studio import _layers_to_rows, _rows_to_layers
    checked = 0
    for path in sorted(Path("examples").glob("*.yaml")):
        video = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("video") or {}
        if not video.get("layers"):
            continue
        back = _rows_to_layers(_layers_to_rows(video))
        assert [_norm_layer(l) for l in back] == [_norm_layer(l or {}) for l in video["layers"]], path
        checked += 1
    assert checked > 10


def test_layer_editor_rejects_bad_yaml_and_keeps_switches():
    import pytest

    from arc_studio import _rows_to_layers
    rows = [{"source": "solid", "enabled": True, "file": "", "blend": "normal", "opacity": 1.0,
             "bars": "", "params": "color: [0, 0, 0]"},
            {"source": "shadertoy", "enabled": False, "file": "shaders/shadertoy/x.glsl",
             "blend": "screen", "opacity": 0.5, "bars": "[[21, 33]]", "params": "speed: 0.5"}]
    assert _rows_to_layers(rows) == [
        {"source": "solid", "color": [0, 0, 0]},
        {"source": "shadertoy", "enabled": False, "file": "shaders/shadertoy/x.glsl",
         "blend": "screen", "opacity": 0.5, "bars": [[21, 33]], "speed": 0.5}]
    rows[1]["params"] = "speed: [0.5"
    with pytest.raises(ValueError, match="layer 2"):
        _rows_to_layers(rows)
    rows[1]["params"] = "just words"
    with pytest.raises(ValueError, match="key: value"):
        _rows_to_layers(rows)


def test_trigger_editor_builds_each_source():
    base = {"name": "t", "notes": "", "audio": "", "track": "", "score": "",
            "events": "chord_changes", "snap": "auto", "voice": "", "actions": "next_clip",
            "until": "", "min_vel": 0, "grav_on": False, "_extra": {}}
    def build(**kw):
        return _rows_to_video_cfg([{**base, **kw}], True, "sequential")["triggers"]["t"]
    assert build(source="notes", notes="36, 38") == {"notes": [36, 38], "actions": ["next_clip"]}
    assert build(source="track", track="kick, kick-2") == {"track": ["kick", "kick-2"],
                                                           "actions": ["next_clip"]}
    assert build(source="track", track="snare", notes="38")["notes"] == [38]
    assert build(source="score", score="input/x/sheetsage", events="melody", snap="beat",
                 voice="vocals") == {"score": "input/x/sheetsage", "events": "melody",
                                     "snap": "beat", "voice": "vocals", "actions": ["next_clip"]}
    # switching a row's source drops the old source's keys
    rows = _video_cfg_to_rows({"triggers": {"t": {"track": "kick", "actions": ["reverse"]}}})
    rows[0]["source"], rows[0]["notes"] = "notes", "36"
    assert _rows_to_video_cfg(rows, True, "sequential")["triggers"]["t"] == {
        "notes": [36], "actions": ["reverse"]}


def test_every_example_scene_keeps_its_triggers_through_the_editor():
    from pathlib import Path

    import yaml
    checked = 0
    for path in sorted(Path("examples").glob("*.yaml")):
        video = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("video") or {}
        trig = video.get("triggers")
        if not trig:
            continue
        back = _rows_to_video_cfg(_video_cfg_to_rows(video), True, "sequential")["triggers"]
        for name, spec in trig.items():
            defaults = {"events": "chord_changes", "snap": "auto"}   # dropped: same meaning
            want = {k: v for k, v in spec.items() if defaults.get(k) != v}
            got = back[name]
            if "audio" in want:                     # the editor spells out onset defaults
                got = {k: v for k, v in got.items() if k in want}
            assert got == want, (path, name)
        checked += 1
    assert checked >= 5


def test_render_options_start_stems_fit_and_named_output():
    _reset_state()
    S.start_time, S.render_stems, S.clip_fit = 0.0, False, "contain"
    cmd = _build_render_cmd()
    assert "--start-time" not in cmd and "--stems" not in cmd and "--clip-fit" not in cmd
    assert cmd[cmd.index("--output") + 1] == "render_output/song_clips_enxame.mp4"
    S.start_time, S.render_stems, S.clip_fit = 53.351, True, "cover"
    cmd = _build_render_cmd()
    assert cmd[cmd.index("--start-time") + 1] == "53.351"
    assert "--stems" in cmd and cmd[cmd.index("--clip-fit") + 1] == "cover"
    assert cmd[cmd.index("--output") + 1] == "render_output/song_clips_enxame_from53s.mp4"
    S.start_time, S.render_stems, S.clip_fit = 0.0, False, "contain"


def test_repaint_command_follows_the_song():
    from arc_studio import _build_repaint_cmd
    _reset_state()
    S.repaint_video = "render_output/song_x.mp4"
    S.repaint_start, S.repaint_fps, S.repaint_seconds = 10.685, 12.0, 4.0
    S.repaint_workflow, S.repaint_scene = "reference/wf.json", "(none)"
    S.repaint_prompt = "make it claymation"
    S.midi_offset, S.meter = 0.018, ""
    cmd = _build_repaint_cmd()
    assert cmd[1] == "repaint.py"
    got = dict(zip(cmd[2::2], cmd[3::2]))
    assert got == {"--video": "render_output/song_x.mp4", "--workflow": "reference/wf.json",
                   "--prompt": "make it claymation", "--midi": "song.mid",
                   "--midi-offset": "0.018", "--start-time": "10.685", "--fps": "12",
                   "--seconds": "4", "--output": "render_output/song_x_repaint.mp4"}
    S.repaint_scene, S.repaint_prompt, S.repaint_seconds = "examples/r.yaml", "", 0.0
    cmd = _build_repaint_cmd()
    assert cmd[cmd.index("--scene") + 1] == "examples/r.yaml"
    assert "--prompt" not in cmd and "--seconds" not in cmd
    S.midi_offset = 0.0

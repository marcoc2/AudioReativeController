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

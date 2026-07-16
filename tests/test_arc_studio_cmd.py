"""Tests for ARC Studio's render command construction (no GUI context needed)."""
import sys

from arc_studio import S, _build_render_cmd


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
    assert cmd[1] == "particle_generator.py"
    assert "--clips" not in cmd
    assert cmd[cmd.index("--midi") + 1] == "song.mid"


def test_no_midi_omits_flag():
    _reset_state()
    S.midi_path = None
    cmd = _build_render_cmd()
    assert "--midi" not in cmd

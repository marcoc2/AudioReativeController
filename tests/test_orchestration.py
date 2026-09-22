"""Orchestrating a scene by the song's form: bar windows, track-following colour, stem-driven level."""
import numpy as np
import pytest

from core.rhythm import MidiNote, RhythmGrid
from core.video.layers import (BarWindow, ChladniLayer, FeedbackLayer, FlameLayer, _feature,
                               build_compositor)

W, H = 64, 36
GRID = RhythmGrid(bpm=120.0, time_signature=(4, 4), start_offset=0.0)      # a bar lasts 2 s


def _note(t, pitch, track):
    return MidiNote(time=t, pitch=pitch, velocity=100, channel=0, duration=0.1, track=track, track_index=1)


def test_bar_window_follows_daw_bar_numbers():
    w = BarWindow({"bars": [[2, 3], [6, 6]]}, GRID)
    assert [w(t) for t in (1.9, 2.0, 5.9, 6.0, 10.5, 12.0)] == [0.0, 1.0, 1.0, 0.0, 1.0, 0.0]
    assert BarWindow({"bars": [2, 3]}, GRID)(3.0) == 1.0                   # a single span needs no nesting
    with pytest.raises(ValueError):
        BarWindow({"bars": [[1, 2]]}, None)


def test_bar_window_fades_inside_the_window():
    w = BarWindow({"bars": [[2, 5]], "fade_in": 0.5, "fade_out": 2}, GRID)  # 2 s .. 10 s
    assert w(2.0) == 0.0 and w(2.5) == pytest.approx(0.5) and w(3.0) == 1.0
    assert w(6.0) == 1.0 and w(8.0) == pytest.approx(0.5) and w(9.9) == pytest.approx(0.025)


def test_feature_path():
    feats = {"stems": {"piano": 0.4}, "centroid": 0.2}
    assert _feature(feats, "stems.piano") == 0.4 and _feature(feats, "centroid") == 0.2
    assert _feature(feats, "stems.cello", 0.7) == 0.7 and _feature(feats, "stems") == 0.0


def test_layers_and_post_ops_are_gated_by_bars():
    scene = {"layers": [{"source": "solid", "color": [0, 0, 0]},
                        {"source": "solid", "color": [200, 0, 0], "blend": "add", "bars": [[2, 2]]},
                        {"source": "rgb_split", "amount": 4, "bars": [[3, 3]],
                         "trigger": {"notes": [38], "envelope": 0.5}}]}
    comp = build_compositor(None, scene, [_note(2.5, 38, "snare"), _note(4.5, 38, "snare")],
                            W, H, fps=24, grid=GRID)
    assert comp.frame_at(1.0).max() == 0 and comp.frame_at(2.5)[0, 0].tolist() == [200, 0, 0]
    assert comp._layers[2][2] is not None and comp._layers[2][2](2.5) == 0.0 and comp._layers[2][2](4.5) == 1.0


def test_sand_level_can_come_from_a_stem():
    spec = {"source": "chladni", "grains": 2000, "level_from": "stems.piano"}
    quiet = ChladniLayer(spec, [], W, H, 24, features_at=lambda t: {"stems": {"piano": 0.0}, "texture": {"loudness": 1.0}})
    loud = ChladniLayer(spec, [], W, H, 24, features_at=lambda t: {"stems": {"piano": 1.0}, "texture": {"loudness": 0.0}})
    start = quiet.sys.pos.copy()
    for i in range(6):
        quiet.frame_at(i / 24), loud.frame_at(i / 24)
    assert np.allclose(quiet.sys.pos, start) and not np.allclose(loud.sys.pos, start)


def test_a_gated_layer_does_not_wake_up_with_a_giant_step():
    layer = ChladniLayer({"source": "chladni", "grains": 2000}, [], W, H, 24,
                         features_at=lambda t: {"texture": {"loudness": 0.9}})
    layer.frame_at(0.0)
    before = layer.sys.pos.copy()
    layer.frame_at(60.0)                                                   # a minute later
    assert np.median(np.abs(layer.sys.pos - before)) < 0.1              # a 60 s step would scatter it all over


def test_flame_colour_follows_a_tracks_notes():
    notes = [_note(0.0, 60, "acid"), _note(0.5, 67, "acid"), _note(0.2, 62, "bass")]
    spec = {"source": "flame", "backend": "cpu", "samples": 0.05, "glide": 0.01, "hue": 0.3,
            "hue_notes": {"track": "acid"}}
    layer = FlameLayer(spec, notes, W, H, 24)
    layer.frame_at(0.0), layer.frame_at(0.25)
    assert layer.hue == pytest.approx(0.0, abs=0.02)                       # C
    layer.frame_at(0.5), layer.frame_at(0.75)
    assert layer.hue == pytest.approx(1 / 12, abs=0.02)                    # G, a fifth away: next on the wheel


def test_feedback_hue_shift_recolours_the_trail_only():
    red = np.zeros((H, W, 3), np.uint8); red[..., 0] = 200
    black = np.zeros_like(red)
    plain = FeedbackLayer({"decay": 0.9, "base_scale": 1.0, "max_rotate": 0}, [], 24)
    drift = FeedbackLayer({"decay": 0.9, "base_scale": 1.0, "max_rotate": 0, "hue_shift": 0.5}, [], 24)
    for fb in (plain, drift):
        fb.process(red, 0.0)
    a, b = plain.process(black, 1 / 24), drift.process(black, 1 / 24)
    mid = (H // 2, W // 2)
    assert a[mid][0] > 100 and a[mid][2] == 0                              # the trail stays red
    assert b[mid][2] > 50 and b[mid][0] < a[mid][0]                        # ... or walks towards blue

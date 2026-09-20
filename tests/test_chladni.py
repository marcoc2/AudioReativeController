"""Chladni layer: the sand must find the still lines, leave them on a hit, and replay identically."""
import numpy as np

from core.chladni import ChladniSystem
from core.rhythm import MidiNote
from core.video.layers import ChladniLayer, build_compositor

W, H = 160, 90


def _mean_amplitude(sys_: ChladniSystem) -> float:
    n, m = sys_.current_modes()
    f, _, _ = sys_.field(sys_.pos[:, 0], sys_.pos[:, 1], n, m)
    return float(np.abs(f).mean())


def _run(sys_: ChladniSystem, frames: int, **controls) -> np.ndarray:
    for _ in range(frames):
        frame = sys_.render(1 / 24, **controls)
    return frame


def test_frame_shape_and_dtype():
    frame = ChladniSystem(W, H, grains=2000).render(1 / 24)
    assert frame.shape == (H, W, 3) and frame.dtype == np.uint8


def test_sand_settles_on_the_nodal_lines():
    sys_ = ChladniSystem(W, H, seed=1, grains=4000)
    before = _mean_amplitude(sys_)
    _run(sys_, 72, level=1.0)
    assert _mean_amplitude(sys_) < 0.4 * before


def test_silence_leaves_the_sand_where_it_is():
    sys_ = ChladniSystem(W, H, seed=1, grains=4000)
    start = sys_.pos.copy()
    _run(sys_, 24, level=0.0)
    assert np.allclose(sys_.pos, start)


def test_a_jolt_throws_the_sand_off_its_lines():
    sys_ = ChladniSystem(W, H, seed=1, grains=4000)
    _run(sys_, 72, level=1.0)
    settled = _mean_amplitude(sys_)
    _run(sys_, 3, level=1.0, jolt=1.0)
    assert _mean_amplitude(sys_) > 1.5 * settled


def test_a_new_chord_moves_the_figure():
    sys_ = ChladniSystem(W, H, seed=1, grains=4000, modes=(2, 3))
    _run(sys_, 72, level=1.0, modes=(3, 5))
    assert abs(sys_.n - 3) < 0.2 and abs(sys_.m - 5) < 0.2


def test_pour_draws_only_the_sand_already_on_the_plate():
    empty = ChladniSystem(W, H, seed=1, grains=4000).render(1 / 24, level=1.0, pour=0.0)
    full = ChladniSystem(W, H, seed=1, grains=4000).render(1 / 24, level=1.0, pour=1.0)
    assert full.astype(int).sum() > empty.astype(int).sum()


def test_same_seed_same_film():
    a = _run(ChladniSystem(W, H, seed=5, grains=3000), 12, level=0.8, jolt=0.5, noisiness=0.7)
    b = _run(ChladniSystem(W, H, seed=5, grains=3000), 12, level=0.8, jolt=0.5, noisiness=0.7)
    c = _run(ChladniSystem(W, H, seed=6, grains=3000), 12, level=0.8, jolt=0.5, noisiness=0.7)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_layer_jolts_on_the_hit_and_is_registered():
    feats = lambda t: {"centroid": 0.08, "texture": {"loudness": 0.9}}
    notes = [MidiNote(time=1.0, pitch=36, velocity=100, channel=9, duration=0.1)]
    spec = {"source": "chladni", "grains": 3000, "seed": 2, "root": 10,
            "modes": {10: [2, 3]}, "jolt": {"notes": [36], "envelope": 0.25}}
    layer = ChladniLayer(spec, notes, W, H, 24, features_at=feats)
    jolts = []
    for i in range(36):
        layer.frame_at(i / 24)
        jolts.append(layer._jolt(i / 24))
    assert max(jolts[:24]) == 0.0 and max(jolts[24:]) > 0.5

    comp = build_compositor(None, {"layers": [spec]}, notes, W, H, fps=24, features_at=feats)
    frame = comp.frame_at(0.0)
    assert frame.shape == (H, W, 3) and frame.dtype == np.uint8

"""Fractal flame: the genome maths and the CPU back end run anywhere; the GPU test skips without a 4.3 context."""
import numpy as np
import pytest

from core.flame import (MAX_XFORMS, VARIATIONS, FlameSystem, Genome, Xform, _variations_np,
                        fit_camera, iterate_cpu, make_palette, pack, random_genome)
from core.rhythm import MidiNote
from core.video.layers import FlameLayer, build_compositor

W, H = 96, 54


def _one(name, x, y):
    w = np.zeros((1, len(VARIATIONS)))
    w[0, VARIATIONS.index(name)] = 1.0
    return _variations_np(np.array([[x, y]], dtype=float), w, np.random.default_rng(0))[0]


def test_variations_match_the_paper():
    assert np.allclose(_one("linear", 0.3, -0.2), [0.3, -0.2])
    assert np.allclose(_one("spherical", 2.0, 0.0), [0.5, 0.0])
    assert np.allclose(_one("sinusoidal", np.pi / 2, 0.0), [1.0, 0.0])
    assert np.allclose(_one("horseshoe", 1.0, 1.0), [0.0, 2 / np.sqrt(2)])
    assert np.allclose(_one("bubble", 2.0, 0.0), [1.0, 0.0])
    assert np.allclose(_one("bent", -1.0, -1.0), [-2.0, -0.5])
    assert np.allclose(np.abs(_one("julia", 4.0, 0.0)), [np.sqrt(2), np.sqrt(2)])   # either branch of the root


def test_pack_weights_pose_and_symmetry():
    g = Genome([Xform([0.5, 0, 0, 0, 0.5, 0], weight=1.0), Xform([0.5, 0, 1, 0, 0.5, 0], weight=3.0)])
    pk = pack(g)
    assert pk.n == 2 and np.allclose(pk.meta[:2, 0], [0.25, 1.0])
    turned = pack(g, phase=0.7)
    det = lambda q, i: q.aff0[i, 0] * q.aff1[i, 1] - q.aff0[i, 1] * q.aff1[i, 0]
    assert det(turned, 0) == pytest.approx(det(pk, 0))                  # a rotation keeps the area
    assert not np.allclose(turned.aff0[0, :2], pk.aff0[0, :2])
    assert np.allclose(turned.aff0[:2, 2], pk.aff0[:2, 2])              # ... and the translation
    assert len(g.with_symmetry(3).xforms) == 4 and len(g.with_symmetry(1).xforms) == 2
    assert pack(random_genome(3)).var.shape == (MAX_XFORMS, len(VARIATIONS))


def test_sierpinski_lands_on_the_gasket():
    g = Genome([Xform([0.5, 0, cx, 0, 0.5, cy]) for cx, cy in ((0, 0), (0.5, 0), (0, 0.5))])
    pts, _, ok = iterate_cpu(pack(g), 500, 8, seed=1)
    assert ok.all() and pts.min() >= -1e-3 and (pts.sum(axis=1) <= 1 + 1e-3).all()
    centre_hole = (pts[:, 0] < 0.5) & (pts[:, 1] < 0.5) & (pts.sum(axis=1) > 0.5)   # the removed middle triangle
    assert centre_hole.mean() < 0.01


def test_a_small_turn_moves_every_sample_a_little():
    # fixed random sequence per sample: the picture flows with the coefficients instead of boiling
    g = random_genome(5)
    a, _, _ = iterate_cpu(pack(g, phase=0.30), 400, 4, seed=2)
    b, _, _ = iterate_cpu(pack(g, phase=0.30), 400, 4, seed=2)
    c, _, _ = iterate_cpu(pack(g, phase=0.31), 400, 4, seed=2)
    assert np.array_equal(a, b)
    assert np.median(np.abs(c - a)) < 0.05


def test_camera_frames_the_attractor():
    g = Genome([Xform([0.5, 0, cx + 10, 0, 0.5, cy]) for cx, cy in ((0, 0), (0.5, 0), (0, 0.5))])
    cx, cy, scale = fit_camera(pack(g), 16 / 9)
    assert 19.5 < cx < 21 and 0 < cy < 1 and scale > 0.5                # fixed points sit around x = 20..21


def test_palette_is_a_gradient():
    pal = make_palette(0.8)
    assert pal.shape == (256, 3) and pal.min() >= 0 and pal.max() <= 1


def test_cpu_render_shape_gestures_and_replay():
    make = lambda: FlameSystem(W, H, random_genome(5), seed=1, samples=0.2, backend="cpu")
    a, b = make().render(1 / 24), make().render(1 / 24)
    assert a.shape == (H, W, 3) and a.dtype == np.uint8 and a.max() > 60
    assert np.array_equal(a, b)
    assert not np.array_equal(a, make().render(1 / 24, punch=1.0))
    assert not np.array_equal(a, make().render(1 / 24, twist=1.0))
    assert make().render(1 / 24, level=0.1).mean() < a.mean()


def test_layer_is_registered_and_reacts_on_the_hit_frame():
    feats = lambda t: {"texture": {"loudness": 0.9}}
    notes = [MidiNote(time=0.5, pitch=36, velocity=100, channel=9, duration=0.1)]
    spec = {"source": "flame", "backend": "cpu", "samples": 0.2, "genome": {"random": 5}, "spin": 0.0,
            "hits": [{"notes": [36], "gesture": "punch", "envelope": 0.1}]}
    layer = FlameLayer(spec, notes, W, H, 24, features_at=feats)
    frames = [layer.frame_at(i / 24) for i in range(14)]
    step = [np.abs(frames[i + 1].astype(int) - frames[i]).mean() for i in range(13)]
    assert int(np.argmax(step)) == 11                                   # frame 12 is the first at or after 0.5 s
    comp = build_compositor(None, {"layers": [spec]}, notes, W, H, fps=24, features_at=feats)
    assert comp.frame_at(0.0).shape == (H, W, 3)
    with pytest.raises(ValueError):
        FlameLayer({**spec, "hits": [{"notes": [36], "gesture": "wobble"}]}, notes, W, H, 24)


def test_gpu_backend_draws_the_same_figure_as_the_cpu():
    try:
        gpu = FlameSystem(160, 90, random_genome(5), seed=1, samples=2.0, supersample=1, backend="gpu")
    except Exception:
        pytest.skip("no GPU/moderngl 4.3 context available")
    cpu = FlameSystem(160, 90, random_genome(5), seed=1, samples=1.5, backend="cpu")
    a = gpu.render(1 / 24).astype(float).mean(axis=2)
    b = cpu.render(1 / 24).astype(float).mean(axis=2)
    assert a.shape == b.shape and a.max() > 60
    blur = lambda m: m.reshape(9, 10, 16, 10).mean(axis=(1, 3))          # compare where the light falls, not the grain
    assert np.corrcoef(blur(a).ravel(), blur(b).ravel())[0, 1] > 0.9

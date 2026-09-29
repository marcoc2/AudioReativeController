"""The hybrid compositor: GPU blends and mattes match the numpy ones; frames move only when needed."""
import numpy as np
import pytest

from core.video.layers import Compositor, SolidLayer, blend_frames
from core.video.matte import LayerMatte, apply_matte

W, H = 97, 61                                   # odd sizes: rows are not 4-byte aligned


def _g():
    try:
        from core.video.gpu_compose import GpuComposer
        return GpuComposer(W, H)
    except Exception as exc:
        pytest.skip(f"no OpenGL: {exc}")


def _img(seed):
    return np.random.default_rng(seed).integers(0, 256, (H, W, 3), dtype=np.uint8)


@pytest.mark.parametrize("mode", ["normal", "add", "screen", "multiply"])
@pytest.mark.parametrize("op", [1.0, 0.37])
def test_blend_matches_numpy(mode, op):
    g = _g()
    a, b = _img(1), _img(2)
    got = g.download(g.blend(g.upload(a), g.upload(b), mode, op)).astype(int)
    want = blend_frames(a, b, mode, op).astype(int)
    assert np.abs(got - want).max() <= 1 and (got != want).mean() < 0.01


def test_apply_matches_numpy():
    g = _g()
    a, b = _img(3), _img(4)
    m = np.random.default_rng(5).random((H, W, 1)).astype(np.float32)
    mt = g._tex(1, "f4")
    mt.write(m.tobytes())
    got = g.download(g.apply(g.upload(a), g.upload(b), mt)).astype(int)
    assert np.abs(got - apply_matte(a, b, m).astype(int)).max() <= 1


class _Track:
    """A stand-in mask track: one frame, a blob at half the render's size."""

    def __init__(self):
        lab = np.zeros((H // 2 + 3, W // 2 + 2), np.uint16)
        lab[8:22, 10:30] = 1
        lab[14:18, 36:44] = 2
        self.lab, self.objects = lab, {1: {}, 2: {}}

    def frame_at(self, t):
        return 0 if 0 <= t < 1 else None

    def labels(self, i):
        return self.lab


@pytest.mark.parametrize("spec", [{"feather": 0}, {"feather": 3}, {"grow": 6, "feather": 0},
                                  {"grow": -3, "feather": 2}, {"where": "outside", "objects": [2], "grow": 4},
                                  {"feather": 5, "grow": 8}])
def test_matte_matches_numpy(spec):
    g = _g()
    lm = LayerMatte({"track": "x", **spec}, W, H, tracks={"x": _Track()})
    for t in (0.5, 3.0):                                   # in the track, and past it
        want = lm(t)[..., 0]
        mt = g.matte(lm.selection(t), lm.grow, lm.feather, lm.where == "outside")
        with g.ctx:
            got = np.frombuffer(g._fbo(mt).read(components=1, dtype="f4"), np.float32).reshape(H, W)
        assert np.abs(got - want).max() < 2e-3, spec


class _Invert:
    def process(self, frame, t):
        return 255 - frame


class _Rest:
    """A post-op at rest: hands the very frame back."""

    def process(self, frame, t):
        return frame


def _stack(use_gpu, matte=None):
    c = Compositor(use_gpu=use_gpu)
    c.add(SolidLayer(W, H, (200, 30, 90)))
    c.add(SolidLayer(W, H, (10, 120, 250)), "screen", lambda t: 0.6)
    c.add(_Invert(), "normal", None, matte)
    c.add(_Rest())
    c.add(SolidLayer(W, H, (90, 90, 90)), "multiply", lambda t: 0.8)
    return c


def test_stack_matches_numpy_and_moves_frames_only_when_needed():
    g = _g()
    lm = LayerMatte({"track": "x", "feather": 2}, W, H, tracks={"x": _Track()})
    gpu, cpu = _stack(None, lm), _stack(False, lm)
    a, b = gpu.frame_at(0.5), cpu.frame_at(0.5)
    assert gpu._g is not None and cpu._g is None
    assert np.abs(a.astype(int) - b.astype(int)).max() <= 2
    # up: the base, the screen's top, the invert's result (the matte mixes it with the frame
    # below, still on the GPU), the multiply's top; down: for the invert, for the post-op at
    # rest (it works in numpy too), and the finished frame
    assert (gpu._g.uploads, gpu._g.downloads) == (4, 3)


def test_numpy_only_when_asked_or_one_layer():
    c = Compositor(use_gpu=False)
    c.add(SolidLayer(W, H, (1, 2, 3)))
    c.add(SolidLayer(W, H, (4, 5, 6)), "add")
    c.frame_at(0.0)
    assert c._g is None
    one = Compositor()
    one.add(SolidLayer(W, H, (1, 2, 3)))
    one.frame_at(0.0)
    assert one._g is None

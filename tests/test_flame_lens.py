"""flame_lens: the clip seen through the flame's variations (numpy reference, GPU checked)."""
import numpy as np
import pytest

from core.rhythm.midi_reader import MidiNote
from core.video.flame_lens import FlameLens, LENS_VARIATIONS

W, H = 64, 40


def _frame():
    yy, xx = np.mgrid[0:H, 0:W]
    f = np.zeros((H, W, 3), np.uint8)
    f[..., 0] = (xx * 4) % 256
    f[..., 1] = (yy * 6) % 256
    f[..., 2] = ((xx // 8 + yy // 8) % 2) * 200
    return f


def _lens(**spec):
    return FlameLens({"variations": {"swirl": 0.5}, **spec}, [], W, H, 30)


def test_linear_or_no_amount_leaves_the_picture():
    f = _frame()
    assert _lens(variations={"linear": 1.0}).process(f, 0.0) is f
    assert _lens(amount=0.0).process(f, 0.0) is f
    same = _lens(variations={"swirl": 1e-9}).process(f, 0.0)
    assert np.abs(same.astype(int) - f).max() <= 1               # all but nothing of the lens: the picture


def test_it_bends_and_the_music_sets_how_much():
    snare = [MidiNote(time=1.0, pitch=38, velocity=100, channel=0, duration=0.1, track="snare", track_index=0)]
    lens = FlameLens({"variations": {"bubble": {"hits": {"track": "snare"}, "envelope": 0.5}}}, snare, W, H, 30)
    f = _frame()
    assert lens.process(f, 0.5) is f                              # before the snare: nothing
    hit = lens.process(f, 1.0)
    assert np.abs(hit.astype(int) - f).mean() > 10
    w, _, _ = lens.state(1.25)
    assert 0 < w[LENS_VARIATIONS.index("bubble")] < 1 and abs(w.sum() - 1) < 1e-6
    w, _, _ = FlameLens({"variations": {"swirl": 2.0, "bubble": 2.0}}, [], W, H, 30).state(0.0)
    assert abs(w.sum() - 1) < 1e-6 and w[0] == 0                  # more than 1: scaled down


def test_errors():
    with pytest.raises(ValueError):
        FlameLens({"variations": {}}, [], W, H, 30)
    with pytest.raises(ValueError):
        FlameLens({"variations": {"julia": 1.0}}, [], W, H, 30)   # random: not a lens


def _gpu(w, h):
    try:
        from core.video.gpu_compose import GpuComposer
        return GpuComposer(w, h)
    except Exception as e:
        if "context" in str(e).lower() or "moderngl" in str(e).lower():
            pytest.skip("no GPU/moderngl context available")
        raise


@pytest.mark.parametrize("variations, extra", [
    ({"swirl": 0.6}, {}),
    ({"bubble": 0.5, "sinusoidal": 0.3}, {"turn": 0.7, "zoom": 1.4}),
    ({"spherical": 0.3, "heart": 0.3}, {"center": [0.3, 0.6], "amount": 0.7}),
    ({"horseshoe": 0.5, "eyefish": 0.4, "cylinder": 0.1}, {})])
def test_the_gpu_matches_numpy(variations, extra):
    from core.video.gpu_compose import Frame
    g = _gpu(W, H)
    lens = FlameLens({"variations": variations, **extra}, [], W, H, 30)
    f = _frame()
    want = lens.process(f, 0.0)
    g.begin()
    got = lens.process_gpu(Frame(g, cpu=f), 0.0, g).cpu()
    d = np.abs(got.astype(int) - want.astype(int))
    assert (d > 2).mean() < 0.005 and (d > 0).mean() < 0.05, ((d > 2).mean(), (d > 0).mean(), d.max())


def test_flame_xforms_weigh_with_the_music():
    from core.video.layers import FlameLayer
    kick = [MidiNote(time=1.0, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=0)]
    layer = FlameLayer({"genome": {"random": 5}, "backend": "cpu", "samples": 0.02, "supersample": 1,
                        "weights": {0: {"hits": {"track": "kick"}, "envelope": 0.5, "gain": 3, "bias": 0.1}}},
                       kick, 32, 24, 30)
    base = layer._base_weights[0]
    layer.frame_at(0.5)
    assert abs(layer.sys.genome.xforms[0].weight - 0.1 * base) < 1e-9          # quiet: the xform all but gone
    layer.frame_at(1.0)
    assert abs(layer.sys.genome.xforms[0].weight - 3.1 * base) < 1e-9          # the kick: it takes over
    with pytest.raises(ValueError):
        FlameLayer({"genome": {"random": 5}, "backend": "cpu", "weights": {9: 1.0}}, [], 32, 24, 30)


def test_the_nebulabrot_builds_up():
    try:
        from core.video.nebulabrot import NebulabrotLayer
        neb = NebulabrotLayer({"samples": 0.2, "blue": 0.0}, [], 48, 64, 30)
    except Exception as e:
        if any(k in str(e).lower() for k in ("context", "moderngl", "compute", "430")):
            pytest.skip("no GL 4.3 compute context available")
        raise
    for i in range(4):
        img = neb.frame_at(i / 30)
    assert img.shape == (64, 48, 3) and img[..., 0].max() > 50
    assert img[..., 2].max() == 0                                   # blue's light at 0: dark
    with pytest.raises(ValueError):
        NebulabrotLayer({"caps": [100, 50, 10]}, [], 48, 64, 30)


def test_the_deep_zoom_dives_and_stays_sharp():
    try:
        from core.video.deep_zoom import DeepZoomLayer
        dz = DeepZoomLayer({"decades": 1.0, "iterations": 1500}, [], 48, 64, 30)
    except Exception as e:
        if "context" in str(e).lower() or "moderngl" in str(e).lower():
            pytest.skip("no GPU/moderngl context available")
        raise
    assert dz.decades(0.0) == 0.0 and dz.decades(20.0) == 10.0 and dz.decades(100.0) == 13.0
    deep = dz.frame_at(20.0)                                        # 1e-10 across: past float32 alone
    assert deep.std() > 10                                          # still detail, not a flat blur


def test_the_flight_keeps_out_of_the_walls():
    from core.rhythm.grid import RhythmGrid
    try:
        from core.video.flythrough import FlythroughLayer, de_np
        fly = FlythroughLayer({"step": 0.6}, [], 32, 24, 30, grid=RhythmGrid(bpm=120.0),
                              cuts=lambda until: np.array([2.0]))
    except Exception as e:
        if "context" in str(e).lower() or "moderngl" in str(e).lower():
            pytest.skip("no GPU/moderngl context available")
        raise
    start = fly.pos.copy()
    for i in range(120):
        fly.advance(i / 30)
        assert de_np(fly.pos) > 0.0                                 # never inside the fractal
    assert np.linalg.norm(fly.pos - start) > 0.5                    # and it went somewhere

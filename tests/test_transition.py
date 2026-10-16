"""transition: the next clip spreads over the last one along a front (numpy reference, GPU checked)."""
import numpy as np
import pytest

from core.segment.mask_track import write_mask_track
from core.video.transition import Transition, make_field
from tests.test_mask_track import FPS, H, W, _ColourSegmenter, _frame

CUT = 1.0
OLD, NEW = np.array([200, 40, 40], np.uint8), np.array([40, 200, 40], np.uint8)


def _cuts(until):
    return np.array([t for t in (CUT, 3.0) if t <= until])


def _solid(c):
    return np.broadcast_to(c, (H, W, 3)).copy()


def _layer(**spec):
    return Transition({"length": 0.5, "soft": 0.2, **spec}, W, H, FPS, cuts=_cuts)


def _run(layer, until, step=None):
    """Frames from 0 to ``until``: the old colour before the cut, the new one after."""
    step = step or 1.0 / FPS
    out = None
    for i in range(int(round(until / step)) + 1):
        t = i * step
        out = layer.process(_solid(OLD if t < CUT - 1e-9 else NEW), t)
    return out


@pytest.mark.parametrize("pattern", ["sweep", "radial", "iris", "plasma", "menger", "julia", "spiral"])
def test_the_new_clip_eats_the_old_one(pattern):
    layer = _layer(pattern=pattern)
    before = _run(_layer(pattern=pattern), CUT - 1.0 / FPS)
    assert (before == OLD).all()
    mid = _run(layer, CUT + 0.25)                               # half way
    is_old = (np.abs(mid.astype(int) - OLD).sum(2) < 10).mean()
    is_new = (np.abs(mid.astype(int) - NEW).sum(2) < 10).mean()
    assert 0.1 < is_old < 0.9 and 0.1 < is_new < 0.9, (is_old, is_new)
    assert (_run(_layer(pattern=pattern), CUT + 0.6) == NEW).all()   # done


def test_fields_and_patterns_in_turn():
    rng = np.random.default_rng(0)
    for pat in ("sweep", "radial", "iris", "plasma", "menger", "julia", "spiral"):
        f = make_field(pat, 40, 30, rng)
        assert f.shape == (30, 40) and f.min() >= 0 and f.max() <= 1 and f.max() - f.min() > 0.9
    radial = make_field("radial", 41, 31, rng)
    assert radial[15, 20] < 0.05 and radial[0, 0] > 0.95       # the centre goes first
    layer = _layer(pattern=["sweep", "plasma", "radial"])
    assert [layer.pattern(k) for k in range(4)] == ["sweep", "plasma", "radial", "sweep"]
    shuffled = _layer(pattern=["sweep", "plasma", "radial"], order="shuffle", seed=3)
    assert sorted(shuffled.pattern(k) for k in range(3)) == ["plasma", "radial", "sweep"]
    with pytest.raises(ValueError):
        _layer(pattern="zigzag")
    with pytest.raises(ValueError):
        Transition({}, W, H, FPS)                               # no clips, no cuts


@pytest.fixture(scope="module")
def track(tmp_path_factory):
    folder = tmp_path_factory.mktemp("transition") / "seg"
    write_mask_track("fake.mp4", lambda: (_frame(i) for i in range(20)), _ColourSegmenter(),
                     str(folder), FPS, start_time=0.0, reseed=0.3, log=lambda m: None)
    return str(folder)


def test_the_figures_go_first(track):
    layer = _layer(pattern="figures", track=track)
    out = _run(layer, CUT + 0.2)                                # early in the transition
    figures = layer.track.labels_at(CUT + 0.2) > 0
    new = np.abs(out.astype(int) - NEW).sum(2) < 10
    assert new[figures].mean() > 0.9 and new[~figures].mean() < 0.3
    with pytest.raises(ValueError):
        _layer(pattern="figures")                               # no mask track


def _gpu(w, h):
    try:
        from core.video.gpu_compose import GpuComposer
        return GpuComposer(w, h)
    except Exception as e:
        if "context" in str(e).lower() or "moderngl" in str(e).lower():
            pytest.skip("no GPU/moderngl context available")
        raise


@pytest.mark.parametrize("pattern", ["sweep", "plasma"])
def test_the_gpu_matches_numpy(pattern):
    from core.video.gpu_compose import Frame
    g = _gpu(W, H)
    cpu, gpu = _layer(pattern=pattern), _layer(pattern=pattern)
    for i in range(int((CUT + 0.6) * FPS)):
        t = i / FPS
        f = _frame(i)
        want = cpu.process(f, t)
        g.begin()
        got = gpu.process_gpu(Frame(g, cpu=f), t, g).cpu()
        d = np.abs(got.astype(int) - want.astype(int))
        assert d.max() <= 2 and (d > 0).mean() < 0.01, (t, d.max(), (d > 0).mean())

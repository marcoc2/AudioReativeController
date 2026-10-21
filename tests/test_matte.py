"""mask: on layers — a layer kept inside / outside the objects of a mask track."""
import numpy as np
import pytest

from core.segment.mask_track import write_mask_track
from core.video.layers import Compositor, SolidLayer, build_compositor
from core.video.matte import LayerMatte
from tests.test_mask_track import FPS, H, RED, W, _ColourSegmenter, _frame

START = 13.0


@pytest.fixture(scope="module")
def track(tmp_path_factory):
    folder = tmp_path_factory.mktemp("matte") / "seg"
    meta = write_mask_track("fake.mp4", lambda: (_frame(i) for i in range(20)), _ColourSegmenter(),
                            str(folder), FPS, start_time=START, reseed=0.3, log=lambda m: None)
    red = next(o["id"] for o in meta["objects"] if tuple(o["colour"]) == RED)
    return str(folder), red


class _Invert:
    """A post-op: the negative of what is below."""

    def process(self, frame, t):
        return 255 - frame


def _red_square(i):
    m = np.zeros((H, W), bool)
    m[40:80, 10 + 4 * i:50 + 4 * i] = True
    return m


def test_a_source_only_inside_one_object(track):
    folder, red = track
    comp = Compositor()
    comp.add(SolidLayer(W, H, (0, 0, 0)))
    comp.add(SolidLayer(W, H, (255, 255, 255)), "normal", None,
             LayerMatte({"track": folder, "objects": [red], "feather": 0}, W, H))
    for i in (0, 7):
        out = comp.frame_at(START + i / FPS)
        assert np.array_equal(out[..., 0] == 255, _red_square(i))      # it moves with the square


def test_a_post_op_only_outside(track):
    folder, _ = track
    comp = Compositor()
    comp.add(SolidLayer(W, H, (10, 20, 30)))
    comp.add(_Invert(), "normal", None, LayerMatte({"track": folder, "where": "outside",
                                                    "feather": 0}, W, H))
    out = comp.frame_at(START)
    objects = (out == (10, 20, 30)).all(axis=2)                         # untouched on the objects
    assert objects[60, 30] and objects[20, 130]                        # red and blue squares
    assert (out[5, 5] == (245, 235, 225)).all()                         # inverted on the background


def test_scene_mask_uses_the_render_default_track_and_scales(track):
    folder, red = track
    cfg = {"layers": [{"source": "solid", "color": [0, 0, 0]},
                      {"source": "solid", "color": [255, 255, 255],
                       "mask": {"objects": [red], "feather": 0}}]}
    comp = build_compositor(None, cfg, [], 2 * W, 2 * H, masks=folder)
    out = comp.frame_at(START)
    assert out.shape == (2 * H, 2 * W, 3)
    white = out[..., 0] > 127
    assert white[120, 60] and not white[120, 150] and 0.08 < white.mean() < 0.09   # 80×80 of 320×240


def test_before_and_after_the_track(track):
    folder, _ = track
    inside = LayerMatte({"track": folder, "feather": 0}, W, H)
    outside = LayerMatte({"track": folder, "where": "outside", "feather": 0}, W, H)
    for t in (START - 1.0, START + 5.0):
        assert inside(t).max() == 0.0 and outside(t).min() == 1.0


def test_grow_and_feather(track):
    folder, red = track
    hard = LayerMatte({"track": folder, "objects": [red], "feather": 0}, W, H)(START)
    grown = LayerMatte({"track": folder, "objects": [red], "feather": 0, "grow": 6}, W, H)(START)
    soft = LayerMatte({"track": folder, "objects": [red], "feather": 6}, W, H)(START)
    assert grown.sum() > hard.sum()
    assert 0.0 < soft[60, 50, 0] < 1.0 and soft[60, 30, 0] > 0.99       # soft edge, solid inside


def test_mistakes_are_named(track):
    folder, _ = track
    with pytest.raises(ValueError, match="needs a track"):
        LayerMatte({}, W, H)
    with pytest.raises(ValueError, match="where"):
        LayerMatte({"track": folder, "where": "around"}, W, H)
    with pytest.raises(ValueError, match="not in"):
        LayerMatte({"track": folder, "objects": [99]}, W, H)
    with pytest.raises(ValueError, match="base layer"):
        build_compositor(None, {"layers": [{"source": "solid", "mask": {"track": folder}}]}, [], W, H)

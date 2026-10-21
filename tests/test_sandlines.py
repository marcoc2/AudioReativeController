"""Sand lines post-op (GPU): skips without a moderngl context."""
import numpy as np
import pytest

from core.rhythm import MidiNote
from core.sandlines import line_mask
from core.video.layers import SandLinesLayer

W, H = 160, 90


def _sand(**kw):
    try:
        from core.sandlines import SandLines
        return SandLines(W, H, **kw)
    except Exception:
        pytest.skip("no GPU/moderngl context available")


def _drawing(x0=40, page=(230, 200, 120)):
    img = np.full((H, W, 3), page, np.uint8)                  # a coloured page ...
    img[20:70, x0:x0 + 3] = 10                                # ... with an inked stroke
    return img


def test_the_inverted_near_white_is_the_ink():
    m = line_mask(_drawing())
    assert m[45, 41] and not m[45, 100]
    assert m.sum() == 50 * 3


def _home_pixels(s):
    pos, _ = s.state()
    px = ((pos[:, 0] + s.aspect) / (2 * s.aspect) * W).astype(int)
    py = ((pos[:, 1] + 1) / 2 * H).astype(int)
    return px, py


def test_the_sand_lies_on_the_lines_and_the_picture_is_gone():
    s = _sand(grains=4000)
    s.lines(_drawing(), 0)
    out = s.render(1 / 30)
    px, py = _home_pixels(s)
    assert np.all((px >= 39) & (px <= 44) & (py >= 19) & (py <= 71))
    assert out[45, 41].mean() > 120                           # the stroke: white sand
    assert out[45, 120].max() < 20                             # the page: gone, black


def test_a_ring_throws_the_sand_and_the_plate_brings_it_back():
    s = _sand(grains=4000, pull=6.0)
    s.lines(_drawing(), 0)
    s.render(1 / 30)
    x0 = s.state()[0][:, 0].copy()
    cx = x0.mean() - 0.3                                       # a ring from left of the stroke
    lifted = False
    for f in range(12):
        age = f / 30
        s.render(1 / 30, [(cx, 0.0, 0.3 + age * 3.2, 1.0 - age / 0.6)])
        lifted |= bool((s.state()[1] > 0).any())
    moved = s.state()[0][:, 0] - x0
    assert moved.mean() > 0.02 and lifted                      # thrown outwards (away from the ring), and up
    for f in range(90):
        s.render(1 / 30)
    back = np.abs(s.state()[0][:, 0] - x0)
    assert back.mean() < moved.mean() * 0.3                    # and brought back to the line


def test_without_the_plate_it_stays_scattered_and_a_new_drawing_calls_it_away():
    s = _sand(grains=4000, pull=0.0)
    s.lines(_drawing(), 0)
    s.render(1 / 30)
    x0 = s.state()[0][:, 0].copy()
    for f in range(12):
        age = f / 30
        s.render(1 / 30, [(x0.mean() - 0.3, 0.0, 0.3 + age * 3.2, 1.0 - age / 0.6)])
    for f in range(60):
        s.render(1 / 30)
    assert (s.state()[0][:, 0] - x0).mean() > 0.02             # it stays where it was thrown
    s2 = _sand(grains=4000)
    s2.lines(_drawing(40), 0)
    s2.render(1 / 30)
    assert s2.lines(_drawing(110), 1)
    for f in range(60):
        s2.render(1 / 30)
    px, _ = _home_pixels(s2)
    assert np.mean((px >= 108) & (px <= 114)) > 0.9            # it travelled to the new stroke
    assert not s2.lines(np.full((H, W, 3), 128, np.uint8), 2)  # a picture with no lines: the old ones stay


def test_layer_takes_a_new_drawing_on_a_cut_and_rings_on_notes():
    _sand()
    notes = [MidiNote(time=0.5, pitch=60, velocity=100, channel=0, duration=0.1, track="piano", track_index=7)]
    spec = {"source": "sandlines", "grains": 4000, "rings": {"track": "piano"}}
    layer = SandLinesLayer(spec, notes, W, H, 30)
    a, b = _drawing(40), _drawing(110, page=(120, 170, 240))  # another clip: another page
    out = layer.process(a, 0.0)
    assert out[45, 41].mean() > 120 and out[45, 111].max() < 20
    for f in range(1, 60):
        out = layer.process(b if f >= 10 else a, f / 30)      # the clip cuts at frame 10
    assert out[45, 111].mean() > 120 and out[45, 41].mean() < 60
    with pytest.raises(ValueError):
        SandLinesLayer({**spec, "center": "left"}, notes, W, H, 30)

"""Shockwave post-op (GPU): skips without a moderngl context."""
import numpy as np
import pytest

from core.rhythm import MidiNote
from core.video.layers import ShockwaveLayer, build_compositor


def _wave(**kw):
    try:
        from core.shockwave import Shockwave
        return Shockwave(160, 90, **kw)
    except Exception:
        pytest.skip("no GPU/moderngl context available")


def _drawing():
    """A cartoon: light fill, a dark outline square in the middle."""
    img = np.full((90, 160, 3), 230, np.uint8)
    img[:, :, 2] = 120                                   # yellowish paper
    img[25:65, 60:100] = 20                              # the square's ink ...
    img[28:62, 63:97] = (200, 120, 90)                   # ... and its fill
    return img


def test_invert_turns_ink_into_white_lines_on_dark():
    out = _wave().render(_drawing())
    assert out[26, 80].min() > 200                       # the outline is white
    assert out[5, 5].max() < 120                         # the paper went dark
    assert out[45, 80].max() < 120                       # so did the fill


def test_the_ring_tints_the_lines_it_crosses_and_bends_the_picture():
    s = _wave()
    calm = s.render(_drawing(), hue=0.9)
    # a ring from the middle, at the square's edge (frame units: 2 tall, the square's side is ~0.44 from centre)
    hit = s.render(_drawing(), rings=[(0.0, 0.0, 0.44, 1.0)], hue=0.9)
    line = (26, 80)
    assert calm[line][1] > 200 and hit[line][1] < calm[line][1] - 40   # white -> magenta (less green)
    ground = (5, 5)
    assert abs(int(hit[ground].sum()) - int(calm[ground].sum())) < 10  # far from the ring: untouched
    assert not np.array_equal(calm, hit)


def test_without_invert_only_the_ink_changes():
    s = _wave(invert=False)
    img = _drawing()
    out = s.render(img, rings=[(0.0, 0.0, 0.44, 1.0)])
    assert np.abs(out[5, 5].astype(int) - img[5, 5]).max() <= 2       # the paper as it was


def test_layer_rings_on_notes_and_punches():
    _wave()
    notes = [MidiNote(time=0.5, pitch=60, velocity=100, channel=0, duration=0.2, track="piano", track_index=7),
             MidiNote(time=0.5, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=3)]
    spec = {"source": "shockwave", "rings": {"track": "piano"}, "punch": {"track": "kick", "envelope": 0.3}}
    layer = ShockwaveLayer(spec, notes, 160, 90, 24)
    before = layer.process(_drawing(), 0.4)
    during = layer.process(_drawing(), 0.6)
    after = layer.process(_drawing(), 1.5)
    assert not np.array_equal(before, during)
    assert np.array_equal(before, after)                 # the ring has died, the punch has faded
    with pytest.raises(ValueError):
        ShockwaveLayer({**spec, "center": "corner"}, notes, 160, 90, 24)
    with pytest.raises(ValueError):
        build_compositor(None, {"layers": [spec]}, notes, 160, 90, fps=24)   # a post-op is not a base


def test_scatter_turns_the_ring_to_sand_that_replays():
    s = _wave(scatter=1.0)
    ring = [(0.0, 0.0, 0.44, 1.0)]
    a = s.render(_drawing(), ring, frame_index=5)
    assert np.array_equal(a, s.render(_drawing(), ring, frame_index=5))       # same frame, same grains
    assert not np.array_equal(a, s.render(_drawing(), ring, frame_index=6))   # the next frame, new grains
    clean = _wave().render(_drawing(), ring)
    assert np.abs(a[2:8, 2:8].astype(int) - clean[2:8, 2:8]).max() <= 2      # far from the ring: the same ground
    assert np.abs(a[20:30, 70:90].astype(int) - clean[20:30, 70:90]).mean() > 10   # in the ring: scattered

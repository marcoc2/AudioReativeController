"""Drops post-op (GPU): skips without a moderngl context."""
import numpy as np
import pytest

from core.rhythm import MidiNote
from core.video.layers import DropsLayer, build_compositor

W, H = 160, 90


def _drops(**kw):
    try:
        from core.drops import Drops
        return Drops(W, H, **kw)
    except Exception:
        pytest.skip("no GPU/moderngl context available")


def _two_squares():
    img = np.full((H, W, 3), 235, np.uint8)
    for x0 in (10, 90):
        img[20:70, x0:x0 + 60] = 15                  # outline ...
        img[23:67, x0 + 3:x0 + 57] = 235             # ... around a light fill
    return img


def test_a_drop_fills_its_region_and_stops_at_the_lines():
    d = _drops(speed=8, grow=5.0, life=10.0, rim=0.0)
    d.drop(40, 45, (1.0, 0.0, 0.0), radius=3, t=0.0)
    for f in range(12):
        out = d.render(_two_squares(), f / 30)
    assert out[45, 30, 0] > 200 and out[45, 30, 1] < 80       # the left square: red
    assert out[25, 15, 0] > 200 and out[25, 15, 1] < 80       # ... right into its corner
    assert tuple(out[21, 40]) == (15, 15, 15)                 # the outline stays drawn
    assert tuple(out[10, 80]) == (235, 235, 235)              # outside: untouched
    assert tuple(out[45, 120]) == (235, 235, 235)             # the other square: untouched


def test_it_spreads_over_time_and_dries():
    d = _drops(speed=2, grow=5.0, life=1.0, fade=0.3, rim=0.0)
    d.drop(40, 45, (1.0, 0.0, 0.0), radius=2, t=0.0)
    early = d.render(_two_squares(), 0.0)
    for f in range(1, 10):
        later = d.render(_two_squares(), f / 30)
    red = lambda img: int(((img[..., 0] > 200) & (img[..., 1] < 80)).sum())
    assert 0 < red(early) < red(later)                        # it grows
    for f in range(10, 45):
        dry = d.render(_two_squares(), f / 30)
    assert red(dry) == 0                                      # and dries away after ``life``


def test_layer_drops_on_notes_with_the_colour_under_it():
    _drops()
    notes = [MidiNote(time=0.5, pitch=38, velocity=100, channel=0, duration=0.1, track="snare", track_index=2)]
    spec = {"source": "drops", "hits": {"track": "snare"}, "at": "center", "speed": 20, "grow": 3, "rim": 0}
    layer = DropsLayer(spec, notes, W, H, 30)
    img = np.full((H, W, 3), (200, 120, 60), np.uint8)       # an orange page
    before = layer.process(img, 0.4)
    assert np.array_equal(before, img)
    for f in range(10):
        after = layer.process(img, 0.5 + f / 30)
    r, g, b = (int(c) for c in after[H // 2, W // 2])
    assert r > g > b and (r - b) > (200 - 60)                 # still orange, but more vivid
    with pytest.raises(ValueError):
        DropsLayer({**spec, "color": "plaid"}, notes, W, H, 30)
    with pytest.raises(ValueError):
        build_compositor(None, {"layers": [spec]}, notes, W, H, fps=30)   # a post-op is not a base


def test_spiral_shows_an_arm_winding_out_with_gaps_between_its_turns():
    d = _drops(speed=20, grow=5.0, life=10.0, rim=0.0, shape="spiral", pitch=12, turns=3.0, arm=0.5)
    page = np.full((H, W, 3), 235, np.uint8)
    d.drop(80, 45, (1.0, 0.0, 0.0), radius=3, t=0.0)
    red = lambda img: (img[..., 0] > 200) & (img[..., 1] < 80)
    early = d.render(page, 0.0)
    for f in range(1, 45):
        out = d.render(page, f / 30)
    assert red(early).sum() < red(out).sum()                  # it winds out
    ring = [red(out)[45, 80 + k] for k in range(4, 36)]       # along a radius: arm, gap, arm, gap ...
    changes = sum(a != b for a, b in zip(ring, ring[1:]))
    assert changes >= 4
    blob = _drops(speed=20, grow=5.0, life=10.0, rim=0.0)
    blob.drop(80, 45, (1.0, 0.0, 0.0), radius=3, t=0.0)
    for f in range(30):
        solid = blob.render(page, f / 30)
    assert red(out).sum() < red(solid).sum()                  # the gaps leave the page showing
    from core.drops import Drops
    with pytest.raises(ValueError):
        Drops(W, H, shape="star")


def test_spiral_whirls_on_its_own_even_when_it_has_stopped_growing():
    d = _drops(speed=20, grow=5.0, life=10.0, rim=0.0, shape="spiral", pitch=12, turns=50.0, arm=0.5, whirl=2.0)
    page = np.full((H, W, 3), 235, np.uint8)
    d.drop(80, 45, (1.0, 0.0, 0.0), radius=3, t=0.0)
    for f in range(40):
        a = d.render(page, f / 30)                            # grown: the whole region is reached
    b = d.render(page, 40 / 30)
    red = lambda img: (img[..., 0] > 200) & (img[..., 1] < 80)
    assert (red(a) != red(b)).sum() > 100                      # ... and the arms still turn

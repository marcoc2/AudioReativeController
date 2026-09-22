"""Slit-scan (GPU): skips without a moderngl context."""
import numpy as np
import pytest

from core.rhythm import MidiNote
from core.video.layers import Compositor, SlitScanLayer, build_compositor


def _scan(mode="down", depth=40, w=64, h=40):
    try:
        from core.slitscan import SlitScan
        return SlitScan(w, h, depth=depth, mode=mode)
    except ValueError:
        raise
    except Exception:
        pytest.skip("no GPU/moderngl context available")


def _bar(x, w=64, h=40):
    f = np.zeros((h, w, 3), np.uint8)
    f[:, x] = 255
    return f


def _col(img, row):
    return int(np.argmax(img[row, :, 0]))


def test_no_delay_is_the_frame_itself():
    s = _scan()
    for x in range(5):
        out = s.render(_bar(x), 0.0)
    assert np.array_equal(out, _bar(4))


def test_a_moving_bar_leans_back_in_time():
    s = _scan("down")
    for x in range(30):
        out = s.render(_bar(x), delay=20.0)
    # the top row is now, the bottom row about 20 frames ago
    assert _col(out, 0) == 29
    assert abs(_col(out, 39) - (29 - 20 * 39 / 40)) <= 1.5
    assert _col(out, 0) > _col(out, 20) > _col(out, 39)


def test_delay_is_clamped_to_what_has_been_seen():
    s = _scan("down")
    for x in range(3):
        out = s.render(_bar(x), delay=30.0)
    assert _col(out, 39) == 0                  # only three frames exist: the oldest is frame 0


def test_modes_and_wave():
    for mode in ("up", "right", "left", "radial", "center"):
        s = _scan(mode)
        for x in range(20):
            out = s.render(_bar(x), delay=10.0)
        assert not np.array_equal(out, _bar(19)), mode
    s = _scan("down")
    for x in range(30):
        calm = s.render(_bar(x), delay=0.0)
    waved = s.render(_bar(30), delay=0.0, wave_front=0.5, wave_depth=20.0)
    assert _col(calm, 20) == 29 and _col(waved, 20) < 20       # the band in the middle shows the past
    with pytest.raises(ValueError):
        _scan("sideways")


class _Moving:
    def __init__(self):
        self.x = 0

    def frame_at(self, t):
        self.x += 1
        return _bar(self.x % 64)


def test_layer_keeps_watching_outside_its_window():
    _scan()
    notes = [MidiNote(time=0.5, pitch=36, velocity=100, channel=0, duration=0.1, track="kick", track_index=1)]
    spec = {"source": "slitscan", "depth": 40, "delay": 20, "amount": 0,
            "wave": {"track": "kick", "envelope": 0.5, "depth": 10}}
    layer = SlitScanLayer(spec, notes, 64, 40, 24)
    comp = Compositor()
    comp.add(_Moving())
    gate = {"open": False}
    comp.add(layer, "normal", lambda t: 1.0 if gate["open"] else 0.0)
    for f in range(30):                         # gated: frames pass untouched, but are remembered
        out = comp.frame_at(f / 24)
    assert _col(out, 39) == _col(out, 0)
    gate["open"] = True
    out = comp.frame_at(30 / 24)
    assert _col(out, 0) - _col(out, 39) >= 15   # the past was there the moment it opened
    with pytest.raises(ValueError):
        build_compositor(None, {"layers": [{"source": "slitscan"}]}, notes, 64, 40, fps=24)   # not a base


def test_works_after_another_gpu_layer():
    """Each GPU layer owns a context; the scan must draw in its own, not the last one used."""
    _scan()
    from core.eyes import EyesSystem
    eyes = EyesSystem(160, 90, rows=3, supersample=1)
    s = _scan(w=160, h=90, depth=8)
    for i in range(4):
        frame = eyes.render(i / 30)
        out = s.render(frame, 2.0)
    assert out.mean() > 30 and abs(out.mean() - frame.mean()) < 10

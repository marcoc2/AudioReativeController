"""datamosh: at a cut, the old shot's pixels move with the new shot's motion until the snare."""
import numpy as np
import pytest

from core.rhythm.midi_reader import MidiNote
from core.video.datamosh import Datamosh

W, H, FPS, CUT = 96, 64, 10, 1.0


def _cuts(until):
    return np.array([t for t in (CUT,) if t <= until])


def _shot(i):
    """Before the cut: a red square, still. After: a blue bar sliding right 3 px a frame."""
    f = np.full((H, W, 3), 30, np.uint8)
    t = i / FPS
    if t < CUT - 1e-9:
        f[20:44, 20:44] = (220, 40, 40)
    else:
        x = 10 + 3 * (i - int(CUT * FPS))
        f[:, x:x + 10] = (40, 40, 220)
        f[::8] = 90                                          # texture, so the flow has something to follow
    return f


def test_the_old_picture_rides_the_new_motion_until_the_snare():
    snare = [MidiNote(time=1.6, pitch=38, velocity=100, channel=0, duration=0.1, track="snare", track_index=0)]
    mosh = Datamosh({"length": 3.0, "until": {"track": "snare"}, "block": 0, "leak": 0.0}, snare, W, H, FPS, cuts=_cuts)
    outs = [mosh.process(_shot(i), i / FPS) for i in range(20)]
    assert np.array_equal(outs[9], _shot(9))                 # before the cut: untouched
    red = lambda img: (img[..., 0] > 150) & (img[..., 2] < 100)
    assert red(outs[10]).sum() > 400                         # the cut: still the old square
    assert red(outs[14]).sum() > 100                         # moshing: the old pixels linger
    assert np.array_equal(outs[16], _shot(16))               # the snare at 1.6: the real picture


def test_it_needs_the_cuts():
    with pytest.raises(ValueError):
        Datamosh({}, [], W, H, FPS)

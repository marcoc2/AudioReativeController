"""Eye wall (GPU): skips without a moderngl context."""
import numpy as np
import pytest

from core.rhythm import MidiNote
from core.video.layers import EyesLayer, build_compositor


def _sys(w=160, h=90):
    try:
        from core.eyes import EyesSystem
        return EyesSystem(w, h, rows=4, seed=1, supersample=1)
    except Exception:
        pytest.skip("no GPU/moderngl context available")


def test_frame_shape_and_determinism():
    a, b = _sys().render(0.5, pupil=0.4), _sys().render(0.5, pupil=0.4)
    assert a.shape == (90, 160, 3) and a.dtype == np.uint8 and a.max() > 100
    assert np.array_equal(a, b)
    assert not np.array_equal(a, _sys().render(0.5, pupil=0.4, gaze=(0.8, 0.5)))    # the eyes moved


def test_blink_closes_the_lids():
    s = _sys()
    assert not np.array_equal(s.render(0.5), s.render(0.5, blink=1.0))


def test_layer_saccades_on_the_hit_and_is_registered():
    _sys()
    notes = [MidiNote(time=0.5, pitch=38, velocity=100, channel=0, duration=0.1, track="snare", track_index=1)]
    spec = {"source": "eyes", "rows": 4, "supersample": 1, "hits": [{"track": "snare", "gesture": "saccade"}]}
    layer = EyesLayer(spec, notes, 160, 90, 24)
    layer.frame_at(0.0)
    before = layer._target.copy()
    layer.frame_at(0.6)
    assert not np.allclose(layer._target, before)
    comp = build_compositor(None, {"layers": [spec]}, notes, 160, 90, fps=24)
    assert comp.frame_at(0.0).shape == (90, 160, 3)
    with pytest.raises(ValueError):
        EyesLayer({**spec, "hits": [{"track": "snare", "gesture": "wink"}]}, notes, 160, 90, 24)

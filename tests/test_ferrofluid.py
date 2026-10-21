"""Ferrofluid pool (GPU): skips without a moderngl context."""
import numpy as np
import pytest

from core.rhythm import MidiNote
from core.video.layers import FerrofluidLayer, build_compositor


def _sys(w=160, h=90):
    try:
        from core.ferrofluid import FerrofluidSystem
        return FerrofluidSystem(w, h, supersample=1)
    except Exception:
        pytest.skip("no GPU/moderngl context available")


def test_frame_shape_and_determinism():
    a, b = _sys().render(0.5), _sys().render(0.5)
    assert a.shape == (90, 160, 3) and a.dtype == np.uint8 and a.max() > 100
    assert np.array_equal(a, b)


def test_the_field_raises_spikes_and_the_lattice_turns():
    s = _sys(320, 180)
    flat, spiky = s.render(0.5, magnets=[(0.0, 0.35, 0.0)]), s.render(0.5, magnets=[(0.0, 0.35, 1.5)])
    centre = (slice(60, 120), slice(110, 210))
    assert np.abs(flat[centre].astype(int) - spiky[centre]).mean() > 10
    assert not np.array_equal(spiky, s.render(0.5, magnets=[(0.0, 0.35, 1.5)], rotation=0.3))


def test_a_hit_rings_the_magnet():
    _sys()
    notes = [MidiNote(time=0.5, pitch=36, velocity=127, channel=0, duration=0.1, track="kick", track_index=1)]
    spec = {"source": "ferrofluid", "supersample": 1, "base": 0.3,
            "magnets": [{"track": "kick", "at": [0, 0.35]}]}
    layer = FerrofluidLayer(spec, notes, 160, 90, 24)
    for f in range(12):                                  # 0 .. 0.46 s: nothing happens yet
        layer.frame_at(f / 24)
    assert abs(layer._magnets[0]["x"]) < 1e-9
    peak = max(abs((layer.frame_at(0.5 + f / 24), layer._magnets[0]["x"])[1]) for f in range(1, 6))
    assert peak > 0.2                                    # the spring was kicked
    for f in range(6, 100):
        layer.frame_at(0.5 + f / 24)
    assert abs(layer._magnets[0]["x"]) < 0.05            # ... and settled again
    comp = build_compositor(None, {"layers": [spec]}, notes, 160, 90, fps=24)
    assert comp.frame_at(0.0).shape == (90, 160, 3)
    with pytest.raises(ValueError):
        FerrofluidLayer({**spec, "magnets": [{"at": [0, 0]}] * 5}, notes, 160, 90, 24)

"""Mouth wall (GPU): skips without a moderngl context."""
import numpy as np
import pytest

from core.rhythm import MidiNote
from core.video.layers import MouthsLayer, build_compositor


def _sys(w=160, h=90):
    try:
        from core.mouths import MouthsSystem
        return MouthsSystem(w, h, rows=2, seed=1, supersample=1)
    except Exception:
        pytest.skip("no GPU/moderngl context available")


def test_frame_shape_and_determinism():
    a, b = _sys().render(0.5, jaw=0.5), _sys().render(0.5, jaw=0.5)
    assert a.shape == (90, 160, 3) and a.dtype == np.uint8 and a.max() > 100
    assert np.array_equal(a, b)


def test_every_control_changes_the_mouths():
    s = _sys()
    base = s.render(0.5, jaw=0.4)
    for kw in ({"jaw": 0.9}, {"jaw": 0.4, "spread": 1.0}, {"jaw": 0.4, "spread": -1.0},
               {"jaw": 0.4, "clench": 1.0}, {"jaw": 0.4, "tint": 1.0}):
        assert not np.array_equal(base, s.render(0.5, **kw)), kw


def test_open_mouth_is_darker_inside():
    s = _sys(320, 180)
    shut, open_ = s.render(0.5, jaw=0.0), s.render(0.5, jaw=1.0)
    assert open_.mean() < shut.mean()          # the wall fills with dark mouths


def test_layer_follows_the_voice_and_the_hits(tmp_path):
    _sys()
    import soundfile as sf
    from core.voice import SR
    # a vowel only in the second half: the mouths open when it starts
    t = np.arange(int(1.0 * SR)) / SR
    y = np.where(t > 0.5, 0.4 * np.sign(np.sin(2 * np.pi * 140 * t)) * np.exp(-((t * 140) % 1) * 8), 0.0)
    wav = tmp_path / "vox.wav"
    sf.write(wav, y, SR)
    notes = [MidiNote(time=0.3, pitch=38, velocity=100, channel=0, duration=0.1, track="snare", track_index=1)]
    spec = {"source": "mouths", "rows": 2, "supersample": 1, "voice": str(wav),
            "hits": [{"track": "snare", "gesture": "clench", "envelope": 0.2}]}
    layer = MouthsLayer(spec, notes, 160, 90, 24)
    quiet = layer.frame_at(0.1)
    assert layer._smooth["jaw"] < 0.05
    clenched = layer.frame_at(0.32)
    assert not np.array_equal(quiet, clenched)
    for f in range(7):                         # 0.6 .. 0.85 s, inside the vowel
        layer.frame_at(0.6 + f / 24)
    assert layer._smooth["jaw"] > 0.2
    comp = build_compositor(None, {"layers": [spec]}, notes, 160, 90, fps=24)
    assert comp.frame_at(0.0).shape == (90, 160, 3)
    with pytest.raises(ValueError):
        MouthsLayer({**spec, "hits": [{"track": "snare", "gesture": "kiss"}]}, notes, 160, 90, 24)

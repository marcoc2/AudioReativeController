"""Voice -> mouth shape: formants of synthetic vowels land where the vowels are."""
import numpy as np
from scipy.signal import lfilter

from core.voice import SR, analyse, formants, read_voice


def _vowel(f1, f2, f3=2800, dur=0.5, f0=140):
    n = int(dur * SR)
    x = np.zeros(n)
    x[::int(SR / f0)] = 1.0                                  # glottal pulses
    for f, bw in ((f1, 80), (f2, 100), (f3, 150)):           # vocal-tract resonances
        r = np.exp(-np.pi * bw / SR)
        th = 2 * np.pi * f / SR
        x = lfilter([1 - r], [1, -2 * r * np.cos(th), r * r], x)
    return x / np.abs(x).max() * 0.5


def test_formants_of_vowels():
    for f1, f2 in ((800, 1250), (300, 2300), (320, 800)):
        e1, e2 = formants(_vowel(f1, f2)[2000:2330])
        assert abs(e1 - f1) < 80 and abs(e2 - f2) < 150, (f1, f2, e1, e2)


def test_vowels_shape_the_mouth():
    a, i, u = (analyse(_vowel(*f)) for f in ((800, 1250), (300, 2300), (320, 800)))
    mid = slice(10, -10)
    assert a.jaw[mid].mean() > i.jaw[mid].mean() + 0.2          # "a" opens the jaw
    assert i.spread[mid].mean() > 0.6                           # "i" spreads the lips
    assert u.spread[mid].mean() < -0.4                          # "u" rounds them


def test_silence_keeps_the_mouth_shut():
    tr = analyse(np.zeros(SR))
    assert tr.jaw.max() == 0.0 and tr.at(0.5) == (0.0, 0.0, 0.0)


def test_read_voice_caches(tmp_path):
    import soundfile as sf
    wav = tmp_path / "v.wav"
    sf.write(wav, _vowel(700, 1200, dur=0.4), SR)
    a = read_voice(wav)
    assert (tmp_path / "v.wav.voice.npz").exists()
    b = read_voice(wav)
    assert np.array_equal(a.jaw, b.jaw)

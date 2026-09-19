"""Texture features: synthetic signals whose surface behaviour is known in advance."""
import librosa
import numpy as np
import pytest
import soundfile as sf

from core.feature_extractor import AudioFeatureExtractor
from core.texture import TEXTURE_NAMES, texture_features

SR, N_FFT, HOP = 22050, 2048, 512
RATE = SR / HOP


def _tone(freq, seconds, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return amp * np.sin(2 * np.pi * freq * t)


def _texture(y):
    mag = np.abs(librosa.stft(y.astype(np.float32), n_fft=N_FFT, hop_length=HOP))
    return texture_features(mag, SR, N_FFT, HOP)


def _middle(x):
    """Ignore the edges, where analysis windows hang over the ends of the signal."""
    n = len(x)
    return x[n // 4: 3 * n // 4]


def test_all_features_present_aligned_and_in_range():
    y = _tone(440, 6.0)
    tex = _texture(y)
    n_frames = 1 + len(y) // HOP
    assert set(tex) == set(TEXTURE_NAMES)
    for name, arr in tex.items():
        assert arr.shape == (n_frames,), name
        assert np.all(np.isfinite(arr)), name
    for name in ("loudness", "harmonic_change", "percussive", "noisiness", "tremolo_depth"):
        assert tex[name].min() >= 0.0 and tex[name].max() <= 1.0, name
    assert np.abs(tex["swell"]).max() <= 1.0


def test_steady_tone_is_a_flat_tonal_surface():
    tex = _texture(_tone(440, 8.0))
    assert _middle(tex["loudness"]).min() > 0.9
    assert np.abs(_middle(tex["swell"])).max() < 0.05
    assert _middle(tex["harmonic_change"]).max() < 0.2
    assert _middle(tex["percussive"]).max() < 0.1
    assert _middle(tex["noisiness"]).max() < 0.1
    assert _middle(tex["tremolo_depth"]).max() < 0.05


def test_loudness_is_on_a_db_scale():
    # 24 dB quieter is half of the 48 dB range, not 1/16 as a linear scale would give
    y = np.concatenate([_tone(440, 4.0, amp=0.5), _tone(440, 4.0, amp=0.5 / 10 ** (24 / 20))])
    tex = _texture(y)
    n = len(tex["loudness"])
    assert tex["loudness"][n // 4] == pytest.approx(1.0, abs=0.03)
    assert tex["loudness"][3 * n // 4] == pytest.approx(0.5, abs=0.03)


def test_swell_sign_follows_fade_direction():
    ramp = 10 ** (np.linspace(-30, 0, int(8.0 * SR)) / 20)      # +3.75 dB/s
    up = _texture(_tone(440, 8.0) * ramp)
    down = _texture(_tone(440, 8.0) * ramp[::-1])
    assert _middle(up["swell"]).mean() == pytest.approx(3.75 / 6.0, abs=0.08)
    assert _middle(down["swell"]).mean() == pytest.approx(-3.75 / 6.0, abs=0.08)


def test_tremolo_rate_and_depth():
    y = _tone(440, 10.0)
    t = np.arange(len(y)) / SR
    tex = _texture(y * (1.0 + 0.5 * np.sin(2 * np.pi * 5.0 * t)))
    assert np.median(_middle(tex["tremolo_rate"])) == pytest.approx(5.0, abs=0.1)
    assert np.median(_middle(tex["tremolo_depth"])) == pytest.approx(0.5, abs=0.12)


def test_harmonic_change_peaks_at_the_note_change():
    y = np.concatenate([_tone(261.63, 5.0), _tone(392.0, 5.0)])   # C4 -> G4 at t = 5 s
    hc = _texture(y)["harmonic_change"]
    peak_t = np.argmax(hc) / RATE
    assert peak_t == pytest.approx(5.0, abs=0.3)
    assert hc[int(2.5 * RATE)] < 0.2 and hc[int(7.5 * RATE)] < 0.2


def test_noise_is_noisy_and_clicks_are_percussive():
    rng = np.random.default_rng(0)
    noise = _texture(0.3 * rng.standard_normal(int(6.0 * SR)))
    assert _middle(noise["noisiness"]).mean() > 0.8

    # percussive is the share of attack *at each instant*: high on a click,
    # nothing between clicks, and nothing at all in a sustained tone
    clicks = np.zeros(int(6.0 * SR))
    clicks[SR // 2:: SR] = 1.0                                   # one click per second, at x.5 s
    perc = _texture(clicks + _tone(440, 6.0, amp=0.01))["percussive"]
    around_click = perc[int(2.4 * RATE): int(2.6 * RATE)]
    assert around_click.max() > 0.8          # an isolated attack keeps its height
    assert perc[int(3.0 * RATE)] < 0.1       # ... and is gone half a second later
    assert _middle(_texture(_tone(440, 6.0))["percussive"]).mean() < 0.1


def test_silence_reads_as_nothing():
    tex = _texture(np.zeros(int(5.0 * SR)))
    for name in TEXTURE_NAMES:
        assert np.allclose(tex[name], 0.0), name


def test_extractor_delivers_texture_per_frame(tmp_path, monkeypatch):
    import core.feature_extractor as mod
    monkeypatch.setattr(mod, "StemService", None)
    wav = tmp_path / "tone.wav"
    sf.write(wav, _tone(440, 5.0).astype(np.float32), SR)
    ex = AudioFeatureExtractor(str(wav), fps=24, separation_mode="none")

    for smoothing in (False, True, True):     # third call takes the smoothed branch
        f = ex.get_features_at_time(2.5, use_smoothing=smoothing)
        assert set(f["texture"]) == set(TEXTURE_NAMES)
        assert f["texture"]["loudness"] > 0.9
        assert all(isinstance(v, float) for v in f["texture"].values())
    assert len(ex.texture["loudness"]) == ex.spectrogram.shape[1]

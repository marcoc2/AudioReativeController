import numpy as np
import pytest
import soundfile as sf

from core.feature_extractor import AudioFeatureExtractor


@pytest.fixture(autouse=True)
def _no_stems(monkeypatch):
    import core.feature_extractor as mod
    monkeypatch.setattr(mod, "StemService", None)


def _sine(freq: float, seconds: float, sr: int = 22050, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def _click_track(bpm: float, seconds: float, sr: int = 22050) -> np.ndarray:
    y = np.zeros(int(seconds * sr), dtype=np.float32)
    beat_period = 60.0 / bpm
    click_len = int(0.01 * sr)
    t_click = np.arange(click_len) / sr
    pulse = (np.sin(2 * np.pi * 1000 * t_click) * np.exp(-t_click * 200)).astype(np.float32)
    t = 0.0
    while t + click_len / sr < seconds:
        i = int(t * sr)
        y[i : i + click_len] += pulse
        t += beat_period
    return y


@pytest.fixture
def click_extractor(tmp_path):
    sr = 22050
    y = _click_track(bpm=120, seconds=4.0, sr=sr)
    wav = tmp_path / "click.wav"
    sf.write(wav, y, sr)
    return AudioFeatureExtractor(str(wav), fps=24, separation_mode="none")


def test_precomputed_arrays_shapes(click_extractor):
    ex = click_extractor
    T = ex.stft_mag.shape[1]
    assert ex.centroid.shape == (T,)
    assert ex.chroma.shape == (12, T)
    assert ex.dominant_pitch.shape == (T,)
    assert ex.flux.shape == (T,)
    assert ex.onset_mask.shape == (T,)
    for name in ex.subband_names:
        assert ex.subbands[name].shape == (T,)


def test_centroid_range_and_values(click_extractor):
    ex = click_extractor
    # centroid normalized 0..1
    assert ex.centroid.min() >= 0.0
    assert ex.centroid.max() <= 1.0
    # 1kHz clicks in a 22050 Hz signal -> non-trivial centroid at active frames
    active = ex.flux > 0.1
    if active.any():
        assert ex.centroid[active].mean() > 0.0


def test_flux_peaks_near_onsets(click_extractor):
    ex = click_extractor
    # flux should have peaks; onset detection should mark at least a few frames
    assert ex.flux.max() == pytest.approx(1.0, abs=1e-6)
    assert ex.onset_mask.sum() >= 4  # ~8 beats in 4s at 120bpm


def test_chroma_dominant_pitch_for_pure_tone(tmp_path):
    sr = 22050
    # A4 = 440 Hz -> pitch class 9 (A)
    y = _sine(440.0, seconds=2.0, sr=sr)
    wav = tmp_path / "a4.wav"
    sf.write(wav, y, sr)
    ex = AudioFeatureExtractor(str(wav), fps=24, separation_mode="none")
    # take median over steady-state frames (skip attack)
    mid = ex.dominant_pitch[len(ex.dominant_pitch) // 4 : -len(ex.dominant_pitch) // 4]
    assert int(np.bincount(mid.astype(np.int64), minlength=12).argmax()) == 9


def test_subbands_energy_dominant_band_for_pure_tone(tmp_path):
    sr = 22050
    # 1 kHz pure tone -> energy concentrated in "mid" (500..2000 Hz) pre-normalization
    y = _sine(1000.0, seconds=2.0, sr=sr)
    wav = tmp_path / "tone_1k.wav"
    sf.write(wav, y, sr)
    ex = AudioFeatureExtractor(str(wav), fps=24, separation_mode="none")
    # Reconstruct un-normalized band energies to compare absolute power
    bin_freqs = np.arange(ex.stft_mag.shape[0]) * (sr / ex.n_fft)
    powers = {}
    for name, (lo, hi) in ex.SUBBAND_RANGES.items():
        mask = (bin_freqs >= lo) & (bin_freqs < hi)
        powers[name] = ex.stft_mag[mask].mean() if mask.any() else 0.0
    assert max(powers, key=powers.get) == "mid"


def test_features_at_time_contains_new_keys(click_extractor):
    ex = click_extractor
    f = ex.get_features_at_time(1.0, apply_gate=False)
    assert f is not None
    assert "centroid" in f and 0.0 <= f["centroid"] <= 1.0
    assert "chroma" in f and f["chroma"].shape == (12,)
    assert "dominant_pitch" in f and 0 <= f["dominant_pitch"] < 12
    assert "flux" in f and 0.0 <= f["flux"] <= 1.0
    assert "onset" in f and isinstance(f["onset"], bool)
    assert "subbands" in f and set(f["subbands"]) == set(ex.subband_names)

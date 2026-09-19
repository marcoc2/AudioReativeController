"""Texture features — what moves in a sound when nothing attacks.

Flux, onsets and band energies describe *events*: something hits, the value
jumps. A pad, a drone or a wash of reverb has no events, and those features
read it as a flat line. What such material does have is a surface that
breathes, drifts and shimmers. These features describe that surface:

    loudness         0..1   how loud, on a dB scale, so quiet passages keep
                            their dynamics instead of being squashed near zero
    swell           -1..1   is it growing (+) or fading (-), and how fast
    harmonic_change  0..1   how different the harmony of the next second is from
                            the last one: peaks where the chord changes
    percussive       0..1   share of the sound (in amplitude) that is attack
                            rather than sustain: 0 = pure pad, ~1 = drums alone.
                            Drums buried under a loud pad read low — it is a
                            share of the mix, not a drum detector
    noisiness        0..1   tonal (0) to noise-like (1): a sine vs. wind or cymbals
    tremolo_depth    0..1   how strongly the volume pulsates ...
    tremolo_rate     Hz     ... and how fast (0 when there is no pulsation)

Everything is computed once for the whole track and returned as one array
per feature, aligned with the STFT frames it was given.
"""
from __future__ import annotations

from typing import Dict

import librosa
import numpy as np
from scipy.ndimage import gaussian_filter1d

TEXTURE_NAMES = ("loudness", "swell", "harmonic_change", "percussive",
                 "noisiness", "tremolo_depth", "tremolo_rate")

DYNAMIC_RANGE_DB = 48.0      # loudness 0..1 spans this many dB below the track's peak
SWELL_FULL_SCALE = 6.0       # dB per second that reads as swell = +-1
HARMONY_WINDOW = 1.0         # seconds of harmony compared on each side of "now"
ANALYSIS_RATE = 20.0         # Hz; texture moves slowly, the heavy steps run at this rate
TREMOLO_RANGE = (1.0, 12.0)  # Hz; slower is a swell, faster is roughness, not pulsation
TREMOLO_WINDOW = 4.0         # seconds of envelope analysed per tremolo estimate
NOISINESS_FLOOR = 1e-3       # spectral flatness that reads as noisiness = 0
NOISINESS_MAX_HZ = 8000.0    # above this, lossy codecs leave silence that fakes "tonal"
HPSS_KERNEL = (11, 31)       # (time frames, freq bins): ~0.5 s sustained counts as harmonic
HPSS_MARGIN = (1.0, 2.0)     # percussive must clearly win a bin to be counted as attack
PERCUSSIVE_RELEASE = 0.15    # seconds for a held attack peak to fall to ~37%
HARMONY_MIN_PEAK = 0.1       # harmonic_change is scaled by the track's own peaks, but never
                             # by less than this: a static drone must not be blown up to 1


def _smooth(x: np.ndarray, seconds: float, rate: float) -> np.ndarray:
    sigma = seconds * rate
    return gaussian_filter1d(x, sigma=sigma, mode="nearest") if sigma > 0.5 else x


def _peak_hold(x: np.ndarray, release_seconds: float, rate: float) -> np.ndarray:
    """Rise instantly, fall exponentially.

    An attack is shorter than any smoothing window worth using, so a symmetric
    blur flattens it (a lone click read 0.23 instead of ~1). Holding the peak
    keeps its height and still gives consumers a few frames to react.
    """
    decay = float(np.exp(-1.0 / max(release_seconds * rate, 1e-6)))
    out = np.empty(len(x))
    level = 0.0
    for i, v in enumerate(x):
        level = v if v > level else level * decay
        out[i] = level
    return out


def _loudness_db(stft_mag: np.ndarray) -> np.ndarray:
    rms = np.sqrt(np.mean(stft_mag ** 2, axis=0))
    peak = float(rms.max())
    if peak <= 0:
        return np.full(len(rms), -DYNAMIC_RANGE_DB)
    return np.maximum(20.0 * np.log10(np.maximum(rms, 1e-12) / peak), -DYNAMIC_RANGE_DB)


def _harmonic_change(chroma: np.ndarray, rate: float) -> np.ndarray:
    """Cosine distance between the mean chroma of the last and the next window."""
    n = chroma.shape[1]
    w = max(1, int(round(HARMONY_WINDOW * rate)))
    csum = np.concatenate([np.zeros((12, 1)), np.cumsum(chroma, axis=1)], axis=1)
    idx = np.arange(n)
    past = csum[:, idx] - csum[:, np.maximum(idx - w, 0)]
    future = csum[:, np.minimum(idx + w, n)] - csum[:, idx]
    norm = np.linalg.norm(past, axis=0) * np.linalg.norm(future, axis=0)
    cos = np.where(norm > 1e-9, np.sum(past * future, axis=0) / np.maximum(norm, 1e-9), 1.0)
    return np.clip(1.0 - cos, 0.0, 1.0)


def _tremolo(envelope: np.ndarray, rate: float):
    """Dominant volume pulsation per frame: (depth 0..1, rate in Hz)."""
    n = len(envelope)
    win = int(TREMOLO_WINDOW * rate)
    if n < win or win < 8:
        return np.zeros(n), np.zeros(n)
    hop = max(1, win // 8)
    freqs = np.fft.rfftfreq(win, 1.0 / rate)
    band = (freqs >= TREMOLO_RANGE[0]) & (freqs <= TREMOLO_RANGE[1])
    if not band.any():
        return np.zeros(n), np.zeros(n)
    window = np.hanning(win)
    centres, depths, rates = [], [], []
    for start in range(0, n - win + 1, hop):
        seg = envelope[start:start + win]
        mean = float(seg.mean())
        centres.append(start + win // 2)
        if mean <= 1e-9:
            depths.append(0.0); rates.append(0.0)
            continue
        # linear detrend: a fade is a swell, not a tremolo
        seg = seg - np.polyval(np.polyfit(np.arange(win), seg, 1), np.arange(win))
        spec = np.abs(np.fft.rfft(seg * window)) / (window.sum() / 2.0)   # sinusoid amplitude
        k = int(np.argmax(np.where(band, spec, 0.0)))
        depth = min(1.0, float(spec[k]) / mean)                            # modulation index
        # parabolic interpolation around the peak: bins are 1/TREMOLO_WINDOW Hz apart
        shift = 0.0
        if 0 < k < len(spec) - 1:
            a, b, c = spec[k - 1], spec[k], spec[k + 1]
            if (a - 2 * b + c) != 0:
                shift = float(np.clip(0.5 * (a - c) / (a - 2 * b + c), -0.5, 0.5))
        depths.append(depth)
        rates.append(float(freqs[k] + shift * (freqs[1] - freqs[0])) if depth > 0.02 else 0.0)
    frames = np.arange(n)
    return np.interp(frames, centres, depths), np.interp(frames, centres, rates)


def texture_features(stft_mag: np.ndarray, sr: int, n_fft: int, hop_length: int) -> Dict[str, np.ndarray]:
    """Compute every texture feature for a magnitude STFT of shape (bins, frames)."""
    n_frames = stft_mag.shape[1]
    rate = sr / hop_length
    frames = np.arange(n_frames)

    db = _loudness_db(stft_mag)
    loudness = np.clip(1.0 + _smooth(db, 0.15, rate) / DYNAMIC_RANGE_DB, 0.0, 1.0)
    swell = np.clip(np.gradient(_smooth(db, 0.5, rate)) * rate / SWELL_FULL_SCALE, -1.0, 1.0)

    # Harmonic/percussive split and harmony run on a decimated STFT: they are
    # the expensive steps and what they measure does not change within 50 ms.
    step = max(1, int(round(rate / ANALYSIS_RATE)))
    slow = stft_mag[:, ::step]
    slow_rate = rate / step
    slow_frames = frames[::step]
    harmonic, percussive_mag = librosa.decompose.hpss(slow, kernel_size=HPSS_KERNEL, margin=HPSS_MARGIN)
    e_all, e_p = np.sum(slow ** 2, axis=0), np.sum(percussive_mag ** 2, axis=0)
    percussive = _peak_hold(np.sqrt(e_p / np.maximum(e_all, 1e-12)), PERCUSSIVE_RELEASE, slow_rate)

    chroma = librosa.feature.chroma_stft(S=harmonic ** 2, sr=sr, n_fft=n_fft)
    harmonic_change = _harmonic_change(chroma, slow_rate)
    harmonic_change = np.clip(
        harmonic_change / max(float(np.percentile(harmonic_change, 99)), HARMONY_MIN_PEAK), 0.0, 1.0)

    bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft) < NOISINESS_MAX_HZ
    flat = librosa.feature.spectral_flatness(S=slow[bins] ** 2, power=1.0)[0]
    span = -np.log10(NOISINESS_FLOOR)
    noisiness = np.clip(1.0 + np.log10(np.maximum(flat, NOISINESS_FLOOR)) / span, 0.0, 1.0)
    noisiness = _smooth(noisiness, 0.15, slow_rate)

    up = lambda x: np.interp(frames, slow_frames, x)
    audible = loudness > 0.02    # in silence chroma and flatness are just noise
    tremolo_depth, tremolo_rate = _tremolo(np.sqrt(np.mean(stft_mag ** 2, axis=0)), rate)

    return {
        "loudness": loudness,
        "swell": swell * audible,
        "harmonic_change": up(harmonic_change) * audible,
        "percussive": up(percussive) * audible,
        "noisiness": up(noisiness) * audible,
        "tremolo_depth": tremolo_depth * audible,
        "tremolo_rate": tremolo_rate * audible,
    }

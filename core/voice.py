"""Voice → mouth shape: what a singing mouth does, read from a vocal stem.

Two things shape a mouth while it sings, and both can be heard:

    level   how loud the voice is                              -> how far the jaw drops
    F1      the first formant, high on open vowels ("a")       -> the jaw drops further
    F2      the second formant, high on front vowels ("i", "e") -> lips spread wide;
            low on back rounded vowels ("o", "u")               -> lips round and pout

Formants come from linear prediction (LPC) over short windows of the stem:
the poles of the all-pole filter sit at the resonances of the vocal tract.
``read_voice(path)`` returns ``VoiceTrack`` sampled every 10 ms; ``at(t)``
interpolates ``(jaw, spread, level)`` with jaw in 0..1 and spread in -1..1
(round .. wide). The analysis is cached next to the stem (``.voice.npz``).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SR = 11025          # formants live below ~3.5 kHz: a low rate keeps the LPC fit on them
HOP = 0.010         # seconds between analysis frames
WIN = 0.030         # analysis window
ORDER = 10          # LPC order: ~5 resonances below Nyquist
_CACHE_VERSION = 1


@dataclass
class VoiceTrack:
    times: np.ndarray
    jaw: np.ndarray
    spread: np.ndarray
    level: np.ndarray

    def at(self, t: float) -> tuple:
        if len(self.times) == 0:
            return 0.0, 0.0, 0.0
        return (float(np.interp(t, self.times, self.jaw, left=0.0, right=0.0)),
                float(np.interp(t, self.times, self.spread, left=0.0, right=0.0)),
                float(np.interp(t, self.times, self.level, left=0.0, right=0.0)))


def formants(frame: np.ndarray, sr: int = SR, order: int = ORDER) -> tuple:
    """(F1, F2) in Hz of one windowed frame, or (nan, nan) if no clear resonances."""
    import librosa
    x = np.append(frame[0], frame[1:] - 0.63 * frame[:-1]) * np.hamming(len(frame))   # pre-emphasis
    if not np.any(x):
        return np.nan, np.nan
    try:
        a = librosa.lpc(x.astype(np.float64), order=order)
    except (FloatingPointError, ValueError):
        return np.nan, np.nan
    roots = np.roots(a)
    roots = roots[np.imag(roots) > 0]
    freqs = np.angle(roots) * sr / (2 * np.pi)
    bw = -sr / np.pi * np.log(np.abs(roots) + 1e-12)
    keep = (freqs > 200) & (bw < 500)
    f = np.sort(freqs[keep])
    f1 = next((x for x in f if 250 <= x <= 1000), np.nan)
    f2 = next((x for x in f if x > (f1 if f1 == f1 else 600) + 150 and 700 <= x <= 3000), np.nan)
    return f1, f2


def analyse(y: np.ndarray, sr: int = SR) -> VoiceTrack:
    """The mouth-shape track of a mono signal at ``sr``."""
    hop, win = int(round(HOP * sr)), int(round(WIN * sr))
    n = max(0, 1 + (len(y) - win) // hop)
    rms = np.array([np.sqrt(np.mean(y[i * hop:i * hop + win] ** 2)) for i in range(n)])
    ref = np.percentile(rms[rms > 0], 95) if np.any(rms > 0) else 1.0
    level = np.clip(rms / (ref + 1e-12), 0.0, 1.0)
    voiced = level > 0.12
    f1 = np.full(n, np.nan)
    f2 = np.full(n, np.nan)
    for i in np.flatnonzero(voiced):
        f1[i], f2[i] = formants(y[i * hop:i * hop + win], sr)
    # hold the last good reading through gaps, then smooth over ~50 ms
    def fill(v, default):
        out, last = np.empty_like(v), default
        for i, x in enumerate(v):
            last = x if x == x else last
            out[i] = last
        return out
    f1, f2 = fill(f1, 500.0), fill(f2, 1400.0)
    k = np.ones(5) / 5
    f1, f2 = np.convolve(f1, k, "same"), np.convolve(f2, k, "same")
    open_ = np.clip((f1 - 300.0) / 500.0, 0.0, 1.0)
    jaw = np.clip(level ** 0.8 * (0.45 + 0.55 * open_), 0.0, 1.0)
    spread = np.clip((f2 - 1350.0) / 700.0, -1.0, 1.0) * np.clip(level * 3.0, 0.0, 1.0)
    return VoiceTrack(np.arange(n) * HOP + WIN / 2, jaw, spread, level)


def read_voice(path) -> VoiceTrack:
    """Analyse a vocal stem (any format librosa reads), cached beside it."""
    path = Path(path)
    cache = path.with_name(path.name + ".voice.npz")
    stamp = np.array([path.stat().st_mtime, path.stat().st_size, _CACHE_VERSION])
    if cache.exists():
        try:
            z = np.load(cache)
            if np.array_equal(z["stamp"], stamp):
                return VoiceTrack(z["times"], z["jaw"], z["spread"], z["level"])
        except Exception:
            pass
    import librosa
    y, _ = librosa.load(str(path), sr=SR, mono=True)
    track = analyse(y)
    try:
        np.savez(cache, stamp=stamp, times=track.times, jaw=track.jaw, spread=track.spread, level=track.level)
    except OSError:
        pass
    return track

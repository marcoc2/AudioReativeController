import json

import numpy as np
import pytest

from core.rhythm.analyzer import analyze, analyze_file
from core.rhythm.grid import RhythmGrid


def _click_track(bpm: float, seconds: float, sr: int = 22050) -> np.ndarray:
    """Synthesize a click track: short pulses at `bpm` for `seconds`."""
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


def test_analyze_click_track_infers_bpm():
    sr = 22050
    y = _click_track(bpm=120, seconds=8.0, sr=sr)
    grid = analyze(y, sr=sr, fps=24)
    assert isinstance(grid, RhythmGrid)
    assert grid.bpm == pytest.approx(120.0, rel=0.05)
    assert grid.beats is not None and len(grid.beats) > 8
    assert grid.downbeats is not None and len(grid.downbeats) > 0


def test_analyze_time_signature_3_4():
    sr = 22050
    y = _click_track(bpm=120, seconds=8.0, sr=sr)
    grid = analyze(y, sr=sr, time_signature=(3, 4), fps=24)
    assert grid.time_signature == (3, 4)
    # downbeats spaced roughly one bar apart
    if len(grid.downbeats) >= 2:
        gap = float(np.median(np.diff(grid.downbeats)))
        assert gap == pytest.approx(grid.bar_duration, rel=0.1)


def test_analyze_file_roundtrip_and_cache(tmp_path):
    import soundfile as sf

    sr = 22050
    y = _click_track(bpm=120, seconds=6.0, sr=sr)
    wav_path = tmp_path / "click.wav"
    sf.write(wav_path, y, sr)

    from core.rhythm import analyzer as mod
    original_cache = mod.CACHE_DIR
    mod.CACHE_DIR = tmp_path / "rhythm_cache"
    try:
        grid1 = analyze_file(str(wav_path), fps=24)
        assert grid1.bpm == pytest.approx(120.0, rel=0.05)
        cache_files = list(mod.CACHE_DIR.glob("*.json"))
        assert len(cache_files) == 1

        # Prove the cache is consulted: rewrite the JSON with a sentinel BPM
        # and assert the next call returns it rather than re-analyzing.
        payload = json.loads(cache_files[0].read_text())
        payload["bpm"] = 42.0
        cache_files[0].write_text(json.dumps(payload))
        grid2 = analyze_file(str(wav_path), fps=24)
        assert grid2.bpm == pytest.approx(42.0)

        # use_cache=False forces a re-analysis and overwrites the sentinel
        grid3 = analyze_file(str(wav_path), fps=24, use_cache=False)
        assert grid3.bpm == pytest.approx(120.0, rel=0.05)
    finally:
        mod.CACHE_DIR = original_cache


def test_analyze_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        analyze_file(str(tmp_path / "does_not_exist.wav"))

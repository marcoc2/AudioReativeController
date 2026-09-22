"""Film grain post-op."""
import numpy as np

from core.rhythm import MidiNote, RhythmGrid
from core.video.layers import GrainLayer, build_compositor

H, W = 36, 64


def _grey(v):
    return np.full((H, W, 3), v, np.uint8)


def test_grain_is_noise_that_replays():
    g = GrainLayer({"amount": 0.2, "size": 1, "seed": 4}, [], 24)
    a, b, c = g.process(_grey(128), 1.0), g.process(_grey(128), 1.0), g.process(_grey(128), 1.5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    assert 20 < a.astype(float).std() < 70 and abs(a.astype(float).mean() - 128) < 6


def test_grain_fades_in_blacks_and_whites_and_grows_with_size():
    fine = GrainLayer({"amount": 0.2, "size": 1, "color": 0.0}, [], 24)
    assert fine.process(_grey(0), 0.0).astype(float).std() < fine.process(_grey(128), 0.0).astype(float).std() / 2
    coarse = GrainLayer({"amount": 0.2, "size": 4, "color": 0.0}, [], 24).process(_grey(128), 0.0).astype(float)
    fine_ = fine.process(_grey(128), 0.0).astype(float)
    step = lambda img: np.abs(np.diff(img[..., 0], axis=1)).mean() / img[..., 0].std()
    assert step(coarse) < 0.5 * step(fine_)                                # bigger grains: neighbours agree more
    assert GrainLayer({"amount": 0.0}, [], 24).process(_grey(90), 0.0).tolist() == _grey(90).tolist()


def test_grain_bursts_on_hits_and_obeys_bars():
    grid = RhythmGrid(bpm=120.0, time_signature=(4, 4), start_offset=0.0)
    notes = [MidiNote(time=3.0, pitch=36, velocity=100, channel=9, duration=0.1, track="kick", track_index=1)]
    scene = {"layers": [{"source": "solid", "color": [128, 128, 128]},
                        {"source": "grain", "amount": 0.05, "burst": 0.6, "bars": [[2, 2]],
                         "trigger": {"track": "kick", "envelope": 0.2}}]}
    comp = build_compositor(None, scene, notes, W, H, fps=24, grid=grid)
    quiet, hit, outside = (comp.frame_at(t).astype(float).std() for t in (2.5, 3.0, 4.5))
    assert hit > 2 * quiet and outside == 0


def test_rgb_noise_matches_gegl_semantics():
    from core.video.layers import RgbNoiseLayer
    mult = RgbNoiseLayer({"amount": 0.3, "seed": 2}, [], 24)
    black, mid = mult.process(_grey(0), 0.0), mult.process(_grey(128), 0.0)
    assert black.max() == 0                                                # multiplicative: blacks stay clean
    assert mid.astype(float).std() > 5 and abs(mid.astype(float).mean() - 128) < 8
    assert np.array_equal(mid, mult.process(_grey(128), 0.0))              # replays
    mono = RgbNoiseLayer({"amount": 0.3, "independent": False}, [], 24).process(_grey(128), 0.0)
    assert np.array_equal(mono[..., 0], mono[..., 1]) and not np.array_equal(mid[..., 0], mid[..., 1])
    add = RgbNoiseLayer({"amount": 0.3, "correlated": False, "linear": False}, [], 24).process(_grey(0), 0.0)
    assert add.max() > 0                                                   # additive noise lifts blacks
    assert RgbNoiseLayer({"amount": 0.0}, [], 24).process(_grey(77), 0.0).tolist() == _grey(77).tolist()

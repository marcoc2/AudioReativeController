"""Tests for the cube TV room — the standalone, non-reactive generator.

The claim under test is the one the whole thing rests on: what the camera
*sees* decides what it *hears*. So the tests check the visibility measurement
itself, and that the mixer turns that measurement into gain and position.
"""
import math
import subprocess

import numpy as np
import pytest


def _field(**over):
    from core.cubes import CubeField
    spec = {"n_cubes": 6, "faces_with_clips": 6, "face_resolution": 32,
            "corridor_depth": 12.0, "radius_min": 1.2, "radius_max": 3.0}
    spec.update(over)
    return CubeField(160, 96, spec, seed=11, n_screens=spec["n_cubes"])


def _gpu_field(**over):
    try:
        return _field(**over)
    except Exception:
        pytest.skip("no GPU/moderngl context available")


# --- what the camera sees -------------------------------------------------

def test_visibility_share_and_pan_track_the_picture():
    field = _gpu_field()
    share, pan = field.visibility(0.0)
    assert share.shape == (6,) and pan.shape == (6,)
    assert share.sum() > 0                       # some TV is on screen
    assert share.sum() <= 1.0                    # it is a share of the frame
    assert np.all(share >= 0)
    # a cube nobody can see contributes nothing and sits dead centre
    for i in np.flatnonzero(share == 0):
        assert pan[i] == 0.0
    assert np.all(np.abs(pan) <= 1.0)


def test_visibility_grows_as_a_tv_approaches():
    """The core promise: coming towards the camera means occupying more frame."""
    field = _gpu_field(n_cubes=1, radius_min=0.35, radius_max=0.4)
    # cube 0 sits at depth phase z0; sweep phase so it walks the corridor in
    phases = np.linspace(0.0, 0.999, 40)
    shares = np.array([field.visibility(p)[0][0] for p in phases])
    assert shares.max() > shares[shares > 0].min() * 3   # a real near/far swing
    # and the swing is monotone while it approaches: find the peak, check the run-up
    peak = int(shares.argmax())
    run_up = shares[max(0, peak - 8):peak + 1]
    assert np.all(np.diff(run_up) >= -1e-9)


def test_visibility_is_deterministic_and_loops():
    field = _gpu_field()
    a, pa = field.visibility(0.25)
    b, pb = field.visibility(0.25)
    assert np.array_equal(a, b) and np.array_equal(pa, pb)
    start, _ = field.visibility(0.0)
    end, _ = field.visibility(0.999999)
    assert np.allclose(start, end, atol=2e-3)    # seamless: phase 0 == phase 1


def test_per_cube_layers_give_each_tv_its_own_clip():
    field = _gpu_field(n_cubes=3, face_resolution=32)
    screens = [np.full((32, 32, 3), c, np.uint8)
               for c in ((220, 0, 0), (0, 220, 0), (0, 0, 220))]
    per_cube = field.render(0.3, screens, [0, 0, 0], cube_layers=np.arange(3))
    # default (per face-index) must be untouched by the new argument
    per_face = field.render(0.3, screens, [0, 0, 0])
    assert not np.array_equal(per_cube, per_face)
    lit = per_cube.reshape(-1, 3)
    lit = lit[lit.max(1) > 60]
    assert len({int(c) for c in lit.argmax(1)}) >= 2   # at least two clips on screen


# --- what the camera hears ------------------------------------------------

def test_smooth_envelope_attacks_and_releases():
    from cube_tv import smooth_envelope
    w = np.zeros((60, 1))
    w[10:30] = 1.0
    env = smooth_envelope(w, fps=30, attack_ms=40, release_ms=200)[:, 0]
    assert env[10] < env[20] <= 1.0          # ramps up, never overshoots
    assert env[31] < env[29]                 # falls after the TV leaves
    assert env[35] > 0.0                     # but tails instead of cutting
    assert np.all(env >= 0)


def _tone(freq=220.0, seconds=1.0, sr=48000):
    t = np.arange(int(seconds * sr)) / sr
    return np.repeat((0.5 * np.sin(2 * math.pi * freq * t))[:, None], 2, 1).astype(np.float32)


def test_mixer_is_louder_when_the_tv_fills_more_of_the_frame():
    """The whole point: a TV that grows in frame grows in the mix."""
    from cube_tv import Mixer
    fps, sr = 10, 48000
    env = np.concatenate([np.full(10, 0.05), np.full(10, 0.50)])[:, None]
    mx = Mixer(2.0, fps, sr, pan_width=1.0)
    mx.add_segment(_tone(seconds=2.0), env[:, 0], np.zeros(20), 0, 20)
    out = mx.finish(0.0)
    far = float(np.sqrt((out[: int(0.8 * sr)] ** 2).mean()))
    near = float(np.sqrt((out[int(1.2 * sr):] ** 2).mean()))
    ratio = near / far
    assert 8.0 < ratio < 12.0, f"near/far = {ratio:.2f}"   # share went up 10x


def test_mixer_pans_a_right_side_tv_to_the_right():
    from cube_tv import Mixer

    def run(pan_value):
        mx = Mixer(1.0, 10, 48000, pan_width=1.0)
        mx.add_segment(_tone(), np.full(10, 0.3), np.full(10, pan_value), 0, 10)
        return mx.finish(0.0)

    right, left = run(1.0), run(-1.0)
    assert np.abs(right[:, 1]).max() > np.abs(right[:, 0]).max() * 10
    assert np.abs(left[:, 0]).max() > np.abs(left[:, 1]).max() * 10


def test_mixer_survives_mute_clips_and_invisible_tvs():
    from cube_tv import Mixer
    mx = Mixer(1.0, 10, 48000, pan_width=1.0)
    mx.add_segment(np.zeros((0, 2), np.float32), np.full(10, 0.3), np.zeros(10), 0, 10)
    mx.add_segment(_tone(), np.zeros(10), np.zeros(10), 0, 10)   # never on screen
    out = mx.finish(0.0)
    assert out.shape == (48000, 2) and not np.any(np.isnan(out))
    assert np.abs(out).max() == 0.0
    assert mx.segments == 2 and mx.heard == 1        # one had audio, one was mute


def test_mixer_segments_land_in_their_own_time_window():
    """Two turns of one cube must not bleed into each other's stretch."""
    from cube_tv import Mixer
    mx = Mixer(2.0, 10, 48000, pan_width=0.0)
    mx.add_segment(_tone(), np.full(20, 0.5), np.zeros(20), 0, 10)    # first second only
    out = mx.finish(0.0)
    assert np.abs(out[:48000]).max() > 0.5
    assert np.abs(out[48000:]).max() == 0.0


# --- the running order ----------------------------------------------------

def test_deck_deals_every_clip_once_before_repeating():
    from cube_tv import Deck
    deck = Deck(12, seed=3)
    first = [deck.draw() for _ in range(12)]
    assert sorted(first) == list(range(12))     # all twelve, none twice
    assert deck.laps == 0
    deck.draw()
    assert deck.laps == 1                       # only now does it wrap
    assert Deck(12, seed=3).order.tolist() == deck.order.tolist()   # seeded


def test_audio_library_evicts_but_still_serves():
    from core.video.clip_audio import AudioLibrary
    lib = AudioLibrary([f"missing_{i}.mp4" for i in range(20)], 48000, cache_size=4)
    for i in range(20):
        assert lib.get(i).shape == (0, 2)       # unreadable -> silence, not a crash
    assert len(lib._cache) <= 4                 # memory stays flat while streaming
    lib.release(19)
    assert 19 not in lib._cache

"""footage: one clip played on cue (lip sync), and the cues found from a stem cut from its sound."""
import json

import cv2
import numpy as np
import pytest

from core.video.footage import Footage, read_cues
from core.video.layers import build_compositor

FPS = 24


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """48 frames, 64×32: frame i is grey 5·i, so a frame tells its own index."""
    path = tmp_path_factory.mktemp("footage") / "clip.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, (64, 32))
    for i in range(48):
        out.write(np.full((32, 64, 3), 5 * i, np.uint8))
    out.release()
    return str(path)


def _index(frame) -> int:
    return int(round(float(np.median(frame)) / 5))


def test_cues_say_which_frame_plays_when(clip):
    f = Footage({"file": clip, "cues": [[2.0, 1.0, 0.5, 0], [1.0, 0.5, 0.5]], "hold": 0.25}, 64, 32)
    assert f.clip_time(0.9) is None                                      # before the first cue
    assert f.clip_time(1.25) == pytest.approx(0.75)
    assert f.clip_time(2.3) == pytest.approx(1.0)                        # rate 0: the frame held
    assert f.clip_time(2.6) == pytest.approx(1.0 - 0.5 / FPS)            # the hold after the cue
    assert f.clip_time(2.8) is None
    below = np.zeros((32, 64, 3), np.uint8)
    assert _index(f.frame_at_over(below, 1.25)) == 18                    # 0.75 s × 24
    assert f.frame_at_over(below, 3.0) is None


def test_contain_places_it_and_leaves_the_rest_alone(clip):
    f = Footage({"file": clip, "cues": [[0.0, 1.0, 1.0]]}, 100, 100)
    below = np.full((100, 100, 3), 7, np.uint8)
    out = f.frame_at_over(below, 0.0)
    assert (out[:25] == 7).all() and (out[75:] == 7).all()               # 100×50, centred
    assert _index(out[25:75]) == 24


def _mask_track(folder, frames=48):
    (folder / "labels").mkdir(parents=True)
    lab = np.zeros((32, 64), np.uint16)
    lab[:, :32], lab[:, 32:] = 1, 2                                      # left: 1, right: 2
    for i in range(frames):
        cv2.imwrite(str(folder / "labels" / f"{i:06d}.png"), lab)
    np.save(folder / "stats.npy", np.zeros((frames, 3, 5), np.float32))
    objects = [{"id": k, "shot": 0, "first": 0, "last": frames - 1} for k in (1, 2)]
    (folder / "track.json").write_text(json.dumps({
        "format": 1, "fps": FPS, "frames": frames, "start_time": 0.0, "width": 64, "height": 32,
        "objects": objects, "shots": [0]}), encoding="utf-8")
    return str(folder)


def test_cutout_keeps_only_the_clips_objects(clip, tmp_path):
    track = _mask_track(tmp_path / "seg")
    f = Footage({"file": clip, "cues": [[0.0, 1.0, 1.0]],
                 "cutout": {"track": track, "objects": [2], "feather": 0}}, 64, 32)
    out = f.frame_at_over(np.zeros((32, 64, 3), np.uint8), 0.0)
    assert (out[:, :30] == 0).all()                                      # object 1: the picture below
    assert _index(out[:, 34:]) == 24


class _Grey:
    def frame_at(self, t):
        return np.full((32, 64, 3), 200, np.uint8)


def test_a_scene_layer(clip):
    comp = build_compositor(_Grey(), {"layers": [{"source": "clips"},
                                                 {"source": "footage", "file": clip, "cues": [[1.0, 0.0, 1.0]]}]},
                            [], 64, 32, fps=FPS)
    comp.use_gpu = False
    assert (comp.frame_at(0.5) == 200).all()                             # no cue: the clips
    assert _index(comp.frame_at(1.5)) == 12


def test_bad_cues_are_named():
    with pytest.raises(ValueError, match="song s, clip s, seconds"):
        read_cues([[1.0, 2.0]])
    with pytest.raises(ValueError, match="needs cues"):
        read_cues([])


def _tones(seconds, sr, seed):
    """A melody of harmonic notes, 0.15 s each, every one a new pitch and timbre."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(int(seconds / 0.15)):
        f0 = 220 * 2 ** (rng.integers(0, 12) / 12)
        t = np.arange(int(0.15 * sr)) / sr
        amps = rng.uniform(0.2, 1.0, 6)
        out.append(sum(a * np.sin(2 * np.pi * f0 * (h + 1) * t) for h, a in enumerate(amps)) * 0.1)
    return np.concatenate(out).astype(np.float32)


def test_a_stem_chopped_from_the_clip_gives_its_cues_back(tmp_path):
    import librosa
    import soundfile as sf
    sr = 16000
    clip = _tones(6.0, sr, seed=3)
    up = librosa.effects.pitch_shift(clip, sr=sr, n_steps=1.0)          # the stem: a semitone up
    stem = np.zeros(10 * sr, np.float32)
    stem[2 * sr:int(3.5 * sr)] = up[1 * sr:int(2.5 * sr)]               # clip 1.0-2.5 at 2.0
    stem[5 * sr:int(6.2 * sr)] = up[3 * sr:int(4.2 * sr)]               # clip 3.0-4.2 at 5.0
    sf.write(tmp_path / "clip.wav", clip, sr)
    sf.write(tmp_path / "stem.wav", stem, sr)
    from core.stem_match import find_cues
    r = find_cues(str(tmp_path / "stem.wav"), str(tmp_path / "clip.wav"))
    assert r["semitones"] == pytest.approx(1.0)
    assert len(r["cues"]) == 2
    for got, want in zip(r["cues"], ([2.0, 1.0, 1.5], [5.0, 3.0, 1.2])):
        assert got == pytest.approx(want, abs=0.05)
    assert r["uncovered"] == []


def test_a_loop_a_still_and_a_place_that_moves(clip, tmp_path):
    f = Footage({"file": clip, "loop": True}, 64, 32)                   # no cues: the whole song
    assert f.clip_time(2.5) == pytest.approx(0.5)                       # 48 frames = 2 s, then again
    still = tmp_path / "still.png"
    cv2.imwrite(str(still), np.full((32, 64, 3), 90, np.uint8))
    s = Footage({"file": str(still), "cues": [[0.0, 0.0, 9.0]],
                 "x": {"lfo": "saw", "period": 1.0, "gain": 0.5, "bias": 0.25}, "scale": 0.5}, 64, 32)
    below = np.zeros((32, 64, 3), np.uint8)
    left = s.frame_at_over(below, 0.0)                                  # centre at x 0.25: 16 px
    assert (left[8:24, 0:32] == 90).all() and (left[:, 33:] == 0).all()
    right = s.frame_at_over(below, 0.999)                               # ... at about x 0.75
    assert (right[8:24, 33:63] == 90).all() and (right[:, :30] == 0).all()
    gone = Footage({"file": str(still), "scale": 0.0}, 64, 32)
    assert gone.frame_at_over(below, 1.0) is None                       # scale 0: nothing


def test_drop_keeps_all_but_the_backdrop(clip, tmp_path):
    track = _mask_track(tmp_path / "seg")
    f = Footage({"file": clip, "cues": [[0.0, 1.0, 1.0]],
                 "cutout": {"track": track, "drop": [1], "feather": 0}}, 64, 32)
    out = f.frame_at_over(np.zeros((32, 64, 3), np.uint8), 0.0)
    assert (out[:, :30] == 0).all() and _index(out[:, 34:]) == 24


def test_blocks_arrange_clips_by_bars(clip, tmp_path):
    from core.rhythm.grid import RhythmGrid
    grid = RhythmGrid(bpm=120.0, fps=30, beats=np.arange(32) * 0.5, downbeats=np.arange(8) * 2.0)
    other = tmp_path / "other.png"
    cv2.imwrite(str(other), np.full((32, 64, 3), 200, np.uint8))
    f = Footage({"blocks": [{"bars": [2, 2], "file": clip, "from": 0.5},
                            {"bars": [3, 4], "file": str(other)}]}, 64, 32, grid=grid)
    below = np.zeros((32, 64, 3), np.uint8)
    assert f.frame_at_over(below, 1.0) is None                          # bar 1: nothing
    assert _index(f.frame_at_over(below, 2.5)) == 24                    # bar 2, 0.5 s in, from 0.5 s
    assert int(np.median(f.frame_at_over(below, 7.9))) == 200           # bars 3-4: the other file
    assert f.frame_at_over(below, 8.1) is None
    with pytest.raises(ValueError, match="needs file"):
        Footage({"blocks": [{"bars": [1, 1]}]}, 64, 32, grid=grid)


def test_an_empty_background_lane_shows_nothing():
    f = Footage({"role": "background", "blocks": []}, 64, 32)          # made in Arrange, no block yet
    assert f.frame_at_over(np.zeros((32, 64, 3), np.uint8), 1.0) is None
    with pytest.raises(ValueError, match="needs file"):
        Footage({"source": "footage"}, 64, 32)


def test_a_tracks_own_fx_work_on_it_alone(tmp_path):
    from core.rhythm.grid import RhythmGrid
    still = tmp_path / "dot.png"
    cv2.imwrite(str(still), np.full((16, 16, 3), 200, np.uint8))
    grid = RhythmGrid(bpm=120.0, fps=30, beats=np.arange(32) * 0.5, downbeats=np.arange(8) * 2.0)
    def frame(**over):
        layers = [{"source": "solid", "color": [40, 40, 40]},
                  {"source": "footage", "file": str(still), "scale": 0.25, **over}]
        return build_compositor(None, {"layers": layers}, [], 64, 32, grid=grid).frame_at(1.0).astype(int)
    plain = frame()
    noisy = frame(fx=[{"source": "rgb_noise", "amount": 0.6, "seed": 2}])
    inside, outside = (slice(13, 19), slice(29, 35)), (slice(0, 6), slice(0, 6))
    assert (plain[outside] == 40).all() and (plain[inside] == 200).all()
    assert (noisy[outside] == 40).all()                              # the stack under it untouched
    assert np.abs(noisy[inside] - 200).mean() > 5                    # the dot got the noise
    gated = frame(fx=[{"source": "rgb_noise", "amount": 0.6, "seed": 2, "bars": [[3, 4]]}])
    assert (gated[inside] == 200).all()                              # bar 1: the FX not on yet


def test_spill_lets_what_an_fx_spreads_show():
    from core.video.track_fx import TrackFXOver, _Given

    class Dot:                                       # a placed layer: one pixel, alpha 1 there
        def frame_at_over(self, below, t): ...
        def frame_rgba(self, t):
            rgb, a = np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.float32)
            rgb[1, 1], a[1, 1] = 200, 1.0
            return rgb, a

    class Trail:                                     # an FX that spreads it to the next pixel
        def frame_at(self, t):
            f = given.frame.copy()
            f[1, 2] = 100
            return f
    given = _Given()
    below = np.full((4, 4, 3), 40, np.uint8)
    kept = TrackFXOver(Dot(), Trail(), given).frame_at_over(below, 0.0)
    spilled = TrackFXOver(Dot(), Trail(), given, spill=True).frame_at_over(below, 0.0)
    assert kept[1, 1, 0] == 200 and kept[1, 2, 0] == 40             # no spill: the trail goes
    assert spilled[1, 2, 0] == 100                                  # spill: the trail covers it
    assert (kept[3, 3] == 40).all() and (spilled[3, 3] == 40).all()


def test_lanes_bg_spliced_fg_on_its_own(tmp_path):
    still = tmp_path / "dot.png"
    cv2.imwrite(str(still), np.full((16, 16, 3), 200, np.uint8))
    noise = {"source": "rgb_noise", "amount": 0.6, "seed": 2}
    bg = [{"source": "solid", "color": [40, 40, 40]}]
    flat = build_compositor(None, {"layers": bg + [noise]}, [], 64, 32).frame_at(1.0)
    grouped = build_compositor(None, {"layers": [bg[0], {"source": "group", "name": "BG", "canvas": "below",
                                                         "layers": [], "fx": [noise]}]}, [], 64, 32).frame_at(1.0)
    assert (flat == grouped).all()                                    # the background lane: spliced in
    fg = {"source": "group", "name": "FG", "fx": [noise],
          "layers": [{"source": "footage", "file": str(still), "scale": 0.25, "x": 0.25},
                     {"source": "footage", "file": str(still), "scale": 0.25, "x": 0.75}]}
    out = build_compositor(None, {"layers": bg + [fg]}, [], 64, 32).frame_at(1.0).astype(int)
    left, right, between = (slice(13, 19), slice(13, 19)), (slice(13, 19), slice(45, 51)), (slice(0, 6), slice(29, 35))
    assert (out[between] == 40).all()                                 # the FG's FX leave the BG alone
    assert np.abs(out[left] - 200).mean() > 5 and np.abs(out[right] - 200).mean() > 5
    off = build_compositor(None, {"layers": bg + [{**fg, "enabled": False}]}, [], 64, 32).frame_at(1.0)
    assert (off == 40).all()


def test_drop_leaves_out_the_unnamed_edge_but_not_a_hole(clip, tmp_path):
    track = _mask_track(tmp_path / "seg")
    lab = np.zeros((32, 64), np.uint16)
    lab[:, :32], lab[:, 32:62] = 1, 2                    # backdrop, the figure; the last 2 columns: unnamed
    lab[14:18, 44:48] = 0                                # and a hole SAM missed inside the figure
    for i in range(48):
        cv2.imwrite(str(tmp_path / "seg" / "labels" / f"{i:06d}.png"), lab)
    f = Footage({"file": clip, "cues": [[0.0, 1.0, 1.0]],
                 "cutout": {"track": track, "drop": [1], "feather": 0}}, 64, 32)
    out = f.frame_at_over(np.zeros((32, 64, 3), np.uint8), 0.0)
    assert (out[:, 63] == 0).all()                       # the unnamed edge: backdrop, not drawn
    assert _index(out[15:17, 45:47]) == 24               # the hole stays with the figure

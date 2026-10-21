"""sam_cutout: a clip's SAM2 cutout as an FX, made once at the render and kept; seeds of
one's own (boxes, points) for SAM2."""
import json

import cv2
import numpy as np
import pytest

from core.segment.mask_track import seeds_in_pixels, write_mask_track
from core.video import sam_cutout
from core.video.layers import build_compositor


@pytest.fixture
def masks(tmp_path, monkeypatch):
    """render_output/footage, here."""
    root = tmp_path / "footage"
    monkeypatch.setattr(sam_cutout, "MASKS", root)
    monkeypatch.setattr(sam_cutout, "status", lambda msg: None)
    return root


def _track(folder, frames=48, shares=(0.2, 0.3)):
    """A mask track: object 1 the frame's left half, 2 its right half."""
    (folder / "labels").mkdir(parents=True)
    lab = np.zeros((32, 64), np.uint16)
    lab[:, :32], lab[:, 32:] = 1, 2
    for i in range(frames):
        cv2.imwrite(str(folder / "labels" / f"{i:06d}.png"), lab)
    np.save(folder / "stats.npy", np.zeros((frames, 3, 5), np.float32))
    objects = [{"id": k, "shot": 0, "first": 0, "last": frames - 1, "share": s} for k, s in zip((1, 2), shares)]
    (folder / "track.json").write_text(json.dumps({
        "format": 1, "fps": 24, "frames": frames, "start_time": 0.0, "width": 64, "height": 32,
        "objects": objects, "shots": [0]}), encoding="utf-8")
    return folder


def test_where_the_mask_track_goes(masks, tmp_path):
    clip = tmp_path / "lion.mp4"
    clip.write_bytes(b"x")
    auto = sam_cutout.config({"source": "sam_cutout"})
    assert auto == {**sam_cutout.DEFAULT}
    assert sam_cutout.mask_folder(str(clip), auto) == masks / "lion_segments"     # the default way's own
    seeded = sam_cutout.config({"seeds": {"at": 1.0, "objects": [{"box": [0.1, 0.1, 0.5, 0.9]}]}})
    other = sam_cutout.config({"seeds": {"at": 1.0, "objects": [{"box": [0.2, 0.1, 0.5, 0.9]}]}})
    a, b = sam_cutout.mask_folder(str(clip), seeded), sam_cutout.mask_folder(str(clip), other)
    assert a != b and a.name.startswith("lion_sam_") and a.name.endswith("_segments")
    assert sam_cutout.mask_folder(str(clip), {**auto, "max_objects": 8}).name.startswith("lion_sam_")
    assert sam_cutout.cached(str(clip), auto) is None
    _track(masks / "dancing" / "lion_segments")                                   # made before, anywhere under it
    assert sam_cutout.cached(str(clip), auto) == masks / "dancing" / "lion_segments"


def test_sam2_runs_once_then_the_cache_is_read(masks, tmp_path, monkeypatch):
    clip = tmp_path / "lion.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(sam_cutout, "weights", lambda: "W")
    runs = []

    def run(cmd, progress):
        runs.append(cmd)
        progress(1, 2)
        out = cmd[cmd.index("--output") + 1]
        if "--seeds" in cmd:
            seeds = json.loads(open(cmd[cmd.index("--seeds") + 1], encoding="utf-8").read())
            assert seeds["objects"][0]["box"] == [0.1, 0.1, 0.5, 0.9]
        from pathlib import Path
        _track(Path(out))
        Path(out + "_preview.mp4").write_bytes(b"p")
        return 0, []
    monkeypatch.setattr(sam_cutout, "_run", run)
    cfg = sam_cutout.config({})
    got = sam_cutout.ensure(str(clip), cfg)
    assert got == masks / "lion_segments" and (got / "track.json").is_file()
    assert (masks / "lion_segments_preview.mp4").is_file() and not (masks / "lion_segments.partial").exists()
    cmd = runs[0]
    assert cmd[cmd.index("--max-objects") + 1] == "4" and cmd[cmd.index("--start-time") + 1] == "0"
    assert sam_cutout.ensure(str(clip), cfg) == got and len(runs) == 1                 # the cache
    seeded = sam_cutout.config({"seeds": {"at": 2.0, "objects": [{"box": [0.1, 0.1, 0.5, 0.9]}]}})
    assert sam_cutout.ensure(str(clip), seeded).name.startswith("lion_sam_") and "--seeds" in runs[1]
    assert "--max-objects" not in runs[1]


def test_a_failed_sam2_run_says_why_and_leaves_nothing(masks, tmp_path, monkeypatch):
    clip = tmp_path / "lion.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(sam_cutout, "weights", lambda: "W")
    monkeypatch.setattr(sam_cutout, "_run", lambda cmd, progress: (1, ["Traceback ...",
                                                                       "torch.OutOfMemoryError: CUDA out of memory"]))
    with pytest.raises(RuntimeError, match="could not cut lion.mp4 out: .*CUDA out of memory"):
        sam_cutout.ensure(str(clip), sam_cutout.config({}))
    assert not list(masks.glob("*")) or not any(p.name.endswith(".partial") for p in masks.iterdir())
    monkeypatch.setattr(sam_cutout, "weights", lambda: None)
    with pytest.raises(RuntimeError, match="ARC_SAM2_WEIGHTS"):
        sam_cutout.ensure(str(clip), sam_cutout.config({}))


def test_the_fx_becomes_the_clips_cutout(tmp_path):
    track = _track(tmp_path / "t", shares=(0.6, 0.2))                 # object 1: a backdrop (60% of it)
    made = []

    def make(clip, cfg, need=None):
        made.append((clip, cfg["seeds"]))
        return track
    clip = {"source": "footage", "file": "lion.mp4", "fx": [{"source": "sam_cutout"}, {"source": "grain"}]}
    out = sam_cutout.resolve(clip, make)
    assert out["cutout"] == {"track": track.as_posix(), "feather": 2, "drop": [1]}   # as always: no backdrop
    assert out["fx"] == [{"source": "grain"}] and made == [("lion.mp4", "auto")]
    every = sam_cutout.resolve({**clip, "fx": [{"source": "sam_cutout", "keep": "all"}]}, make)
    assert every["cutout"] == {"track": track.as_posix(), "feather": 2} and "fx" not in every
    one = sam_cutout.resolve({**clip, "fx": [{"source": "sam_cutout", "drop": [2], "feather": 0}]}, make)
    assert one["cutout"]["drop"] == [2] and one["cutout"]["feather"] == 0
    off = sam_cutout.resolve({**clip, "fx": [{"source": "sam_cutout", "enabled": False}]}, make)
    assert "cutout" not in off and "fx" not in off
    lane = sam_cutout.resolve({"source": "group", "name": "FG", "fx": [{"source": "sam_cutout"}]}, make)
    assert "fx" not in lane and "cutout" not in lane                    # nothing to cut there
    seeded = sam_cutout.resolve({**clip, "fx": [{"source": "sam_cutout", "seeds": {"at": 0, "objects": [
        {"box": [0, 0, 1, 1]}]}}]}, make)
    assert seeded["cutout"] == {"track": track.as_posix(), "feather": 2}        # the seeded ones: all kept
    plain = {"source": "footage", "file": "x.mp4", "fx": [{"source": "grain"}]}
    assert sam_cutout.resolve(plain, make) is plain and len(made) == 4


def test_the_render_cuts_the_clip_out_by_its_fx(tmp_path, monkeypatch):
    clip = tmp_path / "clip.avi"
    out = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 24, (64, 32))
    for i in range(48):
        out.write(np.full((32, 64, 3), 5 * i, np.uint8))
    out.release()
    track = _track(tmp_path / "seg")
    monkeypatch.setattr(sam_cutout, "ensure", lambda c, cfg, need=None: track)

    class Grey:
        def frame_at(self, t):
            return np.full((32, 64, 3), 200, np.uint8)
    layers = [{"source": "clips"}, {"source": "footage", "file": str(clip), "cues": [[0.0, 1.0, 1.0]],
                                    "fx": [{"source": "sam_cutout", "drop": [1], "feather": 0}]}]
    comp = build_compositor(Grey(), {"layers": layers}, [], 64, 32, fps=24)
    comp.use_gpu = False
    frame = comp.frame_at(0.0)
    assert (frame[:, :30] == 200).all()                                 # object 1 left out: the clips
    assert abs(int(np.median(frame[:, 34:])) - 120) <= 5                 # object 2: the clip (frame 24)


def test_seeds_in_pixels():
    got = seeds_in_pixels({"at": 1.5, "objects": [{"box": [0.5, 0.8, 0.1, 0.2], "points": [[0.25, 0.5, 1]]},
                                                  {"points": [[0.5, 0.5, 0]]}, {}]}, 200, 100, 24)
    assert got == {"frame": 36, "objects": [{"box": [20.0, 20.0, 100.0, 80.0], "points": [[50.0, 50.0, 1]]},
                                            {"box": None, "points": [[100.0, 50.0, 0]]}]}


def test_seeded_objects_are_followed_from_their_frame(tmp_path):
    """With seeds: those objects, from the seed frame to the end; nothing before; no search."""
    frames = [np.full((8, 8, 3), i, np.uint8) for i in range(6)]

    class Segmenter:
        def __init__(self):
            self.added, self.live = [], {}

        def set_frame(self, i, f):
            self.i = i

        def add(self, oid, frames, mask=None, point=None, prompt=None):
            self.added.append((self.i, oid, frames, prompt))
            self.live[oid] = oid

        def step(self):
            out = {}
            for oid in self.live:
                m = np.zeros((8, 8), bool)
                m[:, (oid - 1) * 4:oid * 4] = True
                out[oid] = m
            return out

        def forget(self, oid):
            self.live.pop(oid, None)

        def auto_masks(self, *a, **k):
            raise AssertionError("no search with seeds")
    seg = Segmenter()
    prompts = [{"box": [0, 0, 3, 7], "points": []}, {"box": None, "points": [[6, 4, 1]]}]
    meta = write_mask_track("v.mp4", lambda: iter(frames), seg, str(tmp_path / "t"), 24.0,
                            seeds={"frame": 2, "objects": prompts}, log=lambda m: None)
    assert seg.added == [(2, 1, 4, prompts[0]), (2, 2, 4, prompts[1])]
    assert [o["found"] for o in meta["objects"]] == ["seed", "seed"] and meta["objects"][0]["share"] == 0.5
    first = cv2.imread(str(tmp_path / "t" / "labels" / "000001.png"), cv2.IMREAD_UNCHANGED)
    later = cv2.imread(str(tmp_path / "t" / "labels" / "000004.png"), cv2.IMREAD_UNCHANGED)
    assert not first.any() and set(np.unique(later)) == {1, 2}


def test_the_catalog_lists_it_with_the_figures():
    from core import catalog
    e = catalog.by_key("sam_cutout")
    assert e.kind == catalog.FX and e.category == "Figures"
    assert [p.name for p in e.params] == ["seeds", "max_objects", "granularity", "reseed", "keep", "feather"]


def test_sam2s_progress_is_told_and_a_stop_kills_it(monkeypatch):
    """segment_video.py's "  120/714 ..." lines reach the status; stopped() kills the run."""
    import sys
    script = ("import time\n"
              "print('SAM2 base_plus on x', flush=True)\n"
              "for i in range(1, 6):\n"
              "    print(f'  {i * 10:5d}/50  objects in view=2', flush=True)\n"
              "    time.sleep(0.05)\n"
              "time.sleep(30)\n")
    seen = []
    monkeypatch.setattr(sam_cutout, "stopped", lambda: len(seen) >= 3)
    with pytest.raises(sam_cutout.Stopped):
        sam_cutout._run([sys.executable, "-c", script], lambda done, total: seen.append((done, total)))
    assert seen == [(10, 50), (20, 50), (30, 50)]           # killed then, not 30 s later
    monkeypatch.setattr(sam_cutout, "stopped", lambda: False)
    code, tail = sam_cutout._run([sys.executable, "-c", "print('  3/3 x'); raise SystemExit(2)"], lambda d, t: None)
    assert code == 2 and tail == ["  3/3 x"]


def test_a_stopped_sam2_run_keeps_nothing(masks, tmp_path, monkeypatch):
    clip = tmp_path / "lion.mp4"
    clip.write_bytes(b"x")
    monkeypatch.setattr(sam_cutout, "weights", lambda: "W")
    said = []
    monkeypatch.setattr(sam_cutout, "status", said.append)

    def run(cmd, progress):
        from pathlib import Path
        (Path(cmd[cmd.index("--output") + 1]) / "labels").mkdir(parents=True)
        progress(7, 70)
        raise sam_cutout.Stopped()
    monkeypatch.setattr(sam_cutout, "_run", run)
    with pytest.raises(sam_cutout.Stopped):
        sam_cutout.ensure(str(clip), sam_cutout.config({}))
    assert not masks.exists() or not list(masks.iterdir())
    assert "7/70 frames" in said[1] and said[-1].startswith("SAM2: stopped before lion.mp4")


def _grid():
    from core.rhythm.grid import RhythmGrid
    return RhythmGrid(bpm=120.0, fps=30, beats=np.arange(64) * 0.5, downbeats=np.arange(16) * 2.0)   # 2 s bars


def test_the_frames_a_clip_shows_are_the_ones_cut_out():
    """Its blocks (from where each starts, for its bars, at its rate), its cues, inside its
    bars; a looping clip wraps round; parts less than a second apart go in one run."""
    g, need = _grid(), sam_cutout.frames_needed
    block = {"source": "footage", "file": "c.mp4", "blocks": [{"bars": [3, 4], "from": 10.0}]}
    assert need(block, g, 10.0, 1000) == [[98, 143]]                   # 10-14 s of the clip, 2 frames either side
    fast = {**block, "blocks": [{"bars": [3, 4], "from": 10.0, "rate": 2.0}]}
    assert need(fast, g, 10.0, 1000) == [[98, 183]]
    two = {**block, "blocks": [{"bars": [3, 3], "from": 10.0}, {"bars": [9, 9], "from": 50.0}]}
    assert need(two, g, 10.0, 1000) == [[98, 123], [498, 523]]
    near = {**block, "blocks": [{"bars": [3, 3], "from": 10.0}, {"bars": [9, 9], "from": 12.5}]}
    assert need(near, g, 10.0, 1000) == [[98, 148]]                     # 0.3 s apart: one run
    loop = {**block, "loop": True, "blocks": [{"bars": [3, 4], "from": 8.0}]}
    assert need(loop, g, 10.0, 100) == [[0, 23], [78, 100]]             # 8-10 s, then 0-2 s again
    assert need({**loop, "blocks": [{"bars": [1, 12], "from": 0.0}]}, g, 10.0, 100) == [[0, 100]]
    cued = {"source": "footage", "file": "c.mp4", "cues": [[20.0, 3.0, 1.0], [25.0, 30.0, 0.5, 2.0], [40.0, 50.0, 1.0]]}
    assert need(cued, g, 10.0, 1000) == [[28, 43], [298, 313]]                # (the third: after the song)
    windowed = {"source": "footage", "file": "c.mp4", "loop": True, "bars": [[2, 2]]}
    assert need(windowed, g, 10.0, 1000) == [[18, 43]]                  # the song's 2-4 s are the clip's
    other = {**block, "blocks": [{"bars": [3, 4], "from": 10.0, "file": "d.mp4"}]}
    assert need(other, g, 10.0, 1000) == []                             # none of it is this clip
    looped_from_0 = {**block, "loop": True, "blocks": [{"bars": [3, 4]}]}   # (as in the elenco: 1688 frames)
    assert need(looped_from_0, g, 30.0, 1688) == [[0, 123]]                 # the margin never wraps round
    assert need(block, g, 10.0, 1000, span=(4.0, 6.0)) == [[98, 123]]       # rendering bar 3: its half
    assert need(block, g, 10.0, 1000, span=(20.0, 24.0)) == []              # bars 11-12: it does not show
    assert need(block, None, 10.0, 1000) == [[0, 1000]]                 # no grid: all of it
    assert sam_cutout.missing([[0, 20]], [[5, 10], [15, 30]]) == [[0, 5], [10, 15]]
    assert sam_cutout.merged([[10, 20], [0, 5], [5, 8], [30, 31]]) == [[0, 8], [10, 20], [30, 31]]
    assert sam_cutout.merged([[10, 20], [0, 5], [5, 8], [30, 31]], gap=3) == [[0, 20], [30, 31]]


def _clip(tmp_path, frames=48):
    clip = tmp_path / "clip.avi"
    out = cv2.VideoWriter(str(clip), cv2.VideoWriter_fourcc(*"MJPG"), 24, (64, 32))
    for i in range(frames):
        out.write(np.full((32, 64, 3), 5 * i, np.uint8))
    out.release()
    return clip


def test_the_cache_grows_by_what_it_lacks(masks, tmp_path, monkeypatch):
    """A run makes only the frames asked for that the cache lacks, then adds them to it: new
    objects go on numbering; the frames never asked for have nothing; a seeded run follows
    from its seed to the last frame asked for."""
    from core.segment.mask_track import MaskTrack
    clip = _clip(tmp_path)
    monkeypatch.setattr(sam_cutout, "weights", lambda: "W")
    runs = []

    class Seg:                                            # SAM2, faked: one object, the left half
        def __init__(self): self.live = {}
        def set_frame(self, i, f): pass
        def auto_masks(self, *a): m = np.zeros((32, 64), bool); m[:, :32] = True; return [m]
        def add(self, oid, frames, mask=None, point=None, prompt=None): self.live[oid] = mask
        def step(self): return {oid: (m if m is not None else np.ones((32, 64), bool)) for oid, m in self.live.items()}
        def forget(self, oid): self.live.pop(oid, None)

    def run(cmd, progress):
        frames = [np.full((32, 64, 3), i, np.uint8) for i in range(48)]
        arg = lambda k, d=None: cmd[cmd.index(k) + 1] if k in cmd else d
        ranges = [tuple(int(x) for x in p.split(":")) for p in arg("--frames").split(",")] if arg("--frames") else None
        runs.append((ranges, int(arg("--first-id", 1)), "--seeds" in cmd))
        seeds = None
        if "--seeds" in cmd:
            from core.segment.mask_track import seeds_in_pixels
            seeds = seeds_in_pixels(json.loads(open(arg("--seeds"), encoding="utf-8").read()), 64, 32, 24.0)
        write_mask_track(str(clip), lambda: iter(frames), Seg(), arg("--output"), 24.0, cuts=[], log=lambda m: None,
                         ranges=ranges, frames_at=lambda a, c: iter(frames[a:a + c]), total=48,
                         first_id=int(arg("--first-id", 1)), seeds=seeds)
        return 0, []
    monkeypatch.setattr(sam_cutout, "_run", run)
    cfg = sam_cutout.config({})
    folder = sam_cutout.ensure(str(clip), cfg, [[0, 10]])
    assert runs == [([(0, 10)], 1, False)] and sam_cutout.done_of(folder) == [[0, 10]]
    assert sam_cutout.ensure(str(clip), cfg, [[2, 8]]) == folder and len(runs) == 1     # had them
    sam_cutout.ensure(str(clip), cfg, [[5, 20]])
    assert runs[1] == ([(10, 20)], 2, False)                                             # only what it lacked
    track = MaskTrack(str(folder))
    assert sam_cutout.done_of(folder) == [[0, 20]] and [o["id"] for o in track.meta["objects"]] == [1, 2]
    assert set(np.unique(track.labels(3))) == {0, 1} and set(np.unique(track.labels(15))) == {0, 2}
    assert not track.labels(30).any() and track.stats.shape[1] == 3                    # never asked for: nothing
    assert not list(masks.glob("*.partial")) and not list(masks.glob("*.lock"))
    seeded = sam_cutout.config({"seeds": {"at": 1.0, "objects": [{"box": [0, 0, 0.5, 1]}]}})
    sfolder = sam_cutout.ensure(str(clip), seeded, [[30, 40]])
    assert runs[2] == ([(24, 40)], 1, True) and sam_cutout.done_of(sfolder) == [[24, 40]]
    sam_cutout.ensure(str(clip), seeded, [[10, 12], [44, 46]])          # before the seed, and after: from it again
    assert runs[3] == ([(24, 46)], 1, True) and sam_cutout.done_of(sfolder) == [[10, 12], [24, 46]]
    assert [o["id"] for o in MaskTrack(str(sfolder)).meta["objects"]] == [1]            # the same object


def test_an_old_whole_track_counts_as_done_and_the_lock_is_shared(masks, tmp_path, monkeypatch):
    import os
    clip = _clip(tmp_path)
    old = _track(masks / "clip_segments")                 # made whole, before ranges: no done
    assert sam_cutout.done_of(old) == [[0, 48]]
    assert sam_cutout.ensure(str(clip), sam_cutout.config({}), [[40, 48]]) == old
    folder = masks / "x_segments"
    (masks / "x_segments.lock").write_text("999999")      # its owner is gone: taken over
    with sam_cutout._lock(folder):
        assert (masks / "x_segments.lock").read_text() == str(os.getpid())
    assert not (masks / "x_segments.lock").exists()
    (masks / "x_segments.lock").write_text(str(os.getpid()))   # a live owner: wait (Stop ends the wait)
    monkeypatch.setattr(sam_cutout, "stopped", lambda: True)
    with pytest.raises(sam_cutout.Stopped):
        with sam_cutout._lock(folder):
            pass



def test_a_clip_that_does_not_show_in_the_render_is_not_cut_out(masks, tmp_path):
    """Rendering bars 20-21: a clip in bars 3-4 is not cut out now (the cache's track if it
    has one, else none); one in them is, for those bars only."""
    clip = _clip(tmp_path)
    made = []

    def make(c, cfg, need=None):
        made.append(need)
        return _track(masks / "made")
    spec = {"source": "footage", "file": str(clip), "blocks": [{"bars": [3, 4]}], "fx": [{"source": "sam_cutout"}]}
    out = sam_cutout.resolve(spec, make, grid=_grid(), span=(20.0, 24.0))
    assert made == [] and "cutout" not in out and "fx" not in out
    cached = _track(masks / "clip_segments")
    out = sam_cutout.resolve(spec, make, grid=_grid(), span=(20.0, 24.0))
    assert made == [] and out["cutout"]["track"] == cached.as_posix()
    sam_cutout.resolve(spec, make, grid=_grid(), span=(4.0, 6.0))
    assert made == [[[0, 48]]]                                          # bar 3: its first 2 s (all its 48 frames at 24)

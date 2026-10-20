"""sam_cutout: a clip's SAM2 cutout as an FX, made once at the render and kept; seeds of
one's own (boxes, points) for SAM2."""
import json
import subprocess

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

    def run(cmd, **kw):
        runs.append(cmd)
        out = cmd[cmd.index("--output") + 1]
        if "--seeds" in cmd:
            seeds = json.loads(open(cmd[cmd.index("--seeds") + 1], encoding="utf-8").read())
            assert seeds["objects"][0]["box"] == [0.1, 0.1, 0.5, 0.9]
        from pathlib import Path
        _track(Path(out))
        Path(out + "_preview.mp4").write_bytes(b"p")
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(sam_cutout.subprocess, "run", run)
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
    monkeypatch.setattr(sam_cutout.subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 1, "", "Traceback ...\ntorch.OutOfMemoryError: CUDA out of memory"))
    with pytest.raises(RuntimeError, match="could not cut lion.mp4 out: .*CUDA out of memory"):
        sam_cutout.ensure(str(clip), sam_cutout.config({}))
    assert not list(masks.glob("*")) or not any(p.name.endswith(".partial") for p in masks.iterdir())
    monkeypatch.setattr(sam_cutout, "weights", lambda: None)
    with pytest.raises(RuntimeError, match="ARC_SAM2_WEIGHTS"):
        sam_cutout.ensure(str(clip), sam_cutout.config({}))


def test_the_fx_becomes_the_clips_cutout(tmp_path):
    track = _track(tmp_path / "t", shares=(0.6, 0.2))                 # object 1: a backdrop (60% of it)
    made = []

    def make(clip, cfg):
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
    monkeypatch.setattr(sam_cutout, "ensure", lambda c, cfg: track)

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

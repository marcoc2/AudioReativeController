"""sam_cutout: an FX on a clip that cuts its characters out with SAM2, at the render.

The first render (or preview) that needs it runs SAM2 on the clip (segment_video.py, on
the GPU) and keeps the mask track in ``render_output/footage``; every render after reads
it from there, as a ``cutout:`` written by hand would be read. Only the frames the layer
shows are cut out (its blocks, its cues: two bars of a long clip are two bars' worth of
work), and the cache grows as more of the clip comes into use: a run makes only what it
lacks. A clip SAM2 already cut out the default way (its ``<clip>_segments`` folder) is
not cut again.

    - source: footage
      file: input/esfolado/video4.mp4
      fx:
        - source: sam_cutout
          seeds: auto              # auto: SAM2 finds the objects by itself; or seeds of one's own:
                                   # {at: 3.2, objects: [{box: [x0, y0, x1, y1], points: [[x, y, 1]]}]}
                                   # (at: the clip's second they are on; 0..1 of the frame; a point's
                                   # 1 keeps what is there, 0 leaves it out)
          max_objects: 4           # auto: the most objects followed at once
          granularity: whole       # auto: whole (characters, things) or parts (eyes, hands...)
          reseed: 2.0              # auto: looks for new objects every N s (0: only at cuts)
          keep: figures            # figures: all but a backdrop SAM2 took for an object (one
                                   # over 45% of the frame; auto only: seeded, every one is
                                   # kept); all: every object; or drop: [1]
          feather: 2               # its edge, px of softness

``resolve(spec)`` turns the FX into the clip's ``cutout:`` (making the mask track when it
is not there yet); ``status`` (a callable) hears what goes on (the studio shows it).
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional

SOURCE = "sam_cutout"
ROOT = Path(__file__).resolve().parents[2]
MASKS = Path("render_output") / "footage"
DEFAULT = {"seeds": "auto", "max_objects": 4, "granularity": "whole", "reseed": 2.0, "keep": "figures",
           "feather": 2}
WEIGHTS = ("F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/models/sam2",)   # where SAM2 is kept here

status: Callable[[str], None] = lambda msg: print(msg, flush=True)
stopped: Callable[[], bool] = lambda: False      # True: the run is not wanted any more (a preview's Stop)


class Stopped(RuntimeError):
    """SAM2 was stopped before it was done (nothing kept)."""


class SamCutout:
    """Cuts the clip's characters out with SAM2 (made once at the render, then cached).

        - source: sam_cutout
          seeds: auto              # auto: SAM2 finds the objects; or boxes and points of one's own
          max_objects: 4           # auto: the most objects followed at once
          granularity: whole       # auto: whole or parts
          reseed: 2.0              # auto: new objects looked for every N s
          keep: figures            # figures: all but a backdrop; all: every object
          feather: 2               # its edge, px of softness

    It works on its clip (``resolve``): anywhere else it has nothing to cut and leaves the
    picture as it is."""

    def __init__(self, spec: Optional[dict] = None, *args, **kwargs):
        self.spec = spec or {}

    def process(self, frame, t: float):
        return frame


def config(fx: dict) -> dict:
    """The FX's settings, the defaults filled in."""
    out = {**DEFAULT, **{k: v for k, v in fx.items() if k != "source"}}
    if not isinstance(out.get("seeds"), dict):
        out["seeds"] = "auto"
    return out


def weights() -> Optional[str]:
    return os.environ.get("ARC_SAM2_WEIGHTS") or next((w for w in WEIGHTS if Path(w).is_dir()), None)


def _legacy(clip: str) -> Optional[Path]:
    """The mask track SAM2 made of ``clip`` the default way (``<clip stem>_segments``)."""
    from core.arrangement import find_mask_track
    found = find_mask_track(clip, MASKS)
    return Path(found) if found else None


def _key(clip: str, cfg: dict) -> str:
    p = Path(clip)
    try:
        st = p.stat()
        stamp = f"{p.resolve()}|{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        stamp = str(p)
    what = {k: cfg[k] for k in ("seeds", "max_objects", "granularity", "reseed")} if cfg["seeds"] == "auto" \
        else {"seeds": cfg["seeds"]}
    return hashlib.sha1((stamp + json.dumps(what, sort_keys=True)).encode()).hexdigest()[:10]


def is_default(cfg: dict) -> bool:
    """SAM2 left to find the objects with the settings it has always had here."""
    return all(cfg[k] == DEFAULT[k] for k in ("seeds", "max_objects", "granularity", "reseed"))


def mask_folder(clip: str, cfg: dict) -> Path:
    """Where ``clip``'s mask track for these settings is (or goes): the default way's own
    ``<clip>_segments`` when it was made already; else one per settings."""
    if is_default(cfg):
        legacy = _legacy(clip)
        if legacy is not None:
            return legacy
        return MASKS / f"{Path(clip).stem}_segments"
    return MASKS / f"{Path(clip).stem}_sam_{_key(clip, cfg)}_segments"


def cached(clip: str, cfg: dict) -> Optional[Path]:
    """The folder of ``clip``'s mask track for these settings, when there is one (it may hold
    only the frames asked for so far: ``done_of``)."""
    folder = mask_folder(clip, cfg)
    return folder if (folder / "track.json").is_file() else None


_FRAMES: dict = {}


def clip_frames(clip: str) -> tuple:
    """(fps, how many frames) of ``clip``, as OpenCV reads it (the footage layer does)."""
    import cv2
    p = Path(clip)
    try:
        st = p.stat()
        key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    except OSError:
        key = (str(p), 0, 0)
    if key not in _FRAMES:
        cap = cv2.VideoCapture(str(clip))
        _FRAMES[key] = (float(cap.get(cv2.CAP_PROP_FPS) or 24.0), int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
        cap.release()
    return _FRAMES[key]


def merged(ranges, gap: int = 0) -> list:
    """[[first, end)] sorted, those that touch (or are less than ``gap`` frames apart) joined."""
    out = []
    for a, b in sorted([int(a), int(b)] for a, b in ranges if int(b) > int(a)):
        if out and a <= out[-1][1] + gap:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def missing(need, done) -> list:
    """The frames of ``need`` that ``done`` lacks, as ranges."""
    out = []
    for a, b in merged(need):
        for c, d in merged(done):
            if d <= a or c >= b:
                continue
            if c > a:
                out.append([a, c])
            a = max(a, d)
            if a >= b:
                break
        if a < b:
            out.append([a, b])
    return out


GAP = 1.0              # s: parts of a clip nearer than this are cut out in one run
PAD = 2                # frames either side of what plays


def frames_needed(spec: dict, grid, fps: float, n: int, span=None) -> list:
    """[[first, end)] of the clip's frames the layer shows: its blocks (from where each starts
    in the clip, for as long as its bars last, at its rate), its cues, inside its ``bars:``
    and inside ``span`` (the song's seconds being rendered; None: all the song); a looping
    clip wraps round (all of it, when it plays longer than it lasts). Without a grid, or with
    nothing that says when: all of it. [] : none of it shows."""
    import math
    from core.video.footage import read_blocks, read_cues
    whole = [[0, n]]
    if n <= 0:
        return whole
    file = Path(str(spec.get("file") or "")).name
    cues = []
    if spec.get("cues") is not None:
        cues += [c[:4] for c in read_cues(spec["cues"], spec.get("file"))]
    if spec.get("blocks") is not None:
        if grid is None:
            return whole
        cues += [c[:4] for c in read_blocks(spec["blocks"], grid) if not c[4] or Path(c[4]).name == file]
    if spec.get("cues") is None and spec.get("blocks") is None:
        cues = [(0.0, 0.0, 1e9, 1.0)]                    # the song's time is the clip's
    windows = [(0.0, math.inf)]
    if spec.get("bars"):
        if grid is None:
            return whole
        bars = spec["bars"]
        bars = [bars] if not isinstance(bars[0], (list, tuple)) else bars
        windows = [(float(grid.bar_start(int(a) - 1)), float(grid.bar_start(int(b)))) for a, b in bars]
    end = math.inf
    db = getattr(grid, "downbeats", None) if grid is not None else None
    if db is not None and len(db):
        end = float(grid.bar_start(len(db)))
    if span is not None:                                 # only what the render shows
        windows = [(max(w0, float(span[0])), min(w1, float(span[1]))) for w0, w1 in windows]
    hold, loop = float(spec.get("hold", 0.0)), bool(spec.get("loop", False))
    out = []
    for at, start, length, rate in cues:
        for w0, w1 in windows:
            s0, s1 = max(at, w0, 0.0), min(at + length + hold, w1, end)
            if s1 <= s0:
                continue
            c0 = start + (s0 - at) * rate
            c1 = start + (min(s1, at + length) - at) * rate   # a hold stays on the last frame
            f0, f1 = int(math.floor(min(c0, c1) * fps)), int(math.ceil(max(c0, c1) * fps)) + 1
            if loop:
                if f1 - f0 >= n:
                    return whole
                f0, f1 = f0 % n, f0 % n + (f1 - f0)
                pieces = [[f0, min(f1, n)]] + ([[0, f1 - n]] if f1 > n else [])
            else:
                pieces = [[max(0, min(f0, n - 1)), max(1, min(f1, n))]]
            out += [[max(0, a - PAD), min(n, b + PAD)] for a, b in pieces]   # (never round the clip's ends)
    return merged(out, gap=int(round(GAP * fps)))


def done_of(folder: Path) -> list:
    """The frames a mask track was made for (an old one, made whole: all of them)."""
    try:
        meta = json.loads((Path(folder) / "track.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return merged(meta["done"]) if meta.get("done") is not None else [[0, int(meta.get("frames", 0))]]


def _top_id(folder: Path) -> int:
    try:
        meta = json.loads((Path(folder) / "track.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    return max([int(o["id"]) for o in meta.get("objects", [])], default=0)


def command(clip: str, cfg: dict, out: Path, seeds_file: Optional[Path], weights_dir: str,
            ranges=None, first_id: int = 1) -> list:
    cmd = [sys.executable, str(ROOT / "segment_video.py"), "--video", str(clip), "--output", str(out),
           "--start-time", "0", "--sam2-weights", weights_dir, "--cuts", "detect", "--preview"]
    if seeds_file is not None:
        cmd += ["--seeds", str(seeds_file)]
    else:
        cmd += ["--max-objects", str(int(cfg["max_objects"])), "--granularity", str(cfg["granularity"]),
                "--reseed", str(float(cfg["reseed"]))]
    if ranges:
        cmd += ["--frames", ",".join(f"{a}:{b}" for a, b in ranges)]
    if first_id != 1:
        cmd += ["--first-id", str(int(first_id))]
    return cmd


def ensure(clip: str, cfg: dict, need=None) -> Path:
    """``clip``'s mask track for these settings, holding at least the frames ``need``
    ([[first, end)]; None: all of them). What the cache lacks of them is made now by SAM2,
    only that (the clip is read from there: the rest is not even decoded), and added to it:
    the cache grows as more of the clip is used. A run goes into a ``.partial`` folder first
    (one that fails or is cut short is never taken for a track); two runs on one clip wait
    for each other. Seeds are followed forward only: SAM2 runs from their frame to the end
    of the last frames asked for (frames before it have nothing to cut)."""
    if not Path(clip).is_file():
        raise RuntimeError(f"sam_cutout: no clip {clip}")
    fps, n = clip_frames(clip)
    folder = mask_folder(clip, cfg)
    need = (merged(need) if need else [[0, n]]) if n > 0 else None   # frames not known: all of it
    if (need is None and (folder / "track.json").is_file()) or (need and not missing(need, done_of(folder))):
        return folder
    weights_dir = weights()
    if not weights_dir:
        raise RuntimeError("sam_cutout: SAM2's weights were not found (set ARC_SAM2_WEIGHTS)")
    with _lock(folder):
        seeded = isinstance(cfg["seeds"], dict)
        if need is None:
            if (folder / "track.json").is_file():
                return folder
            _make(clip, cfg, folder, None, None, 1, weights_dir, seeded, fps, n)
            return folder
        lacking = missing(need, done_of(folder))           # (another run may have made them meanwhile)
        if not lacking:
            return folder
        if seeded:
            first = max(0, min(n - 1, int(round(float(cfg["seeds"].get("at", 0.0)) * fps))))
            run = [[first, max(first + 1, max(b for _, b in lacking))]]
            done_now, first_id = merged(lacking + run), 1   # before the seed: nothing to cut, counted done
        else:
            run, done_now, first_id = lacking, lacking, _top_id(folder) + 1
        _make(clip, cfg, folder, run, done_now, first_id, weights_dir, seeded, fps, n)
    return folder


def _make(clip, cfg, folder: Path, run, done_now, first_id, weights_dir, seeded, fps, n) -> None:
    tmp = folder.with_name(folder.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    seeds_file = None
    if seeded:
        seeds_file = tmp.with_name(tmp.name + "_seeds.json")
        seeds_file.write_text(json.dumps(cfg["seeds"]), encoding="utf-8")
    name = Path(clip).name
    todo = sum(b - a for a, b in run) if run else n
    status(f"SAM2: cutting {name} out, {todo} of its {n} frames (once: the renders after read them from "
           f"{folder})...")
    try:
        code, tail = _run(command(clip, cfg, tmp, seeds_file, weights_dir,
                                  None if not run or run == [[0, n]] else run, first_id),
                          lambda done, total: status(f"SAM2: cutting {name} out: {done}/{total} frames "
                                                     f"(once: then they are kept)"))
    except Stopped:
        shutil.rmtree(tmp, ignore_errors=True)
        status(f"SAM2: stopped before {name} was cut out (nothing kept)")
        raise
    finally:
        if seeds_file is not None:
            seeds_file.unlink(missing_ok=True)
    if code != 0 or not (tmp / "track.json").is_file():
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"SAM2 could not cut {name} out: " + " | ".join(tail[-4:]))
    grow(folder, tmp, done_now)
    status(f"SAM2: {name} cut out ({todo} frames) -> {folder}")


def grow(folder: Path, part: Path, done_now) -> None:
    """A run's mask track (``part``) added to the cache's (``folder``): its frames' labels
    over the ones there, the stats and objects joined (an object of the same id, a seeded
    one followed again: one), ``done_now`` added to what it was made for. The track's
    ``track.json`` is written last: until then the cache is the one it was."""
    import numpy as np
    folder, part = Path(folder), Path(part)
    new = json.loads((part / "track.json").read_text(encoding="utf-8"))
    if done_now is None:                                         # (all of it, as the run says)
        done_now = new.get("done") or [[0, int(new.get("frames", 0))]]
    preview = Path(str(part) + "_preview.mp4")
    if not (folder / "track.json").is_file():
        shutil.rmtree(folder, ignore_errors=True)                # a run that failed before left it empty
        new["done"] = merged(done_now)
        (part / "track.json").write_text(json.dumps(new, indent=1), encoding="utf-8")
        part.rename(folder)
    else:
        old = json.loads((folder / "track.json").read_text(encoding="utf-8"))
        (folder / "labels").mkdir(exist_ok=True)
        for png in (part / "labels").glob("*.png"):
            png.replace(folder / "labels" / png.name)
        a_stats, b_stats = np.load(folder / "stats.npy"), np.load(part / "stats.npy")
        frames = max(a_stats.shape[0], b_stats.shape[0])
        stats = np.zeros((frames, max(a_stats.shape[1], b_stats.shape[1]), a_stats.shape[2]), np.float32)
        stats[:a_stats.shape[0], :a_stats.shape[1]] = a_stats
        for a, b in new.get("done") or [[0, b_stats.shape[0]]]:
            stats[a:b] = 0
            stats[a:b, :b_stats.shape[1]] = b_stats[a:b]
        np.save(folder / "stats.npy", stats)
        objects = {int(o["id"]): o for o in old.get("objects", [])}
        for o in new.get("objects", []):
            had = objects.get(int(o["id"]))
            if had is not None:                                  # the same (seeded) object, followed again
                o = {**had, "first": min(had["first"], o["first"]), "last": max(had["last"], o["last"]),
                     "share": o.get("share") or had.get("share", 0.0)}
            objects[int(o["id"])] = o
        old.update(frames=max(int(old.get("frames", 0)), int(new.get("frames", 0))),
                   width=old.get("width") or new.get("width"), height=old.get("height") or new.get("height"),
                   objects=[objects[k] for k in sorted(objects)],
                   shots=sorted(old.get("shots", []) + new.get("shots", []), key=lambda s: s["first"]),
                   done=merged((old.get("done") if old.get("done") is not None
                                else [[0, int(old.get("frames", 0))]]) + list(done_now)))
        tmp_json = folder / "track.json.new"
        tmp_json.write_text(json.dumps(old, indent=1), encoding="utf-8")
        tmp_json.replace(folder / "track.json")
        shutil.rmtree(part, ignore_errors=True)
    if preview.is_file():                                        # the objects as the last run found them
        preview.replace(Path(str(folder) + "_preview.mp4"))


def _alive(pid: int) -> bool:
    if sys.platform != "win32":
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    import ctypes
    k32 = ctypes.windll.kernel32
    h = k32.OpenProcess(0x1000, False, int(pid))                 # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    k32.GetExitCodeProcess(h, ctypes.byref(code))
    k32.CloseHandle(h)
    return code.value == 259                                     # STILL_ACTIVE


class _lock:
    """One run at a time on a clip's mask track (the preview's and a render's): a lock file
    beside it with the owner's pid; one left by a process that is gone is taken over."""

    def __init__(self, folder: Path):
        self.path = Path(folder).with_name(Path(folder).name + ".lock")

    def __enter__(self):
        import time
        self.path.parent.mkdir(parents=True, exist_ok=True)
        told = False
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, str(os.getpid()).encode())
                os.close(fd)
                return self
            except FileExistsError:
                try:
                    owner = int(self.path.read_text().strip() or 0)
                except (OSError, ValueError):
                    owner = 0
                if not owner or not _alive(owner):
                    self.path.unlink(missing_ok=True)
                    continue
                if not told:
                    status("SAM2: another run is cutting this clip out; waiting for it...")
                    told = True
                if stopped():
                    raise Stopped()
                time.sleep(0.5)

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def _run(cmd: list, progress: Callable[[int, int], None]) -> tuple:
    """segment_video.py run, its "  120/714 ..." lines told to ``progress``; ``stopped()``
    asked as it goes (True: it is killed, ``Stopped``). -> (exit code, its last lines)."""
    import re
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(ROOT),
                            encoding="utf-8", errors="replace", bufsize=1)
    tail, frames = [], re.compile(r"^\s*(\d+)/(\d+)\s")
    try:
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                tail = (tail + [line])[-20:]
            m = frames.match(line)
            if m:
                progress(int(m.group(1)), int(m.group(2)))
            if stopped():
                raise Stopped()
        return proc.wait(), tail
    except BaseException:
        proc.kill()
        proc.wait()
        raise


def _is_sam(fx) -> bool:
    return isinstance(fx, dict) and fx.get("source") == SOURCE and fx.get("enabled", True) is not False


def resolve(spec: dict, make: Optional[Callable[..., Path]] = None, grid=None, span=None) -> dict:
    """A layer with the sam_cutout FX in its ``fx:``: a footage layer gets its ``cutout:``
    (the mask track of the frames it shows in ``span``, the song's seconds being rendered,
    made if need be: ``frames_needed`` on the song's ``grid``) and loses the FX; one that
    does not show there is not cut out for this render (the cache's track when there is
    one). Anywhere else the FX goes (there is nothing to cut). Other layers come back as
    they were."""
    if not isinstance(spec, dict) or not any(isinstance(f, dict) and f.get("source") == SOURCE
                                             for f in spec.get("fx") or []):
        return spec
    out = copy.copy(spec)
    sams = [f for f in spec["fx"] if _is_sam(f)]
    out["fx"] = [f for f in spec["fx"] if not (isinstance(f, dict) and f.get("source") == SOURCE)]
    if not out["fx"]:
        del out["fx"]
    if not sams:                                          # switched off
        return out
    if spec.get("source") != "footage" or not spec.get("file"):
        status(f"sam_cutout: only a clip (a footage layer with its file) can be cut out; left out on "
               f"{spec.get('source')}")
        return out
    cfg = config(sams[0])
    clip = str(spec["file"])
    need = frames_needed(spec, grid, *clip_frames(clip), span=span) if Path(clip).is_file() else None
    if need == []:                                       # it does not show in this render
        track = cached(clip, cfg)
        if track is None:
            return out
    else:
        track = (make or ensure)(clip, cfg, need)
    cut = {"track": track.as_posix(), "feather": cfg.get("feather", 2)}
    for k in ("drop", "objects", "grow"):
        if cfg.get(k) is not None:
            cut[k] = cfg[k]
    if "drop" not in cut and "objects" not in cut and cfg.get("keep") == "figures" and cfg["seeds"] == "auto":
        from core.arrangement import backdrop_of           # as a character has always been cut out
        backdrop = backdrop_of(str(track))
        if backdrop:
            cut["drop"] = backdrop
    out["cutout"] = cut
    return out

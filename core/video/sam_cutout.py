"""sam_cutout: an FX on a clip that cuts its characters out with SAM2, at the render.

The first render (or preview) that needs it runs SAM2 on the clip (segment_video.py, on
the GPU) and keeps the mask track in ``render_output/footage``; every render after reads
it from there, as a ``cutout:`` written by hand would be read. A clip SAM2 already cut out
the default way (its ``<clip>_segments`` folder) is not cut again.

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
    folder = mask_folder(clip, cfg)
    return folder if (folder / "track.json").is_file() else None


def command(clip: str, cfg: dict, out: Path, seeds_file: Optional[Path], weights_dir: str) -> list:
    cmd = [sys.executable, str(ROOT / "segment_video.py"), "--video", str(clip), "--output", str(out),
           "--start-time", "0", "--sam2-weights", weights_dir, "--cuts", "detect", "--preview"]
    if seeds_file is not None:
        cmd += ["--seeds", str(seeds_file)]
    else:
        cmd += ["--max-objects", str(int(cfg["max_objects"])), "--granularity", str(cfg["granularity"]),
                "--reseed", str(float(cfg["reseed"]))]
    return cmd


def ensure(clip: str, cfg: dict) -> Path:
    """``clip``'s mask track for these settings: the cached one, or made now by SAM2 (once:
    into a ``.partial`` folder first, so a failed or cut-short run is never taken for one)."""
    found = cached(clip, cfg)
    if found is not None:
        return found
    if not Path(clip).is_file():
        raise RuntimeError(f"sam_cutout: no clip {clip}")
    weights_dir = weights()
    if not weights_dir:
        raise RuntimeError("sam_cutout: SAM2's weights were not found (set ARC_SAM2_WEIGHTS)")
    folder = mask_folder(clip, cfg)
    tmp = folder.with_name(folder.name + ".partial")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    seeds_file = None
    if isinstance(cfg["seeds"], dict):
        seeds_file = tmp.with_name(tmp.name + "_seeds.json")
        seeds_file.write_text(json.dumps(cfg["seeds"]), encoding="utf-8")
    status(f"SAM2: cutting {Path(clip).name} out (once: the renders after read it from {folder})...")
    proc = subprocess.run(command(clip, cfg, tmp, seeds_file, weights_dir), capture_output=True, text=True,
                          cwd=str(ROOT))
    if seeds_file is not None:
        seeds_file.unlink(missing_ok=True)
    if proc.returncode != 0 or not (tmp / "track.json").is_file():
        why = (proc.stderr or proc.stdout or "").strip().splitlines()
        shutil.rmtree(tmp, ignore_errors=True)
        raise RuntimeError(f"SAM2 could not cut {Path(clip).name} out: " + " | ".join(why[-4:]))
    shutil.rmtree(folder, ignore_errors=True)                 # a run that failed before left it empty
    tmp.rename(folder)
    preview = Path(str(tmp) + "_preview.mp4")
    if preview.is_file():
        preview.replace(Path(str(folder) + "_preview.mp4"))
    status(f"SAM2: {Path(clip).name} cut out -> {folder}")
    return folder


def _is_sam(fx) -> bool:
    return isinstance(fx, dict) and fx.get("source") == SOURCE and fx.get("enabled", True) is not False


def resolve(spec: dict, make: Optional[Callable[[str, dict], Path]] = None) -> dict:
    """A layer with the sam_cutout FX in its ``fx:``: a footage layer gets its ``cutout:``
    (the mask track, made if need be) and loses the FX; anywhere else the FX goes (there
    is nothing to cut). Other layers come back as they were."""
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
    track = (make or ensure)(str(spec["file"]), cfg)
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

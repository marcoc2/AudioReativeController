"""Small things the web modules share: paths as the projects write them, JSON-safe numbers,
copies that keep the scene's signals, the shaders the page can run."""
from __future__ import annotations

import copy
import math
from pathlib import Path

import yaml

from core.web import ROOT

PREVIEW_RES = "640x360"
# the final video's sizes the choosers offer, by shape (a size can also be typed)
RESOLUTIONS = (("9:16 portrait", ("480x854", "720x1280", "1080x1920")),
               ("10:16 portrait", ("480x768", "720x1152", "1080x1728", "1200x1920")),
               ("4:5 portrait", ("480x600", "720x900", "864x1080", "1080x1350")),
               ("1:1 square", ("480x480", "720x720", "1080x1080", "2160x2160")),
               ("5:4 landscape", ("600x480", "900x720", "1080x864", "1350x1080")),
               ("16:10 landscape", ("768x480", "1152x720", "1728x1080", "1920x1200", "2560x1600")),
               ("16:9 landscape", ("854x480", "1280x720", "1920x1080", "2560x1440", "3840x2160")))


def num(v) -> float:
    """A JSON-safe number (numpy, NaN and infinities made plain)."""
    v = float(v)
    return v if math.isfinite(v) else 0.0


def resolve(path, base: Path = ROOT) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else base / p


def shown_path(p: Path) -> str:
    """A path as the project files write it: relative to the repository when inside it."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def yaml_text(value) -> str:
    from core.scene_doc import plain
    return yaml.safe_dump(plain(value), default_flow_style=True, sort_keys=False, width=4096).strip()


def shader_live(entry) -> bool:
    """A shader the page can run itself: one pass, reading nothing or the picture below."""
    if not entry or entry.source != "shadertoy" or not entry.file or not entry.file.endswith(".glsl"):
        return False
    from core import shadertoy
    try:
        src = shadertoy.load(resolve(entry.file))
    except Exception:
        return False
    return set(src["channels"].get("image", {}).values()) <= {"frame"}


def copy_keeping_signals(obj):
    """A deep copy whose signals (anchored parts of the scene) stay the scene's objects."""
    from core.scene_doc import _anchor_name, _walk
    memo = {id(o): o for _, o in _walk(obj) if _anchor_name(o)}
    return copy.deepcopy(obj, memo)


def cue_key(spec: dict) -> str:
    """Cues are kept by what they are, not by the member's place (members come and go)."""
    return f"{spec.get('file')}|{spec.get('cues')!r}"


def preview_res(render: dict) -> str:
    """The preview's size: the project's own shape, its long side 640 (both even, as the
    encoder needs); never bigger than the project."""
    try:
        w, h = (int(x) for x in str(render.get("resolution", "854x480")).lower().split("x"))
    except ValueError:
        return PREVIEW_RES
    if w <= 0 or h <= 0:
        return PREVIEW_RES
    k = min(1.0, 640 / max(w, h))
    return f"{max(2, round(w * k / 2) * 2)}x{max(2, round(h * k / 2) * 2)}"

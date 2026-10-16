"""The arrangement: what plays in the background and in the foreground, bar by bar.

The Arrange window keeps a scene as the clip montage (``source: clips``), two lanes and
the master over them (core/video/track_fx.py runs them):

    - source: clips                     # the montage, wherever the BG has nothing
    - source: group
      name: BG                          # the background: its members over the montage,
      canvas: below                     # then its FX on all of it
      layers:
        - {source: footage, role: background, fit: cover, loop: true, file: C:/v/a/final.mp4,
           blocks: [{bars: [1, 4], from: 42.0}]}
        - {source: shadertoy, file: shaders/shadertoy/nebulosa_arc.glsl, bars: [[5, 17]]}
      fx: [{source: feedback, bars: [[5, 8]]}]
    - source: group
      name: FG                          # the foreground: drawn on its own, then over the BG
      layers:
        - {source: footage, file: lion.mp4, cutout: {...}, bars: [[5, 17]], fx: [...]}
    - {source: grain}                   # the master: over everything

A member is a clip in its blocks (``blocks:``, each from where in the clip) or a Generator
in its ``bars:`` windows; each can have its own ``fx:``. A scene written as a flat stack is
grouped by where its layers sit (``grouped``).

This module is the window's model, kept apart from the widgets so it can be tested:
reading the members of a lane, placing and removing blocks (a new block takes the bars it
covers from the blocks there), packing overlapping members into sub-rows, new characters,
the clips of a folder, a frame of a clip as a thumbnail, and the mask track SAM2 made of a
clip.
"""
from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

VIDEOS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")
IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
THUMBS = Path("render_output") / ".thumbs"          # never inside the user's clip folders
MASKS = Path("render_output") / "footage"


@dataclass
class Block:
    first: int                 # DAW bar numbers, both included
    last: int
    start: float = 0.0         # where in the clip it starts (s)
    rate: float = 1.0
    file: Optional[str] = None # None: the lane's own clip

    def spec(self) -> dict:
        out = {"bars": [self.first, self.last]}
        if self.file:
            out["file"] = self.file
        if self.start:
            out["from"] = round(float(self.start), 3)
        if self.rate != 1.0:
            out["rate"] = float(self.rate)
        return out


def role_of(spec: dict) -> str:
    """background | foreground: as the layer says, else a cutout or a placed clip is a
    character and a frame-filling one without a file is the background."""
    if spec.get("role") in ("background", "foreground"):
        return spec["role"]
    if spec.get("cutout") or spec.get("fit", "contain") == "contain" and spec.get("file"):
        return "foreground"
    return "background"


def blocks_of(spec: dict) -> List[Block]:
    out = []
    for b in spec.get("blocks") or []:
        if isinstance(b, dict) and isinstance(b.get("bars"), (list, tuple)) and len(b["bars"]) == 2:
            out.append(Block(int(b["bars"][0]), int(b["bars"][1]), float(b.get("from", 0.0)),
                             float(b.get("rate", 1.0)), str(b["file"]) if b.get("file") else None))
    return sorted(out, key=lambda b: b.first)


def windows_of(spec: dict) -> List[Block]:
    """A layer's ``bars:`` windows ([[5, 17], [36, 39]] or [5, 17]) as blocks."""
    bars = spec.get("bars")
    if not bars:
        return []
    bars = [bars] if not isinstance(bars[0], (list, tuple)) else bars
    return [Block(int(a), int(b)) for a, b in bars]


def with_windows(spec: dict, blocks: List[Block]) -> dict:
    """A copy of a Generator's or FX's layer ``spec`` shown in ``blocks`` (``bars:``); with
    none it plays the whole song."""
    out = dict(spec)
    if blocks:
        out["bars"] = [[b.first, b.last] for b in sorted(blocks, key=lambda b: b.first)]
    else:
        out.pop("bars", None)
    return out


def place(blocks: List[Block], new: Block, bar_seconds: float = 0.0) -> List[Block]:
    """``new`` placed: the bars it covers are taken from the blocks there. A block across
    it is cut in two; its second part goes on in its clip from where it would have been
    (``bar_seconds``, the length of a bar, says how far that is; 0: from where it began)."""
    if new.last < new.first:
        new.first, new.last = new.last, new.first
    out = []
    for b in blocks:
        if b.last < new.first or b.first > new.last:
            out.append(b)
            continue
        if b.first < new.first:
            out.append(Block(b.first, new.first - 1, b.start, b.rate, b.file))
        if b.last > new.last:
            skipped = (new.last + 1 - b.first) * bar_seconds * b.rate
            out.append(Block(new.last + 1, b.last, round(b.start + skipped, 3), b.rate, b.file))
    out.append(new)
    return sorted(out, key=lambda b: b.first)


def remove(blocks: List[Block], k: int) -> List[Block]:
    return [b for i, b in enumerate(blocks) if i != k]


def block_at(blocks: List[Block], bar: int) -> Optional[int]:
    for i, b in enumerate(blocks):
        if b.first <= bar <= b.last:
            return i
    return None


def with_blocks(spec: dict, blocks: List[Block]) -> dict:
    """A copy of the layer ``spec`` with ``blocks`` (the key dropped when there are none,
    but for a lane with no clip of its own: ``blocks: []``, it shows nothing yet).
    A layer that was shown by ``bars:`` windows is shown by its blocks instead."""
    out = dict(spec)
    if spec.get("cues") is None and spec.get("bars") and not spec.get("blocks"):
        for key in ("bars", "fade_in", "fade_out"):
            out.pop(key, None)
    if blocks:
        out["blocks"] = [b.spec() for b in sorted(blocks, key=lambda b: b.first)]
    elif not spec.get("file"):
        out["blocks"] = []
    else:
        out.pop("blocks", None)
    return out


# ----------------------------------------------------------------------- the lanes: BG, FG, master

BG, FG = "BG", "FG"
CLIP, GENERATOR, FX = "clip", "generator", "fx"


@dataclass
class Member:
    """One thing in a lane: a clip, a character or a Generator, in its bars."""
    at: int                    # its place in the lane's ``layers:``
    spec: dict
    kind: str                  # clip | generator | fx (an FX among the layers, written by hand)
    label: str
    blocks: List[Block] = field(default_factory=list)
    cued: bool = False         # played by its ``cues:`` (seconds, the REAPER edit): shown, not edited
    always: bool = False       # no blocks nor cues: the whole song
    row: int = 0               # the lane's sub-row it is drawn on (overlaps stack up)

    @property
    def enabled(self) -> bool:
        return self.spec.get("enabled", True) is not False


def kind_of(spec: dict) -> str:
    if spec.get("source") == "footage":
        return CLIP
    from core import catalog
    entry = catalog.of_layer(spec)
    return entry.kind if entry else (FX if spec.get("source") == "meters" else GENERATOR)


def label_of(spec: dict) -> str:
    """What the window calls a layer: its clip, its Generator or FX."""
    if spec.get("source") == "footage" and spec.get("file"):
        return Path(str(spec["file"])).stem
    if spec.get("source") == "footage":
        bl = blocks_of(spec)
        return Path(bl[0].file).stem if bl and bl[0].file else "clip"
    from core import catalog
    entry = catalog.of_layer(spec)
    if entry:
        return entry.label
    return Path(str(spec["file"])).stem if spec.get("file") else str(spec.get("source", "clips"))


def members_of(group: dict) -> List[Member]:
    out = []
    for m, spec in enumerate(group.get("layers") or []):
        if not isinstance(spec, dict):
            continue
        kind = kind_of(spec)
        cued = spec.get("cues") is not None
        blocks = blocks_of(spec) if kind == CLIP else []
        if not blocks and not cued:
            blocks = windows_of(spec)
        out.append(Member(m, spec, kind, label_of(spec), blocks, cued, always=not blocks and not cued))
    return out


def pack(members: List[Member], n_bars: int, cue_bars: Optional[dict] = None) -> int:
    """Each member a sub-row, the topmost (the last in the lane) first: one that overlaps
    nothing already on a row shares it; a cued one by its cues' bars (``cue_bars``: member's
    place -> [(first, last)]) once they are known. Returns how many rows."""
    rows: List[list] = []
    for mem in reversed(members):
        if mem.cued and (cue_bars or {}).get(mem.at):
            spans = cue_bars[mem.at]
        else:
            spans = [(1, n_bars)] if mem.always or mem.cued else [(b.first, b.last) for b in mem.blocks]
        for r, taken in enumerate(rows):
            if not any(a <= d and c <= b for a, b in spans for c, d in taken):
                mem.row = r
                taken.extend(spans)
                break
        else:
            mem.row = len(rows)
            rows.append(list(spans))
    return len(rows)


def group(name: str, layers: Optional[list] = None, fx: Optional[list] = None) -> dict:
    out = {"source": "group", "name": name}
    if name == BG:
        out["canvas"] = "below"
    out["layers"] = list(layers or [])
    if fx:
        out["fx"] = list(fx)
    return out


def _split_blocks(spec: dict) -> list:
    """A background clip lane of several blocks: one member per block (each its own FX)."""
    blocks = spec.get("blocks") or []
    if spec.get("source") != "footage" or len(blocks) < 2 or spec.get("cues") is not None:
        return [spec]
    out = []
    for b in blocks:
        one = {k: v for k, v in spec.items() if k != "blocks"}
        if isinstance(b, dict) and b.get("file"):
            one["file"] = b["file"]
            b = {k: v for k, v in b.items() if k != "file"}
        one["blocks"] = [b]
        out.append(one)
    return out


def grouped(layers: list) -> tuple:
    """The scene's layers as the Arrange window keeps them: the clip montage, the BG lane,
    the FG lane, then the master (the FX, and anything else, over everything). Returns
    (layers, whether they had to be grouped). A scene written as a flat stack is grouped by
    where its layers sit: under the first character, the background (its Generators and
    clips are the BG's members, its FX the BG's FX); up to the last clip above it, the
    foreground; above, the master."""
    layers = [s for s in layers or [] if isinstance(s, dict)]
    have = {s.get("name"): s for s in layers if s.get("source") == "group"}
    if BG in have and FG in have:
        montage = [s for s in layers if s.get("source", "clips") == "clips"]
        before_fg = layers[:layers.index(have[FG])]
        loose = [s for s in before_fg if s.get("source", "clips") not in ("clips", "group")]
        bg = have[BG]
        if loose:                                        # written by hand under the FG: into the BG
            bg = dict(bg)
            bg["layers"] = list(bg.get("layers") or []) + [s for s in loose if kind_of(s) != FX]
            bg["fx"] = list(bg.get("fx") or []) + [s for s in loose if kind_of(s) == FX]
        master = layers[layers.index(have[FG]) + 1:]
        return montage + [bg, have[FG]] + master, bool(loose)
    fg_at = [i for i, s in enumerate(layers) if s.get("source") == "footage" and role_of(s) == "foreground"]
    first = min(fg_at) if fg_at else len(layers)
    last = max((i for i, s in enumerate(layers) if s.get("source") == "footage" and i >= first), default=-1)
    montage, bg_l, bg_fx, fg_l, fg_fx, master = [], [], [], [], [], []
    for i, s in enumerate(layers):
        if s.get("source", "clips") == "clips":
            montage.append(s)
        elif s.get("source") == "group":
            (bg_l if s.get("name") == BG else fg_l).extend(s.get("layers") or [])
        elif i < first:
            (bg_fx if kind_of(s) == FX else bg_l).extend(_split_blocks(s) if kind_of(s) != FX else [s])
        elif i <= last:
            (fg_fx if kind_of(s) == FX else fg_l).append(s)
        else:
            master.append(s)
    return montage + [group(BG, bg_l, bg_fx), group(FG, fg_l, fg_fx)] + master, True


def find_mask_track(clip: str, root: Path = MASKS) -> Optional[str]:
    """The mask track SAM2 made of ``clip`` (segment_video.py --start-time 0 into
    render_output/footage/...): a ``<clip stem>_segments`` folder with a track.json."""
    stem = Path(clip).stem
    if not root.is_dir():
        return None
    for track in sorted(root.rglob(f"{stem}_segments/track.json")):
        return track.parent.as_posix()
    return None


def mask_folder_for(clip: str, root: Path = MASKS) -> str:
    """Where segment_video.py should write ``clip``'s mask track."""
    return (root / f"{Path(clip).stem}_segments").as_posix()


def new_foreground(clip: str, blocks: Optional[List[Block]] = None) -> dict:
    """A character lane for ``clip``: placed low in the frame, looping, cut out by its mask
    track when SAM2 has made one: every object, or all but the backdrop when SAM2 took the
    background for one (``drop:``)."""
    spec = {"source": "footage", "role": "foreground", "file": clip, "fit": "contain",
            "loop": True, "scale": 0.6, "x": 0.5, "y": 0.62}
    track = find_mask_track(clip)
    if track:
        spec["cutout"] = {"track": track, "feather": 2}
        backdrop = backdrop_of(track)
        if backdrop:
            spec["cutout"]["drop"] = backdrop
    return with_blocks(spec, blocks or [])


def backdrop_of(track: str, share: float = 0.45) -> List[int]:
    """The objects of a mask track that are the backdrop, not a figure: those covering more
    than ``share`` of the frame when found (the white behind a character on white)."""
    import json
    try:
        meta = json.loads((Path(track) / "track.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted(int(o["id"]) for o in meta.get("objects", []) if float(o.get("share", 0)) > share)


def list_clips(folder: str, pattern: str = "", limit: int = 2000) -> List[Path]:
    """The videos and images under ``folder`` (all levels) whose path, from it, contains
    ``pattern`` (any case), sorted."""
    root = Path(folder)
    if not root.is_dir():
        return []
    pat = pattern.strip().lower()
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() in VIDEOS + IMAGES and p.is_file() and ".thumbs" not in p.parts \
                and ".arc_thumbs" not in p.parts:
            if not pat or pat in p.relative_to(root).as_posix().lower():
                out.append(p)
                if len(out) >= limit:
                    break
    return out


def duration(path: str) -> float:
    """Seconds of a video (0 for an image or when unknown)."""
    if Path(path).suffix.lower() in IMAGES:
        return 0.0
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip())
    except (ValueError, OSError, subprocess.SubprocessError):
        return 0.0


def thumbnail(path: str, t: float = 0.0, size: int = 192, cache: Path = THUMBS) -> Optional[Path]:
    """A frame of ``path`` at ``t`` s as a PNG at most ``size`` px a side (cached)."""
    cache.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{Path(path).resolve()}|{round(t, 2)}|{size}".encode()).hexdigest()[:16]
    out = cache / f"{key}.png"
    if out.is_file():
        return out
    vf = f"scale='if(gt(iw,ih),{size},-2)':'if(gt(iw,ih),-2,{size})'"
    cmd = ["ffmpeg", "-v", "error", "-y"]
    if Path(path).suffix.lower() not in IMAGES:
        cmd += ["-ss", f"{max(0.0, t):.3f}"]
    cmd += ["-i", str(path), "-frames:v", "1", "-vf", vf, str(out)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out if out.is_file() else None


def colour(file: Optional[str]) -> tuple:
    """A steady colour for a clip, to tell its blocks apart."""
    h = int(hashlib.md5(str(file).encode()).hexdigest()[:6], 16)
    r, g, b = (h >> 16) & 255, (h >> 8) & 255, h & 255
    return (90 + r % 150, 90 + g % 150, 90 + b % 150)

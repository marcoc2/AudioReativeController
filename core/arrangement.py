"""The arrangement: which clip, Generator or FX plays in which bars, behind and in front.

Every layer of a scene is a lane (a track). A ``footage`` layer is a clip lane: a
background lane shows a clip per block
(``file`` on each block, filling the frame); a foreground lane is one character's clip,
cut out by its own mask track and placed in the frame, shown in its blocks' bars:

    - source: footage
      role: background            # what the Arrange window calls it (background | foreground)
      fit: cover
      loop: true
      blocks:
        - {bars: [1, 4], file: C:/v/a/final.mp4, from: 42.0}
        - {bars: [5, 8], file: C:/v/b/final.mp4, from: 3.5}

A Generator or FX lane (core/catalog.py) is shown in its ``bars:`` windows, so its blocks
are bars and nothing else: ``bars: [[5, 8], [17, 20]]``; without them it plays the whole
song. The base clip montage (``source: clips``) is a lane too, always on, under the rest.
A lane's own FX (``fx:``, core/video/track_fx.py) are lanes too, in its ``chain``.

This module is the window's model, kept apart from the widgets so it can be tested:
reading the lanes of a layer list, placing and removing blocks (a new block takes the
bars it covers from the blocks there), new lanes, the clips of a folder, a frame of a
clip as a thumbnail, and the mask track SAM2 made of a clip.
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


CLIP, GENERATOR, FX, MONTAGE = "clip", "generator", "fx", "clips"


@dataclass
class Lane:
    index: int                 # the layer's place in the scene's stack
    role: str                  # a clip lane: background | foreground; else its kind
    file: Optional[str]
    blocks: List[Block] = field(default_factory=list)
    cued: bool = False         # it also has ``cues:`` (seconds, or the REAPER edit): shown, not edited
    spec: dict = field(default_factory=dict)
    always: bool = False       # no cues, blocks nor bars: it plays the whole song
    kind: str = CLIP           # clip | generator | fx | clips (the base montage)
    name: str = ""             # a Generator's or FX's name (catalog)
    sub: Optional[int] = None  # one of a layer's own FX: its place in the layer's ``fx:``
    chain: List["Lane"] = field(default_factory=list)   # the layer's own FX, in order

    @property
    def label(self) -> str:
        if self.kind != CLIP:
            return self.name
        if self.file:
            return Path(self.file).stem
        return "background" if self.role == "background" else f"layer {self.index}"

    @property
    def enabled(self) -> bool:
        return self.spec.get("enabled", True) is not False


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


def lanes_of(layers: list) -> List[Lane]:
    """Every layer of a scene's layer list, as a lane, in stack order (the first is under
    the rest). A clip lane shown by ``bars:`` windows (and no cues or blocks) shows them as
    its blocks: arranging it turns them into blocks (``with_blocks``)."""
    lanes = []
    for i, spec in enumerate(layers or []):
        if not isinstance(spec, dict):
            continue
        if spec.get("source", "clips") != "footage":
            lanes.append(_effect_lane(i, spec))
            lanes[-1].chain = _chain(i, spec)
            continue
        cued = spec.get("cues") is not None
        blocks = blocks_of(spec)
        if not blocks and not cued:
            blocks = windows_of(spec)
        lanes.append(Lane(index=i, role=role_of(spec), file=str(spec["file"]) if spec.get("file") else None,
                          blocks=blocks, cued=cued, spec=spec,
                          always=not blocks and not cued and not spec.get("bars"), chain=_chain(i, spec)))
    return lanes


def _chain(i: int, spec: dict) -> List[Lane]:
    """A layer's own FX (``fx:``) as lanes."""
    out = []
    for j, fx in enumerate(spec.get("fx") or []):
        if isinstance(fx, dict):
            lane = _effect_lane(i, fx)
            lane.sub = j
            out.append(lane)
    return out


def _effect_lane(i: int, spec: dict) -> Lane:
    """A Generator's or FX's lane (or the base montage's): its ``bars:`` windows are its blocks."""
    source = spec.get("source", "clips")
    if source == "clips":
        return Lane(index=i, role=MONTAGE, file=None, spec=spec, always=True, kind=MONTAGE, name="clip montage")
    from core import catalog
    entry = catalog.of_layer(spec)
    kind = entry.kind if entry else (FX if source == "meters" else GENERATOR)
    name = entry.label if entry else (Path(str(spec["file"])).stem if spec.get("file") else source)
    blocks = windows_of(spec)
    return Lane(index=i, role=kind, file=None, blocks=blocks, spec=spec, always=not blocks, kind=kind, name=name)


def with_windows(spec: dict, blocks: List[Block]) -> dict:
    """A copy of a Generator's or FX's layer ``spec`` shown in ``blocks`` (``bars:``); with
    none it plays the whole song."""
    out = dict(spec)
    if blocks:
        out["bars"] = [[b.first, b.last] for b in sorted(blocks, key=lambda b: b.first)]
    else:
        out.pop("bars", None)
    return out


def master_at(layers: list) -> int:
    """Where an FX over everything goes: the top of the stack (under a ``meters`` read-out,
    which stays last)."""
    at = len(layers)
    while at > 0 and isinstance(layers[at - 1], dict) and layers[at - 1].get("source") == "meters":
        at -= 1
    return at


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


def new_background(blocks: Optional[List[Block]] = None) -> dict:
    return with_blocks({"source": "footage", "role": "background", "fit": "cover", "loop": True},
                       blocks or [])


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


def insert_at(layers: list, role: str) -> int:
    """Where a new lane goes in the stack: a background (a clip lane or a Generator) right
    above the base clips (so the background effects work on it), a character above the last
    character (else at the top)."""
    if role in ("background", GENERATOR):
        return 1 if layers else 0
    last = max((i for i, s in enumerate(layers) if isinstance(s, dict) and s.get("source") == "footage"
                and role_of(s) == "foreground"), default=None)
    return len(layers) if last is None else last + 1


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

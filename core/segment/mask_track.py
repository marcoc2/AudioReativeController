"""Mask track — the stems of the picture: the objects of a video, found and followed by SAM2,
saved once so any number of effects can be tried on top without running SAM2 again.

Demucs splits the song into stems; this splits the render into objects. A video is read
once to find its shots (a picture change bigger than ``cut``); on the first frame of each
shot SAM2 finds the objects by itself, and follows them to the end of the shot. Every
``reseed`` seconds it looks again and adds what came into view since (only what no object
already covers). Everything else is the background, id 0.

Folder (``<video stem>_segments/`` next to the video, as ``sheetsage/`` sits next to a song):

    track.json      the video (size, fps, frames, where it starts in the song), the shots,
                    and one entry per object: id, shot, first and last frame it is seen,
                    how it was found, its colour and size when found
    labels/NNNNNN.png   16-bit label map per frame: pixel = the object id there (0 = background);
                    where objects overlap the smaller one is on top
    stats.npy       float32 frames × (ids + 1) × 5: share of the frame, centre x, centre y,
                    width, height (fractions of the frame; share 0 = not in view)

``MaskTrack(folder)`` reads it back by frame or by song time.
"""
from __future__ import annotations

import colorsys
import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import cv2
import numpy as np

FORMAT = 1
STATS = ("share", "cx", "cy", "w", "h")


def default_folder(video: str) -> Path:
    v = Path(video)
    return v.with_name(v.stem + "_segments")


def find_cuts(frames: Iterable[np.ndarray], threshold: float = 40.0):
    """Where each shot starts (frame 0 always) and how many frames there are: a cut is a
    mean change (0..255, on a 1/16 subsample) above ``threshold`` from one frame to the next."""
    starts, prev, n = [0], None, 0
    for i, f in enumerate(frames):
        small = f[::16, ::16].astype(np.int16)
        if prev is not None and np.abs(small - prev).mean() > threshold:
            starts.append(i)
        prev, n = small, i + 1
    return starts, n


def pick_objects(masks: List[np.ndarray], max_objects: int = 8, min_share: float = 0.005,
                 max_share: float = 0.6, granularity: str = "whole",
                 taken: Optional[np.ndarray] = None, max_taken: float = 0.3) -> List[np.ndarray]:
    """The masks worth following. ``whole`` looks at the big ones first (characters, not their
    eyes), ``parts`` at the small ones first; a mask mostly covered by one already picked is
    skipped, and so is one mostly covered by ``taken`` (what is already being followed)."""
    if granularity not in ("whole", "parts"):
        raise ValueError(f"granularity {granularity!r}: use whole or parts")
    if not masks or max_objects <= 0:
        return []
    total = masks[0].size
    ok = [m for m in masks if min_share <= m.sum() / total <= max_share]
    ok.sort(key=lambda m: m.sum(), reverse=granularity == "whole")
    picked, union = [], np.zeros_like(ok[0]) if ok else None
    for m in ok:
        area = m.sum()
        if (m & union).sum() / area > 0.5:
            continue
        if taken is not None and (m & taken).sum() / area > max_taken:
            continue
        picked.append(m)
        union |= m
        if len(picked) >= max_objects:
            break
    return picked


def id_colour(obj_id: int):
    """A stable, well-separated colour per id (golden-ratio hues), RGB 0..255."""
    if obj_id <= 0:
        return (0, 0, 0)
    r, g, b = colorsys.hsv_to_rgb((obj_id * 0.618034) % 1.0, 0.75, 1.0)
    return int(r * 255), int(g * 255), int(b * 255)


def preview_frame(frame: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """The frame with each object tinted in its colour, outlined and numbered."""
    out = frame.copy()
    for oid in np.unique(labels):
        if oid == 0:
            continue
        m = labels == oid
        c = np.array(id_colour(int(oid)), np.float32)
        out[m] = (out[m].astype(np.float32) * 0.55 + c * 0.45).astype(np.uint8)
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, contours, -1, tuple(int(v) for v in c), 2)
        ys, xs = np.nonzero(m)
        cv2.putText(out, str(int(oid)), (int(xs.mean()) - 6, int(ys.mean()) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def write_mask_track(video: str, frames: Callable[[], Iterable[np.ndarray]], segmenter,
                     out_dir: str, fps: float, start_time: float = 0.0, max_objects: int = 8,
                     min_share: float = 0.005, max_share: float = 0.6, granularity: str = "whole",
                     reseed: float = 2.0, cut: float = 40.0, points_per_side: int = 32,
                     min_iou: float = 0.6, min_stability: float = 0.7, preview=None,
                     model: str = "", log: Callable[[str], None] = print) -> dict:
    """Find and follow the objects of a video; write the folder described above.

    ``frames()`` gives a fresh pass over the frames (called twice: cuts, then masks);
    ``segmenter`` is a RegionTracker (or anything with its set_frame/add/step/forget/
    auto_masks); ``preview``, when given, gets ``.stdin.write`` of each tinted frame."""
    out = Path(out_dir)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for old in (out / "labels").glob("*.png"):
        old.unlink()

    starts, n = find_cuts(frames(), cut)
    ends = starts[1:] + [n]
    shot_of = np.zeros(n, int)
    for s, (a, b) in enumerate(zip(starts, ends)):
        shot_of[a:b] = s
    log(f"{n} frames, {len(starts)} shot(s)")

    objects: Dict[int, dict] = {}
    live: List[int] = []
    rows = []                     # (frame, id, share, cx, cy, w, h)
    next_id, last_seed, labels = 1, 0, None
    reseed_frames = int(round(reseed * fps)) if reseed and reseed > 0 else 0
    t0 = time.time()

    def seed(i, frame, shot_end, taken=None):
        nonlocal next_id
        room = max_objects - len(live)
        found = pick_objects(segmenter.auto_masks(frame, points_per_side, min_iou, min_stability),
                             room, min_share, max_share, granularity, taken)
        for m in found:
            oid = next_id
            next_id += 1
            segmenter.add(oid, frames=shot_end - i, mask=m)
            ys, xs = np.nonzero(m)
            objects[oid] = {"id": oid, "shot": int(shot_of[i]), "first": i, "last": i,
                            "found": "shot" if taken is None else "reseed",
                            "colour": [int(v) for v in frame[ys, xs].mean(axis=0)],
                            "share": round(float(m.mean()), 5)}
            live.append(oid)
        return len(found)

    for i, frame in enumerate(frames()):
        if i >= n:
            break
        segmenter.set_frame(i, frame)
        s = int(shot_of[i])
        if i == starts[s]:
            for oid in live:
                segmenter.forget(oid)
            live.clear()
            seed(i, frame, ends[s])
            last_seed = i
        elif reseed_frames and i - last_seed >= reseed_frames and len(live) < max_objects:
            seed(i, frame, ends[s], taken=labels > 0)
            last_seed = i
        masks = segmenter.step()
        live[:] = [oid for oid in live if oid in masks]

        H, W = frame.shape[:2]
        labels = np.zeros((H, W), np.uint16)
        order = sorted(masks.items(), key=lambda kv: -int(kv[1].sum()))   # small ones end on top
        for oid, m in order:
            labels[m] = oid
        for oid in np.unique(labels):
            if oid == 0:
                continue
            m = labels == oid
            ys, xs = np.nonzero(m)
            x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            rows.append((i, int(oid), m.mean(), xs.mean() / W, ys.mean() / H, (x1 - x0) / W, (y1 - y0) / H))
            objects[int(oid)]["last"] = i
        cv2.imwrite(str(out / "labels" / f"{i:06d}.png"), labels)
        if preview is not None:
            preview.stdin.write(preview_frame(frame, labels).tobytes())
        if (i + 1) % max(1, int(round(fps))) == 0:
            log(f"  {i + 1:5d}/{n}  objects in view={len(masks)}  found so far={next_id - 1}  "
                f"{(i + 1) / (time.time() - t0):.1f} fps")

    stats = np.zeros((n, next_id, len(STATS)), np.float32)
    for i, oid, *vals in rows:
        stats[i, oid] = vals
    np.save(out / "stats.npy", stats)
    H, W = (labels.shape if labels is not None else (0, 0))
    meta = {"format": FORMAT, "source": str(video), "width": W, "height": H, "fps": fps,
            "frames": n, "start_time": start_time, "model": model,
            "params": {"max_objects": max_objects, "min_share": min_share, "max_share": max_share,
                       "granularity": granularity, "reseed": reseed, "cut": cut,
                       "points_per_side": points_per_side, "min_iou": min_iou,
                       "min_stability": min_stability},
            "shots": [{"first": a, "last": b - 1} for a, b in zip(starts, ends)],
            "objects": [objects[k] for k in sorted(objects)]}
    (out / "track.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    log(f"done: {out}  ({len(objects)} objects in {len(starts)} shot(s))")
    return meta


class MaskTrack:
    """A mask track folder, read back. Frames are the video's; ``frame_at(t)`` maps song time."""

    def __init__(self, folder: str):
        self.folder = Path(folder)
        meta = json.loads((self.folder / "track.json").read_text(encoding="utf-8"))
        if meta.get("format") != FORMAT:
            raise ValueError(f"{folder}: mask track format {meta.get('format')}, expected {FORMAT}")
        self.meta = meta
        self.fps, self.frames = float(meta["fps"]), int(meta["frames"])
        self.start_time = float(meta.get("start_time", 0.0))
        self.width, self.height = int(meta["width"]), int(meta["height"])
        self.objects = {o["id"]: o for o in meta["objects"]}
        self.shots = meta["shots"]
        self.stats = np.load(self.folder / "stats.npy")
        self._cached = (None, None)

    def frame_at(self, t: float) -> Optional[int]:
        """The frame shown at song time ``t``, or None outside the video."""
        i = int(round((t - self.start_time) * self.fps))
        return i if 0 <= i < self.frames else None

    def labels(self, i: int) -> np.ndarray:
        if self._cached[0] != i:
            lab = cv2.imread(str(self.folder / "labels" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED)
            if lab is None:
                raise FileNotFoundError(f"{self.folder}: no labels for frame {i}")
            self._cached = (i, lab)
        return self._cached[1]

    def labels_at(self, t: float) -> Optional[np.ndarray]:
        i = self.frame_at(t)
        return None if i is None else self.labels(i)

    def mask(self, i: int, obj_id: Optional[int] = None) -> np.ndarray:
        """One object's mask on frame ``i``, or every object's (``obj_id`` None)."""
        lab = self.labels(i)
        return lab > 0 if obj_id is None else lab == obj_id

    def in_view(self, i: int) -> List[int]:
        """The ids seen on frame ``i``, largest first."""
        row = self.stats[i, :, 0]
        ids = [k for k in np.nonzero(row > 0)[0] if k > 0]
        return sorted((int(k) for k in ids), key=lambda k: -row[k])

    def stat(self, i: int, obj_id: int) -> dict:
        if obj_id >= self.stats.shape[1]:
            return dict.fromkeys(STATS, 0.0)
        return dict(zip(STATS, (float(v) for v in self.stats[i, obj_id])))

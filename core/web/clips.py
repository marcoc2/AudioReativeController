"""The browser's clips and the clips put on the lanes, as the Arrange window does it: any
folder (the last one remembered), its clips with their covers and lengths, a clip played from
its "from", clips put as long as they play (a long one asks how many bars first), files dropped
from the Explorer, and a clip's SAM2 cutout set (auto, or seeds of one's own on a frame).
"""
from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Optional

from core.web import WORK
from core.web.util import num, resolve, shown_path

LONG_CLIP = 30.0                  # s: a clip longer asks how many bars it takes (or all of it)
COVERS_MAX = 400
SAM_KEEPS = ("figures", "all", "drop1")


class Clips:
    """Mixed into the Session (it brings media, events, lock, bars(), new_member, put_in_montage...)."""

    # -- the browser's list ----------------------------------------------------------------
    def list_clips(self, folder: str = "", pattern: str = "") -> dict:
        """The folder's clips (the project's clips folder, else the last one used), each with
        its URL; their covers and lengths come as events (``covers``, ``lengths``)."""
        from core import arrangement as arr
        from core.web.api import prefs, remember
        folder = str(folder or "").strip()
        if not folder:
            d = self.clips_dir()
            folder = str(d) if d is not None else prefs().get("clips_folder", "")
        if not folder or not resolve(folder).is_dir():
            return {"folder": folder, "clips": [], "error": f"no folder at {folder}" if folder else None}
        root = resolve(folder)
        paths = arr.list_clips(str(root), pattern or "")
        remember(clips_folder=str(root))
        out = [{"path": str(p), "name": _rel(p, root), "url": self.media.url(p)} for p in paths]
        self._covers_later([str(p) for p in paths[:COVERS_MAX]])
        return {"folder": str(root), "clips": out}

    def _covers_later(self, paths: list) -> None:
        from core import arrangement as arr
        self._covers_gen = getattr(self, "_covers_gen", 0) + 1
        gen = self._covers_gen

        def run():
            covers, lengths = {}, {}
            for i, p in enumerate(paths):
                if gen != self._covers_gen:                # another list since
                    return
                try:
                    png = arr.cover(p)
                    if png is not None:
                        covers[self.media.url(p)] = self.media.url(png)
                    lengths[self.media.url(p)] = num(arr.duration(p))
                except Exception:
                    continue
                if len(covers) >= 12 or i == len(paths) - 1:
                    self.events.emit("covers", covers)
                    self.events.emit("lengths", lengths)
                    covers, lengths = {}, {}
            self.proxies.check(paths)
        threading.Thread(target=run, daemon=True, name="arc-covers").start()

    def clip_info(self, path: str) -> dict:
        """A clip chosen in the browser: its length, its mask track, what the page plays."""
        from core import arrangement as arr
        p = resolve(path)
        if not p.is_file():
            return {"error": f"no clip at {path}"}
        self.proxies.check([str(p)])
        mask = arr.find_mask_track(str(p))
        return {"path": str(p), "name": p.name, "duration": num(arr.duration(str(p))), "url": self.media.url(p),
                "proxy": self.proxies.status(p), "mask": Path(mask).name if mask else None,
                "image": p.suffix.lower() in arr.IMAGES}

    # -- clips put on the lanes ------------------------------------------------------------
    def clip_bars(self, seconds: float) -> int:
        """How many bars a clip put in takes: as long as it plays (an image: 4)."""
        if seconds <= 0:
            return 4
        return max(1, int(round(seconds / (self.bar_seconds() or 2.0))))

    def place_clips(self, lane: str, first: int, todo: list, bars=None, row=None) -> dict:
        """Clips (``[{path, from, seconds?}]``) put on ``lane`` (bg, fg, or clips: the montage)
        one after another from bar ``first``, each as long as it plays; ``bars``: the first
        one's (the answer to the question). One longer than LONG_CLIP stops there and asks:
        the answer is ``{asking: {...}}`` with what is left."""
        from core import arrangement as arr
        n = len(self.bars()) - 1
        first = max(1, min(n, int(first)))
        todo = [dict(t) for t in todo or []]
        last_state, placed = None, 0
        while todo:
            t = todo[0]
            path = str(resolve(t["path"]))
            if t.get("seconds") is None:
                t["seconds"] = num(arr.duration(path)) if Path(path).suffix.lower() not in arr.IMAGES else 0.0
            seconds = max(0.0, float(t["seconds"]) - float(t.get("from") or 0.0))
            if bars is None and seconds > LONG_CLIP:
                whole = min(self.clip_bars(seconds), n - first + 1)
                return {**(last_state or self.state()), "asking": {
                    "lane": lane, "first": first, "row": row, "todo": todo, "name": Path(path).name,
                    "seconds": seconds, "from": float(t.get("from") or 0.0), "bars": self.clip_bars(seconds),
                    "whole": whole, "to_end": whole < self.clip_bars(seconds), "suggest": min(8, whole)}}
            k = self.clip_bars(seconds) if bars in (None, "whole") else max(1, int(bars))
            last = min(n, first + k - 1)
            if lane == "clips":
                last_state = self.put_in_montage(first, last, path, float(t.get("from") or 0.0))
            else:
                last_state = self.new_member(lane, path, first, last, row, float(t.get("from") or 0.0))
            if "error" in last_state:
                return last_state
            placed += 1
            todo, bars = todo[1:], None
            if last >= n:                                    # the song's end
                break
            first = last + 1
        out = last_state or self.state()
        if placed > 1:
            out["note"] = f"{placed} clips put one after another"
        return out

    # -- a frame of a clip, for the SAM2 seeds ----------------------------------------------
    def clip_frame(self, path: str, t: float) -> dict:
        """The clip's frame at ``t`` s as a JPEG the page shows (its size kept)."""
        import cv2
        p = resolve(path)
        cap = cv2.VideoCapture(str(p))
        img = None
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
            ok, img = cap.read()
            img = img if ok else None
        cap.release()
        if img is None:
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            return {"error": f"no frame at {t:.2f} s of {p.name}"}
        h, w = img.shape[:2]
        k = min(1.0, 960 / max(h, w))
        if k < 1.0:
            img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
        key = hashlib.sha1(f"{str(p).lower()}|{float(t):.3f}".encode()).hexdigest()[:16]
        out = WORK / "frames" / f"{key}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return {"url": self.media.url(out), "width": w, "height": h}

    # -- the SAM2 cutout ------------------------------------------------------------------
    def sam_open(self, path) -> dict:
        """The sam_cutout FX at ``path`` (a member's ``fx`` item): its clip, its settings, its
        seeds, the frame its block starts on."""
        from core import arrangement as arr
        from core.scene_doc import plain
        from core.video import sam_cutout
        with self.lock:
            fx = self.get(path)
            owner = self.get(list(path[:-2])) if len(path) >= 2 and str(path[-2]) == "fx" else None
        clip = owner.get("file") if isinstance(owner, dict) and owner.get("source") == "footage" else None
        if not isinstance(fx, dict) or fx.get("source") != sam_cutout.SOURCE or not clip:
            return {"error": "the SAM2 cutout works on a clip: add it to a character's FX (its block in the FG)"}
        cfg = sam_cutout.config(plain(fx))
        blocks = plain(owner).get("blocks") or []
        start = float(blocks[0].get("from", 0.0)) if blocks and isinstance(blocks[0], dict) else 0.0
        seeds = cfg["seeds"] if isinstance(cfg["seeds"], dict) else None
        objects = [{"box": list(o["box"]) if o.get("box") else None, "points": [list(q) for q in o.get("points") or []]}
                   for o in (seeds or {}).get("objects") or []] or [{"box": None, "points": []}]
        keep = "drop1" if cfg.get("drop") == [1] else "all" if cfg.get("keep") == "all" else "figures"
        return {"clip": str(resolve(clip)), "name": Path(str(clip)).name, "duration": num(arr.duration(str(resolve(clip)))),
                "start": start, "mode": "seeds" if seeds else "auto", "at": float(seeds.get("at", start)) if seeds else start,
                "objects": objects, "max_objects": int(cfg["max_objects"]), "granularity": str(cfg["granularity"]),
                "reseed": float(cfg["reseed"]), "keep": keep, "feather": int(cfg.get("feather", 2)),
                "kept": {k: v for k, v in plain(fx).items() if k in ("enabled", "grow", "objects")},
                "cache": self.sam_cache(str(resolve(clip)), self._sam_spec(None, plain(fx)))}

    @staticmethod
    def _sam_spec(form: Optional[dict], fx: Optional[dict] = None) -> Optional[dict]:
        """The FX the panel says (``form``), or ``fx`` as it is; None: seeding, nothing drawn."""
        from core.video import sam_cutout
        if form is None:
            return fx
        spec = {"source": sam_cutout.SOURCE, **(form.get("kept") or {})}
        if form.get("mode") == "seeds":
            objects = [{k: v for k, v in (("box", o.get("box")), ("points", o.get("points"))) if v}
                       for o in form.get("objects") or [] if o.get("box") or o.get("points")]
            if not objects:
                return None
            spec["seeds"] = {"at": round(float(form.get("at") or 0.0), 3), "objects": objects}
        else:
            spec.update(seeds="auto", max_objects=int(form.get("max_objects", 4)),
                        granularity=str(form.get("granularity", "whole")), reseed=round(float(form.get("reseed", 2.0)), 2))
        if form.get("keep") == "drop1":
            spec["drop"] = [1]
        else:
            spec["keep"] = "all" if form.get("keep") == "all" else "figures"
        spec["feather"] = int(form.get("feather", 2))
        return spec

    def sam_cache(self, clip: str, spec: Optional[dict]) -> str:
        from core.video import sam_cutout
        if spec is None:
            return "draw a box (or put a point) on what to cut out"
        done = sam_cutout.cached(clip, sam_cutout.config(spec))
        return (f"cut out already: {shown_path(done)} (the renders read it)" if done else
                "not cut out yet: the next exact preview or render runs SAM2 on the frames its blocks show (GPU), "
                "then they are kept in render_output/footage; more are cut only if more of the clip is used")

    def sam_check(self, clip: str, form: dict) -> dict:
        return {"cache": self.sam_cache(clip, self._sam_spec(form))}

    def sam_apply(self, path, form: dict) -> dict:
        """OK: the FX as the panel has it."""
        spec = self._sam_spec(form)
        if spec is None:
            return {"error": "draw a box (or put a point) on what to cut out first, or choose auto"}
        with self.lock:
            self._edit(list(path), lambda _: spec)
            return self._answer(note="SAM2 cutout set: the next exact preview or render cuts the clip out (once; then it is kept)")

    def sam_for_clip(self, path: str) -> dict:
        """The browser's "Cut it out (SAM2)": the FG member that plays this clip gets the
        sam_cutout FX (or keeps its own); -> the FX's path, and whether it is new."""
        from core.video import sam_cutout
        name = Path(str(path)).name.lower()
        with self.lock:
            for p, spec in self.track("fg"):
                if spec.get("file") and Path(str(spec["file"])).name.lower() == name:
                    fx = spec.get("fx") or []
                    j = next((j for j, f in enumerate(fx) if isinstance(f, dict) and f.get("source") == sam_cutout.SOURCE), None)
                    if j is not None:
                        return {"fx": list(p) + ["fx", j], "new": False}
                    self._edit(p, lambda s: {**s, "fx": list(s.get("fx") or []) + [{"source": sam_cutout.SOURCE}]})
                    return {**self._answer(), "fx": list(p) + ["fx", len(fx)], "new": True}
        return {"error": f"{Path(str(path)).name} is not in the FG: drag it there first, then cut it out"}


def _rel(p: Path, root: Path) -> str:
    try:
        return p.relative_to(root).as_posix()
    except ValueError:
        return p.name

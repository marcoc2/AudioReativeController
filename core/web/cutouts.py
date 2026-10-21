"""Characters cut out for the page: each FG clip with a cutout (its ``cutout:``, or the
SAM2 cache of its ``sam_cutout`` FX) as a video with its alpha, a VP9 WebM the page draws
in the live preview where the render puts the character.

The alpha is the engine's own (``Footage._matte``: the mask track's labels, ``objects`` /
``drop``, ``grow``, ``feather``), frame by frame of the clip, so the page shows the figure
the render shows. Made once per clip, cutout and mask track (a SAM2 track that grows makes a
new one), on the CPU in the background, in ``render_output/.web/cutouts/``. A SAM2 cutout
not made yet (it is made at the first render, on the GPU) has none: the page shows the
whole frame and says so.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import threading
from pathlib import Path
from typing import Optional

import numpy as np

from core.web import WORK
from core.web.util import resolve

FOLDER = WORK / "cutouts"
MAX_SIDE = 720                    # the proxy's longer side at most (the preview is small)


def cut_of(spec: dict) -> Optional[dict]:
    """The cutout the render uses for this clip layer: its ``cutout:``, or the cached track
    of its SAM2 FX (as ``sam_cutout.resolve`` builds it); {"missing": "sam"} when SAM2 has
    not made it yet; None when it is not cut out."""
    from core.scene_doc import plain
    cut = spec.get("cutout")
    if cut:
        return plain(cut) if isinstance(cut, dict) else {"track": str(cut)}
    from core.video import sam_cutout
    sams = [f for f in spec.get("fx") or [] if sam_cutout._is_sam(f)]
    if not sams or not spec.get("file"):
        return None
    cfg = sam_cutout.config(plain(sams[0]))
    track = sam_cutout.cached(str(resolve(spec["file"])), cfg) or sam_cutout.cached(str(spec["file"]), cfg)
    if track is None:
        return {"missing": "sam"}
    out = {"track": track.as_posix(), "feather": cfg.get("feather", 2)}
    for k in ("drop", "objects", "grow"):
        if cfg.get(k) is not None:
            out[k] = cfg[k]
    if "drop" not in out and "objects" not in out and cfg.get("keep") == "figures" and cfg["seeds"] == "auto":
        from core.arrangement import backdrop_of
        stamp = (track.as_posix(), _stamp(resolve(track) / "track.json"))
        if stamp not in _BACKDROPS:                    # read once per track (the lanes ask often)
            _BACKDROPS[stamp] = backdrop_of(str(track))
        if _BACKDROPS[stamp]:
            out["drop"] = _BACKDROPS[stamp]
    return out


_BACKDROPS: dict = {}


def _stamp(path: Path) -> str:
    try:
        st = path.stat()
        return f"{st.st_size}|{st.st_mtime_ns}"
    except OSError:
        return "-"


def key_of(file: str, cut: dict) -> str:
    clip = resolve(file)
    track = resolve(cut["track"]) / "track.json"
    what = json.dumps({k: cut.get(k) for k in ("track", "objects", "drop", "grow", "feather")}, sort_keys=True)
    return hashlib.sha1(f"{str(clip).lower()}|{_stamp(clip)}|{_stamp(track)}|{what}".encode()).hexdigest()[:16]


class Cutouts:
    """The cut-out videos: asked for by the session as it describes the FG, made one at a
    time in the background; when one is ready the page hears the lanes again."""

    def __init__(self, media, events, on_ready=None):
        self.media, self.events, self.on_ready = media, events, on_ready
        self.state: dict = {}            # key -> "making" | "ready" | "failed"
        self.out: dict = {}              # key -> the webm
        self._queue: list = []
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None

    def describe(self, spec: dict) -> Optional[dict]:
        """What the page needs to draw this character cut out: {state, url?, why?}; it is
        queued when it is not made yet."""
        if not spec.get("file"):
            return None
        cut = cut_of(spec)
        if cut is None:
            return None
        if cut.get("missing"):
            return {"state": "none", "why": "SAM2 cuts it out at the first render (on the GPU)"}
        track = resolve(cut["track"])
        if not (track / "track.json").is_file():
            return {"state": "none", "why": f"no mask track at {cut['track']}"}
        clip = resolve(spec["file"])
        if not clip.is_file():
            return {"state": "failed", "why": f"no clip at {spec['file']}"}
        key = key_of(spec["file"], cut)
        out = FOLDER / f"{clip.stem[:40]}_{key}.webm"
        if key not in self.state and out.is_file():
            self.state[key], self.out[key] = "ready", out
        if key not in self.state:
            self.state[key] = "making"
            with self._lock:
                self._queue.append((key, str(clip), cut, out))
            self._start()
        got = {"state": self.state[key]}
        if self.state[key] == "ready":
            got["url"] = self.media.url(self.out[key])
        return got

    def _start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True, name="arc-cutouts")
        self._worker.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    return
                key, clip, cut, out = self._queue.pop(0)
            try:
                self.make(clip, cut, out)
                self.state[key], self.out[key] = "ready", out
                self.events.log("sys", f"the character {Path(clip).name}, cut out for the live preview")
            except Exception as exc:
                self.state[key] = "failed"
                self.events.log("warn", f"{Path(clip).name}: no cut-out video for the live preview ({type(exc).__name__}: {exc})")
            if self.on_ready is not None:
                self.on_ready()

    @staticmethod
    def make(clip: str, cut: dict, out: Path) -> None:
        """The clip with its alpha (the engine's matte), as a VP9 WebM."""
        from core.video.footage import Footage, _Reader
        reader = _Reader(clip)
        k = min(1.0, MAX_SIDE / max(reader.width, reader.height))
        w, h = max(2, int(reader.width * k) // 2 * 2), max(2, int(reader.height * k) // 2 * 2)
        fg = Footage({"file": clip, "cutout": {**cut, "track": str(resolve(cut["track"]))}}, w, h)
        out.parent.mkdir(parents=True, exist_ok=True)
        part = out.with_suffix(".part.webm")
        proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba", "-s", f"{w}x{h}",
                                 "-r", f"{reader.fps:.6f}", "-i", "-", "-an", "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p",
                                 "-b:v", "0", "-crf", "36", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1",
                                 "-auto-alt-ref", "0", str(part)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        import cv2
        try:
            for i in range(max(1, reader.frames)):
                rgb = reader.frame(i)
                if rgb.shape[1] != w or rgb.shape[0] != h:
                    rgb = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
                a = fg._matte(i, w, h)
                rgba = np.dstack([rgb, np.clip(a * 255.0 + 0.5, 0, 255).astype(np.uint8)])
                proc.stdin.write(rgba.tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(proc.stderr.read().decode(errors="replace")[-300:])
        finally:
            if proc.poll() is None:
                proc.kill()
        part.replace(out)

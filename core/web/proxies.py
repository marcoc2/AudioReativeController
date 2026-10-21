"""Clips the page cannot play, made playable: a small H.264 proxy each, made once.

WebView2 plays H.264, VP8/9 and AV1 in MP4 or WebM; ``arrangement.VIDEOS`` also takes
ProRes ``.mov``, ``.mkv``, ``.avi`` and HEVC. Such a clip gets a proxy (540 px high, no
sound) in ``render_output/.web/proxies/``, named by the file's path and time, made by ffmpeg
on the CPU in the background; until it exists the page says "making a proxy" where the clip
shows. The render always reads the clip itself.
"""
from __future__ import annotations

import hashlib
import subprocess
import threading
from pathlib import Path
from typing import Optional

from core.web import WORK

PLAYABLE = {"h264", "vp8", "vp9", "av1"}
CONTAINERS = {".mp4", ".m4v", ".webm"}
FOLDER = WORK / "proxies"


def codec_of(path: Path) -> Optional[str]:
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=codec_name",
                              "-of", "csv=p=0", str(path)], capture_output=True, text=True, timeout=20)
        return out.stdout.strip().splitlines()[0].strip() if out.stdout.strip() else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def proxy_path(path: Path) -> Path:
    st = path.stat()
    key = hashlib.sha1(f"{str(path).lower()}|{st.st_size}|{st.st_mtime_ns}".encode()).hexdigest()[:16]
    return FOLDER / f"{path.stem[:40]}_{key}.mp4"


class Proxies:
    """What the page plays for a clip: itself, or its proxy once made."""

    def __init__(self, media, events):
        self.media, self.events = media, events
        self.state: dict = {}            # str(path) -> "direct" | "making" | "ready" | "failed"
        self.ready: dict = {}            # str(path) -> proxy path
        self._queue: list = []
        self._lock = threading.Lock()
        self._worker: Optional[threading.Thread] = None

    def url(self, path) -> str:
        """The URL the page plays ``path`` from (the proxy's when it is made)."""
        p = Path(path)
        got = self.ready.get(str(p))
        return self.media.url(got if got is not None else p)

    def status(self, path) -> str:
        return self.state.get(str(Path(path)), "unknown")

    def check(self, paths) -> int:
        """Look at each clip (its container, then its codec) and queue the ones that need a
        proxy; -> how many were queued."""
        n = 0
        for p in map(Path, paths):
            key = str(p)
            if key in self.state or not p.is_file():
                continue
            if p.suffix.lower() in CONTAINERS and codec_of(p) in PLAYABLE:
                self.state[key] = "direct"
                continue
            out = proxy_path(p)
            if out.is_file():
                self.state[key], self.ready[key] = "ready", out
                self.events.emit("proxies", {self.media.url(p): self.media.url(out)})
                continue
            self.state[key] = "making"
            with self._lock:
                self._queue.append(p)
            n += 1
        if n:
            self._start()
        return n

    def _start(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._run, daemon=True, name="arc-proxies")
        self._worker.start()

    def _run(self) -> None:
        while True:
            with self._lock:
                if not self._queue:
                    return
                p = self._queue.pop(0)
            out = proxy_path(p)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".part.mp4")
            cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(p), "-an", "-vf", "scale=-2:'min(540,ih)'",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "24", "-pix_fmt", "yuv420p",
                   "-movflags", "+faststart", str(tmp)]
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
                ok = r.returncode == 0 and tmp.is_file()
                if ok:
                    tmp.replace(out)
            except (OSError, subprocess.SubprocessError):
                ok = False
            key = str(p)
            if ok:
                self.state[key], self.ready[key] = "ready", out
                self.events.emit("proxies", {self.media.url(p): self.media.url(out)})
                self.events.log("sys", f"a proxy the page can play: {p.name}")
            else:
                self.state[key] = "failed"
                self.events.log("warn", f"{p.name}: no proxy could be made (the live preview leaves it out; "
                                        "the render reads the clip itself)")

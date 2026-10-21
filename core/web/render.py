"""The engine's render of some bars (clip_generator.py, as the studio calls it), its progress
from the engine's ``frame/total`` lines, and what the GPU is doing before one starts."""
from __future__ import annotations

import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

from core.web import ROOT, WORK
from core.web.session import Session, preview_res, resolve

FRAME_LINE = re.compile(r"\s*(\d+)/(\d+)\s+t=")


def gpu() -> Optional[dict]:
    """The GPU's use and memory (nvidia-smi), or None when there is no NVIDIA GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        name, util, used, total = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "util": int(util), "used_mb": int(used), "total_mb": int(total)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


class EngineRender:
    """One engine render at a time: started, watched, stopped."""

    def __init__(self, session: Session, events, media):
        self.s, self.events, self.media = session, events, media
        self.proc: Optional[subprocess.Popen] = None

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def start(self, first: int, last: int) -> dict:
        if self.running():
            return {"error": "a render is already running"}
        s = self.s
        with s.lock:
            if s.grid is None or s.doc is None:
                return {"error": "the project is not loaded yet"}
            stem = s.scene.stem
            scene = WORK / f"{stem}.yaml"
            s.write_copy(scene)
            db = s.bars()
            first, last = max(1, int(first)), min(len(db) - 1, int(last))
            out = WORK / f"{stem}_bars{first}-{last}.mp4"
            p = s.proj
            render = p.get("render") or {}
            clips = p.get("clips") or {}
            cmd = [sys.executable, "clip_generator.py", "--file", str(s.audio), "--scene", str(scene),
                   "--start-time", f"{db[first - 1]:.3f}", "--bars", str(last - first + 1), "--fps", str(s.fps),
                   "--resolution", preview_res(render), "--output", str(out)]
            if p.get("midi"):
                cmd += ["--midi", str(resolve(p["midi"]))]
            if clips.get("dir"):
                cmd += ["--clips", str(resolve(clips["dir"]))]
            if clips.get("order") and clips["order"] != "(scene)":
                cmd += ["--clip-order", str(clips["order"])]
            if clips.get("seed") is not None:
                cmd += ["--seed", str(clips["seed"])]
            if render.get("clip_fit") == "cover":
                cmd += ["--clip-fit", "cover"]
            if render.get("masks"):
                cmd += ["--masks", str(render["masks"])]
            if p.get("midi_offset"):
                cmd += ["--midi-offset", str(p["midi_offset"])]
            if str(p.get("meter") or "").strip():
                cmd += ["--meter", str(p["meter"]).strip()]
        self.events.log("out", " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)))
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1)
        threading.Thread(target=self._watch, args=(self.proc, out, first, last), daemon=True).start()
        return {"ok": True, "first": first, "last": last}

    def _watch(self, proc: subprocess.Popen, out: Path, first: int, last: int) -> None:
        tail: list = []
        for line in proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-12:]
            m = FRAME_LINE.match(line)
            if m:
                self.events.emit("engine", {"event": "progress", "frame": int(m.group(1)), "of": int(m.group(2))})
            elif line.strip():
                self.events.log("sys", line)
        code = proc.wait()
        if code == 0 and out.is_file():
            self.events.emit("engine", {"event": "done", "url": self.media.url(out), "first": first, "last": last,
                                        "path": str(out)})
        else:
            self.events.emit("engine", {"event": "failed", "code": code, "tail": tail})

    def stop(self) -> dict:
        if self.running():
            self.proc.terminate()
            return {"ok": True}
        return {"ok": False}

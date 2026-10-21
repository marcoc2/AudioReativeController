"""The engine (clip_generator.py), as the studio calls it: a preview of some bars, and the
final video (the whole song or some bars) with the project's render settings. Progress
from the engine's ``frame/total`` lines; the GPU's use and memory before one starts."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from core.web import ROOT, WORK
from core.web.project import render_settings_of
from core.web.util import preview_res, resolve, shown_path

FRAME_LINE = re.compile(r"^\s*(\d+)/(\d+)\s+t=")
FRAMES_OF = re.compile(r"\bframes=(\d+)")


def gpu() -> Optional[dict]:
    """The GPU's use and memory (nvidia-smi), or None when there is no NVIDIA GPU."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=8)
        name, util, used, total = [x.strip() for x in out.stdout.strip().splitlines()[0].split(",")]
        return {"name": name, "util": int(util), "used_mb": int(used), "total_mb": int(total)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def build_cmd(proj: dict, audio: Path, scene: Path, start: float, bars: int, resolution: str, output: Path,
              final: bool = False) -> list:
    """clip_generator.py's command line for the project (the studio's ``_build_render_cmd``):
    ``bars`` 0 is the whole song from ``start``."""
    rs = render_settings_of(proj)
    cmd = [sys.executable, "clip_generator.py", "--file", str(audio)]
    if rs["clips_dir"]:
        cmd += ["--clips", str(resolve(rs["clips_dir"]))]
    cmd += ["--bars", str(int(bars)), "--cache-size", str(rs["cache_size"]), "--fps", str(rs["fps"]),
            "--resolution", resolution, "--scene", str(scene)]
    if rs["order"] != "(scene)":
        cmd += ["--clip-order", rs["order"]]
    if rs["seed"] is not None:
        cmd += ["--seed", str(rs["seed"])]
    if final:
        cmd += ["--codec", rs["codec"]]
    if start > 0:
        cmd += ["--start-time", f"{start:.3f}"]
    if rs["stems"]:
        cmd += ["--stems"]
    if rs["clip_fit"] != "contain":
        cmd += ["--clip-fit", rs["clip_fit"]]
    if rs["turn_wide"]:
        cmd += ["--clip-turn-wide"]
    if rs["masks"]:
        cmd += ["--masks", str(resolve(rs["masks"]))]
    if rs["profile"] and final:
        cmd += ["--profile"]
    cmd += ["--output", str(output)]
    if proj.get("midi"):
        if float(proj.get("midi_offset") or 0.0):
            cmd += ["--midi-offset", str(proj["midi_offset"])]
        if str(proj.get("meter") or "").strip():
            cmd += ["--meter", str(proj["meter"]).strip()]
    g = rs["gravity"]
    if g["enabled"]:
        cmd += ["--gravity-peak", str(g["peak"]), "--gravity-floor", str(g["floor"]),
                "--gravity-radius", str(g["radius"]), "--gravity-curve", str(g["curve"])]
    if proj.get("midi"):
        cmd += ["--midi", str(resolve(proj["midi"]))]
    return cmd


def output_path(audio: Path, scene: Path, first: Optional[int] = None, last: Optional[int] = None,
                start: float = 0.0) -> Path:
    """render_output/<song>_<scene>[_bars<a>-<b> | _from<N>s].mp4, as the studio names it."""
    tail = f"_bars{first}-{last}" if first else (f"_from{start:.0f}s" if start > 0 else "")
    return ROOT / "render_output" / f"{audio.stem}_{scene.stem}{tail}.mp4"


class EngineRender:
    """The engine's preview of some bars, as a subprocess (the exact preview in-process is
    core/web/preview.py: this one is its fallback). One at a time."""

    def __init__(self, session, events, media):
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
            cmd = build_cmd(s.proj, s.audio, scene, db[first - 1], last - first + 1,
                            preview_res(s.proj.get("render") or {}), out)
        self.events.log("out", " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)))
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, encoding="utf-8", errors="replace", bufsize=1,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.events.gpu("the engine renders", True)
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
        self.events.gpu("the engine renders", False)
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


class FinalRender:
    """The final video (the Arrange's "Render video..."): saved first (the engine reads the
    scene file), the project's settings, its progress and the time left, Stop."""

    def __init__(self, session, events, media):
        self.s, self.events, self.media = session, events, media
        self.proc: Optional[subprocess.Popen] = None
        self.job: dict = {"running": False, "result": "", "out": None}

    def running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def where(self, first: Optional[int] = None, last: Optional[int] = None) -> str:
        s = self.s
        start = 0.0 if first else render_settings_of(s.proj)["start_time"]
        return shown_path(output_path(s.audio, s.scene, first, last, start))

    def start(self, first: Optional[int] = None, last: Optional[int] = None, busy: bool = False) -> dict:
        if self.running() or busy:
            return {"error": "the engine is busy with another render: Stop it first"}
        s = self.s
        with s.lock:
            if s.grid is None or s.doc is None or s.audio is None:
                return {"error": "the project is not loaded yet"}
            saved = None
            if s.dirty:
                saved = s.save()["path"]
                self.events.log("in", f"saved {saved} first: the engine reads the scene file")
            rs = render_settings_of(s.proj)
            db = s.bars()
            if first:
                first, last = max(1, int(first)), min(len(db) - 1, int(last or first))
                start, bars = db[first - 1], last - first + 1
            else:
                first = last = None
                start, bars = rs["start_time"], 0
            out = output_path(s.audio, s.scene, first, last, rs["start_time"] if not first else 0.0)
            cmd = build_cmd(s.proj, s.audio, s.scene, start, bars, rs["resolution"], out, final=True)
        self.events.log("out", " ".join(Path(c).name if i < 2 else c for i, c in enumerate(cmd)))
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", bufsize=1,
                                     env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.job = {"running": True, "done": 0, "total": 0, "out": str(out), "result": "", "line": "",
                    "started": time.time(), "first": first, "last": last}
        self.events.gpu("the final render", True)
        self.events.emit("final", {k: v for k, v in self.job.items()})
        threading.Thread(target=self._watch, args=(self.proc,), daemon=True).start()
        return {"ok": True, "out": shown_path(out), "saved": saved}

    def _watch(self, proc) -> None:
        tail: list = []
        last_emit = 0.0
        for line in proc.stdout:
            line = line.rstrip()
            tail = (tail + [line])[-12:]
            m = FRAME_LINE.match(line)
            if m:
                self.job.update(done=int(m.group(1)), total=int(m.group(2)))
            else:
                f = FRAMES_OF.search(line)
                if f and not self.job["total"]:
                    self.job["total"] = int(f.group(1))
                if line.strip():
                    self.job["line"] = line.strip()
                    self.events.log("sys", line)
            if time.time() - last_emit > 0.4:
                last_emit = time.time()
                self.events.emit("final", dict(self.job))
        code = proc.wait()
        self.events.gpu("the final render", False)
        stopped = self.job.get("result") == "stopped"
        self.job.update(running=False, result="stopped" if stopped else "done" if code == 0 and Path(self.job["out"]).is_file()
                        else f"failed (exit {code})", tail=tail if code else [])
        if self.job["result"] == "done":
            self.job["url"] = self.media.url(self.job["out"])
            self.events.log("in", f"rendered {shown_path(Path(self.job['out']))}")
        elif not stopped:
            self.events.log("err", f"the render failed (exit {code}):")
            for t in tail:
                self.events.log("err", t)
        self.events.emit("final", dict(self.job))

    def stop(self) -> dict:
        if self.running():
            self.job["result"] = "stopped"
            self.proc.kill()
            return {"ok": True}
        return {"ok": False}

    def open(self, what: str = "video") -> dict:
        """The rendered video in its player, or its folder with it chosen."""
        out = self.job.get("out")
        if not out or not Path(out).is_file():
            return {"error": "nothing rendered yet"}
        if sys.platform != "win32":
            return {"error": "opening files needs Windows here"}
        if what == "folder":
            subprocess.Popen(["explorer", "/select,", str(Path(out))])
        else:
            os.startfile(str(Path(out)))          # noqa: S606 (the user's own video)
        return {"ok": True}

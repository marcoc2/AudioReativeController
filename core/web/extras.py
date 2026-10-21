"""The catalog's demos and the repaint (ComfyUI), as the Dear PyGui windows do them.

Demos (core/demos.py): picking a Generator or FX shows what it is (about, what it needs, its
file, its parameters) and its demo, a 10 s loop made by ``python -m core.demos render --only
KEY --force`` (on the GPU; a few seconds, minutes for the 3D fractals), with its state: ready,
stale (the layer changed since), rendering, missing, failed.

Repaint (repaint.py): every frame of a render repainted through a ComfyUI workflow, its
settings in the project's ``repaint:``; "Check" says whether ComfyUI is up, its queue and the
GPU; Stop kills it and takes our jobs out of ComfyUI's queue (the frames done stay: run it
again to go on). A finished render becomes the video to repaint.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Optional

from core.web import ROOT
from core.web.util import num, resolve, shown_path

REPAINT_DEFAULTS = {"video": "", "start": 0.0, "workflow": "reference/workflow_api_rundiff.json", "scene": "(none)",
                    "prompt": "", "fps": 12.0, "seconds": 0.0}
REPAINT_LINE = re.compile(r"^\s+(\d+)/(\d+)\s")


class Demos:
    def __init__(self, media, events):
        self.media, self.events = media, events
        self.making: dict = {}            # key -> Popen

    def entry(self, key: str) -> dict:
        from core import catalog, demos
        e = catalog.by_key(key)
        if e is None:
            return {"error": f"{key}: not in the catalog"}
        proc = self.making.get(key)
        state = "rendering" if proc is not None and proc.poll() is None else demos.status(e)
        video = demos.video_path(key)
        info = demos.read_info(key) or {}
        return {"key": e.key, "label": e.label, "kind": "FX" if e.kind == catalog.FX else "Generator",
                "category": e.category, "about": e.about, "file": e.file or "", "takes_input": bool(e.takes_input),
                "needs": list(e.needs),
                "params": [{"name": p.name, "default": p.default if isinstance(p.default, (int, float, str, bool)) or p.default is None
                            else str(p.default), "about": p.about, "knob": p.knob} for p in e.params],
                "demo": {"state": state, "url": self.media.url(video, fresh=True) if video.is_file() and state in ("ready", "stale", "rendering") else None,
                         "error": info.get("error") if state == "failed" else None}}

    def make(self, key: str) -> dict:
        from core import catalog
        e = catalog.by_key(key)
        if e is None:
            return {"error": f"{key}: not in the catalog"}
        proc = self.making.get(key)
        if proc is not None and proc.poll() is None:
            return {"ok": True, "already": True}
        proc = subprocess.Popen([sys.executable, "-m", "core.demos", "render", "--only", key, "--force"], cwd=ROOT,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.making[key] = proc
        self.events.gpu(f"the demo of {e.label}", True)
        self.events.log("out", f"making the demo of {e.label} (a few seconds; minutes for the 3D fractals)")

        def watch():
            for line in proc.stdout:
                if line.strip():
                    self.events.log("sys", line.rstrip())
            code = proc.wait()
            self.events.gpu(f"the demo of {e.label}", False)
            self.events.log("in" if code == 0 else "err", f"the demo of {e.label}: " + ("made" if code == 0 else f"failed (exit {code})"))
            self.events.emit("demo", {"key": key})
        threading.Thread(target=watch, daemon=True, name="arc-demo").start()
        return {"ok": True}


class Repaint:
    """The repaint of a rendered video through ComfyUI (one at a time)."""

    def __init__(self, session, events, media):
        self.s, self.events, self.media = session, events, media
        self.proc: Optional[subprocess.Popen] = None
        self.job: dict = {"running": False, "result": ""}

    def settings(self) -> dict:
        r = {**REPAINT_DEFAULTS, **(self.s.proj.get("repaint") or {})}
        workflows = sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "reference").glob("*.json"))
        scenes = ["(none)"] + sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "examples").glob("*.yaml")
                                     if re.search(r"^repaint:", p.read_text(encoding="utf-8", errors="replace"), re.M))
        return {"video": str(r["video"] or ""), "start": float(r["start"] or 0.0), "workflow": str(r["workflow"] or ""),
                "scene": str(r["scene"] or "(none)"), "prompt": str(r["prompt"] or ""), "fps": float(r["fps"] or 12.0),
                "seconds": float(r["seconds"] or 0.0), "workflows": workflows, "scenes": scenes,
                "output": shown_path(self.output()) if r["video"] else "", "job": dict(self.job)}

    def set(self, changes: dict) -> dict:
        from core.web.project import write_project
        clean = {}
        for k, v in (changes or {}).items():
            if k not in REPAINT_DEFAULTS:
                return {"error": f"{k}: not a repaint setting"}
            if k in ("start", "fps", "seconds"):
                v = float(v or 0.0)
                if k == "fps" and not 0 < v <= 60:
                    return {"error": "fps: above 0, at most 60"}
            elif k == "video" and v and not resolve(v).is_file():
                return {"error": f"no video at {v}"}
            elif k == "video" and v:
                v = shown_path(resolve(v))
            clean[k] = v
        with self.s.lock:
            self.s.proj["repaint"] = {**REPAINT_DEFAULTS, **(self.s.proj.get("repaint") or {}), **clean}
            if self.s.proj_path is not None:
                write_project(self.s.proj_path, {f"repaint.{k}": v for k, v in clean.items()})
        return self.settings()

    def output(self) -> Path:
        v = resolve((self.s.proj.get("repaint") or {}).get("video") or "x.mp4")
        return v.with_name(f"{v.stem}_repaint.mp4")

    def check(self) -> dict:
        """ComfyUI up or not, its queue (ours among it), the GPU."""
        from core.video.h3_client import ComfyH3Client
        from core.video.repaint import queue_status
        from core.web.render import gpu
        client = ComfyH3Client()
        url = client.server_url
        if not client.is_available():
            return {"up": False, "text": f"ComfyUI is not up at {url}", "gpu": gpu()}
        q = queue_status(client)
        g = gpu()
        text = f"ComfyUI ok: {q['running']} running, {q['pending']} waiting ({q['ours']} ours)"
        if g:
            text += f" · GPU {g['util']}% · {g['used_mb'] / 1024:.1f}/{g['total_mb'] / 1024:.1f} GB"
        return {"up": True, "queue": q, "gpu": g, "text": text}

    def start(self) -> dict:
        if self.proc is not None and self.proc.poll() is None:
            return {"error": "the repaint is running already"}
        r = self.settings()
        if not r["video"] or not resolve(r["video"]).is_file():
            return {"error": "repaint: choose a video (or render one first)"}
        if not r["workflow"] and r["scene"] == "(none)":
            return {"error": "repaint: choose a workflow, or a scene with repaint:"}
        p = self.s.proj
        cmd = [sys.executable, "repaint.py", "--video", str(resolve(r["video"]))]
        if r["scene"] != "(none)":
            cmd += ["--scene", r["scene"]]
        if r["workflow"]:
            cmd += ["--workflow", r["workflow"]]
        if r["prompt"].strip():
            cmd += ["--prompt", r["prompt"].strip()]
        if p.get("midi"):
            cmd += ["--midi", str(resolve(p["midi"]))]
            if float(p.get("midi_offset") or 0.0):
                cmd += ["--midi-offset", str(p["midi_offset"])]
            if str(p.get("meter") or "").strip():
                cmd += ["--meter", str(p["meter"]).strip()]
        if r["start"] > 0:
            cmd += ["--start-time", f"{r['start']:.3f}"]
        cmd += ["--fps", f"{r['fps']:g}"]
        if r["seconds"] > 0:
            cmd += ["--seconds", f"{r['seconds']:g}"]
        out = self.output()
        cmd += ["--output", str(out)]
        threading.Thread(target=self._run, args=(cmd, out), daemon=True, name="arc-repaint").start()
        return {"ok": True}

    def _run(self, cmd: list, out: Path) -> None:
        report = self.check()
        self.events.log("sys", report["text"])
        if not report["up"]:
            self.job = {"running": False, "result": "ComfyUI is not up"}
            self.events.emit("repaint", dict(self.job))
            return
        q = report.get("queue") or {}
        if q.get("running") or q.get("pending"):
            self.events.log("warn", "other jobs are in ComfyUI's queue: the repaint waits for them")
        self.events.log("out", " ".join(Path(c).name if c.endswith(".py") else c for c in cmd))
        self.proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                     encoding="utf-8", errors="replace", env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.job = {"running": True, "done": 0, "total": 0, "started": time.time(), "out": str(out), "result": "", "line": ""}
        self.events.gpu("the repaint (ComfyUI)", True)
        self.events.emit("repaint", dict(self.job))
        for line in self.proc.stdout:
            line = line.rstrip()
            if not line.strip():
                continue
            m = REPAINT_LINE.match(line)
            if m:
                self.job.update(done=int(m.group(1)), total=int(m.group(2)))
            else:
                self.job["line"] = line.strip()
            self.events.log("sys", line)
            self.events.emit("repaint", dict(self.job))
        code = self.proc.wait()
        self.events.gpu("the repaint (ComfyUI)", False)
        stopped = self.job.get("result") == "stopped"
        self.job.update(running=False, result="stopped" if stopped else "done" if code == 0 else f"failed (exit {code})")
        if self.job["result"] == "done" and out.is_file():
            self.job["url"] = self.media.url(out, fresh=True)
            self.events.log("in", f"repainted: {shown_path(out)}")
        self.events.emit("repaint", dict(self.job))

    def stop(self) -> dict:
        from core.video.h3_client import ComfyH3Client
        from core.video.repaint import cancel_jobs
        if self.proc is not None and self.proc.poll() is None:
            self.job["result"] = "stopped"
            self.proc.kill()
        try:
            n = cancel_jobs(ComfyH3Client())
            text = f"repaint stopped: {n} of our jobs taken out of ComfyUI's queue; the frames done stay (run it again to go on)"
        except Exception as exc:
            text = f"repaint stopped; ComfyUI's queue could not be cleared: {exc}"
        self.events.log("warn", text)
        return {"ok": True, "text": text}

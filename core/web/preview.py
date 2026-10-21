"""The exact preview: the chosen bars (at most eight) made in this process with the same
stack a render uses (the studio's ``_clip_preview_frames``): the clip composer, brought to
the first bar by a dry run of the montage from the song's start (no frame decoded), under
the arrangement as it is now (saved or not), with the features the load already made.
The frames go to ffmpeg as they come and the page plays the video in step with the song.

It needs the session's clip library probed and its audio features; until then it waits
(queued), and a project without them falls back to the engine's subprocess. A new preview
cancels the one being made.
"""
from __future__ import annotations

import copy
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

from core.web import WORK
from core.web.project import render_settings_of
from core.web.util import preview_res, resolve

MAX_BARS = 8


class Cancelled(Exception):
    pass


class ExactPreview:
    def __init__(self, session, loader, events, media, engine):
        self.s, self.loader, self.events, self.media, self.engine = session, loader, events, media, engine
        self.thread: Optional[threading.Thread] = None
        self.cancel = False
        self.queued: Optional[tuple] = None
        self.n = 0

    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def ready(self) -> Optional[str]:
        """Why it cannot run in this process yet (None: it can)."""
        s = self.s
        if s.grid is None or s.doc is None:
            return "the project is loading"
        if s.features_at is None:
            return "the audio features are loading"
        uses_clips = any(sp.get("source", "clips") == "clips" for sp in s.specs())
        if uses_clips and s.clips_dir() is None:
            return "the scene uses clips: the project needs its clips folder"
        if uses_clips and s.lib is None:
            return "the clips are being probed"
        return None

    def start(self, first: int, last: int) -> dict:
        n = len(self.s.bars()) - 1 if self.s.grid is not None else 0
        first = max(1, int(first))
        last = min(n, max(first, int(last)), first + MAX_BARS - 1)
        why = self.ready()
        if why and "loading" not in why and "probed" not in why:
            return self.engine.start(first, last)              # the subprocess makes it instead
        if why or self.busy():
            self.queued = (first, last)
            if self.busy():
                self.cancel = True
            self.events.emit("engine", {"event": "queued", "why": why or "the one before is stopping", "first": first,
                                        "last": last})
            self._wait_then_start()
            return {"ok": True, "queued": True, "why": why}
        self._run(first, last)
        return {"ok": True, "first": first, "last": last}

    def _wait_then_start(self) -> None:
        if getattr(self, "_waiter", None) is not None and self._waiter.is_alive():
            return

        def wait():
            while self.queued is not None:
                if not self.busy() and self.ready() is None:
                    first, last = self.queued
                    self.queued = None
                    self._run(first, last)
                    return
                time.sleep(0.3)
        self._waiter = threading.Thread(target=wait, daemon=True, name="arc-preview-wait")
        self._waiter.start()

    def stop(self) -> dict:
        self.queued = None
        if self.busy():
            self.cancel = True
            return {"ok": True}
        return self.engine.stop()

    def _run(self, first: int, last: int) -> None:
        self.cancel = False
        self.n += 1
        self.thread = threading.Thread(target=self._make, args=(first, last, self.n), daemon=True, name="arc-preview")
        self.thread.start()

    def _video_cfg(self) -> dict:
        """The scene's video settings as the render resolves them (the studio's
        ``_effective_video_cfg``), with the arrangement as it is now."""
        from core.scene_doc import plain
        s = self.s
        with s.lock:
            cfg = copy.deepcopy(s.video)
            cfg["layers"] = plain(s.specs())
            clips = s.proj.get("clips") or {}
            if clips.get("order") and clips["order"] != "(scene)":
                cfg["clip_order"] = clips["order"]
            g = (clips.get("gravity") or {})
            if g.get("enabled"):
                for spec in (cfg.get("triggers") or {}).values():
                    if isinstance(spec, dict) and "gravity" in spec:
                        spec["gravity"] = {**dict(spec["gravity"]), **{k: float(g[k]) for k in
                                                                       ("peak", "floor", "radius", "curve") if k in g}}
            if "seed" not in cfg and clips.get("seed") is not None:
                cfg["seed"] = clips["seed"]
            if clips.get("overrides"):
                cfg["overrides"] = {**(cfg.get("overrides") or {}), **{int(k): v for k, v in clips["overrides"].items()}}
        return cfg

    def _make(self, first: int, last: int, n: int) -> None:
        from core import gpu as core_gpu
        from core.video.clip_library import ClipLibrary
        from core.video.composer import ClipComposer
        from core.video.layers import build_compositor
        s = self.s
        out = WORK / f"exact_{n % 4}.mp4"
        proc = None
        self.events.gpu("the exact preview", True)
        try:
            with s.lock:
                db = s.bars()
                fps, grid, notes, features_at = s.fps, s.grid, list(s.notes), s.features_at
                start_t, end_t = db[first - 1], db[last]
                rs = render_settings_of(s.proj)
                clips_dir, meta = s.clips_dir(), s.lib
            w, h = (int(x) for x in preview_res(s.proj.get("render") or {}).split("x"))
            n_frames = max(1, int((end_t - start_t) * fps))
            cfg = self._video_cfg()
            layers = cfg.get("layers")
            uses_clips = not layers or any((l or {}).get("source", "clips") == "clips" for l in layers)
            comp = None
            self._tell("loading the clips" if uses_clips else "building the layers", 0, n_frames, first, last)
            if uses_clips:
                lib = ClipLibrary(str(clips_dir), w, h, fps, cache_size=4, fit=rs["clip_fit"], turn_wide=rs["turn_wide"])
                comp = ClipComposer(lib, grid, notes, cfg)
                song_t0 = float(grid.start_offset or 0.0)
                if meta is not None and start_t > song_t0 and [Path(x).name for x in meta.paths] == \
                        [Path(x).name for x in lib.paths]:
                    dry = ClipComposer(meta, grid, notes, cfg)    # the montage from the song's start
                    for fi in range(int(round((start_t - song_t0) * fps))):
                        if fi % 100 == 0:
                            self._check()
                        dry.frame_at(song_t0 + fi / fps)
                    comp.take_state(dry)
                else:
                    comp.seek(start_t)
            stack = build_compositor(comp, cfg, notes, w, h, fps=fps, features_at=features_at, grid=grid,
                                     masks=str(resolve(rs["masks"])) if rs["masks"] else None,
                                     span=(start_t, start_t + n_frames / fps))
            out.parent.mkdir(parents=True, exist_ok=True)
            proc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                     "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "20",
                                     "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)],
                                    stdin=subprocess.PIPE, stderr=subprocess.PIPE)
            t_last = 0.0
            for fi in range(n_frames):
                self._check()
                frame = stack.frame_at(start_t + fi / fps)
                proc.stdin.write(frame[:, :, :3].astype("uint8").tobytes())
                if time.time() - t_last > 0.25:
                    t_last = time.time()
                    self._tell("making", fi + 1, n_frames, first, last)
            proc.stdin.close()
            if proc.wait() != 0:
                raise RuntimeError(f"ffmpeg: {proc.stderr.read().decode(errors='replace')[-300:]}")
            self.events.emit("engine", {"event": "done", "url": self.media.url(out), "first": first, "last": last,
                                        "path": str(out), "exact": True})
        except Cancelled:
            self.events.emit("engine", {"event": "stopped", "first": first, "last": last})
        except Exception as exc:
            import traceback
            self.events.log("trace", traceback.format_exc())
            self.events.emit("engine", {"event": "failed", "code": None, "tail": [f"{type(exc).__name__}: {exc}"]})
        finally:
            if proc is not None and proc.poll() is None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
                proc.kill()
            core_gpu.release()
            self.events.gpu("the exact preview", False)

    def _check(self) -> None:
        if self.cancel:
            raise Cancelled()

    def _tell(self, what: str, done: int, of: int, first: int, last: int) -> None:
        self.events.emit("engine", {"event": "progress", "frame": done, "of": of, "what": what, "first": first,
                                    "last": last})

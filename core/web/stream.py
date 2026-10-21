"""The live preview: the engine itself, played as it is made. From the playhead, the same
stack the exact preview and the render use (``preview.build_stack``) makes the frames a few
seconds ahead of the page, each one a JPEG the page asks for by its number
(``/stream/<gen>/<frame>.jpg``). What the page shows is what the render makes; when the
engine cannot keep up, the page waits for it (the song holds) instead of showing anything
else.

A stream starts at a second of the song (``start``); asked again for a second it has made,
or is about to, it goes on (a loop played again reuses its frames). An edit starts a new
one (``force``). The page asking for frames is what moves it: it makes at most ``AHEAD``
seconds past the last one asked for, and stops after ``IDLE`` seconds without being asked.
"""
from __future__ import annotations

import threading
import time
import traceback
from typing import Optional

AHEAD = 2.5          # seconds made past the last frame the page asked for
KEEP = 60.0          # seconds of frames kept behind it (a loop played again reuses them)
WINDOW = 30.0        # the song's seconds one stack is built for (SAM2 cuts out only those)
IDLE = 120.0         # seconds without a request: the stream ends (its frames stay)
QUALITY = 88


class Cancelled(Exception):
    pass


class EngineStream:
    def __init__(self, session, events):
        self.s, self.events = session, events
        self.lock = threading.Lock()
        self.wake = threading.Event()
        self.gen = 0
        self.t0 = 0.0
        self.fps = 30
        self.size = (0, 0)
        self.frames: dict = {}
        self.made = -1               # the last frame made (they are made in order)
        self.want = 0                # the last frame the page asked for
        self.asked_at = 0.0
        self.error: Optional[str] = None
        self.what = ""               # what it is doing (starting: loading the clips, ...)
        self.ended = False           # it made the song's last frame
        self.thread: Optional[threading.Thread] = None

    def alive(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def info(self) -> dict:
        return {"gen": self.gen, "t0": self.t0, "fps": self.fps, "made": self.made, "error": self.error,
                "what": self.what, "ended": self.ended}

    def start(self, t: float, force: bool = False) -> dict:
        """The stream for second ``t``: this one when it has made it, or will soon (``force``:
        a new one anyway, after an edit)."""
        s = self.s
        if s.grid is None or s.doc is None:
            return {"error": "the project is loading"}
        t = max(0.0, float(t))
        with self.lock:
            fi = int(round((t - self.t0) * self.fps))
            same = (not force and self.gen and self.error is None and self.fps == int(s.fps)
                    and t >= self.t0 - 1e-6)
            coming = self.alive() and self.made - self.fps * KEEP <= fi <= self.made + self.fps * AHEAD * 2
            if same and (fi in self.frames or coming):
                self.want, self.asked_at = fi, time.time()
                self.wake.set()
                return self.info()
            self.gen += 1
            self.t0, self.fps = t, int(s.fps)
            self.frames, self.made, self.want, self.asked_at = {}, -1, 0, time.time()
            self.error, self.what, self.ended = None, "starting", False
            gen = self.gen
        self.wake.set()                                   # the old one sees it is old
        self.thread = threading.Thread(target=self._run, args=(gen, t), daemon=True, name="arc-stream")
        self.thread.start()
        return self.info()

    def stop(self) -> None:
        with self.lock:
            self.gen += 1
        self.wake.set()

    def frame(self, gen: int, fi: int):
        """(the JPEG of frame ``fi`` or None, the stream's state); None, None for an old stream."""
        with self.lock:
            if gen != self.gen:
                return None, None
            self.want, self.asked_at = int(fi), time.time()
            got = self.frames.get(int(fi))
            state = self.info()
            resume = got is None and not self.alive() and not self.ended and self.error is None
        if resume:                                         # it stopped idle: on from its last frame
            self.thread = threading.Thread(target=self._run, args=(gen, self.t0, self.made + 1), daemon=True,
                                           name="arc-stream")
            self.thread.start()
        self.wake.set()
        return got, state

    # -- the thread ------------------------------------------------------------------------
    def _check(self, gen: int) -> None:
        if gen != self.gen:
            raise Cancelled()

    def _run(self, gen: int, t0: float, fi: int = 0) -> None:
        import cv2
        from core import gpu as core_gpu
        from core.web.preview import build_stack
        s = self.s
        try:
            while True:                                    # the project still loading: wait for it
                why = self._not_ready()
                if why is None:
                    break
                if "loading" not in why and "probed" not in why:
                    raise RuntimeError(why)
                self.what = why
                self._check(gen)
                time.sleep(0.3)
            dur = float(s.dur or 0.0)
            fps = self.fps
            while t0 + fi / fps < dur - 1e-6:
                seg_start = t0 + fi / fps
                seg_end = min(dur, seg_start + WINDOW)
                stack, _, w, h = build_stack(s, seg_start, seg_end, lambda: self._check(gen),
                                             lambda what: setattr(self, "what", what))
                self.size, self.what = (w, h), "playing"
                while t0 + fi / fps < seg_end - 1e-6:
                    while fi > self.want + AHEAD * fps:    # far enough ahead: wait for the page
                        self._check(gen)
                        if time.time() - self.asked_at > IDLE:
                            return                         # idle: a request takes it on (frame)
                        self.wake.wait(0.25)
                        self.wake.clear()
                    self._check(gen)
                    frame = stack.frame_at(t0 + fi / fps)
                    ok, jpg = cv2.imencode(".jpg", frame[:, :, 2::-1].astype("uint8"),
                                           [cv2.IMWRITE_JPEG_QUALITY, QUALITY])
                    with self.lock:
                        if gen != self.gen:
                            raise Cancelled()
                        self.frames[fi] = jpg.tobytes() if ok else b""
                        self.made = fi
                        for old in [k for k in self.frames if k < fi - KEEP * fps]:
                            del self.frames[old]
                    fi += 1
            with self.lock:
                if gen == self.gen:
                    self.ended, self.what = True, "the song's end"
        except Cancelled:
            pass
        except Exception as exc:
            self.events.log("trace", traceback.format_exc())
            with self.lock:
                if gen == self.gen:
                    self.error = f"{type(exc).__name__}: {exc}"
            self.events.emit("stream", {"gen": gen, "error": self.error})
        finally:
            core_gpu.release()

    def _not_ready(self) -> Optional[str]:
        s = self.s
        if s.grid is None or s.doc is None:
            return "the project is loading"
        if s.features_at is None:
            return "the audio features are loading"
        uses_clips = any(sp.get("source", "clips") == "clips" for sp in s.specs())
        if uses_clips and s.clips_dir() is None:
            return "the scene uses clips: the project needs its clips folder (Render video…)"
        if uses_clips and s.lib is None:
            return "the clips are being probed"
        return None

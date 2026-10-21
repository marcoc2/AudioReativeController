"""Opening a project, in steps the page shows as they go.

The window is up at once; each step says waiting / running / done / failed / skipped, how
far it is when it can tell, how long it took and why it failed. The lanes are drawn (and
edited) as soon as the scene and the bars are known (step 3); the song, the stems, the
features, the cues and the montage follow, the slow ones side by side. A failed step says
why and can be tried again; the others go on when they can. Cancel stops what is left.

Events (``Events.emit``): ``load`` {gen, step, state, done, of, message, seconds} for each
change of a step, ``opened`` with the whole state once the lanes can be drawn, ``state``
when the lanes change (the cues are read), ``montage``, ``knobs`` (the features came: the
page asks its knobs again), ``covers`` {clip url: cover url}.
"""
from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import yaml

from core.web.session import Session
from core.web.util import cue_key, resolve, shown_path

STEPS = [
    ("project", "The project file"),
    ("scene", "The scene"),
    ("bars", "MIDI and bars"),
    ("song", "The song"),
    ("stems", "Stems"),
    ("features", "Audio features"),
    ("cues", "REAPER cues"),
    ("clips", "The clips and the montage"),
    ("covers", "Clip covers"),
]
NAMES = dict(STEPS)
COVERS_MAX = 400


class Cancelled(BaseException):
    """A newer opening (or Cancel) made this one stale; BaseException, so the ``except
    Exception`` inside the extractor's stem separation lets it through."""


class Loader:
    def __init__(self, session: Session, events, media):
        self.s, self.events, self.media = session, events, media
        self.gen = 0
        self.steps: dict = {}
        self.source = None                  # what was opened: a project path, or a project dict
        self.extractor = None
        self.skip_separation = False        # --skip-separation: no AI separation, whatever the project says
        self._stopped: set = set()          # openings cancelled: what still runs of them stops
        self._lock = threading.Lock()

    # -- the steps' states ---------------------------------------------------------------
    def view(self) -> list:
        with self._lock:
            return [dict(self.steps.get(k) or {"step": k, "state": "waiting"}, title=t) for k, t in STEPS]

    def _set(self, gen: int, step: str, state: str, done=None, of=None, message: str = "", force: bool = False) -> None:
        if gen != self.gen or (gen in self._stopped and not force):
            raise Cancelled()
        with self._lock:
            st = self.steps.setdefault(step, {"step": step, "state": "waiting"})
            if state == "running" and st.get("state") != "running":
                st["t0"] = time.time()
            st.update(state=state, done=done, of=of, message=message)
            if state in ("done", "failed", "skipped"):
                st["seconds"] = round(time.time() - st.get("t0", time.time()), 2)
            elif "t0" in st:
                st["seconds"] = round(time.time() - st["t0"], 2)
            out = {k: v for k, v in st.items() if k != "t0"}
        self.events.emit("load", {"gen": gen, **out, "title": NAMES[step]})

    def _run(self, gen: int, step: str, fn: Callable, *args) -> bool:
        """Run one step; a failure is told (the step says why), never raised."""
        try:
            self._set(gen, step, "running")
            fn(gen, *args)
            if (self.steps.get(step) or {}).get("state") == "running":
                self._set(gen, step, "done", message=(self.steps[step].get("message") or ""))
            return True
        except Cancelled:
            return False
        except Exception as exc:
            traceback.print_exc()
            self.events.log("err", f"{NAMES[step]}: {type(exc).__name__}: {exc}")
            self.events.log("trace", traceback.format_exc())
            try:
                self._set(gen, step, "failed", message=f"{type(exc).__name__}: {exc}")
            except Cancelled:
                pass
            return False

    # -- opening ---------------------------------------------------------------------------
    def start(self, source) -> int:
        """Open ``source`` (a project file's path, or a project as a dict); returns the
        opening's number, which every event of it carries."""
        with self._lock:
            self.gen += 1
            gen = self.gen
            self.steps = {}
            self.source = source
        self.s.reset()
        for k, _ in STEPS:
            self._set(gen, k, "waiting")
        threading.Thread(target=self._open, args=(gen, source), daemon=True, name="arc-load").start()
        return gen

    def cancel(self) -> None:
        """Stop what is left of the opening; what was read so far stays and can be edited."""
        with self._lock:
            gen = self.gen
            self._stopped.add(gen)
            steps = [k for k, st in self.steps.items() if st.get("state") in ("waiting", "running")]
            for k in steps:
                self.steps[k].update(state="skipped", message="cancelled")
        for k in steps:
            self.events.emit("load", {"gen": gen, "step": k, "state": "skipped", "message": "cancelled", "title": NAMES[k]})
        self.events.log("warn", "loading cancelled: what was read so far can be edited")

    def retry(self, step: str) -> None:
        """Run a failed (or skipped) step again, and what depends on it."""
        gen = self.gen
        self._stopped.discard(gen)
        chain = {"project": None, "scene": None, "bars": None, "song": ["song", "stems", "features"],
                 "stems": ["stems", "features"], "features": ["stems", "features"], "cues": ["cues"],
                 "clips": ["clips", "covers"], "covers": ["covers"]}[step]
        if chain is None:
            self.start(self.source)
            return
        threading.Thread(target=self._later, args=(gen, chain), daemon=True, name="arc-retry").start()

    def reload(self, steps: list) -> None:
        """Run again what a change of the project's inputs needs (scene, bars, song, features)."""
        gen = self.gen
        self._stopped.discard(gen)

        def run():
            if "song" in steps:                              # a new song: everything after the project file
                self.start(self.source if not isinstance(self.source, dict) else dict(self.s.proj))
                return
            if "scene" in steps:
                self._run(gen, "scene", self._scene)
            if "bars" in steps:
                self._run(gen, "bars", self._bars)
            if "scene" in steps or "bars" in steps:
                self.events.emit("opened", self.full_state())
                self.s.montage_later()
            if "features" in steps:
                self._later(gen, ["stems", "features"])
        threading.Thread(target=run, daemon=True, name="arc-reload").start()

    def _open(self, gen: int, source) -> None:
        if not self._run(gen, "project", self._project, source):
            return self._skip_rest(gen, "project")
        scene_ok = self._run(gen, "scene", self._scene)
        if not self._run(gen, "bars", self._bars):
            return self._skip_rest(gen, "bars")
        if gen != self.gen:
            return
        self.events.emit("opened", self.full_state())
        for label, file in self.s.missing_clips():
            self.events.log("warn", f"{label}: its clip {file} is not on the disk (the block shows it; the render fails there)")
        if not scene_ok:
            self.events.log("warn", "the scene could not be read: the lanes stay empty until it is fixed (Reload)")
        self._later(gen, ["song", "stems", "features", "cues", "clips", "covers"])

    def _later(self, gen: int, steps: list) -> None:
        """The steps after the bars, side by side where they can be."""
        jobs = []
        if "song" in steps or "stems" in steps:
            def song_and_features():
                if "song" in steps and not self._run(gen, "song", self._song):
                    for k in ("stems", "features"):
                        self._skip(gen, k, "needs the song")
                    return
                self._run(gen, "features", self._features)
            jobs.append(song_and_features)
        if "cues" in steps:
            jobs.append(lambda: self._run(gen, "cues", self._cues))
        if "clips" in steps:
            def clips_and_covers():
                if self._run(gen, "clips", self._clips) and "covers" in steps:
                    self._run(gen, "covers", self._covers)
                elif "covers" in steps:
                    self._skip(gen, "covers", "needs the clips")
            jobs.append(clips_and_covers)
        elif "covers" in steps:
            jobs.append(lambda: self._run(gen, "covers", self._covers))
        for job in jobs:
            threading.Thread(target=job, daemon=True, name="arc-load-step").start()

    def _skip(self, gen: int, step: str, why: str) -> None:
        try:
            self._set(gen, step, "skipped", message=why)
        except Cancelled:
            pass

    def _skip_rest(self, gen: int, after: str) -> None:
        keys = [k for k, _ in STEPS]
        for k in keys[keys.index(after) + 1:]:
            self._skip(gen, k, f"needs {NAMES[after].lower()}")

    def full_state(self) -> dict:
        s = self.s
        with s.lock:
            return {**s.state(), "project": s.project_info(), "song": s.song(), "tracks": s.tracks(),
                    "catalog": s.catalog(), "clips": s.clips(), "knob_hz": 30, "load": self.view()}

    # -- each step -----------------------------------------------------------------------
    def _project(self, gen: int, source) -> None:
        if isinstance(source, dict):
            proj, path = dict(source), None
        else:
            path = resolve(source)
            if not path.is_file():
                raise FileNotFoundError(f"no project at {shown_path(path)}")
            proj = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            if not isinstance(proj, dict):
                raise ValueError(f"{path.name} is not a project (a YAML mapping)")
        if not proj.get("audio") and not proj.get("midi"):
            raise ValueError("the project names neither an audio file nor a MIDI file")
        self.s.set_project(proj, path)
        self._set(gen, "project", "running", message=shown_path(path) if path else "from the command line")

    def _scene(self, gen: int) -> None:
        self.s.load_scene()
        with self.s.lock:
            n = len(self.s.items)
            msg = f"{shown_path(self.s.scene)}: {n} layers" + (" (flat: grouped into BG / FG / master)"
                                                               if self.s.converted else "")
        self._set(gen, "scene", "running", message=msg)

    def _bars(self, gen: int) -> None:
        from core.rhythm.midi_reader import parse_meter_changes, read_midi, shift_in_time
        s = self.s
        proj = s.proj
        meter = str(proj.get("meter") or "").strip()
        dur = 0.0
        if s.audio is not None and s.audio.is_file():
            dur = _audio_seconds(s.audio)
        if proj.get("midi"):
            midi = resolve(proj["midi"])
            if not midi.is_file():
                raise FileNotFoundError(f"no MIDI at {shown_path(midi)}")
            grid, notes = read_midi(midi, fps=s.fps, meter_changes=parse_meter_changes(meter) if meter else None)
            shift_in_time(grid, notes, float(proj.get("midi_offset") or 0.0))
            how = f"{len(notes)} notes"
        else:
            from core.rhythm.analyzer import analyze_file
            if s.audio is None or not s.audio.is_file():
                raise FileNotFoundError("no MIDI, and no song to find the beat in")
            self._set(gen, "bars", "running", message="no MIDI: finding the beat in the song")
            grid, notes = analyze_file(str(s.audio), fps=s.fps), []
            how = "from the song's beat (no MIDI)"
        if not dur:
            ends = [float(n.time) + float(getattr(n, "duration", 0.0) or 0.0) for n in notes]
            dur = max(ends + [float(grid.downbeats[-1]) + float(grid.bar_duration) if len(grid.downbeats) else 0.0])
        s.set_bars(grid, notes, dur)
        self._set(gen, "bars", "running", message=f"{len(s.bars()) - 1} bars, {how}")

    def _song(self, gen: int) -> None:
        s = self.s
        if s.audio is None:
            raise FileNotFoundError("the project names no audio file")
        if not s.audio.is_file():
            raise FileNotFoundError(f"no song at {shown_path(s.audio)}")
        self._set(gen, "song", "running", message=f"{s.audio.name}: decoding in the page")
        self.events.emit("song", {"gen": gen, "url": self.media.url(s.audio)})
        # the page decodes it and tells (page_step): the step is done then

    def _features(self, gen: int) -> None:
        from core.feature_extractor import AudioFeatureExtractor
        s = self.s
        proj = s.proj
        prebuilt = {k: str(resolve(v)) for k, v in (proj.get("stems") or {}).items() if k and v}
        skip = bool(proj.get("skip_separation")) or self.skip_separation

        def progress(stage, done, of, message=""):
            step = "stems" if stage == "stems" else "features"
            state = (self.steps.get(step) or {}).get("state")
            if stage == "stems" and message.startswith("separating with"):
                self.events.gpu("stem separation", True)
            elif stage == "stems" and done >= of:
                self.events.gpu("stem separation", False)
            if stage == "stems" and done >= of and message.endswith("with sound"):
                self._set(gen, "stems", "done", message=message)
            elif state != "done":
                self._set(gen, step, "running", done=done, of=of, message=message)
        self._set(gen, "stems", "running", message="waiting for the spectrum")
        try:
            ex = AudioFeatureExtractor(str(s.audio), fps=s.fps, prebuilt_stems=prebuilt or None,
                                       skip_separation=skip, progress=progress)
        finally:
            self.events.gpu("stem separation", False)
        if gen != self.gen or gen in self._stopped:
            raise Cancelled()
        s.set_features(lambda t: ex.get_features_at_time(t, apply_gate=False))
        s.stem_names = sorted(ex.stems_energy)
        self.extractor = ex
        self._set(gen, "features", "running", message=f"{ex.duration:.1f} s of features")
        self.events.emit("knobs", {"gen": gen})

    def _cues(self, gen: int) -> None:
        from core.video.footage import read_cues
        cued = self.s.cued_specs()
        if not cued:
            self._set(gen, "cues", "skipped", message="no cued layers")
            return
        bad = 0
        for i, spec in enumerate(cued):
            self._set(gen, "cues", "running", done=i, of=len(cued), message=Path(str(spec.get("file") or "")).name)
            try:
                got = [tuple(float(x) for x in c[:4]) for c in read_cues(spec.get("cues"), spec.get("file"))]
            except Exception as exc:                      # told, not fatal: the lane still edits
                got, bad = {"error": f"{type(exc).__name__}: {exc}"}, bad + 1
                self.events.log("warn", f"cues of {spec.get('file')}: {exc}")
            self.s.set_cues(cue_key(spec), got)
            self.events.emit("state", self.s.state())
        self._set(gen, "cues", "running", message=f"{len(cued)} cued layers" + (f", {bad} not read" if bad else ""))

    def _clips(self, gen: int) -> None:
        from core.video.resolver import MetaLibrary
        s = self.s
        d = s.clips_dir()
        if d is None:
            self._set(gen, "clips", "skipped", message="the project has no clips folder")
            return
        lib = MetaLibrary(str(d), s.fps, progress=lambda done, of: self._set(gen, "clips", "running", done=done,
                                                                             of=of, message="probing the clips"))
        with s.lock:
            s.lib = lib
        self._set(gen, "clips", "running", message="resolving the montage")
        out = s.resolve_montage()
        if out is not None:
            self.events.emit("montage", out)
        self._set(gen, "clips", "running", message=f"{len(out or [])} cuts from {len(lib)} clips")

    def _covers(self, gen: int) -> None:
        from core import arrangement as arr
        clips = [c["path"] for c in self.s.clips()][:COVERS_MAX]
        if not clips:
            self._set(gen, "covers", "skipped", message="no clips")
            return
        batch: dict = {}
        for i, path in enumerate(clips):
            got = arr.cover(path)
            if got is not None:
                batch[self.media.url(path)] = self.media.url(got)
            if len(batch) >= 12 or i == len(clips) - 1:
                self.events.emit("covers", batch)
                batch = {}
                self._set(gen, "covers", "running", done=i + 1, of=len(clips))
        self._set(gen, "covers", "running", message=f"{len(clips)} clips")
        used = [c for c in clips] + [str(resolve(f)) for f in self.s.clip_files()]
        n = self.s.proxies.check(used)
        if n:
            self.events.log("sys", f"{n} clips the page cannot play as they are: making a small H.264 proxy of each "
                                   "(the render reads the clips themselves)")

    def page_step(self, step: str, state: str, message: str = "") -> None:
        """What the page did for a step (the song decoded, or not)."""
        if step == "song":
            try:
                self._set(self.gen, step, state, message=message, force=True)   # what the page did stands
            except Cancelled:
                pass


def _audio_seconds(path: Path) -> float:
    """The song's length without decoding it (ffprobe), 0 when unknown."""
    import subprocess
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                              str(path)], capture_output=True, text=True, timeout=20)
        return float(out.stdout.strip() or 0.0)
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0

"""What the page calls: ``window.pywebview.api.<name>(...)`` in the window, ``POST /api/<name>``
in a browser. Each call runs on its own thread and always answers: a Python exception comes
back as ``{error, where}`` (never a promise left hanging) and its traceback goes to the log.
"""
from __future__ import annotations

import functools
import json
import traceback
from pathlib import Path
from typing import Optional

import yaml

from core.web import ROOT
from core.web.load import Loader
from core.web.extras import Demos, Repaint
from core.web.preview import ExactPreview
from core.web.project import checked_render, render_settings_of, write_project
from core.web.render import EngineRender, FinalRender, gpu
from core.web.server import Events, Media
from core.web.session import Session, resolve, shown_path

PREFS = ROOT / "render_output" / ".arrange.json"     # shared with the Arrange window (its clips folder)
RECENT_MAX = 10


def _guarded(fn):
    @functools.wraps(fn)
    def call(self, *args):
        try:
            return fn(self, *args)
        except Exception as exc:
            tb = traceback.extract_tb(exc.__traceback__)
            last = next((f for f in reversed(tb) if str(ROOT) in f.filename), tb[-1] if tb else None)
            where = f"{shown_path(Path(last.filename))}:{last.lineno} in {last.name}" if last else fn.__name__
            self._events.log("err", f"{fn.__name__}: {type(exc).__name__}: {exc} ({where})")
            self._events.log("trace", traceback.format_exc())
            return {"error": f"{type(exc).__name__}: {exc}", "where": where}
    return call


def prefs() -> dict:
    try:
        got = json.loads(PREFS.read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def remember(**values) -> None:
    got = prefs()
    got.update(values)
    try:
        PREFS.parent.mkdir(parents=True, exist_ok=True)
        PREFS.write_text(json.dumps(got, indent=1), encoding="utf-8")
    except OSError:
        pass


class Api:
    """The page's calls. Attributes are private so pywebview exposes only the methods."""

    def __init__(self, first=None, browser: bool = False, skip_separation: bool = False, stems: Optional[dict] = None):
        self._media = Media()
        self._events = Events()
        self._session = Session(self._media, self._events)
        self._loader = Loader(self._session, self._events, self._media)
        self._engine = EngineRender(self._session, self._events, self._media)
        self._final = FinalRender(self._session, self._events, self._media, on_done=self._render_done)
        self._demos = Demos(self._media, self._events)
        self._repaint = Repaint(self._session, self._events, self._media)
        self._preview = ExactPreview(self._session, self._loader, self._events, self._media, self._engine)
        self._preview.final = self._final
        self._loader.skip_separation = skip_separation
        self._loader.extra_stems = dict(stems or {})
        self._first = first                  # what to open at start: a project path or a project dict
        self._browser = browser
        self._window = None

    def _attach(self, window) -> None:
        self._window = window
        self._events.window = window
        window.events.loaded += self._watch_drops

    def _watch_drops(self) -> None:
        """Files dropped from the Explorer: WebView2 gives their full paths to a drop handler
        on the Python side (pywebview's DOM events); the page is told where they came down."""
        from webview.dom import DOMEventHandler

        def dropped(e):
            files = ((e or {}).get("dataTransfer") or {}).get("files") or []
            paths = [f.get("pywebviewFullPath") for f in files if isinstance(f, dict) and f.get("pywebviewFullPath")]
            if paths:
                self._events.emit("dropped", {"paths": paths, "x": e.get("clientX"), "y": e.get("clientY")})
        try:
            self._window.dom.document.events.drop += DOMEventHandler(dropped, prevent_default=True)
        except Exception as exc:
            self._events.log("warn", f"files dropped from the Explorer will not be heard: {exc}")

    def _close(self) -> None:
        if self._repaint.proc is not None and self._repaint.proc.poll() is None:
            self._repaint.proc.kill()
        self._preview.stop()
        self._engine.stop()
        self._final.stop()
        self._loader.cancel()

    # -- starting ------------------------------------------------------------------------
    @_guarded
    def boot(self):
        """What the start view shows, and what to open at once."""
        first = self._first
        self._first = None
        return {"first": first if isinstance(first, (str, dict)) else None, "recent": self.recent(),
                "projects": self.projects(), "in_window": not self._browser}

    @_guarded
    def projects(self):
        return [{"path": shown_path(p), "name": p.stem} for p in sorted((ROOT / "projects").glob("*.yaml"))]

    @_guarded
    def recent(self):
        out = []
        for p in prefs().get("recent_projects") or []:
            if resolve(p).is_file():
                out.append({"path": p, "name": Path(p).stem})
        return out

    @_guarded
    def open(self, source):
        """Open a project (its path, or a project as a mapping); the steps come as events."""
        self._engine.stop()
        gen = self._loader.start(source)
        if isinstance(source, str):
            path = shown_path(resolve(source))
            recent = [path] + [p for p in prefs().get("recent_projects") or [] if p != path]
            remember(recent_projects=recent[:RECENT_MAX])
        return {"gen": gen, "load": self._loader.view()}

    @_guarded
    def cancel_load(self):
        self._loader.cancel()
        return {"ok": True}

    @_guarded
    def retry(self, step: str):
        self._loader.retry(step)
        return {"ok": True}

    @_guarded
    def page_step(self, step: str, state: str, message: str = ""):
        self._loader.page_step(step, state, message)
        return {"ok": True}

    @_guarded
    def load_state(self):
        return self._loader.view()

    @_guarded
    def paths_kind(self, paths: list):
        """Which dropped paths are clips (video or image files) and which folders."""
        from core import arrangement as arr
        ps = [Path(str(p)) for p in paths or []]
        return {"clips": [str(p) for p in ps if p.is_file() and p.suffix.lower() in arr.VIDEOS + arr.IMAGES],
                "folders": [str(p) for p in ps if p.is_dir()]}

    @_guarded
    def pick_file(self, kind: str, save: bool = False, suggested: str = "", at: str = ""):
        """A native file dialog (in the window): kind is project, audio, midi, video, scene,
        masks or folder; ``at``: the value it has now (the dialog starts there)."""
        if self._window is None:
            return {"error": "file dialogs need the app window (the browser cannot hand Python a path): type the path"}
        import webview
        types = {"project": ("Projects (*.yaml)",), "scene": ("Scenes (*.yaml)",),
                 "audio": ("Audio (*.wav;*.mp3;*.flac;*.ogg;*.m4a)",), "midi": ("MIDI (*.mid;*.midi)",),
                 "video": ("Videos (*.mp4;*.mov;*.mkv;*.webm;*.avi)", "All files (*.*)")}.get(kind, ())
        start = {"project": ROOT / "projects", "scene": ROOT / "scenes", "audio": ROOT / "input", "midi": ROOT / "input",
                 "video": ROOT / "render_output", "masks": ROOT / "render_output"}.get(kind, ROOT)
        if at:
            here = resolve(at)
            here = here if here.is_dir() else here.parent
            if here.is_dir():
                start = here
        if kind == "masks":
            kind = "folder"
        mode = (webview.FOLDER_DIALOG if kind == "folder" else webview.SAVE_DIALOG if save else webview.OPEN_DIALOG)
        got = self._window.create_file_dialog(mode, directory=str(start), file_types=types,
                                              save_filename=suggested or "")
        if not got:
            return {"path": None}
        path = got if isinstance(got, str) else got[0]
        return {"path": shown_path(Path(path))}

    @_guarded
    def new_project(self, audio: str):
        """A project for an audio file, with a new scene: projects/<song>.yaml and scenes/<song>.yaml."""
        from core.scene_doc import NEW_SCENE
        song = resolve(audio)
        if not song.is_file():
            return {"error": f"no audio at {audio}"}
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in song.stem).strip("_") or "song"
        proj = ROOT / "projects" / f"{stem}.yaml"
        if proj.exists():
            return {"error": f"{shown_path(proj)} already exists: open it"}
        scene = ROOT / "scenes" / f"{stem}.yaml"
        if not scene.exists():
            scene.parent.mkdir(parents=True, exist_ok=True)
            scene.write_text(NEW_SCENE.format(name=stem), encoding="utf-8", newline="\n")
        midi = next((p for p in (song.with_suffix(".mid"), song.with_suffix(".midi")) if p.is_file()), None)
        data = {"audio": shown_path(song), "midi": shown_path(midi) if midi else "", "scene": shown_path(scene),
                "render": {"fps": 30, "resolution": "854x480"}}
        proj.parent.mkdir(parents=True, exist_ok=True)
        proj.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n")
        self._events.log("in", f"new project {shown_path(proj)} with the scene {shown_path(scene)}")
        return {"path": shown_path(proj)}

    # -- the arrangement -----------------------------------------------------------------
    @_guarded
    def state(self):
        return self._session.state()

    @_guarded
    def full_state(self):
        return self._loader.full_state()

    @_guarded
    def shader(self, key: str):
        return self._session.shader(key)

    @_guarded
    def knob(self, text):
        return self._session.knob(text)

    @_guarded
    def undo(self):
        return self._session.step(True)

    @_guarded
    def redo(self):
        return self._session.step(False)

    # -- the scene file ------------------------------------------------------------------
    @_guarded
    def save(self):
        got = self._session.save()
        self._events.log("in", f"saved {got['path']}")
        return {**got, "text": self._session.scene_text()}

    @_guarded
    def save_as(self, path: str):
        target = resolve(path)
        if target.suffix.lower() not in (".yaml", ".yml"):
            target = target.with_suffix(".yaml")
        old = self._session.project_info()["scene"]
        got = self._session.save(target)
        self._events.log("in", f"saved as {got['path']}; the project file still opens {old} until it is saved")
        return {**got, "text": self._session.scene_text(), "project": self._session.project_info()}

    @_guarded
    def reload(self):
        got = self._session.reload()
        self._events.log("in", f"read {self._session.project_info()['scene']} again")
        return got

    # -- the engine ----------------------------------------------------------------------
    @_guarded
    def gpu(self):
        return gpu() or {"error": "no NVIDIA GPU found (nvidia-smi)"}

    @_guarded
    def render(self, first: int, last: int):
        """The exact preview of these bars (in this process when it can, else the engine's)."""
        return self._preview.start(first, last)

    @_guarded
    def stop_render(self):
        return self._preview.stop()

    # -- the final video -------------------------------------------------------------------
    @_guarded
    def render_settings(self, first=None, last=None, what="song"):
        s = self._session
        with s.lock:
            got = render_settings_of(s.proj)
        return {**got, "gpu": gpu(), "where": self._final.where(first, last, what) if s.audio and s.scene else "",
                "problem": self._final.problem() if s.grid is not None else None,
                "job": {k: v for k, v in self._final.job.items() if k != "tail"}, "dirty": s.dirty,
                "choices": {"codec": ["x264", "nvenc"], "clip_fit": ["contain", "cover"],
                            "order": ["(scene)", "sequential", "random", "shuffle"]}}

    @_guarded
    def set_render(self, changes: dict):
        """The render's settings, written into the project's render: and clips:."""
        from core.web.project import set_in
        s = self._session
        try:
            keys = checked_render(changes)
        except ValueError as exc:
            return {"error": str(exc)}
        with s.lock:
            old_fps = s.fps
            for k, v in keys.items():
                set_in(s.proj, k, v)
            if s.proj_path is not None:
                write_project(s.proj_path, keys)
            s.fps = int((s.proj.get("render") or {}).get("fps") or 24)
        if s.fps != old_fps:                          # the bars, the features and the montage at the new fps
            self._events.log("in", f"fps {old_fps} -> {s.fps}: the bars and the features again")
            self._loader.reload(["bars", "features"])
        elif any(k.startswith("clips.") for k in keys):
            s.montage_later()
        return self.render_settings()

    @_guarded
    def bar_beat(self, text: str = "", playhead: Optional[float] = None):
        """Song seconds of "bar.beat" (or of the bar the playhead is in), and back."""
        from core.rhythm.bar_beat import bar_beat_to_time, time_to_bar_beat
        s = self._session
        if s.grid is None:
            return {"error": "load the project first (no bars yet)"}
        try:
            if playhead is not None:
                db = s.bars()[:-1]
                t = max([d for d in db if d <= float(playhead) + 1e-6] or [db[0]])
            else:
                t = bar_beat_to_time(text, s.grid)
        except ValueError as exc:
            return {"error": str(exc)}
        return {"seconds": round(float(t), 3), "bar_beat": time_to_bar_beat(t, s.grid)}

    @_guarded
    def render_final(self, first=None, last=None, what="song"):
        busy = self._engine.running() or self._preview.busy()
        got = self._final.start(first, last, busy=busy, what=what)
        if got.get("saved"):
            self._events.emit("state", self._session.state())
        return got

    @_guarded
    def stop_final(self):
        return self._final.stop()

    @_guarded
    def open_output(self, what: str = "video"):
        return self._final.open(what)

    def _render_done(self, out: str, start: float) -> None:
        """A finished render becomes the video to repaint (from where it starts in the song)."""
        try:
            self._repaint.set({"video": out, "start": start})
            self._events.emit("repaint", {**self._repaint.job, "settings": True})
        except Exception:
            pass

    # -- the catalog's demos, the repaint ----------------------------------------------------
    @_guarded
    def entry(self, key: str):
        return self._demos.entry(key)

    @_guarded
    def make_demo(self, key: str):
        return self._demos.make(key)

    @_guarded
    def repaint_settings(self):
        return self._repaint.settings()

    @_guarded
    def set_repaint(self, changes: dict):
        return self._repaint.set(changes)

    @_guarded
    def repaint_check(self):
        return self._repaint.check()

    @_guarded
    def repaint_start(self):
        return self._repaint.start()

    @_guarded
    def repaint_stop(self):
        return self._repaint.stop()

    # -- the project's inputs ------------------------------------------------------------
    @_guarded
    def save_project(self, changes: dict):
        got = self._session.save_project(changes)
        if "error" not in got and got.get("reload"):
            self._events.log("in", f"the project: {', '.join(got['changed'])} changed; loading "
                                   f"{', '.join(got['reload'])} again")
            self._loader.reload(got["reload"])
        return got

    @_guarded
    def new_scene(self, name: str):
        got = self._session.new_scene(name)
        if "error" not in got and got.get("reload"):
            self._loader.reload(got["reload"])
        return got

    @_guarded
    def set_resolution(self, text: str):
        got = self._session.set_resolution(text)
        if "error" not in got:
            self._events.log("in", f"the video's size: {got['resolution']}"
                             + (f" (written in {got['path']})" if got.get("path") else " (no project file to keep it)"))
        return got

    # -- for a page in a browser -----------------------------------------------------------
    def _events_since(self, seq: int) -> list:
        return self._events.since(seq)


# The arrangement's edits and questions, answered by the session as they are (core/web/edits.py).
SESSION_CALLS = ("spans", "relane", "add", "remove", "set_param", "set_layer", "pin", "move", "new_member",
                 "set_block", "delete_block", "toggle", "set_windows", "mute", "set_mutes", "add_fx", "reorder",
                 "move_fx", "copy_links", "copy_fx", "paste_links", "paste_fx", "menu", "montage_menu", "set_pins",
                 "to_bg", "put_in_montage", "video_settings", "set_video",
                 "plugin", "set_value", "set_text", "add_trigger", "rename_trigger", "drop_trigger", "set_trigger",
                 "add_gesture", "drop_item", "add_role", "set_role", "set_mask", "keypath_of",
                 "knob_open", "knob_new", "knob_check", "knob_apply", "signals", "edit_signal", "project_fields",
                 "list_clips", "clip_info", "place_clips", "clip_frame", "sam_open", "sam_check", "sam_apply", "sam_for_clip",
                 "layer_text", "set_layer_text")


def _session_call(name):
    def call(self, *args):
        return getattr(self._session, name)(*args)
    call.__name__ = name
    call.__doc__ = getattr(Session, name).__doc__
    return _guarded(call)


for _name in SESSION_CALLS:
    setattr(Api, _name, _session_call(_name))


def stems_from_args(args) -> dict:
    """--stems name=path ... as {name: path}."""
    stems = {}
    for item in getattr(args, "stems", None) or []:
        name, _, path = item.partition("=")
        if name and path:
            stems[name] = path
    return stems


def project_from_args(args) -> Optional[object]:
    """What the command line asks to open: a project file, or one made of --audio/--midi/--scene."""
    if args.project:
        return args.project
    if not (args.audio or args.midi or args.scene):
        return None
    stems = stems_from_args(args)
    return {"audio": args.audio or "", "midi": args.midi or "", "scene": args.scene or "",
            "stems": stems, "skip_separation": bool(args.skip_separation)}

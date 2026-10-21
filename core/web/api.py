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
from core.web.render import EngineRender, gpu
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

    def __init__(self, first=None, browser: bool = False, skip_separation: bool = False):
        self._media = Media()
        self._events = Events()
        self._session = Session(self._media, self._events)
        self._loader = Loader(self._session, self._events, self._media)
        self._engine = EngineRender(self._session, self._events, self._media)
        self._loader.skip_separation = skip_separation
        self._first = first                  # what to open at start: a project path or a project dict
        self._browser = browser
        self._window = None

    def _attach(self, window) -> None:
        self._window = window
        self._events.window = window

    def _close(self) -> None:
        self._engine.stop()
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
    def pick_file(self, kind: str, save: bool = False, suggested: str = ""):
        """A native file dialog (in the window): kind is project, audio, scene or folder."""
        if self._window is None:
            return {"error": "file dialogs need the app window (the browser cannot hand Python a path): type the path"}
        import webview
        types = {"project": ("Projects (*.yaml)",), "scene": ("Scenes (*.yaml)",),
                 "audio": ("Audio (*.wav;*.mp3;*.flac;*.ogg;*.m4a)",)}.get(kind, ())
        start = {"project": ROOT / "projects", "scene": ROOT / "scenes", "audio": ROOT / "input"}.get(kind, ROOT)
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
    def spans(self, ref, spans):
        return self._session.spans(ref, spans)

    @_guarded
    def relane(self, ref, lane, spans=None):
        return self._session.relane(ref, lane, spans)

    @_guarded
    def add(self, lane, what, first, last):
        return self._session.add(lane, what, first, last)

    @_guarded
    def remove(self, ref):
        return self._session.remove(ref)

    @_guarded
    def set_param(self, ref, name, value):
        return self._session.set_param(ref, name, value)

    @_guarded
    def set_layer(self, ref, key, value):
        return self._session.set_layer(ref, key, value)

    @_guarded
    def pin(self, bar, clip=None):
        return self._session.pin(bar, clip)

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
        return self._engine.start(first, last)

    @_guarded
    def stop_render(self):
        return self._engine.stop()

    # -- for a page in a browser -----------------------------------------------------------
    def _events_since(self, seq: int) -> list:
        return self._events.since(seq)


def project_from_args(args) -> Optional[object]:
    """What the command line asks to open: a project file, or one made of --audio/--midi/--scene."""
    if args.project:
        return args.project
    if not (args.audio or args.midi or args.scene):
        return None
    stems = {}
    for item in args.stems or []:
        name, _, path = item.partition("=")
        if name and path:
            stems[name] = path
    return {"audio": args.audio or "", "midi": args.midi or "", "scene": args.scene or "",
            "stems": stems, "skip_separation": bool(args.skip_separation)}

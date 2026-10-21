"""The project's inputs, as the studio's main window edits them (audio, MIDI and its offset,
the meter, the scene, the stems, the AI separation, the picture's offset against the sound),
saved into the project file that was opened with every key it had kept (a key no widget
shows stays as it was). What changed says which loading steps run again.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.web import ROOT
from core.web.util import resolve, shown_path

FIELDS = ("audio", "midi", "midi_offset", "meter", "scene", "stems", "skip_separation", "sync_offset_ms")
RELOADS = {"audio": "song", "midi": "bars", "midi_offset": "bars", "meter": "bars", "scene": "scene",
           "stems": "features", "skip_separation": "features"}


def write_project(path: Path, changes: dict) -> None:
    """``changes`` ({key: value}, dotted keys go inside: ``render.fps``) written into the
    project file round trip; None takes a key out."""
    from ruamel.yaml.comments import CommentedMap
    from core.scene_doc import SceneDoc
    doc = SceneDoc.load(str(path)) if path.is_file() else SceneDoc()
    for key, value in changes.items():
        node = doc.data
        parts = key.split(".")
        for k in parts[:-1]:
            nxt = node.get(k)
            if not isinstance(nxt, dict):
                nxt = CommentedMap()
                node[k] = nxt
            node = nxt
        if value is None:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


def set_in(d: dict, key: str, value) -> None:
    parts = key.split(".")
    for k in parts[:-1]:
        d = d.setdefault(k, {}) if isinstance(d.get(k), dict) else d.__setitem__(k, {}) or d[k]
    if value is None:
        d.pop(parts[-1], None)
    else:
        d[parts[-1]] = value


class ProjectEdits:
    """Mixed into the Session."""

    def project_fields(self) -> dict:
        with self.lock:
            p = self.proj
            return {"path": shown_path(self.proj_path) if self.proj_path else None,
                    "audio": str(p.get("audio") or ""), "midi": str(p.get("midi") or ""),
                    "midi_offset": float(p.get("midi_offset") or 0.0), "meter": str(p.get("meter") or ""),
                    "scene": str(p.get("scene") or ""), "skip_separation": bool(p.get("skip_separation")),
                    "sync_offset_ms": int(p.get("sync_offset_ms") or 0),
                    "stems": [{"name": k, "path": str(v)} for k, v in (p.get("stems") or {}).items()]}

    def save_project(self, changes: dict) -> dict:
        """The project's inputs changed (``changes``: some of FIELDS; stems as [{name, path}])
        and written; -> {fields, reload: the steps to run again}."""
        from core.rhythm.midi_reader import parse_meter_changes
        clean: dict = {}
        for key, value in (changes or {}).items():
            if key not in FIELDS:
                return {"error": f"{key}: not a project input"}
            if key in ("audio", "midi", "scene"):
                v = str(value or "").strip()
                if v and not resolve(v).is_file():
                    return {"error": f"{key}: no file at {v}"}
                if key in ("audio", "scene") and not v:
                    return {"error": f"the project needs its {key}"}
                clean[key] = shown_path(resolve(v)) if v else ""
            elif key == "midi_offset":
                clean[key] = float(value or 0.0)
            elif key == "sync_offset_ms":
                clean[key] = max(-500, min(500, int(value or 0)))
            elif key == "meter":
                v = str(value or "").strip()
                if v:
                    try:
                        parse_meter_changes(v)
                    except ValueError as exc:
                        return {"error": f"meter: {exc} (like 22:6/4,27:5/4)"}
                clean[key] = v
            elif key == "skip_separation":
                clean[key] = bool(value)
            elif key == "stems":
                rows = [r for r in (value or []) if str(r.get("name", "")).strip() and str(r.get("path", "")).strip()]
                for r in rows:
                    if not resolve(r["path"]).is_file():
                        return {"error": f"stem {r['name']}: no file at {r['path']}"}
                clean[key] = {str(r["name"]).strip(): shown_path(resolve(r["path"])) for r in rows}
        with self.lock:
            changed = [k for k, v in clean.items() if self.proj.get(k) != v and not (
                k in ("midi_offset", "sync_offset_ms") and float(self.proj.get(k) or 0) == float(v))]
            if not changed:
                return {"fields": self.project_fields(), "reload": []}
            for k in changed:
                self.proj[k] = clean[k]
            if self.proj_path is None:
                stem = Path(str(self.proj.get("audio") or "song")).stem
                self.proj_path = ROOT / "projects" / f"{stem}.yaml"
                write_project(self.proj_path, {k: v for k, v in self.proj.items() if not k.startswith("_")})
            else:
                write_project(self.proj_path, {k: clean[k] for k in changed})
            self.set_project(self.proj, self.proj_path)
            steps = sorted({RELOADS[k] for k in changed if k in RELOADS},
                           key=["scene", "bars", "song", "features"].index)
            return {"fields": self.project_fields(), "reload": steps, "changed": changed}

    def new_scene(self, name: str) -> dict:
        """A new scene from the studio's template (scenes/<name>.yaml); the project opens it."""
        from core.scene_doc import NEW_SCENE, new_scene_path
        try:
            path = ROOT / new_scene_path(name)
        except ValueError as exc:
            return {"error": str(exc)}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(NEW_SCENE.format(name=path.stem), encoding="utf-8", newline="\n")
        return self.save_project({"scene": shown_path(path)})


def render_settings_of(proj: dict) -> dict:
    """The final render's settings, from the project (``render:`` and ``clips:``)."""
    r = proj.get("render") or {}
    c = proj.get("clips") or {}
    g = c.get("gravity") or {}
    return {"fps": int(r.get("fps") or 30), "resolution": str(r.get("resolution") or "854x480"),
            "codec": str(r.get("codec") or "x264"), "start_time": float(r.get("start_time") or 0.0),
            "stems": bool(r.get("stems")), "clip_fit": str(r.get("clip_fit") or "contain"),
            "turn_wide": bool(r.get("turn_wide")), "masks": str(r.get("masks") or ""), "profile": bool(r.get("profile")),
            "clips_dir": str(c.get("dir") or ""), "order": str(c.get("order") or "(scene)"),
            "seed": c.get("seed"), "cache_size": int(c.get("cache_size") or 8),
            "gravity": {"enabled": bool(g.get("enabled")), "peak": float(g.get("peak", 3.0)),
                        "floor": float(g.get("floor", 0.3)), "radius": float(g.get("radius", 0.45)),
                        "curve": float(g.get("curve", 2.0))}}


RENDER_KEYS = {"fps": ("render.fps", int), "codec": ("render.codec", str), "start_time": ("render.start_time", float),
               "stems": ("render.stems", bool), "clip_fit": ("render.clip_fit", str), "turn_wide": ("render.turn_wide", bool),
               "masks": ("render.masks", str), "profile": ("render.profile", bool), "clips_dir": ("clips.dir", str),
               "order": ("clips.order", str), "seed": ("clips.seed", int), "cache_size": ("clips.cache_size", int),
               "gravity.enabled": ("clips.gravity.enabled", bool), "gravity.peak": ("clips.gravity.peak", float),
               "gravity.floor": ("clips.gravity.floor", float), "gravity.radius": ("clips.gravity.radius", float),
               "gravity.curve": ("clips.gravity.curve", float)}
CHOICES = {"codec": ("x264", "nvenc"), "clip_fit": ("contain", "cover"),
           "order": ("(scene)", "sequential", "random", "shuffle")}


def checked_render(changes: dict) -> dict:
    """``changes`` to the render's settings as (project key -> value); ValueError on a bad one."""
    out = {}
    for k, v in (changes or {}).items():
        if k not in RENDER_KEYS:
            raise ValueError(f"{k}: not a render setting")
        key, kind = RENDER_KEYS[k]
        if v is None or v == "":
            out[key] = None
            continue
        v = kind(v)
        if k in CHOICES and v not in CHOICES[k]:
            raise ValueError(f"{k}: one of {', '.join(CHOICES[k])}")
        if k == "fps" and not 1 <= v <= 120:
            raise ValueError("fps: 1 to 120")
        if k in ("clips_dir", "masks") and v and not resolve(v).is_dir():
            raise ValueError(f"{k}: no folder at {v}")
        out[key] = shown_path(resolve(v)) if k in ("clips_dir", "masks") and v else v
    return out

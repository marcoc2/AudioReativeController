"""The open project and its arrangement, as the page edits it.

The scene is kept as the Arrange window keeps it: one ``[node, base, spec]`` per top-level
layer (the layer's tree in the scene file, what it said when loaded, what it says now), the
layers grouped into the montage, the BG and FG lanes and the master (``arrangement.grouped``).
Each edit from the page is one call here; it changes the specs, keeps the undo history and
answers with the lanes as they now are. Save patches what changed into the scene file
(``SceneDoc.set_layers``): comments, anchors and the layout of what did not change survive.

One lock guards all of it; what takes long (a knob over the song, the montage) is worked
out from a copy taken under the lock and stored back only if nothing changed meanwhile.
"""
from __future__ import annotations

import copy
import json
import math
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import yaml

from core.web import ROOT

KNOB_HZ = 30                       # a knob's value is sampled this often over the song
UNDO_MAX = 200
PREVIEW_RES = "640x360"


def num(v) -> float:
    """A JSON-safe number (numpy, NaN and infinities made plain)."""
    v = float(v)
    return v if math.isfinite(v) else 0.0


def resolve(path, base: Path = ROOT) -> Path:
    p = Path(str(path))
    return p if p.is_absolute() else base / p


def shown_path(p: Path) -> str:
    """A path as the project files write it: relative to the repository when inside it."""
    p = Path(p)
    try:
        return p.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def yaml_text(value) -> str:
    from core.scene_doc import plain
    return yaml.safe_dump(plain(value), default_flow_style=True, sort_keys=False, width=4096).strip()


def shader_live(entry) -> bool:
    """A shader the page can run itself: one pass, reading nothing or the picture below."""
    if not entry or entry.source != "shadertoy" or not entry.file or not entry.file.endswith(".glsl"):
        return False
    from core import shadertoy
    try:
        src = shadertoy.load(resolve(entry.file))
    except Exception:
        return False
    return set(src["channels"].get("image", {}).values()) <= {"frame"}


def copy_keeping_signals(obj):
    """A deep copy whose signals (anchored parts of the scene) stay the scene's objects."""
    from core.scene_doc import _anchor_name, _walk
    memo = {id(o): o for _, o in _walk(obj) if _anchor_name(o)}
    return copy.deepcopy(obj, memo)


def comment_names(seq, first_lead=None) -> list:
    """Names for a YAML list of layers, from the scene's comments: the one at the end of a
    layer's line (``file: lion.mp4  # the lion``), else the line just above it (its first
    clause; a ``# ---- section`` line is no name). As the Arrange window reads them; the lines
    above the first layer are kept by the key holding the list (``first_lead``)."""
    import re
    out = []
    for k, node in enumerate(seq):
        eol = ""
        ca = getattr(node, "ca", None)
        for toks in (ca.items.values() if ca is not None else []):
            tok = toks[2] if isinstance(toks, list) and len(toks) > 2 else None
            line0 = (getattr(tok, "value", "") or "").partition("\n")[0]
            if "#" in line0:
                eol = line0.split("#", 1)[1].strip()
                break
        lead = seq.ca.items.get(k) if getattr(seq, "ca", None) is not None else None
        tok = lead[0] if isinstance(lead, list) and lead else None     # after a layer written on one line
        line0 = (getattr(tok, "value", "") or "").partition("\n")[0]
        if not eol and "#" in line0:
            eol = line0.split("#", 1)[1].strip()
        above = ""
        if k == 0 and first_lead:
            lead = [None, list(first_lead)]
        for tok in (lead[1] or []) if isinstance(lead, list) and len(lead) > 1 and isinstance(lead[1], list) else []:
            for line in (getattr(tok, "value", "") or "").split("\n"):
                line = line.strip().lstrip("#").strip()
                if line and not line.startswith("--"):
                    above = re.split(r"[:,;(]", line)[0].strip()
        out.append((eol or above)[:40])
    return out


def _lead_of(parent, key):
    """The comment lines above the first item of ``parent[key]`` (ruamel keeps them by the key)."""
    entry = getattr(parent, "ca", None) and parent.ca.items.get(key)
    return entry[3] if isinstance(entry, list) and len(entry) > 3 and entry[3] else None


def _name_from_comments(pairs: list, seq, first_lead=None) -> None:
    """A layer without a ``name:`` shows the one its comment gives it. ``pairs``: (spec, base)
    per item of ``seq``; the name goes in both, so only a layer that is edited writes it."""
    for (spec, base), name in zip(pairs, comment_names(seq, first_lead)):
        if (name and isinstance(spec, dict) and not spec.get("name")
                and spec.get("source") not in ("group", "clips")):
            spec["name"] = name
            if isinstance(base, dict):
                base["name"] = name


LANE_TARGETS = {"bg": ("BG", "layers"), "fg": ("FG", "layers"), "bgfx": ("BG", "fx"), "fgfx": ("FG", "fx"),
                "master": ("master",)}


class Session:
    """The open project: its song and bars, its scene, the arrangement being edited."""

    def __init__(self, media, events):
        self.media, self.events = media, events
        self.lock = threading.RLock()
        self.gen = 0                        # which opening this is (a later one makes results stale)
        self.reset()

    def reset(self) -> None:
        with self.lock:
            self.gen += 1
            self.proj: dict = {}
            self.proj_path: Optional[Path] = None
            self.fps = 30
            self.audio: Optional[Path] = None
            self.grid = None
            self.notes: list = []
            self.dur = 0.0
            self.doc = None
            self.scene: Optional[Path] = None
            self.items: list = []           # [node, base, spec] per top-level layer
            self.converted = False
            self.undo: list = []
            self.redo: list = []
            self.dirty = False
            self.saved_at = ""
            self.knobs: dict = {}
            self.features_at: Optional[Callable] = None
            self.cues: dict = {}            # cue key -> [(t0, t1)], or {"error": ...}
            self.lib = None
            self.montage: Optional[list] = None
            self.montage_gen = 0
            self.segments: list = []

    # -- what the loader fills in ------------------------------------------------------
    def set_project(self, proj: dict, path: Optional[Path]) -> None:
        with self.lock:
            self.proj = dict(proj)
            self.proj_path = path
            self.fps = int((proj.get("render") or {}).get("fps") or 30)
            self.audio = resolve(proj["audio"]) if proj.get("audio") else None
            self.scene = resolve(proj["scene"]) if proj.get("scene") else None

    def load_scene(self) -> None:
        """The scene file as the arrangement: grouped, each layer's name read from comments."""
        from core import arrangement as arr
        from core.scene_doc import SceneDoc, plain, plain_keeping_signals
        with self.lock:
            if self.scene is None:
                raise ValueError("the project names no scene")
            if not self.scene.is_file():
                raise FileNotFoundError(f"no scene at {shown_path(self.scene)}")
            doc = SceneDoc.load(str(self.scene))
            nodes = doc.layers()
            specs = [plain_keeping_signals(n) for n in nodes]
            bases = [plain(n) for n in nodes]
            if nodes:
                video = doc.data["video"]
                _name_from_comments(list(zip(specs, bases)), video["layers"], _lead_of(video, "layers"))
            for spec, base, node in zip(specs, bases, nodes):           # a lane's members and FX too
                if spec.get("source") == "group":
                    for key in ("layers", "fx"):
                        if isinstance(node.get(key), list):
                            _name_from_comments(list(zip(spec.get(key) or [], base.get(key) or [])), node[key],
                                                _lead_of(node, key))
            known = {id(s): (n, b) for s, n, b in zip(specs, nodes, bases)}
            layers, self.converted = arr.grouped(specs)
            self.doc = doc
            self.items = [[*known.get(id(s), (None, None)), s] for s in layers]
            self.undo, self.redo, self.knobs = [], [], {}
            self.dirty = self.converted

    def set_bars(self, grid, notes: list, dur: float) -> None:
        with self.lock:
            self.grid, self.notes, self.dur = grid, list(notes), float(dur)
            self.knobs = {}

    def set_features(self, features_at: Optional[Callable]) -> None:
        with self.lock:
            self.features_at = features_at
            self.knobs = {}

    def set_cues(self, key: str, cues) -> None:
        with self.lock:
            self.cues[key] = cues

    # -- reading -----------------------------------------------------------------------
    def specs(self) -> list:
        return [it[2] for it in self.items]

    def bars(self) -> list:
        """Where each bar starts, and where the last one ends; bars go on, at the last one's
        length, as long as the song does (the MIDI may stop before the audio)."""
        db = [float(x) for x in self.grid.downbeats]
        step = float(self.grid.bar_duration)
        if not db:
            db = [float(self.grid.start_offset or 0.0)]
        while db[-1] + step < self.dur - 1e-3:
            db.append(db[-1] + step)
        return db + [db[-1] + step]

    def bar_at(self, t: float) -> int:
        db = self.bars()[:-1]
        return max(1, min(len(db), sum(1 for d in db if d <= t + 1e-6)))

    def groups(self) -> dict:
        return {s.get("name"): i for i, s in enumerate(self.specs()) if s.get("source") == "group"}

    def state(self) -> dict:
        """What the page draws of the arrangement (it asks for this after every edit)."""
        with self.lock:
            if self.grid is None or self.doc is None:
                return {"lanes": [], "pins": {}, "undo": 0, "redo": 0, "montage": None, "dirty": False, "saved_at": ""}
            return {"lanes": self.lanes(), "pins": self.pins(), "undo": len(self.undo), "redo": len(self.redo),
                    "montage": self.montage, "dirty": self.dirty, "saved_at": self.saved_at}

    def song(self) -> dict:
        with self.lock:
            g = self.grid
            n = len(self.bars()) - 1
            bb = [int(x) for x in g.bar_beats.tolist()] if getattr(g, "bar_beats", None) is not None else []
            bb += [int(g.time_signature[0])] * max(0, n - len(bb))
            return {"audio_url": self.media.url(self.audio) if self.audio and self.audio.is_file() else None,
                    "duration": self.dur, "bars": self.bars(), "bar_beats": bb,
                    "beats": [num(x) for x in (g.beats if g.beats is not None else [])],
                    "bpm": num(g.bpm), "meter": f"{g.time_signature[0]}/{g.time_signature[1]}",
                    "from_midi": bool(self.proj.get("midi"))}

    def project_info(self) -> dict:
        with self.lock:
            render = self.proj.get("render") or {}
            return {"path": shown_path(self.proj_path) if self.proj_path else None,
                    "name": self.proj_path.stem if self.proj_path else (self.audio.stem if self.audio else "untitled"),
                    "scene": shown_path(self.scene) if self.scene else None,
                    "audio": self.audio.name if self.audio else None,
                    "midi": Path(str(self.proj["midi"])).name if self.proj.get("midi") else None,
                    "converted": self.converted,
                    "resolution": str(render.get("resolution") or "854x480"),
                    "clip_fit": str(render.get("clip_fit") or "contain")}

    def tracks(self) -> list:
        """The MIDI's tracks (the most played first): when each note starts, how hard."""
        by: dict = {}
        for n in self.notes:
            by.setdefault(n.track or f"channel {n.channel + 1}", []).append(n)
        out = []
        for name, ns in sorted(by.items(), key=lambda kv: -len(kv[1]))[:8]:
            out.append({"name": name, "times": [round(n.time, 4) for n in ns],
                        "vels": [n.velocity for n in ns], "pitches": [n.pitch for n in ns]})
        return out

    def catalog(self) -> list:
        from core import catalog
        out = []
        for kind in (catalog.GENERATOR, catalog.FX):
            for cat, ents in catalog.tree(kind):
                for e in ents:
                    out.append({"key": e.key, "label": e.label, "kind": "gen" if kind == catalog.GENERATOR else "fx",
                                "cat": cat, "about": e.about, "file": e.file or "", "live": shader_live(e),
                                "params": [{"name": p.name, "default": num(p.default), "about": p.about}
                                           for p in e.params if p.knob]})
        return out

    def clips_dir(self) -> Optional[Path]:
        d = (self.proj.get("clips") or {}).get("dir")
        return resolve(d) if d and resolve(d).is_dir() else None

    def clips(self) -> list:
        from core import arrangement as arr
        d = self.clips_dir()
        if d is None:
            return []
        return [{"path": str(p), "name": p.name, "url": self.media.url(p)} for p in arr.list_clips(str(d))[:300]]

    def lanes(self) -> list:
        gi = self.groups()
        specs = self.specs()
        out = []
        for lane, gname, key in (("fg", "FG", "layers"), ("fgfx", "FG", "fx"), ("bg", "BG", "layers"), ("bgfx", "BG", "fx")):
            if gname not in gi:
                continue
            for i, spec in enumerate(specs[gi[gname]].get(key) or []):
                if isinstance(spec, dict):
                    out.append(self.member(lane, [gname, key, i], spec))
        after = (max(gi.values()) + 1) if gi else 0
        for i in range(after, len(specs)):
            spec = specs[i]
            if isinstance(spec, dict) and spec.get("source", "clips") not in ("clips", "group"):
                out.append(self.member("master", ["master", i], spec))
        return out

    def member(self, lane: str, ref: list, spec: dict) -> dict:
        from core import arrangement as arr
        from core import catalog
        kind = arr.kind_of(spec)
        entry = catalog.of_layer(spec)
        cued = spec.get("cues") is not None
        own = spec.get("file")
        got = None
        if cued:
            got = self.cues.get(cue_key(spec))
            spans = [{"first": a, "last": b, "cue": True} for a, b in self.cue_bars(got)] if isinstance(got, list) else []
            cue_state = "error" if isinstance(got, dict) else "read" if got is not None else "reading"
        else:
            blocks = arr.blocks_of(spec) if kind == arr.CLIP else []
            if not blocks:
                blocks = arr.windows_of(spec)
            spans = [{"first": b.first, "last": b.last, "from": b.start, "rate": b.rate, "file": b.file,
                      "url": self.media.url(resolve(b.file or own)) if kind == arr.CLIP and (b.file or own) else None,
                      "missing": bool(kind == arr.CLIP and (b.file or own) and not resolve(b.file or own).is_file())}
                     for b in blocks]
            cue_state = None
        live = (kind == arr.CLIP and lane == "bg" and not cued and not spec.get("cutout")) or shader_live(entry)
        params = []
        if entry:
            here = (spec.get("uniforms") or {}) if entry.source == "shadertoy" else spec
            for p in entry.params:
                if not p.knob:
                    continue
                v = here.get(p.name) if isinstance(here, dict) else None
                row = {"name": p.name, "about": p.about, "default": num(p.default)}
                if v is None or isinstance(v, bool):
                    row["v"] = num(p.default)
                elif isinstance(v, (int, float)):
                    row["v"] = num(v)
                else:
                    row["knob"] = yaml_text(v)
                params.append(row)
        opacity = spec.get("opacity", 1.0)
        return {"ref": ref, "lane": lane, "kind": {"clip": "clip", "generator": "gen", "fx": "fx"}[kind],
                "label": arr.label_of(spec), "key": entry.key if entry else spec.get("source"),
                "file": own or "", "spans": spans, "always": not spans and not cued,
                "cued": cued, "cues": cue_state, "cue_error": got.get("error") if cued and isinstance(got, dict) else None,
                "enabled": spec.get("enabled", True) is not False, "live": bool(live),
                "blend": spec.get("blend", "normal"), "opacity": num(opacity) if isinstance(opacity, (int, float)) else 1.0,
                "own_fx": len(spec.get("fx") or []), "row": spec.get(arr.ROW_KEY),
                "params": params, "yaml": yaml_text({k: v for k, v in spec.items() if k not in ("bars", "blocks")})}

    def missing_clips(self) -> list:
        """(layer, file) of every clip block whose file is not on the disk."""
        with self.lock:
            return [(m["label"], s.get("file") or m["file"]) for m in self.lanes() if m["kind"] == "clip"
                    for s in m["spans"] if s.get("missing")]

    def cue_bars(self, cues: list) -> list:
        return [(self.bar_at(t0), self.bar_at(max(t0, t1 - 1e-3))) for t0, t1 in cues]

    def cued_specs(self) -> list:
        """Every cued layer, the quick ones (a cue file) first: a REAPER edit takes seconds."""
        out, seen = [], set()
        for spec in self.specs():
            for s in [spec] + list(spec.get("layers") or []) + list(spec.get("fx") or []):
                if isinstance(s, dict) and s.get("cues") is not None and cue_key(s) not in seen:
                    seen.add(cue_key(s))
                    out.append(copy.deepcopy(s))
        return sorted(out, key=lambda s: isinstance(s.get("cues"), dict))

    def pins(self) -> dict:
        for s in self.specs():
            if s.get("source", "clips") == "clips":
                return {str(k): v for k, v in (s.get("pins") or {}).items()}
        return {}

    # -- the montage (the composer's dry run) --------------------------------------------
    def montage_inputs(self):
        """What resolving the montage needs, copied under the lock: (generation, cfg)."""
        from core.scene_doc import plain
        with self.lock:
            cfg = dict(self.doc.plain_video()) if self.doc is not None else {}
            cfg["layers"] = plain(self.specs())
            clips = self.proj.get("clips") or {}
            if clips.get("order") and clips["order"] != "(scene)":
                cfg["clip_order"] = clips["order"]
            if clips.get("seed") is not None:
                cfg["seed"] = clips["seed"]
            self.montage_gen += 1
            return self.montage_gen, cfg

    def resolve_montage(self) -> Optional[list]:
        """The montage's cuts as the render will make them (pins included), or None when the
        project has no clips or the library is not probed yet."""
        from core.video.resolver import resolve_song
        if self.lib is None or self.grid is None:
            return None
        gen, cfg = self.montage_inputs()
        start = float(self.grid.start_offset or 0.0)
        segs, _, _ = resolve_song(self.lib, self.grid, self.notes, cfg, self.fps, max(0.0, start), self.dur)
        paths = list(self.lib.paths)
        out = [{"t0": num(s.t0), "t1": num(s.t1), "name": s.clip_name,
                "url": self.media.url(paths[s.clip_idx]) if 0 <= s.clip_idx < len(paths) else None}
               for s in segs]
        with self.lock:
            if gen != self.montage_gen:
                return None                  # an edit came meanwhile: its own resolve follows
            self.montage = out
            self.segments = [s.t0 for s in segs]
            self.knobs = {k: v for k, v in self.knobs.items() if "cut" not in k}
        return out

    def montage_later(self) -> None:
        """Resolve the montage again in the background (after an edit that changes it)."""
        if self.lib is None:
            return
        gen = self.gen

        def run():
            try:
                out = self.resolve_montage()
            except Exception as exc:
                self.events.log("err", f"montage: {type(exc).__name__}: {exc}")
                return
            if out is not None and gen == self.gen:
                self.events.emit("montage", out)
        threading.Thread(target=run, daemon=True).start()

    # -- shaders and knobs ---------------------------------------------------------------
    def shader(self, key: str) -> dict:
        """A one-pass shader's code for the page to compile, its knobs' defaults and which
        channels read the picture below."""
        from core import catalog, shadertoy
        entry = catalog.by_key(key)
        if not shader_live(entry):
            return {"error": f"{key}: not a one-pass shader the page can run"}
        path = resolve(entry.file)
        src = shadertoy.load(path)
        return {"code": src["image"], "file": entry.file,
                "defaults": {u["name"]: num(u["default"]) for u in shadertoy.uniforms_of(path)
                             if isinstance(u["default"], (int, float))},
                "frame_channels": [i for i, s in src["channels"].get("image", {}).items() if s == "frame"]}

    def knob(self, text) -> dict:
        """A knob's values over the song (``KNOB_HZ`` a second), from its YAML text."""
        from core.video.layers import _knob
        try:
            how = yaml.safe_load(text) if isinstance(text, str) else text
        except yaml.YAMLError as exc:
            return {"error": f"not YAML: {exc}"}
        key = json.dumps(how, sort_keys=True, default=str)
        with self.lock:
            if key in self.knobs:
                return self.knobs[key]
            gen, notes, grid, dur = self.gen, self.notes, self.grid, self.dur
            features_at, cuts = self.features_at, list(self.segments)
        if grid is None:
            return {"error": "the bars are not known yet"}
        try:
            f = _knob("web", how, notes, features_at=features_at, grid=grid,
                      cuts=(lambda to: [c for c in cuts if c <= to]) if cuts else None)
            out = {"values": [round(num(f(i / KNOB_HZ)), 4) for i in range(int(dur * KNOB_HZ) + 1)]}
        except Exception as exc:
            waiting = features_at is None and "feature" in str(exc).lower()
            out = {"error": f"{type(exc).__name__}: {exc}" + (" (the audio features are still loading)" if waiting
                                                              else " (shown at the render)")}
        with self.lock:
            if gen == self.gen:
                self.knobs[key] = out
        return out

    # -- edits (each one undoable; they answer with the lanes) -------------------------
    def _snapshot(self) -> list:
        return [[node, base, copy_keeping_signals(spec)] for node, base, spec in self.items]

    def _push(self) -> None:
        self.undo.append(self._snapshot())
        del self.undo[:-UNDO_MAX]
        self.redo.clear()
        self.dirty = True

    def _spec(self, ref: list) -> dict:
        if ref[0] == "master":
            return self.specs()[int(ref[1])]
        return self.specs()[self.groups()[ref[0]]][ref[1]][int(ref[2])]

    def _drop(self, ref: list, spec: dict) -> None:
        if ref[0] == "master":
            self.items = [it for it in self.items if it[2] is not spec]
        else:
            lst = self.specs()[self.groups()[ref[0]]][ref[1]]
            lst[:] = [s for s in lst if s is not spec]

    def _replace(self, ref: list, old: dict, new: dict) -> None:
        if ref[0] == "master":
            for it in self.items:
                if it[2] is old:
                    it[2] = new
        else:
            lst = self.specs()[self.groups()[ref[0]]][ref[1]]
            lst[:] = [new if s is old else s for s in lst]

    def _add_to(self, target: tuple, spec: dict) -> None:
        from core import arrangement as arr
        if target[0] == "master":
            self.items.append([None, None, spec])
            return
        gi = self.groups()
        if target[0] not in gi:
            at = gi.get("FG", len(self.items)) if target[0] == "BG" else gi.get("BG", -1) + 1
            self.items.insert(at, [None, None, arr.group(target[0])])
        self.specs()[self.groups()[target[0]]].setdefault(target[1], []).append(spec)

    def _edited(self, montage: bool = False) -> dict:
        if montage:
            self.montage = None
            self.montage_later()
        return self.state()

    @staticmethod
    def _set_spans(spec: dict, spans: list) -> dict:
        """A member with these blocks; each block keeps its own clip, its ``from`` and speed."""
        from core import arrangement as arr
        if arr.kind_of(spec) == arr.CLIP:
            return arr.with_blocks(spec, [arr.Block(int(s["first"]), int(s["last"]), float(s.get("from") or 0.0),
                                                    float(s.get("rate") or 1.0), s.get("file") or None)
                                          for s in spans])
        return arr.with_windows(spec, [arr.Block(int(s["first"]), int(s["last"])) for s in spans])

    def spans(self, ref: list, spans: list) -> dict:
        """A member's blocks (``[{first, last, from?, rate?, file?}]``); none removes it."""
        with self.lock:
            self._push()
            spec = self._spec(ref)
            if not spans:
                self._drop(ref, spec)
            else:
                self._replace(ref, spec, self._set_spans(spec, spans))
            return self._edited()

    def relane(self, ref: list, lane: str, spans: Optional[list] = None) -> dict:
        """A member moved to another lane (BG <-> FG, or between FX tracks), with its blocks."""
        from core import arrangement as arr
        with self.lock:
            self._push()
            spec = self._spec(ref)
            self._drop(ref, spec)
            if spans:
                spec = self._set_spans(spec, spans)
            spec.pop(arr.ROW_KEY, None)
            self._add_to(LANE_TARGETS[lane], spec)
            return self._edited()

    def add(self, lane: str, what: str, first: int, last: int) -> dict:
        """A Generator / FX (``what`` its catalog key) or a clip (``what`` its path) put in a lane."""
        from core import arrangement as arr
        from core import catalog
        with self.lock:
            entry = catalog.by_key(what)
            if entry:
                spec = arr.with_windows(entry.spec(), [arr.Block(int(first), int(last))])
            elif lane == "fg":
                spec = arr.new_foreground(what, [arr.Block(int(first), int(last))])
            else:
                spec = arr.with_blocks({"source": "footage", "role": "background", "fit": "cover", "loop": True,
                                        "file": shown_path(Path(what))}, [arr.Block(int(first), int(last))])
            self._push()
            self._add_to(LANE_TARGETS[lane], spec)
            return self._edited()

    def remove(self, ref: list) -> dict:
        with self.lock:
            self._push()
            self._drop(ref, self._spec(ref))
            return self._edited()

    def set_param(self, ref: list, name: str, value) -> dict:
        """A knob: a number, or YAML text of what in the music moves it (checked first)."""
        from core import catalog
        from core.scene_doc import flow
        from core import arrangement as arr
        with self.lock:
            spec = self._spec(ref)
            entry = catalog.of_layer(spec)
            where = f"{arr.label_of(spec)} · {name}"
            if isinstance(value, str):
                try:
                    value = yaml.safe_load(value)
                except yaml.YAMLError as exc:
                    first = str(exc).strip().splitlines()
                    return {"error": f"{where}: not YAML ({first[0] if first else exc})", "where": where}
                if not isinstance(value, (int, float)):
                    got = self.knob(value)
                    if "error" in got and "shown at the render" not in got["error"] and "still loading" not in got["error"]:
                        return {"error": f"{where}: {got['error']}", "where": where}
                    value = flow(value)
            self._push()
            holder = spec.setdefault("uniforms", {}) if entry and entry.source == "shadertoy" else spec
            holder[name] = value
            return self._edited()

    def set_layer(self, ref: list, key: str, value) -> dict:
        """blend, opacity, enabled, name."""
        if key not in ("blend", "opacity", "enabled", "name"):
            return {"error": f"{key}: not a layer setting"}
        with self.lock:
            self._push()
            spec = self._spec(ref)
            if (key == "enabled" and value) or (key == "name" and not value):
                spec.pop(key, None)
            else:
                spec[key] = value
            return self._edited()

    def pin(self, bar: int, clip: Optional[str]) -> dict:
        """Bar ``bar`` plays ``clip`` (a clip's name), "none" (nothing), or None (the montage's choice)."""
        with self.lock:
            self._push()
            montage = next((s for s in self.specs() if s.get("source", "clips") == "clips"), None)
            if montage is None:
                montage = {"source": "clips"}
                self.items.insert(0, [None, None, montage])
            pins = {int(k): v for k, v in (montage.get("pins") or {}).items()}
            if clip is None:
                pins.pop(int(bar), None)
            else:
                pins[int(bar)] = clip
            if pins:
                montage["pins"] = dict(sorted(pins.items()))
            else:
                montage.pop("pins", None)
            return self._edited(montage=True)

    def step(self, back: bool) -> dict:
        with self.lock:
            src, dst = (self.undo, self.redo) if back else (self.redo, self.undo)
            if src:
                dst.append(self._snapshot())
                self.items = src.pop()
                self.dirty = True
            return self._edited(montage=True)

    # -- saving --------------------------------------------------------------------------
    def _patched(self, doc, items: list) -> list:
        """``doc`` made to say what ``items`` say; the layers that changed tidied (bars,
        blocks, cutouts on one line). Returns the doc's layers."""
        from core.scene_doc import plain, tidy
        changed = [node is None or plain(spec) != base for node, base, spec in items]
        doc.set_layers([(node, base, copy_keeping_signals(spec)) for node, base, spec in items])
        layers = doc.layers()
        for node, ch in zip(layers, changed):
            if ch:
                tidy(node)
        return layers

    def write_copy(self, path: Path) -> None:
        """The scene as it is now written to ``path`` (for the engine), the open one untouched."""
        from core.scene_doc import SceneDoc
        with self.lock:
            memo: dict = {}
            data = copy.deepcopy(self.doc.data, memo)
            items = [[memo.get(id(node)) if node is not None else None, base, spec] for node, base, spec in self.items]
            doc = SceneDoc(data, self.doc.newline)
            self._patched(doc, items)
            path.parent.mkdir(parents=True, exist_ok=True)
            doc.save(str(path))

    def save(self, path: Optional[Path] = None) -> dict:
        """Write the scene (patched in place); then read it again, as the Arrange does, with
        the undo history moved onto what was read."""
        with self.lock:
            target = Path(path) if path else self.scene
            saved = self._patched(self.doc, self.items)
            target.parent.mkdir(parents=True, exist_ok=True)
            self.doc.save(str(target))
            history = (self.undo, self.redo)
            self.scene = target
            if path:
                self.proj["scene"] = shown_path(target)
            self.load_scene()
            now = {id(node): (self.items[i][0], self.items[i][1]) for i, node in enumerate(saved) if i < len(self.items)}
            self.undo, self.redo = ([[[*now.get(id(node), (None, None)), spec] for node, _, spec in snap] for snap in h]
                                    for h in history)
            self.dirty = False
            self.saved_at = time.strftime("%H:%M:%S")
            return {**self.state(), "path": shown_path(target)}

    def reload(self) -> dict:
        """The scene read from disk again; the undo history goes."""
        with self.lock:
            self.load_scene()
            self.saved_at = ""
            return self._edited(montage=True)

    def scene_text(self) -> str:
        with self.lock:
            return self.scene.read_text(encoding="utf-8") if self.scene and self.scene.is_file() else ""


def cue_key(spec: dict) -> str:
    """Cues are kept by what they are, not by the member's place (members come and go)."""
    return f"{spec.get('file')}|{spec.get('cues')!r}"


def preview_res(render: dict) -> str:
    """The preview's size: the project's shape, small (a portrait project stays portrait)."""
    try:
        w, h = (int(x) for x in str(render.get("resolution", "854x480")).lower().split("x"))
    except ValueError:
        return PREVIEW_RES
    return "360x640" if h > w else PREVIEW_RES


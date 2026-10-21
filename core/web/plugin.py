"""The plugin: what the chosen layer is and does, every parameter with the editor it takes,
as the Arrange window's plugin decides it (arc_arrange.py, ``_draw_plugin``).

``plugin(path)`` answers with rows the page draws; each says its widget: a number, a flag, a
colour, a list of numbers, a knob (a summary and Link / Edit / Who plays / Chords from), a
line of text (with a file or folder dialog), or a section of its own (a clip montage's
reactivity and its triggers, gestures, the cast's roles). Every write is one undo step:
``set_value`` (a value at a key path; None takes the key out: back to the default),
``set_text`` (a line of text, read as YAML unless the parameter is text), and the sections'
own calls.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import yaml

from core.web.knobs import TRIGGER_KEYS, _flow
from core.web.util import copy_keeping_signals, num

SCORES = ("harmony",)
PULSES = ("zoom_pulse", "invert")
CLIP_ACTIONS = ("next_clip", "random_clip", "restart", "reverse")
CLIP_ORDERS = ("sequential", "shuffle", "random")
CLIP_ENDS = ("bounce", "loop")
GRAVITY = {"peak": 3.0, "floor": 0.3, "radius": 0.45, "curve": 2.0}
NEW_TRIGGER = {"notes": [36], "actions": ["next_clip"]}
PICK_DIR = ("_dir", "dir", "folder", "_folder", "masks")
PICK_FILE = ("file", "_file", "path", "_path", "image", "video", "audio", "lut", "voice")
WHO = ("largest", "second", "third", "fourth", "leftmost", "rightmost", "topmost", "bottommost", "all", "background")
ROLE_KEYS = ("who", "plays", "act", "envelope", "amount", "color")
MASK_WHERE = ("inside", "outside")
SKIP_EXTRA = ("file", "uniforms", "mix", "fx", "spill", "name", "lane_row", "blocks", "cues", "cutout", "x", "y",
              "scale", "fit", "flip", "loop", "hold")


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def picks(name: str) -> Optional[str]:
    """dir / file: a parameter a dialog chooses."""
    if name in PICK_DIR or any(name.endswith(s) for s in PICK_DIR if s.startswith("_")):
        return "dir"
    if name in PICK_FILE or any(name.endswith(s) for s in PICK_FILE if s.startswith("_")):
        return "file"
    return None


def is_trigger(name: str, value) -> bool:
    from core.scene_doc import plain
    if isinstance(value, dict):
        keys = set(plain(value))
        return bool(keys & TRIGGER_KEYS) and not keys & {"hits", "feature", "lfo"}
    return value is None and name in ("trigger", "hits") + PULSES


def is_colour(value, default, name: str) -> bool:
    v = value if value is not None else default
    return (name.endswith(("color", "colour")) and isinstance(v, list) and len(v) in (3, 4)
            and all(_is_number(c) for c in v) and (any(c > 1 for c in v) or all(isinstance(c, int) for c in v)))


def gestures_of(spec: dict) -> tuple:
    """The gestures a layer's ``hits:`` rows choose from (its class's GESTURES)."""
    from core import catalog
    import importlib
    got = catalog.NATIVE.get(spec.get("source"))
    if not got:
        return ()
    try:
        cls = getattr(importlib.import_module(got[0]), got[1])
    except (ImportError, AttributeError, TypeError, IndexError):
        return ()
    return tuple(getattr(cls, "GESTURES", ()) or ())


def renamed(trs: dict, old: str, new: Optional[str]) -> dict:
    """Triggers with ``old`` renamed ``new`` (None: dropped); ``until`` and ``exclude`` of the
    others follow."""
    out = {}
    for k, v in trs.items():
        if k == old:
            if new is not None:
                out[new] = v
            continue
        if isinstance(v, dict) and (v.get("until") == old or (v.get("exclude") or {}).get("trigger") == old):
            v = dict(v)
            if v.get("until") == old:
                if new is None:
                    v.pop("until")
                else:
                    v["until"] = new
            ex = v.get("exclude")
            if isinstance(ex, dict) and ex.get("trigger") == old:
                if new is None:
                    v.pop("exclude")
                else:
                    v["exclude"] = {**ex, "trigger": new}
        out[k] = v
    return out


class Plugin:
    """Mixed into the Session."""

    # -- what the plugin shows ---------------------------------------------------------------
    def plugin(self, path) -> dict:
        from core import catalog
        from core.scene_doc import _anchor_name, plain
        with self.lock:
            spec = self.get(path)
            e = catalog.of_layer(spec)
            owner = None
            if len(path) >= 2 and str(path[-2]) == "fx":
                owner = self._label(self.get(list(path[:-2])))
            out = {"ref": list(path), "label": self._label(spec), "about": e.about if e else "",
                   "kind": "FX" if e and e.kind == catalog.FX else "Generator" if e else spec.get("source"),
                   "category": e.category if e else "", "owner": owner, "file": spec.get("file") or "",
                   "enabled": spec.get("enabled", True) is not False, "blend": spec.get("blend", "normal"),
                   "is_fx": bool(e and e.kind == catalog.FX), "sam": spec.get("source") == "sam_cutout",
                   "rows": [], "mask": self._mask_row(spec)}
            if e is None:
                return out
            params = list(e.params)
            lead = catalog.Param("mix", 1.0, "how much of it shows (1: all)") if e.kind == catalog.FX else \
                catalog.Param("opacity", 1.0, "how much of it shows (1: all)")
            known = {p.name for p in params}
            extra = [catalog.Param(k, None, "") for k in spec if k not in known and k not in catalog.COMMON_KEYS
                     and k not in SKIP_EXTRA]
            for prm in [lead] + params + extra:
                keypath = ["uniforms", prm.name] if spec.get("source") == "shadertoy" and prm.name.startswith("u_") \
                    else [prm.name]
                holder = spec
                for k in keypath[:-1]:
                    holder = holder.get(k) if isinstance(holder, dict) else None
                raw = holder.get(keypath[-1]) if isinstance(holder, dict) else None
                out["rows"].append(self._row(spec, prm, keypath, raw, _anchor_name(raw), plain(raw)))
            return out

    def _row(self, spec, prm, keypath, raw, sig, value) -> dict:
        from core.knob_forms import form_of
        name = prm.name
        row = {"name": name, "keypath": keypath, "about": prm.about, "set": value is not None,
               "default": _show(prm.default), "knob": prm.knob, "sig": sig}
        if name == "hits" and gestures_of(spec):
            return {**row, "widget": "gestures", "gestures": list(gestures_of(spec)),
                    "items": [self._who_row(h) for h in (value or []) if isinstance(h, dict)]}
        if name == "clip_reactivity":
            return {**row, "widget": "reactivity", **self._reactivity(value)}
        if name == "roles" and spec.get("source") == "cast":
            from core.video.cast import ACTS, ENVELOPE
            return {**row, "widget": "roles", "acts": list(ACTS), "who": list(WHO), "envelopes": dict(ENVELOPE),
                    "items": [self._role(r) for r in (value or []) if isinstance(r, dict)]}
        is_from = name.endswith("_from") and (value is None or isinstance(value, (str, dict)))
        row["follows"] = bool(prm.knob or is_from)
        if _is_number(value) or (value is None and _is_number(prm.default)):
            row.update(widget="number", value=num(value if value is not None else prm.default),
                       int=isinstance(prm.default, int) and not isinstance(prm.default, bool) and prm.name not in ("opacity", "mix"))
        elif isinstance(value, bool) or (value is None and isinstance(prm.default, bool)):
            row.update(widget="flag", value=bool(value if value is not None else prm.default))
        elif is_colour(value, prm.default, name):
            v = value if value is not None else prm.default or [0, 0, 0]
            row.update(widget="colour", value=[num(c) for c in v][:3])
        elif isinstance(value if value is not None else prm.default, list) and \
                0 < len(value if value is not None else prm.default) <= 4 and \
                all(_is_number(c) for c in (value if value is not None else prm.default)):
            row.update(widget="numbers", value=[num(c) for c in (value if value is not None else prm.default)])
        elif (value is None and (is_trigger(name, None) or name in SCORES)) or (value is not None and (
                sig or isinstance(value, dict) and (form_of(value) is not None or is_trigger(name, value)))):
            row.update(widget="knob", summary=_summary(value) if value is not None else "-",
                       envelope=num(value.get("envelope", 0.25)) if name in PULSES and isinstance(value, dict) else None)
        else:
            is_text = isinstance(value, str) or (value is None and isinstance(prm.default, str))
            row.update(widget="text", is_text=is_text, pick=picks(name),
                       value="" if value is None else value if isinstance(value, str) else _flow(value),
                       hint=("a folder" if picks(name) == "dir" else "a file" if picks(name) == "file" else
                             str(prm.default) if isinstance(prm.default, str) else _flow(prm.default) if prm.default is not None else "")
                       + (" (required)" if "required" in (prm.about or "") else ""))
        # the button: who plays, chords from, link, edit
        if name in SCORES:
            row["button"] = {"label": "Chords from", "mode": "score"}
        elif is_trigger(name, value):
            row["button"] = {"label": "Who plays", "mode": "trigger"}
        elif is_from and not isinstance(value, dict):
            row["button"] = {"label": "Link", "mode": "knob", "from_text": True}
        elif prm.knob or (isinstance(value, dict) and form_of(value) is not None):
            row["button"] = {"label": "Edit" if isinstance(value, dict) else "Link", "mode": "knob"}
        return row

    @staticmethod
    def _who_row(h: dict) -> dict:
        return {"who": _who(h), "gesture": h.get("gesture"), "envelope": num(h.get("envelope", 0.2))}

    @staticmethod
    def _reactivity(cfg) -> dict:
        cfg = cfg if isinstance(cfg, dict) else {}
        trs = cfg.get("triggers") or {}
        out = {"cfg": {"clip_per_bar": bool(cfg.get("clip_per_bar", True)), "bars_per_clip": int(cfg.get("bars_per_clip", 1)),
                       "clip_order": cfg.get("clip_order", "sequential"), "clip_end": cfg.get("clip_end", "bounce"),
                       "seed": cfg.get("seed")},
               "orders": list(CLIP_ORDERS), "ends": list(CLIP_ENDS), "actions": list(CLIP_ACTIONS),
               "triggers": []}
        for name, t in trs.items():
            t = t if isinstance(t, dict) else {}
            ex = t.get("exclude") if isinstance(t.get("exclude"), dict) else {}
            out["triggers"].append({"name": name, "who": _who(t), "actions": [a for a in CLIP_ACTIONS if a in (t.get("actions") or [])],
                                    "min_velocity": int(t.get("min_velocity") or 0), "until": t.get("until"),
                                    "exclude": ex.get("trigger"), "gravity": {k: num(v) for k, v in (t.get("gravity") or {}).items()}
                                    if isinstance(t.get("gravity"), dict) else None})
        return out

    @staticmethod
    def _role(r: dict) -> dict:
        who = r.get("who", "largest")
        return {"who": who if isinstance(who, str) else ", ".join(map(str, who)), "plays": _who(r.get("plays") or {}),
                "act": r.get("act", "punch"), "envelope": num(r["envelope"]) if _is_number(r.get("envelope")) else None,
                "amount": num(r["amount"]) if _is_number(r.get("amount")) else None,
                "color": r.get("color") if isinstance(r.get("color"), str) else _flow(r["color"]) if r.get("color") is not None else "",
                "extra": _flow({k: v for k, v in r.items() if k not in ROLE_KEYS}) if any(k not in ROLE_KEYS for k in r) else ""}

    @staticmethod
    def _mask_row(spec) -> dict:
        m = spec.get("mask") if isinstance(spec.get("mask"), dict) else None
        return {"on": m is not None, "track": (m or {}).get("track") or "", "where": (m or {}).get("where", "inside"),
                "objects": ", ".join(map(str, (m or {}).get("objects") or [])), "grow": num((m or {}).get("grow", 0)),
                "feather": num((m or {}).get("feather", 2)), "wheres": list(MASK_WHERE)}

    # -- writing ---------------------------------------------------------------------------
    def set_value(self, path, keypath, value) -> dict:
        """The value at ``keypath`` in the layer at ``path``; None takes the key out (back to
        its default; an emptied ``uniforms`` goes too). Editing inside a signal gives this
        place its own copy (the signal's other places stay as they were)."""
        from core.scene_doc import _anchor_name, flow
        keypath = list(keypath)
        with self.lock:
            def put(spec):
                node = spec
                for k in keypath[:-1]:
                    nxt = node[int(k)] if isinstance(node, list) else node.get(k)
                    if isinstance(nxt, dict) and _anchor_name(nxt):
                        nxt = dict(nxt)
                        if isinstance(node, list):
                            node[int(k)] = nxt
                        else:
                            node[k] = nxt
                    elif not isinstance(nxt, (dict, list)):
                        nxt = {}
                        node[k] = nxt
                    node = nxt
                last = keypath[-1]
                if value is None:
                    if isinstance(node, dict):
                        node.pop(last, None)
                        if len(keypath) == 2 and keypath[0] == "uniforms" and not node:
                            spec.pop("uniforms", None)
                elif isinstance(node, list):
                    node[int(last)] = flow(value) if isinstance(value, (dict, list)) else value
                else:
                    node[last] = flow(value) if isinstance(value, (dict, list)) else value
                return spec
            self._edit(path, put)
            return self._edited()

    def set_text(self, path, keypath, text: str, is_text: bool = False) -> dict:
        """A one-line cell: empty -> back to the default; text as it is, else read as YAML."""
        text = str(text or "").strip()
        if not text:
            return self.set_value(path, keypath, None)
        if is_text:
            return self.set_value(path, keypath, text)
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            first = str(exc).strip().splitlines()
            return {"error": f"{keypath[-1]}: not YAML ({first[0] if first else exc})"}
        return self.set_value(path, keypath, value)

    def _edit_at(self, path, keypath, fn, drop_empty: bool = True) -> dict:
        """``fn(what is at keypath, a copy)`` -> its new value (None / empty: the key goes)."""
        keypath = list(keypath)
        with self.lock:
            def put(spec):
                node = spec
                for k in keypath[:-1]:
                    nxt = node[int(k)] if isinstance(node, list) else node.get(k)
                    if not isinstance(nxt, (dict, list)):
                        nxt = {}
                        node[k] = nxt
                    node = nxt
                last = keypath[-1]
                old = node[int(last)] if isinstance(node, list) else node.get(last)
                new = fn(copy_keeping_signals(old) if old is not None else None)
                if drop_empty and new in (None, [], {}) and isinstance(node, dict):
                    node.pop(last, None)
                elif isinstance(node, list):
                    node[int(last)] = new
                else:
                    node[last] = new
                return spec
            self._edit(path, put)
            return self._edited()

    # clip reactivity's triggers
    def add_trigger(self, path, keypath) -> dict:
        """A new trigger (``kick``, ``snare``, ``trigger1``...) in the triggers at ``keypath``."""
        with self.lock:
            holder, key = self._holder(path, list(keypath))
            trs = holder.get(key) if isinstance(holder, dict) else None
            names = set(trs or {})
            name = next(n for n in ["kick", "snare"] + [f"trigger{i}" for i in range(1, 99)] if n not in names)
            got = self._edit_at(path, keypath, lambda t: {**dict(t or {}), name: dict(NEW_TRIGGER)})
            return {**got, "added": name}

    def rename_trigger(self, path, keypath, old: str, new: str) -> dict:
        new = str(new or "").strip()
        if not new or new == old:
            return self.state()
        with self.lock:
            holder, key = self._holder(path, list(keypath))
            if new in (holder.get(key) or {}):
                return {"error": f"there is a trigger {new!r} already"}
            return self._edit_at(path, keypath, lambda t: renamed(dict(t or {}), old, new))

    def drop_trigger(self, path, keypath, name: str) -> dict:
        return self._edit_at(path, keypath, lambda t: renamed(dict(t or {}), name, None) or None)

    def set_trigger(self, path, keypath, name: str, key: str, value) -> dict:
        """A trigger's actions (list), min_velocity, until, exclude (another trigger's name),
        gravity (on/off, or one of its numbers: ``gravity.peak``)."""
        def put(t):
            t = dict(t or {})
            if key == "actions":
                acts = [a for a in CLIP_ACTIONS if a in (value or [])]
                if acts:
                    t["actions"] = acts
                else:
                    t.pop("actions", None)
            elif key == "min_velocity":
                if value:
                    t["min_velocity"] = max(0, min(127, int(value)))
                else:
                    t.pop("min_velocity", None)
            elif key == "until":
                if value:
                    t["until"] = value
                else:
                    t.pop("until", None)
            elif key == "exclude":
                if value:
                    old = t.get("exclude") if isinstance(t.get("exclude"), dict) else {}
                    t["exclude"] = {"trigger": value, "window": float(old.get("window", 0.04))}
                else:
                    t.pop("exclude", None)
            elif key == "gravity":
                if value:
                    t["gravity"] = dict(GRAVITY) if not isinstance(t.get("gravity"), dict) else t["gravity"]
                else:
                    t.pop("gravity", None)
            elif key.startswith("gravity."):
                g = dict(t.get("gravity") or GRAVITY)
                g[key.split(".", 1)[1]] = float(value)
                t["gravity"] = g
            return t
        return self._edit_at(path, list(keypath) + [name], put)

    # gestures: hits rows
    def add_gesture(self, path, keypath, gesture: str) -> dict:
        got = self._edit_at(path, keypath, lambda hs: list(hs or []) + [{"notes": [36], "gesture": gesture, "envelope": 0.2}])
        holder, key = self._holder(path, list(keypath))
        return {**got, "added": len(holder.get(key) or []) - 1}

    def drop_item(self, path, keypath, i: int) -> dict:
        """Row ``i`` of the list at ``keypath`` (gestures, roles) taken out."""
        return self._edit_at(path, keypath, lambda xs: [x for j, x in enumerate(xs or []) if j != int(i)] or None)

    # the cast's roles
    def add_role(self, path, keypath) -> dict:
        got = self._edit_at(path, keypath, lambda rs: list(rs or []) + [{"who": "largest", "plays": {"notes": [36]}, "act": "punch"}])
        holder, key = self._holder(path, list(keypath))
        return {**got, "added": len(holder.get(key) or []) - 1}

    def set_role(self, path, keypath, i: int, key: str, value) -> dict:
        """A role's who (a rank or place, or ids "1, 2"), act, envelope, amount, color (r, g, b
        or chord / pitch), or its other keys as YAML (``extra``)."""
        def put(r):
            r = dict(r or {})
            if key == "who":
                v = str(value or "").strip()
                if re.fullmatch(r"[\d,\s]+", v):
                    r["who"] = [int(x) for x in v.replace(" ", "").split(",") if x]
                elif v:
                    r["who"] = v
                else:
                    r.pop("who", None)
            elif key == "act":
                r["act"] = value
            elif key in ("envelope", "amount"):
                if value is None or value == "":
                    r.pop(key, None)
                else:
                    r[key] = float(value)
            elif key == "color":
                v = str(value or "").strip()
                if not v:
                    r.pop("color", None)
                elif v in ("chord", "pitch"):
                    r["color"] = v
                else:
                    got = yaml.safe_load(v if v.startswith("[") else f"[{v}]")
                    r["color"] = [int(c) for c in got][:3]
            elif key == "extra":
                got = yaml.safe_load(value) if str(value or "").strip() else {}
                if not isinstance(got, dict):
                    raise ValueError("its other keys: 'key: value' pairs")
                r = {**{k: v for k, v in r.items() if k in ROLE_KEYS}, **got}
            return r
        try:
            return self._edit_at(path, list(keypath) + [int(i)], put, drop_empty=False)
        except (ValueError, TypeError, yaml.YAMLError) as exc:
            return {"error": f"role {int(i) + 1} · {key}: {exc}"}

    # mask: on any layer
    def set_mask(self, path, changes: dict) -> dict:
        """``mask:`` (keep the layer in or out of the figures of a mask track): on (bool),
        track, where, objects ("1, 2"), grow, feather."""
        def put(m):
            if changes.get("on") is False:
                return None
            m = dict(m or {})
            for k, v in changes.items():
                if k == "on":
                    continue
                if k == "objects":
                    ids = [int(x) for x in str(v or "").replace(" ", "").split(",") if x]
                    if ids:
                        m["objects"] = ids
                    else:
                        m.pop("objects", None)
                elif k in ("grow", "feather"):
                    m[k] = float(v)
                elif k == "track":
                    if v:
                        m["track"] = Path(str(v)).as_posix()
                    else:
                        m.pop("track", None)
                elif k == "where":
                    if v == "inside":
                        m.pop("where", None)
                    else:
                        m["where"] = v
            return m
        try:
            return self._edit_at(path, ["mask"], put, drop_empty=False) if changes.get("on") is not False else \
                self._edit_at(path, ["mask"], lambda m: None)
        except (ValueError, TypeError) as exc:
            return {"error": f"mask: {exc}"}


def _who(t) -> str:
    from core.scene_doc import _anchor_name, plain
    name = _anchor_name(t)
    if name:
        return f"*{name}"
    t = plain(t) if isinstance(t, dict) else {}
    parts = []
    for k in ("track", "notes", "audio", "score", "events"):
        if k in t:
            v = t[k]
            parts.append(f"{k} {', '.join(map(str, v)) if isinstance(v, list) else v}")
    return " · ".join(parts) or "nobody yet"


def _summary(v) -> str:
    text = _flow(v)
    return text if len(text) <= 64 else text[:61] + "..."


def _show(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return _flow(v)


class LayerText:
    """A layer's own YAML, edited as text (the fallback for anything without a widget): the
    scene's own lines when the layer is as it was loaded (its comments kept), the signals it
    follows as ``*name``; its bars and blocks stay with the lanes."""

    SKIP = ("bars", "blocks")

    def _node_of(self, path):
        """The layer's tree in the scene file (None for one added or moved since)."""
        top = self.items[self._top(path)]
        node = top[0]
        for k in self._rest(path):
            if node is None:
                return None
            if isinstance(node, list):                   # an FX added since: the file has fewer
                node = node[int(k)] if 0 <= int(k) < len(node) else None
            else:
                node = node.get(k) if hasattr(node, "get") else None
        return node

    def layer_text(self, path) -> dict:
        from core.scene_doc import plain
        with self.lock:
            spec = self.get(path)
            node = self._node_of(path)
            src = node if node is not None and plain(node) == plain(spec) else spec
            text = self.doc.text_of(src, extra=self._signal_objects(), skip=self.SKIP)
            return {"text": text}

    def set_layer_text(self, path, text: str) -> dict:
        """The layer as the text says (its bars and blocks as they are)."""
        try:
            with self.lock:
                body = self.doc.parse_text(text, extra=self._signal_objects())
        except KeyError as exc:
            return {"error": f"no signal {exc} in the scene"}
        except Exception as exc:
            first = str(exc).strip().splitlines()
            return {"error": f"not YAML: {first[0] if first else exc}"}
        if not body.get("source"):
            return {"error": "a layer needs its source:"}
        with self.lock:
            def put(spec):
                keep = {k: spec[k] for k in self.SKIP if k in spec}
                return {**dict(body), **keep}
            self._edit(path, put)
            return self._answer()

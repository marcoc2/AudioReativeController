"""The knob editor and the scene's signals, on the session (as arc_studio.py's knob editor
and signals panel do it, over core/knob_forms.py).

A knob is edited as a *state* (``knob_forms.state_of``): the form (what in the music), its
source (a track and notes, a stem, the transcription, an audio feature), its settings and
gain/bias, a nested ``from`` for ``rate``. The page draws the state from ``FORMS`` (sent
here, not copied into the page) and sends it back; Python turns it into the knob's YAML,
checks it, samples its curve and writes it.

A *signal* is an anchored part of the scene (``&kick``): defined in one place, followed by
every ``*kick``. Editing it where it is defined changes every place that follows it; a knob
can follow one ("use a signal") or become one ("name this one").
"""
from __future__ import annotations

import re
from typing import Optional

import yaml

from core.web.util import copy_keeping_signals, num, resolve

TRIGGER_KEYS = {"track", "notes", "audio", "threshold", "min_gap", "score", "events", "snap", "voice"}
NAME = re.compile(r"[A-Za-z][A-Za-z0-9_-]*$")


def _jsonable(v):
    if isinstance(v, tuple):
        return [_jsonable(x) for x in v]
    if isinstance(v, list):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    return v


def signal_kind(value) -> str:
    """trigger (who plays), score (a transcription), knob, or other."""
    from core.knob_forms import form_of
    if isinstance(value, dict) and value and set(value) <= TRIGGER_KEYS:
        return "score" if "score" in value and "events" not in value else "trigger"
    return "knob" if form_of(value) is not None else "other"


NOTE_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def _trigger_in(state: dict, spec):
    """The who-plays spec a knob's form follows (a pulse on each hit, a rate moved by hits), or None."""
    from core import knob_forms as kf
    form = (state or {}).get("form")
    if not isinstance(spec, dict) or form not in kf.FORMS:
        return None
    if kf.FORMS[form]["source"] == "trigger" and isinstance(spec.get(form), dict):
        return spec[form]
    if form == "rate" and state.get("from"):
        return _trigger_in(state["from"], spec.get("from"))
    return None


def _note_name(pitch: int) -> str:
    """A MIDI pitch as a musician reads it (60: C4)."""
    return f"{NOTE_NAMES[int(pitch) % 12]}{int(pitch) // 12 - 1}"


class KnobEditor:
    """Mixed into the Session (it brings get, items, specs, lock, _push, _edited...)."""

    # -- where a knob lives ------------------------------------------------------------
    def _holder(self, path, keypath: list, make: bool = False):
        """(the mapping holding the knob, its key) at ``keypath`` inside the layer at ``path``."""
        node = self.get(path)
        for k in keypath[:-1]:
            nxt = node[int(k)] if isinstance(node, list) else node.get(k)
            if nxt is None and make and isinstance(node, dict):
                nxt = {}
                node[k] = nxt
            node = nxt
            if node is None:
                return None, keypath[-1]
        return node, keypath[-1]

    def keypath_of(self, path, name: str) -> list:
        """A parameter's place in its layer: a shader's under ``uniforms``."""
        from core import catalog
        e = catalog.of_layer(self.get(path))
        return ["uniforms", name] if e and e.source == "shadertoy" else [name]

    # -- the signals ---------------------------------------------------------------------
    def _places(self):
        """(ref, keypath, container, lane, layer) of every container in the layers, in the
        scene's stack order (the first holds a signal's ``&name``)."""
        from core.scene_doc import occurrences
        gi = self.groups()
        after = (max(gi.values()) + 1) if gi else 0
        for i, it in enumerate(self.items):
            spec = it[2]
            if spec.get("source") == "group":
                for key in ("layers", "fx"):
                    for j, m in enumerate(spec.get(key) or []):
                        if isinstance(m, dict):
                            lane = {("BG", "layers"): "bg", ("FG", "layers"): "fg", ("BG", "fx"): "bgfx",
                                    ("FG", "fx"): "fgfx"}.get((spec.get("name"), key))
                            for p, o in occurrences(m):
                                yield [spec.get("name"), key, j], list(p), o, lane, m
            elif spec.get("source", "clips") != "clips" and i >= after:
                for p, o in occurrences(spec):
                    yield ["master", i], list(p), o, "master", spec

    def signals(self) -> list:
        """Every signal: its name, kind, value, where it is defined and who follows it."""
        from core.scene_doc import _anchor_name, anchors_in, plain
        with self.lock:
            found: dict = {}
            for ref, p, o, lane, layer in self._places():
                name = _anchor_name(o)
                if not name:
                    continue
                place = {"ref": ref, "keypath": p, "lane": lane,
                         "label": self._label(layer) + (" · " + ".".join(map(str, p)) if p else "")}
                got = found.setdefault(name, {"name": name, "obj": o, "def": None, "uses": []})
                if got["def"] is None and got["obj"] is o:
                    got["def"] = place
                else:
                    got["uses"].append(place)
            for name, o in anchors_in(self.doc.data if self.doc is not None else {}).items():
                found.setdefault(name, {"name": name, "obj": o, "def": None, "uses": []})
            out = []
            for name, s in found.items():
                value = plain(s["obj"])
                out.append({"name": name, "kind": signal_kind(value), "yaml": _flow(value)[:200],
                            "def": s["def"], "uses": s["uses"]})
            return out                                          # in the scene's stack order

    def _defines(self, path, keypath, obj) -> bool:
        """This place is the first in the stack to hold ``obj`` (what the file writes as &name)."""
        for ref, p, o, _, _ in self._places():
            if o is obj:
                return list(ref) == list(path) and list(p) == list(keypath)
        return False

    def _label(self, spec) -> str:
        from core import arrangement as arr
        return arr.label_of(spec)

    def _signal_objects(self) -> dict:
        from core.scene_doc import _anchor_name, _walk, anchors_in
        out: dict = {}
        for it in self.items:
            for _, o in _walk(it[2]):
                name = _anchor_name(o)
                if name and name not in out:
                    out[name] = o
        for name, o in anchors_in(self.doc.data if self.doc is not None else {}).items():
            out.setdefault(name, o)
        return out

    # -- the editor ------------------------------------------------------------------------
    def knob_sources(self) -> dict:
        """What a knob can read in this project: MIDI tracks, audio features (stems among
        them), the transcription."""
        with self.lock:
            tracks = sorted({n.track for n in self.notes if n.track})
            stems = sorted(getattr(self, "stem_names", None) or [])
            score = ""
            if self.audio is not None and (self.audio.parent / "sheetsage").exists():
                score = (self.audio.parent / "sheetsage").as_posix()
        from core import feature_info
        features = feature_info.names(stems)
        info = {n: feature_info.describe(n, stems) for n in features}
        return {"tracks": tracks, "features": features, "feature_info": info, "feature_groups": list(feature_info.GROUPS),
                "score": score, "midi": self.midi_tracks(), "audios": self.project_audios()}

    def midi_tracks(self) -> list:
        """Who can play in the MIDI: each track (the most played first) as a trigger names it
        (its name, or its index when it has none), its notes and how often each is played."""
        with self.lock:
            notes = list(self.notes)
        by: dict = {}
        for n in notes:
            by.setdefault(n.track_index, []).append(n)
        out = []
        for i, ns in sorted(by.items(), key=lambda kv: -len(kv[1])):
            name = ns[0].track or ""
            pitches: dict = {}
            for n in ns:
                pitches[n.pitch] = pitches.get(n.pitch, 0) + 1
            out.append({"track": name or str(i), "label": name or f"channel {ns[0].channel + 1}", "count": len(ns),
                        "pitches": [{"pitch": p, "name": _note_name(p), "count": c} for p, c in sorted(pitches.items())]})
        return out

    def project_audios(self) -> list:
        """Who can play in the project's audio: each stem's attacks, then the song's."""
        from core.web.util import shown_path
        with self.lock:
            proj, audio = dict(self.proj or {}), self.audio
        out = [{"label": str(name), "path": shown_path(resolve(path))}
               for name, path in (proj.get("stems") or {}).items() if name and path and resolve(path).is_file()]
        if audio is not None and audio.is_file():
            out.append({"label": "the song (the mix)", "path": shown_path(audio)})
        return out

    def default_hits(self) -> dict:
        """A new trigger's who plays: the most played MIDI track, else the first stem's attacks."""
        midi = self.midi_tracks()
        if midi:
            t = midi[0]["track"]
            return {"track": int(t) if t.isdigit() else t}
        audios = self.project_audios()
        return {"audio": audios[0]["path"]} if audios else {"notes": [36]}

    def onsets(self, spec: dict) -> list:
        """An audio file's attacks (core.rhythm.onset_reader), kept: the editor asks again and again."""
        from core.rhythm.onset_reader import read_onsets
        key = (str(resolve(spec["audio"])), float(spec.get("threshold", 0.3)), float(spec.get("min_gap", 0.05)))
        cache = self.__dict__.setdefault("_onsets", {})
        if key not in cache:
            cache[key] = read_onsets(key[0], threshold=key[1], min_gap=key[2])
        return cache[key]

    def knob_open(self, path, keypath, mode: str = "knob", default: float = 0.0) -> dict:
        """The editor's state for the knob at ``keypath`` in the layer at ``path`` (mode knob,
        trigger: who plays, score: where chords come from)."""
        from core import knob_forms as kf
        from core.scene_doc import _anchor_name, plain
        with self.lock:
            holder, key = self._holder(path, list(keypath))
            raw = holder.get(key) if isinstance(holder, dict) else None
            name = _anchor_name(raw)
            shared = None
            if name:
                shared = ("&" if self._defines(path, list(keypath), raw) else "*") + name
            value = plain(raw)
            if mode == "trigger":
                st = kf.state_of({"hits": value or self.default_hits()})
            elif mode == "score":
                st = kf.state_of({"harmony": value or {"score": self.knob_sources()["score"]}})
            elif value is None or isinstance(value, bool):
                st = kf.new_state("constant")
                st["main"] = float(default if value is None else value)
            else:
                st = kf.state_of(value)
            sigs = [{"name": s["name"], "kind": s["kind"], "yaml": s["yaml"]} for s in self.signals()
                    if s["kind"] in (("knob",) if mode == "knob" else (mode,)) and s["name"] != name]
            return {"mode": mode, "state": st, "shared": shared, "signals": sigs,
                    "link": name if shared and shared.startswith("*") else None,
                    "param": {"name": str(list(keypath)[-1]), **self.param_range(path, keypath)},
                    "schema": self.knob_schema(), "sources": self.knob_sources(),
                    "yaml": _flow(value) if value is not None else ""}

    @staticmethod
    def knob_schema() -> dict:
        from core import knob_forms as kf
        return {"forms": _jsonable(kf.FORMS), "choices": _jsonable(kf.choices()), "groups": list(kf.GROUPS),
                "shaping": _jsonable(kf.SHAPING), "snaps": list(kf.SNAPS), "trigger_kinds": list(kf.TRIGGER_KINDS)}

    def knob_new(self, form: str, like: Optional[dict] = None) -> dict:
        """A fresh editor state for ``form``; a new who plays is the project's own (default_hits)."""
        from core import knob_forms as kf
        st = kf.new_state(form, like)
        same = like is not None and like.get("form") in kf.FORMS and kf.FORMS[like["form"]]["source"] == kf.FORMS[form]["source"]
        if kf.FORMS[form]["source"] == "trigger" and not same:
            st["source"] = kf.state_of({"hits": self.default_hits()})["source"]
        if form == "rate" and st.get("from") is not None:        # what moves the rate: a new one, too
            st["from"]["source"] = kf.state_of({"hits": self.default_hits()})["source"]
        return st

    def _knob_spec(self, state: dict, mode: str):
        from core import knob_forms as kf
        spec = kf.spec_of(state)
        if mode == "trigger":
            return spec["hits"]
        if mode == "score":
            return spec["harmony"]
        return spec

    def knob_check(self, state: dict, mode: str = "knob", t0: Optional[float] = None, t1: Optional[float] = None,
                   n: int = 600) -> dict:
        """The YAML the state writes, whether the engine reads it, and (knob mode) its curve
        over [t0, t1] with the bars there and the clip cuts."""
        from core import knob_forms as kf
        try:
            spec = self._knob_spec(state, mode)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        out = {"yaml": _flow(spec)}
        if mode == "trigger":
            return {**out, **self._hits_of(spec)}
        if mode != "knob":
            return out
        who = _trigger_in(state, spec)                   # a form that follows hits: its hits too
        if who is not None:
            out.update(self._hits_of(who))
        with self.lock:
            notes, grid, features_at, dur = self.notes, self.grid, self.features_at, self.dur
        cuts = self.cut_times()
        if grid is None:
            return out
        t0 = 0.0 if t0 is None else max(0.0, float(t0))
        t1 = dur if t1 is None else min(dur, float(t1))
        if t1 <= t0:
            t1 = min(dur, t0 + 8.0)
        try:
            import numpy as np
            from core.video.layers import _music_knob
            f = _music_knob("knob", spec, list(notes), features_at, None, grid,
                            (lambda to: [c for c in cuts if c <= to]) if cuts else None)
            ts = np.linspace(float(t0), float(t1), int(n))
            vs = [float(f(t)) for t in ts]
            out.update(times=[round(num(t), 4) for t in ts], values=[round(num(v), 4) for v in vs],
                       bars=[num(b) for b in self.bars() if t0 <= b <= t1], cuts=[num(c) for c in cuts if t0 <= c <= t1])
        except Exception as exc:
            waiting = features_at is None and "feature" in str(exc).lower()
            out["error"] = f"{type(exc).__name__}: {exc}" + (" (the audio features are still loading)" if waiting else "")
        return out

    def _hits_of(self, spec: dict) -> dict:
        """How a who-plays spec hits over the song: how many, when, in which bars."""
        from core.video.layers import _layer_events
        with self.lock:
            notes, grid = list(self.notes), self.grid
        try:
            got = _layer_events(spec, notes, onset_loader=self.onsets, grid=grid)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "hits": 0, "hit_times": []}
        times = [round(num(n.time), 3) for n in got]
        out = {"hits": len(times), "hit_times": times[:4000]}         # (a curve's own are "times")
        if times:
            out.update(first_bar=self.bar_at(times[0]), last_bar=self.bar_at(times[-1]))
        return out

    def knob_apply(self, path, keypath, state: dict, mode: str = "knob", anchor: str = "", link: str = "") -> dict:
        """Write the knob: the state's spec (``link``: follow that signal instead; ``anchor``:
        name this one a new signal). A signal this place defines, changed, stays the signal:
        every place that follows it follows the new value. One undo step."""
        from core.scene_doc import _anchor_name, flow
        anchor, link = (anchor or "").strip(), (link or "").strip()
        sigs = self._signal_objects()
        with self.lock:
            holder, key = self._holder(path, list(keypath))
            old = holder.get(key) if isinstance(holder, dict) else None
            old_name = _anchor_name(old)
            defines = bool(old_name) and self._defines(path, list(keypath), old)
            if link:
                if link not in sigs:
                    return {"error": f"no signal {link!r}"}
                new = sigs[link]
            else:
                try:
                    spec = self._knob_spec(state, mode)
                except Exception as exc:
                    return {"error": f"{type(exc).__name__}: {exc}"}
                if mode == "trigger" and isinstance(old, dict) and set(spec) <= TRIGGER_KEYS:
                    spec = {**{k: v for k, v in dict(old).items() if k not in TRIGGER_KEYS}, **spec}
                if mode == "knob":
                    got = self.knob(spec)
                    if "error" in got and "shown at the render" not in got["error"] and "still loading" not in got["error"]:
                        return got
                    for sub, name in _source_links(state):                   # a source tied to a signal
                        if name in sigs and isinstance(spec, dict):
                            spec = _with_at(spec, sub, sigs[name])
                if not isinstance(spec, (dict, list)):
                    if anchor:
                        return {"error": "a plain number cannot be a signal: make it follow the music first"}
                    new = spec
                else:
                    new = flow(spec)
                    if anchor:
                        if not NAME.match(anchor):
                            return {"error": f"{anchor!r}: a name is a letter, then letters, digits, _ or -"}
                        if anchor in sigs:
                            return {"error": f"there is a signal {anchor!r} already"}
                        new.yaml_set_anchor(anchor, always_dump=True)
                    elif defines:                                            # it stays the signal
                        new.yaml_set_anchor(old_name, always_dump=True)
            self._push()
            spec_now = self.get(path)
            holder2, _ = self._holder_in(spec_now, list(keypath), make=True)
            holder2[key] = new
            if defines and not link and new is not old:                     # the followers follow
                from core.scene_doc import _replace_ids
                for it in self.items:
                    _replace_ids(it[2], {id(old): new})
            return self._edited()

    @staticmethod
    def _holder_in(spec, keypath: list, make: bool = False):
        node = spec
        for k in keypath[:-1]:
            nxt = node[int(k)] if isinstance(node, list) else node.get(k)
            if nxt is None and make and isinstance(node, dict):
                nxt = {}
                node[k] = nxt
            node = nxt
        return node, keypath[-1]

    def edit_signal(self, name: str) -> dict:
        """The knob editor on a signal, where it is defined."""
        s = next((x for x in self.signals() if x["name"] == name), None)
        if s is None:
            return {"error": f"no signal {name!r}"}
        if s["def"] is None:
            return {"error": f"{name} is defined outside the layers (the scene's video settings): edit it there"}
        if s["kind"] == "other":
            return {"error": f"{name} is not a knob, a trigger nor a score"}
        got = self.knob_open(s["def"]["ref"], s["def"]["keypath"], s["kind"] if s["kind"] != "knob" else "knob")
        return {**got, "ref": s["def"]["ref"], "keypath": s["def"]["keypath"], "title": f"&{name}"}


def _source_links(state: dict, prefix: tuple = ()) -> list:
    """(where in the spec, signal name) of each source the editor tied to a signal."""
    out = []
    src = state.get("source")
    if isinstance(src, dict) and src.get("link"):
        out.append((prefix + (state["form"],), src["link"]))
    if state.get("from"):
        out += _source_links(state["from"], prefix + ("from",))
    return out


def _with_at(spec, where: tuple, obj):
    node = spec
    for k in where[:-1]:
        node = node[k]
    node[where[-1]] = obj
    return spec


def _flow(value) -> str:
    from core.scene_doc import plain
    if value is None:
        return ""
    return yaml.safe_dump(plain(value), default_flow_style=True, sort_keys=False, width=4096, allow_unicode=True).strip()

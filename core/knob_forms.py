"""The knob forms (``_knob`` in core/video/layers.py) as the GUI's knob editor shows them.

A knob says what in the music moves a number: a pulse on each kick, the organ's latest
note, the chord's root, the clip cuts, the bass against its recent average. The editor
does not ask for YAML: it asks *what in the music* (a form, grouped by meaning), *from
where* (a track and its notes, an audio stem, the transcription, an audio feature) and
*how it responds* (the form's own settings, and gain/bias for all).

``state_of(spec)`` reads a knob spec (what a scene's YAML says) into an editor state and
``spec_of(state)`` writes it back, leaving out the settings at their defaults; both keep
any key the editor has no widget for. ``FORMS`` is the schema the editor draws from.
"""
from __future__ import annotations

import copy
from typing import Any, Optional

import numpy as np

# a setting: (name, default, kind, help); kind: float | int | bool | text | a tuple of choices
FORMS: dict = {
    "constant": {
        "group": "fixed", "label": "a fixed value", "source": None,
        "main": ("value", 0.0, "float", "the value"), "params": []},
    "hits": {
        "group": "beat", "label": "a pulse on each hit (1 -> 0)", "source": "trigger",
        "params": [("envelope", 0.25, "float", "seconds to fall back to 0"),
                   ("amount", 1.0, "float", "the pulse's height")]},
    "adsr": {
        "group": "beat", "label": "each note's envelope (attack, decay, sustain, release)",
        "source": "trigger",
        "params": [("attack", 0.02, "float", "seconds up"), ("decay", 0.2, "float", "seconds down to sustain"),
                   ("sustain", 0.6, "float", "the level while the note is held"),
                   ("release", 0.4, "float", "seconds away once it ends"),
                   ("velocity", True, "bool", "the note's velocity scales the peak")]},
    "before": {
        "group": "beat", "label": "rises 0 -> 1 just before each hit", "source": "trigger",
        "params": [("window", 0.5, "float", "seconds ahead of the hit it starts rising")]},
    "since": {
        "group": "beat", "label": "seconds since the latest hit", "source": "trigger",
        "params": [("nth", 0, "int", "1: since the hit before that, ...")]},
    "count": {
        "group": "beat", "label": "hits so far in the bar", "source": "trigger", "params": []},
    "random_on": {
        "group": "beat", "label": "a new random value (0..1) on each hit", "source": "trigger",
        "params": [("ease", 0.1, "float", "seconds to slide to it"), ("seed", 0, "int", "which random")]},
    "groove": {
        "group": "beat", "label": "the bar's pattern of hits (one bit a step)", "source": "trigger",
        "params": [("division", 16, "int", "steps to a whole note (16 in 4/4)"),
                   ("ago", 0, "int", "1: the bar before"),
                   ("part", "pattern", ("pattern", "steps"), "steps: how many steps the bar has")]},
    "pitch": {
        "group": "note", "label": "the latest note's pitch (low..high -> 0..1)", "source": "trigger",
        "params": [("low", 36, "float", "MIDI pitch read as 0"), ("high", 84, "float", "MIDI pitch read as 1"),
                   ("glide", 0.1, "float", "seconds to slide to a new note")]},
    "voice": {
        "group": "note", "label": "the k-th note sounding now (lowest first)", "source": "trigger",
        "params": [("k", 0, "int", "0 the lowest"),
                   ("part", "pitch", ("pitch", "level"), "pitch: its MIDI pitch; level: 1 while it sounds"),
                   ("release", 0.3, "float", "seconds the level fades after"),
                   ("velocity", False, "bool", "velocity scales the level")]},
    "harmony": {
        "group": "harmony", "label": "the chord's root as a colour (hue 0..1)", "source": "score",
        "params": [("glide", 1.2, "float", "seconds to slide round the colour wheel"),
                   ("start", 0.6, "float", "the hue before the first chord")]},
    "chord": {
        "group": "harmony", "label": "the chord sounding now: root, a tone, size, mode, tension",
        "source": "score",
        "params": [("part", "root", ("root", "tone", "size", "mode", "tension"),
                    "root 0..11 · tone: its tone-th note · size: how many notes · mode: 1 major, 0 minor"
                    " · tension 0..1"),
                   ("tone", 0, "int", "for part tone: 0 root, 1 third, 2 fifth, 3 seventh"),
                   ("glide", 0.0, "float", "seconds to slide there (0 jumps)")]},
    "cut": {
        "group": "cuts", "label": "the clip cuts (known before they come)", "source": None,
        "main": ("cut", "count", ("count", "since", "until", "phase"),
                 "count: cuts so far · since: seconds since the last · until: seconds to the next"
                 " · phase: 0 -> 1 across the shot"),
        "params": [("morph", 0.0, "float", "for count: seconds to step up")]},
    "feature": {
        "group": "audio", "label": "an audio feature (loudness of a band, a texture)", "source": "feature",
        "params": [("attack", 0.0, "float", "seconds to rise to it (0: as it is, jumpy)"),
                   ("release", 0.0, "float", "seconds to fall back")]},
    "relative": {
        "group": "audio", "label": "an audio feature against its recent average (1 = as usual)",
        "source": "feature",
        "params": [("window", 6.0, "float", "seconds of 'recently'"), ("attack", 0.05, "float", "seconds up"),
                   ("release", 0.3, "float", "seconds down")]},
    "lfo": {
        "group": "time", "label": "an oscillator in step with the song", "source": None,
        "main": ("lfo", "sine", ("sine", "tri", "saw", "square", "pulse"), "the wave"),
        "params": [("period", "1/1", "text", "a note value on the grid (1/1 a bar, 1/4 a beat) or seconds"),
                   ("phase", 0.0, "float", "0..1 of the period"),
                   ("width", 0.5, "float", "for pulse: how long it is high")]},
    "bar": {
        "group": "time", "label": "steps up at each bar (a new shape per bar, melting in)", "source": None,
        "params": [("step", 1.0, "float", "how much a bar adds"), ("offset", 0.0, "float", "the value at bar 1"),
                   ("morph", 0.35, "float", "seconds to melt into the next")]},
    "rate": {
        "group": "time", "label": "keeps adding (base + rate x from) every second", "source": None,
        "main": ("rate", 2.0, "float", "how much 'from' adds a second"),
        "params": [("base", 0.0, "float", "what it adds a second anyway")]},
    "figure": {
        "group": "figure", "label": "a figure of the mask track: where it is, how big", "source": None,
        "main": ("figure", "largest", ("largest", "second", "third"), "which, by its share of the frame"),
        "params": [("part", "x", ("x", "y", "size"), "x, y its centre (0..1), size its share"),
                   ("glide", 0.15, "float", "seconds to slide")]},
}

# the settings a new knob of each form writes out even at their defaults (what one reads
# first in a scene: the kick's envelope, the bar's step and morph)
KEEP = {"hits": ("envelope",), "adsr": ("attack", "decay", "sustain", "release"),
        "before": ("window",), "pitch": ("low", "high", "glide"), "harmony": ("glide",),
        "relative": ("window",), "lfo": ("period",), "bar": ("step", "morph"),
        "rate": ("base",), "random_on": ("ease",), "voice": ("k", "part")}

GROUPS = ("beat", "note", "harmony", "cuts", "audio", "time", "figure", "fixed")
SHAPING = (("gain", 1.0, "float", "the value is bias + gain x it"), ("bias", 0.0, "float", ""))
SNAPS = ("auto", "bar", "beat", "off")
TRIGGER_KINDS = ("track", "notes", "audio", "melody")


def choices() -> list:
    """``(form, "group · label")`` in the order the editor lists them."""
    return [(k, f"{f['group']} · {f['label']}") for g in GROUPS for k, f in FORMS.items() if f["group"] == g]


def form_of(spec: Any) -> Optional[str]:
    """Which form a knob spec is (None for one the editor cannot show)."""
    if isinstance(spec, bool):
        return None
    if isinstance(spec, (int, float)):
        return "constant"
    if isinstance(spec, str):
        return "feature"
    if isinstance(spec, dict):
        for k in FORMS:
            if k != "constant" and k in spec:
                return k
    return None


def split_tracks(text: str) -> list:
    """"kick, kick-2" -> ["kick", "kick-2"]; a comma inside brackets belongs to the name
    ("FM5 MIDI (patches 2, 5, 10)" stays whole, as DAW exports name tracks)."""
    names, cur, depth = [], "", 0
    for ch in str(text):
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if ch == "," and depth == 0:
            names.append(cur)
            cur = ""
        else:
            cur += ch
    names.append(cur)
    return [n.strip() for n in names if n.strip()]


def _trigger_state(src: Any) -> dict:
    src = dict(src) if isinstance(src, dict) else {}
    st = {"kind": "notes", "track": "", "notes": "", "audio": "", "score": "", "voice": "",
          "extra": {}}
    known = {"track", "notes", "audio", "score", "events", "voice"}
    if "audio" in src:
        st["kind"], st["audio"] = "audio", str(src["audio"])
    elif "score" in src and src.get("events") == "melody":
        st["kind"], st["score"], st["voice"] = "melody", str(src["score"]), str(src.get("voice", "") or "")
    elif "track" in src:
        t = src["track"]
        st["kind"], st["track"] = "track", ", ".join(map(str, t)) if isinstance(t, list) else str(t)
    if "notes" in src:
        st["notes"] = ", ".join(str(n) for n in src["notes"])
    st["extra"] = {k: v for k, v in src.items() if k not in known}
    if "score" in src and st["kind"] != "melody":
        st["extra"].update({k: src[k] for k in ("score", "events") if k in src})
    return st


def _trigger_spec(st: dict) -> dict:
    out: dict = {}
    kind = st.get("kind", "notes")
    notes = [int(n) for n in str(st.get("notes", "")).replace(";", ",").split(",") if n.strip().lstrip("-").isdigit()]
    if kind == "audio":
        out["audio"] = st.get("audio", "")
    elif kind == "melody":
        out["score"], out["events"] = st.get("score", ""), "melody"
        if str(st.get("voice", "")).strip():
            out["voice"] = str(st["voice"]).strip()
    elif kind == "track":                             # a track with no name: its index in the file
        names = [int(n) if n.isdigit() else n for n in split_tracks(st.get("track", ""))]
        out["track"] = names[0] if len(names) == 1 else names
    if notes and kind != "audio":
        out["notes"] = notes
    out.update(st.get("extra") or {})
    return out


def state_of(spec: Any) -> dict:
    """An editor state for a knob spec: {form, main, source, params, gain, bias, from, extra}."""
    form = form_of(spec)
    st = {"form": form or "custom", "main": None, "source": None, "audio": None, "params": {},
          "gain": 1.0, "bias": 0.0, "clip": None, "from": None, "extra": {}, "keep": [],
          "custom": spec if form is None else None}
    if form is None:
        return st
    f = FORMS[form]
    st["params"] = {p[0]: p[1] for p in f["params"]}
    if form == "constant":
        st["main"] = float(spec)
        return st
    if isinstance(spec, str):                        # "subbands.bass": the plain feature
        st["source"] = spec
        return st
    spec = dict(spec)
    used = {form, "gain", "bias", "clip"}
    st["gain"], st["bias"] = float(spec.get("gain", 1.0)), float(spec.get("bias", 0.0))
    clip = spec.get("clip")
    st["clip"] = [float(c) for c in clip] if isinstance(clip, (list, tuple)) and len(clip) == 2 else None
    if clip is not None and st["clip"] is None:
        used.discard("clip")                        # not a pair: kept as it is
    st["keep"] = [k for k in ("gain", "bias") if k in spec]
    if f["source"] == "trigger":
        st["source"] = _trigger_state(spec[form])
    elif f["source"] == "score":
        sc = spec[form] if isinstance(spec[form], dict) else {"score": spec[form]}
        st["source"] = {"score": str(sc.get("score", "")), "snap": sc.get("snap", "auto"),
                        "keep_snap": "snap" in sc,
                        "extra": {k: v for k, v in sc.items() if k not in ("score", "snap")}}
    elif f["source"] == "feature":
        st["source"] = str(spec[form])
        st["audio"] = str(spec["audio"]) if spec.get("audio") else None        # whose: a file's own, else the song's
        used.add("audio")
    elif form == "bar":
        st["params"].update(dict(spec["bar"] or {}))
        st["keep"] += list(dict(spec["bar"] or {}))
    elif "main" in f:
        st["main"] = spec[form]
    if form == "rate":
        st["from"] = state_of(spec["from"]) if "from" in spec else None
        used.add("from")
    for name, _, _, _ in f["params"]:
        if name in spec and form != "bar":
            st["params"][name] = spec[name]
            st["keep"].append(name)
            used.add(name)
    st["extra"] = {k: copy.deepcopy(v) for k, v in spec.items() if k not in used}
    return st


def spec_of(st: dict) -> Any:
    """The knob spec an editor state says (settings at their defaults left out)."""
    form = st["form"]
    if form == "custom":
        return copy.deepcopy(st.get("custom"))
    f = FORMS[form]
    if form == "constant":
        return float(st["main"])
    shaped = abs(float(st.get("gain", 1.0)) - 1.0) > 1e-9 or abs(float(st.get("bias", 0.0))) > 1e-9 or bool(st.get("clip"))
    glides = form == "feature" and any(float(st["params"].get(p[0], p[1]) or 0) != p[1] for p in f["params"])
    if form == "feature" and not shaped and not glides and not st.get("audio") and not st.get("extra") and not st.get("keep"):
        return str(st["source"])
    out: dict = {}
    if f["source"] == "trigger":
        out[form] = _trigger_spec(st["source"] or {})
    elif f["source"] == "score":
        src = st["source"] or {}
        sc = {"score": src.get("score", "")}
        if src.get("snap", "auto") != "auto" or src.get("keep_snap"):
            sc["snap"] = src.get("snap", "auto")
        sc.update(src.get("extra") or {})
        out[form] = sc
    elif f["source"] == "feature":
        out[form] = str(st["source"])
        if st.get("audio"):
            out["audio"] = str(st["audio"])
    elif form == "bar":
        defaults = {p[0]: p[1] for p in f["params"]}
        out["bar"] = {k: v for k, v in st["params"].items()
                      if v != defaults.get(k) or k in st.get("keep", ())}
    elif "main" in f:
        out[form] = st["main"]
    if form == "rate" and st.get("from") is not None:
        out["from"] = spec_of(st["from"])
    if form != "bar":
        for name, default, _, _ in f["params"]:
            v = st["params"].get(name, default)
            if v != default or name in st.get("keep", ()):
                out[name] = v
    keep = st.get("keep", ())
    if abs(float(st.get("gain", 1.0)) - 1.0) > 1e-9 or "gain" in keep:
        out["gain"] = float(st["gain"])
    if abs(float(st.get("bias", 0.0))) > 1e-9 or "bias" in keep:
        out["bias"] = float(st["bias"])
    if st.get("clip"):
        out["clip"] = [float(c) for c in st["clip"]]
    out.update(copy.deepcopy(st.get("extra") or {}))
    return out


def new_state(form: str, like: Optional[dict] = None) -> dict:
    """A fresh state for ``form``, keeping the source of ``like`` when it has the same kind."""
    f = FORMS[form]
    st = {"form": form, "main": f["main"][1] if "main" in f else None, "source": None,
          "params": {p[0]: p[1] for p in f["params"]}, "gain": 1.0, "bias": 0.0, "clip": None, "audio": None, "from": None,
          "extra": {}, "keep": list(KEEP.get(form, ())), "custom": None}
    same = like is not None and like.get("form") in FORMS and FORMS[like["form"]]["source"] == f["source"]
    if f["source"] == "trigger":
        st["source"] = copy.deepcopy(like["source"]) if same else _trigger_state({"notes": [36]})
    elif f["source"] == "score":
        st["source"] = copy.deepcopy(like["source"]) if same else {"score": "", "snap": "beat",
                                                                    "keep_snap": True, "extra": {}}
    elif f["source"] == "feature":
        st["source"] = like["source"] if same else "subbands.bass"
        st["audio"] = like.get("audio") if same else None
    if form == "feature":                            # a new one glides: raw, a feature jumps every frame
        st["params"].update(attack=0.1, release=0.5)
    if form == "constant":
        st["main"] = 0.0
    if form == "rate":
        st["from"] = new_state("hits")
    return st


def sample(spec: Any, t0: float, t1: float, n: int = 400, notes=(), features_at=None,
           grid=None, cuts=None, onset_loader=None) -> tuple:
    """The knob's value over [t0, t1]: (times, values), sampled in time order the way a
    render calls it (a knob may keep state from frame to frame)."""
    from core.video.layers import _music_knob
    f = _music_knob("knob", spec, list(notes or ()), features_at, onset_loader, grid,
                    None if cuts is None else np.asarray(cuts, dtype=float))
    ts = np.linspace(float(t0), float(t1), int(n))
    return ts, np.array([float(f(t)) for t in ts])


def describe(v) -> str:
    """A link to the music, said: what moves it, around what, by how much."""
    if not isinstance(v, dict):
        return str(v)

    def who(t):
        if not isinstance(t, dict):
            return str(t)
        track = t.get("track")
        track = ", ".join(map(str, track)) if isinstance(track, (list, tuple)) else track
        notes = f" (notes {', '.join(map(str, t['notes']))})" if t.get("notes") else ""
        return f"{track}{notes}" if track else ", ".join(f"{k}: {x}" for k, x in t.items())
    if "hits" in v:
        what = f"a jump on each {who(v['hits'])} hit, fading in {v.get('envelope', 0.1):g} s"
    elif "adsr" in v:
        what = f"swells and fades with {who(v['adsr'])}"
    elif "pitch" in v:
        what = f"higher with {who(v['pitch'])}'s notes"
    elif "random_on" in v:
        what = f"somewhere new on each {who(v['random_on'])} hit"
    elif "lfo" in v:
        what = f"a {v['lfo']} wave, one every {v.get('period', '?')}"
    elif "feature" in v:
        what = f"the song's {v['feature']}"
    elif "from" in v:
        what = f"moving at {v.get('rate', 1):g}/s, pushed by " + describe(v["from"])
    else:
        what = ", ".join(f"{k}: {x}" for k, x in v.items())
    around = f"; around {v['bias']:g}" if isinstance(v.get("bias"), (int, float)) else ""
    by = f", by {v['gain']:g}" if isinstance(v.get("gain"), (int, float)) else ""
    return what + around + by

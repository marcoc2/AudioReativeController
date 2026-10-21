"""The Feature Lab (mixed into the Session): a link drawn step by step on the song's time.

What the feature listens to (``lab_wave``: a stem's own file, the mix through the band's
frequencies), the feature as the engine reads it, each step of the link (``knob_stages``:
the glide, the relative form's average and ratio, gain / bias / clip) and the value the
parameter gets, inside the range it expects (``param_range``). Every curve is the render's
own code (``_music_knob``) run at the render's fps in time order, so what the Lab draws is
what the render makes.

A link being edited is also tried on the picture before it is applied (``knob_draft``):
the Live stack is built with it in place of what the scene says (``video_cfg(draft=True)``),
and Cancel takes it away.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

from core.web.util import num, resolve

WAVE_HZ = 200          # a waveform's peaks a second
WAVE_SR = 24000        # the rate a waveform is read at (WAVE_HZ times a whole number)
WARM = 30.0            # at most this many seconds run before the shown span (a relative link's past, as in the render)


class Lab:
    """Mixed into the Session (it brings lock, get, notes, grid, features_at, fps, dur, knobs)."""

    # -- the parameter's range ---------------------------------------------------------------
    def param_range(self, path, keypath) -> dict:
        """The values the parameter expects: the user's own (set in the Lab), else what its
        about text says; {range: [lo, hi] | None, from: yours | about | None, key}."""
        from core import catalog
        from core.web.api import prefs
        name = str(list(keypath)[-1])
        with self.lock:
            try:
                e = catalog.of_layer(self.get(path))
            except (KeyError, IndexError, TypeError, ValueError):
                e = None
        key = f"{e.key if e else '?'}:{name}"
        mine = ((prefs().get("ui") or {}).get("ranges") or {}).get(key)
        if isinstance(mine, list) and len(mine) == 2:
            return {"range": [float(mine[0]), float(mine[1])], "from": "yours", "key": key}
        prm = next((p for p in (e.params if e else []) if p.name == name), None)
        if prm is None and name in ("mix", "opacity"):
            prm = catalog.Param(name, 1.0)
        rng = prm.range if prm is not None else None
        return {"range": rng, "from": "about" if rng else None, "key": key}

    # -- the steps of a link -----------------------------------------------------------------
    def knob_stages(self, state: dict, t0: float, t1: float, cols: int = 800) -> dict:
        """Each step of the link over [t0, t1] as min/max per column (``cols`` of them): the
        feature, its glide, the relative form's average and ratio, the final value; and the
        feature's spread over the song (p5, median, p95)."""
        from core import knob_forms as kf
        from core.video.layers import _music_knob
        try:
            spec = kf.spec_of(state)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        with self.lock:
            notes, grid, features_at, fps, dur = list(self.notes), self.grid, self.features_at, int(self.fps), float(self.dur)
        if grid is None:
            return {"error": "the bars are not known yet"}
        form = state.get("form")
        src = state.get("source") if form in ("feature", "relative") else None
        if src is not None and features_at is None:
            return {"error": "the audio features are still loading"}
        t0, t1 = max(0.0, float(t0)), min(dur, float(t1))
        if t1 <= t0:
            return {"error": "an empty span"}
        cuts = self.cut_times()
        cut_fn = (lambda to: [c for c in cuts if c <= to]) if cuts else None
        features_at = _recent(features_at) if features_at is not None else None    # the steps ask at the same times

        def knob(how):
            return _music_knob("lab", how, notes, features_at=features_at, grid=grid, cuts=cut_fn)

        steps = []                                       # (key, label, fn): run together, in order
        if src is not None:
            steps.append(("feature", "the feature", knob(str(src))))
        p = state.get("params") or {}
        if form == "feature" and (float(p.get("attack") or 0) > 0 or float(p.get("release") or 0) > 0):
            steps.append(("glided", "after attack / release",
                          knob({"feature": src, "attack": float(p.get("attack") or 0), "release": float(p.get("release") or 0)})))
        if form == "relative":
            steps.append(("average", "its recent average", _average(knob(str(src)), float(p.get("window") or 6.0))))
            bare = {k: v for k, v in spec.items() if k not in ("gain", "bias", "clip")}
            steps.append(("ratio", "against its average (1 = as usual)", knob(bare)))
        steps.append(("final", "what the parameter gets", knob(spec)))
        p_win = float(p.get("window") or 6.0)
        warm = 0.0 if form in ("constant", "feature") else min(WARM, 5 * p_win) if form == "relative" else WARM
        start = max(0.0, t0 - warm)
        ts = np.arange(math.floor(start * fps), math.ceil(t1 * fps) + 1) / fps
        ts = ts[ts <= t1 + 1e-9]
        vals = {k: np.empty(len(ts)) for k, _, _ in steps}
        try:
            for i, t in enumerate(ts):
                for k, _, f in steps:
                    vals[k][i] = float(f(t))
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        shown = ts >= t0 - 1e-9
        n = int(shown.sum())
        cols = max(1, min(int(cols), n))
        edges = np.linspace(0, n, cols + 1).astype(int)
        out = []
        for k, label, _ in steps:
            v = np.nan_to_num(vals[k][shown])
            lo = [num(v[a:max(a + 1, b)].min()) for a, b in zip(edges[:-1], edges[1:])]
            hi = [num(v[a:max(a + 1, b)].max()) for a, b in zip(edges[:-1], edges[1:])]
            out.append({"key": k, "label": label, "lo": [round(x, 4) for x in lo], "hi": [round(x, 4) for x in hi],
                        "p5": num(np.percentile(v, 5)), "p50": num(np.percentile(v, 50)), "p95": num(np.percentile(v, 95)),
                        "min": num(v.min()), "max": num(v.max())})
        got = {"t0": t0, "t1": t1, "cols": cols, "stages": out, "yaml": _flow_of(spec)}
        if src is not None:
            got["song"] = self._feature_spread(str(src))
        return got

    def _feature_spread(self, path: str) -> dict:
        """The feature's spread over the whole song (kept: it does not change with the link)."""
        key = ("lab-spread", path)
        with self.lock:
            got = self.knobs.get(key)
            features_at, fps, dur = self.features_at, int(self.fps), float(self.dur)
        if got is not None:
            return got
        from core.video.layers import _feature
        v = np.array([_feature(features_at(i / fps) or {}, path) for i in range(int(dur * fps))]) if dur > 0 else np.zeros(1)
        got = {"p5": num(np.percentile(v, 5)), "p50": num(np.percentile(v, 50)), "p95": num(np.percentile(v, 95)),
               "min": num(v.min()), "max": num(v.max())}
        with self.lock:
            self.knobs[key] = got
        return got

    # -- what the feature listens to -----------------------------------------------------------
    def lab_heard(self, feature: str) -> dict:
        """What the feature listens to, and how the page plays and draws it: a stem's file (its
        URL), the mix through a band (lo..hi Hz, the page filters the song), or the mix."""
        from core import feature_info
        with self.lock:
            stems = list(self.stem_names)
            files = dict(getattr(self, "stem_files", {}) or {})
        info = feature_info.describe(feature, stems)
        if info is None:
            return {"error": f"{feature}: not a feature the Lab knows"}
        heard = dict(info["heard"])
        if heard["kind"] == "stem":
            path = files.get(heard["stem"])
            if path and resolve(path).is_file():
                heard["url"] = self.media.url(resolve(path))
            else:
                heard = {"kind": "mix", "why": f"the {heard['stem']} stem's file is not known: the mix instead"}
        return {**info, "heard": heard}

    def lab_wave(self, feature: str) -> dict:
        """The waveform of what ``feature`` listens to: its peaks (0..1, scaled to its loudest),
        ``WAVE_HZ`` a second over the song; the mix's is the page's own."""
        got = self.lab_heard(feature)
        if "error" in got:
            return got
        heard = got["heard"]
        if heard["kind"] == "mix":
            return {"kind": "mix"}
        key = ("lab-wave", heard["kind"], heard.get("stem"), heard.get("lo"), heard.get("hi"))
        with self.lock:
            kept = self.knobs.get(key)
            audio, files = self.audio, dict(getattr(self, "stem_files", {}) or {})
        if kept is not None:
            return kept
        import librosa
        if heard["kind"] == "stem":
            y, sr = librosa.load(str(resolve(files[heard["stem"]])), sr=WAVE_SR, mono=True)
        else:
            from scipy.signal import butter, sosfiltfilt
            y, sr = librosa.load(str(audio), sr=WAVE_SR, mono=True)
            nyq = sr / 2.0
            lo, hi = float(heard["lo"]), min(float(heard["hi"]), nyq * 0.98)
            y = sosfiltfilt(butter(4, [lo / nyq, hi / nyq], btype="band", output="sos"), y)
        step = sr // WAVE_HZ
        n = len(y) // step
        peaks = np.abs(y[:n * step]).reshape(n, step).max(axis=1) if n else np.zeros(0)
        top = float(peaks.max()) if n else 0.0
        kept = {"kind": heard["kind"], "hz": WAVE_HZ, "scale": num(1.0 / top) if top > 0 else 1.0,
                "peaks": [round(float(x), 3) for x in (peaks / top if top > 0 else peaks)]}
        with self.lock:
            self.knobs[key] = kept
        return kept

    # -- a link tried on the picture -----------------------------------------------------------
    def knob_draft(self, path, keypath, state: dict, mode: str = "knob") -> dict:
        """The link being edited, on the Live picture before Apply: the stack is built with it
        at ``keypath`` of the layer at ``path`` (not saved, not an undo step)."""
        try:
            spec = self._knob_spec(state, mode)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}
        if mode == "trigger":
            return {"ok": False}                         # who plays: shown at Apply
        with self.lock:
            self.draft = (list(path), list(keypath), spec)
        return {"ok": True}

    def knob_draft_clear(self) -> dict:
        with self.lock:
            had = getattr(self, "draft", None) is not None
            self.draft = None
        return {"ok": True, "had": had}

    def _with_draft(self, cfg: dict) -> dict:
        """``cfg`` (the video settings, plain) with the draft link in its place."""
        draft = getattr(self, "draft", None)
        if draft is None:
            return cfg
        path, keypath, spec = draft
        try:
            node = cfg["layers"][self._top(path)]
            for k in self._rest(path):
                node = node[int(k)] if isinstance(node, list) else node[k]
            for k in keypath[:-1]:
                nxt = node[int(k)] if isinstance(node, list) else node.get(k)
                if nxt is None and isinstance(node, dict):
                    nxt = node[k] = {}
                node = nxt
            node[keypath[-1]] = spec
        except (KeyError, IndexError, TypeError, ValueError, StopIteration):
            pass                                         # the layer went (an undo): the scene as it is
        return cfg


def _recent(features_at, keep: int = 16):
    """``features_at`` remembering its last few answers: every step of a link asks for the same
    instants one after the other (and a glide on its own grid), so each is read once."""
    import collections
    memo: collections.OrderedDict = collections.OrderedDict()

    def at(t):
        key = round(float(t), 6)
        got = memo.get(key)
        if got is None:
            got = memo[key] = features_at(t)
            if len(memo) > keep:
                memo.popitem(last=False)
        return got
    return at


def _average(raw, window: float):
    """The relative form's recent average of ``raw`` (as ``_knob``'s relative computes it)."""
    state = {"avg": None, "t": None}
    window = max(1e-3, window)

    def avg(t):
        v = max(0.0, float(raw(t)))
        if state["avg"] is None:
            state["avg"] = v
        else:
            state["avg"] += (v - state["avg"]) * (1 - math.exp(-max(0.0, t - state["t"]) / window))
        state["t"] = t
        return state["avg"]
    return avg


def _flow_of(spec) -> str:
    from core.web.knobs import _flow
    return _flow(spec)

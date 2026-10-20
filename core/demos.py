"""Demos: a short, silent, low-resolution video of every Generator and FX of the catalog,
shown under its list in the Arrange window, so a name means something before it is used.

A demo is 512 x 512, ten seconds at 30 fps, without sound. A Generator fills the frame. An
FX is shown half and half: on the left the picture it is given, on the right what it makes
of it, over a backdrop picked to make that obvious:

    Lens & distortion   a grid with a ring and letters moving over it (straight lines bend)
    Time & trails       bright figures crossing a plain ground (their trails show)
    Colour & light,     a colourful clip with sharp edges
    Matter, Tools
    Cuts                three clips cut at every bar
    Figures             a montage with its SAM2 mask track (the figures are known)

Nothing is heard, but the layers still need something to follow: a synthetic loop plays
under every demo, 120 BPM, kick, snare and hats on a MIDI track "drums (midi)" (36, 38,
42, 46, 51), chords on "organ", an arpeggio on "e-piano" and a bass on "bass acid", with
its audio (for the features) and a transcription of its chords (for ``chord:`` and
``harmony:`` knobs). So a demo is the same for everyone, and the scenes of
``examples/me_diz_*.yaml``, tuned on that song's tracks, work on it as they are.

Each layer's settings come from (the first that has it):

    core/demos.yaml               a demo's own recipe (a layer, a backdrop)
    examples/me_diz_*.yaml        the scene where the layer was tuned alone
    the catalog                   its defaults

A strip at the bottom (source: meters) shows the knobs the music moves.

Every demo keeps a hash of what made it (its recipe, the shader's text or the layer's
code, the backdrop): ``render`` makes only the ones missing or out of date.

    python -m core.demos render [--only KEY ...] [--force]
    python -m core.demos list
    python -m core.demos sheet          # a contact sheet of them all
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib
import inspect
import json
import math
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Optional

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
DEMOS = ROOT / "render_output" / "demos"
RECIPES = Path(__file__).with_name("demos.yaml")
VERSION = 1                     # change to make every demo again

SIZE = 512
FPS = 30
SECONDS = 10.0
START = 30.0                    # a downbeat of the loop (bar 16): the layers have warmed up by then
PREROLL = 1.0                   # rendered and thrown away first (trails, feedback build up)
BPM = 120.0
LOOP_SECONDS = 96.0             # the loop's length (meters read a minute ahead)

FUNDO = ROOT / "render_output" / "esfolado_fundo"
CLIP = FUNDO / "2_3_samples_29-6-2026_0.mp4"
CUT_CLIPS = [FUNDO / "2_3_samples_29-6-2026_0.mp4", FUNDO / "tororo_1.mp4", FUNDO / "lofi_0.mp4"]
FIGURES = ROOT / "render_output" / "me_diz_capa_base.mp4"
FIGURE_MASKS = ROOT / "render_output" / "me_diz_capa_base_segments"
FIGURES_START = 27.422          # the song time of FIGURES' first frame (its masks are in song time)

BACKDROPS = ("black", "grid", "figure", "clip", "cuts", "figures")
BY_CATEGORY = {"Lens & distortion": "grid", "Time & trails": "figure", "Colour & light": "clip",
               "Matter": "clip", "Cuts": "cuts", "Figures": "figures", "Tools": "clip"}

DRIVERS = {"hits", "lfo", "adsr", "chord", "harmony", "cut", "feature", "rate", "bar", "since", "count",
           "pitch", "relative", "groove", "before", "random_on", "voice", "figure"}
FEATURE_ROOTS = {"subbands", "texture", "stems", "bands", "bass", "mid", "high", "flux", "centroid", "vocal"}
SCORE = "input/me_diz/sheetsage"


# ----------------------------------------------------------------------- the loop

CHORDS = [("E:min7", [52, 55, 59, 62]), ("C:maj7", [48, 52, 55, 59]), ("G:maj", [55, 59, 62]),
          ("D:7", [50, 54, 57, 60])]


def loop_dir() -> Path:
    return DEMOS / "_loop"


def loop_notes() -> list:
    """The loop's MIDI notes: drums, organ chords, an e-piano arpeggio, a bass."""
    from core.rhythm.midi_reader import MidiNote
    out = []
    beat, bar = 60.0 / BPM, 4 * 60.0 / BPM
    step = beat / 2                                          # eighths

    def note(t, pitch, vel, dur, track, idx):
        out.append(MidiNote(time=round(t, 6), pitch=pitch, velocity=vel, channel=9 if idx == 0 else idx,
                            duration=dur, track=track, track_index=idx))
    for b in range(int(LOOP_SECONDS / bar)):
        t0 = b * bar
        for i in range(8):
            t = t0 + i * step
            if i in (0, 3, 4) or (i == 7 and b % 2):
                note(t, 36, 120 if i in (0, 4) else 90, 0.1, "drums (midi)", 0)
            if i in (2, 6):
                note(t, 38, 110, 0.1, "drums (midi)", 0)
            if i == 7:
                note(t, 46, 80, 0.2, "drums (midi)", 0)
            else:
                note(t, 42, 60 + 20 * (i % 2 == 0), 0.05, "drums (midi)", 0)
            if i == 0 and b % 2 == 0:
                note(t, 51, 70, 0.3, "drums (midi)", 0)
        label, tones = CHORDS[b % len(CHORDS)]
        for p in tones:
            note(t0, p, 90, bar - 0.05, "organ", 1)
        for i in range(8):
            note(t0 + i * step, tones[i % len(tones)] + 12, 70 + 10 * (i % 2 == 0), step * 0.9, "e-piano", 2)
        for i in range(4):
            note(t0 + i * beat, tones[0] - 24, 100, beat * 0.8, "bass acid", 3)
    out.sort(key=lambda n: n.time)
    return out


def loop_grid():
    from core.rhythm.grid import RhythmGrid
    beat = 60.0 / BPM
    n = int(LOOP_SECONDS / beat)
    return RhythmGrid(bpm=BPM, time_signature=(4, 4), fps=FPS, beats=np.arange(n) * beat,
                      downbeats=np.arange(n // 4) * 4 * beat)


def make_loop() -> Path:
    """The loop's audio (for the features) and its transcription, once: ``_loop/``."""
    d = loop_dir()
    wav = d / "loop.wav"
    if wav.is_file() and (d / "score" / "chord.lab").is_file():
        return d
    import soundfile as sf
    d.mkdir(parents=True, exist_ok=True)
    sr = 22050
    y = np.zeros(int(LOOP_SECONDS * sr) + sr, np.float32)
    rng = np.random.default_rng(0)

    def add(t, sig):
        i = int(t * sr)
        y[i:i + len(sig)] += sig[:max(0, len(y) - i)]

    def hz(p):
        return 440.0 * 2 ** ((p - 69) / 12)
    for n in loop_notes():
        v = n.velocity / 127.0
        if n.track == "drums (midi)":
            if n.pitch == 36:
                k = np.arange(int(0.35 * sr)) / sr
                add(n.time, v * np.sin(2 * np.pi * (50 * k + 70 * (1 - np.exp(-k / 0.03)) * 0.03 * 33)) * np.exp(-k / 0.12))
            elif n.pitch == 38:
                k = np.arange(int(0.25 * sr)) / sr
                add(n.time, v * (0.7 * rng.standard_normal(len(k)) + 0.4 * np.sin(2 * np.pi * 190 * k)) * np.exp(-k / 0.07))
            else:
                k = np.arange(int((0.25 if n.pitch == 46 else 0.06) * sr)) / sr
                noise = np.diff(rng.standard_normal(len(k) + 1))
                add(n.time, 0.3 * v * noise * np.exp(-k / (0.08 if n.pitch == 46 else 0.015)))
        else:
            k = np.arange(int(n.duration * sr)) / sr
            f = hz(n.pitch)
            env = np.minimum(1.0, k / 0.02) * (np.exp(-k / 0.4) if n.track in ("e-piano", "bass acid") else 1.0)
            wave = np.sin(2 * np.pi * f * k) + 0.3 * np.sin(4 * np.pi * f * k)
            add(n.time, (0.12 if n.track == "organ" else 0.15) * v * env * wave)
    y = 0.8 * y / max(1e-6, float(np.abs(y).max()))
    sf.write(str(wav), y, sr)

    s = d / "score"
    s.mkdir(exist_ok=True)
    bar = 4 * 60.0 / BPM
    nbars = int(LOOP_SECONDS / bar)
    (s / "chord.lab").write_text("".join(f"{b * bar:.3f}\t{(b + 1) * bar:.3f}\t{CHORDS[b % len(CHORDS)][0]}\n"
                                         for b in range(nbars)), encoding="utf-8")
    (s / "beat.lab").write_text("".join(f"{i * bar / 4:.3f}\t{i % 4 + 1}\t4\t4\n" for i in range(nbars * 4)),
                                encoding="utf-8")
    (s / "downbeat.lab").write_text("".join(f"{b * bar:.3f}\n" for b in range(nbars)), encoding="utf-8")
    (s / "key.lab").write_text(f"0.000\t{LOOP_SECONDS:.3f}\tE:minor\n", encoding="utf-8")
    (s / "structure.lab").write_text(f"0.000\t{LOOP_SECONDS:.3f}\tverse\n", encoding="utf-8")
    return d


class _Features:
    """The loop's audio features, once a frame (both halves of a demo ask at the same time)."""

    def __init__(self, wav: Path):
        from core.feature_extractor import AudioFeatureExtractor
        self.ex = AudioFeatureExtractor(str(wav), fps=FPS, skip_separation=True)
        self.t, self.v = None, None

    def reset(self):
        self.ex.prev_features = None
        self.t, self.v = None, None

    def __call__(self, t):
        if t != self.t:
            self.t, self.v = t, self.ex.get_features_at_time(t, apply_gate=False)
        return self.v


# ----------------------------------------------------------------------- recipes

def file_of(key: str) -> str:
    """"shadertoy:droste_arc" -> "shadertoy__droste_arc"."""
    return key.replace(":", "__")


def video_path(key: str) -> Path:
    return DEMOS / f"{file_of(key)}.mp4"


def info_path(key: str) -> Path:
    return DEMOS / f"{file_of(key)}.json"


def _curated() -> dict:
    if not RECIPES.is_file():
        return {}
    return yaml.safe_load(RECIPES.read_text(encoding="utf-8")) or {}


def _layer_key(spec: dict) -> Optional[str]:
    src = spec.get("source", "clips")
    if src == "shadertoy" and spec.get("file"):
        return "shadertoy:" + Path(str(spec["file"])).stem
    return src


def tuned_scenes() -> dict:
    """key -> (scene, its layer spec): the me_diz scene where the layer was tuned, the one
    with the fewest other layers (alone, ideally)."""
    found: dict = {}
    for f in sorted(glob.glob(str(ROOT / "examples" / "me_diz_*.yaml"))):
        try:
            scene = yaml.safe_load(Path(f).read_text(encoding="utf-8")) or {}
        except yaml.YAMLError:
            continue
        layers = [l for l in ((scene.get("video") or {}).get("layers") or []) if isinstance(l, dict)]
        own = [l for l in layers if l.get("source", "clips") not in ("clips", "meters")]
        for spec in own:
            key = _layer_key(spec)
            if key not in found or len(own) < found[key][2]:
                found[key] = (Path(f).relative_to(ROOT).as_posix(), spec, len(own))
    return {k: (f, s) for k, (f, s, _) in found.items()}


def _swap_score(x):
    """The me_diz transcription -> the loop's."""
    if isinstance(x, dict):
        return {k: _swap_score(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_swap_score(v) for v in x]
    if isinstance(x, str) and x.replace("\\", "/").rstrip("/") == SCORE:
        score = loop_dir() / "score"
        try:
            return score.relative_to(ROOT).as_posix()     # the hash is the same on any machine
        except ValueError:
            return score.as_posix()
    return x


def recipe(entry) -> dict:
    """What a demo shows: {key, label, kind, category, layer, backdrop, origin}."""
    from core import catalog
    own = _curated().get(entry.key) or {}
    tuned = tuned_scenes().get(entry.key)
    if own.get("layer"):
        layer, origin = dict(own["layer"]), "core/demos.yaml"
    elif tuned:
        layer, origin = json.loads(json.dumps(tuned[1])), tuned[0]
    else:
        layer, origin = entry.spec(), "the catalog's defaults"
    for k in own.get("drop") or []:                       # what the demo does without
        layer.pop(k, None)
    _merge(layer, own.get("set") or {})                   # and what it shows otherwise
    if (own.get("drop") or own.get("set")) and not own.get("layer"):
        origin += ", changed in core/demos.yaml"
    backdrop = own.get("backdrop") or (
        "figures" if "masks" in entry.needs or _follows_figures(layer) else
        "black" if entry.kind == catalog.GENERATOR else
        "cuts" if "cuts" in entry.needs else BY_CATEGORY.get(entry.category, "clip"))
    if backdrop != "figures":
        layer.pop("mask", None)
    for k in ("bars", "enabled", "fade_in", "fade_out", "role"):
        layer.pop(k, None)
    return {"key": entry.key, "label": entry.label, "kind": entry.kind, "category": entry.category,
            "layer": _swap_score(layer), "backdrop": backdrop, "origin": origin,
            "lanes": own.get("lanes"), "note": own.get("note", "")}


def _merge(into: dict, changes: dict) -> None:
    """``changes`` over ``into``, key by key inside mappings (``uniforms: {u_slice: ..}``
    changes that one uniform)."""
    for k, v in changes.items():
        if isinstance(v, dict) and isinstance(into.get(k), dict) and not (DRIVERS & set(v)):
            _merge(into[k], v)
        else:
            into[k] = v


def _follows_figures(x) -> bool:
    """A knob that follows the mask track's figures (``figure:``), or a map made of them."""
    if isinstance(x, dict):
        return "figure" in x or x.get("map") == "figures" or any(_follows_figures(v) for v in x.values())
    if isinstance(x, list):
        return any(_follows_figures(v) for v in x)
    return False


def _code_of(entry) -> str:
    """The text that makes the layer: the shader, or the class's source."""
    from core import catalog
    if entry.file:
        p = ROOT / entry.file
        return p.read_text(encoding="utf-8") if p.is_file() else ""
    mod, cls = catalog.NATIVE.get(entry.source, (None, None, None, None))[:2]
    if mod is None:
        return ""
    try:
        return inspect.getsource(getattr(importlib.import_module(mod), cls))
    except (OSError, TypeError, AttributeError, ImportError):
        return ""


def digest(entry, rec: Optional[dict] = None) -> str:
    rec = rec or recipe(entry)
    what = json.dumps({"v": VERSION, "size": SIZE, "seconds": SECONDS, "recipe": rec}, sort_keys=True, default=str)
    h = hashlib.sha1(what.encode("utf-8"))
    h.update(_code_of(entry).encode("utf-8"))
    return h.hexdigest()[:16]


def status(entry) -> str:
    """"ready", "stale" (made from another recipe or code), "missing", "failed", "rendering"."""
    info = read_info(entry.key)
    if info is None:
        return "missing"
    if info.get("state") in ("failed", "rendering"):
        return info["state"]
    if not video_path(entry.key).is_file():
        return "missing"
    return "ready" if info.get("hash") == digest(entry) else "stale"


def read_info(key: str) -> Optional[dict]:
    p = info_path(key)
    try:
        return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
    except (OSError, ValueError):
        return None


# ----------------------------------------------------------------------- backdrops

class _Video:
    """A video file read forwards, frame by frame, fitted (cover) to w x h, looping."""

    def __init__(self, path: Path, w: int, h: int):
        import cv2
        self.path, self.w, self.h = path, w, h
        self.cap = cv2.VideoCapture(str(path))
        self.n = max(1, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or FPS
        self.at, self.frame = -1, np.zeros((h, w, 3), np.uint8)

    def get(self, idx: int) -> np.ndarray:
        import cv2
        idx = idx % self.n
        if idx < self.at:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.at = -1
        while self.at < idx:
            ok, img = self.cap.read()
            self.at += 1
            if not ok:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.at = -1
                break
            if self.at == idx:
                self.frame = self._fit(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        return self.frame

    def _fit(self, img: np.ndarray) -> np.ndarray:
        import cv2
        h, w = img.shape[:2]
        k = max(self.w / w, self.h / h)
        img = cv2.resize(img, (max(self.w, round(w * k)), max(self.h, round(h * k))), interpolation=cv2.INTER_AREA)
        y0, x0 = (img.shape[0] - self.h) // 2, (img.shape[1] - self.w) // 2
        return np.ascontiguousarray(img[y0:y0 + self.h, x0:x0 + self.w])


class Backdrop:
    """What an FX is given (the ``clips`` base of its stack): ``frame_at(t)`` and the cuts."""

    def __init__(self, kind: str, w: int, h: int):
        if kind not in BACKDROPS:
            raise ValueError(f"unknown backdrop {kind!r} ({', '.join(BACKDROPS)})")
        self.kind, self.w, self.h = kind, w, h
        self.every = 4 * 60.0 / BPM * (1 if kind == "cuts" else 2)    # a cut at every bar, or every two
        self.videos = []
        if kind == "clip":
            self.videos = [_Video(CLIP, w, h)]
        elif kind == "cuts":
            self.videos = [_Video(p, w, h) for p in CUT_CLIPS]
        elif kind == "figures":
            self.videos = [_Video(FIGURES, w, h)]
        if kind == "grid":
            self._ground = self._checker()

    def cut_times(self, until: float) -> np.ndarray:
        return np.arange(self.every, max(self.every, until) + 1e-9, self.every)

    def frame_at(self, t: float) -> np.ndarray:
        k = self.kind
        if k == "black":
            return np.zeros((self.h, self.w, 3), np.uint8)
        if k == "grid":
            return self._grid(t)
        if k == "figure":
            return self._figure(t)
        if k == "figures":
            return self.videos[0].get(int(round((t - FIGURES_START) * FPS)))
        if k == "cuts":
            i = int(t // self.every) % len(self.videos)
            v = self.videos[i]
            return v.get(int(round((t % self.every) * v.fps)))
        v = self.videos[0]
        return v.get(int(round(t * v.fps)))

    def _checker(self) -> np.ndarray:
        import cv2
        w, h = self.w, self.h
        cell = max(8, w // 8)
        yy, xx = np.mgrid[0:h, 0:w]
        board = (((xx // cell) + (yy // cell)) % 2).astype(np.uint8)
        img = np.where(board[..., None] == 1, np.uint8([58, 60, 74]), np.uint8([28, 30, 38])).astype(np.uint8)
        for x in range(0, w, cell // 2):
            cv2.line(img, (x, 0), (x, h), (70, 180, 210) if x % cell == 0 else (45, 90, 110), 1)
        for y in range(0, h, cell // 2):
            cv2.line(img, (0, y), (w, y), (70, 180, 210) if y % cell == 0 else (45, 90, 110), 1)
        return img

    def _grid(self, t: float) -> np.ndarray:
        import cv2
        img = self._ground.copy()
        w, h = self.w, self.h
        cx = int(w / 2 + 0.22 * w * math.cos(t * 2 * math.pi / 4.0))
        cy = int(h / 2 + 0.22 * h * math.sin(t * 2 * math.pi / 4.0))
        for r, c in ((0.16, (255, 170, 40)), (0.11, (255, 90, 120)), (0.06, (250, 240, 220))):
            cv2.circle(img, (cx, cy), int(r * w), c, max(2, w // 90), cv2.LINE_AA)
        x = int(w * (0.5 + 0.3 * math.sin(t * 2 * math.pi / 6.0)))
        size = w / 220.0
        (tw, th), _ = cv2.getTextSize("ARC", cv2.FONT_HERSHEY_DUPLEX, size, max(2, w // 120))
        cv2.putText(img, "ARC", (x - tw // 2, int(h * 0.82)), cv2.FONT_HERSHEY_DUPLEX, size, (240, 240, 245),
                    max(2, w // 120), cv2.LINE_AA)
        return img

    def _figure(self, t: float) -> np.ndarray:
        import cv2
        w, h = self.w, self.h
        y = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        img = (np.float32([16, 20, 44]) * (1 - y) + np.float32([40, 18, 50]) * y).repeat(w, 1).astype(np.uint8)
        x = int(w * (0.5 + 0.36 * math.sin(t * 2 * math.pi / 3.0)))
        yc = int(h * (0.5 + 0.12 * math.sin(t * 2 * math.pi / 1.5)))
        cv2.circle(img, (x, yc - int(0.12 * h)), int(0.06 * w), (250, 240, 230), -1, cv2.LINE_AA)
        cv2.rectangle(img, (x - int(0.06 * w), yc - int(0.05 * h)), (x + int(0.06 * w), yc + int(0.14 * h)),
                      (240, 70, 160), -1, cv2.LINE_AA)
        bx = int(w / 2 + 0.3 * w * math.cos(t * 2 * math.pi / 1.2))
        by = int(h / 2 + 0.3 * h * math.sin(t * 2 * math.pi / 1.2))
        cv2.circle(img, (bx, by), int(0.035 * w), (255, 220, 60), -1, cv2.LINE_AA)
        return img


class _Shown:
    """The demo's frame: a Generator's picture, or an FX's two halves side by side (without
    on the left, with on the right), padded to the square; meters draw over it."""

    def __init__(self, dry, wet, w: int, h: int, split: bool, cuts):
        self.dry, self.wet, self.w, self.h, self.split = dry, wet, w, h, split
        self.cut_times = cuts

    def frame_at(self, t: float) -> np.ndarray:
        import cv2
        wet = self.wet.frame_at(t)
        if self.split:
            dry = self.dry.frame_at(t)
            half = self.w // 2
            wet = np.concatenate([dry[:, :half], wet[:, half:]], axis=1)
            wet[:, half - 1:half + 1] = 255
        out = np.zeros((SIZE, SIZE, 3), np.uint8)
        y0, x0 = (SIZE - self.h) // 2, (SIZE - self.w) // 2
        out[y0:y0 + self.h, x0:x0 + self.w] = wet[:self.h, :self.w]
        if self.split:
            for text, x in (("without", x0 + 8), ("with", x0 + self.w // 2 + 8)):
                cv2.putText(out, text, (x, y0 + self.h - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(out, text, (x, y0 + self.h - 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1,
                            cv2.LINE_AA)
        return out


# ----------------------------------------------------------------------- render

def _knob_lanes(layer: dict, limit: int = 3) -> list:
    """The layer's knobs the music moves, as meters lanes."""
    out = []

    def is_driver(v):
        if isinstance(v, dict):
            if "figure" in v:                             # the meters read no mask track
                return False
            return bool(DRIVERS & set(v)) or (isinstance(v.get("from"), dict))
        return isinstance(v, str) and v.split(".")[0] in FEATURE_ROOTS

    def walk(spec, path):
        for k, v in spec.items():
            if k in ("source", "file", "mask", "triggers", "lanes", "fx"):
                continue
            if is_driver(v):
                out.append({"label": ".".join(path + [k]), "knob": v})
            elif isinstance(v, dict):
                walk(v, path + ([] if k == "uniforms" else [k]))
    walk(layer, [])
    return out[:limit]


def _meters(rec: dict) -> dict:
    lanes = rec.get("lanes") or _knob_lanes(rec["layer"])
    if not lanes:
        lanes = [{"label": "the beat (a loop, not heard)", "roll": {"track": "drums (midi)"}}]
    sub = (f"{rec['category']} · left: without it, right: with it" if rec["kind"] == "fx"
           else f"{rec['category']} · Generator")
    return {"source": "meters", "title": rec["label"], "subtitle": sub, "past": 2.0, "future": 0.6,
            "lane": 30, "lanes": lanes}


def render(entry, song: dict, log=print) -> Path:
    """Make one demo: its video and its info."""
    from core import catalog, gpu
    from core.video.layers import build_compositor
    key = entry.key
    DEMOS.mkdir(parents=True, exist_ok=True)
    try:
        rec = recipe(entry)
    except Exception as exc:                              # a recipe that does not read
        info_path(key).write_text(json.dumps({"key": key, "state": "failed",
                                              "error": f"its recipe: {type(exc).__name__}: {exc}"}), encoding="utf-8")
        log(f"  {key}: FAILED its recipe: {exc}")
        raise
    log(f"  {key}: over {rec['backdrop']}, from {rec['origin']}")
    info = {"key": key, "label": entry.label, "kind": entry.kind, "state": "rendering", "hash": digest(entry, rec),
            "recipe": rec, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    info_path(key).write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
    w, h = (288, SIZE) if rec["backdrop"] == "figures" else (SIZE, SIZE)
    masks = str(FIGURE_MASKS) if rec["backdrop"] == "figures" else None
    split = entry.kind == catalog.FX and key != "meters"
    feats = song["features"]
    feats.reset()
    notes, grid = song["notes"], song["grid"]
    t0 = time.perf_counter()
    try:
        dry = Backdrop(rec["backdrop"], w, h)
        base = Backdrop(rec["backdrop"], w, h)
        wet = build_compositor(base, {"layers": [{"source": "clips"}, rec["layer"]]}, notes, w, h,
                               fps=FPS, features_at=feats, grid=grid, masks=masks)
        shown = _Shown(dry, wet, w, h, split, base.cut_times)
        read_out = [] if key == "meters" else [_meters(rec)]        # the meters' demo is itself
        top = build_compositor(shown, {"layers": [{"source": "clips"}] + read_out}, notes, SIZE, SIZE,
                               fps=FPS, features_at=feats, grid=grid, masks=masks)
        out = video_path(key)
        tmp = out.with_suffix(".part.mp4")
        enc = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
                                "-s", f"{SIZE}x{SIZE}", "-r", str(FPS), "-i", "-", "-c:v", "libx264",
                                "-crf", "24", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                                str(tmp)], stdin=subprocess.PIPE)
        try:
            start = START - PREROLL
            n_pre, n = int(PREROLL * FPS), int(SECONDS * FPS)
            for fi in range(n_pre + n):
                t = start + fi / FPS
                if fi < n_pre:
                    wet.frame_at(t)
                    if split:
                        dry.frame_at(t)
                    continue
                enc.stdin.write(np.ascontiguousarray(top.frame_at(t)).tobytes())
        finally:
            enc.stdin.close()
            enc.wait()
        if enc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed (exit {enc.returncode})")
        tmp.replace(out)
        info.update(state="ready", seconds_to_make=round(time.perf_counter() - t0, 1),
                    made=time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"  {key}: {info['seconds_to_make']} s")
        return out
    except Exception as exc:
        info.update(state="failed", error=f"{type(exc).__name__}: {exc}", trace=traceback.format_exc())
        log(f"  {key}: FAILED {type(exc).__name__}: {exc}")
        raise
    finally:
        info_path(key).write_text(json.dumps(info, indent=1, default=str), encoding="utf-8")
        try:
            gpu.release()
        except Exception:
            pass


def song() -> dict:
    d = make_loop()
    return {"features": _Features(d / "loop.wav"), "notes": loop_notes(), "grid": loop_grid()}


def to_make(keys=None, force: bool = False, retry: bool = True) -> list:
    from core import catalog
    out = []
    for e in catalog.entries():
        if keys and e.key not in keys:
            continue
        st = status(e)
        if force or st in ("missing", "stale") or (retry and st in ("failed", "rendering")):
            out.append(e)
    return out


def render_all(keys=None, force: bool = False, log=print) -> dict:
    todo = to_make(keys, force)
    log(f"{len(todo)} demo(s) to make")
    if not todo:
        return {}
    s = song()
    failed = {}
    for i, e in enumerate(todo, 1):
        log(f"[{i}/{len(todo)}] {e.key}")
        try:
            render(e, s, log)
        except Exception as exc:
            failed[e.key] = f"{type(exc).__name__}: {exc}"
    return failed


def contact_sheet(path: Optional[Path] = None, at=(1.0, 4.0, 7.5)) -> Path:
    """Every demo, a row of frames: for looking them over."""
    import cv2
    from core import catalog
    rows = []
    for e in catalog.entries():
        p = video_path(e.key)
        tiles = []
        for t in at:
            img = np.zeros((170, 170, 3), np.uint8)
            if p.is_file():
                cap = cv2.VideoCapture(str(p))
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * FPS))
                ok, frame = cap.read()
                cap.release()
                if ok:
                    img = cv2.resize(frame, (170, 170), interpolation=cv2.INTER_AREA)
            tiles.append(img)
        label = np.zeros((170, 150, 3), np.uint8)
        for j, line in enumerate([e.key.replace("shadertoy:", "")[:20], status(e)]):
            cv2.putText(label, line, (4, 20 + 18 * j), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        rows.append(np.hstack([label] + tiles))
    path = path or DEMOS / "sheet.png"
    per = 11
    cols = [np.vstack(rows[i:i + per] + [np.zeros_like(rows[0])] * (per - len(rows[i:i + per])))
            for i in range(0, len(rows), per)]
    cv2.imwrite(str(path), np.hstack(cols))
    return path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="the catalog's demos")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render", help="make the demos missing or out of date")
    r.add_argument("--only", nargs="*", default=None, help="these keys only")
    r.add_argument("--force", action="store_true", help="make them again even when up to date")
    sub.add_parser("list", help="each demo's state")
    sh = sub.add_parser("sheet", help="a contact sheet of every demo")
    sh.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "render":
        failed = render_all(set(args.only) if args.only else None, args.force)
        for k, why in failed.items():
            print(f"FAILED {k}: {why}")
        return 1 if failed else 0
    if args.cmd == "list":
        from core import catalog
        for e in catalog.entries():
            rec = recipe(e)
            print(f"{status(e):9s} {e.key:34s} {rec['backdrop']:8s} {rec['origin']}")
        return 0
    print(contact_sheet(Path(args.out) if args.out else None))
    return 0


if __name__ == "__main__":
    sys.exit(main())

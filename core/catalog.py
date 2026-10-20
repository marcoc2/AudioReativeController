"""Every Generator and FX the project has, for the GUI to list and insert.

The names (docs/GUI_BACKLOG.md item 14), as in a DAW:

- a **Clip** is a video or image file (``footage`` layers, the ``clips`` montage);
- a **Generator** makes its own picture (a fractal, cells, a nebula): an instrument;
- an **FX** works on the picture it is given (grain, feedback, pixel sort): an effect.

Some Generators can also take the picture below as an input (``flame`` with
``paint: under``, the chord shaders with ``u_video``), like a sidechain: ``takes_input``.

The kind is read from the code, not written by hand: a layer class with ``process()`` gets
the picture so far (an FX); one with only ``frame_at()`` makes its own (a Generator); one
with ``frame_at_over()`` makes its own but sees what is below. A shader is an FX when one of
its passes reads ``frame``. The category (what it is for) is the one thing written here,
and a test keeps every layer source and shader in this list.

Parameters come from what each layer already documents: the YAML example in its docstring
(``key: value  # what it does``), else the keys its code reads (``spec.get("key", default)``);
a shader's are its ``uniform float u_x = default;`` lines with their notes.
"""
from __future__ import annotations

import ast
import importlib
import inspect
import re
import textwrap
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

ROOT = Path(__file__).resolve().parent.parent
SHADERS = ROOT / "shaders" / "shadertoy"

GENERATOR = "generator"
FX = "fx"

# what each one is for, in the order the GUI shows them
GENERATOR_CATEGORIES = ("Fractals", "Organic & particles", "Faces", "Patterns & tunnels", "Colour", "Stage")
FX_CATEGORIES = ("Colour & light", "Time & trails", "Lens & distortion", "Matter", "Cuts", "Figures", "Tools")

# numbers that set the work up rather than the look: not for the music to move
SETUP_KEYS = {"seed", "supersample", "oversample", "resolution", "samples", "subframes"}

# keys every layer has: the GUI sets them itself (the layer row, the timeline's blocks)
COMMON_KEYS = {"source", "enabled", "blend", "opacity", "bars", "fade_in", "fade_out", "mask", "role",
               "mute", "lane_row"}

# source: (module, class, category, needs)
#   needs: what it cannot do without: "masks" (a SAM2 mask track), "cuts" (the clip montage's cuts)
NATIVE = {
    "solid":       ("core.video.layers", "SolidLayer", "Colour", ()),
    "cells":       ("core.video.layers", "CellsLayer", "Organic & particles", ()),
    "veils":       ("core.video.layers", "VeilsLayer", "Organic & particles", ()),
    "chladni":     ("core.video.layers", "ChladniLayer", "Organic & particles", ()),
    "orbiters":    ("core.video.layers", "OrbitersLayer", "Organic & particles", ()),
    "particles":   ("core.video.layers", "ParticlesLayer", "Organic & particles", ()),
    "ferrofluid":  ("core.video.layers", "FerrofluidLayer", "Organic & particles", ()),
    "sam_cutout":  ("core.video.sam_cutout", "SamCutout", "Figures", ()),
    "eyes":        ("core.video.layers", "EyesLayer", "Faces", ()),
    "mouths":      ("core.video.layers", "MouthsLayer", "Faces", ()),
    "flame":       ("core.video.layers", "FlameLayer", "Fractals", ()),
    "julia":       ("core.video.layers", "JuliaLayer", "Fractals", ()),
    "mandelbox":   ("core.video.layers", "MandelboxLayer", "Fractals", ()),
    "mandelbulb":  ("core.video.layers", "MandelbulbLayer", "Fractals", ()),
    "nebulabrot":  ("core.video.nebulabrot", "NebulabrotLayer", "Fractals", ()),
    "deep_zoom":   ("core.video.deep_zoom", "DeepZoomLayer", "Fractals", ()),
    "flythrough":  ("core.video.flythrough", "FlythroughLayer", "Fractals", ()),
    "cubes":       ("core.video.layers", "CubesLayer", "Stage", ()),
    "chord_color": ("core.video.chord_color", "ChordColor", "Colour & light", ()),
    "rgb_split":   ("core.video.layers", "RgbSplit", "Colour & light", ()),
    "rgb_noise":   ("core.video.layers", "RgbNoiseLayer", "Colour & light", ()),
    "grain":       ("core.video.layers", "GrainLayer", "Colour & light", ()),
    "feedback":    ("core.video.layers", "FeedbackLayer", "Time & trails", ()),
    "echoes":      ("core.video.layers", "EchoesLayer", "Time & trails", ()),
    "slitscan":    ("core.video.layers", "SlitScanLayer", "Time & trails", ()),
    "time_map":    ("core.video.time_map", "TimeMap", "Time & trails", ()),
    "flame_lens":  ("core.video.flame_lens", "FlameLens", "Lens & distortion", ()),
    "shockwave":   ("core.video.layers", "ShockwaveLayer", "Lens & distortion", ()),
    "pixel_sort":  ("core.video.pixel_sort", "PixelSort", "Matter", ()),
    "sandlines":   ("core.video.layers", "SandLinesLayer", "Matter", ()),
    "drops":       ("core.video.layers", "DropsLayer", "Matter", ()),
    "transition":  ("core.video.transition", "Transition", "Cuts", ("cuts",)),
    "datamosh":    ("core.video.datamosh", "Datamosh", "Cuts", ("cuts",)),
    "cast":        ("core.video.cast", "CastLayer", "Figures", ("masks",)),
    "meters":      ("core.video.meters", "Meters", "Tools", ()),
}

# layer sources that are not a Generator or an FX: the clips themselves, the shader player
# (each shader is its own entry) and the Arrange window's lanes
NOT_LISTED = {"clips", "footage", "shadertoy", "group"}

# shader (file or folder name under shaders/shadertoy): category
SHADER_CATEGORIES = {
    "bolhas_arc": "Fractals", "fases_arc": "Fractals", "hibrido_arc": "Fractals",
    "julia4d_arc": "Fractals", "kifs_acorde_arc": "Fractals", "newton_acorde_arc": "Fractals",
    "lyapunov_bateria_arc": "Fractals", "nebulosa_arc": "Fractals",
    "kaleidoscope": "Patterns & tunnels", "kaleidoscope_arc": "Patterns & tunnels",
    "octagrams": "Patterns & tunnels", "octagrams_arc": "Patterns & tunnels",
    "starleidoscope": "Patterns & tunnels", "sombras_2d": "Patterns & tunnels",
    "sombras_2d_arc": "Patterns & tunnels", "sombras_2d_kicks": "Patterns & tunnels",
    "diamond_dust": "Patterns & tunnels", "diamond_dust_arc": "Patterns & tunnels",
    "bloom_arc": "Colour & light",
    "fluxo_cromatico_arc": "Time & trails", "milkdrop_arc": "Time & trails",
    "arc_lente_kick": "Lens & distortion", "droste_arc": "Lens & distortion",
    "hiperbolico_arc": "Lens & distortion", "domain_warp_arc": "Lens & distortion",
    "perolas_arc": "Lens & distortion", "prism_liquid": "Patterns & tunnels",
    "prism_liquid_arc": "Patterns & tunnels",
    "copiadora_arc": "Matter", "reacao_arc": "Matter",
    "cenario_arc": "Figures", "julia_figuras_arc": "Figures",
}


@dataclass
class Param:
    name: str
    default: object = None            # None: a structure (a trigger, a list) or required
    about: str = ""

    @property
    def knob(self) -> bool:
        """A number: something the music can move."""
        return (isinstance(self.default, (int, float)) and not isinstance(self.default, bool)
                and self.name not in SETUP_KEYS)


@dataclass
class Entry:
    key: str                          # "grain", "shadertoy:droste_arc"
    label: str                        # what the GUI shows
    kind: str                         # GENERATOR or FX
    category: str
    about: str                        # one line
    source: str                       # the layer's source:
    file: Optional[str] = None        # a shader's path (relative to the project)
    takes_input: bool = False         # a Generator that can also show the picture below
    needs: tuple = ()
    params: List[Param] = field(default_factory=list)

    def spec(self) -> dict:
        """A new layer of it, as a scene writes it."""
        out = {"source": self.source}
        if self.file:
            out["file"] = self.file
        return out


# ----------------------------------------------------------------------- native layers

_EXAMPLE_KEY = re.compile(r"^(\s+)([a-z_][a-z_0-9]*):\s*(.*?)\s*(?:#\s*(.*))?$")


def _example(doc: str, source: str) -> List[Param]:
    """The keys of the ``- source: <source>`` YAML example in a docstring, with their
    comments (a comment that goes on in the next lines, only comment, is kept whole)."""
    m = re.search(rf"^\s*-?\s*source:\s*{re.escape(source)}\b.*$", doc, re.M)
    if not m:
        return []
    import yaml
    out: List[Param] = []
    indent = None
    for line in doc[m.end():].splitlines()[1:]:
        if not line.strip():
            break
        km = _EXAMPLE_KEY.match(line)
        if km and (indent is None or km.group(1) == indent):
            indent = km.group(1)
            try:
                value = yaml.safe_load(km.group(3)) if km.group(3) else None
            except yaml.YAMLError:
                value = None
            if isinstance(value, (dict, list)):
                value = None
            out.append(Param(km.group(2), value, (km.group(4) or "").strip()))
        elif out and line.strip().startswith("#") and indent is not None and len(line) - len(line.lstrip()) > len(indent):
            out[-1].about = (out[-1].about + " " + line.strip().lstrip("#").strip()).strip()
        elif indent is not None and len(line) - len(line.lstrip()) < len(indent):
            break
    return out


def _read_keys(cls) -> List[Param]:
    """``spec.get("key", default)`` and ``spec["key"]`` in the class's code, in order."""
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls)))
    except (OSError, TypeError, SyntaxError):
        return []
    found: dict = {}
    for node in ast.walk(tree):
        key, default = None, None
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "get"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "spec"
                and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)):
            key = node.args[0].value
            if len(node.args) > 1:
                try:
                    default = ast.literal_eval(node.args[1])
                except ValueError:
                    default = None
        elif (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "spec"
              and isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str)):
            key = node.slice.value
        if key and key not in found:
            found[key] = (getattr(node, "lineno", 0), default if not isinstance(default, (dict, list, tuple)) else None)
    return [Param(k, d) for k, (_, d) in sorted(found.items(), key=lambda kv: kv[1][0])]


def _first_line(doc: str) -> str:
    text = " ".join(doc.strip().split("\n\n")[0].split())
    text = re.sub(r"^(Post-op|Source|Source layer)\s*(reference implementation)?:\s*", "", text)
    m = re.match(r"(.+?[.!?])(\s|$)", text)
    text = (m.group(1) if m else text)[:160]
    return text[:1].upper() + text[1:]


def _native(source: str) -> Entry:
    mod_name, cls_name, category, needs = NATIVE[source]
    mod = importlib.import_module(mod_name)
    cls = getattr(mod, cls_name)
    cdoc, mdoc = inspect.getdoc(cls) or "", inspect.getdoc(mod) or ""
    doc = mdoc if "see the module docstring" in cdoc or not cdoc else cdoc
    kind = FX if (hasattr(cls, "process") or hasattr(cls, "process_gpu")) else GENERATOR
    params = _example(cdoc, source) or _example(mdoc, source)
    have = {p.name for p in params}
    params += [p for p in _read_keys(cls) if p.name not in have]
    params = [p for p in params if p.name not in COMMON_KEYS]
    return Entry(key=source, label=source, kind=kind, category=category, about=_first_line(doc),
                 source=source, takes_input=kind == GENERATOR and hasattr(cls, "frame_at_over"),
                 needs=tuple(needs), params=params)


# ----------------------------------------------------------------------- shaders

def shader_paths(root: Path = SHADERS) -> List[Path]:
    """Every shader: a .glsl file, or a folder with an image.glsl (multipass)."""
    if not root.is_dir():
        return []
    return sorted([p for p in root.glob("*.glsl")] + [p for p in root.iterdir() if (p / "image.glsl").is_file()],
                  key=lambda p: p.stem)


def _shader(path: Path) -> Entry:
    from core import shadertoy
    src = shadertoy.load(path)
    reads = {s.split()[0] for chans in src["channels"].values() for s in chans.values() if isinstance(s, str)}
    params = [Param(u["name"], u["default"], u["about"]) for u in shadertoy.uniforms_of(path)]
    kind = FX if "frame" in reads else GENERATOR
    meta = src["meta"]
    title = re.sub(r"\s+—\s+Image$", "", meta.get("title", path.stem))
    notes = re.sub(r"\s+", " ", meta.get("notes", "")).strip()
    about = title + (f": {notes}" if notes and len(notes) < 120 else "")
    needs = ("masks",) if reads & {"mask", "mask_distance"} else ()
    try:
        rel = path.relative_to(ROOT).as_posix()
    except ValueError:
        rel = path.as_posix()
    return Entry(key=f"shadertoy:{path.stem}", label=path.stem, kind=kind,
                 category=SHADER_CATEGORIES.get(path.stem, "Patterns & tunnels" if kind == GENERATOR else "Matter"),
                 about=about, source="shadertoy", file=rel, needs=needs, params=params,
                 takes_input=kind == GENERATOR and any(p.name == "u_video" for p in params))


# ----------------------------------------------------------------------- the list

@lru_cache(maxsize=1)
def entries() -> tuple:
    """Every Generator and FX: the native layers, then the shaders."""
    out = [_native(s) for s in NATIVE]
    out += [_shader(p) for p in shader_paths()]
    return tuple(out)


def by_key(key: str) -> Optional[Entry]:
    return next((e for e in entries() if e.key == key), None)


def of_layer(spec: dict) -> Optional[Entry]:
    """The entry a scene layer is, or None (clips, footage, an unknown shader)."""
    if not isinstance(spec, dict):
        return None
    if spec.get("source") == "shadertoy" and spec.get("file"):
        return by_key(f"shadertoy:{Path(str(spec['file'])).stem}")
    return by_key(str(spec.get("source")))


def tree(kind: str, query: str = "") -> List[tuple]:
    """[(category, [entries])] of one kind, in the GUI's order; ``query`` filters by name,
    category or description (every word must match). A downloaded shader with an ``_arc``
    copy (the one that follows the music) is left out: the copy stands for it."""
    words = query.lower().split()
    cats = GENERATOR_CATEGORIES if kind == GENERATOR else FX_CATEGORIES
    keys = {e.key for e in entries()}
    out = []
    for cat in cats:
        found = [e for e in entries() if e.kind == kind and e.category == cat and f"{e.key}_arc" not in keys
                 and all(w in f"{e.label} {e.category} {e.about}".lower() for w in words)]
        if found:
            out.append((cat, sorted(found, key=lambda e: e.label)))
    return out

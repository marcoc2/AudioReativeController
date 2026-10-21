"""ARC Studio's Arrange window: what plays in the background and in the foreground, bar
by bar, and the FX on each.

Left, the browser: the Clips (a folder, a filter such as "final.mp4", the clip playing from
the point it would start at, muted until its sound is turned on; a clip is dragged from the
list (each with its picture and length) or its picture onto the BG, the FG or CLIPS, or
dropped there from the Explorer: arc_window.py), the Generators (they make a picture: a fractal, cells) and
the FX (they work on the picture they get), by category (core/catalog.py). Right, two
lanes over the song's bars (core/arrangement.py): the FG (the characters, drawn on their
own and put over the rest) and the BG (over the clip montage). Pick a clip or a Generator
and drag across a lane to put it there; things at the same time stack up in sub-rows.
FX live in three places: on a block (it alone), on a lane (all of it: the lane's FX
track) and on the master (over everything: its track). Everything drawn is a block that
is chosen by a click, moved by a drag, resized by a Shift+drag of its ends (the cursor
shows them; where two blocks meet, the one the mouse is in), deleted with Delete, and has
its menu on a right click; an FX's or a Generator's block opens its plugin (its parameters,
linked to the music) with a double click. The CLIPS row is the montage as it plays: a
click (or a drag) chooses bars; a clip picked on the left is pinned there, or nothing,
or they go back to the montage's choice; a clip of it can become a BG block.
Bars are chosen by a drag on the ruler (or on a lane, with nothing picked to put there);
choosing a block chooses its bars. Preview (or Space) plays the chosen bars with the
arrangement as it is, saved or not: the studio renders them (its progress, or why it
failed, shows in the preview window).
Members cued from REAPER show their cues in grey (edited in REAPER). A scene written as a
flat stack is grouped into BG / FG / master when opened; Save writes it so (SceneDoc,
anchors kept). Ctrl+Z undoes an edit, Ctrl+Y (or Ctrl+Shift+Z) redoes it; the history
lives until the window reloads the scene from disk.

``build(host)`` makes the window; ``open_window()`` shows it; ``tick()`` runs once a frame
on the main thread (textures). ``host`` gives what the window needs from the studio:
scene_path(), grid(), notes(), clips_dir(), duration(), log(msg), status(msg),
reload_editor().
"""
from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import dearpygui.dearpygui as dpg
import numpy as np

import arc_window
from core import arrangement as arr

LABEL_W = 150          # the lane names
CANVAS_W = 720                 # the canvas' narrowest: it widens with the window
RULER_H = 30
DENSITY_H = 10
LANE_H = 30                    # a row's height at the vertical zoom's 1 (its least with the notes)
LANE_MIN, LANE_MAX = 20, 120   # the vertical zoom's range
THUMB = 192
COVER = 72                     # a clip's picture in the list
COVERS_MAX = 400               # pictures made for a list (a filter narrows a bigger one)
NAME_CHARS = 27                # a name's line beside its picture
STATE_FILE = Path("render_output") / ".arrange.json"   # what the window remembers: the last clips folder
DEMO = 512                     # a demo's frame (core/demos.py)
DEMO_SHOWN = 300               # how big it plays under the catalog's list
SAM2_DEFAULTS = ("F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/models/sam2",)
UNDO_MAX = 200         # edits remembered

FG_FX, BG_FX, MASTER = "FG fx", "BG fx", "master"     # the tracks of FX: a lane's, the master's
TRACKS = (MASTER, arr.FG, FG_FX, arr.BG, BG_FX)       # top to bottom, over the montage
TRACK_TITLES = {MASTER: "MASTER", arr.FG: "FG", FG_FX: "FX on the whole FG", arr.BG: "BG",
                BG_FX: "FX on the whole BG"}


class _A:
    host = None
    doc = None                 # SceneDoc of the scene being arranged
    items: list = []           # [node or None, base or None, spec, uid] per layer, in stack order
    rows: list = []            # what each row of the canvas shows, top to bottom
    members: dict = {}         # track (FG, BG, FG fx, BG fx, master) -> [core.arrangement.Member]
    converted = False          # the scene was flat: grouped into BG / FG / master on opening
    content = None             # what a drag puts in a lane: ("clip", path, from) or ("gen", key)
    clips: list = []           # the browser's clips (Path)
    clip = None                # the chosen clip (str)
    clip_len = 0.0
    start = 0.0                # where in it a new block starts (s)
    sel = None                 # ("block", track, member, block) | ("lane", lane) | ("master",)
    #                            | ("montage",) | ("montage", first bar, last bar)
    runs: list = []            # the montage as it plays: [(first, last) bar positions, clip, path]
    span = None                # (first, last): the bars chosen (dragged on the ruler, or what is
    #                            chosen plays there): what Preview (Space) plays
    previewing = None          # (asked at, seen rendering): a preview the studio is making
    queued = None              # (first, last, asked at): bars to preview once the studio can
    zoom = 1.0                 # 1: the whole song across the canvas
    canvas_w = CANVAS_W        # the canvas' width: the window's, as it is resized
    lane_h = LANE_H            # a row's height: the vertical zoom
    scroll_to = None           # the lanes' scroll to set on the next frame (a vertical zoom's)
    view0 = 1                  # the first bar shown
    drag = None                # (row, first bar, mode, (member, block)) while dragging
    clip_play = None           # (path, start): what plays in the browser's picture
    clip_frame = None          # its next frame (RGBA floats), from the player's thread
    clip_pos = 0.0             # where it is (s in the clip)
    clip_restart = 0.0         # "from" moved at (s): it plays from there once it rests
    clip_sound = False         # its sound on (off: it plays muted)
    clip_audio: dict = {}      # path -> its sound (mono float32 at CLIP_SR; None: it has none)
    clip_mixer = None          # the clip's sound's own mixer (core.audio_player), beside the studio's
    clip_drag = None           # (path, start, seconds): a clip dragged from the browser to a lane
    sam = None                 # the SAM2 seeds window's state (open_sam), or None
    asking = None              # (lane, first bar, clips to put): a long clip waits for its bars
    list_items: dict = {}      # the list's rows (their picture, their name) -> their clips
    list_names: dict = {}      # the list's names (selectables) -> their clips
    list_pics: dict = {}       # a clip -> its pictures in the list
    covers: dict = {}          # a clip -> its picture's texture (made once)
    covers_ready: list = []    # (list's generation, clip, RGBA floats, seconds) from the worker
    covers_gen = 0             # the list's generation (a new list stops the old one's worker)
    demo_key = None            # the catalog entry whose demo plays
    demo_frame = None          # its next frame (RGBA floats), from the player's thread
    demo_making: dict = {}     # key -> the process making its demo
    busy = False
    cues: dict = {}            # a cued member's [(t0, t1)] seconds (REAPER / seconds cues)
    dirty = False
    redraw = False             # set by workers: the main thread redraws at its next tick
    relane = False
    pending = None             # (uid, fn): an edit a worker asks of the main thread
    entry = None               # the Generator or FX chosen in the browser (catalog key)
    plugin = None              # the layer (uid) the plugin window shows
    plugin_at = ()             # and where in it: () the layer, ("layers", 2, "fx", 0) a block's FX
    param_pick = None          # (uid, path): the parameter a folder/file dialog is choosing for
    catalog_drawn = False
    undo: list = []            # the arrangement before each edit, newest last
    redo: list = []            # what undo took back, newest last


A = _A()


# ----------------------------------------------------------------------- song and bars

def _bars() -> int:
    g = A.host.grid()
    db = getattr(g, "downbeats", None) if g is not None else None
    if db is not None and len(db):
        return int(len(db))
    dur = A.host.duration() or 0.0
    return max(16, int(dur / 2.0) + 1) if dur else 64


def _bar_seconds() -> float:
    g = A.host.grid()
    db = getattr(g, "downbeats", None) if g is not None else None
    if db is not None and len(db) > 1:
        return float(np.median(np.diff(db)))
    return 0.0


def _time_to_bar(t: float) -> float:
    """Song seconds -> bar position (1.0 = the start of bar 1)."""
    g = A.host.grid()
    db = getattr(g, "downbeats", None) if g is not None else None
    if db is None or not len(db):
        return 1.0 + t / 2.0
    i = int(np.searchsorted(db, t, side="right")) - 1
    if i < 0:
        return 1.0 + (t - db[0]) / max(1e-6, _bar_seconds() or 2.0)
    nxt = db[i + 1] if i + 1 < len(db) else db[i] + (_bar_seconds() or 2.0)
    return 1.0 + i + (t - db[i]) / max(1e-6, nxt - db[i])


def _shown() -> int:
    """How many bars the canvas shows (the zoom)."""
    return max(4, min(_bars(), int(round(_bars() / A.zoom))))


def _bw() -> float:
    return (A.canvas_w - LABEL_W - 8) / _shown()


def _x_of(bar: float) -> float:
    return LABEL_W + (bar - A.view0) * _bw()


def _bar_of_x(x: float) -> int:
    return int(np.clip(np.floor((x - LABEL_W) / _bw()) + A.view0, 1, _bars()))


def set_view(zoom: float = None, first: int = None, keep: int = None) -> None:
    """Zoom (1: the whole song) and the first bar shown; ``keep``: a bar that stays where
    it is on the screen while zooming."""
    if zoom is not None:
        x = _x_of(keep) if keep is not None else None
        A.zoom = float(np.clip(zoom, 1.0, max(1.0, _bars() / 4.0)))
        if keep is not None:
            first = int(round(keep - (x - LABEL_W) / _bw()))
    if first is not None:
        A.view0 = int(np.clip(first, 1, max(1, _bars() - _shown() + 1)))
    else:
        A.view0 = int(np.clip(A.view0, 1, max(1, _bars() - _shown() + 1)))
    if dpg.does_item_exist("arr_view0"):
        dpg.configure_item("arr_view0", max_value=max(1, _bars() - _shown() + 1))
        dpg.set_value("arr_view0", A.view0)
        dpg.set_value("arr_zoom_text", f"bars {A.view0}-{A.view0 + _shown() - 1} of {_bars()}")
    _draw()


def set_lane_height(h: float, keep_y: float = None) -> None:
    """The vertical zoom: a row's height; ``keep_y``: a point on the canvas that stays where
    it is on the screen (the row under the mouse)."""
    old = A.lane_h
    A.lane_h = int(np.clip(round(h), LANE_MIN, LANE_MAX))
    if dpg.does_item_exist("arr_rows_text"):
        dpg.set_value("arr_rows_text", f"{A.lane_h} px")
    if A.lane_h == old:
        return
    if keep_y is not None and dpg.does_item_exist("arr_tracks"):
        moved = (keep_y - _row_y(0)) * (A.lane_h / old - 1)
        A.scroll_to = max(0.0, dpg.get_y_scroll("arr_tracks") + moved)   # once the canvas has grown
    _draw()


def on_wheel(sender, delta) -> None:
    """Over the lanes: the wheel scrolls up and down, Ctrl+wheel zooms along the song around the
    bar under the mouse, Shift+wheel makes the rows taller or lower, Alt+wheel moves along."""
    if not delta or not dpg.does_item_exist("arr_canvas"):
        return
    if not (dpg.is_item_hovered("arr_canvas") or dpg.is_item_hovered("arr_tracks")):
        return
    if _down("mvKey_ModCtrl", "mvKey_LControl", "mvKey_RControl"):
        set_view(zoom=A.zoom * (1.25 if delta > 0 else 0.8), keep=_bar_of_x(_mouse_in_canvas()[0]))
    elif _down("mvKey_ModShift", "mvKey_LShift", "mvKey_RShift"):
        set_lane_height(A.lane_h * (1.2 if delta > 0 else 1 / 1.2), keep_y=_mouse_in_canvas()[1])
    elif _down("mvKey_ModAlt", "mvKey_LAlt", "mvKey_RAlt"):
        set_view(first=A.view0 - int(np.sign(delta)) * max(1, _shown() // 8))
    else:
        y = dpg.get_y_scroll("arr_tracks") - np.sign(delta) * 3 * A.lane_h
        dpg.set_y_scroll("arr_tracks", float(np.clip(y, 0, dpg.get_y_scroll_max("arr_tracks"))))


# ----------------------------------------------------------------------- the scene

def _load() -> None:
    from core.scene_doc import SceneDoc, plain, plain_keeping_signals
    path = A.host.scene_path()
    A.sel, A.dirty, A.cues, A.undo, A.redo = None, False, {}, [], []
    if not path or not Path(path).is_file():
        A.doc, A.items = None, []
        _status("no scene file: load or create one in the studio first")
        _relane()
        return
    A.doc = SceneDoc.load(path)
    nodes = A.doc.layers()
    specs = [plain_keeping_signals(node) for node in nodes]
    _name_from_comments(specs, A.doc.data["video"]["layers"])
    for spec, node in zip(specs, nodes):                 # a lane's members and FX too
        for key in ("layers", "fx"):
            if spec.get("source") == "group" and isinstance(node.get(key), list):
                _name_from_comments(spec.get(key) or [], node[key])
    known = {id(s): node for s, node in zip(specs, nodes)}
    layers, A.converted = arr.grouped(specs)
    A.items = [[known.get(id(s)), plain(known[id(s)]) if id(s) in known else None, s, _uid()] for s in layers]
    _relane()
    _status(f"arranging {Path(path).name}"
            + (" - a flat scene, grouped into BG / FG / master: Save writes it so" if A.converted else ""))
    _load_cues()


def _comment_names(seq) -> list:
    """Names for a YAML list of layers, from the scene's comments: the one at the end of a
    line of a layer (``file: lion.mp4  # the lion``), else the line just above it (its
    first clause: ``# the bottles sing the voice, swaying`` -> "the bottles sing the voice";
    a ``# ---- section`` line is no name)."""
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
        above = ""
        lead = seq.ca.items.get(k) if getattr(seq, "ca", None) is not None else None
        for tok in (lead[1] or []) if isinstance(lead, list) and len(lead) > 1 and isinstance(lead[1], list) else []:
            for line in (getattr(tok, "value", "") or "").split("\n"):
                line = line.strip().lstrip("#").strip()
                if line and not line.startswith("--"):
                    above = re.split(r"[:,;(]", line)[0].strip()
        out.append((eol or above)[:40])
    return out


def _name_from_comments(specs: list, seq) -> None:
    """A layer without a ``name:`` takes the one its comment gives it (written on Save)."""
    for spec, name in zip(specs, _comment_names(seq)):
        if name and isinstance(spec, dict) and not spec.get("name") and spec.get("source") not in ("group", "clips"):
            spec["name"] = name


_UIDS = iter(range(1, 1 << 62))


def _uid() -> int:
    """A layer's steady name in the window (its place in the stack changes)."""
    return next(_UIDS)


def _specs() -> list:
    return [it[2] for it in A.items]


def _lane_item(name: str):
    return next((it for it in A.items if it[2].get("source") == "group" and it[2].get("name") == name), None)


def _master_items() -> list:
    """What is over both lanes: the layers above the FG."""
    fg = _lane_item(arr.FG)
    return A.items[A.items.index(fg) + 1:] if fg is not None else []


def _montage_item():
    return next((it for it in A.items if it[2].get("source", "clips") == "clips"), None)


def _owner(key: str) -> str:
    """The lane a track belongs to (an FX track: its lane's), or the master."""
    return {FG_FX: arr.FG, BG_FX: arr.BG}.get(key, key)


def _is_fx(key: str) -> bool:
    """A track of FX: a lane's, the master."""
    return key in (FG_FX, BG_FX, MASTER)


def _track_specs(key: str) -> list:
    """The layers of a track, in order."""
    if key == MASTER:
        return [it[2] for it in _master_items()]
    item = _lane_item(_owner(key))
    return list(item[2].get("fx" if key in (FG_FX, BG_FX) else "layers") or []) if item else []


def _where(key: str, at: int):
    """(uid of the top-level layer, the path in it) of a track's member ``at``."""
    if key == MASTER:
        items = _master_items()
        return (items[at][3], ()) if 0 <= at < len(items) else (None, ())
    item = _lane_item(_owner(key))
    return (item[3] if item else None), ("fx" if key in (FG_FX, BG_FX) else "layers", at)


def _relane() -> None:
    """The rows, top to bottom: each track's sub-rows (things at the same time stack up),
    a lane's with an empty one more to put something new on; the clip montage last."""
    A.members, A.rows = {}, []
    if A.doc is not None:
        for key in TRACKS:
            if key != MASTER and _lane_item(_owner(key)) is None:
                continue
            mems = arr.members_of({"layers": _track_specs(key)})
            cue_bars = {m.at: [(int(_time_to_bar(t0)), int(_time_to_bar(t1))) for t0, t1 in A.cues[_cue_key(m.spec)]]
                        for m in mems if m.cued and A.cues.get(_cue_key(m.spec))}
            n = arr.pack(mems, _bars(), cue_bars)
            A.members[key] = mems
            A.rows += [{"kind": "lane", "lane": key, "sub": r} for r in range(max(1, n if _is_fx(key) else n + 1))]
        if _montage_item() is not None:
            A.rows.append({"kind": "montage"})
    _draw()
    _draw_inspector()
    _draw_plugin()


def _member(lane: str, at: int):
    return next((m for m in A.members.get(lane, []) if m.at == at), None)


def _load_cues() -> None:
    """The cued members' cues (seconds, or the REAPER edit) to show in grey: worked out in
    the background (a REAPER glue takes a few seconds to resolve)."""
    cued = sorted((m for ms in A.members.values() for m in ms if m.cued),       # the quick ones first
                  key=lambda m: isinstance(m.spec.get("cues"), dict))
    if not cued:
        return

    def run():
        from core.video.footage import read_cues
        for m in cued:
            try:
                A.cues[_cue_key(m.spec)] = [(c[0], c[0] + c[2]) for c in read_cues(m.spec.get("cues"),
                                                                                  m.spec.get("file"))]
            except Exception as exc:                    # shown, not fatal: the lane still edits
                A.host.log(f"Arrange: cues of {m.label} not read: {exc}")
            A.relane = True                              # known now: they pack by where they play
    threading.Thread(target=run, daemon=True).start()


def _cue_key(spec: dict) -> str:
    """Cues are kept by what they are, not by the member's place (members come and go)."""
    return f"{spec.get('file')}|{spec.get('cues')!r}"


from core.scene_doc import tidy as _tidy   # noqa: E402  (bars, blocks, cutouts on one line)


def btn_save() -> None:
    if A.doc is None:
        _status("nothing to save: no scene")
        return
    A.doc.set_layers([tuple(it[:3]) for it in A.items])
    for node in A.doc.layers():
        _tidy(node)
    saved = list(A.doc.layers())
    path = A.host.scene_path()
    try:
        A.doc.save(path)
    except OSError as exc:
        _status(f"NOT saved: {exc}")
        _show_saved(f"NOT saved: {exc}", error=True)
        return
    A.host.log(f"Arrange: saved {Path(path).name}")
    A.host.reload_editor()
    history = (A.undo, A.redo)
    _load()
    A.undo, A.redo = (_rebase(h, saved) for h in history)
    when = time.strftime("%H:%M:%S")
    A.saved_at, A.saved_flash = f"saved at {when} in {Path(path).name}", time.time()
    _status(f"saved at {when}: {path} - the studio's editor, previews and renders use it now")
    _show_saved()


SAVED_FLASH = 2.5      # s the Save button says it saved


def _show_saved(text: str = None, error: bool = False) -> None:
    """Beside Save: unsaved changes, or when and where it was saved (the button says
    "Saved!" a moment after a save)."""
    if not dpg.does_item_exist("arr_save_state"):
        return
    if text is None:
        text = "unsaved changes" if A.dirty else getattr(A, "saved_at", "") or ""
    colour = (240, 120, 110) if error else (240, 190, 90) if A.dirty else (130, 220, 140)
    if dpg.get_value("arr_save_state") != text:
        dpg.set_value("arr_save_state", text)
    dpg.configure_item("arr_save_state", color=colour)
    flash = time.time() - getattr(A, "saved_flash", 0.0) < SAVED_FLASH
    label = "Saved!" if flash and not A.dirty else "Save to scene *" if A.dirty else "Save to scene"
    if dpg.get_item_label("arr_save_btn") != label:
        dpg.configure_item("arr_save_btn", label=label)


def _rebase(history: list, saved: list) -> list:
    """The history, its layers moved onto the freshly loaded scene: a layer that was saved
    is the same layer in the file now; one that was not (deleted, then undone) comes back
    as a new layer."""
    now = {id(node): (A.items[i][0], A.items[i][1]) for i, node in enumerate(saved) if i < len(A.items)}
    return [([[*now.get(id(node), (None, None)), spec, uid] for node, _, spec, uid in items], sel)
            for items, sel in history]


# ----------------------------------------------------------------------- edits

def _snapshot() -> tuple:
    from core.scene_doc import plain_keeping_signals            # a copy; the signals stay shared
    return [[node, base, plain_keeping_signals(spec), uid] for node, base, spec, uid in A.items], A.sel


def _checkpoint() -> None:
    """Remember the arrangement as it is, before an edit changes it."""
    A.undo.append(_snapshot())
    del A.undo[:-UNDO_MAX]
    A.redo.clear()


def _step(back: bool) -> None:
    src, dst = (A.undo, A.redo) if back else (A.redo, A.undo)
    if not src or A.drag is not None:
        _status("nothing to undo" if back else "nothing to redo")
        return
    dst.append(_snapshot())
    A.items, A.sel = src.pop()
    A.dirty = True
    _relane()
    _status(("undone" if back else "redone") + f" · {len(A.undo)} to undo, {len(A.redo)} to redo")


def btn_undo() -> None:
    _step(True)


def btn_redo() -> None:
    _step(False)


def _typing() -> bool:
    """A text or number field has the keys (Ctrl+Z there undoes its own typing)."""
    item = dpg.get_focused_item()
    try:
        return bool(item) and dpg.is_item_active(item) and "Input" in dpg.get_item_type(item)
    except Exception:
        return False


def _down(*keys) -> bool:
    return any(dpg.is_key_down(getattr(dpg, k)) for k in keys if hasattr(dpg, k))


def _shortcut() -> bool:
    """Ctrl is held, the window is open and no field is being typed in."""
    return (dpg.does_item_exist("arr_win") and dpg.is_item_shown("arr_win") and not _typing()
            and _down("mvKey_ModCtrl", "mvKey_LControl", "mvKey_RControl"))


def on_key_z(sender, app_data) -> None:
    if _shortcut():
        _step(not _down("mvKey_ModShift", "mvKey_LShift", "mvKey_RShift"))


def on_key_y(sender, app_data) -> None:
    if _shortcut():
        _step(False)


def _write(item, spec: dict) -> bool:
    """A top-level layer's new spec (one undo step); False when nothing changed."""
    if spec == item[2]:
        return False
    _checkpoint()
    item[2] = spec
    A.dirty = True
    return True


def _edit(uid, path: tuple, fn) -> None:
    """``fn(x)`` -> the new value of what is at ``path`` in the layer ``uid`` (on a copy)."""
    from core.scene_doc import get_path, plain_keeping_signals
    item = _item(uid)
    if item is None:
        return
    spec = plain_keeping_signals(item[2])
    if not path:
        spec = fn(spec)
    else:
        holder = get_path(spec, path[:-1])
        new = fn(holder[path[-1]] if isinstance(holder, list) else holder.get(path[-1]))
        if isinstance(holder, dict) and new in (None, []) and path[-1] != "layers":
            holder.pop(path[-1], None)                   # an emptied list goes
        else:
            holder[path[-1]] = new
    if _write(item, spec):
        _relane()


def _set_member(key: str, at: int, spec: dict) -> None:
    uid, path = _where(key, at)
    _edit(uid, path, lambda _: spec)


def _set_blocks(key: str, at: int, blocks: list) -> None:
    mem = _member(key, at)
    if not blocks:                                       # no window left: the member goes
        _remove_member(key, at)
        return
    _set_member(key, at, (arr.with_blocks if mem.kind == arr.CLIP else arr.with_windows)(mem.spec, blocks))


def _remove_member(key: str, at: int) -> None:
    uid, path = _where(key, at)
    A.sel = ("master",) if key == MASTER else ("lane", _owner(key))
    if key == MASTER:
        _delete_master(uid)
    else:
        _edit(uid, path[:1], lambda ls: [x for j, x in enumerate(ls or []) if j != at])
    _draw_inspector()


def _content_spec(lane: str, first: int, last: int):
    """A new member from what the browser has chosen: a clip, or a Generator."""
    from core import catalog
    if A.content is None:
        return None
    if A.content[0] == "gen":
        e = catalog.by_key(A.content[1])
        if e is None:
            return None
        spec = {**e.spec(), "bars": [[first, last]]}
        folder = _clips_folder()
        if folder and any(p.name == "clips_dir" for p in e.params):     # its own clips: these
            spec["clips_dir"] = folder
        return spec
    clip, start = A.content[1], A.content[2]
    block = arr.Block(first, last, start)
    if lane == arr.BG:
        return arr.with_blocks({"source": "footage", "role": "background", "fit": "cover", "loop": True,
                                "file": clip}, [block])
    return arr.new_foreground(clip, [block])


def _new_member(key: str, first: int, last: int, sub=None) -> None:
    """What is picked on the left, new in ``key`` over bars first..last; ``sub``: the sub-row
    it was made on (it stays there, under the rows above it where they overlap)."""
    if _is_fx(key):                                      # an FX track: the FX picked on the left
        from core import catalog
        e = catalog.by_key(A.entry) if A.entry else None
        if e is None or e.kind != catalog.FX:
            _status("pick an FX on the left (FX) first, or right-click the row")
            return
        _put_fx(key, {**e.spec(), "bars": [[first, last]]})
        return
    spec = _content_spec(key, first, last)
    if spec is None:
        _status("pick a clip (Clips) or a Generator (Generators) on the left first")
        return
    at = _insert_member(key, spec, sub, first, last)
    A.sel = ("block", key, at, 0)
    _draw()
    _draw_inspector()
    _status(f"{arr.label_of(spec)} in {key}, bars {first}-{last}"
            + (" - no mask track yet: Segment it to cut it out"
               if key == arr.FG and spec.get("source") == "footage" and not spec.get("cutout") else ""))


def add_fx(uid, path: tuple, key: str) -> None:
    """An FX from the catalog at the end of the list at ``path`` (a block's, a lane's)."""
    from core import catalog
    e = catalog.by_key(key)
    if e is not None:
        _edit(uid, path, lambda fx: list(fx or []) + [e.spec()])
        _status(f"{e.label} added: it plays all the time its owner does - give it bars in the list")
        if key == SAM:                                   # its seeds, first
            from core.scene_doc import get_path
            it = _item(uid)
            got = get_path(it[2], path) if it is not None else None
            if isinstance(got, list) and got:
                open_sam(uid, tuple(path) + (len(got) - 1,), new=True)


def add_master(key: str) -> None:
    """An FX over everything, all the song (its bars are given on its track)."""
    from core import catalog
    e = catalog.by_key(key)
    if A.doc is not None and e is not None:
        _put_fx(MASTER, e.spec())


def _put_fx(key: str, spec: dict) -> None:
    """An FX at the end of a track's chain (the master's: under a ``meters`` read-out); it
    is chosen."""
    if A.doc is None:
        return
    if key == MASTER:
        at = len(A.items)
        while at > 0 and A.items[at - 1][2].get("source") == "meters":
            at -= 1
        _checkpoint()
        item = [None, None, spec, _uid()]
        A.items.insert(at, item)
        A.dirty = True
        k = next(i for i, it in enumerate(_master_items()) if it is item)
    else:
        lane = _lane_item(_owner(key))
        k = len(lane[2].get("fx") or [])
        _edit(lane[3], ("fx",), lambda fx: list(fx or []) + [spec])
    A.sel = ("block", key, k, 0 if spec.get("bars") else None)
    _relane()
    bars = arr.windows_of(spec)
    _status(f"{arr.label_of(spec)} on {TRACK_TITLES[key]}"
            + (f", bars {bars[0].first}-{bars[-1].last}" if bars else ", all the song: drag its ends... or give it bars below"))


def _reorder(key: str, at: int, d: int) -> None:
    """An FX earlier (d -1) or later (+1) in its chain."""
    n = len(_track_specs(key))
    if not 0 <= at + d < n:
        return
    if key == MASTER:
        _move_master(_where(key, at)[0], d)
    else:
        _swap(_where(key, at)[0], ("fx",), at, at + d)
    if A.sel and A.sel[0] == "block" and A.sel[1:3] == (key, at):
        A.sel = ("block", key, at + d, A.sel[3])
    _draw()
    _draw_inspector()


def _toggle(uid, path: tuple) -> None:
    def flip(spec):
        spec = dict(spec)
        if spec.get("enabled", True) is False:
            spec.pop("enabled")
        else:
            spec["enabled"] = False
        return spec
    _edit(uid, path, flip)


def _move_master(uid, d: int) -> None:
    items = _master_items()
    k = next(i for i, it in enumerate(items) if it[3] == uid)
    j = k + d
    if not 0 <= j < len(items):
        return
    _checkpoint()
    a, b = A.items.index(items[k]), A.items.index(items[j])
    A.items[a], A.items[b] = A.items[b], A.items[a]
    A.dirty = True
    _relane()


def _delete_master(uid) -> None:
    _checkpoint()
    A.items = [it for it in A.items if it[3] != uid]
    A.dirty = True
    _relane()


def _close_menu() -> None:
    if dpg.does_item_exist("arr_menu"):
        dpg.delete_item("arr_menu")


def _catalog_menu(label: str, kind: str, pick) -> None:
    """A menu of the catalog's entries of ``kind``, by category; ``pick(key)`` on a choice."""
    from core import catalog
    with dpg.menu(label=label):
        for cat, entries in catalog.tree(kind):
            with dpg.menu(label=cat):
                for e in entries:
                    flags = (" *" if e.takes_input else "") + "".join(f" ({n})" for n in e.needs)
                    item = dpg.add_menu_item(label=e.label + flags, user_data=e.key,
                                             callback=lambda s, a, key: (_close_menu(), pick(key)))
                    with dpg.tooltip(item):
                        dpg.add_text(e.about, wrap=360)


def _menu_at(builder) -> None:
    _close_menu()
    with dpg.window(tag="arr_menu", popup=True, no_title_bar=True, autosize=True,
                    pos=dpg.get_mouse_pos(local=False)):
        builder()


def _put_generator(lane: str, bar: int, key: str) -> None:
    A.content = ("gen", key)
    _show_content()
    _new_member(lane, bar, min(_bars(), bar + 3))


def _put_fx_at(key: str, bar: int, fx_key: str) -> None:
    from core import catalog
    e = catalog.by_key(fx_key)
    if e is not None:
        _put_fx(key, {**e.spec(), "bars": [[bar, min(_bars(), bar + 3)]]})


def context_menu(x: float, y: float) -> None:
    """The right click: on a block (any track), its menu; on a track, a Generator or an FX
    put there; on the montage, its bars' clips; anywhere, an FX on the master."""
    from core import catalog
    if A.doc is None:
        return
    r = _row_at(y)
    row = A.rows[r] if r is not None else {"kind": "none"}
    bar = _bar_of_x(x)
    if row["kind"] == "montage" and x >= LABEL_W:
        if not (A.sel and A.sel[0] == "montage" and len(A.sel) > 2 and A.sel[1] <= bar <= A.sel[2]):
            A.sel = _run_sel(bar)[0]                     # the clip under the mouse is chosen
            _draw()
            _draw_inspector()

    def build():
        if row["kind"] == "montage" and x >= LABEL_W and A.sel and len(A.sel) > 2:
            first, last = A.sel[1], A.sel[2]
            dpg.add_text(f"CLIPS · bars {first}-{last}" if last > first else f"CLIPS · bar {first}",
                         color=LANE_COLOURS["montage"])
            for label, fn in _montage_actions(first, last):
                dpg.add_menu_item(label=label, callback=lambda s, a, *, fn=fn: (_close_menu(), fn()))
            dpg.add_separator()
        if row["kind"] == "lane":
            key = row["lane"]
            hit = _block_at(r, bar)
            mem = None if x < LABEL_W else _member(key, hit[0]) if hit is not None else _unblocked_at(r, x)
            uid, here = _where(key, mem.at) if mem is not None else (None, ())
            if mem is not None:                          # a block, or a cued / whole-song member
                dpg.add_text(f"{TRACK_TITLES[key]} · {mem.label}", color=LANE_COLOURS[key][:3])
                if not _is_fx(key):
                    _catalog_menu("Add FX to it", catalog.FX, lambda k: add_fx(uid, here + ("fx",), k))
                if mem.kind != arr.CLIP:
                    dpg.add_menu_item(label="Open plugin (its parameters, linked to the music)...",
                                      callback=lambda: (_close_menu(), open_plugin(uid, here)))
                dpg.add_menu_item(label="Turn on" if not mem.enabled else "Turn off (bypass)",
                                  callback=lambda: (_close_menu(), _toggle(uid, here)))
                for label, fn in _copy_paste_choices(key, mem):   # its links, its FX: to others
                    dpg.add_menu_item(label=label, callback=lambda s, d, *, fn=fn: (_close_menu(), fn()))
                if hit is None:                          # cued / whole song: hidden in some bars
                    for label, (a, b, show) in _mute_choices(mem, bar):
                        dpg.add_menu_item(label=label, callback=lambda s, d, *, a=a, b=b, show=show: (
                            _close_menu(), mute_bars(key, mem.at, a, b, not show)))
                if _is_fx(key):
                    dpg.add_menu_item(label="Earlier in the chain (works before)",
                                      callback=lambda: (_close_menu(), _reorder(key, mem.at, -1)))
                    dpg.add_menu_item(label="Later in the chain (works after)",
                                      callback=lambda: (_close_menu(), _reorder(key, mem.at, 1)))
                if hit is not None:
                    blk = mem.blocks[hit[1]]
                    first, last = blk.first, blk.last
                else:
                    spans = _cue_bars(mem) or [(1, _bars())]
                    first, last = int(spans[0][0]), int(spans[-1][1])
                dpg.add_menu_item(label=f"Preview its bars ({first}-{min(last, first + PREVIEW_BARS - 1)})",
                                  callback=lambda: (_close_menu(), preview_bars(first, last)))
                if hit is not None and len(mem.blocks) > 1:
                    dpg.add_menu_item(label="Delete this block (Delete)", callback=lambda: (
                        _close_menu(), _delete_block(key, mem.at, hit[1])))
                dpg.add_menu_item(label=f"Delete {mem.label}" + ("" if hit is not None and len(mem.blocks) > 1
                                                                   else " (Delete)"),
                                  callback=lambda: (_close_menu(), _remove_member(key, mem.at)))
                dpg.add_separator()
            else:
                dpg.add_text(TRACK_TITLES[key], color=LANE_COLOURS[key][:3])
            if not _is_fx(key):
                _catalog_menu(f"Put a Generator here (bar {bar})", catalog.GENERATOR,
                              lambda k: _put_generator(key, bar, k))
            fx_key = key if _is_fx(key) else {arr.FG: FG_FX, arr.BG: BG_FX}[key]
            where = "the master" if fx_key == MASTER else f"the whole {_owner(key)}"
            _catalog_menu(f"Add FX on {where} here (bars {bar}-{min(_bars(), bar + 3)})", catalog.FX,
                          lambda k: _put_fx_at(fx_key, bar, k))
            _catalog_menu(f"Add FX on {where}, all the song", catalog.FX,
                          lambda k: _put_fx(fx_key, catalog.by_key(k).spec()))
            dpg.add_separator()
        _catalog_menu("Add FX to the master (over everything)", catalog.FX, add_master)
    _menu_at(build)


def _delete_block(key: str, at: int, i: int) -> None:
    mem = _member(key, at)
    if mem is None:
        return
    if mem.cued or mem.always or len(mem.blocks) <= 1:
        _remove_member(key, at)
        return
    _set_blocks(key, at, arr.remove(mem.blocks, i))
    A.sel = ("master",) if key == MASTER else ("lane", _owner(key))
    _draw_inspector()


def delete_selected() -> None:
    """Delete: the chosen block (a member's last: the member), a cued member, or the
    montage's chosen bars (nothing there)."""
    sel = A.sel or (None,)
    if sel[0] == "block" and _member(sel[1], sel[2]) is not None:
        if sel[3] is None:
            _remove_member(sel[1], sel[2])
        else:
            _delete_block(sel[1], sel[2], sel[3])
    elif sel[0] == "montage" and len(sel) > 2:
        _set_pins(sel[1], sel[2], "none")


def _in_window() -> bool:
    """The mouse or the keys are on the Arrange window (not on the studio behind it)."""
    tags = ("arr_win", "arr_tracks", "arr_insp_win")
    return any(dpg.does_item_exist(t) and (dpg.is_item_hovered(t) or dpg.is_item_focused(t)) for t in tags)


def on_key_delete(sender, app_data) -> None:
    if dpg.does_item_exist("arr_win") and dpg.is_item_shown("arr_win") and not _typing() and _in_window():
        delete_selected()


# ----------------------------------------------------------------------- mouse

def _mouse_in_canvas():
    mx, my = dpg.get_mouse_pos(local=False)
    rx, ry = dpg.get_item_rect_min("arr_canvas")
    return mx - rx, my - ry


def hover_text(x: float, y: float) -> str:
    """What is under the mouse at (x, y) on the canvas, named: the clip the montage plays
    there, a member (its block's bars), else ''."""
    r = _row_at(y)
    if r is None or x < LABEL_W:
        return ""
    row = A.rows[r]
    pos = (x - LABEL_W) / _bw() + A.view0                # the bar position under the mouse
    if row["kind"] == "montage":
        run = next((u for u in A.runs if u[0] <= pos < u[1]), None)
        if run is None:
            return ""
        pinned = "  (pinned)" if int(pos) in _pins() else ""
        return f"bar {int(pos)}: {run[2]}{pinned}" + (f"\n{run[3]}" if run[3] else "")
    key = row["lane"]
    hit = _block_at(r, int(pos))
    if hit is not None:
        mem = _member(key, hit[0])
        blk = mem.blocks[hit[1]]
        return f"{mem.label} ({mem.kind}), bars {blk.first}-{blk.last}"
    mem = _unblocked_at(r, x)
    if mem is None:
        return ""
    hidden = " - hidden in this bar" if any(a <= int(pos) <= b for a, b in _mutes(mem.spec)) else ""
    return f"{mem.label} ({mem.kind}), " + ("cut by its cues (REAPER)" if mem.cued else "the whole song") + hidden


def on_mouse_move(sender, app_data) -> None:
    if not dpg.does_item_exist("arr_tip") or not dpg.is_item_shown("arr_win"):
        return
    text = hover_text(*_mouse_in_canvas()) if dpg.is_item_hovered("arr_canvas") and A.drag is None else ""
    dpg.set_value("arr_tip_text", text)
    dpg.configure_item("arr_tip", show=bool(text))


def _row_y(r: int) -> float:
    return RULER_H + DENSITY_H + 6 + r * A.lane_h


def _scroll() -> float:
    """How far the lanes are scrolled down (the ruler stays on top of what shows)."""
    try:
        return float(dpg.get_y_scroll("arr_tracks"))
    except Exception:
        return 0.0


def _in_ruler(y: float) -> bool:
    """``y`` (on the canvas) is on the ruler, wherever the lanes are scrolled to."""
    return y - _scroll() < _row_y(0)


def _row_at(y: float):
    if _in_ruler(y):                                     # a row scrolled under the ruler
        return None
    r = int((y - _row_y(0)) // A.lane_h)
    return r if 0 <= r < len(A.rows) else None


def _block_at(r: int, bar: int):
    """(member's place, block index) drawn at ``bar`` on row ``r``, or None."""
    row = A.rows[r]
    if row["kind"] != "lane":
        return None
    for mem in A.members.get(row["lane"], []):
        if mem.row == row["sub"]:
            i = arr.block_at(mem.blocks, bar)
            if i is not None:
                return mem.at, i
    return None


def on_canvas_click(sender, app_data) -> None:
    if app_data[0] == 1:                                 # the right button: the menu
        context_menu(*_mouse_in_canvas())
    elif app_data[0] == 0:
        press(*_mouse_in_canvas())


def on_canvas_double_click(sender, app_data) -> None:
    """A double click on a Generator's or FX's block: its plugin."""
    x, y = _mouse_in_canvas()
    r = _row_at(y)
    if r is None or A.rows[r]["kind"] != "lane" or x < LABEL_W:
        return
    key = A.rows[r]["lane"]
    hit = _block_at(r, _bar_of_x(x))
    mem = _member(key, hit[0]) if hit is not None else _unblocked_at(r, x)
    if mem is not None and mem.kind != arr.CLIP:
        open_plugin(*_where(key, mem.at))


def on_mouse_release(sender, app_data) -> None:
    if A.clip_drag is not None:                          # a clip from the browser
        drag, A.clip_drag = A.clip_drag, None
        if dpg.does_item_exist("arr_ghost"):
            dpg.configure_item("arr_ghost", show=False)
        if _over_lanes():
            drop_clip(*_mouse_in_canvas(), drag)
        return
    if A.drag is None or not dpg.is_item_shown("arr_win"):
        A.drag = None
        return
    release(*_mouse_in_canvas())


def on_mouse_drag(sender, app_data) -> None:
    if A.clip_drag is not None:
        _show_clip_ghost()
        return
    if A.drag is None or not dpg.does_item_exist("arr_ghost"):
        return
    mx, my = _mouse_in_canvas()
    got = _dragged(mx)
    if got is None:
        return
    r, a, b = got
    if A.drag[2] == "move" and r >= 0:                   # onto another row: there
        t = _target_row(A.rows[r]["lane"], my)
        r = r if t is None else t
    y0, y1 = ((_scroll(), _scroll() + RULER_H + DENSITY_H) if r < 0 else
              (_row_y(r) + 2, _row_y(r) + A.lane_h - 4))
    dpg.configure_item("arr_ghost", pmin=[_x_of(a), y0], pmax=[_x_of(b + 1), y1], show=True)


EDGE = 8               # pixels at a block's ends that Shift+drag takes (its end, not the block)


def press(x: float, y: float) -> None:
    """The mouse goes down on the canvas: on a row's name it picks the lane (or the
    master, the montage); with Shift, on a block's end it takes that end (``_edge_at``); on
    a block the block, elsewhere on a track it starts a new member there; on the montage,
    bars to choose."""
    if _in_ruler(y) and x >= LABEL_W:                    # the ruler: bars to choose
        A.drag = (-1, _bar_of_x(x), "span", None)
        return
    r = _row_at(y)
    if r is None:
        return
    row = A.rows[r]
    if x < LABEL_W:
        A.sel = (("montage",) if row["kind"] == "montage" else ("master",) if row["lane"] == MASTER
                 else ("lane", _owner(row["lane"])))
        _draw()
        _draw_inspector()
        return
    bar = _bar_of_x(x)
    if row["kind"] == "montage":
        A.drag = (r, bar, "range", None)
        return
    key = row["lane"]
    edge = _edge_at(x, y) if _shift() else None
    if edge is not None:                                 # Shift: an end, to resize the block
        A.drag = (r, bar, edge[1], edge[2])
        return
    hit = _block_at(r, bar)
    if hit is None:
        other = _unblocked_at(r, x)
        if other is not None:                            # cued (REAPER) or the whole song: chosen
            A.sel = ("block", key, other.at, None)
            _draw()
            _draw_inspector()
            return
    A.drag = (r, bar, "new" if hit is None else "move", hit)


def _insert_member(key: str, spec: dict, sub, first: int, last: int) -> int:
    """``spec`` into the lane ``key`` (a lane's FX track: at the end of its chain), on the
    sub-row ``sub`` (None: the topmost free one), under what is above it there over bars
    first..last; -> its place."""
    item = _lane_item(_owner(key))
    where = "fx" if _is_fx(key) else "layers"
    at = len(item[2].get(where) or [])
    if sub is not None:
        spec = {**spec, arr.ROW_KEY: int(sub)}
        if not _is_fx(key):
            above = [m.at for m in A.members.get(key, []) if m.row < sub
                     and any(a <= last and first <= b for a, b in _spans_of(m))]
            at = min(above, default=at)                  # drawn under what is above it
    _edit(item[3], (where,), lambda ls, *, at=at, spec=spec: list(ls or [])[:at] + [spec] + list(ls or [])[at:])
    return at


def _recast(spec: dict, lane: str) -> dict:
    """A clip moved to the other lane: a character in the FG (in its place in the frame), the
    whole frame in the BG."""
    out = dict(spec)
    if spec.get("source") != "footage":
        return out
    if lane == arr.FG:
        out["role"] = "foreground"
        if out.get("fit", "contain") == "cover":
            out["fit"] = "contain"
        for k, v in (("scale", 0.6), ("x", 0.5), ("y", 0.62)):
            out.setdefault(k, v)
    else:
        out.update(role="background", fit="cover")
        for k in ("scale", "x", "y"):
            out.pop(k, None)
    return out


def _target_row(key: str, y: float):
    """The row a block of the lane ``key`` dragged to ``y`` would go to: another sub-row of
    it, or one of the other lane of its kind (BG / FG; a lane's FX track); None: its own."""
    t = _row_at(y)
    if t is None or A.rows[t]["kind"] != "lane":
        return None
    dst = A.rows[t]["lane"]
    kinds = ((arr.BG, arr.FG), (FG_FX, BG_FX))
    if not any(key in k and dst in k for k in kinds):
        return None
    return t


def _move_to_row(key: str, mem, i: int, first: int, last: int, t: int) -> None:
    """Block ``i`` of ``mem`` (in ``key``) dragged to bars first..last on row ``t``: there
    now. The member goes whole when it was its only block, else that block alone becomes a
    member of its own; to the other lane a clip changes its role (``_recast``). One undo."""
    from core.scene_doc import plain
    dst, sub = A.rows[t]["lane"], A.rows[t]["sub"]
    b = mem.blocks[i]
    moved = arr.Block(first, last, b.start, b.rate, b.file)
    shape = arr.with_blocks if mem.kind == arr.CLIP else arr.with_windows
    spec = shape({k: v for k, v in plain(mem.spec).items() if k != arr.ROW_KEY}, [moved])
    rest = [x for j, x in enumerate(mem.blocks) if j != i]
    n = len(A.undo)
    if dst == key and not rest:                          # its own lane: another sub-row
        _set_member(key, mem.at, {**spec, arr.ROW_KEY: int(sub)})
        at = mem.at
    else:
        if dst != key:
            spec = _recast(spec, dst)
        if rest:
            _set_blocks(key, mem.at, rest)
        else:
            _remove_member(key, mem.at)
        at = _insert_member(dst, spec, sub, first, last)
    del A.undo[n + 1:]                                   # one step back undoes all of it
    A.sel = ("block", dst, at, 0)
    _status(f"{mem.label} -> {TRACK_TITLES[dst]}, row {sub + 1}, bars {first}-{last}"
            + (" (Ctrl+Z puts it back)" if dst != key else ""))


def _spans_of(mem) -> list:
    """The bars a member is on: its blocks, its cues, or all of the song."""
    if mem.blocks:
        return [(b.first, b.last) for b in mem.blocks]
    cues = _cue_bars(mem)
    return [(int(a), int(b)) for a, b in cues] if cues else [(1, _bars())]


def _shift() -> bool:
    return _down("mvKey_ModShift", "mvKey_LShift", "mvKey_RShift")


def _edge_at(x: float, y: float):
    """(row, "first" | "last", (member, block), the end's x) Shift+drag takes at (x, y): an
    end within EDGE px; where two blocks meet, the one the mouse is in (never the neighbour's)."""
    if x < LABEL_W or _in_ruler(y):
        return None
    r = _row_at(y)
    if r is None or A.rows[r]["kind"] != "lane":
        return None
    row, bar = A.rows[r], _bar_of_x(x)
    near = []
    for mem in A.members.get(row["lane"], []):
        if mem.row != row["sub"]:
            continue
        for i, b in enumerate(mem.blocks):
            for end, ex in (("last", _x_of(b.last + 1)), ("first", _x_of(b.first))):
                if abs(x - ex) <= EDGE:
                    inside = b.first <= bar <= b.last
                    near.append((not inside, abs(x - ex), end, (mem.at, i), ex))
    if not near:
        return None
    _, _, end, hit, ex = min(near, key=lambda n: n[:2])
    return r, end, hit, ex


def _show_edge() -> None:
    """With Shift held over a block's end (or while one is dragged): the left-right cursor
    and that end lit; else neither."""
    got = None
    resizing = A.drag is not None and A.drag[2] in ("first", "last")
    if dpg.does_item_exist("arr_edge") and dpg.is_item_shown("arr_win"):
        if resizing:
            got = True
        elif _shift() and dpg.is_item_hovered("arr_canvas") and A.clip_drag is None and A.asking is None:
            got = _edge_at(*_mouse_in_canvas())
    if isinstance(got, tuple):
        r, _, _, ex = got
        dpg.configure_item("arr_edge", p1=[ex, _row_y(r) + 1], p2=[ex, _row_y(r) + A.lane_h - 2], show=True)
    elif not resizing and dpg.does_item_exist("arr_edge"):
        dpg.configure_item("arr_edge", show=False)
    arc_window.set_cursor("resize" if got else None)


# ----------------------------------------------------------------------- copy and paste: links, FX

NOT_LINKS = {"source", "file", "name", "role", "blocks", "bars", "cues", "cutout", "fx", "mute", "lane_row",
             "enabled", "fit", "loop", "flip", "spill", "fade_in", "fade_out", "layers", "canvas", "pins",
             "blend", "hold", "mask"}
PLACE_KEYS = {"x", "y", "scale", "opacity"}     # what any clip can be given


def _is_link(v) -> bool:
    """A value the music moves (a knob: a pulse on the kick, an LFO, the notes' pitch...)."""
    from core.knob_forms import form_of
    return isinstance(v, dict) and form_of(v) not in (None, "constant")


def links_of(spec: dict) -> dict:
    """{path: value}: what in a layer the music moves: its knobs (a shader's uniforms too)
    and who plays it (``trigger``, ``triggers``)."""
    out = {}
    for k, v in (spec or {}).items():
        if k in NOT_LINKS:
            continue
        if k == "uniforms" and isinstance(v, dict):
            out.update({("uniforms", u): uv for u, uv in v.items() if _is_link(uv)})
        elif _is_link(v) or k in ("trigger", "triggers") and isinstance(v, dict):
            out[(k,)] = v
    return out


def _takes(spec: dict, path: tuple) -> bool:
    """``spec`` (a layer) has room for the link at ``path``: a key it has or its kind takes
    (a clip's place and size; a Generator's or FX's own parameters; a shader's uniform)."""
    from core import catalog
    key = path[0]
    if key == "uniforms":
        return isinstance(spec.get("uniforms"), dict) and path[1] in spec["uniforms"] or \
            (spec.get("source") == "shadertoy" and any(p.name == path[1] for p in
                                                       (catalog.of_layer(spec).params if catalog.of_layer(spec) else [])))
    if key in spec or spec.get("source") == "footage" and key in PLACE_KEYS or key == "opacity":
        return True
    e = catalog.of_layer(spec)
    return e is not None and any(p.name == key for p in e.params)


def _fx_list(key: str, mem) -> list:
    """The FX a block carries (an FX track's block: itself)."""
    from core.scene_doc import plain
    if _is_fx(key) or mem.kind == arr.FX:
        return [plain(mem.spec)]
    return [plain(f) for f in mem.spec.get("fx") or [] if isinstance(f, dict)]


def _names(paths) -> str:
    return ", ".join(p[-1] for p in paths)


def _copy_paste_choices(key: str, mem) -> list:
    """[(menu label, fn)]: copy this block's links or FX; paste the ones copied before."""
    out = []
    links = links_of(mem.spec)
    if links:
        out.append((f"Copy its music links ({_names(links)})", lambda: copy_links(key, mem.at)))
    fx = _fx_list(key, mem)
    if fx:
        what = "this FX" if (_is_fx(key) or mem.kind == arr.FX) else f"its FX ({len(fx)})"
        out.append((f"Copy {what}: {', '.join(arr.label_of(f) for f in fx)}", lambda: copy_fx(key, mem.at)))
    board = getattr(A, "board_links", None)
    if board:
        fits = [p for p in board["links"] if _takes(mem.spec, p)]
        if fits:
            out.append((f"Paste music links from {board['from']} ({_names(fits)})",
                        lambda: paste_links(key, mem.at)))
    board = getattr(A, "board_fx", None)
    if board and not _is_fx(key):
        out.append((f"Paste FX from {board['from']} ({', '.join(arr.label_of(f) for f in board['fx'])})",
                    lambda: paste_fx(key, mem.at)))
    return out


def copy_links(key: str, at: int) -> None:
    import copy
    mem = _member(key, at)
    links = links_of(mem.spec) if mem else {}
    if links:
        from core.scene_doc import plain
        A.board_links = {"from": mem.label, "links": {p: copy.deepcopy(plain(v)) for p, v in links.items()}}
        _status(f"copied {mem.label}'s music links ({_names(links)}): right-click another block, Paste")


def copy_fx(key: str, at: int) -> None:
    import copy
    mem = _member(key, at)
    fx = _fx_list(key, mem) if mem else []
    if fx:
        A.board_fx = {"from": mem.label, "fx": copy.deepcopy(fx), "file": mem.spec.get("file")}
        _status(f"copied {mem.label}'s FX ({', '.join(arr.label_of(f) for f in fx)}): right-click another "
                "block, Paste")


def paste_links(key: str, at: int) -> None:
    """The links copied before, on this block (the ones its kind takes; the rest said). One undo."""
    import copy
    mem, board = _member(key, at), getattr(A, "board_links", None)
    if mem is None or not board:
        return
    spec = copy.deepcopy(mem.spec)
    put, left = [], []
    for path, v in board["links"].items():
        if not _takes(spec, path):
            left.append(path)
            continue
        if path[0] == "uniforms":
            spec.setdefault("uniforms", {})[path[1]] = copy.deepcopy(v)
        else:
            spec[path[0]] = copy.deepcopy(v)
        put.append(path)
    if put:
        _set_member(key, at, spec)
    _status(f"{mem.label}: {_names(put) or 'nothing'} now follow{'s' if len(put) == 1 else ''} the music as on "
            f"{board['from']}"
            + (f" ({_names(left)}: not something it has)" if left else ""))


def paste_fx(key: str, at: int) -> None:
    """The FX copied before, after this block's own (over all its bars: a copied FX's bars
    were the other block's). A SAM2 cutout's seeds were drawn on the other clip: here SAM2
    finds the objects by itself. One undo."""
    import copy
    mem, board = _member(key, at), getattr(A, "board_fx", None)
    if mem is None or not board:
        return
    fx, note = [], ""
    for f in copy.deepcopy(board["fx"]):
        f.pop("bars", None)
        f.pop(arr.ROW_KEY, None)
        if f.get("source") == SAM:
            if mem.spec.get("source") != "footage":
                continue
            if isinstance(f.get("seeds"), dict) and mem.spec.get("file") != board.get("file"):
                f["seeds"] = "auto"
                note = " (the SAM2 seeds were the other clip's: here it finds the objects by itself)"
        fx.append(f)
    if not fx:
        _status(f"{mem.label}: nothing to paste here")
        return
    _set_member(key, at, {**mem.spec, "fx": list(mem.spec.get("fx") or []) + fx})
    _status(f"{mem.label}: + {', '.join(arr.label_of(f) for f in fx)} from {board['from']}{note}")
    _draw_inspector()


def _mutes(spec: dict) -> list:
    """The bars a layer is hidden in (``mute:``), as sorted [first, last] spans."""
    got = spec.get("mute") or []
    got = [got] if got and not isinstance(got[0], (list, tuple)) else got
    return sorted([int(a), int(b)] for a, b in got)


def _spans(bars: set) -> list:
    out = []
    for b in sorted(bars):
        if out and b == out[-1][1] + 1:
            out[-1][1] = b
        else:
            out.append([b, b])
    return out


def mute_bars(key: str, at: int, first: int, last: int, hide: bool = True) -> None:
    """A member hidden in bars first..last (``hide`` False: shown there again). It goes on
    underneath: a character cut by its REAPER cues stays in sync with its sound."""
    mem = _member(key, at)
    if mem is None:
        return
    first, last = min(first, last), max(first, last)
    bars = {b for a, z in _mutes(mem.spec) for b in range(a, z + 1)}
    bars = bars | set(range(first, last + 1)) if hide else bars - set(range(first, last + 1))
    spec = {k: v for k, v in mem.spec.items() if k != "mute"}
    if bars:
        spec["mute"] = _spans(bars)
    _set_member(key, at, spec)
    what = f"bar {first}" if first == last else f"bars {first}-{last}"
    _status(f"{mem.label} hidden in {what} (its sound and sync go on)" if hide else
            f"{mem.label} shown again in {what}")


def _set_mutes(key: str, at: int, text: str) -> None:
    """The inspector's "hidden in bars" field: 9-12, 30-33 (empty: shown wherever it plays)."""
    mem = _member(key, at)
    try:
        bars = _parse_bars(text)
    except ValueError:
        _status(f"bars {text!r}: write them like 9-12, 30-33")
        return
    if mem is None:
        return
    spec = {k: v for k, v in mem.spec.items() if k != "mute"}
    if bars:
        spec["mute"] = _spans({b for a, z in bars for b in range(a, z + 1)})
    _set_member(key, at, spec)


def _mute_choices(mem, bar: int) -> list:
    """[(menu label, (first, last, show))]: what can be hidden (or shown again) of a member
    with no blocks at ``bar``: the cut there, the chosen bars, the bar."""
    out = []
    muted = next((m for m in _mutes(mem.spec) if m[0] <= bar <= m[1]), None)
    if muted is not None:
        a, b = muted
        out.append((f"Show again in bars {a}-{b}" if b > a else f"Show again in bar {a}", (a, b, True)))
        if b > a:
            out.append((f"Show again in bar {bar}", (bar, bar, True)))
        return out
    cut = next(((int(np.floor(a + 1e-6)), max(int(np.floor(a + 1e-6)), int(np.ceil(b - 1e-6)) - 1))
                for a, b in _cue_bars(mem) if a - 1e-6 <= bar + 0.999 and b + 1e-6 >= bar), None)
    if cut is not None and cut != (bar, bar):
        out.append((f"Hide this cut (bars {cut[0]}-{cut[1]})", (cut[0], cut[1], False)))
    if A.span is not None and A.span[0] <= bar <= A.span[1] and tuple(A.span) != (bar, bar):
        out.append((f"Hide in the chosen bars {A.span[0]}-{A.span[1]}", (A.span[0], A.span[1], False)))
    out.append((f"Hide in bar {bar}", (bar, bar, False)))
    return out


def _cue_bars(mem) -> list:
    """A cued member's cues as (first, last) bar positions (floats), once they are known."""
    return [(_time_to_bar(t0), _time_to_bar(t1)) for t0, t1 in A.cues.get(_cue_key(mem.spec), [])]


def _unblocked_at(r: int, x: float):
    """The member with no blocks drawn at ``x`` on row ``r``: a cued one (on one of its
    cues, or its name), or one that plays the whole song."""
    row = A.rows[r]
    for mem in A.members.get(row["lane"], []):
        if mem.row != row["sub"] or mem.blocks:
            continue
        if mem.always:
            return mem
        name_x = _name_x(mem)
        if name_x <= x <= name_x + 7 * (len(mem.label) + 16):
            return mem
        bar = (x - LABEL_W) / _bw() + A.view0
        if any(a - 0.25 <= bar <= b + 0.25 for a, b in _cue_bars(mem)):
            return mem
    return None


def _dragged(x: float):
    """(row, first bar, last bar) the drag would leave."""
    r, bar, mode, hit = A.drag
    now = _bar_of_x(x)
    if mode in ("new", "range", "span"):
        return r, min(bar, now), max(bar, now)
    b = _member(A.rows[r]["lane"], hit[0]).blocks[hit[1]]
    if mode == "move":
        d = max(1 - b.first, min(_bars() - b.last, now - bar))
        return r, b.first + d, b.last + d
    if mode == "first":
        return r, min(now, b.last), b.last
    return r, b.first, max(now, b.first)


def release(x: float, y: float = None) -> None:
    """The mouse comes up: a new member, a block moved or resized (moved onto another row:
    there now, ``_move_to_row``), or (no move) a block chosen; on the montage, the bars
    dragged over (a click: the clip there) chosen."""
    if A.drag is None:                                   # the press chose something: no drag
        return
    r, bar, mode, hit = A.drag
    got = _dragged(x)
    A.drag = None
    if r >= len(A.rows):
        return
    _, first, last = got
    if mode == "span":                                   # bars chosen on the ruler
        A.sel = None
        _set_span(first, last)
        return
    if mode == "range":
        path = ""
        if first == last:                                # a click: the clip playing there
            sel, path = _run_sel(first)
            first, last = sel[1], sel[2]
        A.sel = ("montage", first, last)
        if path and Path(path).is_file():
            _choose_clip(path, 0.0)                      # picked: to put elsewhere
        _draw()
        _draw_inspector()
        return
    if A.rows[r]["kind"] != "lane":
        return
    key = A.rows[r]["lane"]
    if mode == "new":
        if _can_put(key):
            _new_member(key, first, last, sub=A.rows[r].get("sub"))
        else:                                            # nothing picked to put: bars chosen
            A.sel = None
            _set_span(first, last)
            _status(f"bars {first}-{last} chosen: Preview (Space) plays them · to put something here, "
                    "pick it on the left first")
        return
    mem = _member(key, hit[0])
    b = mem.blocks[hit[1]]
    t = _target_row(key, y) if mode == "move" and y is not None else None
    if t is not None and (A.rows[t]["lane"], A.rows[t]["sub"]) != (key, A.rows[r]["sub"]):
        _move_to_row(key, mem, hit[1], first, last, t)
        _draw()
        _draw_inspector()
        return
    if (first, last) == (b.first, b.last):              # no move: a click picks the block
        A.sel = ("block", key, mem.at, hit[1])
        if mem.kind == arr.CLIP and (b.file or mem.spec.get("file")):
            _choose_clip(b.file or mem.spec.get("file"), b.start)
        _draw()
        _draw_inspector()
        return
    start = b.start
    if mode == "first":                                  # its start moves: the clip keeps its
        start = max(0.0, b.start + (first - b.first) * _bar_seconds() * b.rate)   # place in time
    moved = arr.Block(first, last, round(start, 3), b.rate, b.file)
    rest = [x for i, x in enumerate(mem.blocks) if i != hit[1]]
    blocks = arr.place(rest, moved, _bar_seconds())
    _set_blocks(key, mem.at, blocks)
    A.sel = ("block", key, mem.at, arr.block_at(blocks, first))
    _draw()
    _draw_inspector()


# ----------------------------------------------------------------------- drawing

LANE_COLOURS = {arr.FG: (150, 210, 240, 255), arr.BG: (230, 210, 140, 255), MASTER: (220, 160, 240, 255),
                FG_FX: (200, 160, 230, 255), BG_FX: (200, 160, 230, 255), "montage": (200, 200, 200, 255)}


def _row_chosen(row: dict) -> bool:
    """The row is lit: it holds what is chosen (a lane chosen: its FX track too)."""
    sel = A.sel or (None,)
    if row["kind"] == "montage":
        return sel[0] == "montage"
    key = row["lane"]
    if sel[0] == "master":
        return key == MASTER
    if sel[0] == "lane":
        return key != MASTER and _owner(key) == sel[1]
    return sel[0] == "block" and sel[1] == key


def _short(names: list, room: int) -> str:
    text = ", ".join(names)
    return text if len(text) <= room else text[:room - 3] + "..."


def _draw() -> None:
    if not dpg.does_item_exist("arr_canvas"):
        return
    dpg.delete_item("arr_canvas", children_only=True)
    A.runs = _montage_runs() if A.doc is not None else []
    n, bw = _bars(), _bw()
    h = _row_y(len(A.rows)) + 8
    dpg.configure_item("arr_canvas", height=max(120, int(h)))
    dpg.draw_rectangle([0, 0], [A.canvas_w, h], fill=(18, 20, 26, 255), color=(18, 20, 26, 255),
                       parent="arr_canvas")
    last = min(n, A.view0 + _shown() - 1)
    for b in range(A.view0, last + 1):                   # the bars' lines down the lanes
        x = _x_of(b)
        major = (b - 1) % 4 == 0
        dpg.draw_line([x, _row_y(0)], [x, h], parent="arr_canvas",
                      color=(70, 76, 95, 160) if major else (45, 50, 62, 120))
    for r, row in enumerate(A.rows):
        y = _row_y(r)
        dpg.draw_rectangle([0, y], [A.canvas_w, y + A.lane_h - 2],
                           fill=(34, 38, 50, 255) if _row_chosen(row) else (26, 29, 38, 255), color=(0, 0, 0, 0),
                           parent="arr_canvas")
        if row["kind"] == "montage":
            for w in arr.windows_of(_montage_item()[2]) or [arr.Block(1, n)]:
                dpg.draw_rectangle([_x_of(w.first), y + A.lane_h - 8], [_x_of(w.last + 1), y + A.lane_h - 5],
                                   fill=(120, 120, 130, 160), color=(0, 0, 0, 0), parent="arr_canvas")
            _draw_montage(y)
            continue
        key = row["lane"]
        if row["sub"] == 0:
            dpg.draw_line([0, y - 1], [A.canvas_w, y - 1], parent="arr_canvas",
                          color=(70, 74, 92, 255) if key in (FG_FX, BG_FX) else (90, 96, 120, 255))
        here = [m for m in A.members.get(key, []) if m.row == row["sub"]]
        if not here and _is_fx(key) and row["sub"] == 0 and _roomy():
            dpg.draw_text([LABEL_W + 6, y + 9], "right-click: an FX here · or pick one in FX and drag across",
                          size=11, color=(110, 110, 125, 255), parent="arr_canvas")
        starts = sorted([_name_x(m) for m in here] +       # where the names and the blocks
                        [_x_of(b.first) for m in here for b in m.blocks])   # start on the row
        for mem in here:
            x = _name_x(mem)
            _draw_member(key, mem, y, next((s for s in starts if s > x), A.canvas_w))
    if A.span is not None:                               # the chosen bars
        xa, xb = max(_x_of(A.span[0]), LABEL_W), min(_x_of(A.span[1] + 1), A.canvas_w)
        if xb > xa:
            dpg.draw_rectangle([xa, RULER_H + DENSITY_H], [xb, h], fill=(255, 255, 255, 14),
                               color=(255, 255, 255, 60), parent="arr_canvas")
    _draw_names()
    with dpg.draw_node(tag="arr_head", parent="arr_canvas"):
        _draw_ruler(n, bw, last)
    A.head_y = None
    _pin_ruler()
    if dpg.does_item_exist("arr_prev_btn"):
        dpg.configure_item("arr_prev_btn", label=f"Preview bars {A.span[0]}-{A.span[1]} (Space)"
                           if A.span else "Preview (choose bars)")
    dpg.draw_rectangle([0, 0], [0, 0], fill=(255, 255, 255, 60), color=(255, 255, 255, 200),
                       tag="arr_ghost", show=False, parent="arr_canvas")
    dpg.draw_line([0, 0], [0, 0], color=(255, 210, 90, 255), thickness=3, tag="arr_edge", show=False,
                  parent="arr_canvas")


def _draw_ruler(n: int, bw: float, last: int) -> None:
    """The bar numbers, how busy the MIDI is and the chosen bars, over the lanes that scroll
    under them (``_pin_ruler`` keeps them on top)."""
    head = _row_y(0)
    dpg.draw_rectangle([0, 0], [A.canvas_w, head], fill=(18, 20, 26, 255), color=(18, 20, 26, 255),
                       parent="arr_head")
    for b in range(A.view0, last + 1):
        x = _x_of(b)
        major = (b - 1) % 4 == 0
        dpg.draw_line([x, RULER_H - (10 if major else 4)], [x, head], parent="arr_head",
                      color=(70, 76, 95, 160) if major else (45, 50, 62, 120))
        if major or bw > 22:
            dpg.draw_text([x + 2, 4], str(b), size=12, color=(170, 175, 190, 255), parent="arr_head")
    for b, v in enumerate(_density(n), 1):               # how busy the MIDI is, bar by bar
        if not A.view0 <= b <= last:
            continue
        c = int(40 + 180 * v)
        dpg.draw_rectangle([_x_of(b), RULER_H], [_x_of(b + 1), RULER_H + DENSITY_H],
                           fill=(c, int(c * 0.8), 60, 255), color=(0, 0, 0, 0), parent="arr_head")
    if A.span is not None:                               # the chosen bars
        xa, xb = max(_x_of(A.span[0]), LABEL_W), min(_x_of(A.span[1] + 1), A.canvas_w)
        if xb > xa:
            dpg.draw_rectangle([xa, 0], [xb, RULER_H + DENSITY_H], fill=(255, 255, 255, 70),
                               color=(255, 255, 255, 220), parent="arr_head")
    dpg.draw_rectangle([0, 0], [LABEL_W - 1, head], fill=(18, 20, 26, 255), color=(18, 20, 26, 255),
                       parent="arr_head")
    dpg.draw_line([0, head - 1], [A.canvas_w, head - 1], color=(60, 64, 80, 255), parent="arr_head")


def _pin_ruler() -> None:
    """The ruler moved down as far as the lanes are scrolled: it stays at the top of the view."""
    if not dpg.does_item_exist("arr_head"):
        return
    y = _scroll()
    if y != getattr(A, "head_y", None):
        A.head_y = y
        dpg.apply_transform("arr_head", dpg.create_translation_matrix([0, y]))


def _roomy() -> bool:
    """A row is tall enough for a second line (a lane's subtitle, a cued member's name)."""
    return A.lane_h >= LANE_H


def _draw_names() -> None:
    """The names' column, over whatever a zoomed view runs under it (its top: the ruler's)."""
    grey = (150, 150, 160, 255)
    for r, row in enumerate(A.rows):
        y = _row_y(r)
        dpg.draw_rectangle([0, y], [LABEL_W - 1, y + A.lane_h - 2], fill=(34, 38, 50, 255) if _row_chosen(row) else
                           (26, 29, 38, 255), color=(0, 0, 0, 0), parent="arr_canvas")
        if row["kind"] == "montage":
            dpg.draw_text([6, y + 4], "CLIPS", size=13, color=LANE_COLOURS["montage"], parent="arr_canvas")
            if _roomy():
                dpg.draw_text([6, y + 18], "the montage, under the BG", size=10, color=grey, parent="arr_canvas")
            continue
        key = row["lane"]
        colour = LANE_COLOURS[key]
        if row["sub"] == 0:
            dpg.draw_line([0, y - 1], [LABEL_W, y - 1], color=(90, 96, 120, 255), parent="arr_canvas")
            if key in (FG_FX, BG_FX):
                dpg.draw_text([16, y + 4], "FX", size=13, color=colour, parent="arr_canvas")
                if _roomy():
                    dpg.draw_text([16, y + 18], f"on the whole {_owner(key)}", size=10, color=grey,
                                  parent="arr_canvas")
            elif key == MASTER:
                dpg.draw_text([6, y + 4], "MASTER", size=13, color=colour, parent="arr_canvas")
                if _roomy():
                    dpg.draw_text([6, y + 18], "FX over everything", size=10, color=grey, parent="arr_canvas")
            else:
                dpg.draw_text([6, y + 4], key, size=15, color=colour, parent="arr_canvas")
                dpg.draw_text([30, y + 6], "foreground" if key == arr.FG else "background", size=10,
                              color=grey, parent="arr_canvas")
        if not _is_fx(key) and row["sub"] == max([m.row for m in A.members.get(key, [])], default=-1) + 1 \
                and (_roomy() or row["sub"] > 0):
            dpg.draw_text([6, y + (18 if row["sub"] == 0 else 4)], "drag here: new", size=10,
                          color=(110, 110, 125, 255), parent="arr_canvas")


def _montage_runs() -> list:
    """The montage as the composer plays it with the pins as they are now (saved or not),
    a run per clip (its reverses inside one run): [(first, last) bar positions (floats),
    clip name ("none": pinned to nothing), clip path]."""
    from core.scene_doc import plain
    it = _montage_item()
    if it is None or not hasattr(A.host, "montage"):
        return []
    try:
        segs = A.host.montage([plain(it[2])])
    except TypeError:                                    # a host that only knows the saved montage
        segs = A.host.montage()
    out = []
    for t0, t1, name, path in segs:
        a, b = _time_to_bar(t0), _time_to_bar(t1)
        if out and out[-1][2] == name and abs(out[-1][1] - a) < 1e-6:
            out[-1] = (out[-1][0], b, name, path)
        else:
            out.append((a, b, name, path))
    return out


def _draw_montage(y: float) -> None:
    """The montage's row: which clip plays where, each in its colour, named where it fits;
    a pinned bar has a white mark, a bar pinned to nothing is dark."""
    runs = A.runs
    if not runs:
        if _roomy():
            dpg.draw_text([LABEL_W + 6, y + 6], "load the project (with its clips folder) to see which clip "
                          "plays in each bar", size=11, color=(130, 130, 145, 255), parent="arr_canvas")
        return
    for a, b, name, path in runs:
        xa, xb = max(_x_of(a), LABEL_W), min(_x_of(b), A.canvas_w)
        if xb <= xa:
            continue
        fill = (40, 40, 48, 255) if name == "none" else (*arr.colour(path), 150)
        dpg.draw_rectangle([xa, y + 2], [max(xa + 1, xb - 1), y + A.lane_h - 11], fill=fill,
                           color=(0, 0, 0, 0), parent="arr_canvas")
        room = int((xb - xa - 6) / 6.5)
        if room >= 3 and _roomy():
            dpg.draw_text([xa + 3, y + 6], name if len(name) <= room else name[:room - 1] + "~", size=11,
                          color=(150, 150, 160, 255) if name == "none" else (10, 10, 14, 255), parent="arr_canvas")
    for b in _pins():                                    # pinned bars: a white corner
        x = _x_of(b)
        if LABEL_W <= x < A.canvas_w:
            dpg.draw_triangle([x + 1, y + 2], [x + 8, y + 2], [x + 1, y + 9], fill=(255, 255, 255, 255),
                              color=(255, 255, 255, 255), parent="arr_canvas")
    sel = A.sel or (None,)
    if sel[0] == "montage" and len(sel) > 2:
        dpg.draw_rectangle([max(_x_of(sel[1]), LABEL_W), y + 1], [_x_of(sel[2] + 1), y + A.lane_h - 10],
                           color=(255, 255, 255, 230), thickness=2, parent="arr_canvas")


def _draw_montage_sel(p, first: int, last: int) -> None:
    """The montage's chosen bars: their clips, and what can be done to them."""
    bars = f"bars {first}-{last}" if last > first else f"bar {first}"
    dpg.add_text(f"CLIPS · {bars}", parent=p, color=LANE_COLOURS["montage"][:3])
    pins = _pins()
    runs = _runs_in(first, last)
    if not runs:
        dpg.add_text("the montage is not known yet (load the project with its clips folder)", parent=p,
                     color=(150, 150, 160))
    for a, b, name, path in runs[:12]:
        pinned = any(pins.get(k) for k in range(int(a), int(np.ceil(b))))
        dpg.add_text(f"{name}  (bars {a:.2f} to {b:.2f})" + ("  - pinned" if pinned else ""), parent=p,
                     color=(150, 150, 160) if name == "none" else arr.colour(path))
        if path:
            dpg.add_text(path, parent=p, color=(130, 130, 145), wrap=700)
    if len(runs) > 12:
        dpg.add_text(f"... and {len(runs) - 12} more", parent=p, color=(130, 130, 145))
    with dpg.group(horizontal=True, parent=p):
        for _, label, fn in _montage_choices(first, last):
            dpg.add_button(label=label, callback=lambda s, a, *, fn=fn: fn())
    dpg.add_text("The montage draws a clip for each bar (clip_order, the seed, the triggers' cuts and "
                 "reverses). A pin keeps a clip there (a clip picked on the left: Put); the bars around "
                 "stay as they were drawn. A clip from another folder, or from another point, goes in the "
                 "BG over these bars.", parent=p, color=(150, 150, 160), wrap=700)


def _pins() -> dict:
    """The montage's pins: bar number -> a clip's name, or "none"."""
    it = _montage_item()
    return {int(k): v for k, v in ((it[2].get("pins") or {}) if it else {}).items()}


def _set_pins(first: int, last: int, value) -> None:
    """Bars first..last pinned to ``value`` (a clip's name, "none": nothing), or (None)
    given back to the montage's choice."""
    it = _montage_item()
    if it is None:
        return

    def put(spec):
        pins = {int(k): v for k, v in (spec.get("pins") or {}).items()}
        for b in range(first, last + 1):
            if value is None:
                pins.pop(b, None)
            else:
                pins[b] = value
        out = {k: v for k, v in spec.items() if k != "pins"}
        if pins:
            out["pins"] = dict(sorted(pins.items()))
        return out
    _edit(it[3], (), put)
    A.sel = ("montage", first, last)
    _draw()
    _draw_inspector()
    bars = f"bars {first}-{last}" if last > first else f"bar {first}"
    _status(f"{bars}: " + ("back to the montage's choice" if value is None else
                           "nothing (the BG, or black)" if value == "none" else f"{value}, pinned"))


def _montage_clips() -> list:
    return [str(Path(x)) for x in (A.host.montage_clips() if hasattr(A.host, "montage_clips") else [])]


def _put_in_montage(first: int, last: int) -> None:
    """The clip picked on the left in these bars: pinned, when the montage has it (its
    folder); else a BG block over them."""
    if A.content is None or A.content[0] != "clip":
        _status("pick a clip on the left (Clips) first")
        return
    path, start = A.content[1], A.content[2]
    if str(Path(path)) in _montage_clips() and not start:
        _set_pins(first, last, Path(path).stem)
    else:
        _to_bg(path, start, first, last)


def _to_bg(path: str, start: float, first: int, last: int) -> None:
    """A clip as a BG block over these bars (moved, resized and given FX like any)."""
    item = _lane_item(arr.BG)
    if item is None:
        return
    spec = arr.with_blocks({"source": "footage", "role": "background", "fit": "cover", "loop": True,
                            "file": path}, [arr.Block(first, last, start)])
    _edit(item[3], ("layers",), lambda ls: list(ls or []) + [spec])
    A.sel = ("block", arr.BG, len(item[2].get("layers") or []) - 1, 0)
    _draw()
    _draw_inspector()
    _status(f"{Path(path).stem} in the BG, bars {first}-{last}: over the montage there")


def _runs_in(first: int, last: int) -> list:
    return [u for u in A.runs if u[0] < last + 1 - 1e-6 and u[1] > first + 1e-6]


def _run_sel(bar: int):
    """(the selection, the clip's path) of the montage's clip playing at ``bar``'s middle:
    the bars it plays."""
    run = next((u for u in A.runs if u[0] <= bar + 0.5 < u[1]), None)
    if run is None:
        return ("montage", bar, bar), ""
    first = max(1, int(np.floor(run[0] + 0.05)))
    return ("montage", first, max(first, int(np.ceil(run[1] - 0.05)) - 1)), run[3]


def _montage_actions(first: int, last: int) -> list:
    """[(label, fn)]: what can be done to the montage's bars first..last."""
    return [(label, fn) for label, _, fn in _montage_choices(first, last)]


def _montage_choices(first: int, last: int) -> list:
    """[(menu label, button label, fn)]: what can be done to the montage's bars first..last."""
    out = []
    bars = f"bars {first}-{last}" if last > first else f"bar {first}"
    if A.content is not None and A.content[0] == "clip":
        stem = Path(A.content[1]).stem
        out.append((f"Put {stem} here ({bars})", f"Put {stem[:18]}", lambda: _put_in_montage(first, last)))
    runs = [u for u in _runs_in(first, last) if u[2] != "none"]
    if runs:
        out.append((f"Nothing here ({bars}: the BG, or black) - Delete", "Nothing here",
                    lambda: _set_pins(first, last, "none")))
    if len({u[3] for u in runs}) == 1 and runs[0][3]:
        path = runs[0][3]
        out.append((f"Make {runs[0][2]} a BG block (move it, resize it, FX)", "Make it a BG block",
                    lambda: _to_bg(path, 0.0, first, last)))
    if any(first <= b <= last for b in _pins()):
        out.append(("Back to the montage's choice", "Back to the montage", lambda: _set_pins(first, last, None)))
    out.append((f"Preview {bars}", "Preview", lambda: preview_bars(first, last)))
    return out


def _draw_mutes(mem, y: float) -> set:
    """The bars a member is hidden in, drawn on its row (dark, outlined in red, under its
    name and cuts); -> those bars."""
    hidden = set()
    for a, b in _mutes(mem.spec):
        hidden.update(range(a, b + 1))
        xa, xb = max(_x_of(a), LABEL_W), _x_of(b + 1)
        if xb <= LABEL_W:
            continue
        dpg.draw_rectangle([xa, y + 2], [xb, y + A.lane_h - 3], fill=(40, 16, 20, 255),
                           color=(200, 90, 90, 220), parent="arr_canvas")
        name_at = _name_x(mem)
        if xb - xa > 60 and _roomy() and not xa - 1 <= name_at < xb:   # (never over its name)
            dpg.draw_text([xb - 50, y + 2], "hidden", size=11, color=(230, 140, 140, 255), parent="arr_canvas")
    return hidden


def _name_x(mem) -> float:
    """Where a member's name is written: at its first cue or block (members that never
    meet share a row, each named where it plays), else where the row starts."""
    cues = A.cues.get(_cue_key(mem.spec)) if mem.cued else None
    first = _time_to_bar(cues[0][0]) if cues else mem.blocks[0].first if mem.blocks else 1
    return max(_x_of(first), LABEL_W)


def _draw_member(key: str, mem, y: float, until: float = None) -> None:
    """A member on its row; ``until``: where the next one's name starts (its own stops before)."""
    until = A.canvas_w if until is None else until
    n = _bars()
    fx = [f for f in mem.spec.get("fx") or [] if isinstance(f, dict)]
    tag = ("GEN " if mem.kind == arr.GENERATOR and not _is_fx(key) else "") + mem.label
    chosen = A.sel == ("block", key, mem.at, None)
    if mem.always:                                       # no blocks: the whole song
        r, g, bb = arr.colour(mem.label)
        dpg.draw_rectangle([max(_x_of(1), LABEL_W), y + 3], [_x_of(n + 1), y + A.lane_h - 4],
                           fill=(r, g, bb, 70 if mem.enabled else 25),
                           color=(255, 255, 255, 255) if chosen else (r, g, bb, 160),
                           thickness=2 if chosen else 1, parent="arr_canvas")
        dpg.draw_text([max(_x_of(1), LABEL_W) + 4, y + min(8, (A.lane_h - 14) / 2)], f"{tag} (the whole song)"[:60], size=11,
                      color=(220, 220, 230, 255) if mem.enabled else (130, 130, 140, 255), parent="arr_canvas")
    hidden = _draw_mutes(mem, y) if not mem.blocks else set()
    if mem.cued:
        x = _name_x(mem)
        room = int((min(until, A.canvas_w) - x - 8) / 7)
        name = f"{tag} - cues (REAPER)"
        if room >= 3 and _roomy():                       # low rows: the cues alone
            dpg.draw_text([x + 2, y + 2], name if len(name) <= room else name[:room - 1] + "~", size=12,
                          color=(215, 215, 225, 255), parent="arr_canvas")
        for t0, t1 in A.cues.get(_cue_key(mem.spec), []):  # read-only cues, in grey (white: chosen; dim: hidden)
            xa, xb = _x_of(_time_to_bar(t0)), _x_of(_time_to_bar(t1))
            dim = int(_time_to_bar((t0 + t1) / 2)) in hidden
            dpg.draw_rectangle([xa, y + A.lane_h - 14], [max(xa + 1, xb), y + A.lane_h - 5],
                               fill=(70, 60, 64, 255) if dim else (235, 235, 245, 255) if chosen else (140, 140, 150, 200),
                               color=(0, 0, 0, 0), parent="arr_canvas")
    for i, blk in enumerate(mem.blocks):
        xa, xb = _x_of(blk.first), _x_of(blk.last + 1)
        r, g, bb = arr.colour(blk.file or mem.spec.get("file") or mem.label)
        if not mem.enabled:
            r, g, bb = (r + 240) // 3, (g + 240) // 3, (bb + 240) // 3
        sel = A.sel == ("block", key, mem.at, i)
        dpg.draw_rectangle([xa + 1, y + 3], [xb - 1, y + A.lane_h - 4], fill=(r, g, bb, 230),
                           color=(255, 255, 255, 255) if sel else (10, 10, 14, 255),
                           thickness=2 if sel else 1, parent="arr_canvas")
        stem = Path(blk.file).stem if blk.file else mem.label
        folder = Path(blk.file).parent.name if blk.file else ""
        label = (mem.label if mem.spec.get("name") else
                 folder if folder and stem in ("final", "clip", "video") else stem[-14:])
        if blk.start:
            label += f" @{blk.start:g}s"
        if fx:
            label += f" +{len(fx)}fx"
        xt = max(xa, LABEL_W)                          # its name where it shows
        room = int((min(xb, A.canvas_w) - xt - 6) / 6.5)
        if room >= 3:
            dpg.draw_text([xt + 4, y + min(6, (A.lane_h - 14) / 2)], label[:room], size=11, color=(10, 10, 14, 255), parent="arr_canvas")
        if fx:
            dpg.draw_rectangle([xa + 2, y + A.lane_h - 8], [xb - 2, y + A.lane_h - 6], fill=(200, 120, 230, 255),
                               color=(0, 0, 0, 0), parent="arr_canvas")


def _density(n: int) -> list:
    notes = A.host.notes() or []
    g = A.host.grid()
    db = getattr(g, "downbeats", None) if g is not None else None
    if not notes or db is None or not len(db):
        return [0.0] * n
    counts = np.bincount(np.clip(np.searchsorted(db, [x.time for x in notes], side="right") - 1, 0, n - 1),
                         minlength=n)[:n].astype(float)
    return list(counts / max(1.0, counts.max()))


# ----------------------------------------------------------------------- the preview

def _can_put(key: str) -> bool:
    """Something picked on the left goes on this track: a clip or a Generator on a lane, an
    FX on an FX track."""
    if _is_fx(key):
        from core import catalog
        e = catalog.by_key(A.entry) if A.entry else None
        return e is not None and e.kind == catalog.FX
    return A.content is not None


def _set_span(first: int, last: int) -> None:
    """The bars chosen: drawn over the tracks; Preview plays them."""
    A.span = (min(first, last), max(first, last))
    _draw()
    _draw_inspector()


def _sel_span():
    """The bars of what is chosen: a block's, a cued member's cuts, the montage's bars."""
    sel = A.sel or (None,)
    if sel[0] == "montage" and len(sel) > 2:
        return sel[1], sel[2]
    if sel[0] == "block":
        mem = _member(sel[1], sel[2])
        if mem is not None and sel[3] is not None and sel[3] < len(mem.blocks):
            b = mem.blocks[sel[3]]
            return b.first, b.last
        if mem is not None and mem.cued and _cue_bars(mem):
            spans = _cue_bars(mem)
            return int(spans[0][0]), int(spans[-1][1])
    return None


def preview_span() -> None:
    """Preview: the chosen bars; none chosen, the first bars in view."""
    if A.span is None:
        first = max(1, int(A.view0))
        preview_bars(first, min(_bars(), first + PREVIEW_BARS - 1))
        return
    preview_bars(*A.span)


def on_key_space(sender, app_data) -> None:
    if dpg.does_item_exist("arr_win") and dpg.is_item_shown("arr_win") and not _typing() and _in_window():
        preview_span()



PREVIEW_BARS = 8       # the most a block's preview renders (it is held in memory)


# ----------------------------------------------------------------------- the final video

def open_render() -> None:
    """The final video's window: the whole song or some bars, where it goes, Render / Stop,
    how far it is, and the video opened when it is done."""
    if not dpg.does_item_exist("arr_render"):
        with dpg.window(tag="arr_render", label="Render the video", width=520, height=300, pos=(300, 120)):
            dpg.add_radio_button(["the whole song", "these bars"], tag="arr_render_what", horizontal=True,
                                 default_value="the whole song", callback=lambda: _render_where())
            with dpg.group(horizontal=True):
                dpg.add_input_int(label="to", tag="arr_render_first", width=110, min_value=1, min_clamped=True,
                                  default_value=1, callback=lambda: _render_where(bars=True))
                dpg.add_input_int(tag="arr_render_last", width=110, min_value=1, min_clamped=True,
                                  default_value=1, callback=lambda: _render_where(bars=True))
            dpg.add_text("", tag="arr_render_size", color=(150, 150, 165))
            dpg.add_text("", tag="arr_render_out", color=(170, 210, 170), wrap=500)
            with dpg.group(horizontal=True):
                dpg.add_button(label="Render", tag="arr_render_go", width=110, callback=lambda: btn_render())
                dpg.add_button(label="Stop", tag="arr_render_stop", width=70,
                               callback=lambda: A.host.stop_render() if hasattr(A.host, "stop_render") else None)
            dpg.add_progress_bar(tag="arr_render_bar", width=-1, default_value=0.0, overlay="")
            dpg.add_text("", tag="arr_render_state", wrap=500)
            with dpg.group(horizontal=True, tag="arr_render_done", show=False):
                dpg.add_button(label="Open the video", callback=lambda: _open_render(False))
                dpg.add_button(label="Show it in its folder", callback=lambda: _open_render(True))
    first, last = getattr(A, "span", None) or (1, _bars())
    dpg.set_value("arr_render_first", first)
    dpg.set_value("arr_render_last", last)
    dpg.set_value("arr_render_size", f"{A.host.resolution() if hasattr(A.host, 'resolution') else ''}"
                  "  (the size above: final video)")
    _render_where()
    dpg.show_item("arr_render")
    dpg.focus_item("arr_render")


def _render_bars():
    """(first, last) when some bars are asked for; None: the whole song."""
    if dpg.get_value("arr_render_what") != "these bars":
        return None
    a, b = dpg.get_value("arr_render_first"), dpg.get_value("arr_render_last")
    return max(1, min(a, b)), min(_bars(), max(a, b))


def _render_where(bars: bool = False) -> None:
    if bars:
        dpg.set_value("arr_render_what", "these bars")
    span = _render_bars()
    out = A.host.render_output(*span) if hasattr(A.host, "render_output") and span else \
        A.host.render_output() if hasattr(A.host, "render_output") else ""
    dpg.set_value("arr_render_out", f"-> {out}" if out else "")


def btn_render() -> None:
    """Render: saved first (the render reads the scene file), then the studio's renderer."""
    if not hasattr(A.host, "render"):
        _status("the final video needs the studio (arc_studio.py)")
        return
    if A.dirty:
        btn_save()
        if A.dirty:
            dpg.set_value("arr_render_state", "not rendered: the scene could not be saved")
            return
    span = _render_bars()
    why = A.host.render(*span) if span else A.host.render()
    dpg.set_value("arr_render_state", why or "starting...")
    dpg.configure_item("arr_render_done", show=False)
    if not why:
        A.render_seen = True


def _open_render(folder: bool) -> None:
    import os
    import subprocess
    state = A.host.render_state() if hasattr(A.host, "render_state") else None
    out = Path(state["out"]).resolve() if state and state.get("out") else None
    if out is None or not out.exists():
        _status("the video is not there")
        return
    if folder:
        subprocess.Popen(["explorer", "/select,", str(out)])
    else:
        os.startfile(str(out))                           # noqa: S606 - the user's own video


def _render_tick() -> None:
    """How the final video goes (frame n of N, time left), in its window and the status."""
    if not dpg.does_item_exist("arr_render") or not hasattr(A.host, "render_state"):
        return
    st = A.host.render_state()
    if st is None:
        return
    running = st.get("running")
    dpg.configure_item("arr_render_go", enabled=not running)
    dpg.configure_item("arr_render_stop", enabled=bool(running))
    done, total = st.get("done", 0), st.get("total", 0)
    frac = done / total if total else 0.0
    if running:
        spent = time.time() - st.get("started", time.time())
        left = spent / frac * (1 - frac) if frac > 0.01 else None
        text = (f"rendering: frame {done}/{total}" + (f", about {_mmss(left)} left" if left is not None else "")
                if total else "starting: reading the song and the clips...")
    else:
        result = st.get("result") or ""
        text = (f"done: {st.get('out')}" if result == "done" else
                f"stopped at frame {done}/{total}" if result == "stopped" else result)
        if result == "done":
            frac = 1.0
    dpg.set_value("arr_render_bar", frac)
    dpg.configure_item("arr_render_bar", overlay=f"{100 * frac:.0f}%" if total else "")
    if dpg.get_value("arr_render_state") != text:
        dpg.set_value("arr_render_state", text)
    dpg.configure_item("arr_render_done", show=not running and st.get("result") == "done")
    if dpg.does_item_exist("arr_render_btn"):
        dpg.configure_item("arr_render_btn", label=f"Rendering {100 * frac:.0f}%..." if running else "Render video...")


def _mmss(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds >= 60 else f"{seconds} s"


def preview_bars(first: int, last: int) -> None:
    """The studio's preview of these bars (the first PREVIEW_BARS of them), with the
    arrangement as it is now (saved or not), in a window of its own that says how the
    rendering goes, and why it failed."""
    from core.scene_doc import plain
    if A.clip_sound:
        set_clip_sound(False)                            # the song plays: the clip's sound stops
    first, last = min(first, last), max(first, last)
    A.span = (first, last)
    bars = max(1, min(last - first + 1, PREVIEW_BARS))
    more = f" (the first {PREVIEW_BARS} of {first}-{last})" if last - first + 1 > PREVIEW_BARS else ""
    label = f"bars {first}-{first + bars - 1}{more}"
    if not hasattr(A.host, "preview"):
        _status("the preview needs the studio (arc_studio.py)")
        return
    state = A.host.preview_state() if hasattr(A.host, "preview_state") else "ready"
    if state != "ready":                                 # it plays as soon as the studio can
        why = ""
        if state == "unloaded":
            why = A.host.load() if hasattr(A.host, "load") else "load the project in the studio"
        A.queued = None if why else (first, last, time.time())
        A.previewing = None
        # the preview before is let go (the last one asked for is the one wanted); the video's
        # render is not
        stopping = state == "busy" and hasattr(A.host, "cancel_preview") and A.host.cancel_preview()
        text = (f"the preview can't start: {why}" if why else
                f"loading the project (the song, its MIDI) first: {label} plays when it is loaded..."
                if state in ("unloaded", "loading") else
                f"stopping the preview before: {label} starts right after..." if stopping else
                f"the studio is rendering the video: {label} comes when it is done...")
        _preview_window(label, text, error=bool(why))
        _draw()
        return
    g = A.host.grid()
    db = getattr(g, "downbeats", None) if g is not None else None
    if db is None or not len(db):
        _preview_window(label, "the preview can't start: the song has no bars (load the project)", error=True)
        return
    at = float(db[min(first, len(db)) - 1])
    if not A.host.preview(at, bars, [plain(s) for s in _specs()]):
        A.queued = (first, last, time.time())
        _preview_window(label, f"the studio is busy: {label} comes next...")
        return
    A.queued = None
    A.prev_failed = False
    _preview_window(label, f"rendering {label}...")
    A.previewing = (time.time(), False)
    _status(f"previewing {label}: rendering...")
    _draw()


def _preview_window(label: str, text: str, error: bool = False) -> None:
    """The preview's window (made again each time): what goes on, the frames, Stop."""
    got = A.host.preview_texture() if hasattr(A.host, "preview_texture") else None
    if got is None:
        _status(text)
        return
    tex, w, h = got[:3]
    fw, fh = got[3] if len(got) > 3 else (w, h)          # the part of it the frame fills
    k = 480.0 / max(1, fh) if fh > fw else 1.0           # a portrait frame shown 480 high
    show_w, show_h = int(fw * k), int(fh * k)
    uv0 = [(w - fw) / 2 / w, (h - fh) / 2 / h]
    uv1 = [1 - uv0[0], 1 - uv0[1]]
    if dpg.does_item_exist("arr_prev"):
        dpg.delete_item("arr_prev")
    with dpg.window(tag="arr_prev", label="Arrange preview", width=max(300, show_w + 20), height=show_h + 110,
                    pos=(12, 250), on_close=lambda: A.host.stop_preview() if hasattr(A.host, "stop_preview") else None):
        dpg.add_text("", tag="arr_prev_text", wrap=max(280, show_w))
        dpg.add_image(tex, width=show_w, height=show_h, uv_min=uv0, uv_max=uv1)
        if hasattr(A.host, "stop_preview"):
            dpg.add_button(label="Stop", callback=lambda: A.host.stop_preview())
    dpg.configure_item("arr_prev", label=f"Arrange preview - {label}")
    dpg.set_value("arr_prev_text", text)
    dpg.configure_item("arr_prev_text", color=(240, 120, 110) if error else (200, 200, 210))
    dpg.focus_item("arr_prev")
    _status(text)


def _watch_preview() -> None:
    """Once a frame: the preview window says what the studio says now (never a line from
    before); a preview's progress, then that it plays or why it failed; bars waiting for the
    studio (loading, or making another preview) start when it can."""
    _follow_preview()
    if A.previewing is not None or not dpg.does_item_exist("arr_prev_text") or \
            not hasattr(A.host, "preview_progress") or getattr(A, "prev_failed", False):
        return
    busy, status, _ = A.host.preview_progress()
    status = status.replace("—", "-").replace("…", "...")
    if A.queued is not None:                             # waiting: what holds it up, as it goes
        head = str(dpg.get_value("arr_prev_text")).split("\n")[0]
        status = f"{head}\nnow: {status}" if busy and status else head
    if status and dpg.get_value("arr_prev_text") != status:
        dpg.set_value("arr_prev_text", status)


def _follow_preview() -> None:
    if A.queued is not None and hasattr(A.host, "preview_state"):
        first, last, asked = A.queued
        state = A.host.preview_state()
        if state == "ready":
            preview_bars(first, last)
        elif state == "unloaded" and time.time() - asked > 3.0:   # the load gave up
            A.queued = None
            status = A.host.preview_progress()[1] if hasattr(A.host, "preview_progress") else ""
            text = "the project could not be loaded: " + (status or "see the studio's log")
            text = text.replace("—", "-").replace("…", "...")
            if dpg.does_item_exist("arr_prev_text"):
                dpg.set_value("arr_prev_text", text)
                dpg.configure_item("arr_prev_text", color=(240, 120, 110))
            _status(text)
        return
    if A.previewing is None or not hasattr(A.host, "preview_progress"):
        return
    asked, seen = A.previewing
    busy, status, error = A.host.preview_progress()
    status = status.replace("—", "-").replace("…", "...")   # glyphs the window's font lacks
    text = status
    if busy:
        A.previewing = (asked, True)
    elif seen or time.time() - asked > 3.0:              # it started (a thread) and is over
        A.previewing = None
        text = f"the preview failed: {error}" if error else status
        A.prev_failed = bool(error)                      # (then it stays said)
        _status(text)
    if dpg.does_item_exist("arr_prev_text"):
        dpg.set_value("arr_prev_text", text)
        dpg.configure_item("arr_prev_text", color=(240, 120, 110) if error and A.previewing is None else
                           (200, 200, 210))


# ----------------------------------------------------------------------- the inspector

def _num(spec, key, default):
    v = spec.get(key, default)
    return float(v) if isinstance(v, (int, float)) else None


def _parse_bars(text: str):
    """"9-12, 30-33" -> [[9, 12], [30, 33]]; "" -> None (the whole time its owner plays)."""
    out = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        a, b = int(a), int(b or a)
        out.append([min(a, b), max(a, b)])
    return out or None


def _bars_text(spec: dict) -> str:
    return ", ".join(f"{w.first}-{w.last}" if w.last != w.first else str(w.first) for w in arr.windows_of(spec))


def _draw_fx_list(p, uid, path: tuple, title: str) -> None:
    """An FX list (a block's, a lane's): each with on/off, its bars, its plugin, its place."""
    from core.scene_doc import get_path
    item = _item(uid)
    fx = get_path(item[2], path) if item else None
    fx = fx if isinstance(fx, list) else []
    with dpg.group(horizontal=True, parent=p):
        dpg.add_text(title, color=(220, 160, 240))
        dpg.add_button(label="Add FX...", callback=lambda: _menu_at(lambda: _catalog_menu(
            "Add FX", "fx", lambda key: add_fx(uid, path, key))))
    if not fx:
        dpg.add_text("  none", parent=p, color=(120, 120, 130))
    for j, f in enumerate(fx):
        if not isinstance(f, dict):
            continue
        here = path + (j,)
        with dpg.group(horizontal=True, parent=p):
            dpg.add_checkbox(default_value=f.get("enabled", True) is not False,
                             callback=lambda s, v, *, here=here: _toggle(uid, here))
            dpg.add_text(f"{j + 1}. {arr.label_of(f)}", color=(230, 230, 240))
            dpg.add_text("bars")
            dpg.add_input_text(default_value=_bars_text(f), width=110, hint="all", on_enter=True,
                               callback=lambda s, v, *, here=here: _set_fx_bars(uid, here, v))
            if f.get("source") == SAM:
                dpg.add_button(label="seeds", callback=lambda s, a, *, here=here: open_sam(uid, here))
            else:
                dpg.add_button(label="plugin", callback=lambda s, a, *, here=here: open_plugin(uid, here))
            dpg.add_button(label="^", callback=lambda s, a, *, j=j: _swap(uid, path, j, j - 1))
            dpg.add_button(label="v", callback=lambda s, a, *, j=j: _swap(uid, path, j, j + 1))
            dpg.add_button(label="x", callback=lambda s, a, *, j=j: _edit(
                uid, path, lambda ls, j=j: [x for i, x in enumerate(ls) if i != j]))


def _set_fx_bars(uid, path: tuple, text: str) -> None:
    try:
        bars = _parse_bars(text)
    except ValueError:
        _status(f"bars {text!r}: write them like 9-12, 30-33")
        return

    def put(f):
        f = {k: v for k, v in f.items() if k != "bars"}
        if bars:
            f["bars"] = bars
        return f
    _edit(uid, path, put)


def _swap(uid, path: tuple, i: int, j: int) -> None:
    def swap(ls):
        ls = list(ls)
        if 0 <= i < len(ls) and 0 <= j < len(ls):
            ls[i], ls[j] = ls[j], ls[i]
        return ls
    _edit(uid, path, swap)


def _draw_inspector() -> None:
    if not dpg.does_item_exist("arr_inspector"):
        return
    dpg.delete_item("arr_inspector", children_only=True)
    p = "arr_inspector"
    sel = A.sel
    got = _sel_span()
    if got is not None and got != A.span:                # choosing a thing chooses its bars
        A.span = got
        if dpg.does_item_exist("arr_prev_btn"):
            dpg.configure_item("arr_prev_btn", label=f"Preview bars {got[0]}-{got[1]} (Space)")
    if sel is None and A.span is not None and A.doc is not None:
        first, last = A.span
        dpg.add_text(f"bars {first}-{last} chosen", parent=p, color=(235, 235, 245))
        with dpg.group(horizontal=True, parent=p):
            dpg.add_button(label=f"Preview bars {first}-{min(last, first + PREVIEW_BARS - 1)} (Space)",
                           callback=preview_span)
            dpg.add_button(label="Choose none", callback=lambda: (setattr(A, "span", None), _draw(),
                                                                  _draw_inspector()))
        dpg.add_text("Pick a clip, a Generator or an FX on the left and drag across a track to put it there; "
                     "click a block to choose it.", parent=p, color=(130, 130, 150), wrap=700)
        return
    if sel is None or A.doc is None:
        dpg.add_text("pick a clip (Clips) or a Generator (Generators), then drag across the BG or the FG to "
                     "put it there; an FX (FX), across an FX track · click anything to choose it, drag it to "
                     "move it, Shift+drag its ends to resize it, Delete deletes it · right-click: its menu · double "
                     "click: its plugin · Ctrl+Z undo",
                     parent=p, color=(130, 130, 150), wrap=700)
        return
    if sel[0] == "master":
        _draw_master(p)
    elif sel[0] == "montage" and len(sel) > 2:
        _draw_montage_sel(p, sel[1], sel[2])
    elif sel[0] == "montage":
        dpg.add_text("CLIPS: the clip montage (the project's clips folder, cut by the triggers). It shows "
                     "wherever the BG has nothing; the BG's FX work on it too. Click a clip (or drag across "
                     "bars) to choose it: put another there, nothing, or make it a BG block.", parent=p,
                     wrap=700, color=(150, 150, 160))
    elif sel[0] == "lane":
        _draw_lane(p, sel[1])
    elif sel[0] == "block":
        mem = _member(sel[1], sel[2])
        if mem is None:
            A.sel = ("master",) if sel[1] == MASTER else ("lane", _owner(sel[1]))
            _draw_inspector()
        else:
            _draw_block(p, sel[1], mem, sel[3])


def _draw_lane(p, lane: str) -> None:
    item = _lane_item(lane)
    dpg.add_text(("FG: the foreground, drawn on its own and put over the BG; its FX work on it alone"
                  if lane == arr.FG else
                  "BG: the background, over the clip montage; its FX work on it and the montage"),
                 parent=p, color=LANE_COLOURS[lane][:3], wrap=700)
    if lane == arr.FG:
        dpg.add_checkbox(label="its FX may spill outside the figures (trails, glow, split edges)", parent=p,
                         default_value=bool(item[2].get("spill", False)),
                         callback=lambda s, v: _edit(item[3], (), lambda g: {
                             **{a: b for a, b in g.items() if a != "spill"}, **({"spill": True} if v else {})}))
    _draw_fx_list(p, item[3], ("fx",), f"FX on the whole {lane}, in order:")


def _draw_master(p) -> None:
    dpg.add_text("MASTER: over everything, in order (the last on top)", parent=p, color=LANE_COLOURS[MASTER][:3])
    with dpg.group(horizontal=True, parent=p):
        dpg.add_button(label="Add FX...", callback=lambda: _menu_at(lambda: _catalog_menu("Add FX", "fx", add_master)))
    for it in _master_items():
        spec, uid = it[2], it[3]
        kind = arr.kind_of(spec)
        with dpg.group(horizontal=True, parent=p):
            dpg.add_checkbox(default_value=spec.get("enabled", True) is not False,
                             callback=lambda s, v, *, uid=uid: _toggle(uid, ()))
            dpg.add_text(("" if kind == arr.FX else "GEN " if kind == arr.GENERATOR else "CLIP ")
                         + arr.label_of(spec), color=(230, 230, 240))
            dpg.add_text("bars")
            dpg.add_input_text(default_value=_bars_text(spec), width=110, hint="all", on_enter=True,
                               callback=lambda s, v, *, uid=uid: _set_fx_bars(uid, (), v))
            if kind != arr.CLIP:
                dpg.add_button(label="plugin", callback=lambda s, a, *, uid=uid: open_plugin(uid, ()))
            dpg.add_button(label="^", callback=lambda s, a, *, uid=uid: _move_master(uid, -1))
            dpg.add_button(label="v", callback=lambda s, a, *, uid=uid: _move_master(uid, 1))
            dpg.add_button(label="x", callback=lambda s, a, *, uid=uid: _delete_master(uid))


def _draw_block(p, lane: str, mem, bk) -> None:
    """A member of a track (``lane``: its key): its blocks' bars, on/off, its plugin, its
    place in its chain (an FX), its FX (a lane's member), a character's place in the frame."""
    uid, here = _where(lane, mem.at)
    spec = mem.spec
    what = ("FX" if mem.kind == arr.FX else "Generator" if mem.kind == arr.GENERATOR else
            "character" if lane == arr.FG else "clip")
    dpg.add_text(f"{TRACK_TITLES[lane]} · {what} {mem.label}" + ("" if mem.enabled else " (off)"), parent=p,
                 color=LANE_COLOURS[lane][:3])
    with dpg.group(horizontal=True, parent=p):
        dpg.add_text("name")
        dpg.add_input_text(default_value=str(spec.get("name") or ""), width=260, hint=arr.label_of(
            {k: v for k, v in spec.items() if k != "name"}), on_enter=True,
            callback=lambda s, v: _rename(lane, mem.at, v))
        dpg.add_text("(Enter; empty: its file's or its kind's)", color=(120, 120, 135))
    if spec.get("file"):
        dpg.add_text(str(spec["file"]), parent=p, color=(130, 130, 145), wrap=700)
    if mem.cued:
        _draw_cues_info(p, mem)
    elif mem.always:
        dpg.add_text("it plays the whole song: give it bars here", parent=p, color=(150, 150, 160))
    if (mem.cued or mem.always) and mem.kind == arr.CLIP:  # shown where it plays, but for these bars
        with dpg.group(horizontal=True, parent=p):
            dpg.add_text("hidden in bars")
            dpg.add_input_text(default_value=", ".join(f"{a}-{b}" if b > a else str(a) for a, b in _mutes(spec)),
                               width=160, hint="none", on_enter=True,
                               callback=lambda s, v: _set_mutes(lane, mem.at, v))
            dpg.add_text("(like 9-12, 30-33; Enter; or right-click a cut: Hide) - it stays in sync",
                         color=(120, 120, 135))
    if mem.kind != arr.CLIP:                             # its windows, as text (also: none, the whole song)
        with dpg.group(horizontal=True, parent=p):
            dpg.add_text("plays in bars")
            dpg.add_input_text(default_value=_bars_text(spec), width=160, hint="all the song", on_enter=True,
                               callback=lambda s, v: _set_fx_bars(uid, here, v))
            dpg.add_text("(like 9-12, 30-33; Enter)", color=(120, 120, 135))
    if bk is not None and bk < len(mem.blocks):
        b = mem.blocks[bk]

        def change(**kw):
            blocks = list(_member(lane, mem.at).blocks)
            old = blocks[bk]
            new = arr.Block(old.first, old.last, old.start, old.rate, old.file)
            for key, v in kw.items():
                setattr(new, key, v)
            if new.last < new.first:
                return
            rest = [x for i, x in enumerate(blocks) if i != bk]
            placed = arr.place(rest, new, _bar_seconds())
            _set_blocks(lane, mem.at, placed)
            A.sel = ("block", lane, mem.at, arr.block_at(placed, new.first))
            _draw()
            _draw_inspector()

        def edit(field):
            return lambda s, v: change(**{field: v})
        with dpg.group(horizontal=True, parent=p):
            dpg.add_text("bars")
            dpg.add_input_int(default_value=b.first, width=90, min_value=1, min_clamped=True,
                              callback=edit("first"), on_enter=True)
            dpg.add_text("to")
            dpg.add_input_int(default_value=b.last, width=90, min_value=1, min_clamped=True,
                              callback=edit("last"), on_enter=True)
            if mem.kind == arr.CLIP:
                dpg.add_text("from (s)")
                dpg.add_input_float(default_value=b.start, width=90, step=0.1, format="%.2f",
                                    callback=edit("start"), on_enter=True)
                dpg.add_text("speed")
                dpg.add_input_float(default_value=b.rate, width=80, step=0.1, format="%.2f",
                                    callback=edit("rate"), on_enter=True)
        if mem.kind == arr.CLIP and A.content is not None and A.content[0] == "clip":
            if lane == arr.BG:
                dpg.add_button(label="use the chosen clip and its 'from'", parent=p,
                               callback=lambda: change(file=A.content[1], start=A.content[2]))
            else:
                dpg.add_button(label="use the browser's 'from'", parent=p,
                               callback=lambda: change(start=A.content[2]))
    with dpg.group(horizontal=True, parent=p):
        if bk is not None and bk < len(mem.blocks):
            blk = mem.blocks[bk]
            dpg.add_button(label=f"Preview bars {blk.first}-{min(blk.last, blk.first + PREVIEW_BARS - 1)}",
                           callback=lambda: preview_bars(blk.first, blk.last))
        dpg.add_checkbox(label="on", default_value=mem.enabled, callback=lambda: _toggle(uid, here))
        if mem.kind != arr.CLIP:
            dpg.add_button(label="Open plugin...", callback=lambda: open_plugin(uid, here))
        if _is_fx(lane):
            dpg.add_button(label="earlier", callback=lambda: _reorder(lane, mem.at, -1))
            dpg.add_button(label="later", callback=lambda: _reorder(lane, mem.at, 1))
        if bk is not None and len(mem.blocks) > 1:
            dpg.add_button(label="delete this block", callback=lambda: _delete_block(lane, mem.at, bk))
        dpg.add_button(label=f"delete {mem.label[:20]}" + (" (all its blocks)" if len(mem.blocks) > 1 else ""),
                       callback=lambda: _remove_member(lane, mem.at))
    if _is_fx(lane):
        dpg.add_text("Its chain works in order, top to bottom of the list: earlier/later move it. Delete "
                     "deletes it; drag it, or its ends, on its track.", parent=p, color=(120, 120, 135), wrap=700)
        return
    if mem.kind == arr.CLIP and lane == arr.FG:
        _draw_placement(p, lane, mem)
    if mem.kind == arr.GENERATOR and lane == arr.BG:
        def put(key, cast):
            return lambda s, v: _set_member(lane, mem.at, {**_member(lane, mem.at).spec, key: cast(v)})
        with dpg.group(horizontal=True, parent=p):
            dpg.add_text("blend")
            dpg.add_combo(["normal", "add", "screen", "multiply"], default_value=spec.get("blend", "normal"),
                          width=100, callback=put("blend", str))
            op = spec.get("opacity", 1.0)
            dpg.add_text("opacity")
            if isinstance(op, (int, float)):
                dpg.add_slider_float(default_value=float(op), min_value=0.0, max_value=1.0, width=140,
                                     callback=put("opacity", float))
            else:
                dpg.add_text("(follows the music: plugin)", color=(200, 160, 100))
    dpg.add_separator(parent=p)
    _draw_fx_list(p, uid, here + ("fx",), f"FX on {mem.label[:24]} only, in order:")


def _rename(key: str, at: int, name: str) -> None:
    mem = _member(key, at)
    if mem is not None:
        spec = {k: v for k, v in mem.spec.items() if k != "name"}
        _set_member(key, at, {**spec, "name": name.strip()} if name.strip() else spec)


def _draw_cues_info(p, mem) -> None:
    """Where a cued member's cuts come from, and when they play: they are edited there."""
    cues = mem.spec.get("cues")
    if isinstance(cues, dict) and cues.get("reaper"):
        tracks = cues.get("tracks")
        where = (f"cut by the REAPER project {Path(str(cues['reaper'])).name}"
                 + (f", track(s) {', '.join(map(str, tracks))}" if tracks else ""))
    elif isinstance(cues, (str, Path)):
        where = f"cut by the cue file {cues}"
    else:
        where = f"cut by {len(cues or [])} cue(s) written in the scene"
    spans = _cue_bars(mem)
    when = (f"{len(spans)} cuts, from bar {int(spans[0][0])} to bar {int(spans[-1][1])}"
            if spans else "its cuts are still being read (a REAPER glue takes a moment)")
    dpg.add_text(f"{where}: {when}. Edit the cuts there (in REAPER); here, its place in the frame, "
                 "its cutout and its FX.", parent=p, color=(200, 200, 215), wrap=700)
    if spans:
        first, last = int(spans[0][0]), int(spans[-1][1])
        dpg.add_button(label=f"Preview bars {first}-{min(last, first + PREVIEW_BARS - 1)}", parent=p,
                       callback=lambda: preview_bars(first, last))


PLACES = (("x", 0.5, "0 left .. 1 right; Link: the music moves it"),
          ("y", 0.5, "0 top .. 1 bottom; Link: the music moves it"),
          ("scale", 1.0, "1: as tall as the frame; Link: the music sizes it"))


def _knob_text(v) -> str:
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
        what = f"moving at {v.get('rate', 1):g}/s, pushed by " + _knob_text(v["from"])
    else:
        what = ", ".join(f"{k}: {x}" for k, x in v.items())
    around = f"; around {v['bias']:g}" if isinstance(v.get("bias"), (int, float)) else ""
    by = f", by {v['gain']:g}" if isinstance(v.get("gain"), (int, float)) else ""
    return what + around + by


def _link_place(uid, here: tuple, key: str, default: float, label: str) -> None:
    """The studio's knob editor on a character's x, y or scale: what in the music moves it."""
    if not hasattr(A.host, "edit_knob"):
        _status("Link needs the studio's knob editor")
        return
    A.host.edit_knob(PluginTarget(uid, label), tuple(here) + (key,), default, "knob")


def unlink_place(lane: str, at: int, key: str, default: float) -> None:
    """A character's x, y or scale back to a number: where the music held it (its bias)."""
    mem = _member(lane, at)
    if mem is None:
        return
    v = mem.spec.get(key)
    fixed = float(v["bias"]) if isinstance(v, dict) and isinstance(v.get("bias"), (int, float)) else default
    _set_member(lane, at, {**mem.spec, key: round(fixed, 4)})
    _status(f"{mem.label}: {key} fixed at {fixed:g} (Link... gives it to the music again)")


def _draw_placement(p, lane: str, mem) -> None:
    """A character's place in the frame and its cutout."""
    spec = mem.spec

    def put(key, cast=float):
        return lambda s, v: _set_member(lane, mem.at, {**_member(lane, mem.at).spec, key: cast(v)})
    uid, here = _where(lane, mem.at)
    for key, default, about in PLACES:                   # each: a number, or linked to the music
        v = _num(spec, key, default)
        with dpg.group(horizontal=True, parent=p):
            dpg.add_text(f"{key:<5}")
            if v is None:
                dpg.add_text(f"follows the music: {_knob_text(spec.get(key))}", color=(200, 160, 100))
                dpg.add_button(label="Link...", callback=lambda s, a, *, key=key, d=default: _link_place(
                    uid, here, key, d, mem.label))
                dpg.add_button(label="fixed", callback=lambda s, a, *, key=key, d=default: unlink_place(
                    lane, mem.at, key, d))
            else:
                dpg.add_input_float(default_value=v, width=80, step=0.05, format="%.2f", callback=put(key),
                                    on_enter=True)
                dpg.add_button(label="Link...", callback=lambda s, a, *, key=key, d=default: _link_place(
                    uid, here, key, d, mem.label))
                dpg.add_text(about, color=(120, 120, 135))
    with dpg.group(horizontal=True, parent=p):
        dpg.add_text("fit")
        dpg.add_combo(["contain", "cover"], default_value=spec.get("fit", "contain"), width=80,
                      callback=put("fit", str))
        dpg.add_checkbox(label="flip", default_value=bool(spec.get("flip", False)), callback=put("flip", bool))
        dpg.add_checkbox(label="loop", default_value=bool(spec.get("loop", False)), callback=put("loop", bool))
    j = _sam_fx_of(mem)
    if j is not None:                                    # cut out by its SAM2 FX (at the render)
        from core.video import sam_cutout
        from core.scene_doc import plain
        cfg = sam_cutout.config(plain(spec["fx"][j]))
        seeds = cfg["seeds"]
        how = "SAM2 finds the objects" if seeds == "auto" else f"{len(seeds.get('objects') or [])} object(s) seeded"
        done = sam_cutout.cached(str(spec.get("file")), cfg) if spec.get("file") else None
        with dpg.group(horizontal=True, parent=p):
            dpg.add_text(f"cut out by SAM2 ({how}): " + ("cached, the renders read it" if done else
                                                         "made at the next preview or render, once"),
                         color=(150, 190, 150))
            uid, here = _where(lane, mem.at)
            dpg.add_button(label="seeds...", callback=lambda: open_sam(uid, here + ("fx", j)))
        _draw_spill(p, lane, mem)
        return
    with dpg.group(horizontal=True, parent=p):
        cut = spec.get("cutout") or {}
        track = cut.get("track") if isinstance(cut, dict) else cut
        dpg.add_text(f"cut out by: {Path(str(track)).name}" if track else "not cut out (whole frame)",
                     color=(150, 190, 150) if track else (150, 150, 160))
        file = spec.get("file")
        if file and not track:
            found = arr.find_mask_track(file)
            if found:
                dpg.add_button(label=f"cut out ({Path(found).name})", callback=lambda: _set_member(
                    lane, mem.at, {**_member(lane, mem.at).spec, "cutout": {"track": found, "feather": 2}}))
        if track:
            dpg.add_button(label="whole frame", callback=lambda: _set_member(
                lane, mem.at, {k: v for k, v in _member(lane, mem.at).spec.items() if k != "cutout"}))
            drop = isinstance(cut, dict) and cut.get("drop")
            dpg.add_checkbox(label="all but object 1 (a backdrop)", default_value=bool(drop),
                             callback=lambda s, v: _set_member(lane, mem.at, {**_member(lane, mem.at).spec, "cutout": {
                                 **{a: b for a, b in cut.items() if a not in ("drop", "objects")},
                                 **({"drop": [1]} if v else {})}}))
    _draw_spill(p, lane, mem)


def _draw_spill(p, lane: str, mem) -> None:
    """A character with FX: may they spill outside its figure."""
    spec = mem.spec
    if spec.get("fx"):
        dpg.add_checkbox(label="its FX may spill outside the figure (trails, glow, split edges)", parent=p,
                         default_value=bool(spec.get("spill", False)),
                         callback=lambda s, v: _set_member(lane, mem.at, {
                             **{a: b for a, b in _member(lane, mem.at).spec.items() if a != "spill"},
                             **({"spill": True} if v else {})}))


# ----------------------------------------------------------------------- the plugin window

def _item(uid):
    return next((it for it in A.items if it[3] == uid), None)


def _signals() -> dict:
    """name -> object: the signals (YAML anchors) the arrangement's layers carry, the first
    by stack order, then the rest of the scene's."""
    from core.scene_doc import _walk, _anchor_name, anchors_in
    out: dict = {}
    for it in A.items:
        for _, o in _walk(it[2]):
            name = _anchor_name(o)
            if name and name not in out:
                out[name] = o
    for name, o in (anchors_in(A.doc.data) if A.doc is not None else {}).items():
        out.setdefault(name, o)
    return out


def _defines(uid, obj) -> bool:
    """Whether the layer ``uid`` is the first (by stack order) to hold ``obj``: where a
    signal is defined; the later ones use it."""
    from core.scene_doc import _walk
    for it in A.items:
        if any(o is obj for _, o in _walk(it[2])):
            return it[3] == uid
    return False


def _param_path(spec: dict, name: str) -> tuple:
    return ("uniforms", name) if spec.get("source") == "shadertoy" and name.startswith("u_") else (name,)


def set_param(uid, path: tuple, value, anchor=None, links=None) -> str:
    """Set (``value`` None: take out) a parameter of the layer ``uid``; a knob that follows
    the music, a number, or (``links``) tied to signals: {(): obj} the whole of it, {sub
    path: obj} a part (who plays). A signal this layer defines is changed for every layer
    that uses it. Returns an error, or ""."""
    import copy
    from core.scene_doc import _anchor_name, _replace_ids, flow, get_path, plain_keeping_signals
    it = _item(uid)
    if it is None:
        return "that track is not in the arrangement any more"
    spec = plain_keeping_signals(it[2])                  # a copy: the signals in it stay shared
    holder, parents = spec, []
    for k in path[:-1]:
        nxt = holder[k] if isinstance(holder, list) else holder.get(k)
        if isinstance(nxt, dict) and _anchor_name(nxt):  # editing inside a signal: this place's own
            nxt = dict(nxt)
        elif not isinstance(nxt, (dict, list)):
            nxt = {}
        holder[k] = nxt
        parents.append((holder, k))
        holder = nxt
    last = path[-1]
    old = holder.get(last)
    links = dict(links or {})
    redefined = None                                     # a signal defined here, given a new value
    if value is None and not links:
        new = None
        holder.pop(last, None)
        if parents and parents[-1][1] == "uniforms" and not holder:    # no uniforms left
            parents[-1][0].pop("uniforms", None)
    elif () in links:
        new = links[()]
        holder[last] = new
    else:
        new = flow(copy.deepcopy(value))
        for part, obj in links.items():
            h = get_path(new, tuple(part[:-1]))
            if isinstance(h, (dict, list)):
                h[part[-1]] = obj
        if isinstance(old, (dict, list)) and _anchor_name(old) and _defines(uid, old):
            redefined = old
        name = _anchor_name(redefined) or anchor
        if name:
            if not isinstance(new, (dict, list)):
                return "only a knob that follows the music can be a signal (a plain number cannot)"
            new.yaml_set_anchor(name, always_dump=True)
        holder[last] = new
    _checkpoint()
    it[2] = spec
    if redefined is not None:                            # the layers that use it follow
        for other in A.items:
            if other is not it:
                other[2] = _replace_ids(plain_keeping_signals(other[2]), {id(redefined): new})
    A.dirty = True
    _relane()
    return ""


class PluginTarget:
    """What the studio's knob editor needs to edit a parameter the arrangement owns."""

    def __init__(self, uid, title: str):
        self.uid, self.title = uid, title

    def value(self, path):
        from core.scene_doc import _anchor_name, get_path, plain
        it = _item(self.uid)
        obj = get_path(it[2], tuple(path)) if it else None
        name = _anchor_name(obj)
        shared = (("&" if _defines(self.uid, obj) else "*") + name) if name else None
        return plain(obj), shared

    def anchors(self):
        return _signals()

    def apply(self, path, spec, anchor, links):
        from core.scene_doc import get_path, plain
        it = _item(self.uid)
        old = get_path(it[2], tuple(path)) if it else None
        if isinstance(old, dict) and isinstance(spec, dict) and set(spec) <= TRIGGER_KEYS and () not in (links or {}):
            spec = {**{k: v for k, v in plain(old).items() if k not in TRIGGER_KEYS}, **spec}   # its envelope stays
        return set_param(self.uid, tuple(path), spec, anchor, links) or None


def open_plugin(uid, at: tuple = ()) -> None:
    """A Generator's or FX's window: each parameter with its value and a Link to the music.
    ``at``: where the layer is in the top-level one ``uid`` (a lane's member, an FX of a list)."""
    from core.scene_doc import get_path
    it = _item(uid)
    spec = get_path(it[2], tuple(at)) if it is not None and at else it[2] if it is not None else None
    if isinstance(spec, dict) and spec.get("source") == SAM:
        open_sam(uid, tuple(at))
        return
    A.plugin, A.plugin_at = uid, tuple(at)
    if not dpg.does_item_exist("arr_plugin"):
        with dpg.window(tag="arr_plugin", width=560, height=640, pos=(560, 60), on_close=_plugin_closed):
            dpg.add_group(tag="arr_plugin_body")
    dpg.show_item("arr_plugin")
    _draw_plugin()
    dpg.focus_item("arr_plugin")


def _plugin_closed(*_):
    A.plugin, A.plugin_at = None, ()


def _summary(value) -> str:
    """A knob on one line: what in the music moves it."""
    from core.scene_doc import plain
    import yaml
    text = yaml.safe_dump(plain(value), default_flow_style=True, width=1 << 16).strip()
    return text if len(text) <= 64 else text[:61] + "..."


def _draw_plugin() -> None:
    from core import catalog
    from core.knob_forms import form_of
    from core.scene_doc import _anchor_name, get_path, plain
    if not dpg.does_item_exist("arr_plugin") or not dpg.is_item_shown("arr_plugin") or A.plugin is None:
        return
    body = "arr_plugin_body"
    dpg.delete_item(body, children_only=True)
    it = _item(A.plugin)
    if it is None:
        dpg.configure_item("arr_plugin", label="Plugin")
        dpg.add_text("this track is not in the arrangement now (undone or deleted)", parent=body,
                     color=(200, 160, 100))
        return
    uid = it[3]
    at = A.plugin_at
    spec = get_path(it[2], at) if at else it[2]
    if not isinstance(spec, dict):
        dpg.add_text("this is not in the arrangement now (undone or deleted)", parent=body, color=(200, 160, 100))
        return
    e = catalog.of_layer(spec)
    kind = "FX" if e and e.kind == catalog.FX else "Generator"
    name = e.label if e else str(spec.get("source"))
    owner = get_path(it[2], at[:-2]) if len(at) >= 2 and at[-2] == "fx" else None
    whose = f" on {_layer_name(owner)}" if isinstance(owner, dict) else ""
    dpg.configure_item("arr_plugin", label=f"{name} - {kind}{whose}" + (f" · {e.category}" if e else ""))
    if e:
        dpg.add_text(e.about, parent=body, wrap=530)
    on = spec.get("enabled", True) is not False
    with dpg.group(horizontal=True, parent=body):
        dpg.add_checkbox(label="on", default_value=on,
                         callback=lambda s, v: set_param(uid, at + ("enabled",), None if v else False))
        if kind == "Generator":
            dpg.add_text("blend")
            dpg.add_combo(["normal", "add", "screen", "multiply"], default_value=spec.get("blend", "normal"),
                          width=90, callback=lambda s, v: set_param(uid, at + ("blend",), None if v == "normal" else v))
    params = list(e.params) if e else []
    if kind == "FX":
        params.insert(0, catalog.Param("mix", 1.0, "dry/wet: 0 the picture it was given .. 1 all of its result"))
    else:
        params.insert(0, catalog.Param("opacity", 1.0, "how much of it shows over what is under it, 0..1"))
    known = {p.name for p in params}
    extra = [k for k in spec if k not in known and k not in catalog.COMMON_KEYS
             and k not in ("file", "uniforms", "mix", "fx", "spill")]
    params += [catalog.Param(k, None, "") for k in extra]
    dpg.add_separator(parent=body)
    dpg.add_text("~ the music can move it: Link chooses what (kick, snare, amplitude, notes, chords...)",
                 parent=body, color=(120, 120, 140), wrap=530)
    with dpg.table(parent=body, header_row=False, policy=dpg.mvTable_SizingFixedFit, scrollY=True,
                   height=-1, borders_innerH=True):
        dpg.add_table_column(width_fixed=True, init_width_or_weight=130)
        dpg.add_table_column(width_stretch=True)
        dpg.add_table_column(width_fixed=True, init_width_or_weight=110)
        for prm in params:
            path = at + _param_path(spec, prm.name)
            value = get_path(it[2], path)
            section = _section_of(spec, prm.name)
            if section is not None:                      # a structure: its own controls
                section(uid, path, value)
                continue
            with dpg.table_row():
                movable = prm.knob or _is_from(prm, value)
                label = dpg.add_text(("~ " if movable else "  ") + prm.name,
                                     color=(230, 210, 140) if value is not None else (170, 170, 180))
                if prm.about:
                    with dpg.tooltip(label):
                        dpg.add_text(prm.about, wrap=380)
                numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
                if numeric or (value is None and _is_number(prm.default)):     # fixed ones too (rows, seed)
                    shown = float(value) if numeric else float(prm.default)
                    dpg.add_input_float(default_value=shown, width=-1, step=0, format="%.4g", on_enter=True,
                                        callback=lambda s, v, *, path=path: set_param(uid, path, v))
                elif _is_flag(value, prm):
                    dpg.add_checkbox(default_value=bool(prm.default if value is None else value),
                                     callback=lambda s, v, *, path=path: set_param(uid, path, bool(v)))
                elif _is_colour(value, prm):
                    rgb = [int(c) for c in (value if value is not None else prm.default or [0, 0, 0])[:3]]
                    dpg.add_color_edit(default_value=rgb, no_alpha=True, width=-1,
                                       callback=lambda s, v, *, path=path: set_param(
                                           uid, path, [int(round(c * 255 if max(v[:3]) <= 1 else c)) for c in v[:3]]))
                elif _is_numbers(value, prm):
                    nums = list(value if value is not None else prm.default)
                    with dpg.group(horizontal=True):
                        for k, x in enumerate(nums):
                            dpg.add_input_float(default_value=float(x), width=80, step=0, format="%.4g",
                                                on_enter=True, callback=lambda s, v, *, path=path, k=k, nums=nums:
                                                set_param(uid, path, [v if j == k else x for j, x in enumerate(nums)]))
                elif value is None and (_is_trigger(prm.name, value) or prm.name in SCORES) or value is not None and (
                        _anchor_name(value) or isinstance(value, dict) and (
                            form_of(plain(value)) is not None or _is_trigger(prm.name, value))):
                    sig = _anchor_name(value) if value is not None else None
                    with dpg.group(horizontal=True):
                        dpg.add_text("-" if value is None else ("*" + sig + "  " if sig else "") + _summary(value),
                                     color=(120, 120, 130) if value is None else (230, 210, 140) if sig
                                     else (170, 210, 170))
                        if prm.name in PULSES and isinstance(value, dict):     # each hit: how long
                            dpg.add_text("envelope")
                            dpg.add_input_float(default_value=float(plain(value).get("envelope", 0.25)), width=70,
                                                step=0, format="%.3g", on_enter=True,
                                                callback=lambda s, v, *, path=path: set_param(
                                                    uid, path + ("envelope",), max(0.0, v)))
                            with dpg.tooltip(dpg.last_item()):
                                dpg.add_text("seconds each hit lasts")
                else:                                    # a folder, a name, a list...: typed
                    _text_cell(uid, path, prm, value)
                with dpg.group(horizontal=True):
                    if prm.name in SCORES:                   # the chords: a transcription
                        dpg.add_button(label="Chords from", callback=lambda s, a, *, path=path: _link(
                            uid, path, 0.0, "score"))
                    elif _is_trigger(prm.name, value):       # who plays: a track, notes, a stem
                        dpg.add_button(label="Who plays", callback=lambda s, a, *, path=path: _link(
                            uid, path, 0.0, "trigger"))
                    elif _is_from(prm, value) and not isinstance(value, dict):   # a feature: from it on
                        dpg.add_button(label="Link", callback=lambda s, a, *, path=path, prm=prm, value=value:
                                       _link_from(uid, path, value if value is not None else prm.default))
                    elif prm.knob or (isinstance(value, dict) and form_of(plain(value)) is not None):
                        dpg.add_button(label="Link" if not isinstance(value, dict) else "Edit",
                                       callback=lambda s, a, *, path=path, d=prm.default: _link(uid, path, d))
                    if value is not None:
                        dpg.add_button(label="x", callback=lambda s, a, *, path=path: set_param(uid, path, None))
                        with dpg.tooltip(dpg.last_item()):
                            dpg.add_text("back to its default")


# ----------------------------------------------------------------------- clip reactivity

CLIP_ORDERS = ("sequential", "shuffle", "random")
CLIP_ENDS = ("bounce", "loop")
CLIP_ACTIONS = (("next_clip", "next clip"), ("random_clip", "random clip"), ("restart", "restart"),
                ("reverse", "reverse"))
GRAVITY = (("peak", 3.0, "the clip's speed at the hit"), ("floor", 0.3, "its speed far from any hit"),
           ("radius", 0.45, "how far a hit pulls, seconds"), ("curve", 2.0, "over 1: the pull close to the hit"))
NEW_TRIGGER = {"notes": [36], "actions": ["next_clip"]}


def _cell_label(text: str, about: str = "", on: bool = True, colour=None):
    label = dpg.add_text(text, color=colour or ((230, 210, 140) if on else (170, 170, 180)))
    if about:
        with dpg.tooltip(label):
            dpg.add_text(about, wrap=380)
    return label


def _who(spec) -> str:
    """Who plays a trigger, in a few words."""
    from core.scene_doc import _anchor_name, plain
    sig = _anchor_name(spec)
    if sig:
        return f"*{sig}"
    spec = plain(spec) if isinstance(spec, dict) else {}
    parts = [f"{k} {spec[k]}" for k in ("track", "notes", "audio", "score", "events") if k in spec]
    return ", ".join(str(x) for x in parts) or "nobody yet"


def _draw_reactivity(uid, path: tuple, value) -> None:
    """The clips a layer plays of its own (``clip_reactivity``, a ClipComposer config), each
    option a control: when a clip changes, in what order, and the triggers: who plays (the
    studio's editor) and what it does to the clip."""
    from core.scene_doc import plain
    cfg = plain(value) if isinstance(value, dict) else {}
    put = lambda key, v: set_param(uid, path + (key,), v)      # noqa: E731
    with dpg.table_row():
        _cell_label(path[-1], "the clips it plays: when one changes, in what order, what the music does to them")
        dpg.add_text("the clips it shows", color=(150, 150, 165))
        if value is not None:
            dpg.add_button(label="x", callback=lambda: set_param(uid, path, None))
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("all of these back to their defaults (no triggers)")
    with dpg.table_row():
        _cell_label("   a new clip", "every N bars (counted from the song's first); off: only when a trigger says",
                    "clip_per_bar" in cfg or "bars_per_clip" in cfg)
        with dpg.group(horizontal=True):
            dpg.add_checkbox(label="every", default_value=bool(cfg.get("clip_per_bar", True)),
                             callback=lambda s, v: put("clip_per_bar", bool(v)))
            dpg.add_input_int(default_value=int(cfg.get("bars_per_clip", 1)), width=90, min_value=1,
                              min_clamped=True, on_enter=True, callback=lambda s, v: put("bars_per_clip", max(1, v)))
            dpg.add_text("bar(s)")
        dpg.add_text("")
    with dpg.table_row():
        _cell_label("   clip_order", "sequential: the folder's order (every face in step); shuffle: each clip once "
                    "per round, in a random order (each face its own); random: any clip each time",
                    "clip_order" in cfg)
        dpg.add_combo(CLIP_ORDERS, default_value=cfg.get("clip_order", "sequential"), width=-1,
                      callback=lambda s, v: put("clip_order", v))
        dpg.add_text("")
    with dpg.table_row():
        _cell_label("   clip_end", "a clip shorter than its turn: bounce (back and forth) or loop (start over)",
                    "clip_end" in cfg)
        dpg.add_combo(CLIP_ENDS, default_value=cfg.get("clip_end", "bounce"), width=-1,
                      callback=lambda s, v: put("clip_end", v))
        dpg.add_text("")
    with dpg.table_row():
        _cell_label("   seed", "the same number: the same draw every render; empty: a new one each time", "seed" in cfg)
        dpg.add_input_text(default_value="" if cfg.get("seed") is None else str(cfg["seed"]), hint="none",
                           decimal=True, width=-1, on_enter=True,
                           callback=lambda s, v: put("seed", int(v) if v.strip().lstrip("-").isdigit() else None))
        dpg.add_text("")
    trs = cfg.get("triggers") if isinstance(cfg.get("triggers"), dict) else {}
    tpath = path + ("triggers",)
    with dpg.table_row():
        _cell_label("   triggers", "a hit (a track, notes, a stem, a chord change) and what it does to the clip",
                    bool(trs))
        dpg.add_text("what the music does to the clip" if trs else "none: the clips only change with the bars",
                     color=(150, 150, 165))
        dpg.add_button(label="+ trigger", callback=lambda: _add_trigger(uid, tpath))
    raw = get_path_of(uid, tpath) or {}
    for name, spec in trs.items():
        _draw_trigger(uid, tpath, name, spec, raw.get(name), [k for k in trs if k != name])


def get_path_of(uid, path: tuple):
    from core.scene_doc import get_path
    it = _item(uid)
    return get_path(it[2], tuple(path)) if it is not None else None


def _draw_trigger(uid, tpath: tuple, name: str, spec: dict, raw, others: list) -> None:
    """One trigger: its name, who plays it, what it does; softer hits left out, until another
    comes in, not near another's hits (a mic's bleed), the clip's speed pulled by its hits."""
    spec = spec if isinstance(spec, dict) else {}
    me = tpath + (name,)
    put = lambda key, v: set_param(uid, me + (key,), v)         # noqa: E731
    with dpg.table_row():
        dpg.add_input_text(default_value=name, width=-1, on_enter=True,
                           callback=lambda s, v: _rename_trigger(uid, tpath, name, v))
        with dpg.tooltip(dpg.last_item()):
            dpg.add_text("its name (Enter renames it)")
        dpg.add_text(_who(raw if raw is not None else spec), color=(170, 210, 170))
        with dpg.group(horizontal=True):
            dpg.add_button(label="Who plays", callback=lambda: _link(uid, me, 0.0, "trigger"))
            dpg.add_button(label="x", callback=lambda: _drop_trigger(uid, tpath, name))
            with dpg.tooltip(dpg.last_item()):
                dpg.add_text("take this trigger out")
    acts = list(spec.get("actions") or [])
    with dpg.table_row():
        _cell_label("     does", "what each hit does to the clip (none: only its gravity, if any)", True,
                    (150, 150, 165))
        with dpg.group(horizontal=True):
            for act, label in CLIP_ACTIONS:
                dpg.add_checkbox(label=label, default_value=act in acts,
                                 callback=lambda s, v, *, act=act: put("actions", [
                                     a for a, _ in CLIP_ACTIONS if (a in acts and a != act) or (a == act and v)] or None))
        dpg.add_text("")
    with dpg.table_row():
        _cell_label("     only", "min velocity: softer hits are left out (0..127); until: it stops at the other's "
                    "first hit; not near: its hits close to the other's go (a drum mic's bleed)", True,
                    (150, 150, 165))
        ex = spec.get("exclude") if isinstance(spec.get("exclude"), dict) else {}
        with dpg.group(horizontal=True):
            dpg.add_text("min velocity")
            dpg.add_input_int(default_value=int(spec.get("min_velocity", 0)), width=80, min_value=0, max_value=127,
                              min_clamped=True, max_clamped=True, on_enter=True,
                              callback=lambda s, v: put("min_velocity", v or None))
            dpg.add_text("until")
            dpg.add_combo(["-"] + others, default_value=spec.get("until") or "-", width=90,
                          callback=lambda s, v: put("until", None if v == "-" else v))
            dpg.add_text("not near")
            dpg.add_combo(["-"] + others, default_value=ex.get("trigger") or "-", width=90,
                          callback=lambda s, v: put("exclude", None if v == "-" else
                                                    {"trigger": v, "window": float(ex.get("window", 0.04))}))
        dpg.add_text("")
    g = spec.get("gravity") if isinstance(spec.get("gravity"), dict) else None
    with dpg.table_row():
        _cell_label("     gravity", "its hits pull the clip's speed: fast at the hit, slow far from any", True,
                    (150, 150, 165))
        with dpg.group(horizontal=True):
            dpg.add_checkbox(default_value=g is not None,
                             callback=lambda s, v: put("gravity", {k: d for k, d, _ in GRAVITY} if v else None))
            for key, default, about in (GRAVITY if g is not None else ()):
                dpg.add_text(key)
                dpg.add_input_float(default_value=float(g.get(key, default)), width=70, step=0, format="%.3g",
                                    on_enter=True, callback=lambda s, v, *, key=key: set_param(
                                        uid, me + ("gravity", key), v))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text(about)
        dpg.add_text("")


def _add_trigger(uid, tpath: tuple) -> None:
    """A new trigger (the kick, next clip), then the studio's editor to say who plays it."""
    have = get_path_of(uid, tpath) or {}
    name = next(n for n in ["kick", "snare"] + [f"trigger{i}" for i in range(1, 99)] if n not in have)
    err = set_param(uid, tpath + (name,), dict(NEW_TRIGGER))
    if err:
        _status(err)
        return
    _link(uid, tpath + (name,), 0.0, "trigger")


def _rename_trigger(uid, tpath: tuple, old: str, new: str) -> None:
    """A trigger by another name (``until``/``not near`` that named it follow)."""
    new = new.strip()
    if not new or new == old:
        return
    if new in (get_path_of(uid, tpath) or {}):
        _status(f"there is a trigger {new} already")
        _draw_plugin()
        return
    _edit(uid, tpath, lambda trs: _renamed(trs, old, new))


def _drop_trigger(uid, tpath: tuple, name: str) -> None:
    """A trigger out (``until``/``not near`` that named it go too: nothing left to name)."""
    _edit(uid, tpath, lambda trs: _renamed(trs, name, None) or None)


def _renamed(trs, old: str, new) -> dict:
    """``trs`` with ``old`` called ``new`` (None: taken out) and the references to it."""
    out = {}
    for k, v in (trs or {}).items():
        if k == old and new is None:
            continue
        if isinstance(v, dict) and (v.get("until") == old or (v.get("exclude") or {}).get("trigger") == old):
            v = dict(v)                                  # its own now (a shared signal stays as it was)
            if v.get("until") == old:
                v["until"] = new
                if new is None:
                    v.pop("until")
            if (v.get("exclude") or {}).get("trigger") == old:
                v["exclude"] = {**v["exclude"], "trigger": new} if new is not None else None
                if new is None:
                    v.pop("exclude")
        out[new if k == old else k] = v
    return out


# ----------------------------------------------------------------------- gestures

def _gestures_of(spec: dict) -> tuple:
    """The gestures a layer's ``hits:`` can ask for (its class's GESTURES), or ()."""
    from core import catalog
    import importlib
    got = catalog.NATIVE.get(str(spec.get("source")))
    if not got:
        return ()
    try:
        return tuple(getattr(getattr(importlib.import_module(got[0]), got[1]), "GESTURES", ()))
    except (ImportError, AttributeError):
        return ()


def _draw_gestures(uid, path: tuple, value, gestures: tuple) -> None:
    """A layer's ``hits:``: each gesture a row: who plays it, which gesture, how long."""
    from core.scene_doc import plain
    hits = [h for h in (plain(value) if isinstance(value, list) else [])]
    raw = get_path_of(uid, path) or []
    with dpg.table_row():
        _cell_label(path[-1], "what the music makes it do: " + ", ".join(gestures), bool(hits))
        dpg.add_text(f"gestures: {', '.join(gestures)}" if hits else "none yet", color=(150, 150, 165))
        dpg.add_button(label="+ gesture", callback=lambda: _add_gesture(uid, path, gestures))
    for i, h in enumerate(hits):
        h = h if isinstance(h, dict) else {}
        me = path + (i,)
        with dpg.table_row():
            dpg.add_combo(gestures, default_value=h.get("gesture", gestures[0]), width=-1,
                          callback=lambda s, v, *, me=me: set_param(uid, me + ("gesture",), v))
            with dpg.group(horizontal=True):
                dpg.add_text(_who(raw[i] if i < len(raw) else h), color=(170, 210, 170))
                dpg.add_text("  for")
                dpg.add_input_float(default_value=float(h.get("envelope", 0.2)), width=70, step=0, format="%.3g",
                                    on_enter=True, callback=lambda s, v, *, me=me: set_param(
                                        uid, me + ("envelope",), max(0.001, v)))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("seconds each hit's gesture lasts")
                dpg.add_text("s")
            with dpg.group(horizontal=True):
                dpg.add_button(label="Who plays", callback=lambda s, a, *, me=me: _link(uid, me, 0.0, "trigger"))
                dpg.add_button(label="x", callback=lambda s, a, *, i=i: _edit(
                    uid, path, lambda ls, i=i: [x for j, x in enumerate(ls or []) if j != i] or None))
                with dpg.tooltip(dpg.last_item()):
                    dpg.add_text("take this gesture out")


def _add_gesture(uid, path: tuple, gestures: tuple) -> None:
    """A new gesture (the kick's, the first kind), then the studio's editor: who plays it."""
    have = list(get_path_of(uid, path) or [])
    _edit(uid, path, lambda ls: list(ls or []) + [{"notes": [36], "gesture": gestures[0], "envelope": 0.2}])
    _link(uid, path + (len(have),), 0.0, "trigger")


def _section_of(spec: dict, name: str):
    """The controls of a structure parameter (fn(uid, path, value)), or None."""
    if name == "hits":
        gestures = _gestures_of(spec)
        if gestures:
            return lambda uid, path, value: _draw_gestures(uid, path, value, gestures)
    return SECTIONS.get(name)


SCORES = ("harmony",)   # a transcription's chords (the studio's editor, score mode)


def _is_number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _is_numbers(value, prm) -> bool:
    """A short list of numbers (not a colour): one field each."""
    v = value if value is not None else prm.default
    return isinstance(v, list) and 0 < len(v) <= 4 and all(_is_number(x) for x in v)


def _is_from(prm, value) -> bool:
    """``..._from``: a feature (or any knob) a layer reads from the music."""
    return prm.name.endswith("_from") and (value is None or isinstance(value, (str, dict)))


def _link_from(uid, path: tuple, now) -> None:
    """The knob editor on a feature parameter, starting from the feature it reads now."""
    if isinstance(now, str) and get_path_of(uid, path) is None:
        set_param(uid, path, now)
    _link(uid, path, 0.0)


SECTIONS = {"clip_reactivity": _draw_reactivity}


PICKS = {"dir": ("_dir", "dir", "folder", "_folder", "masks"), "file": ("file", "_file", "path", "_path",
                                                                          "image", "video", "audio", "lut",
                                                                          "voice")}


def _picks(name: str):
    """"dir" or "file": the parameter names a folder or a file (a ... button picks it)."""
    for kind, ends in PICKS.items():
        if name in ends or any(name.endswith(e) for e in ends if e.startswith("_")):
            return kind
    return None


def _is_flag(value, prm) -> bool:
    return isinstance(value, bool) or value is None and isinstance(prm.default, bool)


def _is_colour(value, prm) -> bool:
    """[r, g, b] 0..255 in a parameter named for a colour."""
    v = value if value is not None else prm.default
    return (prm.name.endswith(("color", "colour")) and isinstance(v, list) and len(v) in (3, 4)
            and all(isinstance(c, (int, float)) and not isinstance(c, bool) for c in v)
            and (any(c > 1 for c in v) or all(isinstance(c, int) for c in v)))


def _yaml_line(value) -> str:
    from core.scene_doc import plain
    import yaml
    return yaml.safe_dump(plain(value), default_flow_style=True, width=1 << 16).strip()


def _text_cell(uid, path: tuple, prm, value) -> None:
    """A parameter that is not a number: its text (a folder, a name) or, a list or a table,
    its YAML on one line. Enter sets it; empty takes it out (back to its default)."""
    is_text = isinstance(value, str) or value is None and isinstance(prm.default, str)
    shown = value if isinstance(value, str) else "" if value is None else _yaml_line(value)
    pick = _picks(prm.name)
    hint = {"dir": "a folder", "file": "a file"}.get(pick) if pick else \
        ("" if prm.default is None else str(prm.default) if is_text else _yaml_line(prm.default))
    if pick and "required" in prm.about:
        hint += " (required)"
    with dpg.group(horizontal=True):
        dpg.add_input_text(default_value=shown, hint=hint or "", width=-40 if pick else -1, on_enter=True,
                           callback=lambda s, v, *, is_text=is_text: _set_text(uid, path, v, is_text))
        if pick:
            dpg.add_button(label="...", callback=lambda: _pick_param(uid, path, pick, shown))


def _set_text(uid, path: tuple, text: str, is_text: bool) -> None:
    import yaml
    text = text.strip()
    if not text:
        set_param(uid, path, None)
        return
    if not is_text:
        try:
            value = yaml.safe_load(text)
        except yaml.YAMLError as e:
            _status(f"{path[-1]}: not something it reads ({getattr(e, 'problem', None) or e})")
            return
    else:
        value = text
    err = set_param(uid, path, value)
    _status(err or f"{path[-1]} = {text}")


def _pick_param(uid, path: tuple, kind: str, now: str = "") -> None:
    """The folder (or file) dialog for a parameter (Windows' own: any drive); what it picks
    is set there."""
    A.param_pick = (uid, path)
    if arc_window.pick(kind, lambda chosen: _picked_param(None, {"file_path_name": chosen}),
                       now or (_clips_folder() or ""), f"{path[-1]}: choose a {'folder' if kind == 'dir' else 'file'}"):
        return
    tag = "dlg_arr_param_dir" if kind == "dir" else "dlg_arr_param_file"
    start = Path(now) if now else None
    if start is not None and start.exists():
        dpg.configure_item(tag, default_path=str(start if start.is_dir() else start.parent))
    dpg.show_item(tag)


def _picked_param(sender, app_data) -> None:
    path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
    if path and A.param_pick:
        uid, at = A.param_pick
        err = set_param(uid, at, Path(path).as_posix())
        _status(err or f"{at[-1]} = {Path(path).as_posix()}")
    A.param_pick = None


def _layer_name(spec: dict) -> str:
    from core import catalog
    e = catalog.of_layer(spec)
    if e:
        return e.label
    return Path(str(spec["file"])).stem if spec.get("file") else str(spec.get("source"))


TRIGGER_KEYS = {"track", "notes", "audio", "threshold", "min_gap", "score", "events", "snap", "voice"}


def _is_trigger(name: str, value) -> bool:
    """A parameter that says who plays (a trigger source), not how much."""
    from core.scene_doc import plain
    if isinstance(value, dict):
        return bool(set(plain(value)) & TRIGGER_KEYS) and not set(plain(value)) & {"hits", "feature", "lfo"}
    return value is None and name in ("trigger", "hits") + PULSES


PULSES = ("zoom_pulse", "invert")   # a trigger and how long each hit lasts (``envelope``, seconds)


def _link(uid, path: tuple, default, mode: str = "knob") -> None:
    """The studio's knob editor on this parameter (``mode`` trigger: who plays it)."""
    it = _item(uid)
    if it is None or not hasattr(A.host, "edit_knob"):
        _status("Link needs the studio's knob editor")
        return
    from core.scene_doc import get_path
    at = A.plugin_at if path[:len(A.plugin_at)] == A.plugin_at else ()
    title = _layer_name(get_path(it[2], at) if at else it[2])
    A.host.edit_knob(PluginTarget(uid, title), path, default if isinstance(default, (int, float)) else 0.0, mode)


# ----------------------------------------------------------------------- the browser

def _status(msg: str) -> None:
    if dpg.does_item_exist("arr_status"):
        dpg.set_value("arr_status", msg)


def btn_list() -> None:
    """The folder's clips, each with its picture (made in the background, kept) and name."""
    folder = dpg.get_value("arr_folder")
    pattern = dpg.get_value("arr_filter")
    A.clips = arr.list_clips(folder, pattern)
    dpg.delete_item("arr_list", children_only=True)
    A.list_items, A.list_names, A.list_pics = {}, {}, {}
    for p in A.clips:
        path = str(p)
        with dpg.group(horizontal=True, parent="arr_list"):
            pic = dpg.add_image(A.covers.get(path, "arr_cover_blank"), width=COVER, height=COVER)
            name = dpg.add_selectable(label=_list_label(p.relative_to(folder).as_posix()), height=COVER,
                                      callback=lambda *_: _mark_chosen())
        for row in (pic, name):
            dpg.bind_item_handler_registry(row, "arr_list_handler")
            A.list_items[row] = path
        A.list_names[name] = path
        A.list_pics.setdefault(path, []).append(pic)
    _mark_chosen()
    if A.clips:
        _remember(clips_folder=str(Path(folder).resolve()))
    _make_covers([str(p) for p in A.clips[:COVERS_MAX] if str(p) not in A.covers])
    _status(f"{len(A.clips)} clip(s) in {folder}" + (f" matching '{pattern}'" if pattern else "")
            + " · drag one onto the BG, the FG or CLIPS (or drop one from the Explorer there)")


def _remember(**values) -> None:
    """Kept for the next time the window opens (STATE_FILE)."""
    import json
    got = _remembered()
    got.update(values)
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(got, indent=1), encoding="utf-8")
    except OSError:
        pass


def _remembered() -> dict:
    import json
    try:
        got = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def _list_label(name: str, seconds=None) -> str:
    """A clip's name in the list, in lines beside its picture, and how long it plays."""
    lines = []
    while name and len(lines) < 3:
        cut = len(name) if len(name) <= NAME_CHARS else \
            max((name.rfind(c, 0, NAME_CHARS) + 1 for c in "_-. /"), default=0) or NAME_CHARS
        if cut < NAME_CHARS // 2:                        # nowhere good to break: a full line
            cut = min(len(name), NAME_CHARS)
        lines.append(name[:cut])
        name = name[cut:]
    if seconds:
        lines.append(f"{seconds:.1f} s")
    return "\n".join(lines)


def _make_covers(paths: list) -> None:
    """Each clip's picture (and length), made in the background (cached on disk)."""
    A.covers_gen += 1
    if not paths:
        return

    def run(gen=A.covers_gen):
        import cv2
        for path in paths:
            if A.covers_gen != gen:                      # another list since
                return
            try:
                png = arr.cover(path, COVER)
                seconds = arr.duration(path) if Path(path).suffix.lower() in arr.VIDEOS else 0.0
            except Exception:                            # a clip that can't be read: no picture
                continue
            img = cv2.imread(str(png), cv2.IMREAD_COLOR) if png else None
            A.covers_ready.append((gen, path, _thumb_rgba(img, COVER), seconds))
    threading.Thread(target=run, daemon=True).start()


def _show_covers() -> None:
    """The pictures made since the last frame, put in the list (on the main thread)."""
    while A.covers_ready:
        gen, path, rgba, seconds = A.covers_ready.pop(0)
        if path not in A.covers:
            A.covers[path] = dpg.add_static_texture(COVER, COVER, rgba, parent="arr_covers")
        for pic in A.list_pics.get(path, []):
            if dpg.does_item_exist(pic):
                dpg.configure_item(pic, texture_tag=A.covers[path])
        if gen == A.covers_gen and seconds:
            folder = dpg.get_value("arr_folder") if dpg.does_item_exist("arr_folder") else ""
            for name, p in A.list_names.items():
                if p == path and dpg.does_item_exist(name):
                    try:
                        shown = Path(p).relative_to(folder).as_posix()
                    except ValueError:
                        shown = Path(p).name
                    dpg.configure_item(name, label=_list_label(shown, seconds))


def _mark_chosen() -> None:
    """The list shows the chosen clip (a selectable flips itself when clicked: put back)."""
    for row, path in A.list_names.items():
        if dpg.does_item_exist(row):
            dpg.set_value(row, path == A.clip)


def on_list_press(sender, app_data) -> None:
    """The mouse goes down on a clip in the list: it is chosen (and plays), and dragging it
    onto a lane puts it there."""
    path = A.list_items.get(app_data[1]) if isinstance(app_data, (list, tuple)) else None
    if path is None:
        return
    if path != A.clip:
        _choose_clip(path, 0.0)
    _start_clip_drag()


def on_picture_press(sender, app_data) -> None:
    """The mouse goes down on the playing clip: dragged onto a lane, it goes there."""
    if A.clip:
        _start_clip_drag()


def _start_clip_drag() -> None:
    A.clip_drag = (A.clip, A.start, max(0.0, (A.clip_len or 0.0) - A.start))


def _clips_folder():
    """The folder the browser lists (or the studio's clips), as a scene writes it."""
    folder = dpg.get_value("arr_folder") if dpg.does_item_exist("arr_folder") else ""
    folder = folder or (A.host.clips_dir() if A.host is not None else "") or ""
    return Path(folder).resolve().as_posix() if folder and Path(folder).is_dir() else None


def _choose_folder() -> None:
    """The clips' folder, from Windows' own dialog (any drive); dearpygui's when there is none."""
    if not arc_window.pick("dir", lambda chosen: _pick_folder(None, {"file_path_name": chosen}),
                           dpg.get_value("arr_folder") or "", "A folder of clips"):
        dpg.show_item("dlg_arr_folder")


def _pick_folder(sender, app_data) -> None:
    path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
    if path:
        dpg.set_value("arr_folder", path)
        btn_list()


def _show_content() -> None:
    from core import catalog
    if not dpg.does_item_exist("arr_content"):
        return
    c = A.content
    text = ("drag on BG or FG puts: " + (f"clip {Path(c[1]).name} @{c[2]:g}s (on CLIPS: Put it there)"
                                         if c[0] == "clip" else f"Generator {catalog.by_key(c[1]).label}")) if c else \
        "pick a clip or a Generator to put it in a lane"
    dpg.set_value("arr_content", text)


def _choose_clip(path: str, start: float) -> None:
    A.clip = str(path)
    A.clip_len = arr.duration(A.clip)
    A.start = min(start, A.clip_len)
    dpg.configure_item("arr_from", max_value=max(0.01, A.clip_len))
    dpg.set_value("arr_from", A.start)
    mask = arr.find_mask_track(A.clip)
    dpg.set_value("arr_clip_info", f"{Path(A.clip).name} · {A.clip_len:.1f} s"
                  + (f" · mask: {Path(mask).name}" if mask else " · no mask track"))
    A.content = ("clip", A.clip, A.start)
    _show_content()
    _mark_chosen()
    _play_clip(A.clip, A.start)


def on_from(sender, value) -> None:
    A.start = float(value)
    if A.clip:
        A.clip_restart = time.time()                     # it plays from there once "from" rests
        A.content = ("clip", A.clip, A.start)
        _show_content()


def btn_segment() -> None:
    """The chosen clip's character cut out: the SAM2 cutout FX on its member in the FG (its
    seeds window opens); SAM2 runs at the next preview or render, once (then cached)."""
    if not A.clip:
        _status("choose the clip first: its block in the FG (or the list)")
        return
    name = Path(A.clip).name
    mem = next((m for m in A.members.get(arr.FG, []) if m.spec.get("file")
                and Path(str(m.spec["file"])).name == name), None)
    if mem is None:
        _status(f"{name} is not in the FG: drag it there first, then cut it out")
        return
    uid, here = _where(arr.FG, mem.at)
    j = _sam_fx_of(mem)
    if j is None:
        add_fx(uid, here + ("fx",), SAM)
    else:
        open_sam(uid, here + ("fx", j))


# ----------------------------------------------------------------------- the SAM2 seeds window

SAM = "sam_cutout"
SAM_VIEW = 480         # the seeds window's picture, a side
SAM_MODES = ("auto: SAM2 finds the objects by itself (as always)", "my seeds: boxes and points on a frame")
SAM_TOOLS = ("box", "+ point (keep this)", "- point (leave this out)")
SAM_KEEPS = ("the figures (not a backdrop SAM2 took for an object)", "every object", "all but object 1")
SAM_NEAR = 0.04        # 0..1 of the frame: a right click takes away a seed this near


def _sam_fx_of(mem):
    """Where a member's SAM2 cutout FX is in its ``fx:``, or None."""
    for j, f in enumerate(mem.spec.get("fx") or []):
        if isinstance(f, dict) and f.get("source") == SAM:
            return j
    return None


def open_sam(uid, path: tuple, new: bool = False) -> None:
    """The SAM2 cutout FX at ``path`` in the layer ``uid``: how SAM2 finds what to cut out of
    its clip. By itself (as always: the default), or seeds of one's own: boxes and points
    (+ keep, - leave out) per object, on a frame of the clip (the one its block starts on).
    ``new``: just added (Cancel takes it away again)."""
    from core.scene_doc import get_path, plain
    from core.video import sam_cutout
    it = _item(uid)
    path = tuple(path)
    fx = get_path(it[2], path) if it is not None else None
    owner = get_path(it[2], path[:-2]) if it is not None and len(path) >= 2 and path[-2] == "fx" else None
    clip = owner.get("file") if isinstance(owner, dict) and owner.get("source") == "footage" else None
    if not isinstance(fx, dict) or not clip:
        _status("the SAM2 cutout works on a clip: add it to a character's FX (its block in the FG)")
        return
    cfg = sam_cutout.config(plain(fx))
    blocks = plain(owner).get("blocks") or []
    start = float(blocks[0].get("from", 0.0)) if blocks and isinstance(blocks[0], dict) else 0.0
    seeds = cfg["seeds"] if isinstance(cfg["seeds"], dict) else None
    objects = [{"box": list(o["box"]) if o.get("box") else None, "points": [list(q) for q in o.get("points") or []]}
               for o in (seeds or {}).get("objects") or []] or [{"box": None, "points": []}]
    A.sam = {"uid": uid, "path": path, "clip": str(clip), "new": new, "objects": objects, "obj": 0,
             "drag": None, "rect": (0.0, 0.0, float(SAM_VIEW), float(SAM_VIEW)), "start": start,
             "kept": {k: v for k, v in plain(fx).items() if k in ("enabled", "grow", "objects")}}
    dur = arr.duration(str(clip))
    dpg.configure_item("arr_sam_at", max_value=max(0.01, dur))
    dpg.set_value("arr_sam_at", float(seeds.get("at", start)) if seeds else min(start, dur))
    dpg.set_value("arr_sam_mode", SAM_MODES[1 if seeds else 0])
    dpg.set_value("arr_sam_max", int(cfg["max_objects"]))
    dpg.set_value("arr_sam_gran", str(cfg["granularity"]))
    dpg.set_value("arr_sam_reseed", float(cfg["reseed"]))
    dpg.set_value("arr_sam_keep", SAM_KEEPS[2 if cfg.get("drop") == [1] else 1 if cfg.get("keep") == "all" else 0])
    dpg.set_value("arr_sam_feather", int(cfg.get("feather", 2)))
    dpg.set_value("arr_sam_tool", SAM_TOOLS[0])
    dpg.set_value("arr_sam_clip", f"{Path(clip).name}: the frame its block starts on is {start:g} s into it")
    dpg.configure_item("arr_sam_win", label=f"SAM2 cutout - {Path(clip).name}", show=True)
    _sam_objects()
    on_sam_mode()
    _sam_frame(dpg.get_value("arr_sam_at"))
    dpg.focus_item("arr_sam_win")


def _sam_frame(t: float) -> None:
    """The clip's frame at ``t`` s in the window's picture (its shape kept)."""
    import cv2
    img = None
    cap = cv2.VideoCapture(A.sam["clip"])
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(t)) * 1000.0)
        ok, img = cap.read()
        img = img if ok else None
    cap.release()
    if img is not None:
        h, w = img.shape[:2]
        k = SAM_VIEW / max(h, w)
        A.sam["rect"] = ((SAM_VIEW - w * k) / 2, (SAM_VIEW - h * k) / 2, w * k, h * k)
    dpg.set_value("arr_sam_tex", _thumb_rgba(img, SAM_VIEW))
    _sam_draw()


def _sam_seeding() -> bool:
    return dpg.get_value("arr_sam_mode") == SAM_MODES[1]


def _sam_to_view(u: float, v: float) -> list:
    x0, y0, w, h = A.sam["rect"]
    return [x0 + u * w, y0 + v * h]


def _sam_uv():
    """The mouse on the picture, 0..1 of the frame (clamped)."""
    mx, my = dpg.get_mouse_pos(local=False)
    rx, ry = dpg.get_item_rect_min("arr_sam_canvas")
    x0, y0, w, h = A.sam["rect"]
    return (min(1.0, max(0.0, (mx - rx - x0) / max(1.0, w))), min(1.0, max(0.0, (my - ry - y0) / max(1.0, h))))


def _sam_draw() -> None:
    """The picture and, seeding, each object's box and points in its colour (the chosen
    one's thicker); a box being drawn in white."""
    from core.segment.mask_track import id_colour
    if not dpg.does_item_exist("arr_sam_canvas"):
        return
    dpg.delete_item("arr_sam_canvas", children_only=True)
    dpg.draw_image("arr_sam_tex", [0, 0], [SAM_VIEW, SAM_VIEW], parent="arr_sam_canvas")
    if not _sam_seeding():
        dpg.draw_text([8, 8], "SAM2 finds the objects by itself on each shot", size=13,
                      color=(255, 255, 255, 220), parent="arr_sam_canvas")
        return
    for k, o in enumerate(A.sam["objects"]):
        r, g, b = id_colour(k + 1)
        mine = k == A.sam["obj"]
        if o.get("box"):
            a, z = _sam_to_view(*o["box"][:2]), _sam_to_view(*o["box"][2:])
            dpg.draw_rectangle(a, z, color=(r, g, b, 255), thickness=3 if mine else 1, parent="arr_sam_canvas")
            dpg.draw_text([min(a[0], z[0]) + 3, min(a[1], z[1]) + 2], str(k + 1), size=14,
                          color=(r, g, b, 255), parent="arr_sam_canvas")
        for x, y, keep in o.get("points") or []:
            c = _sam_to_view(x, y)
            dpg.draw_circle(c, 6 if mine else 4, fill=(60, 220, 90, 255) if keep else (230, 60, 60, 255),
                            color=(r, g, b, 255), thickness=2, parent="arr_sam_canvas")
    drag = A.sam.get("drag")
    if drag is not None:
        dpg.draw_rectangle(_sam_to_view(*drag[0]), _sam_to_view(*drag[1]), color=(255, 255, 255, 255),
                           thickness=1, parent="arr_sam_canvas")


def _sam_objects() -> None:
    names = [f"object {k + 1}" for k in range(len(A.sam["objects"]))]
    dpg.configure_item("arr_sam_obj", items=names)
    dpg.set_value("arr_sam_obj", names[A.sam["obj"]])


def on_sam_mode(*_) -> None:
    seeding = _sam_seeding()
    if seeding and dpg.get_value("arr_sam_keep") == SAM_KEEPS[0]:   # the objects asked for: all of them
        dpg.set_value("arr_sam_keep", SAM_KEEPS[1])
    dpg.configure_item("arr_sam_auto", show=not seeding)
    dpg.configure_item("arr_sam_seeds", show=seeding)
    _sam_cache_text()
    _sam_draw()


def on_sam_at(sender, value) -> None:
    _sam_frame(value)
    _sam_cache_text()


def on_sam_obj(sender, value) -> None:
    A.sam["obj"] = max(0, int(str(value).split()[-1]) - 1)
    _sam_draw()


def sam_new_object() -> None:
    A.sam["objects"].append({"box": None, "points": []})
    A.sam["obj"] = len(A.sam["objects"]) - 1
    _sam_objects()
    _sam_draw()


def sam_clear(every: bool = False) -> None:
    if every:
        A.sam["objects"], A.sam["obj"] = [{"box": None, "points": []}], 0
    else:
        A.sam["objects"][A.sam["obj"]] = {"box": None, "points": []}
    _sam_objects()
    _sam_draw()
    _sam_cache_text()


def sam_press(u: float, v: float, button: int = 0) -> None:
    """On the picture at (u, v) (0..1 of the frame): the left button starts a box, or puts a
    point (+ keep, - leave out); the right one takes away the chosen object's nearest seed."""
    if not _sam_seeding():
        return
    o = A.sam["objects"][A.sam["obj"]]
    tool = dpg.get_value("arr_sam_tool")
    if button == 1:
        near = [(abs(x - u) + abs(y - v), i) for i, (x, y, _) in enumerate(o["points"])]
        near = [n for n in near if n[0] <= 2 * SAM_NEAR]
        if near:
            del o["points"][min(near)[1]]
        elif o.get("box") and o["box"][0] <= u <= o["box"][2] and o["box"][1] <= v <= o["box"][3]:
            o["box"] = None
    elif tool == SAM_TOOLS[0]:
        A.sam["drag"] = ((u, v), (u, v))
    else:
        o["points"].append([round(u, 4), round(v, 4), 1 if tool == SAM_TOOLS[1] else 0])
    _sam_draw()
    _sam_cache_text()


def sam_drag(u: float, v: float, done: bool = False) -> None:
    """The box being drawn reaches (u, v); ``done``: the mouse let go (a box too small to be
    one is none)."""
    if A.sam is None or A.sam.get("drag") is None:
        return
    a = A.sam["drag"][0]
    A.sam["drag"] = (a, (u, v))
    if done:
        A.sam["drag"] = None
        if abs(u - a[0]) > 0.01 and abs(v - a[1]) > 0.01:
            A.sam["objects"][A.sam["obj"]]["box"] = [round(min(a[0], u), 4), round(min(a[1], v), 4),
                                                     round(max(a[0], u), 4), round(max(a[1], v), 4)]
        _sam_cache_text()
    _sam_draw()


def on_sam_click(sender, app_data) -> None:
    if A.sam is not None and isinstance(app_data, (list, tuple)):
        sam_press(*_sam_uv(), button=int(app_data[0]))


def _sam_tick() -> None:
    """A box drawn with the mouse: it follows it until it lets go."""
    if A.sam is not None and A.sam.get("drag") is not None:
        sam_drag(*_sam_uv(), done=not dpg.is_mouse_button_down(dpg.mvMouseButton_Left))


def _sam_spec():
    """The FX as the window has it now, or None (seeding, nothing drawn yet)."""
    spec = {"source": SAM, **A.sam["kept"]}
    if _sam_seeding():
        objects = [{k: v for k, v in (("box", o.get("box")), ("points", o.get("points"))) if v}
                   for o in A.sam["objects"] if o.get("box") or o.get("points")]
        if not objects:
            return None
        spec["seeds"] = {"at": round(float(dpg.get_value("arr_sam_at")), 3), "objects": objects}
    else:
        spec.update(seeds="auto", max_objects=int(dpg.get_value("arr_sam_max")),
                    granularity=str(dpg.get_value("arr_sam_gran")),
                    reseed=round(float(dpg.get_value("arr_sam_reseed")), 2))
    keep = dpg.get_value("arr_sam_keep")
    if keep == SAM_KEEPS[2]:
        spec["drop"] = [1]
    else:
        spec["keep"] = "all" if keep == SAM_KEEPS[1] else "figures"
    spec["feather"] = int(dpg.get_value("arr_sam_feather"))
    return spec


def _sam_cache_text() -> None:
    from core.video import sam_cutout
    spec = _sam_spec() if A.sam is not None else None
    if spec is None:
        text = "draw a box (or put a point) on what to cut out"
    else:
        done = sam_cutout.cached(A.sam["clip"], sam_cutout.config(spec))
        text = (f"cut out already: {done} (the renders read it)" if done else
                "not cut out yet: the next preview or render runs SAM2 on the frames its blocks show (GPU), "
                "then they are kept in render_output/footage; more are cut only if more of the clip is used")
    dpg.set_value("arr_sam_cache", text)


def sam_close(ok: bool) -> None:
    """OK: the FX as set here; Cancel: as it was (one just added goes again)."""
    if A.sam is None:
        return
    uid, path = A.sam["uid"], A.sam["path"]
    if ok:
        spec = _sam_spec()
        if spec is None:
            _status("draw a box (or put a point) on what to cut out first - or choose auto")
            return
        _edit(uid, path, lambda _, *, spec=spec: spec)
        _status("SAM2 cutout set: the next preview or render cuts the clip out (once; then it is kept)")
    elif A.sam["new"]:
        j = path[-1]
        _edit(uid, path[:-1], lambda ls, *, j=j: [x for i, x in enumerate(ls or []) if i != j])
        _status("SAM2 cutout not added")
    A.sam = None
    dpg.configure_item("arr_sam_win", show=False)
    _draw()
    _draw_inspector()


def _draw_catalog(kind: str) -> None:
    """One tab of Generators or FX: a header per category (open while searching), an item
    per entry, its description on hover."""
    from core import catalog
    tag = f"arr_cat_{kind}"
    if not dpg.does_item_exist(tag):
        return
    dpg.delete_item(tag, children_only=True)
    query = dpg.get_value(f"arr_cat_q_{kind}") or ""
    groups = catalog.tree(kind, query)
    for cat, entries in groups:
        with dpg.collapsing_header(label=f"{cat} ({len(entries)})", parent=tag, default_open=bool(query)):
            for e in entries:
                flags = "".join([" *" if e.takes_input else "", " (masks)" if "masks" in e.needs else "",
                                 " (cuts)" if "cuts" in e.needs else ""])
                item = dpg.add_selectable(label=e.label + flags, user_data=e.key, callback=_on_pick_entry,
                                          default_value=e.key == A.entry)
                with dpg.tooltip(item):
                    dpg.add_text(e.about, wrap=360)
    if not groups:
        dpg.add_text("nothing matches", parent=tag, color=(150, 150, 160))


def _on_pick_entry(sender, value, key) -> None:
    from core import catalog
    A.entry = key
    for kind in (catalog.GENERATOR, catalog.FX):           # one chosen at a time
        tag = f"arr_cat_{kind}"
        for header in dpg.get_item_children(tag, 1) or []:
            for item in dpg.get_item_children(header, 1) or []:
                if dpg.get_item_type(item).endswith("Selectable") and item != sender:
                    dpg.set_value(item, False)
    e = catalog.by_key(key)
    if e.kind == catalog.GENERATOR:                         # a Generator is put in a lane by dragging
        A.content = ("gen", key)
        _show_content()
    else:
        _status(f"{e.label}: drag across an FX track (a lane's, the master's) to put it there")
    _show_entry(e)


def _show_entry(e) -> None:
    """The chosen entry: its demo playing, what it is, what it needs, its parameters (the
    numbers the music can move are marked ~)."""
    from core import catalog
    tag = f"arr_cat_info_{e.kind}"
    dpg.delete_item(tag, children_only=True)
    _show_demo(e, tag)
    kind = "Generator" if e.kind == catalog.GENERATOR else "FX"
    dpg.add_text(f"{e.label}  ·  {kind} · {e.category}", parent=tag, color=(230, 210, 140), wrap=300)
    dpg.add_text(e.about, parent=tag, wrap=300)
    notes = []
    if e.takes_input:
        notes.append("* can also take the picture below as its input")
    if "masks" in e.needs:
        notes.append("needs a mask track (SAM2)")
    if "cuts" in e.needs:
        notes.append("works at the clip montage's cuts")
    for n in notes:
        dpg.add_text(n, parent=tag, color=(200, 160, 100), wrap=300)
    if e.file:
        dpg.add_text(e.file, parent=tag, color=(120, 120, 140), wrap=300)
    if e.kind == catalog.GENERATOR:
        dpg.add_text("drag across the BG or the FG to put it there", parent=tag, color=(150, 230, 160), wrap=300)
    else:
        dpg.add_button(label="Add to the master", parent=tag, callback=lambda: add_master(e.key))
        dpg.add_text("or drag across an FX track (a lane's, the master's); or right-click a block: its own",
                     parent=tag, color=(120, 120, 140), wrap=300)
    dpg.add_text(f"parameters ({len(e.params)}; ~ the music can move it):", parent=tag, color=(150, 210, 240))
    for prm in e.params:
        value = "" if prm.default is None else f" = {prm.default}"
        line = dpg.add_text(f"{'~' if prm.knob else ' '} {prm.name}{value}", parent=tag)
        if prm.about:
            with dpg.tooltip(line):
                dpg.add_text(prm.about, wrap=360)


# ----------------------------------------------------------------------- demos

def _show_demo(e, parent) -> None:
    """The entry's demo (core/demos.py) playing in a loop; none yet, a button to make it."""
    from core import demos
    making = A.demo_making.get(e.key)
    state = "rendering" if making is not None and making.poll() is None else demos.status(e)
    video = demos.video_path(e.key)
    if video.is_file() and state in ("ready", "stale", "rendering"):
        img = dpg.add_image("arr_demo_tex", width=DEMO_SHOWN, height=DEMO_SHOWN, parent=parent)
        if not dpg.does_item_exist("arr_demo_click"):
            with dpg.item_handler_registry(tag="arr_demo_click"):
                dpg.add_item_clicked_handler(callback=lambda: _demo_window())
        dpg.bind_item_handler_registry(img, "arr_demo_click")
        _play_demo(e.key, video)
    else:
        _play_demo(None, None)
    if state == "rendering":
        dpg.add_text("making its demo...", parent=parent, color=(200, 200, 140))
    elif state == "stale":
        with dpg.group(horizontal=True, parent=parent):
            dpg.add_text("the layer changed since this demo", color=(200, 160, 100))
            dpg.add_button(label="Make it again", callback=lambda: _make_demo(e))
    elif state in ("missing", "failed"):
        info = demos.read_info(e.key) or {}
        dpg.add_text("no demo yet" if state == "missing" else f"its demo failed: {info.get('error', '')}",
                     parent=parent, color=(200, 160, 100) if state == "missing" else (240, 120, 110), wrap=300)
        dpg.add_button(label="Make demo", parent=parent, callback=lambda: _make_demo(e))
    elif video.is_file():
        dpg.add_text("click it: bigger", parent=parent, color=(120, 120, 140))


def _make_demo(e) -> None:
    """Its demo, made in the background (a GPU for shaders); the panel shows it when done."""
    from core import demos
    if A.demo_making.get(e.key) is not None and A.demo_making[e.key].poll() is None:
        return
    root = Path(demos.__file__).resolve().parent.parent
    A.demo_making[e.key] = subprocess.Popen([sys.executable, "-m", "core.demos", "render", "--only", e.key,
                                             "--force"], cwd=str(root), stdout=subprocess.DEVNULL,
                                            stderr=subprocess.DEVNULL)
    _status(f"{e.label}: making its demo (a few seconds; minutes for the 3D fractals)")
    _show_entry(e)


def _watch_demos() -> None:
    """A demo being made is done: the panel shows it (when its entry is still chosen)."""
    for key, proc in list(A.demo_making.items()):
        if proc is None or proc.poll() is None:
            continue
        del A.demo_making[key]
        from core import catalog
        e = catalog.by_key(key)
        if e is not None and A.entry == key:
            _show_entry(e)


def _play_demo(key, video) -> None:
    """Play ``video`` into the demo's texture, looping, until another is asked (None: stop)."""
    if A.demo_key == key:
        return
    A.demo_key = key
    if key is None:
        return

    def run(mine=key):
        import cv2
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        rgba = np.ones((DEMO, DEMO, 4), np.float32)
        nxt = time.time()
        while A.demo_key == mine:
            ok, img = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, img = cap.read()
                if not ok:
                    break
            if img.shape[0] != DEMO or img.shape[1] != DEMO:
                img = cv2.resize(img, (DEMO, DEMO), interpolation=cv2.INTER_AREA)
            rgba[..., :3] = img[..., ::-1] / 255.0
            A.demo_frame = rgba.ravel().copy()
            nxt += 1.0 / fps
            time.sleep(max(0.0, nxt - time.time()))
        cap.release()
    threading.Thread(target=run, daemon=True).start()


def _demo_window() -> None:
    """The demo playing at its own size."""
    from core import catalog
    e = catalog.by_key(A.demo_key) if A.demo_key else None
    if e is None:
        return
    if dpg.does_item_exist("arr_demo_win"):
        dpg.delete_item("arr_demo_win")
    with dpg.window(tag="arr_demo_win", label=f"demo - {e.label}", width=DEMO + 20, height=DEMO + 40,
                    pos=(360, 120)):
        dpg.add_image("arr_demo_tex", width=DEMO, height=DEMO)


def _thumb_rgba(img, size: int = THUMB) -> np.ndarray:
    """A BGR frame (or an image file) in a square ``size`` a side (the browser's picture),
    its shape kept: RGBA floats."""
    import cv2
    if img is not None and not isinstance(img, np.ndarray):
        img = cv2.imread(str(img), cv2.IMREAD_COLOR)
    canvas = np.zeros((size, size, 4), np.float32)
    canvas[..., 3] = 1.0
    if img is not None:
        h, w = img.shape[:2]
        s = size / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        h, w = img.shape[:2]
        y0, x0 = (size - h) // 2, (size - w) // 2
        canvas[y0:y0 + h, x0:x0 + w, :3] = img[..., ::-1] / 255.0
    return canvas.ravel()


# ----------------------------------------------------------------------- the clip's player

CLIP_SR = 44100        # the clip's sound, as the player mixes it


def _play_clip(path, start: float = 0.0) -> None:
    """``path`` plays in the browser's picture from ``start`` (s) to its end, again and again
    (an image: shown); None stops it. Its sound follows when it is on."""
    key = (str(path), round(float(start), 2)) if path else None
    if A.clip_play == key:
        return
    A.clip_play = key
    _sound_off()
    if key is None:
        return

    def run(mine=key):
        import cv2
        if Path(path).suffix.lower() in arr.IMAGES:
            A.clip_frame = _thumb_rgba(path)
            return
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():                           # a codec OpenCV lacks: a still, by ffmpeg
            try:
                png = arr.thumbnail(path, start, THUMB)
            except Exception:                            # none to show
                png = None
            if png and A.clip_play == mine:
                A.clip_frame = _thumb_rgba(png)
            return
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        while A.clip_play == mine:                       # a loop: from "from" to the end
            cap.set(cv2.CAP_PROP_POS_MSEC, start * 1000.0)
            t0, n = time.time(), 0
            A.clip_pos = start
            if A.clip_sound:
                _sound_at(path, start)
            while A.clip_play == mine:
                ok, img = cap.read()
                if not ok:
                    break
                A.clip_frame = _thumb_rgba(img)
                n += 1
                A.clip_pos = start + n / fps
                time.sleep(max(0.0, t0 + n / fps - time.time()))
            if n == 0:                                   # nothing after "from"
                break
        cap.release()
    threading.Thread(target=run, daemon=True).start()


def _clip_audio(path: str):
    """The clip's sound, mono at CLIP_SR (None: it has none), read once by ffmpeg."""
    if path not in A.clip_audio:
        try:
            out = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
                                  "-ar", str(CLIP_SR), "-f", "f32le", "-"], capture_output=True, timeout=60)
            audio = np.frombuffer(out.stdout, np.float32).copy()
        except (OSError, subprocess.SubprocessError):
            audio = np.zeros(0, np.float32)
        if len(A.clip_audio) > 8:
            A.clip_audio.pop(next(iter(A.clip_audio)))
        A.clip_audio[path] = audio if len(audio) else None
    return A.clip_audio[path]


def _sound_at(path: str, t: float) -> None:
    """The clip's sound from ``t`` (s): with its picture."""
    audio = _clip_audio(path)
    if audio is None:
        A.sound_note = f"{Path(path).name} has no sound"
        return
    if A.clip_play is None or A.clip_play[0] != str(path) or not A.clip_sound:
        return                                           # another clip, or muted, while it was read
    if A.clip_mixer is None:
        from core.audio_player import AudioPlayer
        A.clip_mixer = AudioPlayer(CLIP_SR)
    A.clip_mixer.remove_track("clip")
    A.clip_mixer.add_track("clip", audio)
    A.clip_mixer.play(start_t=t)


def _sound_off() -> None:
    if A.clip_mixer is not None:
        A.clip_mixer.pause()


def on_clip_sound(*_) -> None:
    """The clip's sound on or off (it plays muted until turned on)."""
    set_clip_sound(not A.clip_sound)


def set_clip_sound(on: bool) -> None:
    A.clip_sound = bool(on)
    if dpg.does_item_exist("arr_clip_sound"):
        dpg.configure_item("arr_clip_sound", label="sound: on" if on else "sound: off")
    if not on:
        _sound_off()
    elif A.clip_play is not None:
        path = A.clip_play[0]
        threading.Thread(target=lambda: _sound_at(path, A.clip_pos), daemon=True).start()


# ----------------------------------------------------------------------- a clip dragged in

def _over_lanes(pos=None) -> bool:
    """``pos`` (in the viewport; None: the mouse) is over the lanes' pane, by its rectangle:
    while a drag from the browser goes on, the item it started on stays active, and the
    lanes do not count as hovered. (A child window has no rect_min: the pane's top left is
    the canvas', scrolled.)"""
    if not dpg.does_item_exist("arr_tracks") or not dpg.is_item_shown("arr_win"):
        return False
    mx, my = pos if pos is not None else dpg.get_mouse_pos(local=False)
    try:
        x0, y0 = dpg.get_item_rect_min("arr_canvas")
        w, h = dpg.get_item_rect_size("arr_tracks")
    except (KeyError, SystemError):                      # not laid out yet
        return False
    y0 += _scroll()
    return x0 <= mx < x0 + w and y0 <= my < y0 + h


def _canvas_of(pos) -> tuple:
    """A point in the viewport, on the canvas."""
    rx, ry = dpg.get_item_rect_min("arr_canvas")
    return pos[0] - rx, pos[1] - ry


def drop_files(paths: list, pos) -> None:
    """Files dropped from the Explorer at ``pos`` (in the viewport): on a lane, the clips go
    there one after another from the bar under the mouse (as on a drag from the list); on
    the browser, a folder is listed and a clip is chosen (it plays)."""
    if not dpg.does_item_exist("arr_win") or not dpg.is_item_shown("arr_win"):
        return
    clips = [p for p in paths if Path(p).suffix.lower() in arr.VIDEOS + arr.IMAGES and Path(p).is_file()]
    folders = [p for p in paths if Path(p).is_dir()]
    if _over_lanes(pos):
        if not clips:
            _status("only video or image files go on the lanes")
            return
        got = _clip_target(*_canvas_of(pos), 0.0)
        if got is None:
            _status("a clip goes on the BG, the FG or CLIPS (the montage)")
            return
        _place_clips(got[1], got[2], [(path, 0.0, arr.duration(path)) for path in clips],
                     sub=A.rows[got[0]].get("sub"))
        return
    if folders:
        dpg.set_value("arr_folder", folders[0])
        btn_list()
    elif clips:
        _choose_clip(clips[0], 0.0)
        _status(f"{Path(clips[0]).name} chosen - drag it (the picture) onto the BG, the FG or CLIPS")
    else:
        _status("drop video or image files (or a folder of them)")


def _clip_bars(seconds: float) -> int:
    """How many bars a clip dropped in takes: as long as it plays (an image: 4)."""
    if seconds <= 0:
        return 4
    return max(1, int(round(seconds / (_bar_seconds() or 2.0))))


def _clip_target(x: float, y: float, seconds: float):
    """(row, lane, first, last) a clip dropped at (x, y) goes to: the BG, the FG or the
    montage (CLIPS), from the bar under the mouse; None elsewhere."""
    r = _row_at(y)
    if r is None or x < LABEL_W:
        return None
    row = A.rows[r]
    key = "montage" if row["kind"] == "montage" else row["lane"]
    if key not in ("montage", arr.BG, arr.FG):
        return None
    first = _bar_of_x(x)
    return r, key, first, min(_bars(), first + _clip_bars(seconds) - 1)


def _show_clip_ghost() -> None:
    """Where the dragged clip would go, while it is dragged over the lanes."""
    if not dpg.does_item_exist("arr_ghost"):
        return
    got = _clip_target(*_mouse_in_canvas(), A.clip_drag[2]) if _over_lanes() else None
    if got is None:
        dpg.configure_item("arr_ghost", show=False)
        return
    r, key, first, last = got
    dpg.configure_item("arr_ghost", pmin=[_x_of(first), _row_y(r) + 2],
                       pmax=[_x_of(last + 1), _row_y(r) + A.lane_h - 4], show=True)


def drop_clip(x: float, y: float, drag) -> None:
    """A clip dragged from the browser comes down at (x, y): a block in the BG or the FG from
    the bar there, as long as it plays (a long one: as many bars as asked); on CLIPS, pinned
    to the montage there."""
    got = _clip_target(x, y, drag[2])
    if got is None:
        _status("a clip goes on the BG, the FG or CLIPS (the montage)")
        return
    _place_clips(got[1], got[2], [tuple(drag)], sub=A.rows[got[0]].get("sub"))


LONG_CLIP = 30.0       # s: a clip longer asks how many bars it takes (or all of it)


def _place_clips(key: str, first: int, todo: list, bars=None, sub=None) -> None:
    """Clips [(path, start s, seconds)] put on ``key`` one after another from bar ``first``,
    each as long as it plays; ``bars``: the first one's (asked). One longer than LONG_CLIP
    asks first how many bars it takes; the ones after it wait for the answer."""
    todo = list(todo)
    while todo:
        path, start, seconds = todo[0]
        if bars is None and seconds > LONG_CLIP:
            _ask_bars(key, first, todo, sub)
            return
        last = min(_bars(), first + (bars or _clip_bars(seconds)) - 1)
        A.content = ("clip", path, start)
        _show_content()
        if key == "montage":
            _put_in_montage(first, last)
        else:
            _new_member(key, first, last, sub=sub)
        todo, bars = todo[1:], None
        if last >= _bars():                              # the song's end
            break
        first = last + 1


def _ask_bars(key: str, first: int, todo: list, sub=None) -> None:
    """How many bars the long clip ``todo[0]`` takes (from ``first``), or all of it."""
    path, start, seconds = todo[0]
    whole = min(_clip_bars(seconds), _bars() - first + 1)
    A.asking = (key, first, todo, sub)
    where = {"montage": "over the montage (CLIPS)", arr.BG: "BG", arr.FG: "FG"}.get(key, key)
    dpg.set_value("arr_ask_text", f"{Path(path).name} plays {seconds:.0f} s"
                  + (f" from {start:g} s" if start else "") + f": {_clip_bars(seconds)} bars. "
                  f"How many bars should it take in the {where}, from bar {first}?")
    dpg.set_value("arr_ask_n", min(8, whole))
    dpg.configure_item("arr_ask_whole", label=f"The whole video ({whole} bars"
                       + (", to the song's end)" if whole < _clip_bars(seconds) else ")"))
    try:
        vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        dpg.configure_item("arr_ask_win", pos=(max(0, vw // 2 - 230), max(0, vh // 2 - 90)))
    except Exception:                                    # no viewport (the tests)
        pass
    dpg.configure_item("arr_ask_win", show=True)
    dpg.focus_item("arr_ask_n")


def on_ask_n(sender, value) -> None:
    """The bars field: Enter puts the clip. (ImGui calls this for its - and + buttons too,
    on_enter or not: those only change the number.)"""
    if _down("mvKey_Return", "mvKey_NumPadEnter"):
        answer_bars(value)


def answer_bars(n) -> None:
    """The answer: a number of bars, "whole" (all the clip) or None (it is not put)."""
    asking, A.asking = A.asking, None
    if dpg.does_item_exist("arr_ask_win"):
        dpg.configure_item("arr_ask_win", show=False)
    if asking is None:
        return
    key, first, todo, sub = asking
    if n is None:
        _status(f"{Path(todo[0][0]).name} not put" + (f" (nor the {len(todo) - 1} after it)" if len(todo) > 1 else ""))
        return
    _place_clips(key, first, todo, _clip_bars(todo[0][2]) if n == "whole" else max(1, int(n)), sub)


def tick() -> None:
    """Once a frame, on the main thread: a thumbnail asked for (after the slider rests a
    moment) is made in the background and shown when ready; redraws asked by workers; how
    a preview goes."""
    _watch_preview()
    _watch_demos()
    _show_covers()
    _show_edge()
    _show_saved()
    _sam_tick()
    _render_tick()
    for paths, pos in arc_window.take():
        drop_files(paths, pos)
    arc_window.picked()                                  # folders and files chosen in a dialog
    if A.demo_frame is not None:
        frame, A.demo_frame = A.demo_frame, None
        dpg.set_value("arr_demo_tex", frame)
        if A.demo_key is not None and not dpg.is_item_shown("arr_win"):
            A.demo_key = None                                # the window closed: the player stops
    _fit_canvas()
    _pin_ruler()
    if A.scroll_to is not None and dpg.does_item_exist("arr_tracks"):
        y, waited = A.scroll_to if isinstance(A.scroll_to, tuple) else (A.scroll_to, 0)
        top = dpg.get_y_scroll_max("arr_tracks")
        if y > top and waited < 3:                       # the canvas grows a frame or two later
            A.scroll_to = (y, waited + 1)
        else:
            A.scroll_to = None
            dpg.set_y_scroll("arr_tracks", float(min(y, top)))
    g = A.host.grid() if A.host is not None else None
    if g is not getattr(A, "grid_seen", None):          # the song loaded (again): its real bars
        A.grid_seen = g
        if A.doc is not None and dpg.does_item_exist("arr_win") and dpg.is_item_shown("arr_win"):
            _relane()
            set_view()
    if getattr(A, "redraw", False):
        A.redraw = False
        _draw()
    if A.pending is not None:                           # a worker's edit (SAM2's cutouts)
        (uid, fn), A.pending = A.pending, None
        _edit(uid, ("layers",), fn)
    if getattr(A, "relane", False):
        A.relane = False
        _relane()
    if A.clip_frame is not None:
        frame, A.clip_frame = A.clip_frame, None
        dpg.set_value("arr_tex", frame)
    if A.clip_restart and time.time() - A.clip_restart > 0.25 and A.clip:
        A.clip_restart = 0.0
        _play_clip(A.clip, A.start)
    if A.clip_play is not None and dpg.does_item_exist("arr_win") and not dpg.is_item_shown("arr_win"):
        _play_clip(None)                                 # the window closed: the clip stops
    note = getattr(A, "sound_note", "")
    if note:
        A.sound_note = ""
        _status(note)


# ----------------------------------------------------------------------- the window

SCROLLBAR_W = 18


def _fit_canvas() -> None:
    """The canvas as wide as the lanes' pane: widening the window widens the timeline (the
    bars shown spread over it), narrowing it never goes under CANVAS_W."""
    if not dpg.does_item_exist("arr_tracks") or not dpg.is_item_shown("arr_win"):
        return
    pane = dpg.get_item_rect_size("arr_tracks")[0]
    if pane <= 0:                                        # not laid out yet
        return
    w = max(CANVAS_W, int(pane) - SCROLLBAR_W)
    if abs(w - A.canvas_w) >= 2:
        A.canvas_w = w
        dpg.configure_item("arr_canvas", width=w)
        set_view()


def on_video_res(sender, value) -> None:
    """The final video's size and shape, chosen here: the studio's, so a preview shows it."""
    err = A.host.set_resolution(value)
    _status(err or f"final video: {value.strip()} - the next preview shows it; Save Project "
                   "(in the studio) keeps it")


def _show_video_res() -> None:
    dpg.configure_item("arr_video_res", items=A.host.resolutions())
    dpg.set_value("arr_video_res", A.host.resolution())


def open_window() -> None:
    if not getattr(A, "sized", False):          # first opened: as big as the studio's window
        A.sized = True
        try:
            vw, vh = dpg.get_viewport_client_width(), dpg.get_viewport_client_height()
        except Exception:                                # no viewport (the tests)
            vw = vh = 0
        if vw > 0 and vh > 0:
            dpg.configure_item("arr_win", width=max(1090, vw - 10), height=max(700, vh - 45))
    _load()
    set_view()                                  # the song may be another: the view fits it
    _show_video_res()                           # the project may be another: its size
    if not A.catalog_drawn:                     # the catalog is read once, when first shown
        A.catalog_drawn = True
        for kind in ("generator", "fx"):
            _draw_catalog(kind)
    if not dpg.get_value("arr_folder"):                 # the last folder listed, else the project's
        last = _remembered().get("clips_folder") or ""
        dpg.set_value("arr_folder", last if last and Path(last).is_dir() else (A.host.clips_dir() or ""))
    if dpg.get_value("arr_folder") and not A.clips:
        btn_list()
    dpg.show_item("arr_win")
    dpg.focus_item("arr_win")


def build(host) -> None:
    A.host = host
    A.catalog_drawn = False
    A.zoom, A.view0, A.lane_h, A.scroll_to, A.canvas_w = 1.0, 1, LANE_H, None, CANVAS_W
    with dpg.item_handler_registry(tag="arr_list_handler"):
        dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=on_list_press)
    with dpg.item_handler_registry(tag="arr_clip_img_handler"):
        dpg.add_item_clicked_handler(button=dpg.mvMouseButton_Left, callback=on_picture_press)
    A.clip_play, A.clip_drag, A.list_items, A.clip_frame = None, None, {}, None
    A.list_names, A.list_pics, A.covers, A.covers_ready = {}, {}, {}, []
    A.asking, A.saved_at, A.saved_flash = None, "", 0.0
    with dpg.window(tag="arr_ask_win", label="A long clip: how many bars?", modal=True, show=False,
                    width=460, height=125, no_resize=True, on_close=lambda: answer_bars(None)):
        dpg.add_text("", tag="arr_ask_text", wrap=440)
        with dpg.group(horizontal=True):
            dpg.add_text("bars")
            dpg.add_input_int(tag="arr_ask_n", width=110, default_value=8, min_value=1, min_clamped=True,
                              on_enter=True, callback=on_ask_n)
            dpg.add_button(label="Put it", tag="arr_ask_ok", width=80,
                           callback=lambda: answer_bars(dpg.get_value("arr_ask_n")))
        with dpg.group(horizontal=True):
            dpg.add_button(label="The whole video", tag="arr_ask_whole", callback=lambda: answer_bars("whole"))
            dpg.add_button(label="Cancel", tag="arr_ask_cancel", callback=lambda: answer_bars(None))
    with dpg.texture_registry():
        dpg.add_dynamic_texture(THUMB, THUMB, [0.0, 0.0, 0.0, 1.0] * THUMB * THUMB, tag="arr_tex")
        dpg.add_dynamic_texture(DEMO, DEMO, np.zeros(DEMO * DEMO * 4, np.float32), tag="arr_demo_tex")
        dpg.add_static_texture(COVER, COVER, [0.12, 0.12, 0.15, 1.0] * COVER * COVER, tag="arr_cover_blank")
    dpg.add_texture_registry(tag="arr_covers")          # the list's pictures, made as they come
    with dpg.texture_registry():
        dpg.add_dynamic_texture(SAM_VIEW, SAM_VIEW, np.zeros(SAM_VIEW * SAM_VIEW * 4, np.float32), tag="arr_sam_tex")
    A.sam = None
    with dpg.item_handler_registry(tag="arr_sam_handler"):
        dpg.add_item_clicked_handler(callback=on_sam_click)
    with dpg.window(tag="arr_sam_win", label="SAM2 cutout", show=False, width=SAM_VIEW + 440,
                    height=SAM_VIEW + 90, pos=(120, 70), on_close=lambda: sam_close(False)):
        with dpg.group(horizontal=True):
            with dpg.group():
                dpg.add_drawlist(SAM_VIEW, SAM_VIEW, tag="arr_sam_canvas")
                dpg.add_slider_float(tag="arr_sam_at", label="frame (s of the clip)", width=SAM_VIEW - 150,
                                     min_value=0.0, max_value=1.0, callback=on_sam_at)
            with dpg.group():
                dpg.add_text("", tag="arr_sam_clip", wrap=410, color=(200, 200, 215))
                dpg.add_radio_button(SAM_MODES, tag="arr_sam_mode", default_value=SAM_MODES[0], callback=on_sam_mode)
                with dpg.group(tag="arr_sam_auto"):
                    dpg.add_input_int(label="objects at most", tag="arr_sam_max", width=110, default_value=4,
                                      min_value=1, max_value=16, min_clamped=True, max_clamped=True,
                                      callback=lambda: _sam_cache_text())
                    dpg.add_combo(["whole", "parts"], label="whole characters, or their parts", tag="arr_sam_gran",
                                  width=110, default_value="whole", callback=lambda: _sam_cache_text())
                    dpg.add_input_float(label="looks again every (s)", tag="arr_sam_reseed", width=110,
                                        default_value=2.0, step=0.5, min_value=0.0, min_clamped=True, format="%.1f",
                                        callback=lambda: _sam_cache_text())
                    dpg.add_text("on each shot's first frame it finds the objects, then follows them; "
                                 "every few seconds it looks for new ones (0: only at cuts)",
                                 wrap=410, color=(120, 120, 140))
                with dpg.group(tag="arr_sam_seeds", show=False):
                    dpg.add_radio_button(SAM_TOOLS, tag="arr_sam_tool", default_value=SAM_TOOLS[0], horizontal=True)
                    with dpg.group(horizontal=True):
                        dpg.add_combo([], tag="arr_sam_obj", width=110, callback=on_sam_obj)
                        dpg.add_button(label="new object", callback=lambda: sam_new_object())
                        dpg.add_button(label="clear it", callback=lambda: sam_clear())
                        dpg.add_button(label="clear all", callback=lambda: sam_clear(True))
                    dpg.add_text("on the picture: drag a box around it, click points on it (+) or on what is "
                                 "not it (-); right-click takes the nearest away. Each object is followed "
                                 "from this frame to the end of what its blocks show.", wrap=410, color=(120, 120, 140))
                dpg.add_separator()
                dpg.add_radio_button(SAM_KEEPS, tag="arr_sam_keep", default_value=SAM_KEEPS[0],
                                     callback=lambda: _sam_cache_text())
                dpg.add_input_int(label="soft edge (px)", tag="arr_sam_feather", width=110, default_value=2,
                                  min_value=0, max_value=40, min_clamped=True, max_clamped=True)
                dpg.add_text("", tag="arr_sam_cache", wrap=410, color=(150, 190, 150))
                with dpg.group(horizontal=True):
                    dpg.add_button(label="OK", width=90, callback=lambda: sam_close(True))
                    dpg.add_button(label="Cancel", width=90, callback=lambda: sam_close(False))
    dpg.bind_item_handler_registry("arr_sam_canvas", "arr_sam_handler")
    dpg.add_file_dialog(show=False, directory_selector=True, callback=_pick_folder, tag="dlg_arr_folder",
                        width=620, height=420)
    dpg.add_file_dialog(show=False, directory_selector=True, callback=_picked_param, tag="dlg_arr_param_dir",
                        width=620, height=420, cancel_callback=lambda *_: setattr(A, "param_pick", None))
    with dpg.file_dialog(show=False, callback=_picked_param, tag="dlg_arr_param_file", width=620, height=420,
                         cancel_callback=lambda *_: setattr(A, "param_pick", None)):
        dpg.add_file_extension(".*")
    with dpg.window(label="Arrange - the background, the foreground and the master, bar by bar",
                    tag="arr_win", show=False, width=1090, height=900, pos=(5, 40)):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=330, height=-1), dpg.tab_bar():
                with dpg.tab(label="Clips"):
                    dpg.add_text("CLIPS - video or image files", color=(230, 210, 140))
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="arr_folder", width=260, hint="a folder of clips",
                                           on_enter=True, callback=lambda: btn_list())
                        dpg.add_button(label="...", callback=lambda: _choose_folder())
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="arr_filter", width=200, hint="filter, e.g. final.mp4",
                                           on_enter=True, callback=lambda: btn_list())
                        dpg.add_button(label="list", callback=btn_list)
                    dpg.add_child_window(tag="arr_list", height=330, border=True)
                    dpg.add_text("", tag="arr_clip_info", wrap=310, color=(150, 190, 150))
                    with dpg.group(horizontal=True):
                        dpg.add_image("arr_tex", width=THUMB, height=THUMB, tag="arr_clip_img")
                        with dpg.group():
                            dpg.add_button(label="sound: off", tag="arr_clip_sound", width=110, callback=on_clip_sound)
                            with dpg.tooltip("arr_clip_sound"):
                                dpg.add_text("the clip plays muted; this turns its sound on (a preview "
                                             "of the song turns it off)")
                            dpg.add_text("drag the clip\n(from the list\nor here) onto\nthe BG, the FG\n"
                                         "or CLIPS", color=(120, 120, 140))
                    dpg.bind_item_handler_registry("arr_clip_img", "arr_clip_img_handler")
                    dpg.add_slider_float(label="from (s)", tag="arr_from", width=230, min_value=0.0,
                                         max_value=1.0, callback=on_from)
                    dpg.add_text("a new block starts the clip here", color=(120, 120, 140))
                    dpg.add_separator()
                    dpg.add_text("or drag across the BG or the FG to put it over those bars (in the FG it "
                                 "is cut out by its mask track)", wrap=310, color=(150, 190, 150))
                    dpg.add_button(label="Cut it out (SAM2, at the render)", width=-1, callback=btn_segment)
                    dpg.add_text("its block in the FG gets the SAM2 cutout FX (its seeds: the window that "
                                 "opens); SAM2 cuts it out at the next preview or render, once, then it is "
                                 "kept (render_output/footage)", wrap=310, color=(120, 120, 140))
                for kind, title, about in (
                        ("generator", "Generators", "they make a picture: a fractal, cells, a nebula"),
                        ("fx", "FX", "they work on the picture they get")):
                    with dpg.tab(label=title):
                        dpg.add_text(about, color=(120, 120, 140), wrap=310)
                        dpg.add_input_text(tag=f"arr_cat_q_{kind}", width=-1, hint="search",
                                           callback=lambda s, a, *, k=kind: _draw_catalog(k))
                        dpg.add_child_window(tag=f"arr_cat_{kind}", height=420, border=False)
                        dpg.add_separator()
                        dpg.add_child_window(tag=f"arr_cat_info_{kind}", height=-1, border=False)
            with dpg.child_window(width=-1, height=-1):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save to scene", tag="arr_save_btn", width=130, callback=btn_save)
                    dpg.add_button(label="Reload", width=80, callback=lambda: _load())
                    dpg.add_button(label="Undo", width=60, callback=btn_undo)
                    dpg.add_button(label="Redo", width=60, callback=btn_redo)
                    dpg.add_button(label="Preview (choose bars)", tag="arr_prev_btn", callback=preview_span)
                    dpg.add_button(label="Render video...", tag="arr_render_btn", callback=lambda: open_render())
                    with dpg.tooltip("arr_render_btn"):
                        dpg.add_text("the final video: the whole song or some bars (saved first)")
                    dpg.add_text("", tag="arr_save_state")
                with dpg.group(horizontal=True):
                    dpg.add_text("final video:")
                    dpg.add_combo([], tag="arr_video_res", width=200, callback=on_video_res)
                    dpg.add_text("its size and shape (Save Project in the studio keeps it)", color=(120, 120, 140))
                    with dpg.tooltip("arr_video_res"):
                        dpg.add_text("an own size: the studio's field next to PREVIEW")
                dpg.add_text("", tag="arr_status", color=(160, 200, 160), wrap=0)   # wraps: never off the edge
                dpg.add_text("pick a clip or a Generator to put it in a lane", tag="arr_content",
                             color=(150, 230, 160))
                with dpg.group(horizontal=True):
                    dpg.add_text("zoom")
                    dpg.add_button(label="-", width=24, callback=lambda: set_view(zoom=A.zoom / 1.5))
                    dpg.add_button(label="+", width=24, callback=lambda: set_view(zoom=A.zoom * 1.5))
                    dpg.add_button(label="all", width=34, callback=lambda: set_view(zoom=1.0, first=1))
                    dpg.add_slider_int(tag="arr_view0", width=260, min_value=1, max_value=1, default_value=1,
                                       format="from bar %d", callback=lambda s, v: set_view(first=v))
                    dpg.add_text("", tag="arr_zoom_text", color=(120, 120, 140))
                    dpg.add_text("   rows")
                    dpg.add_button(label="-", width=24, callback=lambda: set_lane_height(A.lane_h / 1.2))
                    dpg.add_button(label="+", width=24, callback=lambda: set_lane_height(A.lane_h * 1.2))
                    dpg.add_text(f"{LANE_H} px", tag="arr_rows_text", color=(120, 120, 140))
                dpg.add_text("drag across the bar numbers to choose bars (Space: preview them) · wheel "
                             "scrolls up and down, Ctrl+wheel zooms along the song, Shift+wheel makes the rows "
                             "taller or lower, Alt+wheel moves along · Shift+drag a block's end: resize it · under the bars: how "
                             "busy the MIDI is",
                             color=(120, 120, 140), wrap=0)
                # they scroll (the wheel, the scrollbar: on_wheel's, not ImGui's) and the inspector stays
                with dpg.child_window(height=-250, border=False, tag="arr_tracks", no_scroll_with_mouse=True):
                    dpg.add_drawlist(width=A.canvas_w, height=200, tag="arr_canvas")
                with dpg.tooltip("arr_tracks", tag="arr_tip", show=False):   # (a drawlist takes none)
                    dpg.add_text("", tag="arr_tip_text")
                with dpg.item_handler_registry(tag="arr_canvas_handler"):
                    dpg.add_item_clicked_handler(callback=on_canvas_click)
                    dpg.add_item_double_clicked_handler(callback=on_canvas_double_click)
                dpg.bind_item_handler_registry("arr_canvas", "arr_canvas_handler")
                dpg.add_separator()
                with dpg.child_window(height=-1, border=False, tag="arr_insp_win"):
                    dpg.add_group(tag="arr_inspector")
    with dpg.handler_registry():
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=on_mouse_release)
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, threshold=2, callback=on_mouse_drag)
        dpg.add_key_press_handler(dpg.mvKey_Z, callback=on_key_z)
        dpg.add_mouse_wheel_handler(callback=on_wheel)
        dpg.add_mouse_move_handler(callback=on_mouse_move)
        dpg.add_key_press_handler(dpg.mvKey_Y, callback=on_key_y)
        dpg.add_key_press_handler(dpg.mvKey_Delete, callback=on_key_delete)
        dpg.add_key_press_handler(dpg.mvKey_Spacebar, callback=on_key_space)

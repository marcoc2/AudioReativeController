"""ARC Studio's Arrange window: pick the exact clips, and when the background and the
characters come in, bar by bar.

Left, a clip browser: a folder, a filter ("final.mp4"), the clip's frame at the point it
would start from; beside it the Generators (they make a picture: a fractal, cells) and the
FX (they work on the picture they get), by category (core/catalog.py). Right, the tracks over the song's bars, one per layer of the scene
(core/arrangement.py), the top of the picture on top: drag across a track to put the chosen
clip there (a background track takes any clip; a character track plays its own) or to give
a Generator or FX its bars; click a block to edit or delete it. A right click inserts a
Generator or an FX (above the track under the mouse, or at the Master: over everything).
Lanes cued from REAPER show their cues in grey (edited in REAPER). Save writes the blocks
into the scene through SceneDoc, comments and anchors kept. Ctrl+Z undoes an edit, Ctrl+Y
(or Ctrl+Shift+Z) redoes it; the history lives until the window reloads the scene from disk.

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

from core import arrangement as arr

LABEL_W = 150          # the lane names
CANVAS_W = 720
RULER_H = 30
DENSITY_H = 10
LANE_H = 34
THUMB = 192
SAM2_DEFAULTS = ("F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/models/sam2",)
UNDO_MAX = 200         # edits remembered


class _A:
    host = None
    doc = None                 # SceneDoc of the scene being arranged
    items: list = []           # [node or None, base or None, spec] per layer, in stack order
    lanes: list = []           # core.arrangement.Lane, rebuilt from items
    clips: list = []           # the browser's clips (Path)
    clip = None                # the chosen clip (str)
    clip_len = 0.0
    start = 0.0                # where in it a new block starts (s)
    sel = None                 # (lane k, block k or None)
    drag = None                # (lane k, first bar) while dragging
    want_thumb = None          # (path, t, asked at)
    thumb_ready = None         # RGBA float list for the texture
    busy = False
    cues: dict = {}            # lane index -> [(t0, t1)] seconds (REAPER / seconds cues)
    dirty = False
    redraw = False             # set by workers: the main thread redraws at its next tick
    relane = False
    thumbing = False
    entry = None               # the Generator or FX chosen in the browser (catalog key)
    plugin = None              # the layer (uid) the plugin window shows
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


def _bw() -> float:
    return (CANVAS_W - LABEL_W - 8) / max(1, _bars())


def _x_of(bar: float) -> float:
    return LABEL_W + (bar - 1.0) * _bw()


def _bar_of_x(x: float) -> int:
    return int(np.clip(np.floor((x - LABEL_W) / _bw()) + 1, 1, _bars()))


def _lane_y(k: int) -> float:
    return RULER_H + DENSITY_H + 6 + k * LANE_H


# ----------------------------------------------------------------------- the scene

def _load() -> None:
    from core.scene_doc import SceneDoc, plain
    path = A.host.scene_path()
    if not path or not Path(path).is_file():
        A.doc, A.items = None, []
        _status("no scene file: load or create one in the studio first")
        _relane()
        return
    from core.scene_doc import plain_keeping_signals
    A.doc = SceneDoc.load(path)
    A.items = [[node, plain(node), plain_keeping_signals(node), _uid()] for node in A.doc.layers()]
    A.sel, A.dirty, A.cues = None, False, {}
    A.undo, A.redo = [], []
    _relane()
    _status(f"arranging {Path(path).name}: {len(A.lanes)} lane(s)")
    _load_cues()


_UIDS = iter(range(1, 1 << 62))


def _uid() -> int:
    """A layer's steady name in the window (its place in the stack changes)."""
    return next(_UIDS)


def _specs() -> list:
    return [it[2] for it in A.items]


def _relane() -> None:
    A.lanes = list(reversed(arr.lanes_of(_specs())))      # the top of the picture on top
    _draw()
    _draw_inspector()
    _draw_plugin()


def _load_cues() -> None:
    """The lanes' own cues (seconds, or the REAPER edit) to show in grey: worked out in
    the background (a REAPER glue takes a few seconds to resolve)."""
    lanes = sorted((ln for ln in A.lanes if ln.cued),             # the quick ones first
                   key=lambda ln: isinstance(ln.spec.get("cues"), dict))
    if not lanes:
        return

    def run():
        from core.video.footage import read_cues
        for ln in lanes:
            try:
                A.cues[_cue_key(ln)] = [(c[0], c[0] + c[2]) for c in read_cues(ln.spec.get("cues"), ln.file)]
            except Exception as exc:                    # shown, not fatal: the lane still edits
                A.host.log(f"Arrange: cues of {ln.label} not read: {exc}")
            A.redraw = True
    threading.Thread(target=run, daemon=True).start()


def _cue_key(lane) -> str:
    """A lane's cues are kept by what they are, not by the lane's place (lanes come and go)."""
    return f"{lane.file}|{lane.spec.get('cues')!r}"


def _flow_blocks(doc) -> None:
    """Blocks and cutouts written one per line, ``{bars: [5, 8], from: 2.0}``."""
    from ruamel.yaml.comments import CommentedMap, CommentedSeq
    for node in doc.layers():
        bars = node.get("bars") if isinstance(node, dict) else None
        if isinstance(bars, list) and not isinstance(bars, CommentedSeq):   # just written: [[5, 8], [12, 16]]
            seq = CommentedSeq([_flow(CommentedSeq(w)) if isinstance(w, list) else w for w in bars])
            node["bars"] = _flow(seq)
        if not isinstance(node, dict) or node.get("source") != "footage":
            continue
        if node.get("blocks"):
            seq = CommentedSeq()
            for b in node["blocks"]:
                m = CommentedMap(b.items() if hasattr(b, "items") else b)
                m.fa.set_flow_style()
                seq.append(m)
            node["blocks"] = seq
        if isinstance(node.get("cutout"), dict) and not hasattr(node["cutout"], "fa"):
            m = CommentedMap(node["cutout"])
            m.fa.set_flow_style()
            node["cutout"] = m
        elif hasattr(node.get("cutout"), "fa"):
            node["cutout"].fa.set_flow_style()


def _flow(seq):
    seq.fa.set_flow_style()
    return seq


def btn_save() -> None:
    if A.doc is None:
        _status("nothing to save: no scene")
        return
    A.doc.set_layers([tuple(it[:3]) for it in A.items])
    _flow_blocks(A.doc)
    saved = list(A.doc.layers())
    path = A.host.scene_path()
    A.doc.save(path)
    A.host.log(f"Arrange: saved {Path(path).name} ({len(A.lanes)} lane(s))")
    A.host.reload_editor()
    history = (A.undo, A.redo)
    _load()
    A.undo, A.redo = (_rebase(h, saved) for h in history)
    _status(f"saved {Path(path).name}")


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


def _set_spec(lane_k: int, spec: dict, remember: bool = True) -> None:
    i = A.lanes[lane_k].index
    if spec == A.items[i][2]:                            # nothing changed: nothing to undo
        return
    if remember:
        _checkpoint()
    A.items[i][2] = spec
    A.dirty = True


def _set_blocks(lane_k: int, blocks: list) -> None:
    lane = A.lanes[lane_k]
    _set_spec(lane_k, (arr.with_blocks if lane.kind == arr.CLIP else arr.with_windows)(lane.spec, blocks))
    _relane()


def _new_block(lane_k: int, first: int, last: int) -> None:
    lane = A.lanes[lane_k]
    if lane.kind == arr.MONTAGE:
        _status("the clip montage plays under everything: right-click to put a Generator above it")
        return
    if lane.kind != arr.CLIP:
        block = arr.Block(first, last)
    elif lane.role == "background":
        if not A.clip:
            _status("pick a clip on the left first")
            return
        block = arr.Block(first, last, A.start, 1.0, A.clip)
    else:
        block = arr.Block(first, last, A.start if A.clip == lane.file else 0.0)
    blocks = arr.place([] if lane.always else lane.blocks, block, _bar_seconds())
    _set_blocks(lane_k, blocks)
    A.sel = (lane_k, arr.block_at(A.lanes[lane_k].blocks, min(first, last)))
    _draw()
    _draw_inspector()


def btn_add_background() -> None:
    if A.doc is None:
        return
    at = arr.insert_at(_specs(), "background")
    _checkpoint()
    A.items.insert(at, [None, None, arr.new_background(), _uid()])
    A.dirty = True
    _relane()
    _status("background lane added: drag across it to put the chosen clip there")


def btn_add_character() -> None:
    if A.doc is None or not A.clip:
        _status("pick the character's clip on the left first")
        return
    spec = arr.new_foreground(A.clip)
    at = arr.insert_at(_specs(), "foreground")
    _checkpoint()
    A.items.insert(at, [None, None, spec, _uid()])
    A.dirty = True
    _relane()
    _status(f"character lane {Path(A.clip).stem} added"
            + ("" if spec.get("cutout") else " - no mask track yet: Segment it to cut it out"))


def insert_entry(key: str, at: int, where: str = "") -> None:
    """A Generator or FX from the catalog, as a new layer at ``at`` in the stack."""
    from core import catalog
    e = catalog.by_key(key)
    if A.doc is None or e is None:
        return
    _checkpoint()
    A.items.insert(at, [None, None, e.spec(), _uid()])
    A.dirty = True
    _relane()
    A.sel = (next(k for k, lane in enumerate(A.lanes) if lane.index == at), None)
    _draw()
    _draw_inspector()
    _status(f"{e.label} added{where}: it plays the whole song - drag across its track to give it bars")


def insert_above(key: str, lane_k: int) -> None:
    insert_entry(key, A.lanes[lane_k].index + 1, f" above {A.lanes[lane_k].label}")


def insert_background(key: str) -> None:
    insert_entry(key, arr.insert_at(_specs(), arr.GENERATOR), " as the background")


def insert_master(key: str) -> None:
    insert_entry(key, arr.master_at(_specs()), " to the master (over everything)")


def _toggle(lane_k: int) -> None:
    spec = dict(A.lanes[lane_k].spec)
    if spec.get("enabled", True) is False:
        spec.pop("enabled")
    else:
        spec["enabled"] = False
    _set_spec(lane_k, spec)
    _relane()


def _is_master(lane_k: int) -> bool:
    """An FX with only FX above it: it works on the whole picture."""
    return all(lane.kind == arr.FX for lane in A.lanes[:lane_k + 1])


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


def context_menu(x: float, y: float) -> None:
    """The right click: insert above the track under the mouse, as the background, or at the
    master; turn a track off or delete it."""
    from core import catalog
    _close_menu()
    if A.doc is None:
        return
    k = _lane_at(y)
    with dpg.window(tag="arr_menu", popup=True, no_title_bar=True, autosize=True,
                    pos=dpg.get_mouse_pos(local=False)):
        if k is not None:
            lane = A.lanes[k]
            dpg.add_text(lane.label, color=(230, 210, 140))
            _catalog_menu("Insert generator above", catalog.GENERATOR, lambda key: insert_above(key, k))
            _catalog_menu("Add FX above (works on it and what is under it)", catalog.FX,
                          lambda key: insert_above(key, k))
            if lane.kind in (arr.GENERATOR, arr.FX):
                dpg.add_menu_item(label="Open plugin (its parameters, linked to the music)...",
                                  callback=lambda: (_close_menu(), open_plugin(k)))
            if lane.kind != arr.MONTAGE:
                dpg.add_menu_item(label="Turn on" if not lane.enabled else "Turn off (bypass)",
                                  callback=lambda: (_close_menu(), _toggle(k)))
                dpg.add_menu_item(label="Delete track", callback=lambda: (_close_menu(), _delete_lane(k)))
            dpg.add_separator()
        _catalog_menu("Insert generator as the background", catalog.GENERATOR, insert_background)
        _catalog_menu("Add FX to the master (over everything)", catalog.FX, insert_master)
        dpg.add_menu_item(label="+ background clip track", callback=lambda: (_close_menu(), btn_add_background()))
        if A.clip:
            dpg.add_menu_item(label=f"+ character track ({Path(A.clip).stem})",
                              callback=lambda: (_close_menu(), btn_add_character()))


def _delete_lane(lane_k: int) -> None:
    _checkpoint()
    del A.items[A.lanes[lane_k].index]
    A.sel, A.dirty = None, True
    _relane()


def _delete_block(lane_k: int, block_k: int) -> None:
    _set_blocks(lane_k, arr.remove(A.lanes[lane_k].blocks, block_k))
    A.sel = (lane_k, None)
    _draw_inspector()


# ----------------------------------------------------------------------- mouse

def _mouse_in_canvas():
    mx, my = dpg.get_mouse_pos(local=False)
    rx, ry = dpg.get_item_rect_min("arr_canvas")
    return mx - rx, my - ry


def _lane_at(y: float):
    k = int((y - _lane_y(0)) // LANE_H)
    return k if 0 <= k < len(A.lanes) else None


def on_canvas_click(sender, app_data) -> None:
    if app_data[0] == 1:                                 # the right button: the insert menu
        context_menu(*_mouse_in_canvas())
    elif app_data[0] == 0:
        press(*_mouse_in_canvas())


def on_canvas_double_click(sender, app_data) -> None:
    x, y = _mouse_in_canvas()
    k = _lane_at(y)
    if k is not None and x < LABEL_W and A.lanes[k].kind in (arr.GENERATOR, arr.FX):
        open_plugin(k)


def on_mouse_release(sender, app_data) -> None:
    if A.drag is None or not dpg.is_item_shown("arr_win"):
        A.drag = None
        return
    release(_mouse_in_canvas()[0])


def on_mouse_drag(sender, app_data) -> None:
    if A.drag is None or not dpg.does_item_exist("arr_ghost"):
        return
    got = _dragged(_mouse_in_canvas()[0])
    if got is None:
        return
    k, a, b = got
    y = _lane_y(k)
    dpg.configure_item("arr_ghost", pmin=[_x_of(a), y + 2], pmax=[_x_of(b + 1), y + LANE_H - 4], show=True)


EDGE = 5               # pixels at a block's ends that grab the end instead of the block


def press(x: float, y: float) -> None:
    """The mouse goes down on the canvas: on a lane's name it picks the lane; on a block's
    end it takes that end, on a block the block, elsewhere it starts a new block."""
    k = _lane_at(y)
    if k is None:
        return
    if x < LABEL_W:
        A.sel = (k, None)
        _draw()
        _draw_inspector()
        return
    bar = _bar_of_x(x)
    lane = A.lanes[k]
    if lane.kind == arr.MONTAGE:                         # always on: nothing to drag
        A.sel = (k, None)
        _draw()
        _draw_inspector()
        return
    for i, b in enumerate(lane.blocks):                  # an end within reach, on either side of it
        if abs(x - _x_of(b.last + 1)) <= EDGE:
            A.drag = (k, bar, "last", i)
            return
        if abs(x - _x_of(b.first)) <= EDGE:
            A.drag = (k, bar, "first", i)
            return
    hit = arr.block_at(lane.blocks, bar)
    A.drag = (k, bar, "new" if hit is None else "move", hit)


def _dragged(x: float):
    """(lane, first bar, last bar) the drag would leave, or None."""
    k, bar, mode, hit = A.drag
    now = _bar_of_x(x)
    if mode == "new":
        return k, min(bar, now), max(bar, now)
    b = A.lanes[k].blocks[hit]
    if mode == "move":
        d = now - bar
        d = max(d, 1 - b.first)
        d = min(d, _bars() - b.last)
        return k, b.first + d, b.last + d
    if mode == "first":
        return k, min(now, b.last), b.last
    return k, b.first, max(now, b.first)


def release(x: float) -> None:
    """The mouse comes up: a new block, a block moved or resized, or (no move) a block chosen."""
    k, bar, mode, hit = A.drag
    got = _dragged(x)
    A.drag = None
    if k >= len(A.lanes):
        return
    lane = A.lanes[k]
    _, first, last = got
    if mode == "new":
        _new_block(k, first, last)
        return
    b = lane.blocks[hit]
    if (first, last) == (b.first, b.last):              # no move: a click picks the block
        A.sel = (k, hit)
        if b.file or lane.file:
            _choose_clip(b.file or lane.file, b.start)
        _draw()
        _draw_inspector()
        return
    start = b.start
    if mode == "first":                                  # its start moves: the clip keeps its
        start = max(0.0, b.start + (first - b.first) * _bar_seconds() * b.rate)   # place in time
    moved = arr.Block(first, last, round(start, 3), b.rate, b.file)
    rest = [x for i, x in enumerate(lane.blocks) if i != hit]
    _set_blocks(k, arr.place(rest, moved, _bar_seconds()))
    A.sel = (k, arr.block_at(A.lanes[k].blocks, first))
    _draw()
    _draw_inspector()


# ----------------------------------------------------------------------- drawing

def _draw() -> None:
    if not dpg.does_item_exist("arr_canvas"):
        return
    dpg.delete_item("arr_canvas", children_only=True)
    n, bw = _bars(), _bw()
    h = _lane_y(len(A.lanes)) + 8
    dpg.configure_item("arr_canvas", height=max(120, int(h)))
    dpg.draw_rectangle([0, 0], [CANVAS_W, h], fill=(18, 20, 26, 255), color=(18, 20, 26, 255),
                       parent="arr_canvas")
    for b in range(1, n + 1):                            # the bar ruler
        x = _x_of(b)
        major = (b - 1) % 4 == 0
        dpg.draw_line([x, RULER_H - (10 if major else 4)], [x, h], parent="arr_canvas",
                      color=(70, 76, 95, 160) if major else (45, 50, 62, 120))
        if major or bw > 22:
            dpg.draw_text([x + 2, 4], str(b), size=12, color=(170, 175, 190, 255), parent="arr_canvas")
    dens = _density(n)                                   # how busy the MIDI is, bar by bar
    for b, v in enumerate(dens, 1):
        c = int(40 + 180 * v)
        dpg.draw_rectangle([_x_of(b), RULER_H], [_x_of(b + 1), RULER_H + DENSITY_H],
                           fill=(c, int(c * 0.8), 60, 255), color=(0, 0, 0, 0), parent="arr_canvas")
    for k, lane in enumerate(A.lanes):
        y = _lane_y(k)
        chosen = A.sel is not None and A.sel[0] == k
        dpg.draw_rectangle([0, y], [CANVAS_W, y + LANE_H - 2],
                           fill=(34, 38, 50, 255) if chosen else (26, 29, 38, 255),
                           color=(0, 0, 0, 0), parent="arr_canvas")
        tag, colour = _lane_tag(lane)
        name = lane.label if len(lane.label) <= 16 else "..." + lane.label[-13:]
        if not lane.enabled:
            colour = (110, 110, 120, 255)
        dpg.draw_text([6, y + 4], f"{tag} {name}", size=13, color=colour, parent="arr_canvas")
        note = ("off (bypassed)" if not lane.enabled else
                "master: over everything" if lane.kind == arr.FX and _is_master(k) else
                "on the tracks under it" if lane.kind == arr.FX else
                "under everything" if lane.kind == arr.MONTAGE else "")
        if note:
            dpg.draw_text([6, y + 18], note, size=10, color=(150, 150, 160, 255), parent="arr_canvas")
        if lane.cued:
            dpg.draw_text([6, y + 18], "cues (REAPER / s)", size=10, color=(150, 150, 160, 255),
                          parent="arr_canvas")
        if lane.always:                                  # no blocks: the whole song
            if not note:
                dpg.draw_text([6, y + 18], "the whole song", size=10, color=(150, 150, 160, 255),
                              parent="arr_canvas")
            dpg.draw_rectangle([_x_of(1), y + LANE_H - 8], [_x_of(_bars() + 1), y + LANE_H - 5],
                               fill=(120, 120, 130, 160), color=(0, 0, 0, 0), parent="arr_canvas")
        for t0, t1 in A.cues.get(_cue_key(lane), []):    # read-only cues, in grey
            xa, xb = _x_of(_time_to_bar(t0)), _x_of(_time_to_bar(t1))
            dpg.draw_rectangle([xa, y + LANE_H - 12], [max(xa + 1, xb), y + LANE_H - 5],
                               fill=(140, 140, 150, 200), color=(0, 0, 0, 0), parent="arr_canvas")
        for i, blk in enumerate(lane.blocks):
            xa, xb = _x_of(blk.first), _x_of(blk.last + 1)
            r, g, bb = arr.colour(blk.file or lane.file or lane.label)
            if not lane.enabled:
                r, g, bb = (r + 240) // 3, (g + 240) // 3, (bb + 240) // 3
            sel = A.sel == (k, i)
            dpg.draw_rectangle([xa + 1, y + 3], [xb - 1, y + LANE_H - 14 if lane.cued else y + LANE_H - 4],
                               fill=(r, g, bb, 230), color=(255, 255, 255, 255) if sel else (10, 10, 14, 255),
                               thickness=2 if sel else 1, parent="arr_canvas")
            stem = Path(blk.file or lane.file or "").stem if lane.kind == arr.CLIP else lane.label
            name = Path(blk.file).parent.name if blk.file else ""
            label = name if name and stem in ("final", "clip", "video") else stem[-12:]
            if blk.start:
                label += f" @{blk.start:g}s"
            room = int((xb - xa - 6) / 6.5)
            if room >= 3:
                dpg.draw_text([xa + 4, y + 6], label[:room], size=11, color=(10, 10, 14, 255),
                              parent="arr_canvas")
    dpg.draw_rectangle([0, 0], [0, 0], fill=(255, 255, 255, 60), color=(255, 255, 255, 200),
                       tag="arr_ghost", show=False, parent="arr_canvas")


def _lane_tag(lane) -> tuple:
    """What a track is, as its name starts, and its colour."""
    if lane.kind == arr.CLIP:
        return ("BG", (230, 210, 140, 255)) if lane.role == "background" else ("FG", (150, 210, 240, 255))
    return {arr.GENERATOR: ("GEN", (150, 230, 160, 255)), arr.FX: ("FX", (220, 160, 240, 255)),
            arr.MONTAGE: ("CLIPS", (200, 200, 200, 255))}[lane.kind]


def _density(n: int) -> list:
    notes = A.host.notes() or []
    g = A.host.grid()
    db = getattr(g, "downbeats", None) if g is not None else None
    if not notes or db is None or not len(db):
        return [0.0] * n
    counts = np.bincount(np.clip(np.searchsorted(db, [x.time for x in notes], side="right") - 1, 0, n - 1),
                         minlength=n)[:n].astype(float)
    return list(counts / max(1.0, counts.max()))


def _num(spec, key, default):
    v = spec.get(key, default)
    return float(v) if isinstance(v, (int, float)) else None


def _draw_inspector() -> None:
    if not dpg.does_item_exist("arr_inspector"):
        return
    dpg.delete_item("arr_inspector", children_only=True)
    p = "arr_inspector"
    if A.sel is None or A.sel[0] >= len(A.lanes):
        dpg.add_text("drag across a track to put a clip there or give a Generator/FX its bars · drag a "
                     "block to move it, its ends to resize it · right-click to insert a Generator or FX · "
                     "click a track's name for its settings · Ctrl+Z undo, Ctrl+Y redo",
                     parent=p, color=(130, 130, 150), wrap=700)
        return
    k, bk = A.sel
    lane = A.lanes[k]
    spec = lane.spec
    if bk is not None and bk < len(lane.blocks):
        b = lane.blocks[bk]
        dpg.add_text(f"block in {lane.label}: bars {b.first}-{b.last}"
                     + (f" · {Path(b.file or lane.file or '').as_posix()}" if lane.kind == arr.CLIP else ""),
                     parent=p, color=(230, 210, 140))

        def change(**kw):
            blocks = list(A.lanes[k].blocks)
            old = blocks[bk]
            new = arr.Block(old.first, old.last, old.start, old.rate, old.file)
            for key, v in kw.items():
                setattr(new, key, v)
            if new.last < new.first:
                return
            rest = [x for i, x in enumerate(blocks) if i != bk]
            _set_blocks(k, arr.place(rest, new, _bar_seconds()))
            A.sel = (k, arr.block_at(A.lanes[k].blocks, new.first))
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
        if lane.kind != arr.CLIP:
            dpg.add_button(label="delete block", parent=p, callback=lambda: _delete_block(k, bk))
            dpg.add_separator(parent=p)
        with dpg.group(horizontal=True, parent=p, show=lane.kind == arr.CLIP):
            dpg.add_text("from (s)")
            dpg.add_input_float(default_value=b.start, width=100, step=0.1, format="%.2f",
                                callback=edit("start"), on_enter=True)
            dpg.add_text("speed")
            dpg.add_input_float(default_value=b.rate, width=90, step=0.1, format="%.2f",
                                callback=edit("rate"), on_enter=True)
        with dpg.group(horizontal=True, parent=p, show=lane.kind == arr.CLIP):
            if lane.role == "background":
                dpg.add_button(label="use the chosen clip and its 'from'",
                               callback=lambda: change(file=A.clip, start=A.start) if A.clip else None)
            else:
                dpg.add_button(label="use the browser's 'from'", callback=lambda: change(start=A.start))
            dpg.add_button(label="delete block", callback=lambda: _delete_block(k, bk))
        if lane.kind == arr.CLIP:
            dpg.add_separator(parent=p)
    if lane.kind != arr.CLIP:
        _draw_effect_inspector(k, p)
        return
    # the lane
    dpg.add_text(f"lane {lane.label} ({lane.role})" + (f" · {lane.file}" if lane.file else ""),
                 parent=p, color=(150, 210, 240))

    def put(key, cast=float):
        def cb(s, v):
            new = dict(A.lanes[k].spec)
            new[key] = cast(v)
            _set_spec(k, new)
            _relane()
        return cb
    with dpg.group(horizontal=True, parent=p):
        for key, default in (("x", 0.5), ("y", 0.5), ("scale", 1.0)):
            v = _num(spec, key, default)
            dpg.add_text(key)
            if v is None:
                dpg.add_text("(follows the music - layer editor)", color=(200, 160, 100))
            else:
                dpg.add_input_float(default_value=v, width=90, step=0.05, format="%.2f",
                                    callback=put(key), on_enter=True)
        dpg.add_text("fit")
        dpg.add_combo(["contain", "cover"], default_value=spec.get("fit", "contain"), width=90,
                      callback=put("fit", str))
        dpg.add_checkbox(label="flip", default_value=bool(spec.get("flip", False)), callback=put("flip", bool))
        dpg.add_checkbox(label="loop", default_value=bool(spec.get("loop", False)), callback=put("loop", bool))
    with dpg.group(horizontal=True, parent=p):
        cut = spec.get("cutout") or {}
        track = cut.get("track") if isinstance(cut, dict) else cut
        dpg.add_text(f"cut out by: {track}" if track else "not cut out (whole frame)",
                     color=(150, 190, 150) if track else (150, 150, 160))
        if lane.file and not track:
            found = arr.find_mask_track(lane.file)
            if found:
                dpg.add_button(label=f"cut out ({Path(found).name})", callback=lambda: (
                    _set_spec(k, {**A.lanes[k].spec, "cutout": {"track": found, "feather": 2}}), _relane()))
        if track:
            dpg.add_button(label="whole frame", callback=lambda: (
                _set_spec(k, {key: v for key, v in A.lanes[k].spec.items() if key != "cutout"}), _relane()))
            drop = isinstance(cut, dict) and cut.get("drop")
            dpg.add_checkbox(label="all but object 1 (a backdrop)", default_value=bool(drop),
                             callback=lambda s, v: (_set_spec(k, {**A.lanes[k].spec, "cutout": {
                                 **{a: b for a, b in cut.items() if a not in ("drop", "objects")},
                                 **({"drop": [1]} if v else {})}}), _relane()))
        dpg.add_button(label="delete lane", callback=lambda: _delete_lane(k))


def _draw_effect_inspector(k: int, p) -> None:
    """A Generator's or FX's track (or the montage's): what it is, on/off, how a Generator
    mixes in, and delete."""
    from core import catalog
    lane = A.lanes[k]
    spec = lane.spec
    if lane.kind == arr.MONTAGE:
        dpg.add_text("the clip montage: the project's clips folder, cut by the triggers, under everything. "
                     "Right-click to put a Generator or an FX above it.", parent=p, wrap=700, color=(150, 150, 160))
        return
    e = catalog.of_layer(spec)
    tag, colour = _lane_tag(lane)
    dpg.add_text(f"{'Generator' if lane.kind == arr.GENERATOR else 'FX'} {lane.label}"
                 + (f" · {e.category}" if e else ""), parent=p, color=colour[:3])
    if e:
        dpg.add_text(e.about, parent=p, wrap=700)
    with dpg.group(horizontal=True, parent=p):
        dpg.add_checkbox(label="on", default_value=lane.enabled, callback=lambda: _toggle(k))
        if lane.kind == arr.GENERATOR:
            def put(key, cast):
                return lambda s, v: (_set_spec(k, {**A.lanes[k].spec, key: cast(v)}), _relane())
            dpg.add_text("blend")
            dpg.add_combo(["normal", "add", "screen", "multiply"], default_value=spec.get("blend", "normal"),
                          width=100, callback=put("blend", str))
            op = spec.get("opacity", 1.0)
            dpg.add_text("opacity")
            if isinstance(op, (int, float)):
                dpg.add_slider_float(default_value=float(op), min_value=0.0, max_value=1.0, width=140,
                                     callback=put("opacity", float))
            else:
                dpg.add_text("(follows the music - layer editor)", color=(200, 160, 100))
        else:
            dpg.add_text("master: works on the whole picture" if _is_master(k)
                         else "works on every track under it", color=(150, 150, 160))
        dpg.add_button(label="delete track", callback=lambda: _delete_lane(k))
        dpg.add_button(label="Open plugin...", callback=lambda: open_plugin(k))
    if e and e.needs:
        dpg.add_text("needs " + " and ".join("a mask track (SAM2)" if n == "masks" else "the clip montage's cuts"
                                             for n in e.needs), parent=p, color=(200, 160, 100))
    dpg.add_text("its parameters: Open plugin (or double-click the track's name)", parent=p,
                 color=(120, 120, 140), wrap=700)


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
    spec = plain_keeping_signals(it[2])
    holder = spec
    for k in path[:-1]:
        nxt = holder.get(k)
        nxt = dict(nxt) if isinstance(nxt, dict) else {}
        holder[k] = nxt
        holder = nxt
    last = path[-1]
    old = holder.get(last)
    links = dict(links or {})
    redefined = None                                     # a signal defined here, given a new value
    if value is None and not links:
        new = None
        holder.pop(last, None)
        if len(path) == 2 and not holder:                # no uniforms left
            spec.pop(path[0], None)
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
        return set_param(self.uid, tuple(path), spec, anchor, links) or None


def open_plugin(lane_k: int) -> None:
    """A Generator's or FX's window: each parameter with its value and a Link to the music."""
    lane = A.lanes[lane_k]
    A.plugin = next(it[3] for it in A.items if it[2] is lane.spec)
    if not dpg.does_item_exist("arr_plugin"):
        with dpg.window(tag="arr_plugin", width=560, height=640, pos=(560, 60), on_close=_plugin_closed):
            dpg.add_group(tag="arr_plugin_body")
    dpg.show_item("arr_plugin")
    _draw_plugin()
    dpg.focus_item("arr_plugin")


def _plugin_closed(*_):
    A.plugin = None


def _summary(value) -> str:
    """A knob on one line: what in the music moves it."""
    from core.scene_doc import plain
    import yaml
    text = yaml.safe_dump(plain(value), default_flow_style=True, width=1 << 16).strip()
    return text if len(text) <= 64 else text[:61] + "..."


def _draw_plugin() -> None:
    from core import catalog
    from core.knob_forms import form_of
    from core.scene_doc import _anchor_name, plain
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
    uid, spec = it[3], it[2]
    e = catalog.of_layer(spec)
    kind = "FX" if e and e.kind == catalog.FX else "Generator"
    name = e.label if e else str(spec.get("source"))
    dpg.configure_item("arr_plugin", label=f"{name} - {kind}" + (f" · {e.category}" if e else ""))
    if e:
        dpg.add_text(e.about, parent=body, wrap=530)
    on = spec.get("enabled", True) is not False
    with dpg.group(horizontal=True, parent=body):
        dpg.add_checkbox(label="on", default_value=on,
                         callback=lambda s, v: set_param(uid, ("enabled",), None if v else False))
        if kind == "Generator":
            dpg.add_text("blend")
            dpg.add_combo(["normal", "add", "screen", "multiply"], default_value=spec.get("blend", "normal"),
                          width=90, callback=lambda s, v: set_param(uid, ("blend",), None if v == "normal" else v))
            op = spec.get("opacity", 1.0)
            dpg.add_text("opacity")
            if isinstance(op, (int, float)):
                dpg.add_slider_float(default_value=float(op), min_value=0.0, max_value=1.0, width=120,
                                     callback=lambda s, v: set_param(uid, ("opacity",), None if v >= 1.0 else round(v, 3)))
    params = list(e.params) if e else []
    if kind == "FX":
        params.insert(0, catalog.Param("mix", 1.0, "dry/wet: 0 the picture it was given .. 1 all of its result"))
    known = {p.name for p in params}
    extra = [k for k in spec if k not in known and k not in catalog.COMMON_KEYS and k not in ("file", "uniforms", "mix")]
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
            path = _param_path(spec, prm.name)
            from core.scene_doc import get_path
            value = get_path(spec, path)
            with dpg.table_row():
                label = dpg.add_text(("~ " if prm.knob else "  ") + prm.name,
                                     color=(230, 210, 140) if value is not None else (170, 170, 180))
                if prm.about:
                    with dpg.tooltip(label):
                        dpg.add_text(prm.about, wrap=380)
                numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
                if numeric or (value is None and prm.knob):
                    shown = float(value) if numeric else float(prm.default)
                    dpg.add_input_float(default_value=shown, width=-1, step=0, format="%.4g", on_enter=True,
                                        callback=lambda s, v, path=path: set_param(uid, path, v))
                elif value is None:
                    dpg.add_text("-" if prm.default is None else str(prm.default), color=(120, 120, 130))
                else:
                    sig = _anchor_name(value)
                    dpg.add_text(("*" + sig + "  " if sig else "") + _summary(value),
                                 color=(230, 210, 140) if sig else (170, 210, 170))
                with dpg.group(horizontal=True):
                    if prm.knob or (isinstance(value, dict) and form_of(plain(value)) is not None):
                        dpg.add_button(label="Link" if not isinstance(value, dict) else "Edit",
                                       callback=lambda s, a, path=path, d=prm.default: _link(uid, path, d))
                    if value is not None:
                        dpg.add_button(label="x", callback=lambda s, a, path=path: set_param(uid, path, None))
                        with dpg.tooltip(dpg.last_item()):
                            dpg.add_text("back to its default")


def _link(uid, path: tuple, default) -> None:
    """The studio's knob editor on this parameter."""
    it = _item(uid)
    if it is None or not hasattr(A.host, "edit_knob"):
        _status("Link needs the studio's knob editor")
        return
    from core import catalog
    e = catalog.of_layer(it[2])
    title = e.label if e else str(it[2].get("source"))
    A.host.edit_knob(PluginTarget(uid, title), path, default if isinstance(default, (int, float)) else 0.0)


# ----------------------------------------------------------------------- the browser

def _status(msg: str) -> None:
    if dpg.does_item_exist("arr_status"):
        dpg.set_value("arr_status", msg)


def btn_list() -> None:
    folder = dpg.get_value("arr_folder")
    pattern = dpg.get_value("arr_filter")
    A.clips = arr.list_clips(folder, pattern)
    names = [p.relative_to(folder).as_posix() for p in A.clips]
    dpg.configure_item("arr_list", items=names)
    _status(f"{len(names)} clip(s) in {folder}" + (f" matching '{pattern}'" if pattern else ""))


def _pick_folder(sender, app_data) -> None:
    path = app_data.get("file_path_name") if isinstance(app_data, dict) else None
    if path:
        dpg.set_value("arr_folder", path)
        btn_list()


def on_pick_clip(sender, name) -> None:
    folder = dpg.get_value("arr_folder")
    _choose_clip(str(Path(folder) / name), 0.0)


def _choose_clip(path: str, start: float) -> None:
    A.clip = str(path)
    A.clip_len = arr.duration(A.clip)
    A.start = min(start, A.clip_len)
    dpg.configure_item("arr_from", max_value=max(0.01, A.clip_len))
    dpg.set_value("arr_from", A.start)
    mask = arr.find_mask_track(A.clip)
    dpg.set_value("arr_clip_info", f"{Path(A.clip).name} · {A.clip_len:.1f} s"
                  + (f" · mask: {Path(mask).name}" if mask else " · no mask track"))
    A.want_thumb = (A.clip, A.start, 0.0)


def on_from(sender, value) -> None:
    A.start = float(value)
    if A.clip:
        A.want_thumb = (A.clip, A.start, time.time())


def btn_segment() -> None:
    """SAM2 on the chosen clip (segment_video.py, on the GPU) into render_output/footage."""
    if not A.clip or A.busy:
        return
    weights = os.environ.get("ARC_SAM2_WEIGHTS") or next((w for w in SAM2_DEFAULTS if Path(w).is_dir()), None)
    if not weights:
        _status("SAM2 weights not found: set ARC_SAM2_WEIGHTS")
        return
    try:
        used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        used = "?"
    clip, out = A.clip, arr.mask_folder_for(A.clip)
    A.busy = True
    _status(f"SAM2 on {Path(clip).name} (GPU now: {used})…")

    def run():
        cmd = [sys.executable, "segment_video.py", "--video", clip, "--output", out, "--start-time", "0",
               "--sam2-weights", weights, "--max-objects", "4", "--cuts", "detect", "--preview"]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        A.busy = False
        if proc.returncode == 0:
            A.host.log(f"Arrange: mask track of {Path(clip).name} -> {out}")
            _status(f"segmented: {out} (its preview: {out}_preview.mp4)")
            mine = [k for k, lane in enumerate(A.lanes)       # a lane of this clip gets cut out
                    if lane.file and Path(lane.file).name == Path(clip).name and not lane.spec.get("cutout")]
            if mine:
                _checkpoint()                                  # one undo step for all of them
            for k in mine:
                _set_spec(k, {**A.lanes[k].spec, "cutout": {"track": out, "feather": 2}}, remember=False)
            A.relane = True
        else:
            _status(f"SAM2 failed: {proc.stderr.strip()[-200:]}")
    threading.Thread(target=run, daemon=True).start()


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
    _show_entry(e)


def _show_entry(e) -> None:
    """The chosen entry: what it is, what it needs, its parameters (the numbers the music
    can move are marked ~)."""
    from core import catalog
    tag = f"arr_cat_info_{e.kind}"
    dpg.delete_item(tag, children_only=True)
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
    with dpg.group(horizontal=True, parent=tag):
        if e.kind == catalog.GENERATOR:
            dpg.add_button(label="Insert as background", callback=lambda: insert_background(e.key))
        else:
            dpg.add_button(label="Add to master", callback=lambda: insert_master(e.key))
    dpg.add_text("or right-click a track to put it above that track", parent=tag, color=(120, 120, 140), wrap=300)
    dpg.add_text(f"parameters ({len(e.params)}; ~ the music can move it):", parent=tag, color=(150, 210, 240))
    for prm in e.params:
        value = "" if prm.default is None else f" = {prm.default}"
        line = dpg.add_text(f"{'~' if prm.knob else ' '} {prm.name}{value}", parent=tag)
        if prm.about:
            with dpg.tooltip(line):
                dpg.add_text(prm.about, wrap=360)


def _thumb_rgba(png: Path) -> list:
    import cv2
    img = cv2.imread(str(png), cv2.IMREAD_COLOR)
    canvas = np.zeros((THUMB, THUMB, 4), np.float32)
    canvas[..., 3] = 1.0
    if img is not None:
        h, w = img.shape[:2]
        s = THUMB / max(h, w)
        img = cv2.resize(img, (max(1, int(w * s)), max(1, int(h * s))))
        h, w = img.shape[:2]
        y0, x0 = (THUMB - h) // 2, (THUMB - w) // 2
        canvas[y0:y0 + h, x0:x0 + w, :3] = img[..., ::-1] / 255.0
    return canvas.ravel().tolist()


def tick() -> None:
    """Once a frame, on the main thread: a thumbnail asked for (after the slider rests a
    moment) is made in the background and shown when ready; redraws asked by workers."""
    if getattr(A, "redraw", False):
        A.redraw = False
        _draw()
    if getattr(A, "relane", False):
        A.relane = False
        _relane()
    if A.thumb_ready is not None:
        dpg.set_value("arr_tex", A.thumb_ready)
        A.thumb_ready = None
    if A.want_thumb and time.time() - A.want_thumb[2] > 0.25 and not getattr(A, "thumbing", False):
        path, t, _ = A.want_thumb
        A.want_thumb, A.thumbing = None, True

        def run():
            try:
                png = arr.thumbnail(path, t, THUMB)
                if png:
                    A.thumb_ready = _thumb_rgba(png)
            finally:
                A.thumbing = False
        threading.Thread(target=run, daemon=True).start()


# ----------------------------------------------------------------------- the window

def open_window() -> None:
    _load()
    if not A.catalog_drawn:                     # the catalog is read once, when first shown
        A.catalog_drawn = True
        for kind in ("generator", "fx"):
            _draw_catalog(kind)
    if not dpg.get_value("arr_folder"):
        dpg.set_value("arr_folder", A.host.clips_dir() or "")
    dpg.show_item("arr_win")
    dpg.focus_item("arr_win")


def build(host) -> None:
    A.host = host
    A.catalog_drawn = False
    with dpg.texture_registry():
        dpg.add_dynamic_texture(THUMB, THUMB, [0.0, 0.0, 0.0, 1.0] * THUMB * THUMB, tag="arr_tex")
    dpg.add_file_dialog(show=False, directory_selector=True, callback=_pick_folder, tag="dlg_arr_folder",
                        width=620, height=420)
    with dpg.window(label="Arrange - the clips, and when the background and the characters come in",
                    tag="arr_win", show=False, width=1090, height=900, pos=(5, 40)):
        with dpg.group(horizontal=True):
            with dpg.child_window(width=330, height=-1), dpg.tab_bar():
                with dpg.tab(label="Clips"):
                    dpg.add_text("CLIPS - video or image files", color=(230, 210, 140))
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="arr_folder", width=260, hint="a folder of clips",
                                           on_enter=True, callback=lambda: btn_list())
                        dpg.add_button(label="...", callback=lambda: dpg.show_item("dlg_arr_folder"))
                    with dpg.group(horizontal=True):
                        dpg.add_input_text(tag="arr_filter", width=200, hint="filter, e.g. final.mp4",
                                           on_enter=True, callback=lambda: btn_list())
                        dpg.add_button(label="list", callback=btn_list)
                    dpg.add_listbox([], tag="arr_list", width=-1, num_items=14, callback=on_pick_clip)
                    dpg.add_text("", tag="arr_clip_info", wrap=310, color=(150, 190, 150))
                    dpg.add_image("arr_tex", width=THUMB, height=THUMB)
                    dpg.add_slider_float(label="from (s)", tag="arr_from", width=230, min_value=0.0,
                                         max_value=1.0, callback=on_from)
                    dpg.add_text("a new block starts the clip here", color=(120, 120, 140))
                    dpg.add_separator()
                    dpg.add_button(label="+ background lane", width=-1, callback=btn_add_background)
                    dpg.add_button(label="+ character lane (this clip)", width=-1, callback=btn_add_character)
                    dpg.add_button(label="Segment this clip (SAM2, GPU)", width=-1, callback=btn_segment)
                    dpg.add_text("a character is cut out by its mask track "
                                 "(render_output/footage/<clip>_segments)", wrap=310, color=(120, 120, 140))
                for kind, title, about in (
                        ("generator", "Generators", "they make a picture: a fractal, cells, a nebula"),
                        ("fx", "FX", "they work on the picture they get")):
                    with dpg.tab(label=title):
                        dpg.add_text(about, color=(120, 120, 140), wrap=310)
                        dpg.add_input_text(tag=f"arr_cat_q_{kind}", width=-1, hint="search",
                                           callback=lambda s, a, k=kind: _draw_catalog(k))
                        dpg.add_child_window(tag=f"arr_cat_{kind}", height=420, border=False)
                        dpg.add_separator()
                        dpg.add_child_window(tag=f"arr_cat_info_{kind}", height=-1, border=False)
            with dpg.child_window(width=-1, height=-1):
                with dpg.group(horizontal=True):
                    dpg.add_button(label="Save to scene", width=120, callback=btn_save)
                    dpg.add_button(label="Reload", width=80, callback=lambda: _load())
                    dpg.add_button(label="Undo", width=60, callback=btn_undo)
                    dpg.add_button(label="Redo", width=60, callback=btn_redo)
                    dpg.add_text("", tag="arr_status", color=(160, 200, 160))
                dpg.add_text("bars (the strip under them: how busy the MIDI is) · saving reloads the "
                             "studio's layer editor from the file", color=(120, 120, 140))
                with dpg.child_window(height=-250, border=False):  # the tracks scroll; the inspector stays
                    dpg.add_drawlist(width=CANVAS_W, height=200, tag="arr_canvas")
                with dpg.item_handler_registry(tag="arr_canvas_handler"):
                    dpg.add_item_clicked_handler(callback=on_canvas_click)
                    dpg.add_item_double_clicked_handler(callback=on_canvas_double_click)
                dpg.bind_item_handler_registry("arr_canvas", "arr_canvas_handler")
                dpg.add_separator()
                with dpg.child_window(height=-1, border=False):
                    dpg.add_group(tag="arr_inspector")
    with dpg.handler_registry():
        dpg.add_mouse_release_handler(button=dpg.mvMouseButton_Left, callback=on_mouse_release)
        dpg.add_mouse_drag_handler(button=dpg.mvMouseButton_Left, threshold=2, callback=on_mouse_drag)
        dpg.add_key_press_handler(dpg.mvKey_Z, callback=on_key_z)
        dpg.add_key_press_handler(dpg.mvKey_Y, callback=on_key_y)

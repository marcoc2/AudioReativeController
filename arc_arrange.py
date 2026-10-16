"""ARC Studio's Arrange window: what plays in the background and in the foreground, bar
by bar, and the FX on each.

Left, the browser: the Clips (a folder, a filter such as "final.mp4", the clip's frame at
the point it would start from), the Generators (they make a picture: a fractal, cells) and
the FX (they work on the picture they get), by category (core/catalog.py). Right, two
lanes over the song's bars (core/arrangement.py): the FG (the characters, drawn on their
own and put over the rest) and the BG (over the clip montage). Pick a clip or a Generator
and drag across a lane to put it there; things at the same time stack up in sub-rows.
FX live in three places: on a block (it alone), on a lane (all of it) and on the master
(over everything): a right click adds them, a click on a lane's name or a block lists
them, each with its own bars and its plugin (its parameters, linked to the music).
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
    items: list = []           # [node or None, base or None, spec, uid] per layer, in stack order
    rows: list = []            # what each row of the canvas shows, top to bottom
    members: dict = {}         # lane name -> [core.arrangement.Member]
    converted = False          # the scene was flat: grouped into BG / FG / master on opening
    content = None             # what a drag puts in a lane: ("clip", path, from) or ("gen", key)
    clips: list = []           # the browser's clips (Path)
    clip = None                # the chosen clip (str)
    clip_len = 0.0
    start = 0.0                # where in it a new block starts (s)
    sel = None                 # ("block", lane, member, block) | ("lane", lane) | ("master",) | ("montage",)
    zoom = 1.0                 # 1: the whole song across the canvas
    view0 = 1                  # the first bar shown
    drag = None                # (row, first bar, mode, (member, block)) while dragging
    want_thumb = None          # (path, t, asked at)
    thumb_ready = None         # RGBA float list for the texture
    busy = False
    cues: dict = {}            # a cued member's [(t0, t1)] seconds (REAPER / seconds cues)
    dirty = False
    redraw = False             # set by workers: the main thread redraws at its next tick
    relane = False
    thumbing = False
    pending = None             # (uid, fn): an edit a worker asks of the main thread
    entry = None               # the Generator or FX chosen in the browser (catalog key)
    plugin = None              # the layer (uid) the plugin window shows
    plugin_at = ()             # and where in it: () the layer, ("layers", 2, "fx", 0) a block's FX
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
    return (CANVAS_W - LABEL_W - 8) / _shown()


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


def on_wheel(sender, delta) -> None:
    """Over the canvas: Ctrl+wheel zooms around the bar under the mouse, Shift+wheel moves along."""
    if not dpg.does_item_exist("arr_canvas") or not dpg.is_item_hovered("arr_canvas"):
        return
    if _down("mvKey_ModCtrl", "mvKey_LControl", "mvKey_RControl"):
        set_view(zoom=A.zoom * (1.25 if delta > 0 else 0.8), keep=_bar_of_x(_mouse_in_canvas()[0]))
    elif _down("mvKey_ModShift", "mvKey_LShift", "mvKey_RShift"):
        set_view(first=A.view0 - int(np.sign(delta)) * max(1, _shown() // 8))


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
    known = {id(s): node for s, node in zip(specs, nodes)}
    layers, A.converted = arr.grouped(specs)
    A.items = [[known.get(id(s)), plain(known[id(s)]) if id(s) in known else None, s, _uid()] for s in layers]
    _relane()
    _status(f"arranging {Path(path).name}"
            + (" - a flat scene, grouped into BG / FG / master: Save writes it so" if A.converted else ""))
    _load_cues()


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


def _relane() -> None:
    """The rows, top to bottom: the master, the FG's sub-rows and an empty one to put
    something new on, the same for the BG, the clip montage."""
    A.members, A.rows = {}, []
    if A.doc is not None:
        A.rows.append({"kind": "master"})
        for name in (arr.FG, arr.BG):
            item = _lane_item(name)
            mems = arr.members_of(item[2]) if item else []
            cue_bars = {m.at: [(int(_time_to_bar(t0)), int(_time_to_bar(t1))) for t0, t1 in A.cues[_cue_key(m.spec)]]
                        for m in mems if m.cued and A.cues.get(_cue_key(m.spec))}
            n = arr.pack(mems, _bars(), cue_bars)
            A.members[name] = mems
            A.rows += [{"kind": "lane", "lane": name, "sub": r} for r in range(n + 1)]
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


def _tidy(node, key=None) -> None:
    """Small parts written on one line: bars, blocks, cutouts, each FX of a list; the
    signals (anchored parts) as they are."""
    from core.scene_doc import _anchor_name, flow
    from ruamel.yaml.comments import CommentedMap, CommentedSeq

    def one_line(v):
        if _anchor_name(v):
            return v
        if isinstance(v, dict):
            out = CommentedMap((k, one_line(x)) for k, x in v.items())
        elif isinstance(v, list):
            out = CommentedSeq(one_line(x) for x in v)
        else:
            return v
        out.fa.set_flow_style()
        return out
    if isinstance(node, dict) and not _anchor_name(node):
        for k in list(node.keys()):
            v = node[k]
            if k in ("bars", "cutout", "at") and isinstance(v, (dict, list)):
                node[k] = one_line(v)
            elif k in ("fx", "blocks") and isinstance(v, list):
                seq = v if isinstance(v, CommentedSeq) else CommentedSeq(v)
                for j, x in enumerate(seq):
                    seq[j] = one_line(x) if isinstance(x, dict) else x
                node[k] = seq
            else:
                _tidy(v, k)
    elif isinstance(node, list):
        for x in node:
            _tidy(x)


def btn_save() -> None:
    if A.doc is None:
        _status("nothing to save: no scene")
        return
    A.doc.set_layers([tuple(it[:3]) for it in A.items])
    for node in A.doc.layers():
        _tidy(node)
    saved = list(A.doc.layers())
    path = A.host.scene_path()
    A.doc.save(path)
    A.host.log(f"Arrange: saved {Path(path).name}")
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


def _set_member(lane: str, at: int, spec: dict) -> None:
    _edit(_lane_item(lane)[3], ("layers", at), lambda _: spec)


def _set_blocks(lane: str, at: int, blocks: list) -> None:
    mem = _member(lane, at)
    if not blocks:                                       # no window left: the member goes
        _remove_member(lane, at)
        return
    _set_member(lane, at, (arr.with_blocks if mem.kind == arr.CLIP else arr.with_windows)(mem.spec, blocks))


def _remove_member(lane: str, at: int) -> None:
    item = _lane_item(lane)
    _edit(item[3], ("layers",), lambda ls: [x for j, x in enumerate(ls or []) if j != at])
    A.sel = ("lane", lane)
    _draw_inspector()


def _content_spec(lane: str, first: int, last: int):
    """A new member from what the browser has chosen: a clip, or a Generator."""
    from core import catalog
    if A.content is None:
        return None
    if A.content[0] == "gen":
        e = catalog.by_key(A.content[1])
        return {**e.spec(), "bars": [[first, last]]} if e else None
    clip, start = A.content[1], A.content[2]
    block = arr.Block(first, last, start)
    if lane == arr.BG:
        return arr.with_blocks({"source": "footage", "role": "background", "fit": "cover", "loop": True,
                                "file": clip}, [block])
    return arr.new_foreground(clip, [block])


def _new_member(lane: str, first: int, last: int) -> None:
    spec = _content_spec(lane, first, last)
    if spec is None:
        _status("pick a clip (Clips) or a Generator (Generators) on the left first")
        return
    item = _lane_item(lane)
    _edit(item[3], ("layers",), lambda ls: list(ls or []) + [spec])
    at = len(item[2].get("layers") or []) - 1
    A.sel = ("block", lane, at, 0)
    _draw()
    _draw_inspector()
    _status(f"{arr.label_of(spec)} in {lane}, bars {first}-{last}"
            + (" - no mask track yet: Segment it to cut it out"
               if lane == arr.FG and spec.get("source") == "footage" and not spec.get("cutout") else ""))


def add_fx(uid, path: tuple, key: str) -> None:
    """An FX from the catalog at the end of the list at ``path`` (a block's, a lane's)."""
    from core import catalog
    e = catalog.by_key(key)
    if e is not None:
        _edit(uid, path, lambda fx: list(fx or []) + [e.spec()])
        _status(f"{e.label} added: it plays all the time its owner does - give it bars in the list")


def add_master(key: str) -> None:
    """An FX over everything: the top of the stack, under a ``meters`` read-out."""
    from core import catalog
    e = catalog.by_key(key)
    if A.doc is None or e is None:
        return
    at = len(A.items)
    while at > 0 and A.items[at - 1][2].get("source") == "meters":
        at -= 1
    _checkpoint()
    A.items.insert(at, [None, None, e.spec(), _uid()])
    A.dirty = True
    A.sel = ("master",)
    _relane()
    _status(f"{e.label} added to the master")


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


def context_menu(x: float, y: float) -> None:
    """The right click: on a block, its FX, plugin, on/off, delete; on a lane, a Generator
    there, an FX on the lane; anywhere, an FX on the master."""
    from core import catalog
    if A.doc is None:
        return
    r = _row_at(y)
    row = A.rows[r] if r is not None else {"kind": "none"}
    bar = _bar_of_x(x)

    def build():
        if row["kind"] == "lane":
            lane = row["lane"]
            hit = _block_at(r, bar)
            uid = _lane_item(lane)[3]
            if hit is not None and x >= LABEL_W:
                mem = _member(lane, hit[0])
                dpg.add_text(f"{lane} · {mem.label}", color=(230, 210, 140))
                _catalog_menu("Add FX to this block", catalog.FX,
                              lambda key: add_fx(uid, ("layers", mem.at, "fx"), key))
                if mem.kind != arr.CLIP:
                    dpg.add_menu_item(label="Open plugin (its parameters, linked to the music)...",
                                      callback=lambda: (_close_menu(), open_plugin(uid, ("layers", mem.at))))
                dpg.add_menu_item(label="Turn on" if not mem.enabled else "Turn off (bypass)",
                                  callback=lambda: (_close_menu(), _toggle(uid, ("layers", mem.at))))
                blk = mem.blocks[hit[1]]
                dpg.add_menu_item(label=f"Preview its bars ({blk.first}-{blk.last})",
                                  callback=lambda: (_close_menu(), preview_bars(blk.first, blk.last)))
                dpg.add_menu_item(label="Delete this block", callback=lambda: (
                    _close_menu(), _delete_block(lane, mem.at, hit[1])))
                dpg.add_separator()
            else:
                dpg.add_text(lane, color=(230, 210, 140))
            _catalog_menu(f"Put a Generator here (bar {bar})", catalog.GENERATOR,
                          lambda key: _put_generator(lane, bar, key))
            _catalog_menu(f"Add FX to the {lane} (all of it)", catalog.FX,
                          lambda key: add_fx(uid, ("fx",), key))
            dpg.add_separator()
        _catalog_menu("Add FX to the master (over everything)", catalog.FX, add_master)
    _menu_at(build)


def _delete_block(lane: str, at: int, i: int) -> None:
    mem = _member(lane, at)
    if mem is None:
        return
    if mem.cued or mem.always:
        _remove_member(lane, at)
        return
    _set_blocks(lane, at, arr.remove(mem.blocks, i))
    A.sel = ("lane", lane)
    _draw_inspector()


# ----------------------------------------------------------------------- mouse

def _mouse_in_canvas():
    mx, my = dpg.get_mouse_pos(local=False)
    rx, ry = dpg.get_item_rect_min("arr_canvas")
    return mx - rx, my - ry


def _row_y(r: int) -> float:
    return RULER_H + DENSITY_H + 6 + r * LANE_H


def _row_at(y: float):
    r = int((y - _row_y(0)) // LANE_H)
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
    x, y = _mouse_in_canvas()
    r = _row_at(y)
    if r is None or A.rows[r]["kind"] != "lane":
        return
    hit = _block_at(r, _bar_of_x(x))
    lane = A.rows[r]["lane"]
    if hit is not None and _member(lane, hit[0]).kind != arr.CLIP:
        open_plugin(_lane_item(lane)[3], ("layers", hit[0]))


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
    r, a, b = got
    y = _row_y(r)
    dpg.configure_item("arr_ghost", pmin=[_x_of(a), y + 2], pmax=[_x_of(b + 1), y + LANE_H - 4], show=True)


EDGE = 5               # pixels at a block's ends that grab the end instead of the block


def press(x: float, y: float) -> None:
    """The mouse goes down on the canvas: on a row's name it picks the lane (or the
    master, the montage); on a block's end it takes that end, on a block the block,
    elsewhere on a lane it starts a new member there."""
    r = _row_at(y)
    if r is None:
        return
    row = A.rows[r]
    if row["kind"] != "lane" or x < LABEL_W:
        A.sel = ("lane", row["lane"]) if row["kind"] == "lane" else (row["kind"],)
        _draw()
        _draw_inspector()
        return
    bar = _bar_of_x(x)
    for mem in A.members.get(row["lane"], []):
        if mem.row != row["sub"]:
            continue
        for i, b in enumerate(mem.blocks):               # an end within reach, on either side of it
            if abs(x - _x_of(b.last + 1)) <= EDGE:
                A.drag = (r, bar, "last", (mem.at, i))
                return
            if abs(x - _x_of(b.first)) <= EDGE:
                A.drag = (r, bar, "first", (mem.at, i))
                return
    hit = _block_at(r, bar)
    A.drag = (r, bar, "new" if hit is None else "move", hit)


def _dragged(x: float):
    """(row, first bar, last bar) the drag would leave."""
    r, bar, mode, hit = A.drag
    now = _bar_of_x(x)
    if mode == "new":
        return r, min(bar, now), max(bar, now)
    b = _member(A.rows[r]["lane"], hit[0]).blocks[hit[1]]
    if mode == "move":
        d = max(1 - b.first, min(_bars() - b.last, now - bar))
        return r, b.first + d, b.last + d
    if mode == "first":
        return r, min(now, b.last), b.last
    return r, b.first, max(now, b.first)


def release(x: float) -> None:
    """The mouse comes up: a new member, a block moved or resized, or (no move) a block chosen."""
    r, bar, mode, hit = A.drag
    got = _dragged(x)
    A.drag = None
    if r >= len(A.rows) or A.rows[r]["kind"] != "lane":
        return
    lane = A.rows[r]["lane"]
    _, first, last = got
    if mode == "new":
        _new_member(lane, first, last)
        return
    mem = _member(lane, hit[0])
    b = mem.blocks[hit[1]]
    if (first, last) == (b.first, b.last):              # no move: a click picks the block
        A.sel = ("block", lane, mem.at, hit[1])
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
    _set_blocks(lane, mem.at, blocks)
    A.sel = ("block", lane, mem.at, arr.block_at(blocks, first))
    _draw()
    _draw_inspector()


# ----------------------------------------------------------------------- drawing

LANE_COLOURS = {arr.FG: (150, 210, 240, 255), arr.BG: (230, 210, 140, 255), "master": (220, 160, 240, 255),
                "montage": (200, 200, 200, 255)}


def _strips(specs: list, y0: float, y1: float) -> None:
    """Thin lines for FX with their own bars (the whole song when none), one under another."""
    for j, spec in enumerate(specs):
        y = y0 + 3 + 4 * j
        if y > y1:
            break
        r, g, b = arr.colour(arr.label_of(spec))
        alpha = 230 if spec.get("enabled", True) is not False else 70
        wins = arr.windows_of(spec) or [arr.Block(1, _bars())]
        for w in wins:
            dpg.draw_rectangle([_x_of(w.first), y], [_x_of(w.last + 1), y + 2], fill=(r, g, b, alpha),
                               color=(0, 0, 0, 0), parent="arr_canvas")


def _short(names: list, room: int) -> str:
    text = ", ".join(names)
    return text if len(text) <= room else text[:room - 3] + "..."


def _draw() -> None:
    if not dpg.does_item_exist("arr_canvas"):
        return
    dpg.delete_item("arr_canvas", children_only=True)
    n, bw = _bars(), _bw()
    h = _row_y(len(A.rows)) + 8
    dpg.configure_item("arr_canvas", height=max(120, int(h)))
    dpg.draw_rectangle([0, 0], [CANVAS_W, h], fill=(18, 20, 26, 255), color=(18, 20, 26, 255),
                       parent="arr_canvas")
    last = min(n, A.view0 + _shown() - 1)
    for b in range(A.view0, last + 1):                   # the bar ruler
        x = _x_of(b)
        major = (b - 1) % 4 == 0
        dpg.draw_line([x, RULER_H - (10 if major else 4)], [x, h], parent="arr_canvas",
                      color=(70, 76, 95, 160) if major else (45, 50, 62, 120))
        if major or bw > 22:
            dpg.draw_text([x + 2, 4], str(b), size=12, color=(170, 175, 190, 255), parent="arr_canvas")
    for b, v in enumerate(_density(n), 1):               # how busy the MIDI is, bar by bar
        if not A.view0 <= b <= last:
            continue
        c = int(40 + 180 * v)
        dpg.draw_rectangle([_x_of(b), RULER_H], [_x_of(b + 1), RULER_H + DENSITY_H],
                           fill=(c, int(c * 0.8), 60, 255), color=(0, 0, 0, 0), parent="arr_canvas")
    for r, row in enumerate(A.rows):
        y = _row_y(r)
        sel = A.sel or (None,)
        chosen = (sel[0] == row["kind"] if row["kind"] != "lane"
                  else sel[0] in ("lane", "block") and sel[1] == row["lane"])
        dpg.draw_rectangle([0, y], [CANVAS_W, y + LANE_H - 2],
                           fill=(34, 38, 50, 255) if chosen else (26, 29, 38, 255), color=(0, 0, 0, 0),
                           parent="arr_canvas")
        if row["kind"] == "lane" and row["sub"] == 0:
            dpg.draw_line([0, y - 1], [CANVAS_W, y - 1], color=(90, 96, 120, 255), parent="arr_canvas")
        if row["kind"] == "master":
            _strips([it[2] for it in _master_items()], y, y + LANE_H - 6)
            continue
        if row["kind"] == "montage":
            for w in arr.windows_of(_montage_item()[2]) or [arr.Block(1, n)]:
                dpg.draw_rectangle([_x_of(w.first), y + LANE_H - 8], [_x_of(w.last + 1), y + LANE_H - 5],
                                   fill=(120, 120, 130, 160), color=(0, 0, 0, 0), parent="arr_canvas")
            continue
        lane = row["lane"]
        mems = A.members.get(lane, [])
        rows_here = max([m.row for m in mems], default=-1) + 1
        if row["sub"] == rows_here:                      # the lane's empty row: its FX, and room
            _strips([f for f in _lane_item(lane)[2].get("fx") or [] if isinstance(f, dict)], y + 14, y + LANE_H - 4)
            continue
        for mem in mems:
            if mem.row != row["sub"]:
                continue
            _draw_member(lane, mem, y)
    _draw_names()
    dpg.draw_rectangle([0, 0], [0, 0], fill=(255, 255, 255, 60), color=(255, 255, 255, 200),
                       tag="arr_ghost", show=False, parent="arr_canvas")


def _draw_names() -> None:
    """The names' column, over whatever a zoomed view runs under it."""
    dpg.draw_rectangle([0, 0], [LABEL_W - 1, RULER_H + DENSITY_H], fill=(18, 20, 26, 255),
                       color=(18, 20, 26, 255), parent="arr_canvas")
    for r, row in enumerate(A.rows):
        y = _row_y(r)
        sel = A.sel or (None,)
        chosen = (sel[0] == row["kind"] if row["kind"] != "lane"
                  else sel[0] in ("lane", "block") and sel[1] == row["lane"])
        dpg.draw_rectangle([0, y], [LABEL_W - 1, y + LANE_H - 2], fill=(34, 38, 50, 255) if chosen else
                           (26, 29, 38, 255), color=(0, 0, 0, 0), parent="arr_canvas")
        colour = LANE_COLOURS[row.get("lane") or row["kind"]]
        grey = (150, 150, 160, 255)
        if row["kind"] == "master":
            items = [it[2] for it in _master_items()]
            dpg.draw_text([6, y + 4], f"MASTER ({len(items)})", size=13, color=colour, parent="arr_canvas")
            dpg.draw_text([6, y + 18], _short([arr.label_of(s) for s in items], 22) or "over everything",
                          size=10, color=grey, parent="arr_canvas")
        elif row["kind"] == "montage":
            dpg.draw_text([6, y + 4], "CLIPS", size=13, color=colour, parent="arr_canvas")
            dpg.draw_text([6, y + 18], "the montage, under the BG", size=10, color=grey, parent="arr_canvas")
        else:
            lane = row["lane"]
            if row["sub"] == 0:
                dpg.draw_line([0, y - 1], [LABEL_W, y - 1], color=(90, 96, 120, 255), parent="arr_canvas")
                dpg.draw_text([6, y + 4], lane, size=15, color=colour, parent="arr_canvas")
                dpg.draw_text([30, y + 6], "foreground" if lane == arr.FG else "background", size=10,
                              color=grey, parent="arr_canvas")
            rows_here = max([m.row for m in A.members.get(lane, [])], default=-1) + 1
            if row["sub"] == rows_here:
                fx = [f for f in _lane_item(lane)[2].get("fx") or [] if isinstance(f, dict)]
                dpg.draw_text([6, y + 4], ("FX: " + _short([arr.label_of(f) for f in fx], 17)) if fx else
                              "drag here: new", size=10, color=(200, 160, 230, 255) if fx else (110, 110, 125, 255),
                              parent="arr_canvas")


def _draw_member(lane: str, mem, y: float) -> None:
    n = _bars()
    fx = [f for f in mem.spec.get("fx") or [] if isinstance(f, dict)]
    tag = ("GEN " if mem.kind == arr.GENERATOR else "") + mem.label
    if mem.always:                                       # no blocks: the whole song
        dpg.draw_rectangle([_x_of(1), y + LANE_H - 8], [_x_of(n + 1), y + LANE_H - 5],
                           fill=(120, 120, 130, 160), color=(0, 0, 0, 0), parent="arr_canvas")
        dpg.draw_text([max(_x_of(1), LABEL_W) + 4, y + 4], f"{tag} (the whole song)"[:60], size=11,
                      color=(200, 200, 210, 255), parent="arr_canvas")
    if mem.cued:
        dpg.draw_text([max(_x_of(1), LABEL_W) + 4, y + 2], f"{tag} - cues (REAPER / s)"[:60], size=10,
                      color=(170, 170, 180, 255), parent="arr_canvas")
        for t0, t1 in A.cues.get(_cue_key(mem.spec), []):  # read-only cues, in grey
            xa, xb = _x_of(_time_to_bar(t0)), _x_of(_time_to_bar(t1))
            dpg.draw_rectangle([xa, y + LANE_H - 14], [max(xa + 1, xb), y + LANE_H - 5],
                               fill=(140, 140, 150, 200), color=(0, 0, 0, 0), parent="arr_canvas")
    for i, blk in enumerate(mem.blocks):
        xa, xb = _x_of(blk.first), _x_of(blk.last + 1)
        r, g, bb = arr.colour(blk.file or mem.spec.get("file") or mem.label)
        if not mem.enabled:
            r, g, bb = (r + 240) // 3, (g + 240) // 3, (bb + 240) // 3
        sel = A.sel == ("block", lane, mem.at, i)
        dpg.draw_rectangle([xa + 1, y + 3], [xb - 1, y + LANE_H - 4], fill=(r, g, bb, 230),
                           color=(255, 255, 255, 255) if sel else (10, 10, 14, 255),
                           thickness=2 if sel else 1, parent="arr_canvas")
        stem = Path(blk.file).stem if blk.file else mem.label
        folder = Path(blk.file).parent.name if blk.file else ""
        label = folder if folder and stem in ("final", "clip", "video") else stem[-14:]
        if blk.start:
            label += f" @{blk.start:g}s"
        if fx:
            label += f" +{len(fx)}fx"
        xt = max(xa, LABEL_W)                          # its name where it shows
        room = int((min(xb, CANVAS_W) - xt - 6) / 6.5)
        if room >= 3:
            dpg.draw_text([xt + 4, y + 6], label[:room], size=11, color=(10, 10, 14, 255), parent="arr_canvas")
        if fx:
            dpg.draw_rectangle([xa + 2, y + LANE_H - 8], [xb - 2, y + LANE_H - 6], fill=(200, 120, 230, 255),
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

PREVIEW_BARS = 8       # the most a block's preview renders (it is held in memory)


def preview_bars(first: int, last: int) -> None:
    """The studio's preview of these bars, with the arrangement as it is now (saved or
    not), in a window of its own."""
    from core.scene_doc import plain
    g = A.host.grid()
    db = getattr(g, "downbeats", None) if g is not None else None
    if not hasattr(A.host, "preview") or db is None or not len(db):
        _status("the preview needs the studio with the project loaded")
        return
    bars = max(1, min(last - first + 1, PREVIEW_BARS))
    at = float(db[min(first, len(db)) - 1])
    if not A.host.preview(at, bars, [plain(s) for s in _specs()]):
        _status("the studio is busy rendering, or the project is not loaded")
        return
    tex, w, h = A.host.preview_texture()
    if not dpg.does_item_exist("arr_prev"):
        with dpg.window(tag="arr_prev", label="Arrange preview", width=w + 16, height=h + 40, pos=(400, 120)):
            dpg.add_image(tex, width=w, height=h)
    dpg.configure_item("arr_prev", label=f"Arrange preview - bars {first}-{first + bars - 1}")
    dpg.show_item("arr_prev")
    dpg.focus_item("arr_prev")
    _status(f"previewing bars {first}-{first + bars - 1} (rendering...)")


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
                             callback=lambda s, v, here=here: _toggle(uid, here))
            dpg.add_text(f"{j + 1}. {arr.label_of(f)}", color=(230, 230, 240))
            dpg.add_text("bars")
            dpg.add_input_text(default_value=_bars_text(f), width=110, hint="all", on_enter=True,
                               callback=lambda s, v, here=here: _set_fx_bars(uid, here, v))
            dpg.add_button(label="plugin", callback=lambda s, a, here=here: open_plugin(uid, here))
            dpg.add_button(label="^", callback=lambda s, a, j=j: _swap(uid, path, j, j - 1))
            dpg.add_button(label="v", callback=lambda s, a, j=j: _swap(uid, path, j, j + 1))
            dpg.add_button(label="x", callback=lambda s, a, j=j: _edit(
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
    if sel is None or A.doc is None:
        dpg.add_text("pick a clip (Clips) or a Generator (Generators), then drag across the BG or the FG to "
                     "put it there · drag a block to move it, its ends to resize it · right-click for FX "
                     "(on a block, a lane, the master) · click a lane's name for its FX · Ctrl+Z undo",
                     parent=p, color=(130, 130, 150), wrap=700)
        return
    if sel[0] == "master":
        _draw_master(p)
    elif sel[0] == "montage":
        dpg.add_text("CLIPS: the clip montage (the project's clips folder, cut by the triggers). It shows "
                     "wherever the BG has nothing; the BG's FX work on it too.", parent=p, wrap=700,
                     color=(150, 150, 160))
    elif sel[0] == "lane":
        _draw_lane(p, sel[1])
    elif sel[0] == "block":
        mem = _member(sel[1], sel[2])
        if mem is None:
            A.sel = ("lane", sel[1])
            _draw_lane(p, sel[1])
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
    dpg.add_text("MASTER: over everything, in order (the last on top)", parent=p, color=LANE_COLOURS["master"][:3])
    with dpg.group(horizontal=True, parent=p):
        dpg.add_button(label="Add FX...", callback=lambda: _menu_at(lambda: _catalog_menu("Add FX", "fx", add_master)))
    for it in _master_items():
        spec, uid = it[2], it[3]
        kind = arr.kind_of(spec)
        with dpg.group(horizontal=True, parent=p):
            dpg.add_checkbox(default_value=spec.get("enabled", True) is not False,
                             callback=lambda s, v, uid=uid: _toggle(uid, ()))
            dpg.add_text(("" if kind == arr.FX else "GEN " if kind == arr.GENERATOR else "CLIP ")
                         + arr.label_of(spec), color=(230, 230, 240))
            dpg.add_text("bars")
            dpg.add_input_text(default_value=_bars_text(spec), width=110, hint="all", on_enter=True,
                               callback=lambda s, v, uid=uid: _set_fx_bars(uid, (), v))
            if kind != arr.CLIP:
                dpg.add_button(label="plugin", callback=lambda s, a, uid=uid: open_plugin(uid, ()))
            dpg.add_button(label="^", callback=lambda s, a, uid=uid: _move_master(uid, -1))
            dpg.add_button(label="v", callback=lambda s, a, uid=uid: _move_master(uid, 1))
            dpg.add_button(label="x", callback=lambda s, a, uid=uid: _delete_master(uid))


def _draw_block(p, lane: str, mem, bk) -> None:
    uid = _lane_item(lane)[3]
    spec = mem.spec
    what = "Generator" if mem.kind == arr.GENERATOR else "clip" if lane == arr.BG else "character"
    dpg.add_text(f"{lane} · {what} {mem.label}" + ("" if mem.enabled else " (off)"), parent=p,
                 color=LANE_COLOURS[lane][:3])
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
        dpg.add_checkbox(label="on", default_value=mem.enabled, callback=lambda: _toggle(uid, ("layers", mem.at)))
        if mem.kind != arr.CLIP:
            dpg.add_button(label="Open plugin...", callback=lambda: open_plugin(uid, ("layers", mem.at)))
        if bk is not None:
            dpg.add_button(label="delete this block", callback=lambda: _delete_block(lane, mem.at, bk))
        dpg.add_button(label=f"delete {mem.label[:20]} (all its blocks)", callback=lambda: _remove_member(lane, mem.at))
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
    _draw_fx_list(p, uid, ("layers", mem.at, "fx"), f"FX on {mem.label[:24]} only, in order:")


def _draw_placement(p, lane: str, mem) -> None:
    """A character's place in the frame and its cutout."""
    spec = mem.spec

    def put(key, cast=float):
        return lambda s, v: _set_member(lane, mem.at, {**_member(lane, mem.at).spec, key: cast(v)})
    with dpg.group(horizontal=True, parent=p):
        for key, default in (("x", 0.5), ("y", 0.5), ("scale", 1.0)):
            v = _num(spec, key, default)
            dpg.add_text(key)
            if v is None:
                dpg.add_text("(follows the music)", color=(200, 160, 100))
            else:
                dpg.add_input_float(default_value=v, width=80, step=0.05, format="%.2f", callback=put(key),
                                    on_enter=True)
        dpg.add_text("fit")
        dpg.add_combo(["contain", "cover"], default_value=spec.get("fit", "contain"), width=80,
                      callback=put("fit", str))
        dpg.add_checkbox(label="flip", default_value=bool(spec.get("flip", False)), callback=put("flip", bool))
        dpg.add_checkbox(label="loop", default_value=bool(spec.get("loop", False)), callback=put("loop", bool))
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
                    if _is_trigger(prm.name, value):         # who plays: a track, notes, a stem
                        dpg.add_button(label="Who plays", callback=lambda s, a, path=path: _link(
                            uid, path, 0.0, "trigger"))
                    elif prm.knob or (isinstance(value, dict) and form_of(plain(value)) is not None):
                        dpg.add_button(label="Link" if not isinstance(value, dict) else "Edit",
                                       callback=lambda s, a, path=path, d=prm.default: _link(uid, path, d))
                    if value is not None:
                        dpg.add_button(label="x", callback=lambda s, a, path=path: set_param(uid, path, None))
                        with dpg.tooltip(dpg.last_item()):
                            dpg.add_text("back to its default")


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
    return value is None and name in ("trigger", "hits")


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


def _show_content() -> None:
    from core import catalog
    if not dpg.does_item_exist("arr_content"):
        return
    c = A.content
    text = ("drag on BG or FG puts: " + (f"clip {Path(c[1]).name} @{c[2]:g}s" if c[0] == "clip"
                                         else f"Generator {catalog.by_key(c[1]).label}")) if c else \
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
    A.want_thumb = (A.clip, A.start, 0.0)
    A.content = ("clip", A.clip, A.start)
    _show_content()


def on_from(sender, value) -> None:
    A.start = float(value)
    if A.clip:
        A.want_thumb = (A.clip, A.start, time.time())
        A.content = ("clip", A.clip, A.start)
        _show_content()


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
            item = _lane_item(arr.FG)                        # the characters of this clip get cut out
            if item is not None:
                def cut(layers):
                    return [{**m, "cutout": {"track": out, "feather": 2}}
                            if isinstance(m, dict) and m.get("source") == "footage" and not m.get("cutout")
                            and Path(str(m.get("file", ""))).name == Path(clip).name else m
                            for m in layers or []]
                A.pending = (item[3], cut)                     # done on the main thread (tick)
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
    if e.kind == catalog.GENERATOR:                         # a Generator is put in a lane by dragging
        A.content = ("gen", key)
        _show_content()
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
    if e.kind == catalog.GENERATOR:
        dpg.add_text("drag across the BG or the FG to put it there", parent=tag, color=(150, 230, 160), wrap=300)
    else:
        dpg.add_button(label="Add to the master", parent=tag, callback=lambda: add_master(e.key))
        dpg.add_text("or right-click a block or a lane: its FX", parent=tag, color=(120, 120, 140), wrap=300)
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
    if A.pending is not None:                           # a worker's edit (SAM2's cutouts)
        (uid, fn), A.pending = A.pending, None
        _edit(uid, ("layers",), fn)
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
    set_view()                                  # the song may be another: the view fits it
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
    A.zoom, A.view0 = 1.0, 1
    with dpg.texture_registry():
        dpg.add_dynamic_texture(THUMB, THUMB, [0.0, 0.0, 0.0, 1.0] * THUMB * THUMB, tag="arr_tex")
    dpg.add_file_dialog(show=False, directory_selector=True, callback=_pick_folder, tag="dlg_arr_folder",
                        width=620, height=420)
    with dpg.window(label="Arrange - the background, the foreground and the master, bar by bar",
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
                    dpg.add_text("drag across the BG or the FG to put it there (in the FG it is cut out "
                                 "by its mask track)", wrap=310, color=(150, 190, 150))
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
                dpg.add_text("Ctrl+wheel zooms, Shift+wheel moves along · the strip under the bars: how busy "
                             "the MIDI is", color=(120, 120, 140))
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
        dpg.add_mouse_wheel_handler(callback=on_wheel)
        dpg.add_key_press_handler(dpg.mvKey_Y, callback=on_key_y)

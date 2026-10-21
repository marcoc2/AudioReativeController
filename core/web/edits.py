"""The Arrange's edits, as the Dear PyGui window makes them (arc_arrange.py), on the session.

A thing in the arrangement is named by its *path*: the lane it sits in, then the keys down
to it. ``["BG", "layers", 2]`` is the BG's third member, ``["BG", "layers", 2, "fx", 0]``
that member's first FX, ``["BG", "fx", 1]`` the second FX on the whole BG, ``["BG"]`` the
BG itself, ``["master", 7]`` the eighth top-level layer (one over both lanes), ``["clips"]``
the montage. Each edit is one undo step and answers with the lanes as they now are (and,
when the edit moved what was chosen, ``select``: where it is now).

The page names lanes by id: bg, fg (members), bgfx, fgfx (FX on a whole lane), master.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from core.web.util import copy_keeping_signals, resolve, shown_path

TRACKS = {"bg": ("BG", "layers"), "fg": ("FG", "layers"), "bgfx": ("BG", "fx"), "fgfx": ("FG", "fx"),
          "master": ("master",)}
FX_LANES = ("bgfx", "fgfx", "master")
SAME_KIND = (("bg", "fg"), ("bgfx", "fgfx"))     # where a block may be dragged to
TITLES = {"bg": "BG", "fg": "FG", "bgfx": "FX on the whole BG", "fgfx": "FX on the whole FG", "master": "MASTER"}
SETTABLE = {"name", "blend", "opacity", "enabled", "fit", "flip", "loop", "spill", "x", "y", "scale", "cutout"}
VIDEO_KEYS = {"clip_per_bar": bool, "bars_per_clip": int, "clip_order": str, "clip_end": str, "seed": int,
              "triggers": dict}         # a trigger's gravity is in it (the composer warps the speed by trigger)
VIDEO_CHOICES = {"clip_order": ("sequential", "random", "shuffle"), "clip_end": ("bounce", "loop")}


class Edits:
    """Mixed into the Session: it brings items, specs(), groups(), lock, undo/redo, bars()."""

    # -- paths -------------------------------------------------------------------------
    def _top(self, path) -> int:
        """The top-level item a path starts in."""
        head = path[0]
        if head == "master":
            return int(path[1])
        if head == "clips":
            return next(i for i, s in enumerate(self.specs()) if s.get("source", "clips") == "clips")
        return self.groups()[head]

    def _rest(self, path) -> list:
        return list(path[2:]) if path[0] == "master" else list(path[1:])

    def get(self, path):
        node = self.items[self._top(path)][2]
        for k in self._rest(path):
            node = node[int(k)] if isinstance(node, list) else node[k]
        return node

    def _put(self, path, new) -> None:
        """What is at ``path`` becomes ``new`` (None: it goes, from its list or its mapping)."""
        top, rest = self._top(path), self._rest(path)
        if not rest:
            if new is None:
                del self.items[top]
            else:
                self.items[top][2] = new
            return
        holder = self.items[top][2]
        for k in rest[:-1]:
            holder = holder[int(k)] if isinstance(holder, list) else holder[k]
        last = rest[-1]
        if isinstance(holder, list):
            if new is None:
                del holder[int(last)]
            else:
                holder[int(last)] = new
        elif new is None:
            holder.pop(last, None)
        else:
            holder[last] = new

    def _edit(self, path, fn) -> bool:
        """``fn(a copy of what is at path)`` -> its new value (None: it goes). One undo step;
        False when nothing changed."""
        from core.scene_doc import plain
        old = self.get(path)
        new = fn(copy_keeping_signals(old))
        if new is not None and plain(new) == plain(old):
            return False
        self._push()
        self._put(path, new)
        return True

    def _answer(self, select=None, note: str = "", montage: bool = False) -> dict:
        out = self._edited(montage=montage)
        if select is not None:
            out["select"] = select
        if note:
            out["note"] = note
        return out

    # -- tracks, as the page shows them --------------------------------------------------
    def lane_of(self, path) -> str:
        if path[0] == "master":
            return "master"
        return {("BG", "layers"): "bg", ("FG", "layers"): "fg", ("BG", "fx"): "bgfx",
                ("FG", "fx"): "fgfx"}[(path[0], path[1])]

    def track(self, lane: str) -> list:
        """(path, spec) of each member of a track, in stack order."""
        if lane == "master":
            gi = self.groups()
            after = (max(gi.values()) + 1) if gi else 0
            return [(["master", i], s) for i, s in enumerate(self.specs()) if i >= after
                    and isinstance(s, dict) and s.get("source", "clips") not in ("clips", "group")]
        name, key = TRACKS[lane]
        gi = self.groups()
        if name not in gi:
            return []
        return [([name, key, i], s) for i, s in enumerate(self.specs()[gi[name]].get(key) or []) if isinstance(s, dict)]

    def packed(self, lane: str) -> tuple:
        """({place in the track: sub-row}, rows): the track's members packed as the Arrange
        packs them (``arrangement.pack``), the cued ones by their cues once they are read."""
        from core import arrangement as arr
        from core.web.util import cue_key
        mems = arr.members_of({"layers": [s for _, s in self.track(lane)]})
        cue_bars = {}
        for m in mems:
            got = self.cues.get(cue_key(m.spec)) if m.cued else None
            if isinstance(got, list) and got:
                cue_bars[m.at] = self.cue_bars(got)
        n = arr.pack(mems, len(self.bars()) - 1, cue_bars)
        return {m.at: m.row for m in mems}, n

    def _blocks(self, spec: dict) -> list:
        """A member's blocks as the page numbers them (a clip's, else its windows)."""
        from core import arrangement as arr
        kind = arr.kind_of(spec)
        blocks = arr.blocks_of(spec) if kind == arr.CLIP else []
        return blocks or arr.windows_of(spec)

    def _shaped(self, spec: dict, blocks: list) -> dict:
        from core import arrangement as arr
        return (arr.with_blocks if arr.kind_of(spec) == arr.CLIP else arr.with_windows)(spec, blocks)

    def bar_seconds(self) -> float:
        db = self.bars()
        steps = sorted(b - a for a, b in zip(db, db[1:]))
        return steps[len(steps) // 2] if steps else 2.0

    def _insert(self, lane: str, spec: dict, sub, first: int, last: int) -> list:
        """``spec`` into the track ``lane`` on the sub-row ``sub`` (None: the topmost free one),
        drawn under what is above it there over bars first..last (the master: before a
        trailing ``meters`` read-out); -> its path."""
        from core import arrangement as arr
        spec = dict(spec)
        if sub is not None:
            spec[arr.ROW_KEY] = int(sub)
        if lane == "master":
            at = len(self.items)
            while at > 0 and self.items[at - 1][2].get("source") == "meters":
                at -= 1
            if sub is not None:
                rows, _ = self.packed("master")
                tr = self.track("master")
                above = [tr[k][0][1] for k, r in rows.items() if r < sub
                         and any(a <= last and first <= b for a, b in self._spans_of(tr[k][1]))]
                at = min(above, default=at)
            self.items.insert(at, [None, None, spec])
            return ["master", at]
        name, key = TRACKS[lane]
        if name not in self.groups():
            gi = self.groups()
            at = gi.get("FG", len(self.items)) if name == "BG" else gi.get("BG", -1) + 1
            self.items.insert(at, [None, None, arr.group(name)])
        group = self.specs()[self.groups()[name]]
        lst = group.setdefault(key, [])
        at = len(lst)
        if sub is not None and key == "layers":
            rows, _ = self.packed(lane)
            above = [k for k, r in rows.items() if r < sub
                     and any(a <= last and first <= b for a, b in self._spans_of(lst[k]))]
            at = min(above, default=at)
        lst.insert(at, spec)
        return [name, key, at]

    def _spans_of(self, spec: dict) -> list:
        from core.web.util import cue_key
        got = self.cues.get(cue_key(spec)) if spec.get("cues") is not None else None
        if isinstance(got, list) and got:
            return self.cue_bars(got)
        blocks = self._blocks(spec)
        return [(b.first, b.last) for b in blocks] if blocks else [(1, len(self.bars()) - 1)]

    # -- blocks: move, resize, new, delete --------------------------------------------
    def move(self, path, k: int, mode: str, first: int, last: int, lane: Optional[str] = None,
             row: Optional[int] = None) -> dict:
        """Block ``k`` of the member at ``path`` dragged (``mode`` move) or one of its ends
        (first, last) to bars first..last; dragged onto another sub-row or the other lane of
        its kind (``lane``, ``row``): there now, that block alone a member of its own when it
        had others, a clip recast for its lane. A resized start moves its ``from`` so the
        picture keeps its place in time."""
        from core import arrangement as arr
        with self.lock:
            spec = self.get(path)
            if spec.get("cues") is not None:
                return {"error": "a cued member plays where its cues say (REAPER): hide bars of it instead"}
            src = self.lane_of(path)
            n = len(self.bars()) - 1
            blocks = self._blocks(spec) or [arr.Block(1, n)]
            k = max(0, min(len(blocks) - 1, int(k)))
            first, last = max(1, min(int(first), int(last))), min(n, max(int(first), int(last)))
            b = blocks[k]
            rows, _ = self.packed(src)
            here = rows.get(next(i for i, (p, _) in enumerate(self.track(src)) if p == list(path)), 0)
            if mode == "move" and lane is not None and (lane, row) != (src, here):
                if not any(src in pair and lane in pair for pair in SAME_KIND) and lane != src:
                    return {"error": f"a {TITLES[src]} block goes to {TITLES[src]} rows only"}
                return self._move_to_row(path, spec, blocks, k, first, last, lane, row)
            start = b.start
            if mode == "first":
                start = max(0.0, b.start + (first - b.first) * self.bar_seconds() * b.rate)
            moved = arr.Block(first, last, round(start, 3), b.rate, b.file)
            rest = [x for i, x in enumerate(blocks) if i != k]
            placed = arr.place(rest, moved, self.bar_seconds())
            if not self._edit(path, lambda s: self._shaped(s, placed)):
                return self.state()
            return self._answer(select={"ref": list(path), "k": arr.block_at(placed, first)})

    def _move_to_row(self, path, spec, blocks, k, first, last, lane, sub) -> dict:
        from core import arrangement as arr
        src = self.lane_of(path)
        b = blocks[k]
        moved = arr.Block(first, last, b.start, b.rate, b.file)
        alone = self._shaped({x: v for x, v in copy_keeping_signals(spec).items() if x != arr.ROW_KEY}, [moved])
        rest = [x for j, x in enumerate(blocks) if j != k]
        self._push()                                       # one undo step for all of it
        if lane == src and not rest:                       # its own lane: another sub-row
            self._put(path, {**alone, arr.ROW_KEY: int(sub)})
            return self._answer(select={"ref": list(path), "k": 0})
        if lane != src and src in ("bg", "fg"):
            alone = arr.recast(alone, "FG" if lane == "fg" else "BG")
        if rest:
            self._put(path, self._shaped(copy_keeping_signals(spec), rest))
        else:
            self._put(path, None)
        at = self._insert(lane, alone, sub, first, last)
        return self._answer(select={"ref": at, "k": 0},
                            note=f"{arr.label_of(alone)} -> {TITLES[lane]}, row {int(sub) + 1}, bars {first}-{last}"
                            + (" (Ctrl+Z puts it back)" if lane != src else ""))

    def new_member(self, lane: str, what: str, first: int, last: int, row: Optional[int] = None,
                   start: float = 0.0) -> dict:
        """What the browser picked (a catalog key, or a clip's path) new on the track ``lane``
        over bars first..last; on the sub-row ``row`` (None: the topmost free one)."""
        from core import arrangement as arr
        from core import catalog
        with self.lock:
            n = len(self.bars()) - 1
            first, last = max(1, min(int(first), int(last))), min(n, max(int(first), int(last)))
            e = catalog.by_key(what)
            if lane in FX_LANES:
                if e is None or e.kind != catalog.FX:
                    return {"error": "an FX track takes an FX: pick one in FX"}
                spec = {**e.spec(), "bars": [[first, last]]}
            elif e is not None:
                if e.kind != catalog.GENERATOR:
                    return {"error": f"{e.label} is an FX: drop it on an FX track, or on a block (Add FX)"}
                spec = {**e.spec(), "bars": [[first, last]]}
                d = getattr(self, "browser_folder", None) or self.clips_dir()       # the browser's folder
                if d is not None and any(p.name == "clips_dir" for p in e.params):     # its own clips: these
                    spec["clips_dir"] = shown_path(d)
            else:
                clip = resolve(what)
                if not clip.is_file():
                    return {"error": f"no clip at {what}"}
                block = arr.Block(first, last, round(float(start or 0.0), 3))
                if lane == "bg":
                    spec = arr.with_blocks({"source": "footage", "role": "background", "fit": "cover", "loop": True,
                                            "file": shown_path(clip)}, [block])
                else:
                    spec = arr.new_foreground(shown_path(clip), [block])
            self._push()
            at = self._insert(lane, spec, row, first, last)
            note = f"{arr.label_of(spec)} in {TITLES[lane]}, bars {first}-{last}"
            if lane == "fg" and spec.get("source") == "footage" and not spec.get("cutout"):
                note += " - no mask track yet: cut it out (SAM2) to make it a character"
            return self._answer(select={"ref": at, "k": 0}, note=note)

    def add(self, lane: str, what: str, first: int, last: int) -> dict:
        return self.new_member(lane, what, first, last)

    def set_block(self, path, k: int, changes: dict) -> dict:
        """A block's bars, its ``from`` (s), its speed or its clip (``file``)."""
        from core import arrangement as arr
        with self.lock:
            blocks = self._blocks(self.get(path))
            if not blocks:
                return {"error": "it has no blocks: it plays the whole song (give it bars)"}
            k = max(0, min(len(blocks) - 1, int(k)))
            old = blocks[k]
            new = arr.Block(old.first, old.last, old.start, old.rate, old.file)
            for key, v in (changes or {}).items():
                if key in ("first", "last"):
                    setattr(new, key, int(v))
                elif key in ("start", "rate"):
                    setattr(new, key, float(v))
                elif key == "file":
                    new.file = shown_path(resolve(v)) if v else None
            if new.last < new.first or new.rate <= 0:
                return {"error": "bars run first to last; the speed is above 0"}
            placed = arr.place([x for i, x in enumerate(blocks) if i != k], new, self.bar_seconds())
            if not self._edit(path, lambda s: self._shaped(s, placed)):
                return self.state()
            return self._answer(select={"ref": list(path), "k": arr.block_at(placed, new.first)})

    def delete_block(self, path, k: int) -> dict:
        from core import arrangement as arr
        with self.lock:
            spec = self.get(path)
            blocks = self._blocks(spec)
            if spec.get("cues") is not None or len(blocks) <= 1:
                return self.remove(path)
            rest = arr.remove(blocks, int(k))
            self._edit(path, lambda s: self._shaped(s, rest))
            return self._answer(select={"ref": list(path), "k": None})

    def spans(self, path, spans: list) -> dict:
        """A member's blocks written whole (``[{first, last, from?, rate?, file?}]``); none
        removes it."""
        from core import arrangement as arr
        with self.lock:
            if not spans:
                return self.remove(path)
            blocks = [arr.Block(int(s["first"]), int(s["last"]), float(s.get("from") or 0.0),
                                float(s.get("rate") or 1.0), s.get("file") or None) for s in spans]
            self._edit(path, lambda s: self._shaped(s, blocks))
            return self._answer()

    def relane(self, path, lane: str, spans: Optional[list] = None) -> dict:
        """A member moved whole to another track of its kind (BG <-> FG, a lane's FX tracks)."""
        from core import arrangement as arr
        with self.lock:
            src = self.lane_of(path)
            if lane == src:
                return self.state()
            if not any(src in pair and lane in pair for pair in SAME_KIND):
                return {"error": f"a {TITLES[src]} member goes to {TITLES[src]} only"}
            spec = copy_keeping_signals(self.get(path))
            spec.pop(arr.ROW_KEY, None)
            if spans:
                spec = self._shaped(spec, [arr.Block(int(s["first"]), int(s["last"]), float(s.get("from") or 0.0),
                                                     float(s.get("rate") or 1.0), s.get("file") or None) for s in spans])
            if src in ("bg", "fg"):
                spec = arr.recast(spec, "FG" if lane == "fg" else "BG")
            self._push()
            self._put(path, None)
            at = self._insert(lane, spec, None, 1, len(self.bars()) - 1)
            return self._answer(select={"ref": at, "k": 0})

    def remove(self, path) -> dict:
        with self.lock:
            if len(self._rest(path)) == 0 and path[0] != "master":
                return {"error": "a lane itself is not deleted"}
            self._push()
            self._put(path, None)
            lane = "master" if path[0] == "master" else path[0]
            return self._answer(select={"lane": lane})

    # -- a member's settings ----------------------------------------------------------
    def toggle(self, path) -> dict:
        with self.lock:
            def flip(s):
                s = dict(s)
                if s.get("enabled", True) is False:
                    s.pop("enabled")
                else:
                    s["enabled"] = False
                return s
            self._edit(path, flip)
            return self._answer()

    def set_layer(self, path, key: str, value) -> dict:
        """name, blend, opacity, enabled, fit, flip, loop, spill, x, y, scale, cutout: a value,
        or None / empty / False to take the key out (its default)."""
        if key not in SETTABLE:
            return {"error": f"{key}: not a layer setting here"}
        with self.lock:
            def put(s):
                s = dict(s)
                drop = value is None or value == "" or (key in ("spill", "flip", "loop") and value is False) \
                    or (key == "enabled" and value is True)
                if drop:
                    s.pop(key, None)
                else:
                    s[key] = value.strip() if isinstance(value, str) and key == "name" else value
                return s
            self._edit(path, put)
            return self._answer()

    def set_windows(self, path, text: str) -> dict:
        """A Generator's or FX's windows as text (9-12, 30-33; empty: whenever its owner plays)."""
        from core import arrangement as arr
        try:
            bars = arr.parse_bars(text)
        except ValueError:
            return {"error": f"bars {text!r}: write them like 9-12, 30-33"}
        with self.lock:
            if arr.kind_of(self.get(path)) == arr.CLIP:
                return {"error": "a clip's blocks are edited on its lane (or its block's bars)"}

            def put(s):
                s = {k: v for k, v in s.items() if k != "bars"}
                if bars:
                    s["bars"] = bars
                return s
            self._edit(path, put)
            return self._answer()

    def mute(self, path, first: int, last: int, hide: bool = True) -> dict:
        """A member hidden in bars first..last (``hide`` False: shown there again); it goes on
        underneath, in sync with its sound."""
        from core import arrangement as arr
        with self.lock:
            self._edit(path, lambda s: arr.with_mutes(s, int(first), int(last), bool(hide)))
            what = f"bar {first}" if first == last else f"bars {first}-{last}"
            return self._answer(note=f"hidden in {what} (its sound and sync go on)" if hide else f"shown again in {what}")

    def set_mutes(self, path, text: str) -> dict:
        """The "hidden in bars" field: 9-12, 30-33 (empty: shown wherever it plays)."""
        from core import arrangement as arr
        try:
            bars = arr.parse_bars(text)
        except ValueError:
            return {"error": f"bars {text!r}: write them like 9-12, 30-33"}
        with self.lock:
            def put(s):
                s = {k: v for k, v in s.items() if k != "mute"}
                if bars:
                    s["mute"] = arr.bar_spans({b for a, z in bars for b in range(a, z + 1)})
                return s
            self._edit(path, put)
            return self._answer()

    # -- FX: a block's, a lane's, the master's ---------------------------------------------
    def add_fx(self, owner, key: str, bars: Optional[list] = None) -> dict:
        """An FX from the catalog at the end of ``owner``'s chain: a member's own FX, a lane's
        (``["BG"]``), the master's (``["master"]``: under a trailing ``meters``)."""
        from core import arrangement as arr
        from core import catalog
        e = catalog.by_key(key)
        if e is None or e.kind != catalog.FX:
            return {"error": f"{key}: not an FX"}
        spec = e.spec()
        if bars:
            spec["bars"] = [[int(a), int(b)] for a, b in bars]
        with self.lock:
            if list(owner) == ["master"]:
                self._push()
                at = self._insert("master", spec, None, 1, 1)
                return self._answer(select={"ref": at, "k": 0 if bars else None},
                                    note=f"{e.label} on the master" + ("" if bars else ", all the song: drag its ends, or give it bars"))
            if list(owner) in (["BG"], ["FG"]):
                if owner[0] not in self.groups():
                    return {"error": f"there is no {owner[0]} lane"}
                self._push()
                at = self._insert("bgfx" if owner[0] == "BG" else "fgfx", spec, None, 1, 1)
                return self._answer(select={"ref": at, "k": 0 if bars else None},
                                    note=f"{e.label} on the whole {owner[0]}" + ("" if bars else ", all the song: drag its ends, or give it bars"))
            self._edit(owner, lambda s: {**s, "fx": list(s.get("fx") or []) + [spec]})
            label = arr.label_of(self.get(owner))
            return self._answer(note=f"{e.label} on {label} only: it plays all the time {label} does - give it bars "
                                     "in its list")

    def reorder(self, path, d: int) -> dict:
        """An FX (or any layer of a list) earlier (-1) or later (+1) in its chain."""
        with self.lock:
            if path[0] == "master" and len(path) == 2:
                tr = [p[1] for p, _ in self.track("master")]
                k = tr.index(int(path[1]))
                j = k + int(d)
                if not 0 <= j < len(tr):
                    return self.state()
                self._push()
                a, b = tr[k], tr[j]
                self.items[a], self.items[b] = self.items[b], self.items[a]
                return self._answer(select={"ref": ["master", b], "k": None})
            parent, i = list(path[:-1]), int(path[-1])
            lst = self.get(parent)
            j = i + int(d)
            if not 0 <= j < len(lst):
                return self.state()

            def swap(ls):
                ls = list(ls)
                ls[i], ls[j] = ls[j], ls[i]
                return ls
            self._edit(parent, swap)
            return self._answer(select={"ref": parent + [j], "k": None})

    def move_fx(self, path, to: int) -> dict:
        """An FX dragged to place ``to`` in its list."""
        with self.lock:
            i = int(path[-1])
            d = int(to) - i
            if not d:
                return self.state()
            parent = list(path[:-1])

            def put(ls):
                ls = list(ls)
                x = ls.pop(i)
                ls.insert(max(0, min(len(ls), int(to))), x)
                return ls
            self._edit(parent, put)
            return self._answer()

    # -- copy and paste: music links, FX -------------------------------------------------
    def _fx_list(self, path) -> list:
        from core import arrangement as arr
        from core.scene_doc import plain
        spec = self.get(path)
        if self.lane_of(path) in FX_LANES or arr.kind_of(spec) == arr.FX:
            return [plain(spec)]
        return [plain(f) for f in spec.get("fx") or [] if isinstance(f, dict)]

    def copy_links(self, path) -> dict:
        import copy as _copy
        from core import arrangement as arr
        from core.scene_doc import plain
        with self.lock:
            spec = self.get(path)
            links = arr.links_of(spec)
            if not links:
                return {"error": f"{arr.label_of(spec)}: nothing in it follows the music"}
            self.board_links = {"from": arr.label_of(spec), "links": {p: _copy.deepcopy(plain(v)) for p, v in links.items()}}
            return {"note": f"copied {arr.label_of(spec)}'s music links ({', '.join(p[-1] for p in links)}): "
                            "right-click another block, Paste"}

    def copy_fx(self, path) -> dict:
        import copy as _copy
        from core import arrangement as arr
        with self.lock:
            spec = self.get(path)
            fx = self._fx_list(path)
            if not fx:
                return {"error": f"{arr.label_of(spec)} has no FX"}
            self.board_fx = {"from": arr.label_of(spec), "fx": _copy.deepcopy(fx), "file": spec.get("file")}
            return {"note": f"copied {arr.label_of(spec)}'s FX ({', '.join(arr.label_of(f) for f in fx)}): "
                            "right-click another block, Paste"}

    def paste_links(self, path) -> dict:
        import copy as _copy
        from core import arrangement as arr
        board = getattr(self, "board_links", None)
        if not board:
            return {"error": "no music links copied"}
        with self.lock:
            spec = self.get(path)
            put, left = [], []

            def paste(s):
                for p, v in board["links"].items():
                    if not arr.takes(s, p):
                        left.append(p)
                        continue
                    if p[0] == "uniforms":
                        s.setdefault("uniforms", {})[p[1]] = _copy.deepcopy(v)
                    else:
                        s[p[0]] = _copy.deepcopy(v)
                    put.append(p)
                return s
            self._edit(path, paste)
            names = ", ".join(p[-1] for p in put) or "nothing"
            return self._answer(note=f"{arr.label_of(spec)}: {names} now follow the music as on {board['from']}"
                                + (f" ({', '.join(p[-1] for p in left)}: not something it has)" if left else ""))

    def paste_fx(self, path) -> dict:
        from core import arrangement as arr
        board = getattr(self, "board_fx", None)
        if not board:
            return {"error": "no FX copied"}
        with self.lock:
            spec = self.get(path)
            fx, note = arr.pasted_fx(board["fx"], spec, board.get("file"))
            if not fx:
                return {"error": f"{arr.label_of(spec)}: nothing to paste here"}
            self._edit(path, lambda s: {**s, "fx": list(s.get("fx") or []) + fx})
            return self._answer(note=f"{arr.label_of(spec)}: + {', '.join(arr.label_of(f) for f in fx)} "
                                     f"from {board['from']}{note}")

    # -- the right click -----------------------------------------------------------------
    def menu(self, lane: str, path=None, k=None, bar: int = 1, span=None) -> list:
        """The right click's items: ``{label, call, args}`` (``$key`` in args: a catalog
        entry picked from ``catalog`` (fx | gen) submenus), ``{page, args}`` for what the page
        does itself, ``{head}`` and ``{sep}``."""
        from core import arrangement as arr
        with self.lock:
            n = len(self.bars()) - 1
            bar = max(1, min(n, int(bar)))
            to = min(n, bar + 3)
            out: list = []
            if path is not None:
                spec = self.get(path)
                label = arr.label_of(spec)
                kind = arr.kind_of(spec)
                blocks = self._blocks(spec)
                cued = spec.get("cues") is not None
                hit = k is not None and k >= 0 and k < len(blocks) and not cued
                out.append({"head": f"{TITLES[lane]} · {label}"})
                if lane not in FX_LANES:
                    out.append({"label": "Add FX to it", "catalog": "fx", "call": "add_fx", "args": [path, "$key"]})
                if kind != arr.CLIP:
                    out.append({"label": "Open plugin (its parameters, linked to the music)", "page": "plugin",
                                "args": [path]})
                out.append({"label": "Turn on" if spec.get("enabled", True) is False else "Turn off (bypass)",
                            "call": "toggle", "args": [path]})
                links = arr.links_of(spec)
                if links:
                    out.append({"label": f"Copy its music links ({', '.join(p[-1] for p in links)})",
                                "call": "copy_links", "args": [path]})
                fx = self._fx_list(path)
                if fx:
                    what = "this FX" if lane in FX_LANES or kind == arr.FX else f"its FX ({len(fx)})"
                    out.append({"label": f"Copy {what}: {', '.join(arr.label_of(f) for f in fx)}",
                                "call": "copy_fx", "args": [path]})
                board = getattr(self, "board_links", None)
                fits = [p for p in (board or {}).get("links", {}) if arr.takes(spec, p)]
                if fits:
                    out.append({"label": f"Paste music links from {board['from']} ({', '.join(p[-1] for p in fits)})",
                                "call": "paste_links", "args": [path]})
                board = getattr(self, "board_fx", None)
                if board and lane not in FX_LANES:
                    out.append({"label": f"Paste FX from {board['from']} ({', '.join(arr.label_of(f) for f in board['fx'])})",
                                "call": "paste_fx", "args": [path]})
                if not hit and (cued or not blocks):
                    out += self._mute_choices(path, spec, bar, span)
                if lane in FX_LANES:
                    out.append({"label": "Earlier in the chain (works before)", "call": "reorder", "args": [path, -1]})
                    out.append({"label": "Later in the chain (works after)", "call": "reorder", "args": [path, 1]})
                if hit:
                    a, b = blocks[k].first, blocks[k].last
                else:
                    spans = self._spans_of(spec)
                    a, b = int(spans[0][0]), int(spans[-1][1])
                out.append({"label": f"Play its bars ({a}-{b}, live)", "page": "preview", "args": [a, b]})
                out.append({"label": f"Exact its bars ({a}-{min(b, a + 7)}, the engine)", "page": "exact", "args": [a, b]})
                if hit and len(blocks) > 1:
                    out.append({"label": "Delete this block (Delete)", "call": "delete_block", "args": [path, k],
                                "danger": True})
                out.append({"label": f"Delete {label}" + ("" if hit and len(blocks) > 1 else " (Delete)"),
                            "call": "remove", "args": [path], "danger": True})
                out.append({"sep": True})
            else:
                out.append({"head": TITLES[lane]})
            if lane not in FX_LANES:
                out.append({"label": f"Put a Generator here (bar {bar})", "catalog": "gen", "call": "new_member",
                            "args": [lane, "$key", bar, to]})
            fx_lane = lane if lane in FX_LANES else {"fg": "fgfx", "bg": "bgfx"}[lane]
            where = "the master" if fx_lane == "master" else f"the whole {TRACKS[fx_lane][0]}"
            owner = ["master"] if fx_lane == "master" else [TRACKS[fx_lane][0]]
            out.append({"label": f"Add FX on {where} here (bars {bar}-{to})", "catalog": "fx", "call": "new_member",
                        "args": [fx_lane, "$key", bar, to]})
            out.append({"label": f"Add FX on {where}, all the song", "catalog": "fx", "call": "add_fx",
                        "args": [owner, "$key"]})
            out.append({"sep": True})
            out.append({"label": "Add FX to the master (over everything)", "catalog": "fx", "call": "add_fx",
                        "args": [["master"], "$key"]})
            return out

    def _mute_choices(self, path, spec, bar: int, span) -> list:
        """Hide (or show again) a member with no block at ``bar``: the cut there, the chosen
        bars, the bar."""
        import math
        from core import arrangement as arr
        out = []
        muted = next((m for m in arr.mutes(spec) if m[0] <= bar <= m[1]), None)
        if muted is not None:
            a, b = muted
            out.append({"label": f"Show again in bars {a}-{b}" if b > a else f"Show again in bar {a}",
                        "call": "mute", "args": [path, a, b, False]})
            if b > a:
                out.append({"label": f"Show again in bar {bar}", "call": "mute", "args": [path, bar, bar, False]})
            return out
        cut = None
        if spec.get("cues") is not None:
            for a, b in self._spans_of(spec):
                if a <= bar <= b:
                    cut = (int(math.floor(a)), int(b))
                    break
        if cut is not None and cut != (bar, bar):
            out.append({"label": f"Hide this cut (bars {cut[0]}-{cut[1]})", "call": "mute", "args": [path, cut[0], cut[1], True]})
        if span and span[0] <= bar <= span[1] and tuple(span) != (bar, bar):
            out.append({"label": f"Hide in the chosen bars {span[0]}-{span[1]}", "call": "mute",
                        "args": [path, span[0], span[1], True]})
        out.append({"label": f"Hide in bar {bar}", "call": "mute", "args": [path, bar, bar, True]})
        return out

    # -- the montage -------------------------------------------------------------------
    def set_pins(self, first: int, last: int, value) -> dict:
        """Bars first..last pinned to ``value`` (a clip's name, "none": nothing), or (None)
        given back to the montage's choice."""
        with self.lock:
            if not any(s.get("source", "clips") == "clips" for s in self.specs()):
                self.items.insert(0, [None, None, {"source": "clips"}])

            def put(s):
                pins = {int(k): v for k, v in (s.get("pins") or {}).items()}
                for b in range(int(first), int(last) + 1):
                    if value is None:
                        pins.pop(b, None)
                    else:
                        pins[b] = value
                out = {k: v for k, v in s.items() if k != "pins"}
                if pins:
                    out["pins"] = dict(sorted(pins.items()))
                return out
            self._edit(["clips"], put)
            bars = f"bars {first}-{last}" if last > first else f"bar {first}"
            return self._answer(montage=True, select={"montage": [int(first), int(last)]},
                                note=f"{bars}: " + ("back to the montage's choice" if value is None else
                                                    "nothing (the BG, or black)" if value == "none" else f"{value}, pinned"))

    def pin(self, bar: int, clip=None) -> dict:
        return self.set_pins(bar, bar, clip)

    def to_bg(self, clip: str, start: float, first: int, last: int) -> dict:
        """A clip as a BG block over these bars (moved, resized and given FX like any)."""
        return self.new_member("bg", clip, first, last, None, start)

    def put_in_montage(self, first: int, last: int, clip: str, start: float = 0.0) -> dict:
        """The clip picked in the browser in these bars: pinned when the montage has it (its
        folder) and plays it from its start; else a BG block over them."""
        paths = [str(p).lower() for p in (self.lib.paths if self.lib is not None else [])]
        if str(resolve(clip)).lower() in paths and not start:
            return self.set_pins(first, last, Path(clip).stem)
        return self.to_bg(clip, start, first, last)

    def montage_menu(self, first: int, last: int, clip: Optional[str] = None) -> list:
        """What can be done to the montage's bars first..last (``clip``: the browser's)."""
        with self.lock:
            bars = f"bars {first}-{last}" if last > first else f"bar {first}"
            runs = [s for s in (self.montage or []) if s["name"] != "none" and self.bar_at(s["t0"] + 1e-3) <= last
                    and self.bar_at(max(s["t0"], s["t1"] - 1e-3)) >= first]
            out = [{"head": f"CLIPS · {bars}"}]
            if clip:
                stem = Path(clip).stem
                out.append({"label": f"Put {stem} here ({bars})", "call": "put_in_montage", "args": [first, last, clip, "$from"]})
            if runs:
                out.append({"label": f"Nothing here ({bars}: the BG, or black) - Delete", "call": "set_pins",
                            "args": [first, last, "none"]})
            if len({s["path"] for s in runs}) == 1 and runs[0].get("path"):
                out.append({"label": f"Make {runs[0]['name']} a BG block (move it, resize it, FX)", "call": "to_bg",
                            "args": [runs[0]["path"], 0.0, first, last]})
            pins = {int(k) for k in self.pins()}
            if any(first <= b <= last for b in pins):
                out.append({"label": "Back to the montage's choice", "call": "set_pins", "args": [first, last, None]})
            out.append({"label": f"Play {bars} (live)", "page": "preview", "args": [first, last]})
            out.append({"label": f"Exact {bars} (the engine)", "page": "exact", "args": [first, last]})
            out.append({"sep": True})
            out.append({"label": "Add FX to the master (over everything)", "catalog": "fx", "call": "add_fx",
                        "args": [["master"], "$key"]})
            return out

    def video_settings(self) -> dict:
        """The montage's own settings (the scene's ``video:``), as the inspector edits them."""
        from core.scene_doc import plain
        with self.lock:
            v = self.video or {}
            out = {k: plain(v.get(k)) for k in VIDEO_KEYS}
            out["triggers_text"] = _yaml_block(v.get("triggers")) if v.get("triggers") else ""
            out["choices"] = {k: list(c) for k, c in VIDEO_CHOICES.items()}
            return out

    def set_video(self, key: str, value) -> dict:
        """One of the montage's settings (``triggers`` as YAML text)."""
        import yaml
        if key not in VIDEO_KEYS:
            return {"error": f"{key}: not a montage setting"}
        want = VIDEO_KEYS[key]
        try:
            if want is dict and isinstance(value, str):
                value = yaml.safe_load(value) if value.strip() else None
                if value is not None and not isinstance(value, dict):
                    raise ValueError("must be 'name: ...' lines")
            elif value is not None and value != "":
                value = want(value)
            if key in VIDEO_CHOICES and value not in (None, "") and value not in VIDEO_CHOICES[key]:
                raise ValueError(f"one of {', '.join(VIDEO_CHOICES[key])}")
        except (ValueError, yaml.YAMLError) as exc:
            return {"error": f"{key}: {exc}"}
        with self.lock:
            v = dict(self.video or {})
            if value is None or value == "":
                v.pop(key, None)
            else:
                v[key] = value
            if v == self.video:
                return self.state()
            self._push()
            self.video = v
            return self._answer(montage=True)


def _yaml_block(v) -> str:
    import yaml
    from core.scene_doc import plain
    return yaml.safe_dump(plain(v), sort_keys=False, allow_unicode=True, width=100).rstrip()

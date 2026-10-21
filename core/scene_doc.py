"""A scene file as the GUI edits it: comments, anchors and layout survive a save.

The scene editor shows rows (triggers, layers) built from plain dicts, and each layer's
extra keys as a block of YAML text. Saving used to rebuild the file from those rows,
which dropped the inline comments, renamed the anchors (``&kick`` -> ``&id001``) and,
since every layer's text was parsed on its own, cut the link an alias makes between two
layers (the ``meters`` lane that shows the same knob the shader gets).

``SceneDoc`` keeps the file as ruamel's round-trip tree and applies only what changed:
the GUI snapshots what its rows say right after loading (the *base*) and, on save, hands
the base and what the rows say now; the difference is patched into the tree in place.
A layer's text shows the anchors of other layers as ``*name`` and resolves them again
when it is parsed, so the link survives the editor.
"""
from __future__ import annotations

import copy
import io
import re
from pathlib import Path
from typing import Any, Iterable, Optional

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq

# the keys a layer row edits with its own widgets (the rest is the row's text)
WIDGET_KEYS = ("source", "enabled", "file", "blend", "opacity", "bars")

_ALIAS = re.compile(r"(?<![\w&*])\*([A-Za-z0-9_.-]+)")
_ANCHOR = re.compile(r"(?<![\w&*])&([A-Za-z0-9_.-]+)")


def _yaml() -> YAML:
    y = YAML()
    y.preserve_quotes = True
    y.width = 4096
    y.indent(mapping=2, sequence=4, offset=2)
    return y


def plain(obj: Any) -> Any:
    """The tree as plain Python (dict, list, int, float, str, bool, None)."""
    if isinstance(obj, dict):
        return {plain(k): plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    return obj


def plain_keeping_signals(obj: Any) -> Any:
    """``plain(obj)``, but its signals (the anchored parts) kept as the scene's objects: what
    is edited as plain values can be written back (``apply_diff``) with them still tied."""
    if isinstance(obj, (dict, list)) and _anchor_name(obj):
        return obj
    if isinstance(obj, dict):
        return {plain(k): plain_keeping_signals(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [plain_keeping_signals(v) for v in obj]
    return plain(obj)


def _anchor_name(obj: Any) -> Optional[str]:
    a = getattr(obj, "anchor", None)
    return getattr(a, "value", None) if a is not None else None


def _walk(obj: Any, path: tuple = ()) -> Iterable[tuple[tuple, Any]]:
    """Every container in the tree (with its path), in document order; shared ones once."""
    seen: set = set()

    def go(o, p):
        if isinstance(o, (dict, list)):
            if id(o) in seen:
                return
            seen.add(id(o))
            yield p, o
            items = o.items() if isinstance(o, dict) else enumerate(o)
            for k, v in items:
                yield from go(v, p + (k,))
    yield from go(obj, path)


def anchors_in(obj: Any) -> dict:
    """name -> the object carrying that anchor (the first one, in document order)."""
    out: dict = {}
    for _, o in _walk(obj):
        name = _anchor_name(o)
        if name and name not in out:
            out[name] = o
    return out


def _replace_ids(obj: Any, table: dict) -> Any:
    """``obj`` with every container whose id is in ``table`` swapped for its entry."""
    if id(obj) in table:
        return table[id(obj)]
    if isinstance(obj, dict):
        for k in list(obj.keys()):
            v = obj[k]
            r = _replace_ids(v, table)
            if r is not v:
                obj[k] = r
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            r = _replace_ids(v, table)
            if r is not v:
                obj[i] = r
    return obj


def _dump_text(node: dict, outside: list, skip: Iterable[str] = ()) -> str:
    """``node`` (less the ``skip`` keys) as block YAML text; the ``outside`` objects (anchored
    elsewhere) show as ``*name`` and the node's own anchors as ``&name``."""
    ctx, body = copy.deepcopy((outside, node))     # one copy: the identities stay shared
    for k in skip:
        if k in body:
            del body[k]
    if not body:
        return ""
    if getattr(body, "fa", None) is not None and body.fa.flow_style():
        body.fa.set_block_style()
    for o in ctx:                  # emitted first, so ``body`` refers to them by alias
        o.yaml_set_anchor(_anchor_name(o), always_dump=True)
    for _, o in _walk(body):       # and its own anchors show their names
        if _anchor_name(o):
            o.yaml_set_anchor(_anchor_name(o), always_dump=True)
    holder = CommentedMap([("__ctx__", CommentedSeq(ctx)), ("__body__", body)])
    out = io.StringIO()
    _yaml().dump(holder, out)
    lines = out.getvalue().splitlines()
    start = next(n for n, ln in enumerate(lines) if ln.startswith("__body__:"))
    return "\n".join(ln[2:] if ln.startswith("  ") else ln for ln in lines[start + 1:]) + "\n"


def flow(value: Any) -> Any:
    """Plain lists/dicts as flow-style ruamel containers (a knob on one line)."""
    if isinstance(value, dict):
        out = CommentedMap((k, flow(v)) for k, v in value.items())
    elif isinstance(value, list):
        out = CommentedSeq(flow(v) for v in value)
    else:
        return value
    out.fa.set_flow_style()
    return out


def _get(node: Any, key: Any) -> Any:
    if isinstance(node, dict):
        return node.get(key)
    if isinstance(node, list) and isinstance(key, int) and -len(node) <= key < len(node):
        return node[key]
    return None


def get_path(node: Any, path: tuple) -> Any:
    """``node[path[0]][path[1]]...`` (None where it stops); list steps are ints."""
    for k in path:
        node = _get(node, k)
    return node


def occurrences(node: Any, path: tuple = ()) -> Iterable[tuple]:
    """Every container in the tree with its path, in document order, each time it occurs
    (an alias too: what ``*name`` points at shows up where the alias is)."""
    if isinstance(node, (dict, list)):
        yield path, node
        items = node.items() if isinstance(node, dict) else enumerate(node)
        for k, v in items:
            yield from occurrences(v, path + (k,))


def _graft(old: Any, new: Any, known: set) -> Any:
    """``new`` keeping the anchored parts of ``old`` (by place): a signal defined there is
    changed in place, so whatever aliases it follows; a ``*name`` (``known``: anchored
    elsewhere) stays the alias while its value is the same, else this place gets its own."""
    if isinstance(old, (dict, list)) and _anchor_name(old):
        if id(old) in known:
            return old if plain(old) == plain(new) else new
        if isinstance(old, dict) and isinstance(new, dict):
            kids = [(k, _graft(old[k], v, known) if k in old else v) for k, v in new.items()]
            for k in [k for k in old if k not in new]:
                del old[k]
            for k, v in kids:
                if old.get(k, _MISSING) is not v:
                    old[k] = v
            return old
        if isinstance(old, list) and isinstance(new, list):
            old[:] = [_graft(old[i], v, known) if i < len(old) else v for i, v in enumerate(new)]
            return old
        return new
    if isinstance(old, dict) and isinstance(new, dict):
        for k in list(new.keys()):
            if k in old:
                new[k] = _graft(old[k], new[k], known)
    elif isinstance(old, list) and isinstance(new, list) and len(old) == len(new):
        for i in range(len(new)):
            new[i] = _graft(old[i], new[i], known)
    return new


def set_knob(doc: "SceneDoc", text: str, key_path: tuple, spec: Any, extra: Optional[dict] = None,
             anchor: Optional[str] = None, links: Optional[dict] = None) -> str:
    """A layer's params ``text`` with the knob at ``key_path`` (e.g. ("uniforms", "u_kick"),
    ("lanes", 0, "knob")) set to ``spec``, on one line; the rest of the text (comments,
    anchors) as it was. A signal (anchor) the old knob or a part of it defined is changed
    in place, so the layers that alias it follow; a part that aliased one stays tied while
    its value is the same. ``anchor`` names the knob a new signal; ``links`` ties parts of
    it to anchored objects (``extra``'s or the scene's): {(): obj} makes the whole knob
    ``*name``, {("hits",): obj} only its source."""
    body = doc.parse_text(text, extra)
    node = body
    for k in key_path[:-1]:
        nxt = _get(node, k)
        if not isinstance(nxt, (dict, list)):
            if not isinstance(node, dict):
                raise KeyError(f"{key_path}: no {k!r} there")
            nxt = CommentedMap()
            node[k] = nxt
        node = nxt
    last = key_path[-1]
    old = _get(node, last)
    links = dict(links or {})
    if () in links:
        new = links.pop(())
    else:
        known = {id(o) for o in list(anchors_in(doc.data).values()) + list((extra or {}).values())}
        new = flow(copy.deepcopy(spec))
        for sub, obj in links.items():
            holder = get_path(new, tuple(sub[:-1]))
            if isinstance(holder, (dict, list)):
                holder[sub[-1]] = obj
        new = _graft(old, new, known | {id(o) for o in links.values()})
        if anchor and hasattr(new, "yaml_set_anchor"):
            new.yaml_set_anchor(anchor, always_dump=True)
    node[last] = new
    return doc.text_of(body, extra)


_MISSING = object()


def apply_diff(node: Any, base: Any, new: Any, root: Any = None) -> Any:
    """Patch ``node`` (the tree) with what changed from ``base`` (plain) to ``new``.

    Returns what to store in the node's place: ``node`` itself when it was patched in
    place (so its comments, anchor and style stay), else the new value. Where ``new`` is
    another signal than ``node`` (an anchor tied or untied there), the place takes it
    whole: a ``*name`` becomes the ``root`` tree's object of that name, so it stays an
    alias, and untying a ``*name`` leaves the signal itself alone."""
    a_node, a_new = _anchor_name(node), _anchor_name(new)
    if plain(new) == base and a_node == a_new:
        return node
    if a_node != a_new and (a_node or a_new):
        if a_new and root is not None and a_new in anchors_in(root):
            return anchors_in(root)[a_new]
        if a_new:                                   # a name new to the scene: written out
            new.yaml_set_anchor(a_new, always_dump=True)
        return new
    if isinstance(new, dict) and isinstance(base, dict) and isinstance(node, dict):
        for k in list(base):
            if k not in new and k in node:
                del node[k]
        order = list(new.keys())
        for n, (k, v) in enumerate(new.items()):
            have = node.get(k, _MISSING)
            if k in base and have is not _MISSING:
                r = apply_diff(have, base[k], v, root)
                if r is not have:
                    node[k] = r
            elif have is _MISSING and hasattr(node, "insert"):
                # next to the key before it in ``new`` (bars_per_clip by clip_per_bar)
                prev = [p for p in order[:n] if p in node]
                keys = list(node.keys())
                node.insert(keys.index(prev[-1]) + 1 if prev else len(keys), k, v)
            elif have is _MISSING or plain(have) != plain(v):
                node[k] = v
        return node
    if (isinstance(new, list) and isinstance(base, list) and isinstance(node, list)
            and len(new) == len(base) == len(node)):
        for i, v in enumerate(new):
            r = apply_diff(node[i], base[i], v, root)
            if r is not node[i]:
                node[i] = r
        return node
    return new


def _tail(node):
    """(container, key, slot) of the place where the comment lines after ``node`` are
    kept: the post-comment of its deepest last entry (block style)."""
    while True:
        if isinstance(node, CommentedMap) and len(node):
            key = list(node.keys())[-1]
            val = node[key]
            if isinstance(val, (CommentedMap, CommentedSeq)) and len(val) and not val.fa.flow_style():
                node = val
                continue
            return node, key, 2
        if isinstance(node, CommentedSeq) and len(node):
            k = len(node) - 1
            val = node[k]
            if isinstance(val, (CommentedMap, CommentedSeq)) and len(val) and not val.fa.flow_style():
                node = val
                continue
            return node, k, 0
        return None


def _keep_comments_with_layers(seq: CommentedSeq, out: list) -> None:
    """Replace the layers of ``seq`` by ``out`` with each comment line above a layer still
    above that layer. ruamel keeps such a comment either by the layer's place in the list
    (after a blank line) or at the end of the layer before it: both move with the layer
    they stand above, and go with it when it is removed."""
    old = list(seq)
    above = {}                                       # id(layer) -> its comment, by place
    for j, cmt in seq.ca.items.items():
        if isinstance(j, int) and j < len(old):
            above[id(old[j])] = cmt
    trailing = {}                                    # id(layer) -> the comment ending the one before it
    for j, node in enumerate(old):
        tail = _tail(node)
        if tail is None:
            continue
        cont, key, slot = tail
        entry = cont.ca.items.get(key)
        if entry and len(entry) > slot and entry[slot] is not None:
            target = old[j + 1] if j + 1 < len(old) else None
            trailing[id(target)] = entry[slot]
            entry[slot] = None
    for i, node in enumerate(out):                   # a new layer before a comment can hold it
        if isinstance(node, dict) and not isinstance(node, CommentedMap):
            out[i] = CommentedMap(node)
    keep = {k: v for k, v in seq.ca.items.items() if not isinstance(k, int)}
    seq[:] = out
    seq.ca.items.clear()
    seq.ca.items.update(keep)
    for i, node in enumerate(out):
        if id(node) in above:
            seq.ca.items[i] = above[id(node)]
        if id(node) in trailing and i > 0:
            tail = _tail(out[i - 1])
            if tail is not None:
                cont, key, slot = tail
                entry = cont.ca.items.setdefault(key, [None] * (4 if isinstance(cont, CommentedMap) else 2))
                while len(entry) <= slot:
                    entry.append(None)
                entry[slot] = trailing[id(node)]
    if None in trailing and out:                    # the comment after the last layer stays last
        tail = _tail(out[-1])
        if tail is not None:
            cont, key, slot = tail
            entry = cont.ca.items.setdefault(key, [None] * (4 if isinstance(cont, CommentedMap) else 2))
            entry[slot] = trailing[None]


class SceneDoc:
    """A scene file, round-trip: ``load``, read the rows, ``apply`` the edits, ``dumps``."""

    def __init__(self, data: Optional[CommentedMap] = None, newline: str = "\n"):
        self.data = data if data is not None else CommentedMap()
        self.newline = newline         # the file's own line ends, kept on save

    # -- reading ---------------------------------------------------------------------
    @classmethod
    def from_text(cls, text: str) -> "SceneDoc":
        data = _yaml().load(text) if text.strip() else None
        return cls(data if isinstance(data, CommentedMap) else None)

    @classmethod
    def load(cls, path: str) -> "SceneDoc":
        p = Path(path)
        if not p.is_file():
            return cls()
        raw = p.read_bytes().decode("utf-8")
        doc = cls.from_text(raw.replace("\r\n", "\n"))
        doc.newline = "\r\n" if "\r\n" in raw else "\n"
        return doc

    def video(self) -> CommentedMap:
        v = self.data.get("video")
        if not isinstance(v, dict):
            v = CommentedMap()
            self.data["video"] = v
        return v

    def layers(self) -> list:
        v = self.data.get("video")
        return list((v.get("layers") if isinstance(v, dict) else None) or [])

    def plain_video(self) -> dict:
        v = self.data.get("video")
        return plain(v) if isinstance(v, dict) else {}

    def layer_text(self, i: int, skip: Iterable[str] = WIDGET_KEYS) -> str:
        """Layer ``i`` as YAML text, without the ``skip`` keys. What other layers (or the
        rest of the scene) anchor shows as ``*name``; this layer's own anchors as ``&name``."""
        node = self.layers()[i]
        if not isinstance(node, dict):
            return ""
        own = {id(o) for _, o in _walk(node)}
        first: dict = {}               # anchored object id -> path of its first occurrence
        for p, o in _walk(self.data):
            if _anchor_name(o) and id(o) not in first:
                first[id(o)] = p
        here = ("video", "layers", i)
        outside = [o for _, o in _walk(self.data)
                   if _anchor_name(o) and id(o) in own and first[id(o)][:3] != here]
        return _dump_text(node, outside, skip)

    def text_of(self, body: dict, extra: Optional[dict] = None, skip: Iterable[str] = ()) -> str:
        """What ``parse_text`` gave (perhaps edited) as text again: the anchors it resolved
        (the scene's, or ``extra``) show as ``*name`` once more."""
        known = {id(o) for o in list(anchors_in(self.data).values()) + list((extra or {}).values())}
        outside = [o for _, o in _walk(body) if _anchor_name(o) and id(o) in known]
        return _dump_text(body, outside, skip)

    def parse_text(self, text: str, extra: Optional[dict] = None) -> CommentedMap:
        """A layer's text back into the tree's types; ``*name`` resolves to the scene's
        anchor of that name (``extra`` first: anchors other edited rows define)."""
        text = text or ""
        if not text.strip():
            return CommentedMap()
        known = dict(anchors_in(self.data))
        known.update(extra or {})
        wanted = [n for n in dict.fromkeys(_ALIAS.findall(text)) if n not in set(_ANCHOR.findall(text))]
        missing = [n for n in wanted if n not in known]
        if missing:
            raise KeyError(", ".join(f"*{n}" for n in missing))
        y = _yaml()
        prefix = ""
        if wanted:
            stubs = copy.deepcopy([known[n] for n in wanted])   # one copy: shared stays shared
            for n, o in zip(wanted, stubs):
                o.yaml_set_anchor(n, always_dump=True)
            buf = io.StringIO()
            y.dump(CommentedMap([("__ctx__", CommentedSeq(stubs))]), buf)
            prefix = buf.getvalue()
        body = "".join("  " + ln + "\n" for ln in text.splitlines())
        doc = y.load(prefix + "__body__:\n" + body)
        result = doc.get("__body__") or CommentedMap()
        if not isinstance(result, dict):
            raise ValueError("must be 'key: value' lines")
        if wanted:
            parsed = anchors_in(doc["__ctx__"])
            _replace_ids(result, {id(parsed[n]): known[n] for n in wanted if n in parsed})
        return result

    # -- writing ---------------------------------------------------------------------
    def apply_video(self, base: dict, new: dict, skip: Iterable[str] = ("layers",)) -> None:
        """Patch the video section's keys (all but ``skip``) with what changed."""
        skip = set(skip)               # in neither, so apply_diff leaves them be
        b = {k: v for k, v in base.items() if k not in skip}
        n = {k: v for k, v in new.items() if k not in skip}
        if plain(n) != b:              # a scene without video: stays without one
            apply_diff(self.video(), b, n, self.data)

    def set_layers(self, items: list) -> None:
        """``items``: one ``(node, base, new)`` per layer, in stack order; ``node`` is the
        layer's original tree (None for a new one) and ``base`` what it said when loaded."""
        out = []
        for node, base, new in items:
            if node is None or base is None:
                out.append(new)
            else:
                out.append(apply_diff(node, base, new, self.data))
        if not out and not self.layers():
            return
        video = self.video()
        seq = video.get("layers")
        if not out:
            del video["layers"]
            return
        if not (isinstance(seq, list) and len(seq) == len(out) and all(a is b for a, b in zip(seq, out))):
            if isinstance(seq, CommentedSeq):
                _keep_comments_with_layers(seq, out)
            else:
                video["layers"] = CommentedSeq(out)
        self._dedupe_anchors()           # a signal edited in one layer and used in another

    def _dedupe_anchors(self) -> None:
        """Two different objects under one anchor name would not load: the later one goes
        (its references use the first) when they say the same, else it is renamed."""
        firsts: dict = {}
        table: dict = {}
        for _, o in _walk(self.data):
            name = _anchor_name(o)
            if not name:
                continue
            if name not in firsts:
                firsts[name] = o
            elif firsts[name] is not o:
                if plain(firsts[name]) == plain(o):
                    table[id(o)] = firsts[name]
                else:
                    k = 2
                    while f"{name}_{k}" in firsts:
                        k += 1
                    o.yaml_set_anchor(f"{name}_{k}", always_dump=True)
                    firsts[f"{name}_{k}"] = o
        if table:
            _replace_ids(self.data, table)

    def dumps(self) -> str:
        """The scene as text. A copy is written: ruamel's emitter moves the comment tokens
        it writes (one before a list's first item came back as ``layers:   -``, which no
        layer's text could be read from again)."""
        out = io.StringIO()
        _yaml().dump(copy.deepcopy(self.data), out)
        return out.getvalue()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline=self.newline) as f:
            f.write(self.dumps())

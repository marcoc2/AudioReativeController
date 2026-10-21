"""A REAPER project (.rpp) read back: its tracks and the items on them.

REAPER is the timeline: what plays when, cut where, from which point of which file, at
what speed and pitch. Reading the .rpp gives that edit exactly (no guessing it from the
sound), so a ``footage`` layer can show a clip cut the way its sound was cut:

    proj = read_project("C:/Audio/.../song.rpp")
    for track in proj.tracks:
        for item in track.items:
            item.position, item.length       # where it sits in the song (s)
            item.offset, item.rate           # where it starts in its file, the speed
            item.pitch                       # semitones it is transposed
            item.file, item.reverse          # the file it plays (a section's own file)

The .rpp is a text of nested ``<BLOCK args ... >`` chunks, one statement a line.
"""
from __future__ import annotations

import math
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class Chunk:
    name: str
    args: List[str]
    lines: List[List[str]] = field(default_factory=list)       # plain statements
    children: List["Chunk"] = field(default_factory=list)
    events: List[List[str]] = field(default_factory=list)      # a MIDI source's E/e lines

    def get(self, key: str) -> Optional[List[str]]:
        """The arguments of the first statement ``key``, or None."""
        for row in self.lines:
            if row and row[0] == key:
                return row[1:]
        return None

    def child(self, name: str) -> Optional["Chunk"]:
        return next((c for c in self.children if c.name == name), None)


def _split(line: str) -> List[str]:
    try:
        return shlex.split(line, posix=True)
    except ValueError:                                         # an odd quote: split on spaces
        return line.split()


def parse(text: str) -> Chunk:
    """The project's root chunk (``REAPER_PROJECT``)."""
    root, stack = None, []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("<"):
            row = _split(line[1:])
            c = Chunk(row[0] if row else "", row[1:])
            if stack:
                stack[-1].children.append(c)
            else:
                root = c
            stack.append(c)
        elif line == ">":
            if stack:
                stack.pop()
        elif stack:
            if stack[-1].name == "SOURCE" and stack[-1].args[:1] == ["MIDI"]                     and line[:1] in ("E", "e", "X", "x") and line[1:2] in (" ", "m"):
                stack[-1].events.append(line.split())          # MIDI events, kept apart
                continue
            stack[-1].lines.append(_split(line))
    if root is None:
        raise ValueError("not a REAPER project (no <REAPER_PROJECT chunk)")
    return root


@dataclass
class Item:
    position: float            # song seconds
    length: float              # seconds on the timeline
    offset: float              # where it starts in its file (s), the section's start included
    rate: float                # file seconds per song second
    pitch: float               # semitones
    file: str                  # the file played ("" for MIDI / empty items)
    kind: str                  # WAVE, MP3, VIDEO, MIDI, ...
    name: str = ""
    muted: bool = False
    reverse: bool = False
    loop: bool = True          # past the file's end it starts it over
    markers: List[tuple] = field(default_factory=list)   # stretch markers (item s, file s)
    midi: Optional[Chunk] = None                          # a MIDI item's source

    @property
    def end(self) -> float:
        return self.position + self.length


@dataclass
class Span:
    """Song time [t0, t1) plays ``file`` from ``s0`` on, ``rate`` file seconds a song second."""
    t0: float
    t1: float
    file: str
    s0: float
    rate: float

    def at(self, t: float) -> float:
        return self.s0 + (t - self.t0) * self.rate


@dataclass
class Track:
    name: str
    items: List[Item]
    muted: bool = False
    depth: int = 0             # folder depth (0: top level)


@dataclass
class Project:
    path: str
    tempo: Optional[float]
    tracks: List[Track]
    # the time signatures, [(s, (numerator, denominator))]: the project's at 0, then its
    # tempo map's markers (each starts a new bar in REAPER, even mid-bar: the bar before it
    # is short)
    meters: List[Tuple[float, Tuple[int, int]]] = field(default_factory=list)

    def track(self, name: str) -> Track:
        found = [t for t in self.tracks if t.name.lower() == name.lower()]
        if not found:
            raise KeyError(f"no track {name!r}; tracks: {[t.name for t in self.tracks]}")
        return found[0]

    def items_of(self, file: str) -> List[Item]:
        """Every unmuted item that plays ``file`` (matched by file name), in time order."""
        stem = Path(file).name.lower()
        return sorted((it for t in self.tracks if not t.muted for it in t.items
                       if not it.muted and Path(it.file).name.lower() == stem),
                      key=lambda it: it.position)


def _f(row: Optional[List[str]], k: int = 0, default: float = 0.0) -> float:
    try:
        return float(row[k]) if row and len(row) > k else default
    except ValueError:
        return default


def _item(c: Chunk, base: Path) -> Item:
    src = c.child("SOURCE")
    midi = src if src is not None and src.args[:1] == ["MIDI"] else None
    kind, file, start, reverse = "", "", 0.0, False
    while src is not None:
        kind = src.args[0] if src.args else ""
        if kind == "SECTION":                                  # a part of another source
            start += _f(src.get("STARTPOS"))
            reverse = reverse or _f(src.get("MODE")) % 2 == 1
            src = src.child("SOURCE")
            continue
        f = src.get("FILE")
        file = f[0] if f else ""
        break
    if file and not Path(file).is_absolute():
        file = str(base / file)
    rate = c.get("PLAYRATE")
    name = c.get("NAME")
    markers = []
    sm = c.get("SM")
    if sm:                                                     # "pos src [slope] + pos src ..."
        group = []
        for tok in sm + ["+"]:
            if tok == "+":
                if len(group) >= 2:
                    markers.append((group[0], group[1]))
                group = []
            else:
                try:
                    group.append(float(tok))
                except ValueError:
                    pass
    loop = c.get("LOOP")
    return Item(position=_f(c.get("POSITION")), length=_f(c.get("LENGTH")),
                offset=_f(c.get("SOFFS")) + start, rate=_f(rate, 0, 1.0), pitch=_f(rate, 2),
                file=file, kind=kind, name=name[0] if name else "",
                muted=_f(c.get("MUTE")) == 1, reverse=reverse,
                loop=loop is None or _f(loop) == 1, markers=sorted(markers), midi=midi)


def item_spans(item: Item, source_length: Optional[float] = None) -> List[Span]:
    """What the item plays, as straight pieces: song time -> its file's time.

    Without stretch markers the file runs from ``offset`` at ``rate``. Stretch markers
    pin item time (seconds from the item's start, at the item's rate) to file time; the
    file is stretched straight between two of them and runs at the item's rate before the
    first and after the last. A looped item past the file's end starts it over."""
    if item.length <= 0 or item.rate == 0:
        return []
    r = item.rate
    if item.markers:
        pts = [(p / r, s) for p, s in item.markers]             # (item seconds, file seconds)
    else:
        pts = [(0.0, item.offset)]
    # straight pieces in item time: before the first point, between points, after the last
    pieces = []
    cuts = sorted({0.0, item.length} | {p for p, _ in pts if 0.0 < p < item.length})
    for a, b in zip(cuts, cuts[1:]):
        mid = (a + b) / 2
        if mid <= pts[0][0]:
            s_a, slope = pts[0][1] - (pts[0][0] - a) * r, r
        elif mid >= pts[-1][0]:
            s_a, slope = pts[-1][1] + (a - pts[-1][0]) * r, r
        else:
            k = max(i for i in range(len(pts) - 1) if pts[i][0] <= mid)
            (p0, s0), (p1, s1) = pts[k], pts[k + 1]
            slope = (s1 - s0) / (p1 - p0)
            s_a = s0 + (a - p0) * slope
        pieces.append((a, b, s_a, slope))
    spans = []
    L = source_length if (item.loop and source_length) else None
    for a, b, s_a, slope in pieces:
        while b - a > 1e-9:
            s = s_a
            if L:
                wraps = math.floor(s / L)
                s -= wraps * L
                # where the file runs out within this piece
                left = (L - s) / slope if slope > 0 else math.inf
                stop = min(b, a + left) if left > 1e-9 else b
            else:
                stop = b
            spans.append(Span(item.position + a, item.position + stop, item.file, s, slope))
            s_a += (stop - a) * slope
            a = stop
    return spans


def read_project(path: str) -> Project:
    p = Path(path)
    root = parse(p.read_text(encoding="utf-8", errors="replace"))
    tempo = root.get("TEMPO")
    tracks, depth = [], 0
    for c in root.children:
        if c.name != "TRACK":
            continue
        name = c.get("NAME")
        bus = c.get("ISBUS")
        tracks.append(Track(name=name[0] if name else "",
                            items=[_item(i, p.parent) for i in c.children if i.name == "ITEM"],
                            muted=_f(c.get("MUTESOLO")) == 1, depth=depth))
        if bus and len(bus) > 1:
            depth += int(_f(bus, 1))
    return Project(path=str(p), tempo=_f(tempo) if tempo else None, tracks=tracks,
                   meters=_meters(root))


def _meters(root: Chunk) -> List[Tuple[float, Tuple[int, int]]]:
    """[(s, (numerator, denominator))]: TEMPO's meter at 0, then each tempo-map point that
    sets one ("PT s bpm shape sig ..."; sig = numerator + denominator * 65536, 0: none)."""
    tempo = root.get("TEMPO")
    first = (int(_f(tempo, 1, 4)), int(_f(tempo, 2, 4))) if tempo and len(tempo) > 2 else (4, 4)
    out = [(0.0, first)]
    env = root.child("TEMPOENVEX")
    for row in env.lines if env is not None else []:
        if row[:1] != ["PT"] or len(row) < 5:
            continue
        sig = int(_f(row, 4))
        if sig <= 0:
            continue
        num, den = sig & 0xFFFF, sig >> 16
        at = _f(row, 1)
        if num and den:
            out = [m for m in out if abs(m[0] - at) > 1e-6] + [(at, (num, den))]
    return sorted(out)


def midi_meters(meters, tempo: float) -> List[Tuple[float, Tuple[int, int]]]:
    """REAPER's meters as a MIDI file can hold them, [(quarter-note beat, signature)]: a
    marker that comes mid-bar makes the bar before it short, written as a bar of its own
    meter (3 beats of 4/4 before it: a 3/4 bar)."""
    beats = [(t * tempo / 60.0, sig) for t, sig in sorted(meters)]
    out = []
    for k, (b, sig) in enumerate(beats):
        bar = sig[0] * 4.0 / sig[1]                               # the bar in quarter notes
        end = beats[k + 1][0] if k + 1 < len(beats) else None
        full = int(math.floor((end - b) / bar + 1e-6)) if end is not None else 1
        if full > 0:
            out.append((b, sig))
        rest = (end - b) - full * bar if end is not None else 0.0
        if rest > 1e-6:
            den = next((d for d in (sig[1], 8, 16, 32) if abs(rest * d / 4 - round(rest * d / 4)) < 1e-4), 32)
            out.append((b + full * bar, (max(1, int(round(rest * den / 4))), den)))
    return [m for k, m in enumerate(out) if k == 0 or m[1] != out[k - 1][1]]


# ----------------------------------------------------------- a clip cut as its sound was

def _audio(path: str):
    """(mono float32 samples, sample rate) of an audio file (a video's sound through librosa)."""
    try:
        import soundfile as sf
        y, sr = sf.read(str(path), dtype="float32", always_2d=True)
        return y.mean(1), sr
    except Exception:
        import librosa
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y, sr = librosa.load(str(path), sr=None, mono=True)
        return y.astype("float32"), sr


def _duration(path: str) -> Optional[float]:
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        try:
            import librosa
            return float(librosa.get_duration(path=str(path)))
        except Exception:
            return None


def _same_file(a: Path, b: Path) -> bool:
    import filecmp
    try:
        return a.resolve() == b.resolve() or (a.stat().st_size == b.stat().st_size
                                               and filecmp.cmp(a, b, shallow=False))
    except OSError:
        return False


def exact_pieces(rendered, sources, sr: int, chunk: int = 4096, floor: float = 1e-3,
                 min_score: float = 0.999, bridge: float = 2.0) -> List[tuple]:
    """Where ``rendered`` (samples) comes from, sample for sample, in ``sources`` ({name:
    samples}, same rate): a glue or a consolidation of unprocessed items keeps their
    samples, so every sounding stretch is found exactly. Stretches that keep the same
    offset into the same source (one item, with a quiet bit inside) are joined across up
    to ``bridge`` seconds, and each piece then runs on until the next one begins (and the
    first from the start), so the quiet between pieces belongs to the piece before it.
    -> [(start s, end s, source name, source s at start)], rate 1."""
    import numpy as np
    spectra = {}
    for name, src in sources.items():
        if len(src) < chunk:
            continue
        n = 1 << int(np.ceil(np.log2(len(src) + chunk)))
        energy = np.concatenate([[0.0], np.cumsum(src.astype(np.float64) ** 2)])
        spectra[name] = (n, np.fft.rfft(src, n), energy[chunk:] - energy[:-chunk])
    hits = []                                                  # (rendered sample, name, offset)
    for i in range(0, len(rendered) - chunk + 1, chunk):
        q = rendered[i:i + chunk].astype(np.float64)
        if np.abs(q).max() < floor:
            continue
        best = None
        for name, (n, F, energy) in spectra.items():
            c = np.fft.irfft(F * np.conj(np.fft.rfft(q, n)), n)[:len(energy)]
            score = c / (np.linalg.norm(q) * np.sqrt(np.maximum(energy, 1e-18)))
            k = int(np.argmax(score))
            if score[k] >= min_score and (best is None or score[k] > best[0]):
                best = (score[k], name, k - i)
        if best:
            hits.append((i, best[1], best[2]))
    pieces = []                                                # [start, end, name, offset]
    for i, name, off in hits:
        last = pieces[-1] if pieces else None
        if last and last[2] == name and abs(last[3] - off) <= 2 and i - last[1] <= bridge * sr:
            last[1] = i + chunk
        else:
            pieces.append([i, i + chunk, name, off])
    for k, p in enumerate(pieces):                             # the exact edges of the sound
        src = sources[p[2]]
        a, b, off = p[0], p[1], p[3]
        while a > 0 and 0 <= a - 1 + off < len(src) and abs(rendered[a - 1] - src[a - 1 + off]) < 1e-4 \
                and abs(rendered[a - 1]) >= 1e-6:
            a -= 1
        while b < len(rendered) and b + off < len(src) and abs(rendered[b] - src[b + off]) < 1e-4 \
                and abs(rendered[b]) >= 1e-6:
            b += 1
        p[0], p[1] = a, b
    for k, p in enumerate(pieces):                             # the quiet after a piece is its own
        nxt = pieces[k + 1][0] if k + 1 < len(pieces) else len(rendered)
        p[1] = max(p[1], min(nxt, len(sources[p[2]]) - p[3]))
        if k == 0:
            p[0] = min(p[0], max(0, -p[3]))
    return [(a / sr, b / sr, name, (a + off) / sr) for a, b, name, off in pieces]


class ClipSounds:
    """The audio files that are a clip's own sound, and the ones made from them.

    A clip's sound reaches a REAPER project as the clip itself, as its stems (a folder
    named after the clip, next to it or next to the project: ``<clip stem>/vocals.flac``,
    all starting where the clip starts), as a copy of one of those (REAPER's
    ``Media/vocals-imported.flac``), or as a glue of items cut from them.
    ``clip_time(file)`` -> the pieces of that file that are the clip's sound, as
    [(file s from, file s to, clip s at from)], or None when it is not the clip's."""

    def __init__(self, clip: str, project_dir: str):
        self.clip = Path(clip)
        self.own = [self.clip]
        for d in (self.clip.parent / self.clip.stem, Path(project_dir) / self.clip.stem):
            if d.is_dir():
                self.own += sorted(p for p in d.iterdir() if p.is_file())
        twin = Path(project_dir) / self.clip.name                 # the clip as the project holds it
        if twin.is_file():
            self.own.append(twin)
        self._known: dict = {}

    def clip_time(self, file: str) -> Optional[List[tuple]]:
        key = str(Path(file)).lower()
        if key in self._known:
            return self._known[key]
        f = Path(file)
        found = None
        if f.is_file():
            if any(_same_file(f, o) for o in self.own):
                found = [(0.0, _duration(str(f)) or 0.0, 0.0)]
            elif f.suffix.lower() in (".wav", ".flac", ".aif", ".aiff"):
                found = self._glued(f)
        self._known[key] = found
        return found

    def _glued(self, f: Path) -> Optional[List[tuple]]:
        y, sr = _audio(str(f))
        sources = {}
        for o in self.own:
            if o.suffix.lower() in (".wav", ".flac", ".aif", ".aiff"):
                s, r = _audio(str(o))
                if r == sr:
                    sources[str(o)] = s
        if not sources:
            return None
        return [(a, b, at) for a, b, _, at in exact_pieces(y, sources, sr)] or None


def clip_cues(project: str, clip: str, tracks: Optional[Sequence[str]] = None,
              shortest: float = 0.02) -> List[list]:
    """The cues (core/video/footage.py) that show ``clip`` exactly as its sound plays in the
    REAPER ``project``: every unmuted item, on the unmuted ``tracks`` (all when None), that
    plays the clip's sound (the cuts, the item's rate, its stretch markers, loops and glues
    followed) becomes [song s, clip s, seconds(, rate)] pieces.

    Kept in ``render_output/.cache/reaper_cues`` and worked out again only when the project,
    the clip or the asked tracks change (a glue takes seconds to find)."""
    import hashlib
    import json
    stamp = []
    for f in (project, clip):
        try:
            st = Path(f).stat()
            stamp.append(f"{Path(f).resolve()}|{st.st_mtime_ns}|{st.st_size}")
        except OSError:
            stamp.append(str(f))
    key = hashlib.sha1(("|".join(stamp) + f"|{sorted(tracks) if tracks else None}|2").encode()).hexdigest()[:20]
    cache = Path("render_output") / ".cache" / "reaper_cues" / f"{key}.json"
    if cache.is_file():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    cues = _clip_cues(project, clip, tracks, shortest)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(cues), encoding="utf-8")
    except OSError:
        pass
    return cues


def _clip_cues(project: str, clip: str, tracks: Optional[Sequence[str]], shortest: float) -> List[list]:
    proj = read_project(project)
    sounds = ClipSounds(clip, str(Path(project).parent))
    wanted = None if tracks is None else {str(t).lower() for t in tracks}
    cues = []
    for track in proj.tracks:
        if track.muted or (wanted is not None and track.name.lower() not in wanted):
            continue
        for item in track.items:
            if item.muted or not item.file or item.kind == "MIDI":
                continue
            pieces = sounds.clip_time(item.file)
            if not pieces:
                continue
            for sp in item_spans(item, _duration(item.file)):
                if sp.rate <= 0:
                    continue
                for f0, f1, c0 in pieces:                      # the span's file time inside the piece
                    lo, hi = max(sp.s0, f0), min(sp.at(sp.t1), f1)
                    if hi <= lo:
                        continue
                    t_lo = sp.t0 + (lo - sp.s0) / sp.rate
                    t_hi = sp.t0 + (hi - sp.s0) / sp.rate
                    if t_hi - t_lo >= shortest:
                        cues.append([round(t_lo, 4), round(c0 + lo - f0, 4), round(t_hi - t_lo, 4),
                                     round(sp.rate, 5)])
    cues.sort()
    for k in range(len(cues) - 1):                             # where items overlap, the later one wins
        nxt = cues[k + 1][0]
        if cues[k][0] + cues[k][2] > nxt:
            cues[k][2] = round(nxt - cues[k][0], 4)
    return [c[:3] if c[3] == 1 else c for c in cues if c[2] > 0]


# ----------------------------------------------------------- MIDI items back to notes

def item_notes(item: Item, tempo: float) -> List[tuple]:
    """The notes a MIDI item plays, in song time: [(start s, pitch, velocity, seconds,
    channel)]. Its events are read on a fixed ``tempo`` (the project's; REAPER's own
    ``E delta status data data`` lines, ``ppq`` from HASDATA), from the item's offset on,
    over and over when it loops, cut at the item's end; muted events are left out."""
    src = item.midi
    if src is None or item.muted:
        return []
    has = src.get("HASDATA") or []
    ppq = _f(has, 1, 960.0) or 960.0
    sec = 60.0 / tempo / ppq                                   # seconds a tick
    tick, events = 0, []
    for ev in src.events:
        try:
            tick += int(ev[1])
            status, d1, d2 = (int(x, 16) for x in ev[2:5])
        except (ValueError, IndexError):
            continue                                           # an X (sysex / meta) event
        if not ev[0].endswith("m"):
            events.append((tick, status, d1, d2))
    if not events:
        return []
    source_len = tick * sec                                    # the last event ends the source
    notes, held = [], {}
    for t, status, d1, d2 in events:
        kind, ch = status & 0xF0, status & 0x0F
        if kind == 0x90 and d2 > 0:
            held[(ch, d1)] = (t * sec, d2)
        elif kind in (0x80, 0x90) and (ch, d1) in held:
            t0, vel = held.pop((ch, d1))
            notes.append((t0, d1, vel, t * sec - t0, ch))
    out = []
    loops = (math.ceil((item.offset + item.length) / source_len) + 1) if item.loop and source_len > 0 else 1
    for k in range(loops):
        for t0, pitch, vel, dur, ch in notes:
            t = item.position + t0 + k * source_len - item.offset
            if item.position - 1e-6 <= t < item.end - 1e-6:
                out.append((round(t, 6), pitch, vel, round(min(dur, item.end - t), 6), ch))
    return sorted(out)


def write_midi(project: str, tracks: Sequence[str], out: str, merge: Optional[str] = None) -> dict:
    """The notes of the REAPER ``tracks`` (by name; unmuted items) as a MIDI file ``out``,
    one MIDI track each, named as in REAPER; ``merge``: a MIDI file whose tracks go first
    (the song's own export, so a render reads them all from one file). Tempo: the
    project's (a single tempo); meters: the project's (``midi_meters``), unless merged into
    a file that has its own. -> {track name: notes written}."""
    import mido
    proj = read_project(project)
    tempo = proj.tempo or 120.0
    if merge:
        mid = mido.MidiFile(merge)
        ppq = mid.ticks_per_beat
    else:
        ppq = 960
        mid = mido.MidiFile(type=1, ticks_per_beat=ppq)
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(tempo), time=0))
        now = 0
        for beat, (num, den) in midi_meters(proj.meters or [(0.0, (4, 4))], tempo):
            at = int(round(beat * ppq))
            meta.append(mido.MetaMessage("time_signature", numerator=num, denominator=den, time=at - now))
            now = at
        meta.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(meta)
    tick = ppq * tempo / 60.0                                  # ticks a second
    written = {}
    for name in tracks:
        found = [t for t in proj.tracks if t.name.lower() == str(name).lower() and not t.muted]
        if not found:
            raise KeyError(f"no unmuted track {name!r} in {project}")
        notes = sorted(n for t in found for it in t.items for n in item_notes(it, tempo))
        # tracks of one name playing the same note at once (a doubled part): it sounds once
        once = {}
        for n in notes:
            key = (round(n[0], 3), n[1], n[4])
            if key not in once or n[2] > once[key][2]:
                once[key] = n
        notes = sorted(once.values())
        msgs = []
        for t0, pitch, vel, dur, ch in notes:
            msgs.append((round(t0 * tick), 1, mido.Message("note_on", note=pitch, velocity=vel, channel=ch)))
            msgs.append((round((t0 + dur) * tick), 0, mido.Message("note_off", note=pitch, velocity=0, channel=ch)))
        msgs.sort(key=lambda m: (m[0], m[1]))
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=str(name), time=0))
        now = 0
        for at, _, msg in msgs:
            track.append(msg.copy(time=at - now))
            now = at
        track.append(mido.MetaMessage("end_of_track", time=0))
        mid.tracks.append(track)
        written[str(name)] = len(notes)
    mid.save(out)
    return written


def markers(path: str) -> List[Tuple[float, str, Optional[float]]]:
    """A REAPER project's markers and regions: (seconds, name, the region's end or None)."""
    import shlex
    out: dict = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("MARKER "):
            continue
        try:
            parts = shlex.split(line, posix=True)
            idx, pos = int(parts[1]), float(parts[2])
        except (ValueError, IndexError):
            continue
        name = parts[3] if len(parts) > 3 else ""
        region = len(parts) > 4 and parts[4].isdigit() and int(parts[4]) & 1
        if region and idx in out and out[idx][2] is None and out[idx][3]:
            t, n, _, _ = out[idx]
            out[idx] = (t, n, pos, True)                 # a region's second line: its end
        else:
            out[idx] = (pos, name, None, bool(region))
    return sorted((t, n, end) for t, n, end, _ in out.values())

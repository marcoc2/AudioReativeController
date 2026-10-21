"""One clip played on cue: at given song times, given stretches of one video — lip sync.

When a stem was cut from a clip's own sound (a voice chopped from a singing clip, a
sample lifted from a cartoon), the clip can be shown doing exactly what is heard: each
cue says when, in the song, a stretch of the clip starts, where in the clip, and for how
long. ``tools/match_stem.py`` finds the cues by matching the stem to the clip's sound.

    - source: footage
      file: input/esfolado/MiniMax_H3_00039_.mp4
      cues: [[12.14, 8.0, 1.5], [24.73, 14.56, 3.41], [44.39, 7.43, 2.57, 0]]
                          # [song s, clip s, seconds(, rate)]: rate 1 by default, 0 holds
                          # the frame (a sustained note), 0.5 half speed; or the path of a
                          # cue file from tools/match_stem.py; or, exact, the REAPER edit:
                          #   cues: {reaper: C:/.../song.rpp, tracks: [vocals]}
                          # every item there that plays this clip's sound (its stems, a
                          # copy, a glue of them) shows the clip cut, stretched and sped
                          # as the sound is (core/reaper_project.py)
      fit: contain        # contain: all of the clip in the frame | cover: the frame filled
      scale: 1.0          # of the fitted size
      at: [0.5, 0.5]      # where its centre sits (fractions of the frame)
      x: {lfo: sine, period: 2/1, gain: 0.3, bias: 0.35}   # x / y / scale may be knobs that
      y: 0.6              # follow the music (layers._knob): it sways, jumps on the kick...
      flip: false         # mirrored left to right
      loop: false         # past the clip's end, from its start again (a dance loop)
      cutout:             # only the clip's objects, the picture below elsewhere: a mask
        track: render_output/footage/MiniMax_H3_00039_segments   # track of the clip itself
        objects: [2, 3]   # (segment_video.py --video <the clip> --start-time 0); default all
        drop: [1]         # or: all but these (the backdrop: a figure on white, where SAM
                          # names the white and leaves bits of the figure unnamed)
        grow: 0           # pixels (at 720p of the placed clip) to grow (+) or shrink (-)
        feather: 2        # pixels (at 720p) of soft edge
      hold: 0             # seconds the last frame of a cue stays when the next is not there yet

Or arranged by bars (what the GUI's Arrange window writes), alone or with cues:

      blocks:
        - {bars: [5, 8], from: 12.0}              # bars 5 to 8 (DAW numbers, both in),
        - {bars: [9, 12], file: C:/v/final.mp4, from: 42.0, rate: 1}   # from 12 s of the clip;
                          # a block may name its own file (a background lane: a clip a block;
                          # the cutout's mask track is the layer's own file's)

Outside the cues (and the holds) the layer shows nothing; with no cues at all it plays
the whole song from the clip's start (a ``bars:`` window gates it). A still image (png,
jpg) is a one-frame clip. Frames are read from the file as they are needed (a seek where
a cue jumps), so a long clip costs no memory.
"""
from __future__ import annotations

import bisect
from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np

FITS = ("contain", "cover")


def read_cues(cues, clip: Optional[str] = None) -> List[tuple]:
    """``cues`` as sorted (song s, clip s, seconds, rate) tuples; a path reads a cue file
    (YAML or JSON with a ``cues:`` list, as tools/match_stem.py writes); ``{reaper: <.rpp>,
    tracks: [...]}`` takes them from the REAPER edit of ``clip``'s sound."""
    if isinstance(cues, dict):
        if not cues.get("reaper"):
            raise ValueError("footage: cues as a mapping needs reaper: <the .rpp>")
        if not clip:
            raise ValueError("footage: cues from reaper: need the layer's file")
        from core.reaper_project import clip_cues
        project = cues["reaper"]
        cues = clip_cues(project, clip, cues.get("tracks"))
        if not cues:
            raise ValueError(f"footage: no item in {project} plays the sound of {clip}")
    if isinstance(cues, (str, Path)):
        import yaml
        data = yaml.safe_load(Path(cues).read_text(encoding="utf-8"))
        cues = data.get("cues") if isinstance(data, dict) else data
    out = []
    for c in cues or []:
        if not isinstance(c, (list, tuple)) or len(c) not in (3, 4):
            raise ValueError(f"footage: a cue is [song s, clip s, seconds(, rate)], not {c!r}")
        at, start, length = (float(v) for v in c[:3])
        if length <= 0:
            raise ValueError(f"footage: cue {list(c)} lasts no time")
        out.append((at, start, length, float(c[3]) if len(c) == 4 else 1.0))
    if not out:
        raise ValueError("footage: needs cues ([song s, clip s, seconds] each)")
    return sorted(out)


IMAGES = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class _Reader:
    """Frames of a video by index (BGR->RGB), read on: a seek only where the index jumps.
    A still image is a one-frame video."""

    def __init__(self, path: str):
        self.cap, self._last = None, (None, None)
        if Path(path).suffix.lower() in IMAGES:
            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(f"footage: cannot open {path}")
            self.fps, self.frames = 24.0, 1
            self.height, self.width = bgr.shape[:2]
            self._last = (0, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            return
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"footage: cannot open {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 24.0
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._next = 0                                  # the index the next read() gives

    @property
    def duration(self) -> float:
        return self.frames / self.fps

    def frame(self, i: int) -> np.ndarray:
        i = max(0, min(self.frames - 1, int(i)))
        if self._last[0] == i:
            return self._last[1]
        if i != self._next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, bgr = self.cap.read()
        if not ok:
            raise RuntimeError(f"footage: no frame {i} (the file has {self.frames})")
        self._next = i + 1
        self._last = (i, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        return self._last[1]


def read_blocks(blocks, grid) -> List[tuple]:
    """``blocks`` (the arrangement: ``{bars: [first, last], from: s, rate: 1, file: path}``,
    DAW bar numbers, both included) as (song s, clip s, seconds, rate, file or None) cues
    on the song's ``grid``."""
    out = []
    for b in blocks or []:
        if not isinstance(b, dict) or not isinstance(b.get("bars"), (list, tuple)) or len(b["bars"]) != 2:
            raise ValueError(f"footage: a block is {{bars: [first, last], from: s}}, not {b!r}")
        if grid is None:
            raise ValueError("footage: blocks: need a rhythm grid (render with --midi)")
        first, last = (int(v) for v in b["bars"])
        if last < first:
            raise ValueError(f"footage: block {b!r} ends before it starts")
        t0, t1 = float(grid.bar_start(first - 1)), float(grid.bar_start(last))
        out.append((t0, float(b.get("from", 0.0)), t1 - t0, float(b.get("rate", 1.0)),
                    str(b["file"]) if b.get("file") else None))
    return out


class Footage:
    """Source layer: see the module docstring."""

    def __init__(self, spec: dict, width: int, height: int, tracks: Optional[dict] = None,
                 notes: Sequence = (), features_at=None, onset_loader=None, grid=None):
        self.file = str(spec["file"]) if spec.get("file") else None
        cues = [] if spec.get("cues") is None else [c + (None,) for c in read_cues(spec["cues"], self.file)]
        cues += read_blocks(spec.get("blocks"), grid)
        if not self.file and any(c[4] is None for c in cues):
            raise ValueError("footage: needs file: <the clip> (or a file: on every block)")
        if not self.file and not cues and spec.get("blocks") is None:   # blocks: [] is an empty lane
            raise ValueError("footage: needs file: <the clip> or blocks with files")
        if spec.get("cues") is None and spec.get("blocks") is None:
            cues = [(0.0, 0.0, 1e9, 1.0, None)]        # no cues: all the song, from the clip's start
        self.cues = sorted(cues, key=lambda c: c[0])
        self._starts = [c[0] for c in self.cues]
        self._readers: dict = {}
        self.reader = self._reader(None) if self.file else None
        self.loop = bool(spec.get("loop", False))
        self.hold = float(spec.get("hold", 0.0))
        self.fit = spec.get("fit", "contain")
        if self.fit not in FITS:
            raise ValueError(f"footage: fit {self.fit!r} (use {', '.join(FITS)})")
        from core.video.layers import _music_knob
        at = spec.get("at", [0.5, 0.5])

        def knob(name, default):
            return _music_knob(name, spec.get(name, default), notes, features_at, onset_loader, grid)
        self._scale = knob("scale", 1.0)
        self._x = knob("x", float(at[0]))
        self._y = knob("y", float(at[1]))
        self.flip = bool(spec.get("flip", False))
        self.W, self.H = width, height
        self.track, self.objects, self.drop = None, None, None
        cut = spec.get("cutout")
        if cut:
            from core.segment.mask_track import MaskTrack
            cut = dict(cut) if isinstance(cut, dict) else {"track": cut}
            folder = cut.get("track")
            if not folder:
                raise ValueError("footage: cutout needs track: <the clip's mask track>")
            tracks = tracks if tracks is not None else {}
            if folder not in tracks:
                tracks[folder] = MaskTrack(folder)
            self.track = tracks[folder]
            objects = cut.get("objects")
            self.objects = None if objects is None else np.array(sorted(int(o) for o in objects))
            drop = cut.get("drop")
            self.drop = None if drop is None else np.array(sorted(int(o) for o in drop))
            self._grow, self._feather = float(cut.get("grow", 0)), float(cut.get("feather", 2))

    def _reader(self, file: Optional[str]) -> "_Reader":
        key = file or self.file
        if key not in self._readers:
            self._readers[key] = _Reader(key)
        return self._readers[key]

    def _cue_at(self, t: float):
        """(clip s, file or None) at song time ``t``, or None when no cue (or hold) covers it."""
        k = bisect.bisect_right(self._starts, t) - 1
        if k < 0:
            return None
        at, start, length, rate, file = self.cues[k]
        if t < at + length:
            ct = start + (t - at) * rate
        elif t < at + length + self.hold:                # bisect_right: the next cue has not begun
            ct = start + length * rate - 0.5 / self._reader(file).fps
        else:
            return None
        if self.loop:
            ct %= self._reader(file).duration
        return ct, file

    def clip_time(self, t: float) -> Optional[float]:
        """Where in its clip song time ``t`` falls, or None when no cue (or hold) covers it."""
        found = self._cue_at(t)
        return None if found is None else found[0]

    def _matte(self, i: int, w: int, h: int) -> np.ndarray:
        lab = self.track.labels(min(i, self.track.frames - 1))
        if self.drop is not None:                       # all but these (a backdrop): what SAM
            m = ~np.isin(lab, self.drop)                # left unnamed stays with the figure,
            unnamed = (lab == 0).astype(np.uint8)       # but not where it touches the frame's edge:
            if unnamed.any():                           # that is backdrop SAM missed (a line there)
                _, cc = cv2.connectedComponents(unnamed)
                edge = np.unique(np.concatenate([cc[0], cc[-1], cc[:, 0], cc[:, -1]]))
                m &= ~np.isin(cc, edge[edge > 0])
        else:
            m = lab > 0 if self.objects is None else np.isin(lab, self.objects)
        m = cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        grow = int(round(self._grow * h / 720.0))
        if grow:
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * abs(grow) + 1,) * 2)
            m = cv2.dilate(m, k) if grow > 0 else cv2.erode(m, k)
        feather = self._feather * h / 720.0
        if feather > 0.25:
            m = cv2.GaussianBlur(m, (0, 0), feather)
        return m

    def _placed(self, t: float):
        """(patch, its alpha or None, (X0, Y0, X1, Y1)) the clip puts in the frame at ``t``,
        or None when nothing shows."""
        found = self._cue_at(t)
        scale = float(self._scale(t))                   # the knobs are asked every frame, in order
        cx, cy = float(self._x(t)) * self.W, float(self._y(t)) * self.H
        if found is None or scale <= 0:
            return None
        ct, file = found
        reader = self._reader(file)
        fit = (min if self.fit == "contain" else max)(self.W / reader.width, self.H / reader.height)
        w = max(1, int(round(reader.width * fit * scale)))
        h = max(1, int(round(reader.height * fit * scale)))
        x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
        X0, Y0, X1, Y1 = max(0, x0), max(0, y0), min(self.W, x0 + w), min(self.H, y0 + h)
        if X1 <= X0 or Y1 <= Y0:                         # all of it outside the frame
            return None
        i = int(np.floor(ct * reader.fps + 1e-6))
        img = cv2.resize(reader.frame(i), (w, h), interpolation=cv2.INTER_AREA
                         if w < reader.width else cv2.INTER_LINEAR)
        own = file is None or file == self.file          # the mask track is the lane's own clip's
        m = self._matte(i, w, h) if self.track is not None and own else None
        if self.flip:
            img = img[:, ::-1]
            m = None if m is None else m[:, ::-1]
        sx, sy = X0 - x0, Y0 - y0
        patch = img[sy:sy + Y1 - Y0, sx:sx + X1 - X0]
        a = None if m is None else m[sy:sy + Y1 - Y0, sx:sx + X1 - X0]
        return patch, a, (X0, Y0, X1, Y1)

    def frame_at_over(self, below: np.ndarray, t: float) -> Optional[np.ndarray]:
        got = self._placed(t)
        if got is None:
            return None
        patch, a, (X0, Y0, X1, Y1) = got
        out = below.copy()
        if a is None:
            out[Y0:Y1, X0:X1] = patch
        else:
            dst = out[Y0:Y1, X0:X1].astype(np.float32)
            out[Y0:Y1, X0:X1] = (dst + (patch.astype(np.float32) - dst) * a[..., None] + 0.5).astype(np.uint8)
        return out

    def frame_rgba(self, t: float):
        """(the clip on black, its alpha 0..1 (H, W)) at ``t``, or None when nothing shows:
        what a track's own FX work on (core/video/track_fx.py)."""
        got = self._placed(t)
        if got is None:
            return None
        patch, a, (X0, Y0, X1, Y1) = got
        rgb = np.zeros((self.H, self.W, 3), np.uint8)
        alpha = np.zeros((self.H, self.W), np.float32)
        rgb[Y0:Y1, X0:X1] = patch if a is None else (patch.astype(np.float32) * a[..., None] + 0.5).astype(np.uint8)
        alpha[Y0:Y1, X0:X1] = 1.0 if a is None else a
        return rgb, alpha

    def frame_at(self, t: float) -> np.ndarray:
        """As the base layer: on black (black also outside the cues)."""
        black = np.zeros((self.H, self.W, 3), np.uint8)
        out = self.frame_at_over(black, t)
        return black if out is None else out

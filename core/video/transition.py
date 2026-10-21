"""The cut as a gesture: the next clip spreads over the last one along a moving front.

The composer knows its cuts before they come (``ClipComposer.cut_times``), so this
post-op keeps the last picture of a shot, and over the first moments of the next one
lets the new picture eat the old one through a pattern. Each pixel has a threshold in
a field (0..1) and switches when the progress passes it, softened over ``soft``. This
is MilkDrop's preset blending (Butterchurn ``blendPattern.js``, MIT): the field is a
plane, a circle or a plasma.

    - source: transition
      pattern: [sweep, figures, plasma]   # one per cut, taken in turn (or a single name):
                                          #   sweep    a straight front, a new angle each cut
                                          #   radial   from the centre out
                                          #   iris     from the edges in
                                          #   plasma   blotches, everywhere at once
                                          #   figures  the figures of the new shot switch first
                                          #            and the world follows them out
                                          #            (needs a mask track: --masks or track:)
                                          #   menger   a Menger carpet's holes open, the biggest
                                          #            first, then the next level, and the next
                                          #   julia    through a Julia set's level curves, the
                                          #            set itself last (a new c each cut)
                                          #   spiral   along the arms of a logarithmic spiral out
                                          #            of the centre (Droste's, as a front)
      order: cycle          # cycle | shuffle (seeded; a bag, no pattern twice in a row)
      length: 1/4           # how long it takes: seconds or a note value (default: a beat)
      soft: 0.3             # the front's width (0 a hard edge)
      seed: 0

Put it right above the clips: the old picture it keeps is everything below it, still
(the clip that went out does not keep playing). Numpy is the reference; the GPU pass
matches it within a level or two.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import cv2
import numpy as np

PATTERNS = ("sweep", "radial", "iris", "plasma", "figures", "menger", "julia", "spiral")

_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_old, u_new, u_field;
uniform float u_p, u_b;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 o = floor(texelFetch(u_old, p, 0).rgb * 255.0 + 0.5);
    vec3 n = floor(texelFetch(u_new, p, 0).rgb * 255.0 + 0.5);
    float f = texelFetch(u_field, p, 0).r;
    float m = clamp((u_p * (1.0 + u_b) - f) / u_b, 0.0, 1.0);
    vec3 v = o * (1.0 - m) + n * m;
    f_color = vec4(floor(clamp(v + 0.5, 0.0, 255.0)) / 255.0, 1.0);
}
"""


def _ranked(field: np.ndarray) -> np.ndarray:
    """``field`` with its values replaced by their rank (0..1): the front then moves at an
    even pace over the picture, whatever the shape of the field."""
    flat = field.ravel()
    out = np.empty(flat.size, np.float32)
    out[np.argsort(flat, kind="stable")] = np.linspace(0.0, 1.0, flat.size, dtype=np.float32)
    return out.reshape(field.shape)


def make_field(pattern: str, w: int, h: int, rng: np.random.Generator,
               figures: Optional[np.ndarray] = None) -> np.ndarray:
    """The thresholds (h×w float32, 0..1) a cut's front sweeps through."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if pattern == "sweep":
        a = rng.uniform(0, 2 * np.pi)
        d = (xx - w / 2) * np.cos(a) + (yy - h / 2) * np.sin(a)
        f = _ranked(d)
    elif pattern in ("radial", "iris"):
        r = np.hypot(xx - w / 2, yy - h / 2)
        f = _ranked(r if pattern == "radial" else -r)
    elif pattern == "menger":
        # in square cells as big as the frame's short side: the level at which a pixel falls
        # in a hole (the middle third both ways), the biggest holes opening first
        side = float(min(w, h))
        u = (xx - (w - side) / 2) / side
        v = (yy - (h - side) / 2) / side
        level = np.full((h, w), 5.0, np.float32)
        for k in range(4, 0, -1):
            a = np.floor(np.mod(u, 1.0) * 3 ** k) % 3
            b = np.floor(np.mod(v, 1.0) * 3 ** k) % 3
            level = np.where((a == 1) & (b == 1), float(k), level)
        # what never falls in a hole goes last, in soft blotches; ranked for an even pace
        blot = cv2.resize(rng.uniform(0, 1, (5, 4)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
        blot = np.clip(blot, 0, 1)                        # also settles ties within a level softly
        f = _ranked(level + 0.3 * blot + 0.6 * blot * (level > 4))
    elif pattern == "spiral":
        # log-polar: radius and angle tilted into each other, a few arms; out from the centre
        r = np.hypot(xx - w / 2, yy - h / 2) / (0.5 * min(w, h)) + 1e-3
        th = np.arctan2(yy - h / 2, xx - w / 2)
        arms = float(rng.integers(2, 5))
        turn = np.mod(np.log(r) * 1.6 - th * arms / (2 * np.pi) + rng.uniform(0, 1), 1.0)
        f = _ranked(np.log(r) + 0.9 * turn)
    elif pattern == "julia":
        ang = rng.uniform(0, 2 * np.pi)
        c = complex(0.7885 * np.cos(ang), 0.7885 * np.sin(ang))
        z = ((xx - w / 2) + 1j * (yy - h / 2)) / (0.5 * min(w, h)) * 1.4
        n = np.zeros(z.shape, np.float32)
        alive = np.ones(z.shape, bool)
        for i in range(48):
            z = np.where(alive, z * z + c, z)
            out = alive & (np.abs(z) > 4.0)
            n[out] = i + 1 - np.log2(np.log(np.abs(z[out])) / np.log(4.0))
            alive &= ~out
        n[alive] = 60.0                                   # the set itself goes last
        f = _ranked(n + rng.uniform(0, 1e-3, n.shape).astype(np.float32))
    elif pattern == "figures" and figures is not None and figures.any():
        # 0 on the figures, rising with the distance from them: they go first
        d = cv2.distanceTransform((~figures).astype(np.uint8), cv2.DIST_L2, 5)
        f = _ranked(d + rng.uniform(0, 1e-3, d.shape).astype(np.float32))
        f = np.where(figures, 0.0, 0.08 + 0.92 * f)
    else:                                                # plasma (and figures with no figures)
        small = rng.uniform(0, 1, (5, 4)).astype(np.float32)
        f = cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)
        f = _ranked(cv2.GaussianBlur(f, (0, 0), max(w, h) / 40.0))
    return np.ascontiguousarray(f, dtype=np.float32)


class Transition:
    """Post-op: see the module docstring."""

    def __init__(self, spec: dict, width: int, height: int, fps: float, grid=None, cuts=None,
                 default_track: Optional[str] = None, tracks: Optional[Dict] = None):
        from core.video.cast import _Role
        if cuts is None:
            raise ValueError("transition: needs the clips' cuts (a scene with - source: clips)")
        pats = spec.get("pattern", "sweep")
        self.patterns = [pats] if isinstance(pats, str) else list(pats)
        bad = [p for p in self.patterns if p not in PATTERNS]
        if bad or not self.patterns:
            raise ValueError(f"transition: pattern {bad or pats!r} (use {', '.join(PATTERNS)})")
        self.order = spec.get("order", "cycle")
        if self.order not in ("cycle", "shuffle"):
            raise ValueError(f"transition: order {self.order!r} (cycle or shuffle)")
        length = spec.get("length", "1/4" if grid is not None else 0.5)
        self.length = _Role._period(length, grid, 0)
        self.soft = max(1e-3, float(spec.get("soft", 0.3)))
        self.seed = int(spec.get("seed", 0))
        self.W, self.H, self.fps = width, height, float(fps)
        self._cuts = cuts
        self._known = (-np.inf, np.zeros(0))
        self.track = None
        if "figures" in self.patterns:
            folder = spec.get("track") or default_track
            if not folder:
                raise ValueError("transition: figures needs a mask track (track: <folder>) or the render's --masks")
            from core.segment.mask_track import MaskTrack
            tracks = tracks if tracks is not None else {}
            if folder not in tracks:
                tracks[folder] = MaskTrack(folder)
            self.track = tracks[folder]
        self._old: Dict[int, object] = {}        # cut index -> the shot's last picture
        self._field: Dict[int, np.ndarray] = {}
        self._uploaded = None                    # (cut index) whose field the GPU holds
        self._keep = None                        # the GPU's copy of the old picture
        self.on_gpu = True

    # ---------------------------------------------------------------- timing
    def _times(self, t: float) -> np.ndarray:
        if t + 600.0 > self._known[0]:
            self._known = (t + 3600.0, np.asarray(self._cuts(t + 3600.0), dtype=float))
        return self._known[1]

    def pattern(self, k: int) -> str:
        """The pattern of cut ``k`` (0 the first cut of the song)."""
        n = len(self.patterns)
        if self.order == "cycle" or n == 1:
            return self.patterns[k % n]
        rounds, i = divmod(k, n)
        bag = list(np.random.default_rng(self.seed + 7919 * rounds).permutation(n))
        return self.patterns[bag[i]]

    def _where(self, t: float):
        """(the cut a transition is running from or None, its progress, keep this frame for
        the cut coming next or None)."""
        ts = self._times(t)
        i = int(np.searchsorted(ts, t + 1e-6, side="right"))       # cuts at or before t
        keep = i if i < len(ts) and t + 1.0 / self.fps >= ts[i] - 1e-6 else None
        if i > 0 and t - ts[i - 1] < self.length:
            p = (t - ts[i - 1]) / self.length
            return i - 1, p * p * (3 - 2 * p), keep
        return None, 0.0, keep

    def _field_of(self, k: int, t: float) -> np.ndarray:
        if k not in self._field:
            rng = np.random.default_rng(self.seed * 1009 + k)
            pat = self.pattern(k)
            figures = None
            if pat == "figures" and self.track is not None:
                lab = self.track.labels_at(t)
                if lab is not None:
                    figures = cv2.resize((lab > 0).astype(np.uint8), (self.W, self.H),
                                         interpolation=cv2.INTER_NEAREST) > 0
            self._field = {k: make_field(pat, self.W, self.H, rng, figures)}   # only the current one
        return self._field[k]

    # --------------------------------------------------------------- drawing
    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        k, p, keep = self._where(t)
        out = frame
        if k is not None and k in self._old:
            f = self._field_of(k, t)
            b = np.float32(self.soft)
            m = np.clip((np.float32(p) * (1 + b) - f) / b, 0.0, 1.0)[..., None]
            v = self._old[k].astype(np.float32) * (1 - m) + frame.astype(np.float32) * m
            out = np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)
        if keep is not None:
            self._old = {keep: frame.copy()}
        elif k is None:
            self._old = {}
        return out

    def process_gpu(self, frame, t: float, g):
        from core.video.gpu_compose import Frame
        k, p, keep = self._where(t)
        out = frame
        if k is not None and k in self._old:
            f = self._field_of(k, t)
            if self._uploaded != k:
                self._ftex = g.small(f, f"transition_field_{id(self)}")
                self._uploaded = k
            tex = g.run("transition", g.texture(), {"u_old": self._keep, "u_new": frame.tex(),
                                                    "u_field": self._ftex}, fs=_FS,
                        u_p=float(np.float32(p)), u_b=float(np.float32(self.soft)))
            out = Frame(g, tex=tex)
        if keep is not None:
            if self._keep is None:
                self._keep = g.keep()
            g.copy(frame.tex(), self._keep)
            self._old = {keep: True}
        elif k is None:
            self._old = {}
        return out

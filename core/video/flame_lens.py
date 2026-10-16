"""The flame's variations as a lens: the clip seen through swirl, bubble, spherical..

A fractal flame bends the plane with its variations (core/flame.py). Here the plane is
the picture: each pixel becomes a point of the plane, goes through a weighted mix of
variations, and takes its colour from where it lands, mirrored at the edges so the
picture never runs out. The shot stays recognisable, pulled through a liquid lens, and
the music sets how much of each variation there is (Draves & Reckase, "The Fractal
Flame Algorithm", the final transform as a camera).

    - source: flame_lens
      variations:                 # each one's share: a number or any knob that follows the
        swirl: {hits: {track: snare}, envelope: 0.6, amount: 0.8}     # music (see the
        bubble: {feature: subbands.bass, gain: 1.5}                   # shadertoy layer's
        spherical: 0.15                                               # uniforms)
      amount: 1.0                 # 0 the plain picture .. 1 the full lens (a knob too)
      zoom: 1.0                   # how much of the plane the picture spans (bigger: calmer)
      turn: {rate: 0.4}           # the plane's rotation in radians (a knob: here it keeps turning)
      center: [0.5, 0.5]          # where the lens sits, as a share of the width and height

Shares adding up to less than 1 leave the rest to the plain picture; more than 1 are
scaled down. The variations: linear, sinusoidal, spherical, swirl, horseshoe, polar,
handkerchief, heart, disc, spiral, hyperbolic, diamond, ex, bent, eyefish, bubble,
cylinder. Numpy is the reference; the GPU matches it within a level or two, except where
a variation folds the plane so hard that a hair's difference lands far away (a few
pixels near the poles of spherical or spiral).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

LENS_VARIATIONS = ("linear", "sinusoidal", "spherical", "swirl", "horseshoe", "polar",
                   "handkerchief", "heart", "disc", "spiral", "hyperbolic", "diamond",
                   "ex", "bent", "eyefish", "bubble", "cylinder")
NV = len(LENS_VARIATIONS)
LIMIT = 1.0e5                     # coordinates past this are folded back as if they were here

_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src;
uniform float u_w[NV];
uniform float u_amount, u_unit, u_turn;
uniform vec2 u_center;
const float PI = 3.14159265358979;

vec2 lens(vec2 q){
    float x = q.x, y = q.y;
    float r2 = x * x + y * y + 1e-12;
    float r = sqrt(r2);
    float th = atan(x, y);
    vec2 o = vec2(0.0);
    o += u_w[0] * q;
    o += u_w[1] * vec2(sin(x), sin(y));
    o += u_w[2] * q / r2;
    o += u_w[3] * vec2(x * sin(r2) - y * cos(r2), x * cos(r2) + y * sin(r2));
    o += u_w[4] * vec2((x - y) * (x + y) / r, 2.0 * x * y / r);
    o += u_w[5] * vec2(th / PI, r - 1.0);
    o += u_w[6] * vec2(r * sin(th + r), r * cos(th - r));
    o += u_w[7] * vec2(r * sin(th * r), -r * cos(th * r));
    o += u_w[8] * vec2(th / PI * sin(PI * r), th / PI * cos(PI * r));
    o += u_w[9] * vec2((cos(th) + sin(r)) / r, (sin(th) - cos(r)) / r);
    o += u_w[10] * vec2(sin(th) / r, r * cos(th));
    o += u_w[11] * vec2(sin(th) * cos(r), cos(th) * sin(r));
    float p0 = pow(sin(th + r), 3.0), p1 = pow(cos(th - r), 3.0);
    o += u_w[12] * vec2(r * (p0 + p1), r * (p0 - p1));
    o += u_w[13] * vec2(x >= 0.0 ? x : 2.0 * x, y >= 0.0 ? y : y / 2.0);
    o += u_w[14] * 2.0 * q / (r + 1.0);
    o += u_w[15] * 4.0 * q / (r2 + 4.0);
    o += u_w[16] * vec2(sin(x), y);
    return o;
}

int fold(int i, int n){            // OpenCV's BORDER_REFLECT: ..cba|abc..xyz|zyx..
    int m = i - 2 * n * int(floor(float(i) / float(2 * n)));   // a modulo that stays >= 0
    return m < n ? m : 2 * n - 1 - m;
}

vec3 texel(int x, int y, ivec2 n){
    return floor(texelFetch(u_src, ivec2(fold(x, n.x), fold(y, n.y)), 0).rgb * 255.0 + 0.5);
}

void main(){
    ivec2 n = textureSize(u_src, 0);
    vec2 pix = gl_FragCoord.xy - 0.5;
    vec2 c = u_center * vec2(n);
    float cs = cos(u_turn), sn = sin(u_turn);
    vec2 p = (pix - c) / u_unit;
    vec2 q = vec2(cs * p.x - sn * p.y, sn * p.x + cs * p.y);
    vec2 v = q + u_amount * (lens(q) - q);
    vec2 back = vec2(cs * v.x + sn * v.y, -sn * v.x + cs * v.y);
    vec2 u = clamp(back * u_unit + c, vec2(-LIMIT), vec2(LIMIT));
    vec2 f0 = floor(u);
    vec2 f = u - f0;
    int x0 = int(f0.x), y0 = int(f0.y);
    vec3 a = texel(x0, y0, n) * (1.0 - f.x) + texel(x0 + 1, y0, n) * f.x;
    vec3 b = texel(x0, y0 + 1, n) * (1.0 - f.x) + texel(x0 + 1, y0 + 1, n) * f.x;
    vec3 s = a * (1.0 - f.y) + b * f.y;
    f_color = vec4(floor(clamp(s + 0.5, 0.0, 255.0)) / 255.0, 1.0);
}
""".replace("NV]", f"{NV}]").replace("LIMIT)", f"{LIMIT:.1f})")


def lens_np(q: np.ndarray, w: np.ndarray) -> np.ndarray:
    """The weighted variations at the points ``q`` (N×2 float32); ``w`` one weight each."""
    x, y = q[:, 0], q[:, 1]
    r2 = x * x + y * y + np.float32(1e-12)
    r = np.sqrt(r2)
    th = np.arctan2(x, y)
    pi = np.float32(np.pi)
    o = np.zeros_like(q)

    def add(j, vx, vy):
        if w[j]:
            o[:, 0] += w[j] * vx
            o[:, 1] += w[j] * vy

    add(0, x, y)
    add(1, np.sin(x), np.sin(y))
    add(2, x / r2, y / r2)
    add(3, x * np.sin(r2) - y * np.cos(r2), x * np.cos(r2) + y * np.sin(r2))
    add(4, (x - y) * (x + y) / r, 2 * x * y / r)
    add(5, th / pi, r - 1)
    add(6, r * np.sin(th + r), r * np.cos(th - r))
    add(7, r * np.sin(th * r), -r * np.cos(th * r))
    add(8, th / pi * np.sin(pi * r), th / pi * np.cos(pi * r))
    add(9, (np.cos(th) + np.sin(r)) / r, (np.sin(th) - np.cos(r)) / r)
    add(10, np.sin(th) / r, r * np.cos(th))
    add(11, np.sin(th) * np.cos(r), np.cos(th) * np.sin(r))
    if w[12]:
        p0, p1 = np.sin(th + r) ** 3, np.cos(th - r) ** 3
        add(12, r * (p0 + p1), r * (p0 - p1))
    add(13, np.where(x >= 0, x, 2 * x), np.where(y >= 0, y, y / 2))
    add(14, 2 * x / (r + 1), 2 * y / (r + 1))
    add(15, 4 * x / (r2 + 4), 4 * y / (r2 + 4))
    add(16, np.sin(x), y)
    return o


def _fold(i: np.ndarray, n: int) -> np.ndarray:
    m = np.mod(i, 2 * n)
    return np.where(m < n, m, 2 * n - 1 - m)


class FlameLens:
    """Post-op: see the module docstring."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: int,
                 features_at=None, onset_loader=None, grid=None):
        from core.video.layers import _music_knob
        knob = lambda name, how: _music_knob(name, how, notes, features_at, onset_loader, grid)
        vs = spec.get("variations") or {}
        if not vs:
            raise ValueError("flame_lens: variations: is empty (e.g. {swirl: 0.5})")
        bad = [v for v in vs if v not in LENS_VARIATIONS]
        if bad:
            raise ValueError(f"flame_lens: unknown variations {bad} (use {', '.join(LENS_VARIATIONS)})")
        self._weights = {LENS_VARIATIONS.index(v): knob(v, how) for v, how in vs.items()}
        self._amount = knob("amount", spec.get("amount", 1.0))
        self._turn = knob("turn", spec.get("turn", 0.0))
        self.zoom = float(spec.get("zoom", 1.0))
        cx, cy = spec.get("center", (0.5, 0.5))
        self.center = (float(cx), float(cy))
        self.W, self.H = width, height
        self.unit = 0.5 * min(width, height) * self.zoom          # pixels per unit of the plane
        self._grid = None
        self.on_gpu = True

    def state(self, t: float):
        """(weights, amount, turn) at ``t``: the shares scaled as the docstring says."""
        w = np.zeros(NV, np.float32)
        for j, f in self._weights.items():
            w[j] = max(0.0, float(f(t)))
        total = float(w.sum())
        if total > 1.0:
            w /= np.float32(total)
        else:
            w[0] += np.float32(1.0 - total)                      # the rest: the plain picture
        amount = float(np.clip(self._amount(t), 0.0, 1.0))
        return w, np.float32(amount), np.float32(self._turn(t))

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        w, amount, turn = self.state(t)
        if amount <= 0.0 or (w[0] == 1.0):
            return frame
        H, W = frame.shape[:2]
        if self._grid is None or self._grid[0].shape != (H, W):
            yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
            self._grid = (xx, yy)
        xx, yy = self._grid
        unit = np.float32(0.5 * min(W, H) * self.zoom)
        cx, cy = np.float32(self.center[0] * W), np.float32(self.center[1] * H)
        cs, sn = np.cos(turn, dtype=np.float32), np.sin(turn, dtype=np.float32)
        px, py = (xx - cx) / unit, (yy - cy) / unit
        q = np.stack([cs * px - sn * py, sn * px + cs * py], -1).reshape(-1, 2)
        v = q + amount * (lens_np(q, w) - q)
        bx = cs * v[:, 0] + sn * v[:, 1]
        by = -sn * v[:, 0] + cs * v[:, 1]
        ux = np.clip(bx * unit + cx, -LIMIT, LIMIT).reshape(H, W)
        uy = np.clip(by * unit + cy, -LIMIT, LIMIT).reshape(H, W)
        x0, y0 = np.floor(ux), np.floor(uy)
        fx, fy = (ux - x0)[..., None], (uy - y0)[..., None]
        x0, y0 = x0.astype(np.int64), y0.astype(np.int64)
        X0, X1 = _fold(x0, W), _fold(x0 + 1, W)
        Y0, Y1 = _fold(y0, H), _fold(y0 + 1, H)
        src = frame.astype(np.float32)
        a = src[Y0, X0] * (1 - fx) + src[Y0, X1] * fx
        b = src[Y1, X0] * (1 - fx) + src[Y1, X1] * fx
        s = a * (1 - fy) + b * fy
        return np.clip(np.floor(s + 0.5), 0, 255).astype(np.uint8)

    def process_gpu(self, frame, t: float, g):
        from core.video.gpu_compose import Frame
        w, amount, turn = self.state(t)
        if amount <= 0.0 or (w[0] == 1.0):
            return frame
        unit = float(np.float32(0.5 * min(g.W, g.H) * self.zoom))
        tex = g.run("flame_lens", g.texture(), {"u_src": frame.tex()}, fs=_FS,
                    u_w=w.astype(np.float32), u_amount=float(amount), u_unit=unit,
                    u_turn=float(turn), u_center=self.center)
        return Frame(g, tex=tex)

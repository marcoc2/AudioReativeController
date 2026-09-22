"""Flame — fractal flames (Scott Draves' algorithm) as a layer source. GPU first.

A flame is a handful of functions ("xforms"): an affine map followed by a blend
of non-linear *variations*, each with a weight and a colour. Play the chaos
game — start anywhere, keep applying an xform picked at random — and the points
pile up on the attractor. The picture is the *histogram* of those points, shown
on a log scale, coloured by which xforms each point came through.

    GPU (moderngl, the normal path)
        compute shader   N threads x K iterations -> a buffer of N*K points
        draw pass        the buffer as GL points, additive blend, into a float texture
        tone pass        log density, gamma, palette -> 8 bit
    CPU (numpy, for the days the card is busy)
        the same three steps with fewer samples; same picture, grainier

Every sample follows a *fixed* random sequence (seeded by its index, not by the
frame), so a point's position is a continuous function of the coefficients: as
the music turns the xforms, the flame flows instead of boiling. Implemented from
the paper ("The Fractal Flame Algorithm", Draves & Reckase); variation numbering
follows its appendix.
"""
from __future__ import annotations

import colorsys
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np

VARIATIONS = ("linear", "sinusoidal", "spherical", "swirl", "horseshoe", "polar",
              "handkerchief", "heart", "disc", "spiral", "hyperbolic", "diamond",
              "ex", "julia", "bent", "eyefish", "bubble", "cylinder")
NVAR = len(VARIATIONS)
MAX_XFORMS = 16
WARMUP = 12              # iterations thrown away while a point falls onto the attractor

# gestures, in screen units (the frame is 2 high) — same vocabulary as core/chladni
PUNCH = 0.06
TWIST = 0.07
SIZZLE = 0.012
RING_LIFE = 0.6
RING_SPEED = 3.2
RING_WIDTH = 0.12
RING_PUSH = 0.06
MAX_RINGS = 4


# ---------------------------------------------------------------------------
# genome
# ---------------------------------------------------------------------------
@dataclass
class Xform:
    affine: Sequence[float]                  # a, b, c, d, e, f:  x' = ax + by + c ; y' = dx + ey + f
    variations: dict = field(default_factory=lambda: {"linear": 1.0})
    weight: float = 1.0
    color: float = 0.0                       # where on the palette this xform pulls a point
    color_speed: float = 0.5
    spin: float = 1.0                        # how much (and which way) the animation turns it


@dataclass
class Genome:
    xforms: List[Xform]

    @staticmethod
    def from_spec(spec) -> "Genome":
        """``{random: seed}`` or ``{xforms: [{affine: [...], variations: {...}, ...}, ...]}``."""
        if "random" in spec:
            return random_genome(int(spec["random"]))
        xs = [Xform(affine=[float(v) for v in x["affine"]],
                    variations={k: float(v) for k, v in (x.get("variations") or {"linear": 1.0}).items()},
                    weight=float(x.get("weight", 1.0)), color=float(x.get("color", 0.0)),
                    color_speed=float(x.get("color_speed", 0.5)), spin=float(x.get("spin", 1.0)))
              for x in spec["xforms"]]
        return Genome(xs)

    def with_symmetry(self, order: int) -> "Genome":
        """Rotational symmetry the flam3 way: extra xforms that only rotate, and leave the colour alone."""
        order = int(order)
        if order <= 1:
            return self
        extra = []
        for k in range(1, order):
            a = 2 * np.pi * k / order
            extra.append(Xform(affine=[np.cos(a), -np.sin(a), 0.0, np.sin(a), np.cos(a), 0.0],
                               weight=sum(x.weight for x in self.xforms) / len(self.xforms),
                               color_speed=0.0, spin=0.0))
        return Genome(list(self.xforms) + extra)


_RANDOM_POOL = ("sinusoidal", "spherical", "swirl", "horseshoe", "polar", "handkerchief", "heart",
                "disc", "spiral", "hyperbolic", "diamond", "ex", "julia", "bent", "eyefish",
                "bubble", "cylinder")


def random_genome(seed: int) -> Genome:
    """A random flame in the spirit of Apophysis' random batch; the seed *is* the preset name."""
    rng = np.random.default_rng([int(seed), 0xF1A3E])
    n = int(rng.integers(3, 5))
    xs = []
    for i in range(n):
        ang, skew = rng.uniform(0, 2 * np.pi), rng.uniform(-0.5, 0.5)
        sx, sy = rng.uniform(0.35, 0.95, size=2)
        a, b = sx * np.cos(ang), -sy * np.sin(ang + skew)
        d, e = sx * np.sin(ang), sy * np.cos(ang + skew)
        c, f = rng.uniform(-1.0, 1.0, size=2)
        var = {"linear": float(rng.uniform(0.0, 0.6))}
        for name in rng.choice(_RANDOM_POOL, size=int(rng.integers(1, 3)), replace=False):
            var[str(name)] = float(rng.uniform(0.4, 1.0))
        xs.append(Xform(affine=[a, b, c, d, e, f], variations=var,
                        weight=float(rng.uniform(0.5, 1.5)), color=i / max(1, n - 1),
                        spin=float(rng.choice([-1.0, 1.0]) * rng.uniform(0.5, 1.5))))
    return Genome(xs)


@dataclass
class Packed:
    """A genome in a given pose, as the flat arrays both back ends read."""
    n: int
    aff0: np.ndarray      # (MAX, 4): a, b, c, _
    aff1: np.ndarray      # (MAX, 4): d, e, f, _
    meta: np.ndarray      # (MAX, 4): cumulative weight (0..1), colour, colour speed, _
    var: np.ndarray       # (MAX, NVAR)


def pack(genome: Genome, phase: float = 0.0, pose: float = 0.0) -> Packed:
    """Turn every xform by ``spin * phase`` (the animation) and the first two by
    ``pose`` in opposite senses (the chord's figure), and flatten."""
    xs = genome.xforms[:MAX_XFORMS]
    aff0, aff1 = np.zeros((MAX_XFORMS, 4), "f4"), np.zeros((MAX_XFORMS, 4), "f4")
    meta, var = np.zeros((MAX_XFORMS, 4), "f4"), np.zeros((MAX_XFORMS, NVAR), "f4")
    total = sum(max(0.0, x.weight) for x in xs) or 1.0
    cum = 0.0
    for i, x in enumerate(xs):
        a, b, c, d, e, f = (float(v) for v in x.affine)
        th = x.spin * phase + (pose if i == 0 else -0.6 * pose if i == 1 else 0.0)
        co, si = np.cos(th), np.sin(th)
        aff0[i, :3] = (co * a - si * d, co * b - si * e, c)
        aff1[i, :3] = (si * a + co * d, si * b + co * e, f)
        cum += max(0.0, x.weight) / total
        meta[i, :3] = (cum, x.color, x.color_speed)
        for name, w in x.variations.items():
            var[i, VARIATIONS.index(name)] = w
    meta[len(xs) - 1, 0] = 1.0
    return Packed(len(xs), aff0, aff1, meta, var)


def make_palette(hue: float, n: int = 256) -> np.ndarray:
    """(n, 3) float 0..1: deep -> the chord's colour -> pale -> a warm accent."""
    stops = [colorsys.hsv_to_rgb((hue - 0.10) % 1, 0.95, 0.55), colorsys.hsv_to_rgb(hue % 1, 0.85, 1.0),
             colorsys.hsv_to_rgb((hue + 0.05) % 1, 0.25, 1.0), colorsys.hsv_to_rgb((hue + 0.30) % 1, 0.85, 1.0)]
    xs = np.linspace(0, 1, n)
    at = np.linspace(0, 1, len(stops))
    return np.stack([np.interp(xs, at, [s[ch] for s in stops]) for ch in range(3)], axis=1).astype("f4")


# ---------------------------------------------------------------------------
# the chaos game on the CPU (fallback back end, camera fitting, and the reference for tests)
# ---------------------------------------------------------------------------
def _variations_np(q: np.ndarray, w: np.ndarray, rng) -> np.ndarray:
    x, y = q[:, 0], q[:, 1]
    r2 = x * x + y * y + 1e-12
    r = np.sqrt(r2)
    th = np.arctan2(x, y)
    out = np.zeros_like(q)

    def add(j, vx, vy):
        out[:, 0] += w[:, j] * vx
        out[:, 1] += w[:, j] * vy

    used = w.any(axis=0)
    if used[0]: add(0, x, y)
    if used[1]: add(1, np.sin(x), np.sin(y))
    if used[2]: add(2, x / r2, y / r2)
    if used[3]: add(3, x * np.sin(r2) - y * np.cos(r2), x * np.cos(r2) + y * np.sin(r2))
    if used[4]: add(4, (x - y) * (x + y) / r, 2 * x * y / r)
    if used[5]: add(5, th / np.pi, r - 1)
    if used[6]: add(6, r * np.sin(th + r), r * np.cos(th - r))
    if used[7]: add(7, r * np.sin(th * r), -r * np.cos(th * r))
    if used[8]: add(8, th / np.pi * np.sin(np.pi * r), th / np.pi * np.cos(np.pi * r))
    if used[9]: add(9, (np.cos(th) + np.sin(r)) / r, (np.sin(th) - np.cos(r)) / r)
    if used[10]: add(10, np.sin(th) / r, r * np.cos(th))
    if used[11]: add(11, np.sin(th) * np.cos(r), np.cos(th) * np.sin(r))
    if used[12]:
        p0, p1 = np.sin(th + r) ** 3, np.cos(th - r) ** 3
        add(12, r * (p0 + p1), r * (p0 - p1))
    if used[13]:
        om = np.pi * (rng.random(len(x)) < 0.5)
        add(13, np.sqrt(r) * np.cos(th / 2 + om), np.sqrt(r) * np.sin(th / 2 + om))
    if used[14]: add(14, np.where(x >= 0, x, 2 * x), np.where(y >= 0, y, y / 2))
    if used[15]: add(15, 2 * x / (r + 1), 2 * y / (r + 1))
    if used[16]: add(16, 4 * x / (r2 + 4), 4 * y / (r2 + 4))
    if used[17]: add(17, np.sin(x), y)
    return out


def iterate_cpu(pk: Packed, threads: int, keep: int, seed: int = 0):
    """``threads`` points, ``keep`` kept iterations each -> (points (M, 2), colour (M,), valid (M,))."""
    rng = np.random.default_rng([int(seed), 0xC4A05])
    p = rng.uniform(-1, 1, size=(threads, 2))
    c = rng.random(threads)
    cum = pk.meta[:pk.n, 0].astype(float)
    pts, cols, oks = [], [], []
    with np.errstate(all="ignore"):
        for i in range(WARMUP + keep):
            k = np.minimum(np.searchsorted(cum, rng.random(threads), side="left"), pk.n - 1)
            q = np.stack([pk.aff0[k, 0] * p[:, 0] + pk.aff0[k, 1] * p[:, 1] + pk.aff0[k, 2],
                          pk.aff1[k, 0] * p[:, 0] + pk.aff1[k, 1] * p[:, 1] + pk.aff1[k, 2]], axis=1)
            p = _variations_np(q, pk.var[k].astype(float), rng)
            c = c + (pk.meta[k, 1] - c) * pk.meta[k, 2]
            bad = ~np.isfinite(p).all(axis=1) | ((p * p).sum(axis=1) > 1e10)
            fresh = rng.uniform(-1, 1, size=(threads, 2))          # drawn every time: keeps the sequence fixed
            p = np.where(bad[:, None], fresh, p)
            if i >= WARMUP:
                pts.append(p.copy()); cols.append(c.copy()); oks.append(~bad)
    return np.concatenate(pts), np.concatenate(cols), np.concatenate(oks)


def fit_camera(pk: Packed, aspect: float, seed: int = 0, centered: bool = False) -> tuple:
    """(cx, cy, scale) that frames the bulk of the attractor (2nd–98th percentile).
    ``centered``: a figure with rotational symmetry sits on the origin — keep it
    there and fit its radius to the frame height."""
    pts, _, ok = iterate_cpu(pk, 3000, 8, seed)
    pts = pts[ok]
    if len(pts) < 50:
        return 0.0, 0.0, 1.0
    if centered:
        return 0.0, 0.0, float(0.92 / max(1e-3, np.percentile(np.sqrt((pts * pts).sum(axis=1)), 90)))
    lo, hi = np.percentile(pts, 2, axis=0), np.percentile(pts, 98, axis=0)
    half = np.maximum((hi - lo) / 2, 1e-3)
    scale = 0.82 / max(half[0] / aspect, half[1])
    return float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), float(scale)


def _tone(count, rgb_sum, norm, exposure, white, gamma, gain):
    d = count * norm
    lum = np.clip(np.log1p(exposure * d) / np.log1p(exposure * white), 0, 1) ** (1 / gamma)
    rgb = rgb_sum / np.maximum(count, 1e-6)[..., None] * (lum * gain)[..., None]
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------
_CS = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Points { vec4 pts[]; };
#define MAXX %(maxx)d
#define NVAR %(nvar)d
#define PI 3.14159265358979
uniform int u_nx;
uniform int u_keep;
uniform int u_warm;
uniform uint u_seed;
uniform uint u_threads;
uniform vec4 u_aff0[MAXX];
uniform vec4 u_aff1[MAXX];
uniform vec4 u_meta[MAXX];
uniform float u_var[MAXX * NVAR];

uint pcg(inout uint s) {
    s = s * 747796405u + 2891336453u;
    uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
    return (w >> 22u) ^ w;
}
float rnd(inout uint s) { return float(pcg(s)) * (1.0 / 4294967296.0); }

vec2 variations(vec2 q, int k, inout uint s) {
    float x = q.x, y = q.y;
    float r2 = x * x + y * y + 1e-12, r = sqrt(r2), th = atan(x, y);
    float om = PI * float(rnd(s) < 0.5);          // drawn always: keeps the sequence fixed
    int o = k * NVAR;
    vec2 v = vec2(0.0);
    float w;
    w = u_var[o + 0];  if (w != 0.0) v += w * q;
    w = u_var[o + 1];  if (w != 0.0) v += w * sin(q);
    w = u_var[o + 2];  if (w != 0.0) v += w * q / r2;
    w = u_var[o + 3];  if (w != 0.0) v += w * vec2(x * sin(r2) - y * cos(r2), x * cos(r2) + y * sin(r2));
    w = u_var[o + 4];  if (w != 0.0) v += w * vec2((x - y) * (x + y), 2.0 * x * y) / r;
    w = u_var[o + 5];  if (w != 0.0) v += w * vec2(th / PI, r - 1.0);
    w = u_var[o + 6];  if (w != 0.0) v += w * r * vec2(sin(th + r), cos(th - r));
    w = u_var[o + 7];  if (w != 0.0) v += w * r * vec2(sin(th * r), -cos(th * r));
    w = u_var[o + 8];  if (w != 0.0) v += w * (th / PI) * vec2(sin(PI * r), cos(PI * r));
    w = u_var[o + 9];  if (w != 0.0) v += w * vec2(cos(th) + sin(r), sin(th) - cos(r)) / r;
    w = u_var[o + 10]; if (w != 0.0) v += w * vec2(sin(th) / r, r * cos(th));
    w = u_var[o + 11]; if (w != 0.0) v += w * vec2(sin(th) * cos(r), cos(th) * sin(r));
    w = u_var[o + 12]; if (w != 0.0) { float p0 = pow(sin(th + r), 3.0), p1 = pow(cos(th - r), 3.0);
                                       v += w * r * vec2(p0 + p1, p0 - p1); }
    w = u_var[o + 13]; if (w != 0.0) v += w * sqrt(r) * vec2(cos(th / 2.0 + om), sin(th / 2.0 + om));
    w = u_var[o + 14]; if (w != 0.0) v += w * vec2(x >= 0.0 ? x : 2.0 * x, y >= 0.0 ? y : y / 2.0);
    w = u_var[o + 15]; if (w != 0.0) v += w * 2.0 * q / (r + 1.0);
    w = u_var[o + 16]; if (w != 0.0) v += w * 4.0 * q / (r2 + 4.0);
    w = u_var[o + 17]; if (w != 0.0) v += w * vec2(sin(x), y);
    return v;
}

void main() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= u_threads) return;
    uint s = id * 1664525u + u_seed * 1013904223u + 12345u;
    pcg(s);
    vec2 p = vec2(rnd(s), rnd(s)) * 2.0 - 1.0;
    float c = rnd(s);
    for (int i = 0; i < u_warm + u_keep; i++) {
        float pick = rnd(s);
        int k = 0;
        for (int j = 0; j < u_nx - 1; j++) if (pick > u_meta[j].x) k = j + 1;
        vec2 q = vec2(dot(u_aff0[k].xy, p) + u_aff0[k].z, dot(u_aff1[k].xy, p) + u_aff1[k].z);
        p = variations(q, k, s);
        c = mix(c, u_meta[k].y, u_meta[k].z);
        vec2 fresh = vec2(rnd(s), rnd(s)) * 2.0 - 1.0;
        bool bad = any(isnan(p)) || any(isinf(p)) || dot(p, p) > 1e10;
        if (bad) p = fresh;
        if (i >= u_warm) pts[id * uint(u_keep) + uint(i - u_warm)] = bad ? vec4(0.0) : vec4(p, c, 1.0);
    }
}
""" % {"maxx": MAX_XFORMS, "nvar": NVAR}

_DRAW_VS = """
#version 430
in vec4 in_pt;
uniform vec4 u_cam;        // centre x, centre y, scale, rotation
uniform float u_aspect;
uniform vec3 u_gest;       // punch, twist (signed), sizzle
uniform float u_rings[%(rings)d];   // age in seconds, negative = none
out float v_c;
out float v_heat;

float hash(uint n) { n = (n << 13u) ^ n; n = n * (n * n * 15731u + 789221u) + 1376312589u;
                     return float(n & 0x7fffffffu) / float(0x7fffffff); }

void main() {
    vec2 q = (in_pt.xy - u_cam.xy) * u_cam.z;
    float a = u_cam.w + %(twist)f * u_gest.y;
    q = vec2(q.x * cos(a) - q.y * sin(a), q.x * sin(a) + q.y * cos(a));
    float r = length(q) + 1e-4, heat = u_gest.z;
    float push = %(punch)f * u_gest.x * r;
    for (int i = 0; i < %(rings)d; i++) {
        float age = u_rings[i];
        if (age >= 0.0 && age < %(life)f) {
            float band = exp(-pow((r - age * %(speed)f) / %(width)f, 2.0)) * (1.0 - age / %(life)f);
            push += %(rpush)f * band;
            heat += band;
        }
    }
    q *= 1.0 + push / r;
    if (u_gest.z > 0.0) {
        uint id = uint(gl_VertexID);
        float ang = 6.2831853 * hash(id), mag = sqrt(-2.0 * log(max(1e-6, hash(id * 7919u + 13u))));
        q += %(sizzle)f * u_gest.z * mag * vec2(cos(ang), sin(ang));
    }
    v_c = in_pt.z;
    v_heat = clamp(heat, 0.0, 1.0);
    gl_Position = in_pt.w > 0.5 ? vec4(q.x / u_aspect, q.y, 0.0, 1.0) : vec4(4.0, 4.0, 4.0, 1.0);
}
""" % {"rings": MAX_RINGS, "twist": TWIST, "punch": PUNCH, "life": RING_LIFE, "speed": RING_SPEED,
       "width": RING_WIDTH, "rpush": RING_PUSH, "sizzle": SIZZLE}

_DRAW_FS = """
#version 430
uniform sampler2D u_pal;
uniform sampler2D u_paint;     // the picture below the layer, when the flame is painted with it
uniform vec2 u_size;           // accumulation buffer size in pixels
uniform float u_paint_mix;
uniform float u_paint_gain;    // the layers below may have been dimmed on purpose: brighten what is sampled
uniform vec3 u_accent;
in float v_c;
in float v_heat;
out vec4 f_acc;
void main() {
    vec3 col = texture(u_pal, vec2(clamp(v_c, 0.002, 0.998), 0.5)).rgb;
    if (u_paint_mix > 0.0) col = mix(col, texture(u_paint, gl_FragCoord.xy / u_size).rgb * u_paint_gain, u_paint_mix);
    f_acc = vec4(mix(col, u_accent, v_heat), 1.0);
}
"""

_QUAD_VS = """
#version 430
in vec2 in_pos;
out vec2 v_uv;
void main() { v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_TONE_FS = """
#version 430
uniform sampler2D u_acc;
uniform int u_ss;
uniform float u_norm;
uniform float u_exposure;
uniform float u_white;
uniform float u_gamma;
uniform float u_gain;
in vec2 v_uv;
out vec4 f_col;
void main() {
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ss;
    vec4 acc = vec4(0.0);
    for (int j = 0; j < u_ss; j++) for (int i = 0; i < u_ss; i++) acc += texelFetch(u_acc, base + ivec2(i, j), 0);
    acc /= float(u_ss * u_ss);
    float lum = clamp(log(1.0 + u_exposure * acc.a * u_norm) / log(1.0 + u_exposure * u_white), 0.0, 1.0);
    lum = pow(lum, 1.0 / u_gamma);
    f_col = vec4(acc.rgb / max(acc.a, 1e-6) * lum * u_gain, 1.0);
}
"""


class _GpuBackend:
    def __init__(self, width, height, threads, keep, supersample, seed):
        import moderngl
        self._mgl = moderngl
        self.W, self.H, self.ss = width, height, max(1, int(supersample))
        self.threads, self.keep, self.seed = int(threads), int(keep), int(seed)
        self.ctx = ctx = moderngl.create_standalone_context(require=430)
        self.cs = ctx.compute_shader(_CS)
        self.points = ctx.buffer(reserve=self.threads * self.keep * 16)
        self.draw = ctx.program(vertex_shader=_DRAW_VS, fragment_shader=_DRAW_FS)
        self.draw_vao = ctx.vertex_array(self.draw, [(self.points, "4f", "in_pt")])
        self.acc = ctx.texture((self.W * self.ss, self.H * self.ss), 4, dtype="f4")
        self.acc_fbo = ctx.framebuffer([self.acc])
        self.pal = ctx.texture((256, 1), 3, dtype="f4")
        self.pal.repeat_x = False
        self.paint = ctx.texture((self.W, self.H), 3)        # uint8 picture from the layers below
        self.tone = ctx.program(vertex_shader=_QUAD_VS, fragment_shader=_TONE_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._quad = ctx.buffer(quad.tobytes())
        self.tone_vao = ctx.vertex_array(self.tone, [(self._quad, "2f", "in_pos")])
        self.out_fbo = ctx.simple_framebuffer((self.W, self.H), 3)

    @staticmethod
    def _set(prog, name, value):
        if name in prog:                      # the compiler drops what a shader ends up not using
            prog[name].value = value

    @staticmethod
    def _write(prog, name, array):
        if name in prog:
            prog[name].write(np.ascontiguousarray(array, dtype="f4").tobytes())

    def render(self, pk, cam, palette, accent, gest, rings, exposure, white, gamma, gain,
               paint=None, paint_mix=0.0, paint_gain=1.0):
        ctx, mgl = self.ctx, self._mgl
        cs = self.cs
        self._set(cs, "u_nx", pk.n); self._set(cs, "u_keep", self.keep); self._set(cs, "u_warm", WARMUP)
        self._set(cs, "u_seed", self.seed); self._set(cs, "u_threads", self.threads)
        self._write(cs, "u_aff0", pk.aff0); self._write(cs, "u_aff1", pk.aff1)
        self._write(cs, "u_meta", pk.meta); self._write(cs, "u_var", pk.var)
        self.points.bind_to_storage_buffer(0)
        cs.run(group_x=(self.threads + 255) // 256)
        ctx.memory_barrier()

        self.pal.write(np.ascontiguousarray(palette, dtype="f4").tobytes())
        self.acc_fbo.use()
        self.acc_fbo.clear(0.0, 0.0, 0.0, 0.0)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.ONE, mgl.ONE
        self.pal.use(0)
        self._set(self.draw, "u_pal", 0)
        if paint is not None and paint_mix > 0:
            self.paint.write(np.ascontiguousarray(paint[::-1]).tobytes())   # GL rows run bottom-up
        self.paint.use(1)
        self._set(self.draw, "u_paint", 1)
        self._set(self.draw, "u_size", (float(self.W * self.ss), float(self.H * self.ss)))
        self._set(self.draw, "u_paint_mix", float(paint_mix) if paint is not None else 0.0)
        self._set(self.draw, "u_paint_gain", float(paint_gain))
        self._set(self.draw, "u_cam", tuple(cam))
        self._set(self.draw, "u_aspect", self.W / self.H)
        self._set(self.draw, "u_gest", tuple(gest))
        self._set(self.draw, "u_accent", tuple(float(v) for v in accent))
        self._write(self.draw, "u_rings", (list(rings) + [-1.0] * MAX_RINGS)[:MAX_RINGS])
        self.draw_vao.render(mgl.POINTS)
        ctx.disable(mgl.BLEND)

        self.out_fbo.use()
        self.acc.use(0)
        self._set(self.tone, "u_acc", 0)
        self._set(self.tone, "u_ss", self.ss)
        self._set(self.tone, "u_norm", (self.W * self.ss * self.H * self.ss) / (self.threads * self.keep))
        self._set(self.tone, "u_exposure", float(exposure)); self._set(self.tone, "u_white", float(white))
        self._set(self.tone, "u_gamma", float(gamma)); self._set(self.tone, "u_gain", float(gain))
        self.tone_vao.render(mgl.TRIANGLE_STRIP)
        data = self.out_fbo.read(components=3)
        return np.frombuffer(data, dtype=np.uint8).reshape(self.H, self.W, 3)[::-1].copy()


class _CpuBackend:
    def __init__(self, width, height, threads, keep, supersample, seed):
        self.W, self.H = width, height
        self.threads, self.keep, self.seed = int(threads), int(keep), int(seed)

    def render(self, pk, cam, palette, accent, gest, rings, exposure, white, gamma, gain,
               paint=None, paint_mix=0.0, paint_gain=1.0):
        W, H, aspect = self.W, self.H, self.W / self.H
        pts, col, ok = iterate_cpu(pk, self.threads, self.keep, self.seed)
        punch, twist, sizzle = gest
        q = (pts - np.array(cam[:2])) * cam[2]
        a = cam[3] + TWIST * twist
        q = np.stack([q[:, 0] * np.cos(a) - q[:, 1] * np.sin(a), q[:, 0] * np.sin(a) + q[:, 1] * np.cos(a)], axis=1)
        r = np.sqrt((q * q).sum(axis=1)) + 1e-4
        push, heat = PUNCH * punch * r, np.full(len(q), float(sizzle))
        for age in rings:
            if 0.0 <= age < RING_LIFE:
                band = np.exp(-(((r - age * RING_SPEED) / RING_WIDTH) ** 2)) * (1 - age / RING_LIFE)
                push, heat = push + RING_PUSH * band, heat + band
        q = q * (1 + push / r)[:, None]
        if sizzle > 0:
            q = q + np.random.default_rng([self.seed, 1]).standard_normal(q.shape) * SIZZLE * sizzle
        ix = np.floor((q[:, 0] / aspect + 1) * 0.5 * W).astype(np.int64)
        iy = np.floor((1 - (q[:, 1] + 1) * 0.5) * H).astype(np.int64)
        ok = ok & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
        idx, heat = (iy * W + ix)[ok], np.clip(heat[ok], 0, 1)[:, None]
        rgb = palette[np.clip((col[ok] * 255).astype(int), 0, 255)]
        if paint is not None and paint_mix > 0:
            rgb = rgb * (1 - paint_mix) + paint[iy[ok], ix[ok]].astype("f4") / 255.0 * paint_gain * paint_mix
        rgb = rgb * (1 - heat) + np.asarray(accent) * heat
        count = np.bincount(idx, minlength=W * H).astype("f4")
        rgb_sum = np.stack([np.bincount(idx, weights=rgb[:, ch], minlength=W * H) for ch in range(3)], axis=1)
        img = _tone(count, rgb_sum.astype("f4"), (W * H) / (self.threads * self.keep), exposure, white, gamma,
                    np.full(W * H, gain, "f4"))
        return img.reshape(H, W, 3)


# ---------------------------------------------------------------------------
class FlameSystem:
    """``backend``: ``gpu``, ``cpu`` or ``auto`` (GPU when a 4.3 context can be had)."""

    KEEP = 32               # kept iterations per thread
    FIT_EVERY = 6           # frames between camera fits

    def __init__(self, width: int, height: int, genome: Genome, seed: int = 0, samples: float = 6.0,
                 supersample: int = 2, backend: str = "auto", exposure: float = 3.0,
                 white: float = 40.0, gamma: float = 2.4, centered: bool = False):
        self.width, self.height = int(width), int(height)
        self.genome, self.seed = genome, int(seed)
        self.exposure, self.white, self.gamma = float(exposure), float(white), float(gamma)
        self.centered = bool(centered)
        self.backend_name, self._backend = self._open(backend, samples, supersample)
        self._cam: Optional[np.ndarray] = None
        self._frame = 0

    def _open(self, backend, samples, supersample):
        if backend not in ("auto", "gpu", "cpu"):
            raise ValueError(f"flame: unknown backend {backend!r}")
        if backend != "cpu":
            try:
                threads = max(256, int(samples * 1e6 / self.KEEP))
                return "gpu", _GpuBackend(self.width, self.height, threads, self.KEEP, supersample, self.seed)
            except Exception:
                if backend == "gpu":
                    raise
        threads = max(256, int(min(samples, 1.5) * 1e6 / self.KEEP))     # the CPU gets a smaller budget
        return "cpu", _CpuBackend(self.width, self.height, threads, self.KEEP, 1, self.seed)

    def render(self, dt: float, phase: float = 0.0, pose: float = 0.0, hue: float = 0.8,
               level: float = 1.0, zoom: float = 1.0, bloom: float = 0.0, punch: float = 0.0,
               twist: float = 0.0, sizzle: float = 0.0, ring_ages=(),
               paint: Optional[np.ndarray] = None, paint_mix: float = 0.0,
               paint_gain: float = 1.0) -> np.ndarray:
        """Draw the genome turned to ``phase``/``pose`` (uint8, H x W x 3). The camera
        follows the attractor with a lag, so a pose change reframes without a cut.
        ``paint`` (H x W x 3 uint8) colours each point with the picture under it."""
        pk = pack(self.genome, phase, pose)
        if self._frame % self.FIT_EVERY == 0:             # the fit runs on the CPU; the camera lags anyway
            self._target = np.array(fit_camera(pk, self.width / self.height, self.seed, self.centered))
        self._frame += 1
        if self._cam is None:
            self._cam = self._target.copy()
        else:
            self._cam += (self._target - self._cam) * (1 - np.exp(-max(0.0, dt) / 1.5))
        cam = (self._cam[0], self._cam[1], self._cam[2] * float(zoom), 0.0)
        accent = colorsys.hsv_to_rgb((hue + 0.30) % 1.0, 0.85, 1.0)
        gain = (0.30 + 0.70 * float(np.clip(level, 0, 1))) * (1 + 0.4 * punch + 0.5 * bloom)
        return self._backend.render(pk, cam, make_palette(hue), accent, (float(punch), float(twist), float(sizzle)),
                                    [float(a) for a in ring_ages][-MAX_RINGS:], self.exposure, self.white,
                                    self.gamma, gain, paint=paint, paint_mix=float(paint_mix),
                                    paint_gain=float(paint_gain))

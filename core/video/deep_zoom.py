"""A zoom into the Mandelbrot set as long as the song, by perturbation.

The camera falls towards one point of the set's edge, steadily, a decade at a time, from
the whole set down to ~1e-13 across the frame: one continuous dive, not a loop. Floats of
32 bits run out near 1e-6, so the deep frames use perturbation (K. I. Martin; Zhuoran's
rebasing): the orbit of the target point is worked out once, on the CPU in 64 bits, and
every pixel iterates only its small difference from it on the GPU,
d -> 2 Z d + d^2 + dc, moving back onto the reference (rebasing) whenever its own value
gets smaller than the difference.

    - source: deep_zoom
      target: [-0.743643887037151, 0.131825904205330]   # where it falls (default: seahorse valley)
      from: 0                 # song seconds where the dive starts (the whole set in view)
      decades: 0.5            # how many tenfold zooms per bar (or per 2 s without a grid)
      depth: 13               # the deepest it goes, in decades (it stays there)
      iterations: 2000        # the most a pixel tries (deeper needs more)
      hue: 0.6                # the palette's colour, 0..1 (a knob: e.g. harmony:)
      flow: {rate: 0.0, base: 0.05}   # how fast the palette runs along the iterations (a knob)
      kick: {hits: {track: kick}, envelope: 0.3}   # a jolt forward on each hit (a knob, in decades x 0.2)

GPU (a fragment pass, GL 3.3); the reference is a float texture.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

SEAHORSE = (-0.743643887037151, 0.131825904205330)

_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_ref;          // the target's orbit Z_n (r, g), one texel each
uniform int u_n;                  // its length (it may have escaped earlier than u_iter)
uniform int u_iter;
uniform vec2 u_size;
uniform float u_scale;            // the half-height of the frame in the plane (as a float: fine to ~1e-30)
uniform float u_hue, u_flow;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}
vec2 ref(int i) { return texelFetch(u_ref, ivec2(i, 0), 0).rg; }

void main() {
    vec2 p = (gl_FragCoord.xy - 0.5 * u_size) / (0.5 * u_size.y);
    vec2 dc = p * u_scale;
    vec2 d = vec2(0.0);
    int m = 0;                                     // where on the reference this pixel is
    float n = 0.0;
    bool out_ = false;
    vec2 z = vec2(0.0);
    for (int i = 0; i < u_iter; i++) {
        vec2 Z = ref(m);
        d = vec2(2.0 * (Z.x * d.x - Z.y * d.y) + d.x * d.x - d.y * d.y,
                 2.0 * (Z.x * d.y + Z.y * d.x) + 2.0 * d.x * d.y) + dc;
        m++;
        z = ref(m) + d;
        float zz = dot(z, z);
        if (zz > 256.0) { out_ = true; n = float(i) + 1.0 - log2(log(zz) * 0.5 / log(2.0)); break; }
        if (zz < dot(d, d) || m >= u_n - 1) { d = z; m = 0; }   // rebase onto the reference's start
    }
    vec3 col = vec3(0.0);
    if (out_) {
        float k = sqrt(n) * 0.35 + u_flow;
        col = hsv(u_hue + 0.08 * sin(k), 0.55 + 0.3 * sin(k * 1.7), 0.5 + 0.5 * cos(k * 1.3));
        col *= 0.4 + 0.6 * clamp(n / 60.0, 0.0, 1.0);
    }
    f_color = vec4(col, 1.0);
}
"""


def reference_orbit(c: complex, iterations: int) -> np.ndarray:
    """The orbit of ``c`` (float64 on the CPU, stored as float32 pairs), until it escapes."""
    z = 0j
    out = [0j]
    for _ in range(iterations + 1):
        z = z * z + c
        out.append(z)
        if abs(z) > 1e3:
            break
    return np.array([[v.real, v.imag] for v in out], np.float32)


class DeepZoomLayer:
    """Source: see the module docstring."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: int,
                 features_at=None, onset_loader=None, grid=None):
        from core.gpu import context
        from core.shader_pass import VERTEX_SHADER
        from core.video.layers import _music_knob
        self.W, self.H = int(width), int(height)
        tx, ty = spec.get("target", SEAHORSE)
        self.iterations = int(spec.get("iterations", 2000))
        self.start = float(spec.get("from", 0.0))
        self.per_bar = float(spec.get("decades", 0.5))
        self.depth = float(spec.get("depth", 13.0))
        self.grid = grid
        knob = lambda name, default: _music_knob(name, spec.get(name, default), notes, features_at, onset_loader, grid)
        self._hue, self._flow, self._kick = knob("hue", 0.6), knob("flow", 0.0), knob("kick", 0.0)
        orbit = reference_orbit(complex(float(tx), float(ty)), self.iterations)
        self.ctx = ctx = context()
        self.ref = ctx.texture((len(orbit), 1), 2, data=np.ascontiguousarray(orbit).tobytes(), dtype="f4")
        self.n = len(orbit)
        self.prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vao = ctx.vertex_array(self.prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        self.fbo = ctx.simple_framebuffer((self.W, self.H), 3)

    def decades(self, t: float) -> float:
        """How deep the camera is at ``t``, in decades below the whole set."""
        if self.grid is not None:
            bars = (t - self.start) / max(1e-6, float(self.grid.bar_duration))
        else:
            bars = (t - self.start) / 2.0
        return float(np.clip(bars * self.per_bar + 0.2 * float(self._kick(t)), 0.0, self.depth))

    def frame_at(self, t: float) -> np.ndarray:
        scale = 1.5 * 10.0 ** (-self.decades(t))
        p = self.prog
        for k, v in dict(u_ref=0, u_n=self.n, u_iter=self.iterations, u_size=(float(self.W), float(self.H)),
                         u_scale=float(scale), u_hue=float(self._hue(t)), u_flow=float(self._flow(t))).items():
            if k in p:
                p[k].value = v
        self.ref.use(0)
        self.fbo.use()
        import moderngl
        self.vao.render(moderngl.TRIANGLE_STRIP)
        data = self.fbo.read(components=3)
        return np.frombuffer(data, np.uint8).reshape(self.H, self.W, 3)[::-1].copy()

"""Flesh wall — the base the body effects (``eyes``, ``mouths``) are built on.

A Voronoi tiling of the frame into patches of skin that meet in deep raw seams,
shaded in linear light and mapped to sRGB through a filmic curve. This module
holds the GLSL every such effect shares and a small GPU runner:

    FLESH_GLSL      noise, the cell lookup (seed, smooth distance to the edge and its
                    gradient, nearest neighbour), skin albedo and lighting, the
                    rounded shoulder into the seam, the ACES finish
    FleshWall       a moderngl context that renders a fragment shader supersampled
                    and averages it down on the GPU

An effect writes its own ``main()`` after ``FLESH_GLSL``: it calls ``fleshCell()``
to know which cell it is in, builds its organ in the cell's frame, lights the
skin around it with ``skinBase`` / ``seamFlesh`` / ``skinLight`` and ends with
``finish()``. The shader must declare ``u_aspect``, ``u_rows``, ``u_seed`` and
``u_light`` (``FLESH_HEADER`` does).
"""
from __future__ import annotations

import numpy as np

FLESH_HEADER = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform float u_aspect, u_rows, u_seed, u_light;
#define PI 3.14159265
"""

FLESH_GLSL = """
const vec3 L = vec3(-0.466, 0.621, 0.630);   // key light (normalized), upper left

float hash1(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
vec2  hash2(vec2 p){ float h = hash1(p); return vec2(h, hash1(p + h + 7.13)); }
float vnoise(vec2 p){
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash1(i), hash1(i + vec2(1, 0)), f.x), mix(hash1(i + vec2(0, 1)), hash1(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++){ s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }
// thin ridges where a warped noise crosses its middle: veins
float veins(vec2 p, float k){ float n = fbm(p + 2.0 * fbm(p * 0.7 + 3.3)); return pow(1.0 - abs(2.0 * n - 1.0), k); }
vec3 hsv(float h, float s, float v){ vec3 k = fract(h + vec3(0, 2.0/3.0, 1.0/3.0)) * 6.0; return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s); }
float smax(float a, float b, float k){ float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0); return mix(b, a, h) + k * h * (1.0 - h); }

vec2 seedOf(vec2 cell){ return cell + 0.5 + (hash2(cell + u_seed) - 0.5) * 0.50; }

// which cell this pixel is in. q: the pixel in cell units; best: the cell's seed; cell: its
// integer id; e: distance to the edge (a smooth minimum over the bisectors, so the seams
// round off at the corners instead of mitring) and ge its gradient; dn: nearest seed
struct Cell { vec2 q, best, cell; float e, dn; vec2 ge; };
Cell fleshCell(){
    Cell C;
    vec2 p = vec2((v_uv.x - 0.5) * 2.0 * u_aspect, (v_uv.y - 0.5) * 2.0);   // frame is 2 units tall
    float cs = 2.0 / u_rows;
    vec2 q = p / cs, c = floor(q);
    float d1 = 1e9; vec2 best = vec2(0.0), bestCell = vec2(0.0);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
        vec2 cc = c + vec2(i, j); vec2 s = seedOf(cc); float d = distance(q, s);
        if (d < d1){ d1 = d; best = s; bestCell = cc; }
    }
    const float KS = 0.06;
    float ws = 0.0, dn = 1e9; vec2 ge = vec2(0.0);
    for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++){
        if (i == 0 && j == 0) continue;
        vec2 sj = seedOf(bestCell + vec2(i, j));
        vec2 nb = normalize(sj - best);
        float w = exp(-dot((best + sj) * 0.5 - q, nb) / KS);
        ws += w; ge -= w * nb;
        dn = min(dn, distance(best, sj));
    }
    ge /= ws;
    C.q = q; C.best = best; C.cell = bestCell; C.dn = dn; C.ge = ge;
    C.e = max(0.0, -KS * log(ws) + KS * log(2.0));
    return C;
}

// the skin's height meets the seam: a rounded shoulder, then a groove
float seamShape(float h, float eL){
    float sh = 1.0 - pow(1.0 - clamp(eL / 0.8, 0.0, 1.0), 2.5);
    return h * sh - 0.25 * (1.0 - smoothstep(0.0, 0.35, eL));
}
vec3 skinBase(vec2 v, float id){ return mix(vec3(0.60, 0.34, 0.25), vec3(0.48, 0.23, 0.17), fbm(v * 1.7 + id * 13.0)); }
vec3 mottle(vec3 alb, vec2 v, float id){ return alb * (0.9 + 0.2 * vnoise(v * 6.0 + id * 21.0)); }
vec3 seamFlesh(vec3 alb, float eL){ return mix(alb, vec3(0.34, 0.08, 0.07), 0.7 * (1.0 - smoothstep(0.0, 0.35, eL))); }   // raw flesh in the seams
float seamAO(float eL){ return mix(0.12, 1.0, smoothstep(0.0, 0.45, eL)); }
// lambert with red light bleeding into the shadows, an oily sheen, and a wet gloss where ``wet``
vec3 skinLight(vec3 n, vec3 alb, float ao, vec2 v, float id, float wet){
    float ndl = dot(n, L), diff = max(ndl, 0.0), wrap = max((ndl + 0.45) / 1.45, 0.0);
    vec3 lit = alb * (diff + (wrap - diff) * vec3(0.60, 0.14, 0.09)) + alb * 0.07;
    vec3 hv = normalize(L + vec3(0.0, 0.0, 1.0));
    float nh = max(dot(n, hv), 0.0);
    return lit * ao + ao * (0.05 * pow(nh, 25.0) * (0.5 + vnoise(v * 12.0 + id)) + 0.3 * pow(nh, 90.0) * wet);
}
vec3 finish(vec3 col){
    col *= u_light * 1.15;
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);        // ACES filmic
    return pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2));
}
"""

_VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_DOWN_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_tex; uniform int u_ss;
void main(){
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ss; vec3 acc = vec3(0.0);
    for (int j = 0; j < u_ss; j++) for (int i = 0; i < u_ss; i++) acc += texelFetch(u_tex, base + ivec2(i, j), 0).rgb;
    f_color = vec4(acc / float(u_ss * u_ss), 1.0);
}
"""


class FleshWall:
    """Runs a flesh-wall fragment shader: ``render(**uniforms)`` -> uint8 H x W x 3."""

    def __init__(self, fragment_shader: str, width: int, height: int, rows: float = 5.0,
                 seed: int = 0, supersample: int = 2):
        import moderngl
        self.W, self.H = int(width), int(height)
        self.ss = max(1, int(supersample))
        self.rows, self.seed = float(rows), float(seed)
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=fragment_shader)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self.prog, [(self._vbo, "2f", "in_pos")])
        # supersample into a texture, average it down on the GPU (a numpy mean-pool of the
        # big image costs ~120 ms at 720p; this costs ~1 ms)
        self._tex = self.ctx.texture((self.W * self.ss, self.H * self.ss), 3)
        self._fbo = self.ctx.framebuffer([self._tex])
        self._down = self.ctx.program(vertex_shader=_VS, fragment_shader=_DOWN_FS)
        self._down_vao = self.ctx.vertex_array(self._down, [(self._vbo, "2f", "in_pos")])
        self._out = self.ctx.simple_framebuffer((self.W, self.H), 3)
        self._mgl = moderngl

    def draw(self, light: float = 1.0, **uniforms) -> np.ndarray:
        u = self.prog
        u["u_aspect"].value = self.W / self.H
        u["u_rows"].value = self.rows
        u["u_seed"].value = self.seed
        u["u_light"].value = float(light)
        for name, value in uniforms.items():
            if name in u:                                   # the compiler drops unused uniforms
                u[name].value = value
        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0)
        self._vao.render(self._mgl.TRIANGLE_STRIP)
        self._out.use()
        self._tex.use(0)
        self._down["u_tex"].value = 0
        self._down["u_ss"].value = self.ss
        self._down_vao.render(self._mgl.TRIANGLE_STRIP)
        img = np.frombuffer(self._out.read(components=3), dtype=np.uint8).reshape(self.H, self.W, 3)[::-1]
        return img.copy()

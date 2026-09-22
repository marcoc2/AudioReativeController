"""Eyes — a Voronoi wall of eyeballs. GPU fragment shader (moderngl), one pass.

Random seeds tile the frame into Voronoi cells (about ``rows`` cells down the
height, more across); every cell holds one eyeball set into wet flesh, as if
seen from outside the skull: a shaded sphere with a veined sclera, a fibrous
iris, a pupil, a specular glint, lids that blink now and then. All eyes look at
one point in front of the screen (each with its own small deviation), so the
wall follows a gaze:

    gaze        -> where the eyes look (screen units), glided in Python
    saccade     -> the gaze jumps and every eye rerolls its deviation (a hit)
    pupil       -> dilation 0..1 (loudness, a kick)
    blink       -> 0..1 closes every lid at once (a hit); eyes also blink on their own
    hue         -> the iris colour (the chord)
    light       -> overall brightness

Everything an eye owns (seed position, iris fibres, veins, blink timing) comes
from a hash of its cell, so the same seed gives the same wall.
"""
from __future__ import annotations

import numpy as np

_VS = """
#version 330
in vec2 in_pos; out vec2 v_uv;
void main(){ v_uv = in_pos * 0.5 + 0.5; gl_Position = vec4(in_pos, 0.0, 1.0); }
"""

_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform float u_aspect, u_rows, u_seed, u_time, u_saccade, u_pupil, u_blink, u_hue, u_light, u_depth;
uniform vec2 u_gaze;
#define PI 3.14159265

float hash1(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
vec2  hash2(vec2 p){ float h = hash1(p); return vec2(h, hash1(p + h + 7.13)); }
float vnoise(vec2 p){
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash1(i), hash1(i + vec2(1, 0)), f.x), mix(hash1(i + vec2(0, 1)), hash1(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++){ s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }
// thin bright ridges where a warped noise crosses its middle: veins
float veins(vec2 p, float k){ float n = fbm(p + 2.0 * fbm(p * 0.7 + 3.3)); return pow(1.0 - abs(2.0 * n - 1.0), k); }
vec3 hsv(float h, float s, float v){ vec3 k = fract(h + vec3(0, 2.0/3.0, 1.0/3.0)) * 6.0; return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s); }

vec2 seedOf(vec2 cell){ return cell + 0.5 + (hash2(cell + u_seed) - 0.5) * 0.50; }

void main(){
    vec2 p = vec2((v_uv.x - 0.5) * 2.0 * u_aspect, (v_uv.y - 0.5) * 2.0);   // frame is 2 units tall
    float cs = 2.0 / u_rows;
    vec2 q = p / cs, c = floor(q);
    float d1 = 1e9; vec2 best = vec2(0.0), bestCell = vec2(0.0);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
        vec2 cc = c + vec2(i, j); vec2 s = seedOf(cc); float d = distance(q, s);
        if (d < d1){ d1 = d; best = s; bestCell = cc; }
    }
    // distance from this pixel to the edge of its cell: the nearest bisector with a neighbour
    float e = 1e9;
    for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++){
        if (i == 0 && j == 0) continue;
        vec2 sj = seedOf(bestCell + vec2(i, j));
        e = min(e, dot((best + sj) * 0.5 - q, normalize(sj - best)));
    }
    e = max(e, 0.0);
    // the whole cell is one eyeball, squashed to the cell's outline: r runs 0 at the
    // seed to 1 on the edge whatever the shape, so the sphere shading and the veins
    // are what show where one eye ends and the next begins
    float r = d1 / (d1 + e);
    vec2 dir = d1 > 1e-5 ? (q - best) / d1 : vec2(0.0, 1.0);
    vec2 lp = dir * r;
    // ... while the iris sits on a round ball (radius from the nearest neighbour), so the
    // pupil stays a disc: the corners of the cell are sclera pressed out of that ball
    float dn = 1e9;
    for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++){
        if (i == 0 && j == 0) continue;
        dn = min(dn, distance(best, seedOf(bestCell + vec2(i, j))));
    }
    float rr = d1 / (0.55 * dn);
    vec3 nr = vec3(dir * min(rr, 1.0), sqrt(max(0.0, 1.0 - rr * rr)));
    float id = hash1(bestCell * 3.7 + u_seed);

    // this eye's own blink: closes for ~0.25 s every few seconds, plus the global one
    float period = 3.5 + 4.0 * hash1(bestCell + 11.0);
    float ph = fract(u_time / period + id);
    float own = smoothstep(0.0, 0.5, ph / 0.07) * smoothstep(1.0, 0.5, ph / 0.07) * step(ph, 0.07);
    float lid = clamp(max(own, u_blink), 0.0, 1.0);
    float openY = 1.0 - lid;                            // lids meet at the middle when closed

    vec3 L = normalize(vec3(-0.45, 0.6, 0.65));
    vec3 n = vec3(lp, sqrt(max(0.0, 1.0 - r * r)));
    vec3 col;
    if (abs(lp.y) < openY){
        vec2 sp = best * cs;                            // the eye's centre in frame units
        vec2 dev = (hash2(bestCell + 5.0 * floor(u_saccade) + u_seed) - 0.5) * 0.45;
        vec3 g = normalize(vec3((u_gaze + dev - sp) * 0.7, u_depth));
        float ang = rr < 1.0 ? acos(clamp(dot(nr, g), -1.0, 1.0)) : PI;
        vec3 t1 = normalize(cross(g, vec3(0.0, 1.0, 0.0))), t2 = cross(g, t1);
        float phi = atan(dot(nr, t2), dot(nr, t1));

        // sclera: veins gather towards the rim, where the ball sinks into the socket
        vec3 sclera = vec3(0.93, 0.90, 0.86);
        sclera = mix(sclera, vec3(0.70, 0.40, 0.38), smoothstep(0.55, 1.0, r));
        float w = smoothstep(0.65, 1.2, ang) * (0.15 + 0.85 * pow(r, 1.5));
        float v = veins(n.xy * 5.0 + id * 40.0, 18.0) + 0.8 * veins(n.xy * 11.0 + id * 90.0, 26.0);
        sclera = mix(sclera, vec3(0.68, 0.06, 0.05), clamp(v * w, 0.0, 1.0) * 0.9);

        // iris: radial fibres, a dark limbal ring, a lighter collarette by the pupil
        float irisA = 0.70, pupilA = 0.15 + 0.32 * u_pupil;
        float fib = fbm(vec2(phi * 6.0 + id * 20.0, ang * 16.0));
        float fib2 = vnoise(vec2(phi * 40.0, ang * 9.0 + id * 7.0));
        vec3 iris = hsv(u_hue + 0.06 * (fib - 0.5), 0.72, 0.30 + 0.75 * fib) * (0.7 + 0.5 * fib2);
        iris = mix(iris, iris * 0.25, smoothstep(irisA - 0.18, irisA, ang));
        iris = mix(iris, iris * 1.5 + 0.1, smoothstep(pupilA + 0.20, pupilA, ang) * 0.5);
        float irisM = smoothstep(irisA + 0.012, irisA - 0.012, ang);
        float pupilM = smoothstep(pupilA + 0.015, pupilA - 0.01, ang);
        col = mix(sclera, iris, irisM);
        col = mix(col, vec3(0.01), pupilM);

        // shading: lambert on the squashed sphere, the socket swallowing the rim, a hard glint
        vec3 ns = normalize(mix(nr, n, smoothstep(0.7, 1.0, rr)));   // round where the ball is, cell-shaped beyond
        float diff = 0.30 + 0.70 * max(dot(ns, L), 0.0);
        col *= diff * (0.15 + 0.85 * smoothstep(1.0, 0.72, r));
        col *= 1.0 - 0.35 * smoothstep(0.35, 1.0, lp.y) * (1.0 - n.z * 0.5);   // shadow of the upper lid
        vec3 h = normalize(L + vec3(0.0, 0.0, 1.0));
        col += vec3(0.95) * pow(max(dot(ns, h), 0.0), 120.0);
        col += vec3(0.25) * pow(max(dot(ns, normalize(vec3(0.5, -0.3, 1.0))), 0.0), 60.0);   // fill glint
        col *= 0.55 + 0.45 * smoothstep(0.0, 0.18, openY - abs(lp.y));
    } else {
        // the lid, shut: skin stretched over the ball. Wrinkles run along the lid (stretched
        // in x), pores are a fine noise, and both bend the normal; the skin reddens towards
        // the seam and the socket, and a row of lashes hangs from the upper lid.
        vec2 sk = q * 5.0 + id * 10.0;
        vec2 wf = vec2(1.2, 9.0);                                          // wrinkle frequencies (x, y)
        float w0 = fbm(sk * wf), wx = fbm((sk + vec2(0.015, 0.0)) * wf), wy = fbm((sk + vec2(0.0, 0.015)) * wf);
        float pores = vnoise(sk * 14.0);
        float p0 = pores, px = vnoise((sk + vec2(0.01, 0.0)) * 14.0), py = vnoise((sk + vec2(0.0, 0.01)) * 14.0);
        vec3 nb = normalize(n + vec3((wx - w0) * 6.0 + (px - p0) * 1.5, (wy - w0) * 6.0 + (py - p0) * 1.5, 0.0));
        float blotch = fbm(sk * 0.7 + 3.0);
        vec3 skin = mix(vec3(0.74, 0.50, 0.42), vec3(0.62, 0.36, 0.32), blotch);
        float seam = smoothstep(0.35, 0.0, abs(lp.y) - openY);           // near where the lids meet
        skin = mix(skin, vec3(0.62, 0.22, 0.22), 0.55 * seam + 0.35 * smoothstep(0.6, 1.0, r));
        float diff = 0.30 + 0.70 * max(dot(nb, L), 0.0);
        col = skin * diff * (0.92 + 0.16 * pores);
        vec3 h = normalize(L + vec3(0.0, 0.0, 1.0));
        col += vec3(0.18, 0.14, 0.12) * pow(max(dot(nb, h), 0.0), 18.0);   // soft oily sheen
        col *= 0.15 + 0.85 * smoothstep(1.0, 0.72, r);
        col *= 0.55 + 0.45 * smoothstep(0.0, 0.05, abs(lp.y) - openY);     // the crease of the seam
        // lashes: dark hairs on the upper lid, just above the seam
        float above = lp.y - openY;
        float hair = pow(vnoise(vec2(q.x * 90.0 + id * 30.0, 0.0)), 4.0);
        float lash = hair * smoothstep(0.0, 0.01, above) * smoothstep(0.12, 0.02, above);
        col *= 1.0 - 0.8 * lash;
    }
    f_color = vec4(clamp(col * u_light, 0.0, 1.0), 1.0);
}
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


class EyesSystem:
    def __init__(self, width: int, height: int, rows: float = 5.0, seed: int = 0,
                 supersample: int = 2, depth: float = 0.9):
        import moderngl
        self.W, self.H = int(width), int(height)
        self.ss = max(1, int(supersample))
        self.rows, self.seed, self.depth = float(rows), float(seed), float(depth)
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
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

    def render(self, time: float, gaze=(0.0, 0.0), saccade: int = 0, pupil: float = 0.3,
               blink: float = 0.0, hue: float = 0.55, light: float = 1.0) -> np.ndarray:
        """One frame (uint8, H x W x 3). ``time`` drives the eyes' own blinks."""
        u = self.prog
        u["u_aspect"].value = self.W / self.H
        u["u_rows"].value = self.rows
        u["u_seed"].value = self.seed
        u["u_time"].value = float(time)
        u["u_gaze"].value = (float(gaze[0]), float(gaze[1]))
        u["u_depth"].value = self.depth
        u["u_saccade"].value = float(int(saccade))
        u["u_pupil"].value = float(np.clip(pupil, 0.0, 1.0))
        u["u_blink"].value = float(np.clip(blink, 0.0, 1.0))
        u["u_hue"].value = float(hue) % 1.0
        u["u_light"].value = float(light)
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

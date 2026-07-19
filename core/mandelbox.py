"""MandelboxSystem — kaleidoscopic IFS with a TRUE infinite zoom loop.

The fold (tetrahedral abs-folds + scale) is exactly self-similar with
factor SCALE around its fixed point, so evaluating the world at
``q = FIX + (p - FIX) / SCALE**phase`` makes phase 0.0 and 1.0 render the
same geometry: the dive loops seamlessly, forever, with no precision loss.

``phase`` is supplied in *musical* time (e.g. one loop every 4 bars), so
the eternal fall breathes with the song. Audio: flux -> orbit/roll speed
and glow, centroid -> hue, kick pulse -> brief extra plunge.
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
uniform float u_z, u_angle, u_roll, u_hue, u_glow, u_aspect, u_spread;

const float SCALE = 2.0;
const vec3  FIX   = vec3(1.0, 1.0, 1.0);

float DE0(vec3 p){
    // classic Sierpinski-tetra KIFS: FIX is a vertex with infinite
    // substructure, exactly self-similar under SCALE about FIX
    float k = 1.0;
    for (int i = 0; i < 16; i++){
        if (p.x + p.y < 0.0) p.xy = -p.yx;
        if (p.x + p.z < 0.0) p.xz = -p.zx;
        if (p.y + p.z < 0.0) p.yz = -p.zy;
        p = p * SCALE - FIX * (SCALE - 1.0);
        k *= SCALE;
    }
    return (length(p) - 2.0) / k;
}

float map(vec3 p){
    vec3 q = FIX + (p - FIX) / u_z;   // self-similar zoom: seamless at wrap
    return DE0(q) * u_z;
}

vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(){
    vec2 uv = v_uv * 2.0 - 1.0; uv.x *= u_aspect;
    // orbit around the vertex axis (1,1,1), looking at the vertex
    vec3 ax = normalize(FIX);
    vec3 b1 = normalize(cross(ax, vec3(0.0, 0.0, 1.0)));
    vec3 b2 = cross(ax, b1);
    vec3 ro = FIX + 1.5 * (cos(u_angle) * b1 + sin(u_angle) * b2) + 0.9 * ax;
    vec3 fw = normalize(FIX - 0.5 * ax - ro);   // aim below the vertex, into the gasket
    vec3 rt0 = normalize(cross(fw, vec3(0.0, 0.0, 1.0)));
    vec3 up0 = cross(rt0, fw);
    float cr = cos(u_roll), sr = sin(u_roll);
    vec3 rt = rt0 * cr + up0 * sr;
    vec3 up = -rt0 * sr + up0 * cr;
    vec3 rd = normalize(fw * 1.5 + uv.x * rt + uv.y * up);

    float t = 0.0; int steps = 0; bool hit = false;
    for (int i = 0; i < 150; i++){
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.0006 * t){ hit = true; break; }
        t += d * 0.9; steps = i;
        if (t > 8.0) break;
    }
    float ao = 1.0 - float(steps) / 150.0;
    vec3 col;
    if (hit){
        vec3 p = ro + rd * t;
        float e = 0.0004 + 0.0008 * t;
        vec3 n = normalize(vec3(
            map(p + vec3(e,0,0)) - map(p - vec3(e,0,0)),
            map(p + vec3(0,e,0)) - map(p - vec3(0,e,0)),
            map(p + vec3(0,0,e)) - map(p - vec3(0,0,e))));
        vec3 ld = normalize(vec3(0.5, 0.7, 0.6));
        float dif = clamp(dot(n, ld), 0.0, 1.0);
        // log-radial hue bands: one hue cycle per self-similarity octave,
        // colors flow outward with the dive; world-position based, so the
        // loop wrap stays seamless by construction
        float r = length(p - FIX);
        float band = 0.5 + 0.5 * sin(6.2832 * log2(max(r, 1e-4)));
        float az = atan(dot(p - FIX, b1), dot(p - FIX, b2));
        vec3 base = hsv2rgb(vec3(u_hue + u_spread * band + 0.06 * sin(3.0 * az),
                                 0.5 + 0.3 * band, 0.92));
        float rim = pow(1.0 - abs(dot(n, -rd)), 2.0);
        col = base * (0.12 + 0.88 * dif) * ao
            + rim * hsv2rgb(vec3(u_hue + 0.45, 0.45, 1.0)) * 0.5;
    } else {
        col = hsv2rgb(vec3(u_hue + 0.5, 0.35, 0.04))
            + u_glow * (float(steps) / 150.0) * hsv2rgb(vec3(u_hue, 0.85, 0.45));
    }
    f_color = vec4(col, 1.0);
}
"""


class MandelboxSystem:
    def __init__(self, width: int, height: int, supersample: int = 2):
        import moderngl
        self.W, self.H = int(width), int(height)
        self.ss = max(1, int(supersample))
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self.prog, [(self._vbo, "2f", "in_pos")])
        self._fbo = self.ctx.simple_framebuffer(
            (self.W * self.ss, self.H * self.ss), 3)
        self._mgl = moderngl
        self.angle = 0.0
        self.roll = 0.0
        self._flux = 0.0
        self._hue = 0.58
        self.hue_spread = 0.35

    def step(self, dt: float, controls: dict) -> None:
        flux_raw = float(controls.get("flux", 0.0) or 0.0)
        self._flux += (flux_raw - self._flux) * (1.0 - float(np.exp(-dt / 0.35)))
        cen = controls.get("centroid")
        if cen is not None:
            self._hue = 0.5 + 0.38 * float(cen)
        self.angle += dt * (0.05 + 0.35 * self._flux)
        self.roll += dt * (0.02 + 0.20 * self._flux)

    def render(self, phase: float, pulse: float = 0.0) -> np.ndarray:
        """``phase`` 0..1 = one seamless self-similarity period (musical)."""
        p = pulse * pulse * (3.0 - 2.0 * pulse)
        # the continuous dive dominates; the kick is an accent on top
        z = float(2.0 ** ((phase % 1.0) + 0.15 * p))
        self.prog["u_z"].value = z
        self.prog["u_angle"].value = float(self.angle)
        self.prog["u_roll"].value = float(self.roll)
        self.prog["u_hue"].value = float(self._hue)
        self.prog["u_glow"].value = float(0.15 + 0.85 * self._flux)
        self.prog["u_spread"].value = float(self.hue_spread)
        self.prog["u_aspect"].value = self.W / self.H
        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0)
        self._vao.render(self._mgl.TRIANGLE_STRIP)
        data = self._fbo.read(components=3)
        sw, sh = self.W * self.ss, self.H * self.ss
        img = np.frombuffer(data, dtype=np.uint8).reshape(sh, sw, 3)[::-1]
        if self.ss > 1:
            img = img.reshape(self.H, self.ss, self.W, self.ss, 3).mean(
                axis=(1, 3)).astype(np.uint8)
        return img.copy()

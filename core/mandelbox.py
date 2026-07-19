"""MandelboxSystem — infinite forward flight through a fractal corridor.

The world is a space-periodic kaleidoscopic IFS (tetra folds mirrored in
±x/±y, tiled along z): the camera flies forward through an endless tunnel
of fractal structures with real parallax and depth fog. World, camera
path, sway and colors are all periodic in one TILE, and the camera covers
exactly one TILE per musical loop (loop_bars) — so the flight is both
infinite and seamlessly loopable, with no float growth (z stays bounded
by the modulo).

Audio: flux -> glow/light energy, centroid -> hue, kick pulse -> brief
forward surge on top of the dominant constant flight.
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
uniform float u_ph, u_hue, u_glow, u_aspect, u_spread, u_focal;
uniform float u_c0, u_sts;   // corridor clearance, structure density

const float TILE = 2.0;

float DEt(vec3 p){
    float k = 1.0;
    for (int i = 0; i < 12; i++){
        if (p.x + p.y < 0.0) p.xy = -p.yx;
        if (p.x + p.z < 0.0) p.xz = -p.zx;
        if (p.y + p.z < 0.0) p.yz = -p.zy;
        p = p * 2.0 - vec3(1.0);
        k *= 2.0;
    }
    return (length(p) - 2.0) / k;
}

float map(vec3 p){
    p.z = mod(p.z, TILE) - 0.5 * TILE;   // endless corridor
    p.xy = abs(p.xy) - vec2(u_c0);       // fractal walls on all four sides
    return DEt(p * u_sts) / u_sts;
}

vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(){
    vec2 uv = v_uv * 2.0 - 1.0; uv.x *= u_aspect;

    // flight path: forward along z, gentle periodic sway + banked look
    float w = 6.2832 * u_ph;
    vec3 ro = vec3(0.30 * sin(w), 0.22 * cos(w), TILE * u_ph);
    vec3 tangent = normalize(vec3(0.30 * cos(w) * 6.2832 / TILE * 0.12,
                                  -0.22 * sin(w) * 6.2832 / TILE * 0.12, 1.0));
    vec3 fw = tangent;
    vec3 rt0 = normalize(cross(fw, vec3(0.0, 1.0, 0.0)));
    vec3 up0 = cross(rt0, fw);
    float roll = 0.35 * sin(w);
    float crr = cos(roll), srr = sin(roll);
    vec3 rt = rt0 * crr + up0 * srr;
    vec3 up = -rt0 * srr + up0 * crr;
    vec3 rd = normalize(fw * u_focal + uv.x * rt + uv.y * up);

    float t = 0.0; int steps = 0; bool hit = false;
    for (int i = 0; i < 150; i++){
        vec3 p = ro + rd * t;
        float d = map(p);
        if (d < 0.0007 * t + 0.0002){ hit = true; break; }
        t += d * 0.9; steps = i;
        if (t > 12.0) break;
    }
    float ao = 1.0 - float(steps) / 150.0;
    vec3 bg = hsv2rgb(vec3(u_hue + 0.5, 0.35, 0.03));
    vec3 col;
    if (hit){
        vec3 p = ro + rd * t;
        float e = 0.0004 + 0.0008 * t;
        vec3 n = normalize(vec3(
            map(p + vec3(e,0,0)) - map(p - vec3(e,0,0)),
            map(p + vec3(0,e,0)) - map(p - vec3(0,e,0)),
            map(p + vec3(0,0,e)) - map(p - vec3(0,0,e))));
        vec3 ld = normalize(vec3(0.4, 0.75, -0.5));
        float dif = clamp(dot(n, ld), 0.0, 1.0);
        // colors periodic along the corridor: bands stream past with flight
        float band = 0.5 + 0.5 * sin(6.2832 * p.z / TILE);
        float az = atan(p.y, p.x);
        vec3 base = hsv2rgb(vec3(u_hue + u_spread * band + 0.07 * sin(2.0 * az),
                                 0.5 + 0.3 * band, 0.92));
        float rim = pow(1.0 - abs(dot(n, -rd)), 2.0);
        col = base * (0.10 + 0.90 * dif) * ao
            + rim * hsv2rgb(vec3(u_hue + 0.45, 0.45, 1.0)) * 0.5;
        col = mix(bg, col, exp(-0.22 * t));   // depth fog: sells the motion
    } else {
        col = bg + u_glow * (float(steps) / 150.0)
                 * hsv2rgb(vec3(u_hue, 0.85, 0.4));
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
        self._flux = 0.0
        self._hue = 0.58
        self.hue_spread = 0.35
        self.focal = 1.1        # wide lens: tunnel walls wrap the view
        self.clearance = 0.85   # corridor half-width before the walls
        self.density = 1.3      # structure scale inside the walls

    def step(self, dt: float, controls: dict) -> None:
        flux_raw = float(controls.get("flux", 0.0) or 0.0)
        self._flux += (flux_raw - self._flux) * (1.0 - float(np.exp(-dt / 0.35)))
        cen = controls.get("centroid")
        if cen is not None:
            self._hue = 0.5 + 0.38 * float(cen)

    def render(self, phase: float, pulse: float = 0.0) -> np.ndarray:
        """``phase`` 0..1 advances the camera one TILE (one musical loop)."""
        p = pulse * pulse * (3.0 - 2.0 * pulse)
        ph = (phase % 1.0) + 0.06 * p     # kick: brief forward surge
        self.prog["u_ph"].value = float(ph)
        self.prog["u_hue"].value = float(self._hue)
        self.prog["u_glow"].value = float(0.15 + 0.85 * self._flux)
        self.prog["u_spread"].value = float(self.hue_spread)
        self.prog["u_focal"].value = float(self.focal)
        self.prog["u_c0"].value = float(self.clearance)
        self.prog["u_sts"].value = float(self.density)
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

"""MandelbulbSystem — GPU ray-marched 3D fractal (moderngl fragment shader).

Audio mapping:
  chroma (dominant pitch) -> target power of the bulb (3..11), gliding —
                             harmony morphs the creature's species
  bass energy             -> camera dives closer (plus external kick zoom)
  flux                    -> orbit speed + background glow
  centroid                -> light/palette hue

Renders offscreen at the output resolution (an RTX-class GPU does this in
well under a millisecond; readback dominates). Deterministic per frame.
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
uniform float u_power, u_angle, u_elev, u_dist, u_hue, u_glow, u_aspect;

float DE(vec3 p){
    vec3 z = p; float dr = 1.0; float r = 0.0;
    for (int i = 0; i < 12; i++){
        r = length(z);
        if (r > 2.0) break;
        float th = acos(clamp(z.z / r, -1.0, 1.0)) * u_power;
        float ph = atan(z.y, z.x) * u_power;
        float zr = pow(r, u_power);
        dr = pow(r, u_power - 1.0) * u_power * dr + 1.0;
        z = zr * vec3(sin(th) * cos(ph), sin(ph) * sin(th), cos(th)) + p;
    }
    return 0.5 * log(r) * r / dr;
}

vec3 hsv2rgb(vec3 c){
    vec4 K = vec4(1.0, 2.0/3.0, 1.0/3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main(){
    vec2 uv = v_uv * 2.0 - 1.0; uv.x *= u_aspect;
    vec3 ro = vec3(u_dist * cos(u_elev) * cos(u_angle),
                   u_dist * cos(u_elev) * sin(u_angle),
                   u_dist * sin(u_elev));
    vec3 fw = normalize(-ro);
    vec3 rt = normalize(cross(fw, vec3(0.0, 0.0, 1.0)));
    vec3 up = cross(rt, fw);
    vec3 rd = normalize(fw * 1.6 + uv.x * rt + uv.y * up);

    float t = 0.0; int steps = 0; bool hit = false;
    for (int i = 0; i < 100; i++){
        vec3 p = ro + rd * t;
        float d = DE(p);
        if (d < 0.0012 * t){ hit = true; break; }
        t += d; steps = i;
        if (t > 6.0) break;
    }
    float ao = 1.0 - float(steps) / 100.0;
    vec3 col;
    if (hit){
        vec3 p = ro + rd * t;
        float e = 0.0015;
        vec3 n = normalize(vec3(
            DE(p + vec3(e,0,0)) - DE(p - vec3(e,0,0)),
            DE(p + vec3(0,e,0)) - DE(p - vec3(0,e,0)),
            DE(p + vec3(0,0,e)) - DE(p - vec3(0,0,e))));
        vec3 ld = normalize(vec3(0.6, 0.5, 0.8));
        float dif = clamp(dot(n, ld), 0.0, 1.0);
        vec3 base = hsv2rgb(vec3(u_hue, 0.65, 0.9));
        float rim = pow(1.0 - abs(dot(n, -rd)), 2.0);
        col = base * (0.15 + 0.85 * dif) * ao
            + rim * hsv2rgb(vec3(u_hue + 0.12, 0.5, 1.0)) * 0.6;
    } else {
        col = hsv2rgb(vec3(u_hue + 0.5, 0.4, 0.05))
            + u_glow * (float(steps) / 100.0) * hsv2rgb(vec3(u_hue, 0.8, 0.5));
    }
    f_color = vec4(col, 1.0);
}
"""


class MandelbulbSystem:
    def __init__(self, width: int, height: int):
        import moderngl
        self.W, self.H = int(width), int(height)
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self.prog, [(self._vbo, "2f", "in_pos")])
        self._fbo = self.ctx.simple_framebuffer((self.W, self.H), 3)
        self._mgl = moderngl
        # animated state
        self.power = 8.0
        self.angle = 0.0
        self.elev_phase = 0.0
        self._bass = 0.0
        self._flux = 0.0
        self._hue = 0.6

    def step(self, dt: float, controls: dict) -> None:
        ch = controls.get("chroma")
        if ch is not None and len(ch) == 12:
            target = 3.0 + (int(np.argmax(ch)) / 11.0) * 8.0   # 3..11
            self.power += (target - self.power) * min(1.0, dt * 1.5)
        self._flux = float(controls.get("flux", 0.0) or 0.0)
        sub = controls.get("subbands") or {}
        self._bass = float(sub.get("bass", controls.get("bass_energy", 0.0) or 0.0))
        cen = controls.get("centroid")
        if cen is not None:
            self._hue = 0.52 + 0.36 * float(cen)
        self.angle += dt * (0.15 + 1.2 * self._flux)
        self.elev_phase += dt * 0.23

    def render(self, zoom: float = 0.0) -> np.ndarray:
        dist = max(1.15, 2.7 - 1.1 * self._bass - 0.7 * float(zoom))
        self.prog["u_power"].value = float(self.power)
        self.prog["u_angle"].value = float(self.angle)
        self.prog["u_elev"].value = 0.35 * float(np.sin(self.elev_phase))
        self.prog["u_dist"].value = float(dist)
        self.prog["u_hue"].value = float(self._hue)
        self.prog["u_glow"].value = float(0.2 + 0.8 * self._flux)
        self.prog["u_aspect"].value = self.W / self.H
        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0)
        self._vao.render(self._mgl.TRIANGLE_STRIP)
        data = self._fbo.read(components=3)
        img = np.frombuffer(data, dtype=np.uint8).reshape(self.H, self.W, 3)
        return img[::-1].copy()   # GL origin is bottom-left

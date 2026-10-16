"""A flight through a fractal's caves, phrased by the bars.

The space is Knighty's pseudo-Kleinian fractal (box folds and sphere inversions: arches,
halls and colonnades; written afresh). The camera is worked out on the CPU, frame by
frame, with the same distance estimate the GPU draws with: it glides forward one step a
bar, easing in on the downbeat and settling (a dolly, then a hold), the kick pushes it a
little further, and it steers by itself towards open space (looking ahead in a few
directions and turning towards the one with the most room), never into a wall. At every
clip cut it takes another branch (it turns hard and finds its way again).

    - source: flythrough
      step: 0.6             # how far it goes each bar (units of the fractal; it is ~2 across)
      dolly: 0.5            # the share of the bar the move takes (then it holds)
      kick: {hits: {track: kick}, envelope: 0.3}   # pushes it on a little (a knob)
      hue: 0.6              # the colour (a knob: harmony:)
      fog: 0.3              # how thick the air is
      room: 0.08            # the clearance it keeps from the walls (smaller: it hugs them)

GPU (a fragment pass, GL 3.3); the camera's path is numpy.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

CSIZE = np.array([0.92436, 0.90756, 0.92436])
SIZE = 1.0
FOLDS = 8

_FS = """
#version 330
out vec4 f_color;
uniform vec2 u_size;
uniform vec3 u_pos, u_fw, u_rt, u_up;
uniform float u_hue, u_fog, u_flash;
const vec3 CSIZE = vec3(0.92436, 0.90756, 0.92436);

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

float gTrap;

float de(vec3 p) {
    float f = 1.0;
    float trap = 1e9;
    for (int i = 0; i < %(FOLDS)d; i++) {
        p = 2.0 * clamp(p, -CSIZE, CSIZE) - p;
        float k = max(%(SIZE)f / dot(p, p), 1.0);
        p *= k;
        f *= k;
        trap = min(trap, dot(p, p));
    }
    gTrap = trap;
    float rxy = length(p.xy);
    return max(rxy - 0.92784, abs(rxy * p.z) / length(p)) / f;
}

void main() {
    vec2 uv = (gl_FragCoord.xy - 0.5 * u_size) / u_size.y;
    vec3 rd = normalize(u_fw * 1.2 + uv.x * u_rt + uv.y * u_up);
    float t = 0.0;
    int steps = 0;
    bool hit = false;
    for (int i = 0; i < 180; i++) {
        float d = de(u_pos + rd * t);
        if (d < 0.0004 * (1.0 + 10.0 * t)) { hit = true; break; }
        t += d * 0.9;
        steps = i;
        if (t > 8.0) break;
    }
    vec3 air = hsv(u_hue + 0.5, 0.35, 0.10);
    vec3 col = air;
    if (hit) {
        vec3 p = u_pos + rd * t;
        vec2 e = vec2(0.0006 * (1.0 + 10.0 * t), 0.0);
        vec3 n = normalize(vec3(de(p + e.xyy) - de(p - e.xyy), de(p + e.yxy) - de(p - e.yxy), de(p + e.yyx) - de(p - e.yyx)));
        de(p);
        vec3 base = hsv(u_hue + 0.15 * log(gTrap + 1.0), 0.55, 1.0);
        float head = clamp(dot(n, -rd), 0.0, 1.0);                 // a lamp on the camera
        float ao = 1.0 - float(steps) / 180.0;
        col = base * (0.15 + 0.9 * head) * (0.3 + 0.7 * ao) * (1.0 + u_flash);
    }
    col = mix(col, air, 1.0 - exp(-u_fog * t * t));
    f_color = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}
""" % {"FOLDS": FOLDS, "SIZE": SIZE}


def de_np(p: np.ndarray) -> np.ndarray:
    """The same distance estimate on the CPU, for points ``p`` (..., 3)."""
    p = np.array(p, dtype=np.float64)
    f = np.ones(p.shape[:-1])
    for _ in range(FOLDS):
        p = 2.0 * np.clip(p, -CSIZE, CSIZE) - p
        k = np.maximum(SIZE / np.maximum((p * p).sum(-1), 1e-12), 1.0)
        p = p * k[..., None]
        f = f * k
    rxy = np.hypot(p[..., 0], p[..., 1])
    return np.maximum(rxy - 0.92784, np.abs(rxy * p[..., 2]) / np.linalg.norm(p, axis=-1)) / f


def _turn(d: np.ndarray, yaw: float, pitch: float) -> np.ndarray:
    up = np.array([0.0, 1.0, 0.0])
    rt = np.cross(up, d)
    rt = rt / max(1e-9, np.linalg.norm(rt))
    v = d * np.cos(yaw) + rt * np.sin(yaw)
    u = np.cross(v, rt)
    v = v * np.cos(pitch) + u * np.sin(pitch)
    return v / np.linalg.norm(v)


class FlythroughLayer:
    """Source: see the module docstring."""

    LOOK = [(y, p) for y in (-0.5, -0.25, 0.0, 0.25, 0.5) for p in (-0.25, 0.0, 0.25)]

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: int,
                 features_at=None, onset_loader=None, grid=None, cuts=None):
        from core.gpu import context
        from core.shader_pass import VERTEX_SHADER
        from core.video.layers import _music_knob
        self.W, self.H, self.fps = int(width), int(height), float(fps)
        self.grid = grid
        self.step = float(spec.get("step", 0.6))
        self.dolly = float(np.clip(float(spec.get("dolly", 0.5)), 0.05, 1.0))
        self.fog = float(spec.get("fog", 0.3))
        self.room = float(spec.get("room", 0.08))
        knob = lambda name, default: _music_knob(name, spec.get(name, default), notes, features_at, onset_loader, grid)
        self._kick, self._hue = knob("kick", 0.0), knob("hue", 0.6)
        self._cuts = cuts
        self._known = (-np.inf, np.zeros(0))
        self.pos = np.array([0.1, 0.05, 0.12])
        self.fw = np.array([0.0, 0.0, 1.0])
        self._s: Optional[float] = None
        self._shot = None
        self._settle()
        self.ctx = ctx = context()
        self.prog = ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self.vao = ctx.vertex_array(self.prog, [(ctx.buffer(quad.tobytes()), "2f", "in_pos")])
        self.fbo = ctx.simple_framebuffer((self.W, self.H), 3)

    def _settle(self) -> None:
        """Out of any wall: step towards where there is more room until there is enough."""
        for _ in range(60):
            if de_np(self.pos) > 0.02:
                return
            e = 1e-3
            g = np.array([de_np(self.pos + v) - de_np(self.pos - v) for v in np.eye(3) * e])
            self.pos = self.pos + (g / max(1e-9, np.linalg.norm(g))) * 0.01 + 0.003 * self.fw

    def _travel(self, t: float) -> float:
        """How far along the flight is at ``t`` (bars phrase it: a dolly, then a hold)."""
        if self.grid is not None:
            i = self.grid.bar_index(t)
            start, end = self.grid.bar_start(i), self.grid.bar_start(i + 1)
            x = min(1.0, (t - start) / max(1e-6, (end - start) * self.dolly))
            return (i + x * x * (3 - 2 * x)) * self.step
        return t * self.step / 2.0

    def _shot_at(self, t: float) -> int:
        if self._cuts is None:
            return 0
        if t + 600.0 > self._known[0]:
            self._known = (t + 3600.0, np.asarray(self._cuts(t + 3600.0), dtype=float))
        return int(np.searchsorted(self._known[1], t + 1e-6, side="right"))

    def advance(self, t: float) -> None:
        """Move the camera to where it is at ``t`` (called in time order)."""
        s = self._travel(t) + 0.15 * float(self._kick(t))
        shot = self._shot_at(t)
        if self._shot is not None and shot != self._shot:          # a cut: another branch
            self.fw = _turn(self.fw, 1.3 if shot % 2 else -1.3, 0.2)
        self._shot = shot
        ds = 0.0 if self._s is None else max(0.0, s - self._s)
        self._s = s
        for _ in range(max(1, int(np.ceil(ds / 0.01)))):            # small steps: never through a wall
            room = de_np(self.pos)
            look = max(0.05, 3.0 * min(room, self.room))
            dirs = np.array([_turn(self.fw, y, p) for y, p in self.LOOK])
            ahead = de_np(self.pos[None, :] + dirs * look)
            # not the most room (that leads out into the void) but room like a corridor's
            best = dirs[int(np.argmin(np.abs(np.log(np.maximum(ahead, 1e-6) / self.room))))]
            self.fw = self.fw * 0.85 + best * 0.15
            self.fw /= np.linalg.norm(self.fw)
            move = min(ds, 0.01) if ds > 0 else 0.0
            self.pos = self.pos + self.fw * min(move, 0.5 * room)
            ds -= move
        self._settle()

    def frame_at(self, t: float) -> np.ndarray:
        import moderngl
        self.advance(t)
        up = np.array([0.0, 1.0, 0.0])
        rt = np.cross(up, self.fw)
        rt /= max(1e-9, np.linalg.norm(rt))
        upv = np.cross(self.fw, rt)
        p = self.prog
        for k, v in dict(u_size=(float(self.W), float(self.H)), u_pos=tuple(self.pos), u_fw=tuple(self.fw),
                         u_rt=tuple(rt), u_up=tuple(upv), u_hue=float(self._hue(t)), u_fog=self.fog,
                         u_flash=0.3 * float(self._kick(t))).items():
            if k in p:
                p[k].value = v
        self.fbo.use()
        self.vao.render(moderngl.TRIANGLE_STRIP)
        data = self.fbo.read(components=3)
        return np.frombuffer(data, np.uint8).reshape(self.H, self.W, 3)[::-1].copy()

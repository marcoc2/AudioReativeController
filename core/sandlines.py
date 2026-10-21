"""Sand lines — the white outlines of a drawing turned into sand, and shock waves that scatter it. GPU.

The picture itself goes away; only its lines are left, as white sand on black:
the drawing is inverted (its ink becomes white), its contrast stretched, and only
what is left near white is kept (``white``: how near). Each grain of sand is
given a home on those lines (``lines(frame)``, typically once a bar, from the
first frame of the bar's clip); when a new drawing comes the grains travel from
the lines they are on to the nearest lines of the new one.

The grains are simulated on the GPU, each with a position on the plate, a
velocity, a height and a heat (state in float textures, one texel per grain):

    ring      a shock wave running out through the sand (as in ``core/shockwave``):
              where it passes it throws the grains outwards and up (``push``, ``lift``: the
              speed a ring gives a grain over its whole passage, every grain with its own
              weight and a little sideways spray; a chord throws no harder than a note) and
              heats them
    flight    a grain in the air falls under ``gravity``, lands, bounces a little
              (losing most of its speed) and settles
    friction  on the plate a grain slides and stops (Coulomb friction: it slows by
              ``friction`` whatever its speed, so it stops dead, as sand does)
    plate     the plate hums, so the sand creeps back to its lines (``pull``; 0: it
              stays where it was thrown until the next drawing)
    jump      a hit makes every grain on the plate hop (``jump(strength)``)
    heat      a heated grain takes the ring's colour and cools back to white

A grain is drawn as a point, bigger and brighter while in the air; the points add
up (lines are dense and white, scattered sand is a dust) with a soft glow.
Every random draw is seeded by ``seed`` and the frame: same inputs, same film.
"""
from __future__ import annotations

import math

import numpy as np

from core.shader_pass import VERTEX_SHADER, ShaderPass

MAX_RINGS = 16

_STEP_FS = """
#version 330
in vec2 v_uv;
layout(location = 0) out vec4 o_move;      // x, y, vx, vy   (frame units: the frame is 2 tall, y down)
layout(location = 1) out vec4 o_air;       // height, vertical speed, heat, the grain's own draw 0..1
uniform sampler2D u_move, u_air, u_home;
uniform vec4 u_ring[16];                   // x, y, radius, strength
uniform float u_aspect, u_dt, u_width, u_push, u_lift, u_gravity, u_friction, u_pull, u_jump, u_cool,
              u_bounce, u_frame, u_travel, u_rspeed;
float hash1(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

void main(){
    ivec2 id = ivec2(gl_FragCoord.xy);
    vec4 m = texelFetch(u_move, id, 0), a = texelFetch(u_air, id, 0);
    vec2 p = m.xy, v = m.zw, home = texelFetch(u_home, id, 0).xy;
    float z = a.x, vz = a.y, heat = a.z, g = a.w;
    float dt = u_dt;
    // the rings: outwards and up, each grain by its own weight, sprayed a little sideways
    vec2 f = vec2(0.0); float band = 0.0;
    for (int i = 0; i < 16; i++){
        vec4 R = u_ring[i];
        if (R.w <= 0.0) continue;
        vec2 d = p - R.xy;
        float r = length(d) + 1e-4;
        float b = exp(-pow((r - R.z) / u_width, 2.0)) * R.w;
        f += d / r * b; band += b;
    }
    // a chord's rings on top of each other throw no harder than one
    if (band > 1.0){ f /= band; band = 1.0; }
    float spray = (hash1(vec2(id) * 1.37 + u_frame * 0.917 + 5.1) - 0.5) * 1.2;   // a new draw each step: no swirl
    f = mat2(cos(spray), sin(spray), -sin(spray), cos(spray)) * f;
    float weight = 0.55 + 0.9 * g;                               // light grains fly further
    // over its whole passage a ring gives a grain ``u_push`` of speed outwards and ``u_lift`` upwards
    float pass = u_rspeed / (u_width * 1.7725);
    v += f * u_push * weight * pass * dt;
    vz += band * u_lift * weight * pass * dt;
    float sp = length(v);
    if (sp > 3.0) v *= 3.0 / sp;
    vz = min(vz, 4.0);
    if (z <= 0.0 && vz < 0.1) vz = 0.0;                          // too weak a kick to lift it off the plate
    heat = max(heat * exp(-dt / u_cool), min(band * 1.5, 1.0));
    bool ground = z <= 0.0 && vz <= 0.0;
    if (ground) vz += u_jump * (0.5 + g);                        // a hit: the whole plate hops
    // flight
    if (z > 0.0 || vz > 0.0){
        vz -= u_gravity * dt;
        z += vz * dt;
        v *= exp(-0.4 * dt);                                     // a little air drag
        if (z <= 0.0){                                           // lands: bounces a little, loses most of it
            z = 0.0;
            vz = -vz * u_bounce;
            if (vz < 0.08) vz = 0.0;
            v *= 0.6;
        }
    } else {
        // on the plate: it hums, carrying the grain back towards its line ...
        vec2 to = home - p;
        float dist = length(to);
        vec2 carry = dist > 1e-5 ? to / dist * min(dist * u_pull, u_travel) : vec2(0.0);
        vec2 rel = v - carry;                                    // ... and friction stops it dead
        float s = length(rel);
        rel *= s > 1e-6 ? max(0.0, s - u_friction * dt) / s : 0.0;
        v = carry + rel;
        // the hum shivers a grain a little, more the further it is from its line
        float j1 = hash1(vec2(id) + u_frame * 3.17), j2 = hash1(vec2(id) * 0.71 + u_frame * 5.93 + 1.3);
        p += (vec2(j1, j2) - 0.5) * min(dist, 0.03) * sqrt(dt) * 0.8 * min(u_pull, 1.0);
    }
    p += v * dt;
    // the rim of the plate: a grain that hits it drops there, most of its speed gone
    if (abs(p.x) > u_aspect){ p.x = sign(p.x) * u_aspect; v.x *= -0.2; }
    if (abs(p.y) > 1.0){ p.y = sign(p.y); v.y *= -0.2; }
    o_move = vec4(p, v);
    o_air = vec4(z, vz, heat, g);
}
"""

_GRAIN_VS = """
#version 330
uniform sampler2D u_move, u_air;
uniform int u_side;
uniform float u_aspect, u_size, u_hue, u_sat, u_bright;
out vec3 v_col;
vec3 hsv(float h, float s, float v){ vec3 k = fract(h + vec3(0, 2.0/3.0, 1.0/3.0)) * 6.0; return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s); }
void main(){
    ivec2 id = ivec2(gl_VertexID % u_side, gl_VertexID / u_side);
    vec4 m = texelFetch(u_move, id, 0), a = texelFetch(u_air, id, 0);
    gl_Position = vec4(m.x / u_aspect, -m.y, 0.0, 1.0);
    float up = clamp(a.x * 8.0, 0.0, 1.0);
    gl_PointSize = u_size * (1.0 + 1.5 * up);                   // nearer the eye in the air
    vec3 tint = hsv(u_hue + 0.08 * (a.w - 0.5), u_sat, 1.0) * 1.4;
    v_col = mix(vec3(1.0), tint, a.z) * u_bright * (1.0 + 0.8 * up + 0.8 * a.z) * (0.75 + 0.5 * a.w);
}
"""

_GRAIN_FS = """
#version 330
in vec3 v_col; out vec4 f_color;
void main(){
    vec2 c = gl_PointCoord - 0.5;
    f_color = vec4(v_col * (1.0 - smoothstep(0.25, 0.5, length(c))), 1.0);
}
"""

_SHOW_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_acc;
uniform float u_gain, u_glow, u_lod;
void main(){
    vec3 acc = texture(u_acc, v_uv).rgb;
    vec3 halo = textureLod(u_acc, v_uv, u_lod).rgb + 0.5 * textureLod(u_acc, v_uv, u_lod + 1.5).rgb;
    vec3 col = 1.0 - exp(-(acc * u_gain + halo * u_glow));
    f_color = vec4(col, 1.0);
}
"""


def line_mask(frame: np.ndarray, white: float = 0.8) -> np.ndarray:
    """Where the inverted drawing is near white: the drawing inverted (ink turns white),
    its contrast stretched (2nd..98th percentile to 0..1), and kept above ``white``."""
    lum = frame[..., :3].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    inv = 255.0 - lum
    lo, hi = np.percentile(inv, (2, 98))
    inv = np.clip((inv - lo) / max(1.0, hi - lo), 0.0, 1.0)
    return inv >= white


def _morton(ix: np.ndarray, iy: np.ndarray) -> np.ndarray:
    """Z-order key of 10-bit coordinates: points close on the plate are close in the key."""
    def spread(v):
        v = v.astype(np.uint32) & 0x3FF
        v = (v | (v << 8)) & 0x00FF00FF
        v = (v | (v << 4)) & 0x0F0F0F0F
        v = (v | (v << 2)) & 0x33333333
        return (v | (v << 1)) & 0x55555555
    return spread(ix) | (spread(iy) << 1)


class SandLines:
    def __init__(self, width: int, height: int, grains: int = 200_000, seed: int = 0,
                 white: float = 0.8, push: float = 1.2, lift: float = 2.5, width_: float = 0.12, ring_speed: float = 3.2,
                 gravity: float = 22.0, friction: float = 3.0, bounce: float = 0.3, pull: float = 3.0,
                 travel: float = 2.5, cool: float = 0.8, size: float = 1.5, bright: float = 0.35,
                 glow: float = 0.25):
        self.W, self.H = int(width), int(height)
        self.aspect = self.W / self.H
        self.side = int(math.ceil(math.sqrt(max(1, int(grains)))))
        self.n = self.side * self.side
        self.seed = int(seed)
        self.white = float(white)
        self.push, self.lift, self.width = float(push), float(lift), max(1e-3, float(width_))
        self.ring_speed = max(1e-3, float(ring_speed))
        self.gravity, self.friction, self.bounce = float(gravity), float(friction), float(np.clip(bounce, 0.0, 0.9))
        self.pull, self.travel, self.cool = float(pull), float(travel), max(1e-3, float(cool))
        self.size, self.bright, self.glow = float(size) * self.H / 720.0, float(bright), float(glow)
        self._pass = ShaderPass(_SHOW_FS, width, height, supersample=1)
        ctx, mgl = self._pass.ctx, self._pass._mgl
        self._step_prog, self._step_vao = self._pass.quad_program(_STEP_FS)
        with ctx:
            self._grain_prog = ctx.program(vertex_shader=_GRAIN_VS, fragment_shader=_GRAIN_FS)
            self._grain_vao = ctx.vertex_array(self._grain_prog, [])
            rng = np.random.default_rng(self.seed)
            air = np.zeros((self.n, 4), np.float32)
            air[:, 3] = rng.random(self.n, dtype=np.float32)
            move = np.zeros((self.n, 4), np.float32)
            move[:, 0] = rng.uniform(-self.aspect, self.aspect, self.n)
            move[:, 1] = rng.uniform(-1, 1, self.n)
            self._move = [ctx.texture((self.side, self.side), 4, data=move.tobytes(), dtype="f4") for _ in range(2)]
            self._air = [ctx.texture((self.side, self.side), 4, data=air.tobytes(), dtype="f4") for _ in range(2)]
            self._home = ctx.texture((self.side, self.side), 4, data=move.tobytes(), dtype="f4")
            for t in (*self._move, *self._air, self._home):
                t.filter = (mgl.NEAREST, mgl.NEAREST)
            self._fbo = [ctx.framebuffer([self._move[i], self._air[i]]) for i in range(2)]
            self._acc = ctx.texture((self.W, self.H), 4, dtype="f2")
            self._acc.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            self._acc_fbo = ctx.framebuffer([self._acc])
        self._cur = 0
        self._placed = False
        self._frame = 0
        self._jump = 0.0
        self._lod = float(math.log2(max(2.0, 6.0 * self.H / 720.0)))

    # -- the drawing -----------------------------------------------------------
    def lines(self, frame: np.ndarray, bar: int = 0) -> bool:
        """Give the grains homes on the white lines of ``frame`` (inverted, stretched, near white).
        The first time they are laid right there; after that they travel to them. False when
        the picture has (almost) no lines, and the old homes are kept."""
        mask = line_mask(frame, self.white)
        ys, xs = np.nonzero(mask)
        if len(xs) < 50:
            return False
        rng = np.random.default_rng([self.seed, int(bar) & 0x7FFFFFFF])
        pick = rng.integers(0, len(xs), self.n)
        hx = (xs[pick] + rng.random(self.n)) / self.W * 2.0 * self.aspect - self.aspect
        hy = (ys[pick] + rng.random(self.n)) / self.H * 2.0 - 1.0
        homes = np.zeros((self.n, 4), np.float32)
        ctx = self._pass.ctx
        with ctx:
            if not self._placed:
                homes[:, 0], homes[:, 1] = hx, hy
                self._move[self._cur].write(homes.tobytes())       # the first drawing: the sand is laid on it
                self._placed = True
            else:
                # the grains go to the nearest lines of the new drawing: pair the grains and the new
                # homes in the order of a space-filling curve, so each goes somewhere near
                cur = np.frombuffer(self._move[self._cur].read(), np.float32).reshape(self.n, 4)
                key = lambda x, y: _morton(((x + self.aspect) / (2 * self.aspect) * 1023).clip(0, 1023),
                                           ((y + 1) / 2 * 1023).clip(0, 1023))
                order_g = np.argsort(key(cur[:, 0], cur[:, 1]), kind="stable")
                order_h = np.argsort(key(hx, hy), kind="stable")
                homes[order_g, 0], homes[order_g, 1] = hx[order_h], hy[order_h]
            self._home.write(homes.tobytes())
        return True

    def jump(self, strength: float) -> None:
        """A hit: every grain resting on the plate hops (``strength`` 0..1)."""
        self._jump += float(strength)

    # -- a frame -----------------------------------------------------------------
    def render(self, dt: float, rings=(), hue: float = 0.9, sat: float = 0.75, substeps: int = 2) -> np.ndarray:
        """Advance ``dt`` seconds and draw. ``rings``: up to 16 of (x, y, radius, strength) in frame units."""
        rs = [(float(x), float(y), float(r), float(s)) for x, y, r, s in list(rings)[:MAX_RINGS]]
        rs += [(0.0, 0.0, 0.0, 0.0)] * (MAX_RINGS - len(rs))
        ctx, mgl = self._pass.ctx, self._pass._mgl
        steps = max(1, int(substeps))
        h = max(0.0, float(dt)) / steps
        with ctx:
            sp = self._step_prog
            for name, value in (("u_move", 1), ("u_air", 2), ("u_home", 3), ("u_ring", rs),
                                ("u_aspect", self.aspect), ("u_dt", h), ("u_width", self.width),
                                ("u_push", self.push), ("u_lift", self.lift), ("u_gravity", self.gravity),
                                ("u_friction", self.friction), ("u_pull", self.pull), ("u_cool", self.cool),
                                ("u_bounce", self.bounce), ("u_travel", self.travel), ("u_rspeed", self.ring_speed)):
                if name in sp:
                    sp[name].value = value
            self._home.use(3)
            for k in range(steps):
                sp["u_jump"].value = self._jump * 1.2 if k == 0 else 0.0
                sp["u_frame"].value = float((self._frame * steps + k) % 8192)
                self._move[self._cur].use(1)
                self._air[self._cur].use(2)
                self._fbo[1 - self._cur].use()
                self._step_vao.render(mgl.TRIANGLE_STRIP)
                self._cur = 1 - self._cur
            self._jump = 0.0
            # the grains, added up
            self._acc_fbo.use()
            self._acc_fbo.clear(0.0, 0.0, 0.0, 0.0)
            ctx.enable(mgl.BLEND | mgl.PROGRAM_POINT_SIZE)
            ctx.blend_func = mgl.ONE, mgl.ONE
            gp = self._grain_prog
            self._move[self._cur].use(1)
            self._air[self._cur].use(2)
            for name, value in (("u_move", 1), ("u_air", 2), ("u_side", self.side), ("u_aspect", self.aspect),
                                ("u_size", self.size), ("u_hue", float(hue) % 1.0),
                                ("u_sat", float(np.clip(sat, 0.0, 1.0))), ("u_bright", self.bright)):
                if name in gp:
                    gp[name].value = value
            self._grain_vao.render(mgl.POINTS, vertices=self.n)
            ctx.disable(mgl.BLEND | mgl.PROGRAM_POINT_SIZE)
            self._acc.build_mipmaps()
        self._frame += 1
        return self._pass.draw(textures={1: self._acc}, u_acc=1, u_gain=1.0, u_glow=self.glow, u_lod=self._lod)

    def state(self) -> tuple:
        """(positions N x 2, heights N) of the grains, for tests."""
        with self._pass.ctx:
            m = np.frombuffer(self._move[self._cur].read(), np.float32).reshape(self.n, 4)
            a = np.frombuffer(self._air[self._cur].read(), np.float32).reshape(self.n, 4)
        return m[:, :2].copy(), a[:, 0].copy()

"""Drops — a drop of colour falls on the drawing and spreads until it meets the lines. GPU post-op.

A paint layer lives on the GPU, as big as the frame, and outlasts it: every
pixel is either dry (empty) or holds the colour of the drop that reached it and
the time that drop fell. Each frame:

    ink     the lines of the picture below are found (``core/frame_read``): these are
            the walls the paint cannot cross
    seed    a new drop is a small disc of its colour where it fell
    grow    the paint creeps one pixel at a time into empty neighbours that are not
            lines, ``speed`` pixels per frame, for ``grow`` seconds after the drop fell;
            so a drop fills the region of the drawing it fell in, like a slow paint
            bucket, and stops at its outline
    dry     after ``life`` seconds the stain fades out and the pixels are empty again

With ``shape="spiral"`` the paint still fills the region, but it shows only on an arm
that winds out from where the drop fell (``pitch`` pixels between turns, ``turns``
turns a second, ``arm`` the painted share of each turn; 1 is a solid snail shell)
and whirling (``whirl`` turns a second when it lands, slowing as it dries), and the
drawing around it is twisted like a whirlpool (``twirl`` radians at its centre),
each drop turning its own way (``drop(..., spin=±1)``).

The stain is laid over the picture (``flood``: the colour replaces the fill, or
``stain``: it tints it, keeping its shading), with a darker wet rim like watercolour,
and the lines stay drawn on top. As the video moves the stain stays where it is on
the screen and keeps filling whatever region is now under it; lines that appear
erase the paint under them, and when the picture cuts to another (``cut(t)``) the
stains laid on the old one dry in a moment.

``drop(x, y, colour, radius)`` lets a drop fall (image pixels, y down);
``render(frame, t)`` runs a frame.
"""
from __future__ import annotations

import numpy as np

from core.frame_read import INK_GLSL, ink_lod
from core.shader_pass import ShaderPass

MAX_NEW = 8                 # drops that can fall on one frame
MAX_WET = 32                # drops a spiral can remember at once
SHAPES = ("blob", "spiral")

_INK_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_img;
uniform float u_lod, u_ink;
""" + INK_GLSL + """
void main(){
    // the walls, thickened by a pixel all round so thin or broken strokes still hold the paint
    ivec2 p = ivec2(gl_FragCoord.xy), size = textureSize(u_img, 0);
    float ink = 0.0;
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
        ivec2 q = clamp(p + ivec2(i, j), ivec2(0), size - 1);
        vec2 uv = (vec2(q) + 0.5) / vec2(size);
        ink = max(ink, inkOf(texelFetch(u_img, q, 0).rgb, luma(textureLod(u_img, uv, u_lod).rgb), u_ink));
    }
    f_color = vec4(ink, 0.0, 0.0, 1.0);
}
"""

# paint texel: rgb = colour, a = when its drop fell (-1: dry). Texture rows are image rows (y down).
_GROW_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_paint, u_inkmap;
uniform float u_t, u_life, u_grow, u_wall, u_cut, u_step, u_rough;
uniform vec4 u_new[8];          // x, y, radius (pixels), birth time (< 0: none)
uniform vec3 u_newc[8];
const float CUT_DRY = 0.3;      // seconds for the stains of a cut-away picture to dry
// Dave Hoskins' hash without sine
float hash1(vec2 p){ vec3 p3 = fract(vec3(p.xyx) * 0.1031); p3 += dot(p3, p3.yzx + 33.33); return fract((p3.x + p3.y) * p3.z); }

void main(){
    ivec2 p = ivec2(gl_FragCoord.xy), size = textureSize(u_paint, 0);
    vec4 me = texelFetch(u_paint, p, 0);
    bool wall = texelFetch(u_inkmap, p, 0).r > u_wall;
    bool stale = me.a >= 0.0 && me.a < u_cut;                                           // painted before a cut
    if (me.a >= 0.0 && (u_t - me.a >= u_life || (stale && u_t - u_cut >= CUT_DRY) || wall))
        me = vec4(0.0, 0.0, 0.0, -1.0);                                                  // dried out, or a line now
    for (int i = 0; i < 8; i++){                                                          // a drop falls here
        if (u_new[i].w < 0.0) continue;
        if (distance(vec2(p), u_new[i].xy) <= u_new[i].z && !wall) me = vec4(u_newc[i], u_new[i].w);
    }
    // the paint creeps in, a little unevenly (a random draw per pixel and step, and 4 or 8
    // neighbours in turn) so its front is a ragged round, not an octagon
    if (me.a < 0.0 && !wall && hash1(vec2(p) + u_step * 7.31) >= u_rough){
        float best = -1.0; vec3 col = vec3(0.0);
        bool diag = mod(u_step, 2.0) > 0.5;
        for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
            if (i == 0 && j == 0) continue;
            if (!diag && i != 0 && j != 0) continue;
            ivec2 q = clamp(p + ivec2(i, j), ivec2(0), size - 1);
            vec4 n = texelFetch(u_paint, q, 0);
            if (n.a >= u_cut && u_t - n.a < u_grow && n.a > best){ best = n.a; col = n.rgb; }   // the youngest wet drop wins
        }
        if (best >= 0.0) me = vec4(col, best);
    }
    f_color = me;
}
"""

_SHOW_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_img, u_paint, u_inkmap;
uniform float u_t, u_life, u_fade, u_opacity, u_rim, u_stain, u_cut, u_lod, u_ink;
uniform float u_spiral, u_pitch, u_turns, u_arm, u_whirl, u_twirl;
uniform vec4 u_drops[32];       // the drops still wet: x, y, birth time, turning (+1 / -1; 0: none)
""" + INK_GLSL + """
// the spiral: the paint of a drop shows only on an arm that winds out from where it fell,
// ``u_turns`` turns a second, ``u_pitch`` pixels between turns, ``u_arm`` of each turn painted
float spiralAt(ivec2 p, float birth){
    if (u_spiral < 0.5) return 1.0;
    vec4 o = vec4(0.0); float near = 1e9;
    for (int i = 0; i < 32; i++){                       // which drop this paint came from
        vec4 d = u_drops[i];
        if (d.w == 0.0 || abs(d.z - birth) > 1e-3) continue;
        float r = distance(vec2(p), d.xy);
        if (r < near){ near = r; o = d; }
    }
    if (o.w == 0.0) return 1.0;
    vec2 d = vec2(p) - o.xy;
    float e = u_t - birth;
    // it whirls: fast when the drop lands, slowing as the stain dries, the arms flowing outwards
    float phase = u_whirl * u_life / 3.0 * (1.0 - exp(-3.0 * e / u_life));
    float s = length(d) / u_pitch + fract(o.w * atan(d.y, d.x) / 6.2831853 - phase + 2.0);   // turns out from the centre
    float aa = 1.2 / u_pitch, front = u_turns * e;
    float m = 1.0 - smoothstep(front - aa, front + aa, s);                              // how far the arm has wound
    if (u_arm < 0.999){
        float f = fract(s);
        m *= smoothstep(0.0, aa, f) * (1.0 - smoothstep(u_arm - aa, u_arm, f));        // the arm, and the gap between turns
    }
    return m;
}
float wetAt(ivec2 p, ivec2 size){
    ivec2 q = clamp(p, ivec2(0), size - 1);
    vec4 n = texelFetch(u_paint, q, 0);
    if (n.a < 0.0) return 0.0;
    float v = (1.0 - smoothstep(u_life - u_fade, u_life, u_t - n.a)) * spiralAt(q, n.a);
    return n.a < u_cut ? v * (1.0 - smoothstep(0.0, 0.3, u_t - u_cut)) : v;             // a cut: the old stains dry
}
void main(){
    ivec2 size = textureSize(u_paint, 0);
    ivec2 p = ivec2(int(gl_FragCoord.x), size.y - 1 - int(gl_FragCoord.y));             // image rows: y down
    // the whirlpool: around each spiral the drawing itself is twisted, most at the centre,
    // winding in as the drop spins up and letting go as the stain dries
    vec2 at = vec2(p) + 0.5;
    if (u_spiral > 0.5 && u_twirl != 0.0){
        for (int i = 0; i < 32; i++){
            vec4 o = u_drops[i];
            if (o.w == 0.0) continue;
            float e = u_t - o.z;
            float k = (1.0 - exp(-3.0 * e / u_life)) * (1.0 - smoothstep(u_life - u_fade, u_life, e));
            if (o.z < u_cut) k *= 1.0 - smoothstep(0.0, 0.3, u_t - u_cut);
            float reach = u_pitch * clamp(u_turns * e, 1.0, 8.0);
            vec2 d = at - o.xy;
            float a = o.w * u_twirl * k * exp(-dot(d, d) / (reach * reach));
            at = o.xy + mat2(cos(a), sin(a), -sin(a), cos(a)) * d;
        }
    }
    vec2 uv = at / vec2(size);
    vec3 c = texture(u_img, uv).rgb;
    vec4 paint = texelFetch(u_paint, p, 0);
    float v = wetAt(p, size);
    // the wet rim: paint whose neighbourhood is not all paint
    float around = 0.0;
    for (int k = 0; k < 8; k++){
        float a = float(k) * 0.7853982;
        around += wetAt(p + ivec2(round(3.0 * vec2(cos(a), sin(a)))), size);
    }
    float rim = v * (1.0 - around / 8.0);
    vec3 drop = mix(paint.rgb, c * paint.rgb * 1.6, u_stain);                             // flood .. stain
    vec3 col = mix(c, drop, v * u_opacity);
    col *= 1.0 - u_rim * rim;                                                             // watercolour edge
    col = mix(col, c, inkOf(c, luma(textureLod(u_img, uv, u_lod).rgb), u_ink));          // the lines stay on top
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


class Drops:
    def __init__(self, width: int, height: int, speed: int = 6, grow: float = 2.0, life: float = 4.0,
                 fade: float = 1.0, ink: float = 1.0, wall: float = 0.5, opacity: float = 0.9,
                 rim: float = 0.35, stain: float = 0.0, rough: float = 0.35, shape: str = "blob",
                 pitch: float = 30.0, turns: float = 2.0, arm: float = 0.55, whirl: float = 1.5,
                 twirl: float = 1.5):
        if shape not in SHAPES:
            raise ValueError(f"drops: unknown shape {shape!r} (use one of {SHAPES})")
        self.W, self.H = int(width), int(height)
        self.speed = max(1, int(speed))
        self.grow, self.life, self.fade = float(grow), float(life), max(1e-3, float(fade))
        self.ink, self.wall, self.opacity = float(ink), float(wall), float(opacity)
        self.rim, self.stain, self.rough = float(rim), float(stain), float(np.clip(rough, 0.0, 0.9))
        self.spiral = shape == "spiral"
        self.pitch, self.turns, self.arm = max(2.0, float(pitch)), float(turns), float(np.clip(arm, 0.05, 1.0))
        self.whirl, self.twirl = float(whirl), float(twirl)
        self._wet: list = []                    # (x, y, birth, spin) of the drops that may still show
        self._cut = -1e9
        self._steps = 0
        self._pass = ShaderPass(_SHOW_FS, width, height, supersample=1)
        ctx, mgl = self._pass.ctx, self._pass._mgl
        self._ink_prog, self._ink_vao = self._pass.quad_program(_INK_FS)
        self._grow_prog, self._grow_vao = self._pass.quad_program(_GROW_FS)
        with ctx:
            self._img = ctx.texture((self.W, self.H), 3)
            self._img.filter = (mgl.LINEAR_MIPMAP_LINEAR, mgl.LINEAR)
            self._inkmap = ctx.texture((self.W, self.H), 1, dtype="f2")
            self._inkfbo = ctx.framebuffer([self._inkmap])
            empty = np.tile(np.array([0, 0, 0, -1], np.float32), self.W * self.H).tobytes()
            self._paint = [ctx.texture((self.W, self.H), 4, data=empty, dtype="f4") for _ in range(2)]
            self._pfbo = [ctx.framebuffer([t]) for t in self._paint]
            for t in (self._inkmap, *self._paint):
                t.filter = (mgl.NEAREST, mgl.NEAREST)
        self._cur = 0
        self._pending: list = []
        self._lod = ink_lod(self.H)

    def drop(self, x: float, y: float, colour, radius: float = 6.0, t: float = 0.0, spin: float = 1.0) -> None:
        """A drop falls at image pixel (x, y) at time ``t``; ``colour`` is RGB 0..1; ``spin``
        (+1 / -1) the way its spiral turns."""
        self._pending.append((float(x), float(y), float(radius), float(t), tuple(float(c) for c in colour)))
        self._wet.append((float(x), float(y), float(t), 1.0 if spin >= 0 else -1.0))

    def cut(self, t: float) -> None:
        """The picture below changed completely at ``t``: the stains laid on the old one dry at once."""
        self._cut = float(t)

    def render(self, frame: np.ndarray, t: float) -> np.ndarray:
        ctx = self._pass.ctx
        with ctx:
            self._img.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            self._img.build_mipmaps()
            self._inkfbo.use()
            self._img.use(1)
            self._ink_prog["u_img"].value = 1
            self._ink_prog["u_lod"].value = self._lod
            self._ink_prog["u_ink"].value = max(0.1, self.ink)
            self._ink_vao.render(self._pass._mgl.TRIANGLE_STRIP)
            new, self._pending = self._pending[:MAX_NEW], self._pending[MAX_NEW:]
            g = self._grow_prog
            for name, value in (("u_inkmap", 2), ("u_paint", 3), ("u_t", float(t)), ("u_life", self.life),
                                ("u_grow", self.grow), ("u_wall", self.wall), ("u_cut", self._cut),
                                ("u_rough", self.rough)):
                if name in g:
                    g[name].value = value
            self._inkmap.use(2)
            for step in range(self.speed):
                seeds = new if step == 0 else []
                pos = [(x, y, r, b) for x, y, r, b, _ in seeds] + [(0.0, 0.0, 0.0, -1.0)] * (MAX_NEW - len(seeds))
                col = [c for *_, c in seeds] + [(0.0, 0.0, 0.0)] * (MAX_NEW - len(seeds))
                g["u_new"].value = pos
                g["u_newc"].value = col
                g["u_step"].value = float(self._steps % 4096)
                self._steps += 1
                self._paint[self._cur].use(3)
                self._pfbo[1 - self._cur].use()
                self._grow_vao.render(self._pass._mgl.TRIANGLE_STRIP)
                self._cur = 1 - self._cur
        self._wet = [d for d in self._wet if t - d[2] < self.life][-MAX_WET:]
        wet = self._wet + [(0.0, 0.0, -1.0, 0.0)] * (MAX_WET - len(self._wet))
        return self._pass.draw(textures={1: self._img, 2: self._inkmap, 3: self._paint[self._cur]},
                               u_spiral=1.0 if self.spiral else 0.0, u_pitch=self.pitch, u_turns=self.turns,
                               u_arm=self.arm, u_whirl=self.whirl, u_twirl=self.twirl, u_drops=wet,
                               u_img=1, u_inkmap=2, u_paint=3, u_t=float(t), u_life=self.life, u_fade=self.fade,
                               u_opacity=self.opacity, u_rim=self.rim, u_stain=self.stain, u_cut=self._cut,
                               u_lod=self._lod, u_ink=max(0.1, self.ink))

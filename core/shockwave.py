"""Shockwave — rings that travel out through a drawing and light up its lines. GPU post-op.

The ``ring`` gesture of the sand (``core/chladni``) on any image: a ring is born
at a point and runs outwards; where it passes it bends the image like a lens
(the picture is pushed out along the radius) and tints **only the lines** of
the drawing, leaving the ground as it is.

The lines are found the way ink is: a pixel much darker than its neighbourhood
(a high-pass on the luminance, the neighbourhood read from a mipmap) or simply
very dark. With ``invert`` the drawing is turned inside out, as the sand looks:
the ink becomes white lines, the fills become their dim negatives (``fill``
sets how much of them is left), so a cartoon reads as white strokes on dark.

    rings       up to 16 of (x, y, radius, strength): the centre in frame units
                (the frame is 2 tall, 0 in the middle), how far the ring has run,
                and how strong it still is (it fades as it goes)
    punch       0..1 the whole picture swells a little and flashes (a hit)
    hue, sat    the colour the rings give to the lines
"""
from __future__ import annotations

import numpy as np

from core.shader_pass import ShaderPass

MAX_RINGS = 16
# the sand's numbers (core/chladni: RING_SPEED, RING_WIDTH, RING_PUSH, RING_LIFE)
RING_SPEED = 3.2
RING_WIDTH = 0.12
RING_PUSH = 0.06
RING_LIFE = 0.6

_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_img;
uniform vec4 u_ring[16];
uniform float u_aspect, u_width, u_push, u_punch, u_hue, u_sat, u_fill, u_invert, u_ink, u_lod;

vec3 hsv(float h, float s, float v){ vec3 k = fract(h + vec3(0, 2.0/3.0, 1.0/3.0)) * 6.0; return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s); }
float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }

void main(){
    vec2 q = (vec2(v_uv.x, 1.0 - v_uv.y) - 0.5) * vec2(u_aspect, 1.0) * 2.0;    // frame units, y down
    vec2 disp = vec2(0.0);
    float heat = 0.0;
    for (int i = 0; i < 16; i++){
        vec4 R = u_ring[i];
        if (R.w <= 0.0) continue;
        vec2 d = q - R.xy;
        float r = length(d) + 1e-4;
        float band = exp(-pow((r - R.z) / u_width, 2.0)) * R.w;
        disp += d / r * u_push * band;                        // pushed out along the radius, like the sand
        heat += band;
    }
    vec2 src = (q - disp) / (1.0 + 0.05 * u_punch);                // the punch swells the picture
    vec2 uv = src / (vec2(u_aspect, 1.0) * 2.0) + 0.5;
    vec3 c = texture(u_img, uv).rgb;
    float L = luma(c), Lm = luma(textureLod(u_img, uv, u_lod).rgb);
    // ink: darker than its neighbourhood (high-pass), or plainly dark
    float ink = max(smoothstep(0.03, 0.03 + 0.25 / u_ink, Lm - L), smoothstep(0.30, 0.10, L));
    vec3 tint = hsv(u_hue, u_sat, 1.0);
    vec3 col;
    if (u_invert > 0.5){
        vec3 ground = (1.0 - c) * u_fill;                                     // the fills, dim negatives
        vec3 line = mix(vec3(1.0), tint * 1.3, clamp(heat * 1.5, 0.0, 1.0));  // white lines, tinted by the ring
        col = mix(ground, line, ink) + tint * ink * heat * 0.6;               // ... and glowing where it passes
    } else {
        col = mix(c, tint, ink * clamp(heat * 1.5, 0.0, 1.0));
    }
    col *= 1.0 + 0.4 * u_punch;
    f_color = vec4(clamp(col, 0.0, 1.0), 1.0);
}
"""


class Shockwave:
    def __init__(self, width: int, height: int, invert: bool = True, fill: float = 0.3,
                 ink: float = 1.0, width_: float = RING_WIDTH, push: float = RING_PUSH):
        self.W, self.H = int(width), int(height)
        self.invert, self.fill, self.ink = bool(invert), float(fill), float(ink)
        self.width, self.push = float(width_), float(push)
        self._pass = ShaderPass(_FS, width, height, supersample=1)
        with self._pass.ctx:
            self._img = self._pass.ctx.texture((self.W, self.H), 3)
            self._img.filter = (self._pass._mgl.LINEAR_MIPMAP_LINEAR, self._pass._mgl.LINEAR)
            self._img.repeat_x = self._img.repeat_y = False
        # the neighbourhood the ink is judged against: ~12 px at 720p, whatever the size
        self._lod = float(np.log2(max(2.0, 12.0 * self.H / 720.0)))

    def render(self, frame: np.ndarray, rings=(), punch: float = 0.0, hue: float = 0.9,
               sat: float = 0.75) -> np.ndarray:
        """``rings``: up to 16 of (x, y, radius, strength) in frame units."""
        rs = [(float(x), float(y), float(r), float(s)) for x, y, r, s in list(rings)[:MAX_RINGS]]
        rs += [(0.0, 0.0, 0.0, 0.0)] * (MAX_RINGS - len(rs))
        with self._pass.ctx:
            self._img.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
            self._img.build_mipmaps()
        return self._pass.draw(textures={1: self._img}, u_img=1, u_ring=rs, u_width=self.width,
                               u_push=self.push, u_punch=float(np.clip(punch, 0.0, 1.0)),
                               u_hue=float(hue) % 1.0, u_sat=float(np.clip(sat, 0.0, 1.0)),
                               u_fill=self.fill, u_invert=1.0 if self.invert else 0.0,
                               u_ink=max(0.1, self.ink), u_lod=self._lod)

"""Slit-scan — every line of the frame shows a different moment of the past. GPU.

The frames that pass through are kept in a ring buffer on the GPU (a texture
array). Each output pixel reads the buffer ``delay`` frames back, where the
delay grows along one direction of the frame, so a moving image smears and
bends through time, as in a slit-scan camera (the stargate in "2001"):

    down / up       rows: the bottom (down) or the top (up) of the frame is the oldest
    right / left    columns: the right or the left edge is the oldest
    radial          rings: the rim is the oldest, the centre is now
    center          rings: the centre is the oldest, the rim is now

``render(frame, delay, wave_front, wave_depth)``: ``delay`` is the delay (in
frames) at the far end of the direction, ``wave_front`` 0..1 a band of extra
delay ``wave_depth`` frames deep crossing the frame (negative = no wave).
Delays are fractional: the two nearest frames are blended, so nothing stutters.
Until the buffer has filled, delays are clamped to what has been seen.
"""
from __future__ import annotations

import numpy as np

from core.shader_pass import ShaderPass

MODES = ("down", "up", "right", "left", "radial", "center")

_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2DArray u_hist;
uniform float u_aspect, u_delay, u_front, u_depth, u_head, u_n, u_filled;
uniform int u_mode;

void main(){
    vec2 img = vec2(v_uv.x, 1.0 - v_uv.y);                  // image coordinates: y runs down
    float s;                                                 // 0 = now, 1 = the far (oldest) end
    if (u_mode == 0) s = img.y;
    else if (u_mode == 1) s = 1.0 - img.y;
    else if (u_mode == 2) s = img.x;
    else if (u_mode == 3) s = 1.0 - img.x;
    else {
        float r = length((img - 0.5) * vec2(u_aspect, 1.0)) / length(vec2(u_aspect, 1.0) * 0.5);
        s = u_mode == 4 ? r : 1.0 - r;
    }
    float d = u_delay * s;
    if (u_front >= 0.0) d += u_depth * exp(-pow((s - u_front) / 0.12, 2.0));   // a wave of the past sweeping across
    d = clamp(d, 0.0, u_filled - 1.0);
    float d0 = floor(d), k = d - d0;
    float l0 = mod(u_head - d0 + u_n, u_n), l1 = mod(u_head - d0 - 1.0 + u_n, u_n);
    vec3 a = texture(u_hist, vec3(img, l0)).rgb, b = texture(u_hist, vec3(img, l1)).rgb;
    f_color = vec4(mix(a, b, k), 1.0);
}
"""


class SlitScan:
    def __init__(self, width: int, height: int, depth: int = 60, mode: str = "down"):
        if mode not in MODES:
            raise ValueError(f"slitscan: unknown mode {mode!r} (use one of {MODES})")
        self.W, self.H, self.n = int(width), int(height), max(2, int(depth))
        self.mode = MODES.index(mode)
        self._pass = ShaderPass(_FS, width, height, supersample=1)
        self._hist = self._pass.ctx.texture_array((self.W, self.H, self.n), 3)
        self._head = -1
        self._filled = 0

    def push(self, frame: np.ndarray) -> None:
        """Remember a frame (uint8 H x W x 3) as the newest moment."""
        self._head = (self._head + 1) % self.n
        self._filled = min(self.n, self._filled + 1)
        with self._pass.ctx:
            self._hist.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes(),
                             viewport=(0, 0, self._head, self.W, self.H, 1))

    def render(self, frame: np.ndarray, delay: float, wave_front: float = -1.0,
               wave_depth: float = 0.0) -> np.ndarray:
        """Push ``frame`` and return it scanned through the buffer."""
        self.push(frame)
        return self._pass.draw(textures={1: self._hist}, u_hist=1, u_delay=float(max(0.0, delay)),
                               u_front=float(wave_front), u_depth=float(max(0.0, wave_depth)), u_head=float(self._head),
                               u_n=float(self.n), u_filled=float(self._filled), u_mode=self.mode)

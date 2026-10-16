"""Nebulabrot: the orbits that escape the Mandelbrot set, as a nebula, its channels the stems.

The Buddhabrot keeps, for every point c that escapes the Mandelbrot set, the path its
orbit took, and adds it up: a ghostly density. Split by how long the orbits last, one
colour per iteration cap, it is the Nebulabrot (short orbits make the broad red glow,
long ones the fine blue filaments). A GPU compute shader samples the c's and adds the
orbits into three histograms; they build up across frames, fading slowly, and each
channel's light follows the music.

    - source: nebulabrot
      caps: [40, 250, 1500]       # the iteration caps of the red, green and blue channels
      samples: 1.0                # millions of c tried per frame
      memory: 0.94                # how much of the density stays from frame to frame
      red: {relative: subbands.bass, window: 6}        # each channel's light (a knob: any form
      green: {relative: subbands.mid, window: 6}       # of the shadertoy layer's uniforms);
      blue: {relative: subbands.presence, window: 6}   # 1 = as built up
      turn: 0.0                   # radians the set is turned (a knob: e.g. rate)
      zoom: 1.0                   # how much of the set the frame spans (a knob)

GPU only (a compute shader, GL 4.3, like ``flame``); the test skips without one.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

_CS = """
#version 430
layout(local_size_x = 256) in;
layout(std430, binding = 0) buffer Counts { uint counts[]; };
uniform int u_w, u_h, u_per, u_seed;
uniform ivec3 u_caps;
uniform float u_cs, u_sn, u_scale;

uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16;
    return x;
}
float rnd(inout uint s) { s = hash(s); return float(s) / 4294967295.0; }

void plot(vec2 z, int n) {
    // the real axis runs down the frame (the classic Buddhabrot pose), turned by u_turn
    vec2 q = vec2(z.y, z.x + 0.5);
    q = vec2(u_cs * q.x - u_sn * q.y, u_sn * q.x + u_cs * q.y);
    float side = float(min(u_w, u_h));
    int x = int(floor(q.x * u_scale * side * 0.5 + float(u_w) * 0.5));
    int y = int(floor(q.y * u_scale * side * 0.5 + float(u_h) * 0.5));
    if (x < 0 || y < 0 || x >= u_w || y >= u_h) return;
    uint i = uint(y * u_w + x) * 3u;
    if (n <= u_caps.x) atomicAdd(counts[i], 1u);
    if (n <= u_caps.y) atomicAdd(counts[i + 1u], 1u);
    atomicAdd(counts[i + 2u], 1u);
}

void main() {
    uint s = hash(gl_GlobalInvocationID.x * 747796405u + uint(u_seed) * 2891336453u);
    for (int k = 0; k < u_per; k++) {
        vec2 c = vec2(rnd(s) * 3.0 - 2.0, rnd(s) * 3.0 - 1.5);
        // skip the main cardioid and the period-2 bulb: they never escape
        float q = (c.x - 0.25) * (c.x - 0.25) + c.y * c.y;
        if (q * (q + c.x - 0.25) < 0.25 * c.y * c.y) continue;
        if ((c.x + 1.0) * (c.x + 1.0) + c.y * c.y < 0.0625) continue;
        vec2 z = vec2(0.0);
        int n = 0;
        for (; n < u_caps.z; n++) {
            z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
            if (dot(z, z) > 4.0) break;
        }
        if (n >= u_caps.z || n < 5) continue;              // stayed (or fled at once): not part of it
        z = vec2(0.0);
        for (int j = 0; j < n; j++) {
            z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
            if (j >= 2) plot(z, n);                        // the first steps only repeat where c was
        }
    }
}
"""


class NebulabrotLayer:
    """Source: see the module docstring."""

    def __init__(self, spec: dict, notes: Sequence, width: int, height: int, fps: int,
                 features_at=None, onset_loader=None, grid=None):
        from core.gpu import context
        from core.video.layers import _music_knob
        self.W, self.H = int(width), int(height)
        caps = [int(c) for c in spec.get("caps", [40, 250, 1500])]
        if len(caps) != 3 or not caps[0] < caps[1] < caps[2]:
            raise ValueError(f"nebulabrot: caps {caps} (three, rising: red, green, blue)")
        self.caps = tuple(caps)
        self.per = 8
        self.threads = max(256, int(float(spec.get("samples", 1.0)) * 1e6 / self.per))
        self.memory = float(np.clip(float(spec.get("memory", 0.94)), 0.0, 0.999))
        knob = lambda name, default: _music_knob(name, spec.get(name, default), notes, features_at, onset_loader, grid)
        self._light = [knob("red", 1.0), knob("green", 1.0), knob("blue", 1.0)]
        self._turn, self._zoom = knob("turn", 0.0), knob("zoom", 1.0)
        self.ctx = ctx = context()
        self.cs = ctx.compute_shader(_CS)
        self.buf = ctx.buffer(reserve=self.W * self.H * 3 * 4)
        self.acc = np.zeros((self.H, self.W, 3), np.float32)
        self._frame = 0
        self._peak = np.full(3, 1e-6, np.float32)

    def frame_at(self, t: float) -> np.ndarray:
        cs = self.cs
        self.buf.clear()
        turn = float(self._turn(t))
        values = dict(u_w=self.W, u_h=self.H, u_per=self.per, u_seed=self._frame * 7919 + 13,
                      u_caps=self.caps, u_cs=float(np.cos(turn)), u_sn=float(np.sin(turn)),
                      u_scale=float(0.62 * max(0.1, self._zoom(t))))
        for k, v in values.items():
            if k in cs:
                cs[k].value = v
        self.buf.bind_to_storage_buffer(0)
        cs.run(group_x=(self.threads + 255) // 256)
        self.ctx.memory_barrier()
        counts = np.frombuffer(self.buf.read(), dtype=np.uint32).reshape(self.H, self.W, 3).astype(np.float32)
        self._frame += 1
        self.acc = self.acc * np.float32(self.memory) + counts
        # each channel against its own bright end, slowly followed, so it neither flickers nor clips
        peak = np.percentile(self.acc.reshape(-1, 3), 99.7, axis=0).astype(np.float32) + 1e-6
        self._peak = np.maximum(self._peak * np.float32(0.98), peak)
        v = np.log1p(self.acc / self._peak * 6.0) / np.log1p(6.0)
        light = np.array([max(0.0, float(f(t))) for f in self._light], np.float32)
        img = np.clip(v * light[None, None, :], 0.0, 1.0) ** np.float32(0.8)
        return (img * 255.0 + 0.5).astype(np.uint8)

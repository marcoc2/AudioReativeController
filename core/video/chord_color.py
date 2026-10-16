"""The chord colours the picture: a gradient map built from the chord that is sounding.

The picture keeps its light and shade, and its colours come from the harmony. Each
pixel's brightness picks a colour on a ramp whose stops are the chord's notes, from the
root in the shadows up through the third and the fifth (and a seventh, a ninth..) to
the highlights. A note's hue is its place on the circle of fifths, the same wheel
``harmony:`` uses elsewhere, so the usual chord moves are short glides.

The chord's quality sets the mood of the ramp, after how listeners match music to colour
(Palmer et al., PNAS 2013: major and bright with light, saturated, yellower colours;
minor with darker, greyer, bluer ones):

    major   lighter, more saturated, leaning yellow
    minor   darker, less saturated, leaning blue
    sus, 5  in between
    tension (dim/aug, added notes) adds saturation

At a chord change the new ramp fades in over ``glide`` (a beat by default).

    - source: chord_color
      harmony: {score: input/song/sheetsage, snap: beat}   # the chords (required)
      amount: 0.7          # how much of the picture's colour the ramp takes (0..1), or any
                           # knob that follows the music (see the shadertoy layer's uniforms)
      pulse: {hits: {track: kick}, envelope: 0.3, amount: 0.25}   # optional: added to amount
      glide: 1/4           # seconds, or a note value on the song's grid (default: a beat)
      saturation: 1.0      # scales the ramp's saturation
      mask: {where: outside}   # as any layer: e.g. the world takes the chord, the figures keep theirs

Stateless: every frame is worked out from its time alone. Numpy is the reference; the GPU
pass matches it within a level or two.
"""
from __future__ import annotations

import colorsys
from typing import Sequence

import numpy as np

LUMA = (0.299, 0.587, 0.114)
BLUE, YELLOW = 0.62, 0.14             # where minor and major lean on the hue wheel

_FS = """
#version 330
out vec4 f_color;
uniform sampler2D u_src, u_lut;
uniform float u_amount;
void main(){
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec3 s = floor(texelFetch(u_src, p, 0).rgb * 255.0 + 0.5);
    int i = int(clamp(floor(dot(s, vec3(0.299, 0.587, 0.114)) + 0.5), 0.0, 255.0));
    vec3 c = vec3(texelFetch(u_lut, ivec2(0, i), 0).r, texelFetch(u_lut, ivec2(1, i), 0).r,
                  texelFetch(u_lut, ivec2(2, i), 0).r);
    vec3 v = s * (1.0 - u_amount) + c * u_amount;
    f_color = vec4(floor(clamp(v + 0.5, 0.0, 255.0)) / 255.0, 1.0);
}
"""


def _toward(h: float, target: float, k: float) -> float:
    """Hue ``h`` moved a fraction ``k`` of the short way round to ``target``."""
    return (h + k * ((target - h + 0.5) % 1.0 - 0.5)) % 1.0


def chord_ramp(root: int, quality: str, tones: Sequence[int], saturation: float = 1.0) -> np.ndarray:
    """The 256-step colour ramp (256x3 float32, 0..255) of one chord: dark to light,
    through its notes in order."""
    from core.video.layers import VeilsLayer, _chord_part
    mode = _chord_part("mode", root, quality, tuple(tones))
    tension = _chord_part("tension", root, quality, tuple(tones))
    hues = []
    for pc in tones:
        h = VeilsLayer.hue_of_root(pc)
        h = _toward(h, YELLOW, 0.15 * mode) if mode > 0.5 else _toward(h, BLUE, 0.15 * (1.0 - mode))
        hues.append(h)
    sat = np.clip((0.45 + 0.30 * mode + 0.15 * tension) * saturation, 0.0, 1.0)
    gamma = 1.25 - 0.4 * mode                           # minor darker, major lighter
    n = len(hues)
    out = np.zeros((256, 3), np.float32)
    for i in range(256):
        L = i / 255.0
        x = L * (n - 1)
        k = min(int(x), n - 2) if n > 1 else 0
        f = x - k if n > 1 else 0.0
        h = _toward(hues[k], hues[min(k + 1, n - 1)], f)
        v = L ** gamma
        s = sat * (1.0 - 0.6 * L * L)                   # highlights whiten a little
        out[i] = np.array(colorsys.hsv_to_rgb(h, s, v), np.float32) * 255.0
    return out


class ChordColor:
    """Post-op: the chord sounding now recolours the picture (see the module docstring)."""

    def __init__(self, spec: dict, notes: Sequence, fps: int, features_at=None, onset_loader=None, grid=None):
        from core.score.sheetsage_reader import parse_chord
        from core.video.cast import _Role
        from core.video.composer import load_score
        from core.video.layers import _music_knob
        hspec = spec.get("harmony")
        if not hspec or "score" not in hspec:
            raise ValueError("chord_color: needs harmony: {score: <a SheetSage folder>}")
        sat = float(spec.get("saturation", 1.0))
        times, ramps = [], []
        for span in load_score(hspec, grid).chords:
            root, quality, tones = parse_chord(span.label)
            if root is not None:                         # a no-chord stretch keeps the last chord
                times.append(span.start)
                ramps.append(chord_ramp(root, quality, tones, sat))
        if not ramps:
            raise ValueError("chord_color: the score has no chords")
        self._times = np.array(times)
        self._ramps = ramps
        glide = spec.get("glide", "1/4" if grid is not None else 0.5)
        self._glide = _Role._period(glide, grid, 0) if glide else 0.0
        self._amount = _music_knob("amount", spec.get("amount", 0.7), notes, features_at, onset_loader, grid)
        p = spec.get("pulse")
        self._pulse = _music_knob("pulse", p, notes, features_at, onset_loader, grid) if p else None
        self.on_gpu = True

    def lut(self, t: float) -> np.ndarray:
        """The ramp at ``t``: the chord's, fading in over the glide from the one before."""
        i = max(0, int(np.searchsorted(self._times, t, side="right")) - 1)
        cur = self._ramps[i]
        if i == 0 or not self._glide:
            return cur
        x = (t - self._times[i]) / self._glide
        if x >= 1.0:
            return cur
        x = max(0.0, x)
        w = np.float32(x * x * (3 - 2 * x))
        return self._ramps[i - 1] * (1 - w) + cur * w

    def amount(self, t: float) -> float:
        a = float(self._amount(t)) + (float(self._pulse(t)) if self._pulse else 0.0)
        return float(np.clip(a, 0.0, 1.0))

    def process(self, frame: np.ndarray, t: float) -> np.ndarray:
        a = np.float32(self.amount(t))
        if a <= 0.0:
            return frame
        src = frame.astype(np.float32)
        idx = np.clip(np.floor(src.dot(np.array(LUMA, np.float32)) + 0.5), 0, 255).astype(np.int32)
        out = src * (1 - a) + self.lut(t)[idx] * a
        return np.clip(np.floor(out + 0.5), 0, 255).astype(np.uint8)

    def process_gpu(self, frame, t: float, g):
        from core.video.gpu_compose import Frame
        a = self.amount(t)
        if a <= 0.0:
            return frame
        lut = g.small(np.ascontiguousarray(self.lut(t), dtype=np.float32), "chord_color")
        tex = g.run("chord_color", g.texture(), {"u_src": frame.tex(), "u_lut": lut}, fs=_FS,
                    u_amount=float(np.float32(a)))
        return Frame(g, tex=tex)

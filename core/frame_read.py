"""Frame reading — what effects that play with the picture below them extract from it.

GLSL shared by the post-ops that read the frame (``shockwave``, ``drops``):

    INK_GLSL    ``luma(c)`` and ``inkOf(c, Lm, gain)``: how much a pixel of colour ``c``
                is a drawn line, judged the way ink is: much darker than its
                neighbourhood (a high-pass on luminance; ``Lm`` is the neighbourhood's
                luminance, read from a mipmap level) or simply very dark. 0..1;
                ``gain`` > 1 finds more lines, < 1 fewer.

``ink_lod(height)`` is the mipmap level that makes the neighbourhood ~12 px at
720p, whatever the frame size.
"""
from __future__ import annotations

import math

INK_GLSL = """
float luma(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }
float inkOf(vec3 c, float Lm, float gain){
    float L = luma(c);
    return max(smoothstep(0.03, 0.03 + 0.25 / gain, Lm - L), smoothstep(0.30, 0.10, L));
}
"""


def ink_lod(height: int) -> float:
    return float(math.log2(max(2.0, 12.0 * height / 720.0)))

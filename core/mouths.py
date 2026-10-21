"""Mouths — a Voronoi wall of singing mouths. GPU fragment shader (moderngl), one pass.

Built on ``core/flesh`` (the same wall of skin as ``eyes``): every cell is a
patch of skin with one mouth in it, seen from the front:

    lips        an upper lip with a cupid's bow, a fuller lower lip, both rolling
                in towards the opening; vertical lip lines, a wet inner edge, gloss
    teeth       upper teeth fixed to the skull, lower teeth riding the jaw; each
                tooth rounded, translucent at the edge, yellower towards the gum;
                gums show when the lips pull back
    inside      a dark mouth, a tongue behind the lower teeth, strands of saliva
                that stretch and snap as the mouth opens
    face        a muzzle bulging around it, the philtrum above, the folds from
                the nose to the corners, the groove under the lower lip

Controls (all the mouths sing together, each a little differently):

    jaw         0..1 how far the jaw drops (voice level and F1, see ``core/voice``)
    spread      -1..1 round and pouting ("o", "u") .. wide ("i", "e") (F2)
    clench      0..1 lips pull back, teeth bared and shut (a hit)
    tremble     0..1 the lips quiver (a hit)
    tint        0..1 lip colour, rosy .. bruised (the chord)
    light       overall brightness
"""
from __future__ import annotations

import numpy as np

from core.flesh import FLESH_GLSL, FLESH_HEADER, FleshWall

_FS = FLESH_HEADER + """
uniform float u_time, u_jaw, u_spread, u_clench, u_tremble, u_tint;
""" + FLESH_GLSL + """
// the mouth in its own frame (x = +-1 at the corners of a relaxed mouth):
// M = (jaw, spread, clench, width); lines of the lips at x, with a quiver
struct Lips { float ui, uo, li, lo; };
Lips lipsAt(float x, vec4 M, float id, float quiver){
    float J = M.x, S = M.y, C = M.z, w = M.w;
    float u = clamp(x / w, -1.0, 1.0), k = max(0.0, 1.0 - u * u);
    float H = J * (0.85 - 0.25 * max(S, 0.0));
    float pe = mix(0.6, 0.35, max(-S, 0.0));                                   // rounder when pouting
    float y0 = -0.02 + 0.06 * S * u * u;
    float q = quiver * sin(u_time * 70.0 + x * 9.0 + id * 20.0);
    Lips l;
    l.ui = y0 + (0.45 * H + 0.14 * C) * pow(k, pe) + 0.020 * q;
    l.li = y0 - (0.55 * H + 0.10 * C) * pow(k, pe * 0.9) - 0.015 * q;
    float pout = 1.0 + 0.35 * max(-S, 0.0), thin = 1.0 - 0.3 * J - 0.25 * C;
    float tu = (0.20 + 0.06 * hash1(vec2(id, 1.0))) * pow(k, 0.55) * pout * thin
             + (0.05 * exp(-pow((abs(x) - 0.2) / 0.11, 2.0)) - 0.035 * exp(-pow(x / 0.07, 2.0))) * k;   // cupid's bow
    float tl = (0.26 + 0.07 * hash1(vec2(id, 2.0))) * pow(k, 0.6) * pout * thin;
    l.uo = l.ui + tu;
    l.lo = l.li - tl;
    return l;
}

float skinH(vec2 v, float eL, vec4 M, float id){
    float ax = abs(v.x), w = M.w;
    Lips l = lipsAt(v.x, M, id, 0.0);
    float rb = length(v * vec2(0.75, 1.0));
    float h = smax(0.9 * pow(max(0.0, 1.0 - rb * rb / 2.6), 1.5), 0.18 + 0.30 * smoothstep(1.0, 2.2, rb), 0.25);
    h += 0.12 * (fbm(v * 0.6 + id * 11.0) - 0.5);
    float dO = max(max(v.y - l.uo, l.lo - v.y), ax - w);                        // outside the lips
    h -= 0.03 * exp(-max(dO, 0.0) / 0.05);                                      // the skin dips to meet the lips
    Lips l0 = lipsAt(0.0, M, id, 0.0);
    float above = smoothstep(l0.uo + 0.55, l0.uo + 0.05, v.y) * step(l0.uo, v.y);
    h += (0.03 * exp(-pow((ax - 0.11) / 0.04, 2.0)) - 0.015 * exp(-pow(v.x / 0.05, 2.0))) * above;   // philtrum
    h -= 0.006 * pow(vnoise(vec2(v.x * 18.0, v.y * 2.0) + id), 3.0) * smoothstep(0.35, 0.0, v.y - l.uo) * step(l.uo, v.y);
    float xf = w + 0.22 - 0.5 * (v.y + 0.1);                                    // nose-to-corner folds
    float fold = ax - xf - 0.12 * sin((v.y + 0.1) * 2.0);
    h -= (0.035 + 0.04 * max(M.y, 0.0) + 0.03 * M.z) * exp(-pow(fold / 0.16, 2.0)) * smoothstep(1.1, 0.5, v.y) * smoothstep(-0.9, -0.3, v.y);
    h += 0.03 * smoothstep(0.0, 0.25, fold) * smoothstep(0.6, 0.25, fold) * smoothstep(0.9, 0.3, v.y) * smoothstep(-0.9, -0.3, v.y);   // the cheek beyond
    h -= 0.04 * exp(-pow((v.y - (l0.lo - 0.22)) / 0.07, 2.0)) * smoothstep(0.9, 0.2, ax);   // under the lower lip
    h -= 0.04 * exp(-dot(vec2(ax - w - 0.06, v.y + 0.02), vec2(ax - w - 0.06, v.y + 0.02)) / 0.01);   // the corners
    h -= 0.025 * fbm(vec2(v.x * 9.0, v.y * 2.5) + id * 5.0) * smoothstep(1.6, 0.8, rb) * smoothstep(0.2, 0.5, abs(v.y));
    h += 0.010 * vnoise(v * 15.0 + id * 7.0) + 0.006 * vnoise(v * 31.0 + id * 2.0);   // pores
    return seamShape(h, eL);
}

// one lip, seen from the front: tt runs 0 at the opening to 1 at the border with the skin
vec3 shadeLip(vec2 v, float tt, bool upper, float id, float wetness){
    float lines = vnoise(vec2(v.x * 38.0 + id * 9.0, v.y * 3.0 + tt * 1.5));
    float ny = upper ? mix(-0.8, 0.45, tt) : mix(0.55, -0.85, tt);
    vec3 n = normalize(vec3(v.x * 0.35 + (lines - 0.5) * 0.25, ny, 1.0));
    vec3 lip = mix(vec3(0.44, 0.17, 0.15), vec3(0.28, 0.10, 0.15), u_tint);
    lip *= 0.85 + 0.3 * fbm(v * 4.0 + id * 3.0);                                // uneven colour
    lip = mix(vec3(0.36, 0.06, 0.06), lip, smoothstep(0.0, 0.35, tt));         // the wet inside is darker, redder
    lip *= 0.92 + 0.16 * lines;
    lip = mix(lip, skinBase(v, id), smoothstep(0.7, 1.0, tt) * 0.55);            // the border fades into skin
    float ndl = dot(n, L), diff = max(ndl, 0.0), wrap = max((ndl + 0.5) / 1.5, 0.0);
    vec3 col = lip * (diff + (wrap - diff) * vec3(0.7, 0.18, 0.12)) + lip * 0.08;
    vec3 hv = normalize(L + vec3(0.0, 0.0, 1.0));
    float gloss = (upper ? 0.15 : 0.4) * wetness * (0.5 + 0.5 * lines) * smoothstep(0.9, 0.3, tt);
    col += gloss * pow(max(dot(n, hv), 0.0), 70.0) + 0.05 * pow(max(dot(n, hv), 0.0), 12.0);
    return col;
}

// a row of teeth: y0 the biting edge, dir = -1 for the upper row (crowns go up), +1 lower
vec3 shadeTeeth(vec2 v, float y0, float dir, float id, out float gumEdge){
    float ax = abs(v.x);
    float f = ax / 0.19 * (1.0 + 0.7 * ax);                                     // narrower as the arch turns away
    float k = floor(f), fr = fract(f);
    float th = hash1(vec2(k * sign(v.x + 1e-4) + dir * 7.0, id));
    float edge = y0 + dir * (0.02 * pow(abs(fr - 0.5) * 2.0, 3.0) - (k == 2.0 ? 0.02 : 0.0) + 0.008 * (th - 0.5));
    float along = (v.y - edge) * -dir;                                          // 0 at the edge, grows towards the gum
    gumEdge = 0.22 + 0.03 * v.x * v.x;
    if (along < 0.0) return vec3(-1.0);                                        // past the edge: not tooth
    vec3 n = normalize(vec3((fr - 0.5) * 0.9 + sign(v.x) * ax * 0.9, -dir * 0.15, 1.0));
    vec3 enamel = mix(vec3(0.80, 0.74, 0.64), vec3(0.72, 0.60, 0.44), smoothstep(0.02, gumEdge, along));
    enamel = mix(enamel, vec3(0.62, 0.55, 0.40), 0.35 * th);
    enamel = mix(vec3(0.46, 0.45, 0.47), enamel, smoothstep(0.0, 0.035, along));   // translucent at the edge
    enamel *= 0.9 + 0.2 * vnoise(vec2(fr * 3.0 + k * 7.0, along * 12.0));      // faint streaks in the enamel
    vec3 col = enamel * (0.12 + 0.55 * max(dot(n, L), 0.0)) * (1.0 - 0.35 * smoothstep(0.05, gumEdge, along));
    col += 0.35 * pow(max(dot(n, normalize(L + vec3(0.0, 0.0, 1.0))), 0.0), 50.0);
    col *= smoothstep(0.0, 0.035, fr) * smoothstep(1.0, 0.965, fr) * 0.45 + 0.55;   // gaps between teeth
    col *= smoothstep(1.05, 0.35, ax);                                          // the arch recedes into the dark
    return col;
}

void main(){
    Cell C = fleshCell();
    float id = hash1(C.cell * 3.7 + u_seed);
    vec4 hh = vec4(hash1(C.cell + 1.3), hash1(C.cell + 2.9), hash1(C.cell + 4.1), hash1(C.cell + 6.7));
    float a = 0.36 * C.dn * (0.90 + 0.20 * hh.x);
    float th = (hh.y - 0.5) * 0.18;
    mat2 rot = mat2(cos(th), sin(th), -sin(th), cos(th));
    vec2 v = rot * (C.q - C.best) / a;
    vec2 gE = rot * C.ge;
    float eL = C.e / a;

    // each mouth sings the same line its own way: a little more or less open, wider or rounder
    float idle = 0.03 + 0.02 * sin(u_time * 1.7 + id * 6.28);
    float J = clamp(u_jaw * (0.75 + 0.5 * hh.x) + idle, 0.0, 1.0);
    float S = clamp(u_spread * (0.7 + 0.6 * hh.w), -1.0, 1.0);
    float Cl = clamp(u_clench * (0.8 + 0.4 * hh.z), 0.0, 1.0);
    vec4 M = vec4(J, S, Cl, 1.0 + 0.15 * max(S, 0.0) - 0.28 * max(-S, 0.0) + 0.12 * (hh.z - 0.5));
    float quiver = u_tremble * (0.6 + 0.8 * hh.y);
    Lips l = lipsAt(v.x, M, id, quiver);
    float ax = abs(v.x);

    vec3 col;
    bool inMouth = ax < M.w && v.y < l.ui && v.y > l.li;
    bool upperLip = ax < M.w && v.y >= l.ui && v.y < l.uo;
    bool lowerLip = ax < M.w && v.y <= l.li && v.y > l.lo;
    if (inMouth){
        float depth = min(v.y - l.li, l.ui - v.y);
        col = mix(vec3(0.16, 0.025, 0.025), vec3(0.008, 0.0, 0.0), smoothstep(0.0, 0.18, depth));
        // the tongue, low in the mouth behind the lower teeth
        float yTu = -0.02 + 0.08 * v.x * v.x;
        float yTl = yTu - (J * 0.40 * (1.0 - Cl) + 0.01);
        float yTg = yTl + (0.04 + 0.14 * J) * max(0.0, 1.0 - pow(v.x / 0.8, 2.0));
        if (v.y < yTg){
            vec3 tn = normalize(vec3(v.x * 0.8 + (vnoise(v * 40.0) - 0.5) * 0.3, (yTg - v.y) * -2.0 + 0.5, 1.0));
            vec3 tg = vec3(0.55, 0.18, 0.18) * (0.85 + 0.3 * vnoise(v * 30.0 + id));
            tg *= 1.0 - 0.5 * exp(-pow(v.x / 0.05, 2.0));                          // the groove down its middle
            col = tg * (0.2 + 0.6 * max(dot(tn, L), 0.0)) * 0.3;
            col += 0.08 * pow(max(dot(tn, normalize(L + vec3(0.0, 0.0, 1.0))), 0.0), 40.0);
        }
        float gum;
        vec3 up = shadeTeeth(v, yTu, -1.0, id, gum);
        if (up.x >= 0.0 && ax < 0.95){
            float along = v.y - yTu;
            col = along < gum ? up : vec3(0.58, 0.20, 0.22) * (0.25 + 0.75 * max(dot(normalize(vec3(v.x * 0.5, 0.3, 1.0)), L), 0.0));   // gum
        }
        vec3 lo = shadeTeeth(v, yTl, 1.0, id, gum);
        if (lo.x >= 0.0 && ax < 0.9){
            float along = yTl - v.y;
            col = along < gum * 0.9 ? lo * 0.85 : vec3(0.55, 0.18, 0.20) * 0.5;
        }
        col *= 1.0 - 0.7 * exp(-(l.ui - v.y) / 0.05);                           // the upper lip's shadow
        col *= 1.0 - 0.4 * exp(-(v.y - l.li) / 0.04);
        col *= smoothstep(1.0, 0.6, ax / M.w) * 0.8 + 0.2;                       // deep into the corners
        // strands of saliva, stretched between the lips; they thin and snap as the jaw opens
        for (int s = 0; s < 3; s++){
            float hs = hash1(vec2(float(s) + 0.5, id * 13.0));
            if (hs < 0.35) continue;
            float xs = (hash1(vec2(float(s), id * 5.0)) - 0.5) * 1.2 * M.w;
            float span = max(l.ui - l.li, 1e-3);
            float tt = clamp((v.y - l.li) / span, 0.0, 1.0);
            float cx = xs + 0.04 * sin(tt * PI) * (hs - 0.5) * 4.0;
            float wdt = (0.004 + 0.012 * pow(abs(tt - 0.5) * 2.0, 2.0)) * smoothstep(0.6, 0.25, J) * smoothstep(0.03, 0.08, J);
            if (wdt < 1e-3) continue;
            float m = smoothstep(wdt, wdt * 0.3, abs(v.x - cx));
            col = mix(col, vec3(0.55, 0.45, 0.45), m * 0.6);
            col += m * 0.4 * smoothstep(0.3, 0.0, abs(v.x - cx - wdt * 0.3) / max(wdt, 1e-4));
        }
    } else if (upperLip){
        col = shadeLip(v, (v.y - l.ui) / max(l.uo - l.ui, 1e-4), true, id, 1.0);
        col *= 0.55 + 0.45 * smoothstep(1.0, 0.8, ax / M.w);
    } else if (lowerLip){
        col = shadeLip(v, (l.li - v.y) / max(l.li - l.lo, 1e-4), false, id, 1.0);
        col *= 0.55 + 0.45 * smoothstep(1.0, 0.8, ax / M.w);
    } else {
        float eps = 0.02;
        float h0 = skinH(v, eL, M, id);
        float hx = skinH(v + vec2(eps, 0.0), eL + gE.x * eps, M, id);
        float hy = skinH(v + vec2(0.0, eps), eL + gE.y * eps, M, id);
        vec3 n = normalize(vec3(-(hx - h0) / eps, -(hy - h0) / eps, 1.0));
        float dO = max(max(v.y - l.uo, l.lo - v.y), ax - M.w);
        vec3 alb = skinBase(v, id);
        alb = mix(alb, vec3(0.52, 0.22, 0.18), 0.35 * exp(-max(dO, 0.0) / 0.12));   // flushed around the lips
        alb = seamFlesh(mottle(alb, v, id), eL);
        float ao = seamAO(eL) * (1.0 - 0.25 * exp(-max(dO, 0.0) / 0.03));
        col = skinLight(n, alb, ao, v, id, 0.0);
    }
    f_color = vec4(finish(col), 1.0);
}
"""


class MouthsSystem:
    def __init__(self, width: int, height: int, rows: float = 5.0, seed: int = 0, supersample: int = 2):
        self.W, self.H = int(width), int(height)
        self._wall = FleshWall(_FS, width, height, rows=rows, seed=seed, supersample=supersample)

    def render(self, time: float, jaw: float = 0.0, spread: float = 0.0, clench: float = 0.0,
               tremble: float = 0.0, tint: float = 0.0, light: float = 1.0) -> np.ndarray:
        """One frame (uint8, H x W x 3). ``time`` drives the idle breathing and the quiver."""
        return self._wall.draw(
            light=light, u_time=float(time), u_jaw=float(np.clip(jaw, 0.0, 1.0)),
            u_spread=float(np.clip(spread, -1.0, 1.0)), u_clench=float(np.clip(clench, 0.0, 1.0)),
            u_tremble=float(np.clip(tremble, 0.0, 1.0)), u_tint=float(np.clip(tint, 0.0, 1.0)))

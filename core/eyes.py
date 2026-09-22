"""Eyes — a Voronoi wall of eyes. GPU fragment shader (moderngl), one pass.

Random seeds tile the frame into Voronoi cells (about ``rows`` cells down the
height, more across); every cell is a patch of skin with one eye in it, and
the cells meet in deep raw seams. Each eye is built the way a real one is seen:

    lids        an almond opening, fuller towards the nose, a fold above the upper
                lid, wrinkles at the corners, lashes fanning out of the margins
    eyeball     a sphere behind the lids: off-white sclera, pink and veined towards
                the corners, shadowed under the upper lid, a tear film on the lower
    iris        a flat disc *behind* the cornea, seen through it by refraction (so it
                shifts and squashes as the eye turns): fibres, crypts, a collarette,
                a warmer ring by the pupil, a dark limbal ring, a light caustic
    cornea      wet: a softbox window reflected in it, sharp; softer on the sclera

Shading is done in linear light and mapped to sRGB through a filmic curve, and
the skin lets red light bleed into its shadows. All eyes look at one point in
front of the screen (each with its own small deviation, and no further than an
eye can turn), and the lids follow the gaze up and down:

    gaze        -> where the eyes look (screen units), glided in Python
    saccade     -> the gaze jumps and every eye rerolls its deviation (a hit)
    pupil       -> dilation 0..1 (loudness, a kick)
    blink       -> 0..1 closes every lid at once (a hit); eyes also blink on their own
    hue         -> the iris colour (the chord)
    light       -> overall brightness

Everything an eye owns (seed position, size, tilt, lid shape, iris, veins, blink
timing) comes from a hash of its cell, so the same seed gives the same wall.
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
uniform float u_aspect, u_rows, u_seed, u_time, u_saccade, u_pupil, u_blink, u_hue, u_light, u_depth;
uniform vec2 u_gaze;
#define PI 3.14159265
const float R = 0.92;          // eyeball radius, in units of the eye's half-width
const float LIMB = 0.52;       // angular radius of the iris on the ball (12 mm iris, 24 mm eye)
const vec3 L = vec3(-0.466, 0.621, 0.630);   // key light (normalized), upper left

float hash1(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
vec2  hash2(vec2 p){ float h = hash1(p); return vec2(h, hash1(p + h + 7.13)); }
float vnoise(vec2 p){
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash1(i), hash1(i + vec2(1, 0)), f.x), mix(hash1(i + vec2(0, 1)), hash1(i + vec2(1, 1)), f.x), f.y);
}
float fbm(vec2 p){ float a = 0.5, s = 0.0; for (int i = 0; i < 4; i++){ s += a * vnoise(p); p = p * 2.03 + 17.1; a *= 0.5; } return s; }
// thin ridges where a warped noise crosses its middle: veins
float veins(vec2 p, float k){ float n = fbm(p + 2.0 * fbm(p * 0.7 + 3.3)); return pow(1.0 - abs(2.0 * n - 1.0), k); }
vec3 hsv(float h, float s, float v){ vec3 k = fract(h + vec3(0, 2.0/3.0, 1.0/3.0)) * 6.0; return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s); }
float smax(float a, float b, float k){ float h = clamp(0.5 + 0.5 * (a - b) / k, 0.0, 1.0); return mix(b, a, h) + k * h * (1.0 - h); }

vec2 seedOf(vec2 cell){ return cell + 0.5 + (hash2(cell + u_seed) - 0.5) * 0.50; }

// the lid lines in the eye's frame (x in -1..1 corner to corner). P = (upper height, lower
// depth, which side the nose is on, the corners' height): an almond, fuller towards the nose
float lidUp(float x, vec4 P){ return P.w + P.x * pow(max(0.0, 1.0 - x * x), 0.7) * (1.0 + 0.22 * P.z * x); }
float lidLo(float x, vec4 P){ return P.w - P.y * pow(max(0.0, 1.0 - x * x), 0.9) * (1.0 - 0.18 * P.z * x); }
// where the lids are now: a blink brings the upper lid down onto a slightly raised lower lid
vec2 lidsAt(float x, vec4 P, float lid){
    float u = lidUp(x, P), l = lidLo(x, P);
    l = mix(l, u, 0.15 * lid);
    return vec2(mix(u, l, lid), l);
}

// height of the skin: lids stretched over the ball, the hollow of the socket rising to brow and
// cheek, the fold above the upper lid, fine wrinkles, and the seam where two cells meet
float skinH(vec2 v, float eL, vec4 P, float lid, float id){
    float rb = length(v);
    float ball = 0.95 * pow(max(0.0, 1.0 - rb * rb / 1.9), 1.5);                  // the lids over the ball, easing out
    float ring = length(v * vec2(0.72, 1.0));
    float h = smax(ball, 0.18 + 0.30 * smoothstep(0.9, 2.1, ring), 0.25);
    h += 0.12 * (fbm(v * 0.6 + id * 11.0) - 0.5);                                // the flesh is never flat
    vec2 lids = lidsAt(clamp(v.x, -1.0, 1.0), P, lid);
    float dF = max(max(v.y - lids.x, lids.y - v.y), abs(v.x) - 1.0);
    h -= 0.07 * exp(-max(dF, 0.0) / 0.05);                                        // lid margins roll inwards
    float ax = abs(v.x);
    float crease = lidUp(clamp(v.x * 0.92, -1.0, 1.0), P) + 0.30 + 0.12 * (1.0 - lid);
    h -= 0.06 * exp(-pow((v.y - crease) / 0.07, 2.0)) * smoothstep(1.35, 0.7, ax);
    h -= 0.035 * exp(-pow((v.y - lidLo(clamp(v.x * 0.9, -1.0, 1.0), P) + 0.30) / 0.08, 2.0)) * smoothstep(1.2, 0.5, ax);
    h -= 0.03 * fbm(vec2(v.x * 2.2, v.y * 11.0) + id * 5.0) * smoothstep(2.0, 0.9, ring);   // fine folds along the lids
    vec2 cv = v - vec2(sign(v.x) * 1.08, P.w);                                     // crow's feet at the corners
    h -= 0.025 * pow(vnoise(vec2(atan(cv.y, abs(cv.x)) * 7.0, length(cv) * 1.2) + id * 3.0), 3.0)
         * smoothstep(0.9, 0.25, length(cv)) * smoothstep(0.05, 0.2, length(cv));
    h += 0.010 * vnoise(v * 15.0 + id * 7.0) + 0.006 * vnoise(v * 31.0 + id * 2.0);   // pores
    float sh = 1.0 - pow(1.0 - clamp(eL / 0.8, 0.0, 1.0), 2.5);                 // a rounded shoulder into the seam
    return h * sh - 0.25 * (1.0 - smoothstep(0.0, 0.35, eL));
}

// a softbox (a window with a cross bar) reflected in the wet surface; sharp on the cornea
vec3 wetReflection(vec3 N, float sharp){
    vec3 Rv = reflect(vec3(0.0, 0.0, -1.0), N);
    vec3 acc = vec3(0.0);
    float d = dot(Rv, L);
    if (d > 0.0){
        vec3 b1 = normalize(cross(vec3(0.0, 0.0, 1.0), L)), b2 = cross(L, b1);
        vec2 uv = vec2(dot(Rv, b1), dot(Rv, b2)) / d;
        float soft = mix(0.22, 0.05, sharp);
        vec2 bq = max(abs(uv * vec2(1.35, 1.0)) - 0.10, 0.0);                     // a rounded window
        float box = smoothstep(0.07 + soft, 0.07 - soft * 0.5, length(bq));
        acc += box * mix(0.15, 2.2, sharp) * vec3(1.0, 0.98, 0.95);
    }
    vec3 F2 = normalize(vec3(0.55, -0.25, 1.0));                                  // a small fill light
    acc += vec3(0.9, 0.95, 1.0) * sharp * smoothstep(0.9975, 0.9993, dot(Rv, F2));
    // the room around: a bright ceiling, a dark floor, stronger at grazing angles (Fresnel)
    float fres = 0.04 + 0.96 * pow(1.0 - max(N.z, 0.0), 5.0);
    vec3 env = mix(vec3(0.04, 0.035, 0.03), vec3(0.55, 0.58, 0.62), smoothstep(-0.3, 0.7, Rv.y));
    acc += env * fres * (0.5 + 1.5 * sharp);
    return acc;
}

// the iris, laid out flat: s runs 0 at the pupil to 1 at the limbus, dir is the angle as a vector
// (sampling noise on a circle keeps it seamless, and a slow drift in s draws the radial fibres)
vec3 irisColor(vec2 dir, float s, float id, float het){
    vec3 outer = pow(hsv(u_hue, 0.60, 0.66), vec3(2.2));
    vec3 inner = pow(hsv(0.075 + 0.03 * id, 0.72, 0.55), vec3(2.2));
    float fib = vnoise(dir * 26.0 + vec2(s * 2.4, s * 1.3) + id * 13.0);
    float fib2 = vnoise(dir * 52.0 + vec2(s * 3.1, -s * 2.2) + id * 29.0);
    float crypt = smoothstep(0.60, 0.74, vnoise(dir * 9.0 + vec2(s * 4.2, s * 2.6) + id * 5.0))
                  * smoothstep(0.15, 0.35, s) * smoothstep(0.85, 0.6, s);
    float cz = 0.33 + 0.08 * (vnoise(dir * 6.0 + id * 3.0) - 0.5);                 // the collarette, a wavy ring
    float coll = exp(-pow((s - cz) / 0.05, 2.0));
    vec3 c = mix(outer, inner, smoothstep(cz + 0.14, cz - 0.08, s) * (0.25 + 0.7 * het));
    c *= 0.40 + 0.80 * fib + 0.40 * (fib2 - 0.5);
    c = mix(c, c * 0.22, crypt * 0.85);
    c += coll * 0.25 * mix(inner, outer, 0.5);
    c *= mix(1.0, 0.15, smoothstep(0.76, 1.0, s));                                // dark limbal ring
    c = mix(c, vec3(0.03, 0.015, 0.01), smoothstep(0.06, 0.0, s) * 0.8);          // the pupil's ruff
    return c;
}

// eyelashes: strands from the lid margin fanning outwards; they hang down once the lid is shut
float lashes(vec2 v, vec4 P, float lid, float id, bool upper){
    float N = upper ? 18.0 : 11.0;
    float acc = 0.0, k0 = floor(v.x * N);
    for (int k = -5; k <= 5; k++){
        float kk = k0 + float(k);
        float h = hash1(vec2(kk, id * 17.0 + (upper ? 0.0 : 9.0)));
        float x0 = (kk + 0.5 + 0.8 * (h - 0.5)) / N;
        if (abs(x0) > (upper ? 0.93 : 0.85)) continue;
        vec2 lids = lidsAt(x0, P, lid);
        float y0 = upper ? lids.x : lids.y;
        float len = (upper ? 0.22 : 0.09) * (0.6 + 0.5 * hash1(vec2(kk, 3.1 + id))) * pow(1.0 - x0 * x0, 0.35);
        float dy = mix(1.0, -0.75, smoothstep(0.2, 0.9, lid));                     // swinging down as the lid shuts
        float dirY = upper ? (dy >= 0.0 ? max(dy, 0.3) : min(dy, -0.3)) : -1.0;
        float s = (v.y - y0) / (len * dirY);
        if (s < -0.03 || s > 1.0) continue;
        float fan = x0 * 0.9 + (h - 0.5) * 0.4;
        float cx = x0 + len * (fan * s + 0.6 * fan * s * s);
        float w = (upper ? 0.018 : 0.012) * (1.0 - 0.85 * s);
        acc = max(acc, smoothstep(w, w * 0.3, abs(v.x - cx)) * (1.0 - 0.3 * s) * (upper ? 1.0 : 0.6));
    }
    return acc;
}

void main(){
    vec2 p = vec2((v_uv.x - 0.5) * 2.0 * u_aspect, (v_uv.y - 0.5) * 2.0);   // frame is 2 units tall
    float cs = 2.0 / u_rows;
    vec2 q = p / cs, c = floor(q);
    float d1 = 1e9; vec2 best = vec2(0.0), bestCell = vec2(0.0);
    for (int j = -1; j <= 1; j++) for (int i = -1; i <= 1; i++){
        vec2 cc = c + vec2(i, j); vec2 s = seedOf(cc); float d = distance(q, s);
        if (d < d1){ d1 = d; best = s; bestCell = cc; }
    }
    // distance to the cell's edge (the nearest bisector, and its gradient) and to the nearest seed
    // (a smooth minimum, so the seams round off at the corners instead of mitring)
    const float KS = 0.06;
    float ws = 0.0, dn = 1e9; vec2 ge = vec2(0.0);
    for (int j = -2; j <= 2; j++) for (int i = -2; i <= 2; i++){
        if (i == 0 && j == 0) continue;
        vec2 sj = seedOf(bestCell + vec2(i, j));
        vec2 nb = normalize(sj - best);
        float w = exp(-dot((best + sj) * 0.5 - q, nb) / KS);
        ws += w; ge -= w * nb;
        dn = min(dn, distance(best, sj));
    }
    ge /= ws;
    float e = max(0.0, -KS * log(ws) + KS * log(2.0));
    float id = hash1(bestCell * 3.7 + u_seed);
    vec4 hh = vec4(hash1(bestCell + 1.3), hash1(bestCell + 2.9), hash1(bestCell + 4.1), hash1(bestCell + 6.7));

    // the eye's own frame: centred on the seed, a little tilted, x = +-1 at the corners
    float a = 0.40 * dn * (0.88 + 0.24 * hh.x);
    float th = (hh.y - 0.5) * 0.22;
    mat2 rot = mat2(cos(th), sin(th), -sin(th), cos(th));
    vec2 v = rot * (q - best) / a;
    vec2 gE = rot * ge;
    float eL = e / a;

    // where it looks: the shared point plus its own deviation, as far as an eye can turn
    vec2 sp = best * cs;
    vec2 dev = (hash2(bestCell + 5.0 * floor(u_saccade) + u_seed) - 0.5) * 0.35;
    vec3 g = normalize(vec3((u_gaze + dev - sp) * 0.7, u_depth));
    if (g.z < cos(0.6)) g = vec3(normalize(g.xy) * sin(0.6), cos(0.6));
    g.xy = rot * g.xy;

    // lids follow the gaze up and down, as real ones do
    vec4 P = vec4(0.40 + 0.10 * hh.w + 0.28 * g.y, 0.30 + 0.06 * hh.x - 0.10 * g.y, hh.z < 0.5 ? -1.0 : 1.0, -0.04);
    float period = 3.5 + 4.0 * hash1(bestCell + 11.0);
    float ph = fract(u_time / period + id);
    float own = ph < 0.06 ? sin(PI * ph / 0.06) : 0.0;
    float lid = clamp(max(own, u_blink), 0.0, 1.0);
    vec2 lids = lidsAt(clamp(v.x, -1.0, 1.0), P, lid);
    float dU = lids.x - v.y, dL = v.y - lids.y;
    float rb = length(v);

    vec3 col;
    if (abs(v.x) < 1.0 && dU > 0.0 && dL > 0.0){
        if (rb < R){
            vec3 N = vec3(v, sqrt(R * R - rb * rb)) / R;
            float cosA = dot(N, g), ang = acos(clamp(cosA, -1.0, 1.0));
            // sclera: off-white, pinker and veined towards the corners, faintly yellow in patches
            float side = smoothstep(0.45, 1.0, abs(v.x));
            vec3 scl = vec3(0.64, 0.58, 0.52);
            scl = mix(scl, vec3(0.70, 0.60, 0.46), 0.30 * fbm(N.xy * 3.0 + id * 9.0));
            scl = mix(scl, vec3(0.74, 0.34, 0.30), 0.55 * side * side);
            float vv = veins(N.xy * 4.0 + id * 40.0, 30.0) + 0.7 * veins(N.xy * 9.0 + id * 77.0, 40.0);
            scl = mix(scl, vec3(0.50, 0.04, 0.035), clamp(vv * (0.12 + 0.88 * side) * smoothstep(0.55, 0.85, ang), 0.0, 1.0) * 0.8);
            float wrap = max(0.0, (dot(N, L) + 0.35) / 1.35);
            col = scl * (0.10 + 0.95 * wrap * wrap);
            col += scl * vec3(0.20, 0.06, 0.04) * (1.0 - wrap);                        // light scattered in the ball
            // the iris lies flat behind the cornea: follow the view ray refracted into the eye
            if (ang < LIMB + 0.03){
                vec3 Rr = refract(vec3(0.0, 0.0, -1.0), N, 1.0 / 1.376);
                float t = (0.83 - cosA) / min(dot(Rr, g), -1e-3);
                vec3 X = N + t * Rr;
                vec3 t1 = normalize(cross(g, vec3(0.0, 1.0, 0.0))), t2 = cross(g, t1);
                vec2 wp = vec2(dot(X, t1), dot(X, t2));
                float rho = length(wp) / sin(LIMB);
                vec2 dir = rho > 1e-4 ? normalize(wp) : vec2(1.0, 0.0);
                float pR = 0.22 + 0.42 * u_pupil;
                float s = clamp((rho - pR) / (1.0 - pR), 0.0, 1.0);
                vec3 ic = irisColor(dir, s, id, hh.w);
                vec2 lxy = normalize(vec2(dot(L, t1), dot(L, t2)) + 1e-5);
                float caustic = smoothstep(0.2, 1.0, dot(dir, -lxy)) * smoothstep(0.3, 0.95, s) * 0.7;
                ic *= 0.35 + 0.65 * max(dot(g, L), 0.0) + caustic;
                ic = mix(ic, vec3(0.004), smoothstep(pR + 0.015, pR - 0.015, rho));
                col = mix(col, ic, smoothstep(LIMB + 0.02, LIMB - 0.015, ang));
            }
            col *= 1.0 - 0.75 * exp(-dU / 0.09);                                 // the upper lid's shadow
            col *= 1.0 - 0.35 * exp(-dL / 0.05);
            col *= 0.55 + 0.45 * smoothstep(0.0, 0.25, 1.0 - abs(v.x));                // the ball turning away into the corners
            float onCornea = smoothstep(LIMB + 0.07, LIMB, ang);
            vec3 Nc = normalize(N + g * 0.3 * onCornea);                          // the cornea bulges
            col += wetReflection(Nc, onCornea) * (1.0 - 0.85 * exp(-dU / 0.05));
            col += vec3(0.15) * exp(-pow((dL - 0.02) / 0.012, 2.0));             // tear film on the lower lid
        } else {
            // the corner of the eye: caruncle, wet and pink, deeper in shadow
            float k = smoothstep(R, 1.0, rb);
            vec3 n2 = normalize(vec3(v * 0.6 + (vnoise(v * 20.0 + id) - 0.5) * 0.4, 0.6));
            col = mix(vec3(0.62, 0.22, 0.22), vec3(0.30, 0.05, 0.05), k) * (0.3 + 0.7 * max(dot(n2, L), 0.0));
            col += 0.4 * pow(max(dot(n2, normalize(L + vec3(0.0, 0.0, 1.0))), 0.0), 60.0);
            col *= 1.0 - 0.6 * exp(-dU / 0.08);
        }
    } else {
        float eps = 0.02;
        float h0 = skinH(v, eL, P, lid, id);
        float hx = skinH(v + vec2(eps, 0.0), eL + gE.x * eps, P, lid, id);
        float hy = skinH(v + vec2(0.0, eps), eL + gE.y * eps, P, lid, id);
        vec3 n = normalize(vec3(-(hx - h0) / eps, -(hy - h0) / eps, 1.0));
        float dF = max(max(v.y - lids.x, lids.y - v.y), abs(v.x) - 1.0);
        vec3 alb = mix(vec3(0.60, 0.34, 0.25), vec3(0.48, 0.23, 0.17), fbm(v * 1.7 + id * 13.0));
        alb = mix(alb, vec3(0.50, 0.17, 0.15), 0.45 * exp(-max(dF, 0.0) / 0.18));   // thin lid skin, blood beneath
        alb = mix(alb, vec3(0.40, 0.22, 0.24), 0.35 * smoothstep(0.0, -0.25, v.y - lids.y) * exp(-max(lids.y - v.y - 0.1, 0.0) / 0.3) * step(abs(v.x), 1.2));   // bruised under the eye
        alb *= 0.9 + 0.2 * vnoise(v * 6.0 + id * 21.0);                               // mottling
        alb = mix(alb, vec3(0.34, 0.08, 0.07), 0.7 * (1.0 - smoothstep(0.0, 0.35, eL)));   // raw flesh in the seams
        float ndl = dot(n, L), diff = max(ndl, 0.0), wrap = max((ndl + 0.45) / 1.45, 0.0);
        vec3 lit = alb * (diff + (wrap - diff) * vec3(0.60, 0.14, 0.09)) + alb * 0.07;   // light bleeding red through skin
        float ao = mix(0.12, 1.0, smoothstep(0.0, 0.45, eL)) * (1.0 - 0.55 * exp(-max(dF, 0.0) / 0.03));
        vec3 hv = normalize(L + vec3(0.0, 0.0, 1.0));
        float nh = max(dot(n, hv), 0.0);
        col = lit * ao;
        col += ao * (0.05 * pow(nh, 25.0) * (0.5 + vnoise(v * 12.0 + id)) + 0.3 * pow(nh, 90.0) * exp(-max(dF, 0.0) / 0.03));
    }
    col = mix(col, vec3(0.012, 0.009, 0.008), lashes(v, P, lid, id, true));
    col = mix(col, vec3(0.03, 0.02, 0.015), lashes(v, P, lid, id, false));

    col *= u_light * 1.15;
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);        // ACES filmic
    f_color = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
"""


_DOWN_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform sampler2D u_tex; uniform int u_ss;
void main(){
    ivec2 base = ivec2(gl_FragCoord.xy) * u_ss; vec3 acc = vec3(0.0);
    for (int j = 0; j < u_ss; j++) for (int i = 0; i < u_ss; i++) acc += texelFetch(u_tex, base + ivec2(i, j), 0).rgb;
    f_color = vec4(acc / float(u_ss * u_ss), 1.0);
}
"""


class EyesSystem:
    def __init__(self, width: int, height: int, rows: float = 5.0, seed: int = 0,
                 supersample: int = 2, depth: float = 0.9):
        import moderngl
        self.W, self.H = int(width), int(height)
        self.ss = max(1, int(supersample))
        self.rows, self.seed, self.depth = float(rows), float(seed), float(depth)
        self.ctx = moderngl.create_standalone_context()
        self.prog = self.ctx.program(vertex_shader=_VS, fragment_shader=_FS)
        quad = np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4")
        self._vbo = self.ctx.buffer(quad.tobytes())
        self._vao = self.ctx.vertex_array(self.prog, [(self._vbo, "2f", "in_pos")])
        # supersample into a texture, average it down on the GPU (a numpy mean-pool of the
        # big image costs ~120 ms at 720p; this costs ~1 ms)
        self._tex = self.ctx.texture((self.W * self.ss, self.H * self.ss), 3)
        self._fbo = self.ctx.framebuffer([self._tex])
        self._down = self.ctx.program(vertex_shader=_VS, fragment_shader=_DOWN_FS)
        self._down_vao = self.ctx.vertex_array(self._down, [(self._vbo, "2f", "in_pos")])
        self._out = self.ctx.simple_framebuffer((self.W, self.H), 3)
        self._mgl = moderngl

    def render(self, time: float, gaze=(0.0, 0.0), saccade: int = 0, pupil: float = 0.3,
               blink: float = 0.0, hue: float = 0.55, light: float = 1.0) -> np.ndarray:
        """One frame (uint8, H x W x 3). ``time`` drives the eyes' own blinks."""
        u = self.prog
        u["u_aspect"].value = self.W / self.H
        u["u_rows"].value = self.rows
        u["u_seed"].value = self.seed
        u["u_time"].value = float(time)
        u["u_gaze"].value = (float(gaze[0]), float(gaze[1]))
        u["u_depth"].value = self.depth
        u["u_saccade"].value = float(int(saccade))
        u["u_pupil"].value = float(np.clip(pupil, 0.0, 1.0))
        u["u_blink"].value = float(np.clip(blink, 0.0, 1.0))
        u["u_hue"].value = float(hue) % 1.0
        u["u_light"].value = float(light)
        self._fbo.use()
        self._fbo.clear(0.0, 0.0, 0.0)
        self._vao.render(self._mgl.TRIANGLE_STRIP)
        self._out.use()
        self._tex.use(0)
        self._down["u_tex"].value = 0
        self._down["u_ss"].value = self.ss
        self._down_vao.render(self._mgl.TRIANGLE_STRIP)
        img = np.frombuffer(self._out.read(components=3), dtype=np.uint8).reshape(self.H, self.W, 3)[::-1]
        return img.copy()

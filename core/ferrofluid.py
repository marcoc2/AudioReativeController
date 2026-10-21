"""Ferrofluid — a pool of black magnetic liquid raising spikes. GPU fragment shader (moderngl).

Seen at an angle, as in a studio photograph: a pool of ferrofluid under up to
four magnets. Each magnet's field is a soft gaussian on the pool; where the
field is strong the liquid first heaps into a mound, then breaks into the
Rosensweig instability, a hexagonal lattice of sharp spikes with concave
flanks, taller where the field is stronger. The spikes sway a little, each on
its own phase. The surface is black and mirror-like: it shows almost nothing
but the reflections of the studio (a large softbox ahead, a key light behind
the camera, strip lights behind at the sides, a small top light), with Fresnel, dark valleys between the spikes and
fog into the dark.

    magnets     up to 4 of (x, z, strength): where each magnet sits on the pool and
                how strongly it pulls now (the layer drives this from the music)
    rotation    the angle of the spike lattice (radians)
    hue, sat    tint of the side strip lights (sat 0 = plain white studio)
    light       overall brightness

The pool is the plane y = 0; x runs right, z away from the camera. The camera
looks at (0, 0, 0.35), so a magnet there sits in the middle of the frame.
"""
from __future__ import annotations

import numpy as np

from core.shader_pass import ShaderPass

MAX_MAGNETS = 4

_FS = """
#version 330
in vec2 v_uv; out vec4 f_color;
uniform float u_aspect, u_time, u_rot, u_light, u_hue, u_sat, u_spacing;
uniform vec3 u_mag[4];
#define PI 3.14159265
const float HMAX = 0.75;       // nothing on the pool rises above this

float hash1(vec2 p){ p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p){
    vec2 i = floor(p), f = fract(p); f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash1(i), hash1(i + vec2(1, 0)), f.x), mix(hash1(i + vec2(0, 1)), hash1(i + vec2(1, 1)), f.x), f.y);
}
vec3 hsv(float h, float s, float v){ vec3 k = fract(h + vec3(0, 2.0/3.0, 1.0/3.0)) * 6.0; return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s); }

// offset from the nearest point of a hexagonal lattice with unit spacing
vec2 hexOffset(vec2 p){
    const vec2 s = vec2(1.0, 1.7320508);
    vec2 a = mod(p, s) - s * 0.5;
    vec2 b = mod(p - s * 0.5, s) - s * 0.5;
    return dot(a, a) < dot(b, b) ? a : b;
}

float field(vec2 p){
    float f = 0.0;
    for (int i = 0; i < 4; i++){ vec2 d = p - u_mag[i].xy; f += u_mag[i].z * exp(-dot(d, d) / 0.25); }
    return f;
}

// the pool's height at p; amp: how tall the spikes are here (for the step size and the shading)
float heightAt(vec2 p, out float amp, out float mound){
    float F = field(p);
    mound = 0.20 * F / (1.0 + 0.6 * F);                                        // the liquid heaps over the magnet
    amp = min(0.34 * smoothstep(0.05, 1.5, F) + 0.05 * max(F - 1.5, 0.0), 0.45);   // a hedgehog: tallest in the middle
    float c = cos(u_rot), s = sin(u_rot);
    vec2 pr = mat2(c, s, -s, c) * p / u_spacing;
    vec2 o = hexOffset(pr);
    float j = hash1(floor((pr - o) * 4.0 + 0.5));
    o += 0.05 * vec2(sin(u_time * 2.3 + j * 6.28), cos(u_time * 1.9 + j * 4.0)) * smoothstep(0.0, 0.3, amp);   // sway
    float k = max(0.0, 1.0 - sqrt(dot(o, o) + 1e-4) / 0.5);
    float spike = amp * (0.85 + 0.3 * j) * pow(k, 2.2);                         // sharp tip, concave flanks
    float rip = 0.0015 * (vnoise(p * 2.5 + u_time * 0.1) - 0.5);                 // an oily, nearly still surface
    return mound + spike + rip;
}
float heightAt(vec2 p){ float a, m; return heightAt(p, a, m); }

// the studio, as the mirror-black liquid sees it
vec3 env(vec3 r){
    float el = asin(clamp(r.y, -1.0, 1.0)), az = atan(r.x, r.z);
    vec3 tint = mix(vec3(1.0), hsv(u_hue, 1.0, 1.0), u_sat);
    vec3 c = mix(vec3(0.004), vec3(0.03, 0.03, 0.034), smoothstep(-0.2, 0.7, r.y));
    float box = smoothstep(0.95, 0.65, abs(az)) * smoothstep(0.02, 0.12, el) * smoothstep(0.75, 0.6, el);
    c += (1.2 + 4.0 * smoothstep(0.1, 0.6, el)) * box;                                               // big softbox ahead, brighter up top
    c += 3.0 * smoothstep(0.55, 0.45, PI - abs(az)) * smoothstep(0.22, 0.15, abs(el - 0.35));         // key light, behind the camera
    c += 4.0 * tint * smoothstep(0.09, 0.05, abs(az + 2.3)) * smoothstep(0.6, 0.52, abs(el - 0.55));  // strip, back left
    c += 3.0 * tint * smoothstep(0.06, 0.03, abs(az - 2.1)) * smoothstep(0.55, 0.48, abs(el - 0.5));  // strip, back right
    c += 5.0 * smoothstep(0.20, 0.12, length(vec2(az * cos(el), el - 1.3)));                           // top light
    return c;
}

void main(){
    vec2 uv = (v_uv - 0.5) * vec2(u_aspect, 1.0) * 2.0;
    vec3 ro = vec3(0.0, 1.0, -1.9), ta = vec3(0.0, 0.05, 0.35);
    vec3 fw = normalize(ta - ro), rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw)), up = cross(fw, rt);
    vec3 rd = normalize(fw * 2.0 + rt * uv.x + up * uv.y);
    vec3 bg = vec3(0.004) + vec3(0.01) * smoothstep(-0.3, 0.3, rd.y);
    vec3 col = bg;
    if (rd.y < 0.0){
        float t = (ro.y - HMAX) / -rd.y, tprev = t;
        bool hit = false;
        for (int i = 0; i < 400; i++){
            vec3 p = ro + rd * t;
            float amp, mound;
            float dh = p.y - heightAt(p.xz, amp, mound);
            if (dh < 0.0){ hit = true; break; }
            tprev = t;
            float lip = 1.0 + amp * 2.2 / (0.5 * u_spacing);                    // how steep the spikes can get here
            t += max(dh / lip, 0.0008 * t);
            if (t > 14.0) break;
        }
        if (hit){
            float a = tprev, b = t;
            for (int i = 0; i < 8; i++){
                float m = 0.5 * (a + b);
                vec3 p = ro + rd * m;
                if (p.y < heightAt(p.xz)) b = m; else a = m;
            }
            t = b;
            vec3 p = ro + rd * t;
            float amp, mound;
            float h = heightAt(p.xz, amp, mound);
            float e = 0.0012 * (1.0 + 0.5 * t);
            vec3 n = normalize(vec3(heightAt(p.xz - vec2(e, 0.0)) - heightAt(p.xz + vec2(e, 0.0)), 2.0 * e,
                                    heightAt(p.xz - vec2(0.0, e)) - heightAt(p.xz + vec2(0.0, e))));
            float cosv = max(dot(n, -rd), 0.0);
            float F = 0.05 + 0.95 * pow(1.0 - cosv, 5.0);
            vec3 r = reflect(rd, n);
            float ao = amp > 1e-3 ? mix(0.25, 1.0, smoothstep(0.0, 0.8, (h - mound) / amp)) : 1.0;   // dark between spikes
            col = env(r) * F * mix(ao, 1.0, smoothstep(0.9, 1.2, asin(clamp(r.y, -1.0, 1.0))));
            col += vec3(0.004) * max(dot(n, normalize(vec3(-0.4, 0.8, -0.3))), 0.0);   // the liquid itself: nearly black
            col = mix(col, bg, 1.0 - exp(-0.02 * t * t));
        }
    }
    col *= 1.0 - 0.35 * dot(uv * 0.5, uv * 0.5);                                  // vignette
    col *= u_light;
    col = (col * (2.51 * col + 0.03)) / (col * (2.43 * col + 0.59) + 0.14);      // ACES filmic
    f_color = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / 2.2)), 1.0);
}
"""


class FerrofluidSystem:
    def __init__(self, width: int, height: int, supersample: int = 2, spacing: float = 0.11):
        self.W, self.H = int(width), int(height)
        self.spacing = float(spacing)
        self._pass = ShaderPass(_FS, width, height, supersample)

    def render(self, time: float, magnets=((0.0, 0.35, 1.0),), rotation: float = 0.0,
               hue: float = 0.0, sat: float = 0.0, light: float = 1.0) -> np.ndarray:
        """One frame (uint8, H x W x 3); ``magnets`` is up to 4 of (x, z, strength)."""
        mags = [(float(x), float(z), max(0.0, float(s))) for x, z, s in list(magnets)[:MAX_MAGNETS]]
        mags += [(0.0, 0.0, 0.0)] * (MAX_MAGNETS - len(mags))
        return self._pass.draw(u_time=float(time), u_mag=mags, u_rot=float(rotation), u_spacing=self.spacing,
                               u_hue=float(hue) % 1.0, u_sat=float(np.clip(sat, 0.0, 1.0)), u_light=float(light))

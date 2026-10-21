// title: Diamond Dust — Sound (não roda no ARC)
// url: https://www.shadertoy.com/view/ffKXRw
// author: Tonny Espeset (2026)
// license: livre com crédito ("Feel free to use this code, but please keep this credit and link.")
// notes: a trilha dub do próprio shader (mainSound); o ARC usa a música do vídeo, esta aba fica só guardada

/*

Created by Tonny Espeset, 2026
https://www.shadertoy.com/view/ffKXRw

Diamond Dust

Feel free to use this code, but please keep this credit and link.

Check my other shaders at

https://www.shadertoy.com/user/Espeset


My dub rhythm, aimed at matching the visuals, and to put you in that deep
theta state. Headphones recommended for that effect. :)

There is no bassline. One ear gets 55 Hz, the other 63, and they never meet
in the air - your brain puts them together and hears a throb at the
difference, 8 times a second. This is a binaural beat. The tones themselves
are steady; the rhythm is something you generate, not something I play.

I slide the second tone to 59 and back, walking that throb from 8 down to 4.
4 Hz is where shamanic drumming sits, 8 Hz is where your normal mind is.

*/

#define TAU 6.283185307
#define SPB 0.625    // seconds per beat @ 96 BPM

// split-index hash: sin() loses all precision past ~1e5, so fold the index hi/lo first
float hs(float x)
{
    float hi = floor(x * 0.000244140625);
    float lo = x - hi * 4096.0;
    return fract(sin(lo * 12.9898 + hi * 78.233) * 43758.545);
}

// additive saw through an emulated resonant lowpass
float voice(float f, float t, float cut, float res)
{
    float v = 0.0;
    for (int h = 1; h <= 14; h++)
    {
        float fh = float(h) * f;
        float x = fh / cut;
        float g = (1.0 / float(h)) / (1.0 + x * x * x * x);
        g *= 1.0 + res * exp(-9.0 * (x - 1.0) * (x - 1.0));
        v += g * sin(TAU * fh * t);
    }
    return v;
}

// offbeat dub chord, stepping Am - F - C - G every two bars
vec2 stab(float t, float cutmul)
{
    if (t < 0.0) return vec2(0.0);
    float fr = fract(t / SPB + 0.5);
    float e = exp(-8.0 * fr) * smoothstep(0.0, 0.003, fr);
    float semi[4];
    semi[0]=0.; semi[1]=-4.; semi[2]=3.; semi[3]=-2.;
    float r = 110.0 * exp2(semi[int(mod(floor(t / 5.0), 4.0))] / 12.0);
    float cut = 420.0 * cutmul;
    float v = voice(r, t, cut, 1.2) + voice(r * 1.5, t, cut, 1.2) + voice(r * 2.5, t, cut, 1.0);
    return vec2(v * e * 0.13);
}

vec2 mainSound(int samp, float time)
{
    float t = mod(time, 180.0);
    float slow = 0.5 + 0.5 * sin(TAU * t / 45.0);
    float cutmul = 0.6 + 0.85 * slow;

    // right carrier is 61 + 2*cos(TAU*t/180), so the beat walks 8 -> 4 -> 8; phase must be its integral, not rate*t
    float phL = 55.0 * t;
    float phR = 61.0 * t + (360.0 / TAU) * sin(TAU * t / 180.0);
    float br = 0.78 + 0.22 * sin(TAU * t / 30.0);
    vec2 s = vec2(sin(TAU * phL), sin(TAU * phR)) * 0.42 * br;

    // gated on the beat phase, so the sub pulses along the 8 -> 4 -> 8 Hz walk
    float phB = phR - phL;
    float cB = 0.5 + 0.5 * cos(TAU * phB);
    float gB = 0.74 + 0.26 * cB * cB;
    float br2 = 0.90 + 0.10 * sin(TAU * t / 30.0);
    float sub = sin(TAU * 27.5 * t) + sin(TAU * 27.55 * t) * 0.28;
    sub += sin(TAU * 55.0 * t) * (0.16 + 0.10 * sin(TAU * t / 45.0));
    s += sub * 0.26 * br2 * gB;

    // dotted eighth dub delay, three taps, ping-ponged
    float d = SPB * 0.75;
    s += stab(t, cutmul) * vec2(0.9, 0.35);
    s += stab(mod(t - d, 180.0), cutmul) * vec2(0.30, 0.55);
    s += stab(mod(t - d * 2.0, 180.0), cutmul) * vec2(0.33, 0.14);
    s += stab(mod(t - d * 3.0, 180.0), cutmul) * vec2(0.10, 0.22);

    // soft kick, felt more than heard: pitch drops 110 Hz -> 45 Hz
    float kp = fract(t / SPB) * SPB;
    float kph = 45.0 * kp + (65.0 / 28.0) * (1.0 - exp(-28.0 * kp));
    s += sin(TAU * kph) * exp(-6.5 * kp) * 0.34;

    // tape hiss floor
    float n = mod(float(samp), 7938000.0);
    s += (hs(floor(n / 4.0)) - 0.5) * 0.012;

    return tanh(s * 0.88);
}

// title: Híbrido da levada (fórmula por semicolcheia)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// notes: um fractal 3D cuja fórmula muda a cada iteração conforme a bateria do compasso
//        (as sequências híbridas do Mandelbulber/Mandelbulb3D como ideia; escrito do zero):
//        a iteração k olha a semicolcheia k do compasso:
//          bumbo  -> dobra de caixa e esfera (Mandelbox)
//          caixa  -> dobra octaédrica (KIFS)
//          prato  -> dobra tetraédrica com um giro
//          nada   -> dobra de Menger
//        Cada levada vira uma "espécie"; uma virada muda a forma; um compasso derrete no outro.
//          u_kick_seq, u_snare_seq, u_hat_seq   groove: {track: ...}: as levadas do compasso
//          u_kick_prev, u_snare_prev, u_hat_prev  as do compasso anterior (ago: 1)
//          u_bar    bar: {step: 1, morph: ...}: durante o morph, de uma levada para a outra
//          u_kick   pulso 1 -> 0: a escala incha
//          u_hue    a cor (harmony:)
//          u_orbit  a câmera em volta

uniform float u_kick_seq = 4369.0, u_snare_seq = 4112.0, u_hat_seq = 21845.0;
uniform float u_kick_prev = 4369.0, u_snare_prev = 4112.0, u_hat_prev = 21845.0;
uniform float u_bar = 0.0;
uniform float u_kick = 0.0;
uniform float u_hue = 0.1;
uniform float u_orbit = 0.0;

const int ITER = 16;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

float gOps;           // what the last call did, for colour: a mix of the ops' counts

float de(vec3 p, int ks, int ss, int hs) {
    vec3 z = p;
    float dr = 1.0;
    float boost = 1.0 + 0.15 * u_kick;
    float tally = 0.0;
    float c = 0.7, s = 0.7;                           // the hats' small turn
    for (int i = 0; i < ITER; i++) {
        int bit = 1 << i;
        if ((ks & bit) != 0) {                        // Mandelbox
            z = clamp(z, -1.0, 1.0) * 2.0 - z;
            float r2 = dot(z, z);
            if (r2 < 0.25) { z *= 4.0; dr *= 4.0; }
            else if (r2 < 1.0) { z /= r2; dr /= r2; }
            z = z * 2.0 * boost + p * 0.5;
            dr = dr * 2.0 * boost + 1.0;
            tally += 1.0;
        } else if ((ss & bit) != 0) {                 // octahedral fold
            z = abs(z);
            if (z.x < z.y) z.xy = z.yx;
            if (z.x < z.z) z.xz = z.zx;
            if (z.y < z.z) z.yz = z.zy;
            z = z * 2.0 * boost - vec3(1.0) * (2.0 * boost - 1.0);
            dr *= 2.0 * boost;
            tally += 3.0;
        } else if ((hs & bit) != 0) {                 // tetrahedral fold and a turn
            if (z.x + z.y < 0.0) z.xy = -z.yx;
            if (z.x + z.z < 0.0) z.xz = -z.zx;
            if (z.y + z.z < 0.0) z.zy = -z.yz;
            z.xz = mat2(c, s, -s, c) * z.xz;
            z = z * 2.0 * boost - vec3(1.0) * (2.0 * boost - 1.0);
            dr *= 2.0 * boost;
            tally += 9.0;
        } else {                                      // Menger
            z = abs(z);
            if (z.x < z.y) z.xy = z.yx;
            if (z.x < z.z) z.xz = z.zx;
            if (z.y < z.z) z.yz = z.zy;
            z = z * 3.0 - vec3(2.0, 2.0, 0.0);
            if (z.z < -1.0) z.z += 2.0;
            dr *= 3.0;
        }
        if (dot(z, z) > 1e4) break;
    }
    gOps = tally;
    return (length(z) - 1.5) / abs(dr);
}

float map(vec3 p) {
    float now = de(p, int(u_kick_seq), int(u_snare_seq), int(u_hat_seq));
    float m = fract(u_bar);
    if (m > 0.0 && m < 1.0) {
        float ops = gOps;
        float before = de(p, int(u_kick_prev), int(u_snare_prev), int(u_hat_prev));
        gOps = mix(gOps, ops, m);
        return mix(before, now, m);
    }
    return now;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float a = 6.2831853 * u_orbit + 0.05 * iTime;
    vec3 ro = vec3(4.5 * sin(a), 1.6, 4.5 * cos(a));
    vec3 fw = normalize(-ro), rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw)), up = cross(fw, rt);
    vec3 rd = normalize(fw * 1.7 + uv.x * rt + uv.y * up);
    float t = 0.0;
    int steps = 0;
    bool hit = false;
    for (int i = 0; i < 140; i++) {
        float d = map(ro + rd * t);
        if (d < 0.0008 * t) { hit = true; break; }
        t += d;
        steps = i;
        if (t > 12.0) break;
    }
    vec3 bg = hsv(u_hue + 0.5, 0.4, 0.08) * (1.0 - 0.3 * length(uv));
    vec3 col = bg;
    if (hit) {
        vec3 p = ro + rd * t;
        vec2 e = vec2(0.001 * t, 0.0);
        vec3 n = normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy), map(p + e.yyx) - map(p - e.yyx)));
        map(p);
        vec3 base = hsv(u_hue + 0.02 * gOps, 0.65, 1.0);
        vec3 l = normalize(vec3(0.5, 0.9, -0.3));
        float dif = clamp(dot(n, l), 0.0, 1.0);
        float ao = 1.0 - float(steps) / 140.0;
        col = base * (0.18 + 0.85 * dif) * (0.35 + 0.65 * ao);
        col = mix(col, bg, 1.0 - exp(-0.01 * t * t));
    }
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}

// title: Bolhas apolonianas
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// notes: uma espuma de esferas dentro de esferas (o fractal apoloniano: dobrar o espaço numa
//        célula e invertê-lo numa esfera, de novo e de novo; escrito do zero), com a câmera
//        lá dentro, devagar. As cascas têm cor de película de sabão (muda com o ângulo).
//          u_s       a força da inversão (1.0..1.3): o kick incha (hits com bias)
//          u_note    a nota que soa (a altura MIDI, ou só a classe 0..11): acende as cascas
//                    cuja célula é dela
//          u_light   0..1: o quanto elas acendem (adsr da melodia)
//          u_hue     a cor da película, 0..1 (harmony:)
//          u_travel  a viagem (rate: a câmera anda)

uniform float u_s = 1.1;
uniform float u_note = 0.0;
uniform float u_light = 0.0;
uniform float u_hue = 0.6;
uniform float u_travel = 0.0;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

float gCell;

float de(vec3 p) {
    float scale = 1.0;
    float cell = 0.0;
    for (int i = 0; i < 8; i++) {
        vec3 c = floor(0.5 * p + 0.5);
        if (i == 2) cell = dot(c, vec3(7.0, 13.0, 29.0));
        p = -1.0 + 2.0 * fract(0.5 * p + 0.5);
        float r2 = dot(p, p);
        float k = u_s / r2;
        p *= k;
        scale *= k;
    }
    gCell = cell;
    return 0.25 * abs(p.y) / scale;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float tt = 0.3 * u_travel + 0.02 * iTime;
    vec3 ro = vec3(0.7 * cos(tt * 0.7), 0.8 + 0.4 * sin(tt * 0.5), 0.6 * sin(tt));
    vec3 ta = ro + vec3(cos(tt * 0.9 + 1.0), 0.3 * sin(tt * 0.6), sin(tt * 0.8 + 2.0));
    vec3 fw = normalize(ta - ro), rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw)), up = cross(fw, rt);
    vec3 rd = normalize(fw * 1.3 + uv.x * rt + uv.y * up);
    float t = 0.002;
    int steps = 0;
    for (int i = 0; i < 160; i++) {
        float d = de(ro + rd * t);
        if (d < 0.0004 * t || t > 6.0) break;
        t += d;
        steps = i;
    }
    vec3 col = hsv(u_hue + 0.5, 0.4, 0.12);
    if (t < 6.0) {
        vec3 p = ro + rd * t;
        de(p);
        float cell = gCell;
        vec2 e = vec2(0.0005 * t, 0.0);
        vec3 n = normalize(vec3(de(p + e.xyy) - de(p - e.xyy), de(p + e.yxy) - de(p - e.yxy), de(p + e.yyx) - de(p - e.yyx)));
        float fres = pow(1.0 - abs(dot(n, rd)), 2.0);
        vec3 film = hsv(u_hue + 0.6 * fres + 0.1 * sin(cell), 0.55, 1.0);
        float ao = 1.0 - float(steps) / 160.0;
        col = film * (0.45 + 0.65 * ao) * (0.65 + 0.9 * fres);
        // the shells whose cell belongs to the sounding note light up
        if (abs(mod(cell, 12.0) - mod(floor(u_note + 0.5), 12.0)) < 0.5) col += film * 1.5 * u_light;
        col = mix(col, hsv(u_hue + 0.5, 0.4, 0.12), 1.0 - exp(-0.06 * t * t));
    }
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}

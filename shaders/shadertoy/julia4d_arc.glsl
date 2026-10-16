// title: Julia quaterniônico
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// notes: um conjunto de Julia em quatro dimensões (z -> z² + c com quatérnios), visto numa
//        fatia 3D: uma criatura lisa e brilhante que se transforma sem parar. Distância
//        estimada pela derivada analítica (|z| log|z| / |dz|; escrito do zero).
//          u_root   a tônica, 0..11: leva c por uma volta perto da borda do Mandelbrot
//          u_mode   o modo do acorde (0 menor .. 1 maior): a quarta coordenada de c
//          u_slice  a fatia da 4ª dimensão (-0.5..0.5; ex.: a altura da melodia)
//          u_kick   pulso 1 -> 0: c dá um tranco (a superfície ferve)
//          u_hue    a cor (harmony:)
//          u_orbit  a câmera em volta, em voltas
//          u_dist   a distância da câmera ao centro (4.2; num quadro em pé, ~8 para caber inteiro)

uniform float u_root = 4.0;
uniform float u_mode = 0.0;
uniform float u_slice = 0.0;
uniform float u_kick = 0.0;
uniform float u_hue = 0.6;
uniform float u_orbit = 0.0;
uniform float u_dist = 4.2;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

vec4 qsq(vec4 a) { return vec4(a.x * a.x - dot(a.yzw, a.yzw), 2.0 * a.x * a.yzw); }

vec4 gC;
float gTrap;

float de(vec3 p) {
    vec4 z = vec4(p, u_slice);
    float dz2 = 1.0;
    float m2 = dot(z, z);
    float trap = 1e9;
    for (int i = 0; i < 11; i++) {
        dz2 *= 4.0 * m2;
        z = qsq(z) + gC;
        m2 = dot(z, z);
        trap = min(trap, length(z.xyz - vec3(0.2, 0.0, 0.0)));
        if (m2 > 256.0) break;
    }
    gTrap = trap;
    return 0.25 * log(m2) * sqrt(m2 / dz2);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    float a = 6.2831853 * u_root / 12.0;
    gC = vec4(0.40 * cos(a) - 0.2, 0.45 * sin(a), 0.25 * sin(2.0 * a), 0.35 * (u_mode - 0.5));
    gC.xy += 0.08 * u_kick * vec2(cos(7.0 * iTime), sin(5.0 * iTime));
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float ca = 6.2831853 * u_orbit + 0.05 * iTime;
    vec3 ro = vec3(u_dist * sin(ca), 0.9 * u_dist / 4.2, u_dist * cos(ca));
    vec3 fw = normalize(-ro), rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw)), up = cross(fw, rt);
    vec3 rd = normalize(fw * 1.9 + uv.x * rt + uv.y * up);
    float t = 0.0;
    int steps = 0;
    bool hit = false;
    for (int i = 0; i < 160; i++) {
        float d = de(ro + rd * t);
        if (d < 0.0006) { hit = true; break; }
        t += d;
        steps = i;
        if (t > u_dist + 4.8) break;
    }
    vec3 bg = hsv(u_hue + 0.5, 0.4, 0.07) * (1.0 - 0.4 * length(uv));
    vec3 col = bg;
    if (hit) {
        vec3 p = ro + rd * t;
        vec2 e = vec2(0.0008, 0.0);
        vec3 n = normalize(vec3(de(p + e.xyy) - de(p - e.xyy), de(p + e.yxy) - de(p - e.yxy), de(p + e.yyx) - de(p - e.yyx)));
        de(p);
        vec3 base = hsv(u_hue + 0.25 * gTrap, 0.6, 1.0);
        vec3 l = normalize(vec3(0.6, 0.8, -0.3));
        float dif = clamp(dot(n, l), 0.0, 1.0);
        float spec = pow(clamp(dot(reflect(rd, n), l), 0.0, 1.0), 32.0);
        float ao = 1.0 - float(steps) / 160.0;
        float fres = pow(1.0 - abs(dot(n, rd)), 3.0);
        col = base * (0.15 + 0.85 * dif) * (0.4 + 0.6 * ao) + spec * 0.7 + fres * hsv(u_hue + 0.5, 0.5, 0.6);
    }
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}

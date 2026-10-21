// title: Newton do acorde
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: none
// notes: o fractal de Newton de um polinômio cujas raízes são as notas do acorde, cada
//        uma no seu lugar da roda de 12 (Dó em cima, um semitom a cada 30°). Cada ponto
//        do plano escorrega pelo método de Newton até uma raiz e ganha a cor dela (o lugar
//        da nota no círculo das quintas); as bordas entre as bacias trançam. Numa troca
//        de acorde as raízes deslizam (ligue com glide) e as tranças se refazem.
//          u_t0..u_t3   as notas do acorde, 0..11 (chord: part: tone, tone: 0..3)
//          u_size       quantas notas usar (chord: part: size; 2..4)
//          u_relax      o passo de Newton (1 o clássico; abaixo amolece, acima retorce)
//          u_kick       pulso 1 -> 0: o passo dá um tranco (as tranças fervem)
//          u_spin       gira tudo (radianos)
//          u_zoom       o tamanho do plano na tela (1: as raízes perto da borda)
//          u_video      0..1: com channel0: frame, cada bacia mostra o clipe girado pelo
//                       ângulo da sua nota (um caleidoscópio do acorde)
//          u_nova       0..1: o modo Nova: a cada passo soma-se c (no ângulo de u_bass), e as
//                       bacias se enchem de filamentos e mini-cópias
//          u_bass       a nota do baixo (MIDI ou 0..11): onde c aponta

uniform float u_t0 = 4.0;
uniform float u_t1 = 7.0;
uniform float u_t2 = 11.0;
uniform float u_t3 = 2.0;
uniform float u_size = 3.0;
uniform float u_relax = 1.0;
uniform float u_kick = 0.0;
uniform float u_spin = 0.0;
uniform float u_zoom = 1.0;
uniform float u_video = 0.0;
uniform float u_nova = 0.0;
uniform float u_bass = 4.0;

const int ITER = 48;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}
vec3 note(float pc) { return hsv(mod(pc * 7.0, 12.0) / 12.0, 0.8, 1.0); }

vec2 cdiv(vec2 a, vec2 b) { return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }

float angleOf(float pc) { return 1.5707963 - 6.2831853 * pc / 12.0 + u_spin; }   // C on top, clockwise

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / (0.5 * min(iResolution.x, iResolution.y));
    vec2 z = uv * 1.25 / u_zoom;
    int n = int(clamp(floor(u_size + 0.5), 2.0, 4.0));
    float pcs[4] = float[4](u_t0, u_t1, u_t2, u_t3);
    vec2 roots[4];
    for (int k = 0; k < 4; k++) roots[k] = vec2(cos(angleOf(pcs[k])), sin(angleOf(pcs[k])));
    float a = u_relax + 0.6 * u_kick;
    int hit = -1;
    float it = float(ITER);
    for (int i = 0; i < ITER; i++) {
        // f'/f = sum of 1 / (z - root): one Newton step is z -= a / (f'/f)
        vec2 s = vec2(0.0);
        for (int k = 0; k < 4; k++) {
            if (k >= n) break;
            s += cdiv(vec2(1.0, 0.0), z - roots[k]);
        }
        vec2 before = z;
        z -= a * cdiv(vec2(1.0, 0.0), s);
        z += u_nova * 0.15 * vec2(cos(angleOf(u_bass)), sin(angleOf(u_bass)));
        if (u_nova > 0.0) {                               // Nova: it settles where the steps stop
            if (length(z - before) < 1e-4) { it = float(i); break; }
            continue;
        }
        for (int k = 0; k < 4; k++) {
            if (k >= n) break;
            if (length(z - roots[k]) < 1e-3) { hit = k; break; }
        }
        if (hit >= 0) { it = float(i); break; }
    }
    if (hit < 0) {                                    // never settled: the nearest root
        float best = 1e9;
        for (int k = 0; k < 4; k++) {
            if (k >= n) break;
            float d = length(z - roots[k]);
            if (d < best) { best = d; hit = k; }
        }
    }
    float shade = pow(1.0 - it / float(ITER), 1.6);
    vec3 col = note(pcs[hit]) * (0.15 + 0.85 * shade);
    if (u_video > 0.0) {
        float ang = angleOf(pcs[hit]) - 1.5707963;
        vec2 c = fragCoord / iResolution.xy - 0.5;
        c.x *= iResolution.x / iResolution.y;
        c = mat2(cos(ang), sin(ang), -sin(ang), cos(ang)) * c;
        c.x /= iResolution.x / iResolution.y;
        vec3 clip = texture(iChannel0, fract(c + 0.5)).rgb;
        col = mix(col, clip * mix(vec3(1.0), note(pcs[hit]) * 1.4, 0.35) * (0.35 + 0.65 * shade), u_video);
    }
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}

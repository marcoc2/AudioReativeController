// title: Nebulosa (kaliset volumétrico)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// notes: nuvens e estrelas de um fractal volumétrico: em cada passo do raio, o ponto passa
//        pela fórmula de Kali (p = |p| / |p|² - k) algumas vezes, e o quanto ele pula de um
//        passo para o outro vira luz. A câmera viaja pelo relógio do shader (speed e rush da
//        camada: o kick acelera a viagem).
//          u_form   o k da fórmula (0.45..0.6): a textura das nuvens (ex.: o modo do acorde)
//          u_hue    a cor, 0..1 (harmony:)
//          u_dark   0..1: a "matéria escura" que apaga as nuvens perto (ex.: pulso na caixa)
//          u_bright o brilho

uniform float u_form = 0.53;
uniform float u_hue = 0.6;
uniform float u_dark = 0.3;
uniform float u_bright = 1.0;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    vec3 dir = normalize(vec3(uv * 0.9, 1.0));
    float a1 = 0.3 + 0.05 * sin(0.07 * iTime), a2 = 0.6;
    mat2 r1 = mat2(cos(a1), sin(a1), -sin(a1), cos(a1));
    mat2 r2 = mat2(cos(a2), sin(a2), -sin(a2), cos(a2));
    dir.xz = r1 * dir.xz;
    dir.xy = r2 * dir.xy;
    vec3 from = vec3(1.0, 0.5, 0.5) + vec3(0.0, 0.0, 0.02 * iTime);
    from.xz = r1 * from.xz;
    from.xy = r2 * from.xy;
    float s = 0.1, fade = 1.0;
    vec3 v = vec3(0.0);
    for (int r = 0; r < 18; r++) {
        vec3 p = from + s * dir * 0.5;
        p = abs(vec3(0.85) - mod(p, vec3(1.7)));                // tile the space
        float pa = 0.0, a = 0.0;
        for (int i = 0; i < 16; i++) {
            p = abs(p) / dot(p, p) - u_form;                     // Kali's formula
            a += abs(length(p) - pa);
            pa = length(p);
        }
        float dm = max(0.0, 0.3 - a * a * 0.001);               // dark matter
        a *= a * a;
        if (r > 6) fade *= 1.0 - dm * u_dark * 3.0;
        v += fade;
        v += hsv(u_hue + 0.08 * s, 0.6, 1.0) * s * a * 0.0015 * fade;
        fade *= 0.73;
        s += 0.1;
    }
    v = mix(vec3(length(v)), v, 0.85);
    fragColor = vec4(clamp(v * 0.01 * u_bright, 0.0, 1.0), 1.0);
}

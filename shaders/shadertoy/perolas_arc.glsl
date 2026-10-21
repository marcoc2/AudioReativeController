// title: Pérolas de Indra (inversões)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// notes: correntes de círculos que se beijam, à moda de "Indra's Pearls" (Mumford, Series e
//        Wright): cada ponto é dobrado para uma célula e invertido num círculo, de novo e de
//        novo; o fractal que sobra (a gaxeta apoloniana) é feito de pérolas cada vez menores,
//        e cada pérola mostra o clipe, virado do avesso pela inversão.
//          u_s      a força da inversão (1.0..1.5): muda o desenho das correntes (ex.: a melodia)
//          u_kick   pulso 1 -> 0: as pérolas incham
//          u_depth  quantas dobras (3..12)
//          u_zoom   o tamanho do plano na tela
//          u_hue    a cor do brilho nas bordas das pérolas, 0..1 (harmony:)
//          u_drift  desliza o plano (ex.: rate: 0.1)

uniform float u_s = 1.2;
uniform float u_kick = 0.0;
uniform float u_depth = 8.0;
uniform float u_zoom = 1.0;
uniform float u_hue = 0.6;
uniform float u_drift = 0.0;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 z = (fragCoord - 0.5 * res) / (0.5 * min(res.x, res.y)) * 1.2 / u_zoom;
    z += vec2(0.37, 0.21) * u_drift;
    float scale = 1.0;
    float s = u_s + 0.12 * u_kick;
    int depth = int(clamp(u_depth, 3.0, 12.0));
    float trap = 1e9;
    int n = 0;
    for (int i = 0; i < 12; i++) {
        if (i >= depth) break;
        z = -1.0 + 2.0 * fract(0.5 * z + 0.5);                     // into a cell
        float r2 = dot(z, z);
        float k = s / max(r2, 1e-4);
        z *= k;                                                      // inverted in the circle
        scale *= k;
        trap = min(trap, abs(r2 - 1.0));
        n++;
    }
    // the distance to the pearls' rims, in screen units
    float d = 0.25 * abs(z.y) / scale;
    vec2 uv = fract(z * 0.25 + 0.5);
    vec3 col = texture(iChannel0, uv).rgb;
    float rim = exp(-d * 1600.0);
    col = mix(col, hsv(u_hue + 0.05 * float(n), 0.5, 1.0), clamp(rim, 0.0, 0.6));
    col *= 0.55 + 0.45 * exp(-trap * 3.0);
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}

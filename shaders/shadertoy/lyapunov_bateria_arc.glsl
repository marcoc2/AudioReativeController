// title: Lyapunov da bateria
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: none
// notes: o fractal de Lyapunov (Markus): a cada ponto (a, b) do plano, a sequência logística
//        x -> r x (1 - x) alterna r entre a e b seguindo uma palavra de As e Bs, e o
//        expoente de Lyapunov diz se ela se acalma (conchas e fitas lisas, claras) ou
//        entra em caos (vazio escuro). Aqui a palavra é a bateria do compasso: cada
//        semicolcheia com bumbo é um A, cada uma sem é um B. Cada levada desenha a sua
//        paisagem; uma virada a muda; um compasso de outro tamanho também (5/4 = 20 passos).
//          u_seq    a levada do compasso (groove: {track: kick}): bit k = passo k tem bumbo
//          u_prev   a do compasso anterior (groove: ..., ago: 1), para derreter de uma na outra
//          u_len    quantos passos o compasso tem (groove: ..., part: steps)
//          u_bar    bar: {step: 1, morph: 0.5}: durante o morph, a mistura da anterior para a atual
//          u_hue    a cor das conchas, 0..1 (harmony:)
//          u_pan    desliza a janela do plano (ex.: rate: 0.02)
//          u_kick   pulso 1 -> 0: as conchas acendem

uniform float u_seq = 4369.0;        // x...x...x...x... (four on the floor)
uniform float u_prev = 4369.0;
uniform float u_len = 16.0;
uniform float u_bar = 0.0;
uniform float u_hue = 0.12;
uniform float u_pan = 0.0;
uniform float u_kick = 0.0;

const int WARM = 40;
const int SUM = 160;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

float lyapunov(vec2 ab, int seq, int len) {
    float x = 0.5;
    float acc = 0.0;
    for (int i = 0; i < WARM + SUM; i++) {
        int k = i % len;
        float r = ((seq >> k) & 1) == 1 ? ab.x : ab.y;
        x = r * x * (1.0 - x);
        if (i >= WARM) acc += log(max(abs(r * (1.0 - 2.0 * x)), 1e-6));
    }
    return acc / float(SUM);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    float aspect = iResolution.x / iResolution.y;
    // the window on the plane: around (3.35, 3.35), where order and chaos braid, the long
    // side of the frame spanning 0.9; drifting with u_pan
    vec2 ab = vec2(3.35) + (uv - 0.5) * vec2(aspect, 1.0) * (0.9 / max(aspect, 1.0));
    ab += 0.25 * vec2(sin(u_pan), cos(0.7 * u_pan));
    int len = int(clamp(floor(u_len + 0.5), 1.0, 24.0));
    float now = lyapunov(ab, int(u_seq), len);
    float m = fract(u_bar);
    float lam = now;
    if (m > 0.0 && m < 1.0) lam = mix(lyapunov(ab, int(u_prev), len), now, m);
    vec3 col;
    if (lam < 0.0) {                                  // order: shells, light where it is calmest
        float v = 1.0 - exp(lam * 2.2);
        col = hsv(u_hue, 0.75 - 0.4 * v, 0.15 + 0.85 * v);
        col *= 1.0 + 0.6 * u_kick;
    } else {                                          // chaos: the dark between them
        col = hsv(u_hue + 0.5, 0.6, 0.12 * exp(-lam * 3.0));
    }
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}

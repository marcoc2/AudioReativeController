// title: Copiadora (MRCM) — Buffer A
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// iChannel1: buffer_a
// notes: a máquina copiadora de Peitgen, Jürgens e Saupe: quatro lentes, cada uma uma cópia
//        menor e girada do quadro anterior. Realimentado, o quadro converge para um atrator
//        feito do próprio clipe. Cada lente é de um instrumento: o peso dela (u_l0..u_l3,
//        0..1) diz o quanto a sua cópia aparece. u_turn gira as lentes, u_scale o tamanho.

uniform float u_l0 = 0.8;
uniform float u_l1 = 0.8;
uniform float u_l2 = 0.8;
uniform float u_l3 = 0.0;
uniform float u_turn = 0.0;
uniform float u_scale = 0.5;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 uv = fragCoord / res;
    vec3 col = texture(iChannel0, uv).rgb;
    if (iFrame < 1) { fragColor = vec4(col, 1.0); return; }
    // lens k: x -> c_k + s R(a_k) (x - 1/2), in a square space so the copies keep their shape
    vec2 asp = vec2(res.x / res.y, 1.0);
    vec2 p = (uv - 0.5) * asp;
    float w[4] = float[4](u_l0, u_l1, u_l2, u_l3);
    vec2 cs[4] = vec2[4](vec2(-0.25, -0.25), vec2(0.25, -0.25), vec2(0.0, 0.22), vec2(0.0, 0.0));
    float ang[4] = float[4](u_turn, -u_turn, 2.0 * u_turn, 3.1416 + u_turn);
    for (int k = 0; k < 4; k++) {
        if (w[k] <= 0.001) continue;
        float s = k == 3 ? u_scale * 0.6 : u_scale;
        vec2 d = (p - cs[k] * vec2(asp.x, 1.0)) / s;
        float a = -ang[k];
        d = mat2(cos(a), sin(a), -sin(a), cos(a)) * d;
        vec2 x = d / asp + 0.5;                                    // where in the last frame
        if (all(greaterThan(x, vec2(0.0))) && all(lessThan(x, vec2(1.0)))) {
            vec3 copy = texture(iChannel1, x).rgb;
            col = mix(col, copy, clamp(w[k], 0.0, 1.0));
        }
    }
    fragColor = vec4(col, 1.0);
}

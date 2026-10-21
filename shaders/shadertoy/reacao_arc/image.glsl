// title: Reação-difusão (Gray-Scott)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: buffer_d
// iChannel1: frame
// notes: a reação-difusão de Gray-Scott (Pearson; Karl Sims), escrita do zero, crescendo das
//        bordas do clipe: corais, vermes, labirintos ou manchas que se dividem, conforme
//        (u_f, u_k). O acorde pode saltar de um regime para outro (ex.: u_f e u_k pelo modo:
//        maior manchas, menor labirinto); o kick semeia de novo nas bordas.
//          u_f, u_k  alimentação e morte (0.029, 0.057 labirinto; 0.0367, 0.0649 manchas;
//                    0.0545, 0.062 coral)
//          u_kick    pulso 1 -> 0: semeia nas bordas do quadro
//          u_seed    o quanto as bordas semeiam o tempo todo
//          u_hue     a cor da química, 0..1 (harmony:)
//          u_amount  0..1: o quanto a química cobre o clipe

uniform float u_hue = 0.3;
uniform float u_amount = 0.8;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 px = 1.0 / iResolution.xy;
    float b = texture(iChannel0, uv).g;
    float bx = texture(iChannel0, uv + vec2(px.x, 0.0)).g - b, by = texture(iChannel0, uv + vec2(0.0, px.y)).g - b;
    float shade = clamp(0.6 + 6.0 * (bx - by), 0.0, 1.4);          // embossed
    vec3 clip = texture(iChannel1, uv).rgb;
    vec3 ink = hsv(u_hue + 0.15 * b, 0.65, 1.0) * shade;
    float m = smoothstep(0.08, 0.3, b) * u_amount;
    fragColor = vec4(mix(clip, ink * (0.5 + 0.8 * clip), m), 1.0);
}

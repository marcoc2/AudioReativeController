// title: Fluxo cromático (modulação cruzada)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// iChannel1: previous
// notes: a imagem empurra a si mesma pelas próprias cores: cada pixel anda na direção
//        (verde - vermelho, azul - verde), a ideia do modulateHue da Hydra (escrita do
//        zero), e o resultado se acumula sobre o quadro anterior, então as cores escorrem
//        como tinta. O clipe novo entra por baixo, sem parar.
//          u_flow   o quanto as cores empurram (ex.: o grave relativo)
//          u_keep   0..1: quanto do rastro anterior fica (0: só o clipe)
//          u_kick   pulso 1 -> 0: o rastro se apaga e o clipe volta inteiro
//          u_turn   gira a direção do fluxo (radianos)

uniform float u_flow = 1.0;
uniform float u_keep = 0.85;
uniform float u_kick = 0.0;
uniform float u_turn = 0.0;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 now = texture(iChannel0, uv).rgb;
    vec3 last = texture(iChannel1, uv).rgb;
    vec2 d = vec2(last.g - last.r, last.b - last.g);
    d = mat2(cos(u_turn), sin(u_turn), -sin(u_turn), cos(u_turn)) * d;
    vec2 px = 1.0 / iResolution.xy;
    vec3 moved = texture(iChannel1, uv - d * u_flow * 6.0 * px * min(iResolution.x, iResolution.y) / 480.0).rgb;
    float keep = clamp(u_keep * (1.0 - u_kick), 0.0, 0.98);
    if (iFrame < 2) keep = 0.0;
    fragColor = vec4(mix(now, moved, keep), 1.0);
}

// title: Bloom musical
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// iChannel1: buffer_b
// notes: o brilho das luzes do quadro (o blur1/2/3 do MilkDrop, escrito do zero): o que passa
//        do limiar se espalha e soma por cima, tingido. A música mexe no limiar e no ganho.
//          u_thresh  a luminância a partir da qual brilha (0..1; baixe na batida)
//          u_gain    o quanto o brilho soma (ex.: hits nos pratos)
//          u_hue     a cor da tinta, 0..1 (harmony:)
//          u_tint    0..1: o quanto o brilho toma a cor da tinta
//          u_size    o tamanho do halo

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = texture(iChannel0, uv).rgb;
    vec3 glow = texture(iChannel1, uv).rgb;
    float l = dot(glow, vec3(0.299, 0.587, 0.114));
    glow = mix(glow, hsv(u_hue, 0.7, 1.0) * l * 1.6, u_tint);
    col = 1.0 - (1.0 - col) * (1.0 - clamp(glow * u_gain, 0.0, 1.0));     // screen
    fragColor = vec4(col, 1.0);
}

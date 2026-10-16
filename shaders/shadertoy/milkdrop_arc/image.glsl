// title: Campo de warp (à moda do MilkDrop)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: buffer_a
// notes: a composição do MilkDrop: o rastro do buffer, com um eco espelhado e ampliado por
//        cima (u_echo: 0 sem eco; u_flip: 0 sem espelho, 1 espelha na horizontal, 2 na
//        vertical, 3 nos dois) e um leve escurecer no centro (u_dark).

uniform float u_echo = 0.25;
uniform float u_flip = 1.0;
uniform float u_dark = 0.2;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec3 col = texture(iChannel0, uv).rgb;
    vec2 e = (uv - 0.5) / 1.35 + 0.5;
    int f = int(mod(floor(u_flip + 0.5), 4.0));            // cut: count steps through the four
    if (f == 1 || f == 3) e.x = 1.0 - e.x;
    if (f == 2 || f == 3) e.y = 1.0 - e.y;
    col = mix(col, max(col, texture(iChannel0, e).rgb), u_echo);
    col *= 1.0 - u_dark * exp(-dot(uv - 0.5, uv - 0.5) * 18.0);
    fragColor = vec4(col, 1.0);
}

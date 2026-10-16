// title: Bloom musical — Buffer B
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: buffer_a
// notes: e na vertical

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(blur(iChannel0, fragCoord / iResolution.xy, vec2(0.0, 1.0)), 1.0);
}

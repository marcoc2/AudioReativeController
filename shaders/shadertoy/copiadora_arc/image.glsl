// title: Copiadora (MRCM)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: buffer_a
// notes: ver buffer_a.glsl. A imagem só mostra o atrator.

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(texture(iChannel0, fragCoord / iResolution.xy).rgb, 1.0);
}

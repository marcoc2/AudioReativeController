// title: Reação-difusão (Gray-Scott) — Buffer A
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: buffer_d
// iChannel1: frame
// notes: um passo da simulação; os quatro buffers em fila dão quatro passos por quadro

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    fragColor = vec4(rd_step(iChannel0, fragCoord, iChannel1), 0.0, 1.0);
}

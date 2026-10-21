// title: Julia das figuras (orbit trap)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// iChannel1: mask_distance
// notes: um conjunto de Julia cuja armadilha de órbita são as figuras do vídeo (a trilha
//        de máscaras do SAM2): cada ponto itera z -> z² + c e, na primeira vez que a órbita
//        cai dentro de uma figura, pega a cor do clipe ali. As figuras se repetem ao longo
//        dos filamentos do fractal, em todas as escalas (iq, "bitmap orbit traps").
//          u_c      o ângulo de c na volta de raio 0.7885 (0..12: a tônica, chord: root)
//          u_trap   o tamanho da armadilha (1: o quadro inteiro; menor: figuras maiores)
//          u_kick   pulso 1 -> 0: a armadilha incha (as cópias crescem)
//          u_depth  quantas iterações (4..64; ligue no compasso para a recursão crescer)
//          u_zoom   o tamanho do plano na tela
//          u_spin   gira o plano (radianos)
//        Onde a órbita foge sem tocar figura nenhuma: o clipe na direção da fuga, escurecido.

uniform float u_c = 4.0;
uniform float u_trap = 1.2;
uniform float u_kick = 0.0;
uniform float u_depth = 40.0;
uniform float u_zoom = 1.6;
uniform float u_spin = 0.0;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 p = (fragCoord - 0.5 * res) / (0.5 * min(res.x, res.y)) * 1.5 / u_zoom;
    p = mat2(cos(u_spin), sin(u_spin), -sin(u_spin), cos(u_spin)) * p;
    float ang = 6.2831853 * u_c / 12.0;
    vec2 c = 0.7885 * vec2(cos(ang), sin(ang));
    vec2 z = p;
    float trap = u_trap * (1.0 + 0.35 * u_kick);
    int depth = int(clamp(u_depth, 4.0, 64.0));
    vec3 col = vec3(0.0);
    bool caught = false;
    float n = 0.0;
    for (int i = 0; i < 64; i++) {
        if (i >= depth) break;
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        // the plane's point as a place on the frame (the frame's short side spans 2 * trap)
        vec2 uv = z / (2.0 * trap) * vec2(min(res.x, res.y) / res.x, min(res.x, res.y) / res.y) + 0.5;
        if (all(greaterThan(uv, vec2(0.0))) && all(lessThan(uv, vec2(1.0)))) {
            vec2 g = vec2(uv.x, uv.y);
            if (texture(iChannel1, g).r > 0.502) {      // inside a figure
                col = texture(iChannel0, g).rgb * (1.0 - 0.025 * float(i));
                caught = true;
                break;
            }
        }
        if (dot(z, z) > 64.0) break;
        n += 1.0;
    }
    if (!caught) {
        // escaped: the clip seen along the direction it fled, darker the faster it went
        float k = n / float(depth);
        vec2 dir = normalize(z) * 0.35 + 0.5;
        col = texture(iChannel0, dir).rgb * (0.12 + 0.45 * sqrt(k));
    }
    fragColor = vec4(col, 1.0);
}

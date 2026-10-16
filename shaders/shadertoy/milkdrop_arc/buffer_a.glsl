// title: Campo de warp (à moda do MilkDrop) — Buffer A
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: buffer_a
// iChannel1: frame
// notes: a realimentação do MilkDrop (Ryan Geiss), escrita do zero: o quadro anterior volta
//        puxado por um campo de movimento que muda de pixel para pixel — zoom que depende do
//        raio (u_zoomexp), giro, turbulência senoidal — em volta de um centro que pode seguir
//        a figura principal (figure: largest). O clipe entra por cima aos poucos (u_feed).
//          u_cx, u_cy  o centro do vórtice (0..1)
//          u_zoom      o zoom por quadro (1.0 parado; 1.02 entra)
//          u_zoomexp   o quanto o zoom cresce para fora (0: igual em tudo)
//          u_rot       o giro por quadro (radianos; o sinal é o sentido)
//          u_warp      a turbulência
//          u_decay     quanto do rastro fica (0.9..0.99)
//          u_feed      quanto do clipe entra por quadro (0..1)

uniform float u_cx = 0.5;
uniform float u_cy = 0.5;
uniform float u_zoom = 1.015;
uniform float u_zoomexp = 1.0;
uniform float u_rot = 0.01;
uniform float u_warp = 0.5;
uniform float u_decay = 0.96;
uniform float u_feed = 0.12;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 uv = fragCoord / res;
    vec3 now = texture(iChannel1, uv).rgb;
    if (iFrame < 1) { fragColor = vec4(now, 1.0); return; }
    vec2 asp = vec2(res.x / res.y, 1.0);
    vec2 c = vec2(u_cx, 1.0 - u_cy);                         // figure y runs down; GL's up
    vec2 d = (uv - c) * asp;
    float r = length(d) / length(asp * 0.5);
    float zoom = pow(max(u_zoom, 0.5), 1.0 + u_zoomexp * (r - 0.5));
    float a = -u_rot * (1.0 + 0.5 * r);
    d = mat2(cos(a), sin(a), -sin(a), cos(a)) * d / zoom;
    float t = iTime;
    d += u_warp * 0.006 * vec2(sin(d.y * 9.0 + t * 1.3) + sin(d.y * 17.0 - t * 0.7),
                               cos(d.x * 11.0 - t * 1.1) + cos(d.x * 7.0 + t * 0.9));
    vec2 from = c + d / asp;
    vec3 last = texture(iChannel0, from).rgb * u_decay;
    fragColor = vec4(mix(last, now, u_feed), 1.0);
}

// title: Droste (espiral de Escher)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// notes: o quadro contém a si mesmo numa espiral logarítmica, como na "Galeria de gravuras"
//        de Escher (de Smit & Lenstra): no plano do logaritmo, o anel entre o quadro e a sua
//        cópia menor se repete, inclinado, e cada volta da espiral desce um nível. O zoom
//        de um nível é periódico, então ligado no compasso ele faz um loop perfeito.
//          u_phase  quantos níveis já desceu (bar: {step: 1, morph: <um compasso>}: um nível
//                   por compasso, acelerando e freando)
//          u_ratio  o tamanho da cópia interna (0.15..0.6)
//          u_twist  1 a espiral de Escher, 0 anéis retos (zoom sem girar), 2 mais torcida
//          u_kick   pulso 1 -> 0: um tranco na torção
//          u_cx, u_cy  o centro da espiral, em fração da largura e da altura

uniform float u_phase = 0.0;
uniform float u_ratio = 0.3;
uniform float u_twist = 1.0;
uniform float u_kick = 0.0;
uniform float u_cx = 0.5;
uniform float u_cy = 0.5;

vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 cdiv(vec2 a, vec2 b) { return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    float unit = 0.5 * min(res.x, res.y);
    vec2 p = (fragCoord - vec2(u_cx, u_cy) * res) / unit;          // the copy's edge at radius ~1
    float L = log(clamp(u_ratio, 0.05, 0.9));                      // < 0: one level in log radius
    vec2 w = vec2(log(max(length(p), 1e-6)), atan(p.y, p.x));
    // tilt the log plane so one turn of the angle walks one level down: the spiral
    float k = (u_twist + 0.35 * u_kick) * (-L) / 6.2831853;
    vec2 beta = vec2(1.0, k);
    w = cmul(w, beta) / dot(beta, beta) * 1.0;
    w = cdiv(w, vec2(1.0, 0.0));
    // shift down by u_phase levels and fold into the ring between the copy and the frame
    w.x = mod(w.x + u_phase * (-L), -L) + L;
    vec2 q = exp(w.x) * vec2(cos(w.y), sin(w.y));                 // back to the frame, |q| in (ratio, 1)
    vec2 uv = q * unit / res + vec2(u_cx, u_cy);
    // past the frame's short side the corners mirror back in
    uv = abs(mod(uv - 1.0, 2.0) - 1.0);
    fragColor = vec4(texture(iChannel0, uv).rgb, 1.0);
}

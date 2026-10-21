// title: Ladrilhos hiperbólicos ({p,q}, Circle Limit)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// notes: o clipe ladrilhado ao infinito no disco de Poincaré, como nos "Circle Limit" de
//        Escher: polígonos de p lados, q em cada vértice, encolhendo para a borda do disco.
//        Cada ponto é dobrado para o polígono central (giros e inversões no círculo da
//        aresta), onde o clipe está. Uma translação de Möbius anda o plano: u_phase de 0
//        a 1 leva exatamente um ladrilho ao lugar do vizinho, então um loop por tempo.
//          u_p      lados de cada polígono (o numerador do compasso: 4 em 4/4, 5 em 5/4)
//          u_q      quantos se encontram em cada vértice (ex.: tamanho do acorde + 2)
//                   (precisa (p - 2)(q - 2) > 4; senão sobe q)
//          u_phase  o passeio: 0 -> 1 = um ladrilho (lfo: saw, period: 1/4: um por tempo)
//          u_kick   pulso 1 -> 0: os ladrilhos ímpares invertem as cores
//          u_spin   gira o disco (radianos)
//          u_edge   0..1: o brilho das arestas
//          u_fill   quanto do clipe cabe num ladrilho (1: o miolo; 2: quase tudo)
//        Fora do disco: o clipe, escurecido.

uniform float u_p = 4.0;
uniform float u_q = 5.0;
uniform float u_phase = 0.0;
uniform float u_kick = 0.0;
uniform float u_spin = 0.0;
uniform float u_edge = 0.5;
uniform float u_fill = 1.6;

const float PI = 3.14159265358979;

vec2 cmul(vec2 a, vec2 b) { return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x); }
vec2 cdiv(vec2 a, vec2 b) { return vec2(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y) / dot(b, b); }
vec2 conj(vec2 a) { return vec2(a.x, -a.y); }

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 z = (fragCoord - 0.5 * res) / (0.5 * min(res.x, res.y)) * 1.02;
    float rr = dot(z, z);
    float p = max(3.0, floor(u_p + 0.5));
    float q = max(3.0, floor(u_q + 0.5));
    if ((p - 2.0) * (q - 2.0) <= 4.0) q = floor(4.0 / (p - 2.0)) + 3.0;
    float a = PI / p, b = PI / q;
    // the edge's mirror: a circle square to the disc, through the edge's midpoint x0
    float ch = cos(b) / sin(a);                                   // cosh(centre to edge midpoint)
    float x0 = sqrt((ch - 1.0) / (ch + 1.0));
    float D = (1.0 + x0 * x0) / (2.0 * x0), R = D - x0;
    vec3 outside = texture(iChannel0, fragCoord / res).rgb * 0.3;
    if (rr >= 1.0) { fragColor = vec4(outside, 1.0); return; }
    z = cmul(z, vec2(cos(u_spin), sin(u_spin)));
    // walk the plane: a Moebius translation along x, one tile (2h) per unit of u_phase
    float h = log(ch + sqrt(ch * ch - 1.0));                      // centre to edge midpoint
    vec2 w = vec2(tanh(h * fract(u_phase)), 0.0);
    z = cdiv(z + w, vec2(1.0, 0.0) + cmul(conj(w), z));
    int flips = 0;
    for (int i = 0; i < 40; i++) {
        float th = atan(z.y, z.x);
        float k = floor(th / (2.0 * a) + 0.5);
        z = cmul(z, vec2(cos(-2.0 * a * k), sin(-2.0 * a * k)));  // into the sector around +x
        vec2 c = vec2(D, 0.0);
        vec2 d = z - c;
        if (dot(d, d) < R * R) {                                  // beyond the edge: invert back in
            z = c + d * (R * R / dot(d, d));
            flips++;
        } else break;
    }
    // the central polygon holds the frame; its corner at radius rv
    float chv = 1.0 / (tan(a) * tan(b));                          // cosh(centre to corner)
    float rv = sqrt((chv - 1.0) / (chv + 1.0));
    vec2 uv = z / (2.0 * rv) * u_fill * vec2(min(res.x, res.y) / res.x, min(res.x, res.y) / res.y) + 0.5;
    vec3 col = texture(iChannel0, uv).rgb;
    if ((flips & 1) == 1) col = mix(col, 1.0 - col, u_kick);
    float edge = abs(length(z - vec2(D, 0.0)) - R);
    col += u_edge * vec3(1.0, 0.95, 0.8) * exp(-edge * 140.0 * (1.0 - rr + 0.05));
    col = mix(outside, col, smoothstep(1.0, 0.97, rr));
    fragColor = vec4(col, 1.0);
}

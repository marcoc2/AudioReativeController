// title: Mármore (domain warp)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// notes: o clipe derretido como mármore: um ruído fractal (fbm) desloca outro, que desloca
//        um terceiro (a técnica de "domain warping" de Inigo Quilez, escrita do zero), e o
//        resultado empurra cada pixel do clipe. Os vetores do meio tingem o fluxo.
//          u_warp1  o primeiro estágio, o grande (ex.: o grave)
//          u_warp2  o segundo, o fino (ex.: a voz ou os agudos)
//          u_jump   pulso 1 -> 0: o ruído salta no tempo (o mármore se rearranja no kick)
//          u_tint   0..1: quanto os vetores do meio tingem a imagem
//          u_hue    a cor da tinta, 0..1 (harmony:)
//          u_scale  o tamanho dos veios

uniform float u_warp1 = 0.5;
uniform float u_warp2 = 0.3;
uniform float u_jump = 0.0;
uniform float u_tint = 0.3;
uniform float u_hue = 0.6;
uniform float u_scale = 2.0;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
float noise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), u.x), mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), u.x), u.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    mat2 r = mat2(0.8, 0.6, -0.6, 0.8);
    for (int i = 0; i < 5; i++) { v += a * noise(p); p = r * p * 2.02; a *= 0.5; }
    return v;
}
vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 p = uv * vec2(iResolution.x / iResolution.y, 1.0) * u_scale;
    float t = 0.15 * iTime + 3.0 * u_jump;
    vec2 q = vec2(fbm(p + vec2(0.0, t)), fbm(p + vec2(5.2, 1.3 - t)));
    vec2 r = vec2(fbm(p + 4.0 * q + vec2(1.7, 9.2) + 0.5 * t), fbm(p + 4.0 * q + vec2(8.3, 2.8) - 0.3 * t));
    vec2 push = (q - 0.5) * 0.25 * u_warp1 + (r - 0.5) * 0.12 * u_warp2;
    vec3 col = texture(iChannel0, uv + push).rgb;
    float f = fbm(p + 4.0 * r);
    vec3 ink = hsv(u_hue + 0.15 * (q.x - 0.5), 0.6, 0.6 + 0.6 * f);
    col = mix(col, col * ink * 1.4, u_tint * clamp(length(r - 0.5) * 2.0, 0.0, 1.0));
    fragColor = vec4(clamp(col, 0.0, 1.0), 1.0);
}

// title: Lente do kick (exemplo do ARC)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// notes: exemplo do adaptador: o clipe passa por uma lente que respira; u_kick (pulso no
//        kick) incha a lente e separa as cores; u_bass gira o redemoinho no centro

uniform float u_kick;
uniform float u_bass;

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 p = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float r = length(p);
    float lens = 1.0 - (0.25 + 0.35 * u_kick) * exp(-r * r * 3.0) * (0.8 + 0.2 * sin(iTime * 2.0));
    float a = (0.6 * u_bass + 0.1 * sin(iTime)) * exp(-r * r * 6.0);
    p = mat2(cos(a), -sin(a), sin(a), cos(a)) * p * lens;
    vec2 q = p * iResolution.y / iResolution.xy + 0.5;
    vec2 split = p * 0.02 * u_kick;
    vec3 col = vec3(texture(iChannel0, q + split).r, texture(iChannel0, q).g, texture(iChannel0, q - split).b);
    col *= 1.0 - 0.35 * r * r;
    fragColor = vec4(col, 1.0);
}

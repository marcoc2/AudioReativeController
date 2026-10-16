// title: Bloom musical — Buffer A
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// notes: só as luzes acima do limiar, desfocadas na horizontal

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = fragCoord / iResolution.xy;
    vec2 px = 1.0 / iResolution.xy * (iResolution.y / 480.0) * 2.5 * u_size;
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int i = -8; i <= 8; i++) {
        float w = exp(-float(i * i) / 18.0);
        vec3 c = texture(iChannel0, uv + vec2(px.x * float(i), 0.0)).rgb;
        float l = dot(c, vec3(0.299, 0.587, 0.114));
        acc += c * smoothstep(u_thresh, u_thresh + 0.15, l) * w;
        wsum += w;
    }
    fragColor = vec4(acc / wsum, 1.0);
}

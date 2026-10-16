uniform float u_thresh = 0.7;
uniform float u_gain = 1.0;
uniform float u_hue = 0.1;
uniform float u_tint = 0.5;
uniform float u_size = 1.0;

// one direction of a 17-tap Gaussian, spread with the frame's size
vec3 blur(sampler2D s, vec2 uv, vec2 dir) {
    vec2 px = dir / iResolution.xy * (iResolution.y / 480.0) * 2.5 * u_size;
    vec3 acc = vec3(0.0);
    float wsum = 0.0;
    for (int i = -8; i <= 8; i++) {
        float w = exp(-float(i * i) / 18.0);
        acc += texture(s, uv + px * float(i)).rgb * w;
        wsum += w;
    }
    return acc / wsum;
}

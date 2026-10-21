// Gray-Scott, one step (Karl Sims' weights; dt 1): the texture holds (A, B) in r, g
uniform float u_f = 0.029;
uniform float u_k = 0.057;
uniform float u_kick = 0.0;
uniform float u_seed = 0.5;

vec2 rd_step(sampler2D s, vec2 fragCoord, sampler2D frame) {
    vec2 res = iResolution.xy;
    vec2 uv = fragCoord / res;
    vec2 px = 1.0 / res;
    vec2 c = texture(s, uv).rg;
    if (iFrame < 2) {                                         // start: all A, B where the picture has edges
        vec3 f = texture(frame, uv).rgb;
        float e = length(texture(frame, uv + vec2(px.x, 0.0)).rgb - f) + length(texture(frame, uv + vec2(0.0, px.y)).rgb - f);
        return vec2(1.0, step(0.15, e) * 0.5);
    }
    vec2 lap = -c;
    lap += 0.2 * (texture(s, uv + vec2(px.x, 0.0)).rg + texture(s, uv - vec2(px.x, 0.0)).rg
                + texture(s, uv + vec2(0.0, px.y)).rg + texture(s, uv - vec2(0.0, px.y)).rg);
    lap += 0.05 * (texture(s, uv + px).rg + texture(s, uv - px).rg
                 + texture(s, uv + vec2(px.x, -px.y)).rg + texture(s, uv + vec2(-px.x, px.y)).rg);
    float a = c.r, b = c.g;
    float abb = a * b * b;
    a += 1.0 * lap.r - abb + u_f * (1.0 - a);
    b += 0.5 * lap.g + abb - (u_k + u_f) * b;
    // the picture's edges keep seeding B a little; a kick seeds a lot
    vec3 f = texture(frame, uv).rgb;
    float e = length(texture(frame, uv + vec2(px.x, 0.0)).rgb - f) + length(texture(frame, uv + vec2(0.0, px.y)).rgb - f);
    b += step(0.25, e) * (0.004 * u_seed + 0.05 * u_kick);
    return clamp(vec2(a, b), 0.0, 1.0);
}

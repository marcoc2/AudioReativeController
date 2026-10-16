// title: Fases (um planeta por clipe)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// notes: um fractal de dobras como o de Marble Marcher (abs, giro, dobra de Menger, giro,
//        escala, deslocamento, 16 vezes; escrito do zero), com seis "fases" de parâmetros
//        próprios: torres, pontes, treliças. u_level escolhe a fase e, fracionário, derrete
//        de uma para a seguinte (cut: count com morph: uma fase nova a cada corte). Dentro
//        da fase, os ângulos balançam presos ao compasso (u_sway).
//          u_level  a fase (0..5, e volta)
//          u_sway   o balanço, -1..1 (lfo: sine, period: 2/1, gain: 2, bias: -1)
//          u_kick   pulso 1 -> 0: um tranco no primeiro ângulo
//          u_hue    a cor (harmony:)
//          u_orbit  a câmera em volta, em voltas

uniform float u_level = 0.0;
uniform float u_sway = 0.0;
uniform float u_kick = 0.0;
uniform float u_hue = 0.1;
uniform float u_orbit = 0.0;

// scale, angle 1, angle 2, shift x, shift y, shift z
const float LEVELS[36] = float[36](
    1.80, -0.12,  0.50, -2.12, -2.75,  0.49,
    1.66,  1.52,  0.19, -3.83, -1.94, -1.09,
    1.72, -0.30,  0.85, -2.40, -1.60, -0.40,
    1.87, -3.12,  0.02, -3.57,  0.13, -0.20,
    1.81, -4.84, -2.99, -2.91,  0.03, -0.07,
    1.93,  1.35,  1.24, -2.31, -1.65, -2.98
);

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

float gScale, gA1, gA2; vec3 gShift; vec3 gOrbit;

void level(float x) {
    float l = mod(x, 6.0);
    int i = int(floor(l)), j = (i + 1) % 6;
    float f = fract(l);
    f = f * f * (3.0 - 2.0 * f);
    gScale = mix(LEVELS[i * 6], LEVELS[j * 6], f);
    gA1 = mix(LEVELS[i * 6 + 1], LEVELS[j * 6 + 1], f) + 0.08 * u_sway + 0.25 * u_kick;
    gA2 = mix(LEVELS[i * 6 + 2], LEVELS[j * 6 + 2], f) - 0.06 * u_sway;
    gShift = mix(vec3(LEVELS[i * 6 + 3], LEVELS[i * 6 + 4], LEVELS[i * 6 + 5]),
                 vec3(LEVELS[j * 6 + 3], LEVELS[j * 6 + 4], LEVELS[j * 6 + 5]), f);
}

void mengerFold(inout vec3 z) {
    float a = min(z.x - z.y, 0.0); z.x -= a; z.y += a;
    a = min(z.x - z.z, 0.0); z.x -= a; z.z += a;
    a = min(z.y - z.z, 0.0); z.y -= a; z.z += a;
}

float de(vec3 p) {
    vec4 z = vec4(p, 1.0);
    float c1 = cos(gA1), s1 = sin(gA1), c2 = cos(gA2), s2 = sin(gA2);
    vec3 orbit = vec3(0.0);
    for (int i = 0; i < 16; i++) {
        z.xyz = abs(z.xyz);
        z.xy = mat2(c1, s1, -s1, c1) * z.xy;
        mengerFold(z.xyz);
        z.yz = mat2(c2, s2, -s2, c2) * z.yz;
        z *= gScale;
        z.xyz += gShift;
        orbit = max(orbit, z.xyz * vec3(0.12, 0.09, 0.05));
    }
    gOrbit = orbit;
    vec3 q = abs(z.xyz) - vec3(6.0);
    return (length(max(q, 0.0)) + min(max(q.x, max(q.y, q.z)), 0.0)) / z.w;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    level(u_level);
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float a = 6.2831853 * u_orbit + 0.04 * iTime;
    vec3 ro = vec3(9.0 * sin(a), 3.5, 9.0 * cos(a));
    vec3 fw = normalize(vec3(0.0, -1.0, 0.0) - ro), rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw)), up = cross(fw, rt);
    vec3 rd = normalize(fw * 1.5 + uv.x * rt + uv.y * up);
    float t = 0.0;
    int steps = 0;
    for (int i = 0; i < 120; i++) {
        float d = de(ro + rd * t);
        if (d < 0.0008 * t || t > 30.0) break;
        t += d;
        steps = i;
    }
    vec3 sky = hsv(u_hue + 0.5, 0.35, 0.18) * (1.0 - 0.5 * uv.y);
    vec3 col = sky;
    if (t < 30.0) {
        vec3 p = ro + rd * t;
        de(p);
        vec3 base = clamp(gOrbit, 0.0, 1.0);
        base = mix(base, hsv(u_hue + dot(base, vec3(0.25)), 0.8, 1.0), 0.75);
        vec2 e = vec2(0.002 * t, 0.0);
        vec3 n = normalize(vec3(de(p + e.xyy) - de(p - e.xyy), de(p + e.yxy) - de(p - e.yxy), de(p + e.yyx) - de(p - e.yyx)));
        vec3 l = normalize(vec3(0.5, 0.9, -0.3));
        float dif = clamp(dot(n, l), 0.0, 1.0);
        float ao = 1.0 - float(steps) / 120.0;
        col = base * (0.2 + 0.85 * dif) * (0.4 + 0.6 * ao);
        col = mix(col, sky, 1.0 - exp(-0.003 * t * t));
    }
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}

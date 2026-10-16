// title: Figuras no cenário fractal
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// iChannel0: frame
// iChannel1: mask
// notes: as figuras do vídeo (a trilha de máscaras) viram um recorte de papel em pé dentro de
//        um espaço fractal (o pseudo-kleiniano de Knighty, escrito do zero): o fractal que
//        passa na frente do recorte esconde a figura, o que fica atrás aparece em volta dela,
//        e ele joga sombra suave sobre ela. Fora das figuras, o fundo é o próprio fractal.
//          u_depth  a que distância da câmera o recorte está (0.2..1)
//          u_kick   pulso 1 -> 0: o recorte salta para a frente
//          u_dolly  a câmera anda pelo salão, ida e volta (bar: {step: 1, morph: ...}: um passo por compasso)
//          u_hue    a cor do fractal (harmony:)
//          u_shadow 0..1: o quanto o fractal escurece a figura

uniform float u_depth = 0.5;
uniform float u_kick = 0.0;
uniform float u_dolly = 0.0;
uniform float u_hue = 0.6;
uniform float u_shadow = 0.7;

const vec3 CSIZE = vec3(0.92436, 0.90756, 0.92436);

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

float gTrap;
float de(vec3 p) {
    float f = 1.0, trap = 1e9;
    for (int i = 0; i < 8; i++) {
        p = 2.0 * clamp(p, -CSIZE, CSIZE) - p;
        float k = max(1.0 / dot(p, p), 1.0);
        p *= k;
        f *= k;
        trap = min(trap, dot(p, p));
    }
    gTrap = trap;
    float rxy = length(p.xy);
    return max(rxy - 0.92784, abs(rxy * p.z) / length(p)) / f;
}

float softShadow(vec3 p, vec3 l) {
    float res = 1.0, t = 0.01;
    for (int i = 0; i < 40; i++) {
        float h = de(p + l * t);
        res = min(res, 10.0 * h / t);
        t += clamp(h, 0.005, 0.1);
        if (res < 0.01 || t > 2.0) break;
    }
    return clamp(res, 0.0, 1.0);
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 res = iResolution.xy;
    vec2 uv01 = fragCoord / res;
    vec2 uv = (fragCoord - 0.5 * res) / res.y;
    // a hall with room round it (found on the CPU: >= 0.35 from the walls along the way);
    // the camera goes up it and back, a step per unit of u_dolly
    float walk = 5.0 - abs(mod(u_dolly, 10.0) - 5.0);
    vec3 ro = vec3(0.934, 0.901, -0.329 + 0.35 * walk);
    vec3 fw = normalize(vec3(0.15 * sin(0.2 * u_dolly), 0.05, 1.0));
    vec3 rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw)), up = cross(fw, rt);
    vec3 rd = normalize(fw + uv.x * rt + uv.y * up);
    float t = 0.0;
    bool hit = false;
    int steps = 0;
    for (int i = 0; i < 160; i++) {
        float d = de(ro + rd * t);
        if (d < 0.0005 * (1.0 + 8.0 * t)) { hit = true; break; }
        t += d * 0.9;
        steps = i;
        if (t > 6.0) break;
    }
    vec3 air = hsv(u_hue + 0.5, 0.3, 0.12);
    vec3 col = air;
    if (hit) {
        vec3 p = ro + rd * t;
        vec2 e = vec2(0.0006 * (1.0 + 8.0 * t), 0.0);
        vec3 n = normalize(vec3(de(p + e.xyy) - de(p - e.xyy), de(p + e.yxy) - de(p - e.yxy), de(p + e.yyx) - de(p - e.yyx)));
        de(p);
        vec3 base = hsv(u_hue + 0.12 * log(gTrap + 1.0), 0.5, 1.0);
        vec3 l = normalize(vec3(-0.4, 0.8, -0.5));
        float dif = clamp(dot(n, l), 0.0, 1.0);
        float ao = 1.0 - float(steps) / 160.0;
        col = base * (0.3 + 0.9 * dif) * (0.5 + 0.5 * ao);
        col = mix(col, air, 1.0 - exp(-0.03 * t * t));
    }
    // the cut-out: where a figure is and nothing of the fractal comes before the card
    float depth = u_depth * (1.0 - 0.3 * u_kick);
    float card = depth / max(dot(rd, fw), 1e-3);
    if (texture(iChannel1, uv01).r > 0.5 && (!hit || t > card)) {
        vec3 fig = texture(iChannel0, uv01).rgb;
        vec3 pc = ro + rd * card;
        float sh = softShadow(pc, normalize(vec3(-0.4, 0.8, -0.5)));
        col = fig * mix(1.0, 0.35 + 0.65 * sh, u_shadow);
    }
    fragColor = vec4(col, 1.0);
}

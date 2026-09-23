// title: kaleidoscope — anéis no kick, cor da harmonia
// url: https://www.shadertoy.com/view/ttKGDt (derivado)
// author: kasari39 (original, 2020-01-28); mudanças: ARC
// license: CC BY-NC-SA 3.0 (a do original; o derivado herda)
// notes: no original os anéis de luz azul correm sozinhos pelo túnel, a cada 30 unidades.
//        Aqui cada kick solta um anel da câmera túnel adentro: u_ring0..2 são os segundos
//        desde os três kicks mais recentes (>= 0 liga o modo; -1, o padrão, deixa o original).
//        u_hue (>= 0) pinta o cristal com a cor da harmonia (u_sat a saturação, u_paint o
//        quanto pinta); os anéis saem na mesma cor, mais quentes. Sem eles: o original.

precision highp float;

uniform float u_ring0 = -1.0;
uniform float u_ring1 = -1.0;
uniform float u_ring2 = -1.0;
uniform float u_ring_speed = 24.0;      // how fast a ring runs down the tunnel (units a second)
uniform float u_ring_life = 1.2;        // seconds until it has faded
uniform float u_hue = -1.0;
uniform float u_sat = 0.7;
uniform float u_paint = 0.6;

mat2 rot(float a) {
    float c = cos(a), s = sin(a);
    return mat2(c,s,-s,c);
}

const float pi = acos(-1.0);
const float pi2 = pi*2.0;

vec2 pmod(vec2 p, float r) {
    float a = atan(p.x, p.y) + pi/r;
    float n = pi2 / r;
    a = floor(a/n)*n;
    return p*rot(-a);
}

float box( vec3 p, vec3 b ) {
    vec3 d = abs(p) - b;
    return min(max(d.x,max(d.y,d.z)),0.0) + length(max(d,0.0));
}

float ifsBox(vec3 p) {
    for (int i=0; i<5; i++) {
        p = abs(p) - 1.0;
        p.xy *= rot(iTime*0.3);
        p.xz *= rot(iTime*0.1);
    }
    p.xz *= rot(iTime);
    return box(p, vec3(0.4,0.8,0.3));
}

float map(vec3 p, vec3 cPos) {
    vec3 p1 = p;
    p1.x = mod(p1.x-5., 10.) - 5.;
    p1.y = mod(p1.y-5., 10.) - 5.;
    p1.z = mod(p1.z, 16.)-8.;
    p1.xy = pmod(p1.xy, 5.0);
    return ifsBox(p1);
}

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

// ARC: how much of a kick's ring is at distance t from the camera (0 .. 1, fading as it goes)
float ringAt(float t, float age) {
    if (age < 0.0 || age > u_ring_life) return 0.0;
    float r = age * u_ring_speed;
    return step(abs(t - r), 1.5) * (1.0 - age / u_ring_life);
}

void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
    vec2 p = (fragCoord.xy * 2.0 - iResolution.xy) / min(iResolution.x, iResolution.y);

    vec3 cPos = vec3(0.0,0.0, -3.0 * iTime);
    // vec3 cPos = vec3(0.3*sin(iTime*0.8), 0.4*cos(iTime*0.3), -6.0 * iTime);
    vec3 cDir = normalize(vec3(0.0, 0.0, -1.0));
    vec3 cUp  = vec3(sin(iTime), 1.0, 0.0);
    vec3 cSide = cross(cDir, cUp);

    vec3 ray = normalize(cSide * p.x + cUp * p.y + cDir);

    // Phantom Mode https://www.shadertoy.com/view/MtScWW by aiekick
    bool kicks = u_ring0 >= 0.0;
    float acc = 0.0;
    float acc2 = 0.0;
    float t = 0.0;
    for (int i = 0; i < 99; i++) {
        vec3 pos = cPos + ray * t;
        float dist = map(pos, cPos);
        dist = max(abs(dist), 0.02);
        float a = exp(-dist*3.0);
        if (kicks) {                                    // ARC: the kicks' rings, running out from the camera
            float k = max(ringAt(t, u_ring0), max(ringAt(t, u_ring1), ringAt(t, u_ring2)));
            if (k > 0.0) {
                a *= 1.0 + k;
                acc2 += a * k;
            }
        } else if (mod(length(pos)+24.0*iTime, 30.0) < 3.0) {
            a *= 2.0;
            acc2 += a;
        }
        acc += a;
        t += dist * 0.5;
    }

    vec3 col = vec3(acc * 0.01, acc * 0.011 + acc2*0.002, acc * 0.012+ acc2*0.005);
    if (u_hue >= 0.0) {                                 // ARC: the chord paints the crystal
        vec3 tint = hsv(u_hue, u_sat, 1.0);
        vec3 hot = hsv(u_hue + 0.04, min(1.0, u_sat + 0.2), 1.0);
        col = acc * 0.011 * mix(vec3(1.0), tint, u_paint) + acc2 * 0.006 * hot;
    }
    fragColor = vec4(col, 1.0 - t * 0.03);
}

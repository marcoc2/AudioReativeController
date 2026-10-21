// title: Caleidoscópio do acorde (KIFS)
// url: (escrito para o ARC, não vem do Shadertoy)
// author: ARC
// license: MIT
// notes: um fractal 3D de dobras espelhadas (KIFS, de Knighty; escrito do zero a partir
//        da matemática), cuja simetria segue o acorde:
//          u_mode   0 menor: octaédrica · 0.5 sus/5: tetraédrica · 1 maior: icosaédrica
//                   (valores no meio misturam as duas formas: ligue com glide)
//          u_root   a tônica, 0..11: o eixo de uma torção entre as dobras (cada tom, um cristal)
//          u_third, u_fifth   as outras notas do acorde, 0..11: as cores dos orbit traps
//                   (a tônica pinta o perto do eixo x, a terça o y, a quinta o z; a cor de
//                   cada nota é o lugar dela no círculo das quintas)
//          u_kick   pulso 1 -> 0: a escala incha (o fractal abre)
//          u_snare  pulso 1 -> 0: uma torção na segunda rotação
//          u_orbit  a câmera em volta, em voltas (ex.: bar: {step: 0.125}: 1/8 de volta por compasso)
//          u_glow   0..1: o brilho das bordas
//          u_video  0..1: com channel0: frame, o clipe pintado na superfície (projeção
//                   triplanar: cada face pega o clipe pelo eixo em que mais olha)
//          u_slice  > -2: um plano corta o fractal ali (de -2 a 2 atravessa tudo; ligue no
//                   compasso para uma tomografia por compasso); a face cortada mostra o
//                   interior pelas cores dos traps, com a borda acesa. -9: sem corte
//          u_tilt   inclina o plano de corte (radianos)
//          u_p0..u_p3, u_l0..u_l3   até quatro notas como luzes dentro do fractal: a altura
//                   MIDI (o ângulo na roda das quintas, a altura pela oitava) e o brilho
//                   (voice: {track: ...}, k: 0..3, part: pitch | level)
//        Sem os uniforms: Mi menor parado, octaédrico.

uniform float u_mode = 0.0;
uniform float u_root = 4.0;
uniform float u_third = 7.0;
uniform float u_fifth = 11.0;
uniform float u_kick = 0.0;
uniform float u_snare = 0.0;
uniform float u_orbit = 0.0;
uniform float u_glow = 0.5;
uniform float u_video = 0.0;
uniform float u_slice = -9.0;
uniform float u_tilt = 0.0;
uniform float u_p0 = 60.0, u_p1 = 64.0, u_p2 = 67.0, u_p3 = 71.0;
uniform float u_l0 = 0.0, u_l1 = 0.0, u_l2 = 0.0, u_l3 = 0.0;

const int ITER = 9;
const float PHI = 1.61803399;

vec3 hsv(float h, float s, float v) {
    vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
    return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

// a note's colour: its place on the circle of fifths (the wheel harmony: uses)
vec3 note(float pc) { return hsv(mod(pc * 7.0, 12.0) / 12.0, 0.9, 1.0); }

mat3 rotAxis(vec3 a, float t) {
    a = normalize(a);
    float c = cos(t), s = sin(t), k = 1.0 - c;
    return mat3(c + a.x * a.x * k, a.y * a.x * k + a.z * s, a.z * a.x * k - a.y * s,
                a.x * a.y * k - a.z * s, c + a.y * a.y * k, a.z * a.y * k + a.x * s,
                a.x * a.z * k + a.y * s, a.y * a.z * k - a.x * s, c + a.z * a.z * k);
}

vec4 gTrap;          // the orbit trap of the last call: distances to the x, y, z axes and the origin

void fold(inout vec3 z, int sym) {
    if (sym == 0) {                                   // tetrahedral
        if (z.x + z.y < 0.0) z.xy = -z.yx;
        if (z.x + z.z < 0.0) z.xz = -z.zx;
        if (z.y + z.z < 0.0) z.zy = -z.yz;
    } else if (sym == 1) {                            // octahedral
        z = abs(z);
        if (z.x < z.y) z.xy = z.yx;
        if (z.x < z.z) z.xz = z.zx;
        if (z.y < z.z) z.yz = z.zy;
    } else {                                          // icosahedral
        const vec3 n1 = normalize(vec3(-PHI, PHI - 1.0, 1.0));
        const vec3 n2 = normalize(vec3(1.0, -PHI, PHI + 1.0));
        z = abs(z);
        z -= 2.0 * max(0.0, dot(z, n1)) * n1;
        z -= 2.0 * max(0.0, dot(z, n2)) * n2;
        z -= 2.0 * max(0.0, dot(z, n1)) * n1;
        z -= 2.0 * max(0.0, dot(z, n2)) * n2;
    }
}

float kifs(vec3 z, int sym) {
    float size = sym == 1 ? 1.9 : (sym == 0 ? 1.7 : 1.0);   // the three about as big
    z *= size;
    float scale = 2.0 + 0.35 * u_kick;
    vec3 offset = sym == 2 ? vec3(0.850650808, 0.525731112, 0.0) : vec3(1.0);
    // the root turns the axis of a small twist round the wheel: twelve keys, twelve crystals
    float ang = 6.2831853 * u_root / 12.0;
    mat3 r1 = rotAxis(vec3(cos(ang), 1.2, sin(ang)), 0.22 + 0.06 * sin(3.0 * ang));
    mat3 r2 = rotAxis(vec3(1.0, 0.2, 0.5), 0.2 + 0.9 * u_snare);
    vec4 trap = vec4(1e9);
    for (int i = 0; i < ITER; i++) {
        z = r1 * z;
        fold(z, sym);
        z = r2 * z;
        z = z * scale - offset * (scale - 1.0);
        trap = min(trap, vec4(length(z.yz), length(z.xz), length(z.xy), length(z)));
    }
    gTrap = trap;
    return (length(z) - 2.0) * pow(scale, -float(ITER)) / size;
}

float gCut;          // how far the last call's point is past the slicing plane (> 0: on the cut face)

float shape(vec3 p);

vec3 planeNormal() { return normalize(vec3(sin(u_tilt), cos(u_tilt), 0.3 * sin(2.0 * u_tilt))); }

float map(vec3 p) {
    float d = shape(p);
    gCut = -9.0;
    if (u_slice > -8.0) {
        float c = dot(p, planeNormal()) - u_slice;               // keep what is below the plane
        gCut = c - d;
        d = max(d, c);
    }
    return d;
}

float shape(vec3 p) {
    // u_mode: 0 octahedral, 0.5 tetrahedral, 1 icosahedral, in between a blend of two
    float m = clamp(u_mode, 0.0, 1.0);
    if (m <= 0.001) return kifs(p, 1);
    if (abs(m - 0.5) <= 0.001) return kifs(p, 0);
    if (m >= 0.999) return kifs(p, 2);
    if (m < 0.5) {
        float a = kifs(p, 1); vec4 ta = gTrap;
        float b = kifs(p, 0);
        float k = m * 2.0;
        gTrap = mix(ta, gTrap, k);
        return mix(a, b, k);
    }
    float a = kifs(p, 0); vec4 ta = gTrap;
    float b = kifs(p, 2);
    float k = (m - 0.5) * 2.0;
    gTrap = mix(ta, gTrap, k);
    return mix(a, b, k);
}

vec3 normalAt(vec3 p) {
    vec2 e = vec2(1e-3, 0.0);
    return normalize(vec3(map(p + e.xyy) - map(p - e.xyy), map(p + e.yxy) - map(p - e.yxy),
                          map(p + e.yyx) - map(p - e.yyx)));
}

void mainImage(out vec4 fragColor, in vec2 fragCoord) {
    vec2 uv = (fragCoord - 0.5 * iResolution.xy) / iResolution.y;
    float a = 6.2831853 * u_orbit + 0.05 * iTime;
    vec3 ro = vec3(5.0 * sin(a), 1.2 + 0.4 * sin(0.21 * iTime), 5.0 * cos(a));
    vec3 fw = normalize(-ro), rt = normalize(cross(vec3(0.0, 1.0, 0.0), fw)), up = cross(fw, rt);
    vec3 rd = normalize(fw * 1.6 + uv.x * rt + uv.y * up);

    // the notes' lights: round the crystal by their place on the wheel of fifths, up by octave
    float lp[4] = float[4](u_p0, u_p1, u_p2, u_p3);
    float ll[4] = float[4](u_l0, u_l1, u_l2, u_l3);
    vec3 lpos[4];
    for (int j = 0; j < 4; j++) {
        float an = 6.2831853 * mod(lp[j] * 7.0, 12.0) / 12.0;
        lpos[j] = vec3(1.1 * cos(an), (lp[j] - 60.0) / 18.0, 1.1 * sin(an));
    }
    vec3 lights = vec3(0.0);
    float t = 0.0, d = 1.0, glow = 0.0;
    int steps = 0;
    for (int i = 0; i < 140; i++) {
        d = map(ro + rd * t);
        glow += exp(-d * 60.0) * 0.012;
        vec3 q = ro + rd * t;
        for (int j = 0; j < 4; j++) {
            if (ll[j] <= 0.0) continue;
            vec3 dl = q - lpos[j];
            lights += note(lp[j]) * ll[j] * 0.006 / (dot(dl, dl) + 0.003) * max(d, 0.01);
        }
        if (d < 6e-4 * t || t > 12.0) break;
        t += d * 0.9;
        steps = i;
    }
    vec3 bg = mix(note(u_root) * 0.08, vec3(0.0), clamp(length(uv), 0.0, 1.0));
    vec3 col = bg;
    if (t < 12.0) {
        vec3 p = ro + rd * t;
        map(p);
        vec4 tr = gTrap;
        // the chord's notes, where the orbit came near each axis; white near the origin
        vec3 w = exp(-tr.xyz * 4.0);
        vec3 base = (note(u_root) * w.x + note(u_third) * w.y + note(u_fifth) * w.z) / (w.x + w.y + w.z + 1e-4);
        base = mix(base, vec3(1.0), clamp(exp(-tr.w * 0.8) * 0.3, 0.0, 0.3));
        bool cutFace = gCut > 0.0;
        if (cutFace) {                                          // inside, laid open: bands like a scan
            float band = 0.5 + 0.5 * sin(tr.w * 30.0 + tr.x * 18.0);
            base = mix(note(u_root), note(u_fifth), band) * (0.5 + 0.7 * exp(-tr.y * 2.0));
        }
        vec3 n = normalAt(p);
        if (u_video > 0.0) {                                    // the clip on the surface, triplanar
            vec3 bw = pow(abs(n), vec3(4.0));
            bw /= bw.x + bw.y + bw.z;
            vec3 q = p * 0.22 + 0.5;
            vec3 clip = texture(iChannel0, q.yz).rgb * bw.x + texture(iChannel0, q.xz).rgb * bw.y
                      + texture(iChannel0, q.xy).rgb * bw.z;
            base = mix(base, clip * 1.5, u_video);
        }
        float lift = 0.18 + 0.3 * u_video;                     // a painted surface wants more light
        vec3 l = normalize(vec3(0.6, 0.8, -0.4));
        float dif = clamp(dot(n, l), 0.0, 1.0);
        float ao = 1.0 - float(steps) / 140.0;
        col = base * (lift + 0.85 * dif) * (0.35 + 0.65 * ao);
        col += pow(clamp(dot(reflect(rd, n), l), 0.0, 1.0), 24.0) * 0.4;
        if (cutFace) col += note(u_third) * 0.6 * exp(-abs(shape(p)) * 400.0);   // the cut's rim
        for (int j = 0; j < 4; j++) {                           // the notes light the crystal
            if (ll[j] <= 0.0) continue;
            vec3 dl = lpos[j] - p;
            float fall = ll[j] * 0.35 / (dot(dl, dl) + 0.05);
            col += base * note(lp[j]) * fall * (0.3 + 0.7 * clamp(dot(n, normalize(dl)), 0.0, 1.0));
        }
        col = mix(col, bg, 1.0 - exp(-0.012 * t * t));
    }
    col += glow * u_glow * note(u_third);
    col += lights;
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(0.4545)), 1.0);
}

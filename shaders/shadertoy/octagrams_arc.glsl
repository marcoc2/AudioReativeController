// title: Octgrams — formas trocáveis
// url: https://www.shadertoy.com/view/tlVGDt (derivado)
// author: whisky_shusuky (original, 2020-01-28); mudanças: ARC
// license: CC BY-NC-SA 3.0 (a do original; o derivado herda)
// notes: a caixa achatada do original virou uma forma 2D extrudada, escolhida por u_shape:
//        0 quadrado (o original) · 1 círculo · 2 triângulo · 3 hexágono · 4 cruz · 5 estrela de 5
//        valores fracionários fazem morph de uma forma para a próxima (2.5 = meio triângulo,
//        meio hexágono). A cópia girada usa o ângulo que casa com a simetria da forma
//        (quadrado ~46° = octagrama; triângulo 60° = hexagrama; hexágono 30°; estrela 36°),
//        ou u_twin se for >= 0. u_size escala a forma.
//        u_fly (>= 0): a distância voada pelo túnel no lugar de iTime*4 (a música acelera só o voo);
//        u_hue (>= 0): a cor no lugar do azul do original (u_sat a saturação). Sem eles: o original.

precision highp float;

uniform float u_shape = 0.0;
uniform float u_twin = -1.0;
uniform float u_size = 1.0;
uniform float u_fly = -1.0;
uniform float u_hue = -1.0;
uniform float u_sat = 0.8;

vec3 hsv(float h, float s, float v) {
	vec3 k = fract(h + vec3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0;
	return v * mix(vec3(1.0), clamp(abs(k - 3.0) - 1.0, 0.0, 1.0), s);
}

float gTime = 0.;
const float REPEAT = 5.0;

// 回転行列
mat2 rot(float a) {
	float c = cos(a), s = sin(a);
	return mat2(c,s,-s,c);
}

float sdBox( vec3 p, vec3 b )
{
	vec3 q = abs(p) - b;
	return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

// ARC: 2D shapes (Inigo Quilez's exact distances), each about as big as the original's square
float sd2Circle(vec2 p, float r) { return length(p) - r; }
float sd2Triangle(vec2 p, float r) {
	const float k = sqrt(3.0);
	p.x = abs(p.x) - r;
	p.y = p.y + r / k;
	if (p.x + k * p.y > 0.0) p = vec2(p.x - k * p.y, -k * p.x - p.y) / 2.0;
	p.x -= clamp(p.x, -2.0 * r, 0.0);
	return -length(p) * sign(p.y);
}
float sd2Hexagon(vec2 p, float r) {
	const vec3 k = vec3(-0.866025404, 0.5, 0.577350269);
	p = abs(p);
	p -= 2.0 * min(dot(k.xy, p), 0.0) * k.xy;
	p -= vec2(clamp(p.x, -k.z * r, k.z * r), r);
	return length(p) * sign(p.y);
}
float sd2Cross(vec2 p, float b, float w) {
	p = abs(p); p = (p.y > p.x) ? p.yx : p.xy;
	vec2 q = p - vec2(b, w);
	float k = max(q.y, q.x);
	vec2 v = (k > 0.0) ? q : vec2(w - p.x, -k);
	return sign(k) * length(max(v, 0.0));
}
float sd2Star5(vec2 p, float r, float rf) {
	const vec2 k1 = vec2(0.809016994375, -0.587785252292);
	const vec2 k2 = vec2(-k1.x, k1.y);
	p.x = abs(p.x);
	p -= 2.0 * max(dot(k1, p), 0.0) * k1;
	p -= 2.0 * max(dot(k2, p), 0.0) * k2;
	p.x = abs(p.x);
	p.y -= r;
	vec2 ba = rf * vec2(-k1.y, k1.x) - vec2(0, 1);
	float h = clamp(dot(p, ba) / dot(ba, ba), 0.0, r);
	return length(p - ba * h) * sign(p.y * ba.x - p.x * ba.y);
}
float sd2Shape(vec2 p, int s) {
	if (s == 1) return sd2Circle(p, 0.42);
	if (s == 2) return sd2Triangle(p, 0.5);
	if (s == 3) return sd2Hexagon(p, 0.38);
	if (s == 4) return sd2Cross(p, 0.45, 0.14);
	if (s == 5) return sd2Star5(p, 0.55, 0.45);
	vec2 q = abs(p) - 0.4;                                      // 0: the original's square
	return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0);
}
float twinOf(int s) {                                           // the turn that matches its symmetry
	if (s == 1) return 0.0;
	if (s == 2) return 1.0471976;
	if (s == 3) return 0.5235988;
	if (s == 4) return 0.7853982;
	if (s == 5) return 0.6283185;
	return 0.8;                                                 // the original's
}
float shapeTwin() {
	if (u_twin >= 0.0) return u_twin;
	float s = mod(u_shape, 6.0);
	int a = int(floor(s)), b = int(mod(floor(s) + 1.0, 6.0));
	return mix(twinOf(a), twinOf(b), smoothstep(0.0, 1.0, fract(s)));
}

float box(vec3 pos, float scale) {
	pos *= scale / u_size;
	// ARC: the flat box of the original, now any 2D shape extruded 0.1 deep (morphing on fractions)
	float s = mod(u_shape, 6.0);
	int a = int(floor(s)), b = int(mod(floor(s) + 1.0, 6.0));
	float d2 = mix(sd2Shape(pos.xy, a), sd2Shape(pos.xy, b), smoothstep(0.0, 1.0, fract(s)));
	vec2 w = vec2(d2, abs(pos.z) - 0.1);
	float base = (min(max(w.x, w.y), 0.0) + length(max(w, 0.0))) * u_size / 1.5;
	float result = -base;
	return result;
}

float box_set(vec3 pos, float iTime) {
	float tw = shapeTwin();
	vec3 pos_origin = pos;
	pos = pos_origin;
	pos .y += sin(gTime * 0.4) * 2.5;
	pos.xy *=   rot(tw);
	float box1 = box(pos,2. - abs(sin(gTime * 0.4)) * 1.5);
	pos = pos_origin;
	pos .y -=sin(gTime * 0.4) * 2.5;
	pos.xy *=   rot(tw);
	float box2 = box(pos,2. - abs(sin(gTime * 0.4)) * 1.5);
	pos = pos_origin;
	pos .x +=sin(gTime * 0.4) * 2.5;
	pos.xy *=   rot(tw);
	float box3 = box(pos,2. - abs(sin(gTime * 0.4)) * 1.5);
	pos = pos_origin;
	pos .x -=sin(gTime * 0.4) * 2.5;
	pos.xy *=   rot(tw);
	float box4 = box(pos,2. - abs(sin(gTime * 0.4)) * 1.5);
	pos = pos_origin;
	pos.xy *=   rot(tw);
	float box5 = box(pos,.5) * 6.;
	pos = pos_origin;
	float box6 = box(pos,.5) * 6.;
	float result = max(max(max(max(max(box1,box2),box3),box4),box5),box6);
	return result;
}

float map(vec3 pos, float iTime) {
	vec3 pos_origin = pos;
	float box_set1 = box_set(pos, iTime);

	return box_set1;
}


void mainImage( out vec4 fragColor, in vec2 fragCoord ) {
	vec2 p = (fragCoord.xy * 2. - iResolution.xy) / min(iResolution.x, iResolution.y);
	vec3 ro = vec3(0., -0.2 , u_fly >= 0.0 ? u_fly : iTime * 4.);
	vec3 ray = normalize(vec3(p, 1.5));
	ray.xy = ray.xy * rot(sin(iTime * .03) * 5.);
	ray.yz = ray.yz * rot(sin(iTime * .05) * .2);
	float t = 0.1;
	vec3 col = vec3(0.);
	float ac = 0.0;


	for (int i = 0; i < 99; i++){
		vec3 pos = ro + ray * t;
		pos = mod(pos-2., 4.) -2.;
		gTime = iTime -float(i) * 0.01;

		float d = map(pos, iTime);

		d = max(abs(d), 0.01);
		ac += exp(-d*23.);

		t += d* 0.55;
	}

	col = vec3(ac * 0.02);

	if (u_hue < 0.0) {
		col +=vec3(0.,0.2 * abs(sin(iTime)),0.5 + sin(iTime) * 0.2);
	} else {                                                    // ARC: the chord's colour
		vec3 tint = hsv(u_hue, u_sat, 1.0);
		col = col * mix(vec3(1.0), tint, 0.3) + tint * (0.5 + 0.15 * sin(iTime));
	}


	fragColor = vec4(col ,1.0 - t * (0.02 + 0.02 * sin (iTime)));
}

/** SHADERDATA
{
	"title": "Octgrams",
	"description": "Lorem ipsum dolor",
	"model": "person"
}
*/

// title: 2D distance field shadows — luzes da música — Buffer A
// url: https://www.shadertoy.com/view/XsK3RR (derivado)
// author: Flyguy (original, 2016-01-17); mudanças: ARC
// license: CC BY-NC-SA 3.0 (a do original; o derivado herda)
// notes: cada luz segue um instrumento: vermelha = kick, rosa/verde (as do mouse) = caixa, azul = hi-hat gira a órbita, amarela = acid (altura = nota), roxa = baixo (altura = nota); a fenda do anel gira com o baixo (u_spin no lugar de iTime, nas duas abas)

uniform float u_kick;
uniform float u_snare;
uniform float u_turn;
uniform float u_orbit;
uniform float u_hat;
uniform float u_acid;
uniform float u_acid_y;
uniform float u_bass;
uniform float u_bass_y;
uniform float u_spin;

//1D Distance field shadow map.
//Each row is a radial shadow map for a light.
//r = distance
//gba = light info (position / color)

#define MAX_STEPS 48
#define INF 1e8
#define EPS 1e-4
float tau = atan(1.0)*8.0;

//Globals
vec2 res = vec2(0);
vec2 mouse = vec2(0);

// Shapes
float sdCircle(float r, vec2 uv)
{
	return length(uv) - r;    
}

float sdRing(float ir, float or, vec2 uv)
{
	return abs(length(uv) - (ir+or)/2.0) - (or - ir);   
}

float sdBox(float s, vec2 uv)
{
	return max(abs(uv.x), abs(uv.y)) - s;   
}

float sdRect(vec2 s, vec2 uv)
{
    uv = abs(uv) - s;
	return max(uv.x, uv.y);
}

float sdPlane(vec2 dir, vec2 uv)
{
	return dot(normalize(dir), uv);   
}

// Operations
float opU(float a, float b)
{
	return min(a, b);   
}

float opI(float a, float b)
{
	return max(a, b);   
}

float opS(float a, float b)
{
	return max(-a, b);   
}

//Domain modifiers
mat2 Rotate(float a)
{
	return mat2(cos(a), sin(a),-sin(a), cos(a));   
}

vec2 Rep1(vec2 uv, float r)
{
	uv.x = mod(uv.x, r) - r/2.0;
    return uv;
}

vec2 Rep2(vec2 uv, vec2 r)
{
	return mod(uv, r) - r/2.0;  
}

// Scene function (must be changed in both tabs)
float Scene(vec2 uv)
{
	float d = -sdRect(res/2.0 - 0.05, uv);
    
    vec2 rp = Rep2(uv, vec2(0.2));
    
    d = opU(sdCircle(0.02, rp), d);
    
    rp = Rep1(uv, 0.2);
    
    d = opU(sdRect(vec2(0.005,0.1), rp), d);
    
    d = opS(sdBox(0.2, uv), d);
    
    d = opU(sdRing(0.08, 0.09, uv), d);
    
    d = opS(sdRect(vec2(0.11,0.03), uv * Rotate(u_spin)), d);
    
    return d;
}

float MarchShadow(vec2 orig, vec2 dir)
{
    float d = 0.0;
    
    for(int i = 0;i < MAX_STEPS;i++)
    {
        float ds = Scene(dir * d - orig);
        
        d += ds;
        
        if(ds < EPS)
        {
        	break;   
        }
    }
    
    return d;
}

//Data slots
#define SLOT_POSITION 0
#define SLOT_COLOR 1

struct Light
{
	vec2 origin;
    vec3 color;
    float brightness;
    
};

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
    res = iResolution.xy / iResolution.y;  
    mouse = iMouse.xy / iResolution.y - res/2.0;
    
    float a = (fragCoord.x / iResolution.x) * tau;
    vec2 dir = vec2(cos(a), sin(a));
    
    int id = int(fragCoord.y);
    
    Light light;
    
    light.origin = vec2(0);
    light.color = vec3(0);
    light.brightness = 0.0;
    
    // ARC: each light follows an instrument (the original's lights, same places and colours)
    if(id == 0)                                   // red beacon inside the ring: the kick
    {
    	light.origin = vec2(0);
        light.color = vec3(1.0, 0.2, 0.2);
        light.brightness = 0.3 + 4.5 * u_kick;
    }
    if(id == 1)                                   // the two mouse lights, now a mirrored pair: the snare
    {
        vec2 o = vec2(0.6 * cos(u_turn), 0.33 * sin(u_turn));
    	light.origin = o;
        light.color = vec3(1.0, 0.6, 0.6);
        light.brightness = 3.0 * u_snare;
    }
    if(id == 2)
    {
        vec2 o = vec2(0.6 * cos(u_turn), 0.33 * sin(u_turn));
    	light.origin = -o;
        light.color = vec3(0.6, 1.0, 0.6);
        light.brightness = 3.0 * u_snare;
    }
    if(id == 3)                                   // blue, circling the ring: the hi-hat drives its orbit
    {
        float a = -u_orbit;
    	light.origin = vec2(cos(a), sin(a)) * 0.2;
        light.color = vec3(0.4, 0.4, 1.0);
        light.brightness = 2.5 + 1.5 * u_hat;
    }
    if(id == 4)                                   // yellow, right: the acid line (height = its pitch)
    {
    	light.origin = vec2(0.4, mix(-0.25, 0.25, u_acid_y));
        light.color = vec3(1.0, 1.0, 0.4);
        light.brightness = 0.4 + 2.5 * u_acid;
    }
    if(id == 5)                                   // purple, left: the bass (height = its pitch)
    {
    	light.origin = vec2(-0.4, mix(-0.25, 0.25, u_bass_y));
        light.color = vec3(1.0, 0.4, 1.0);
        light.brightness = 0.4 + 2.5 * u_bass;
    }

    int slot = int(fragCoord.x);
    vec3 data = vec3(0);
    
    if(slot == SLOT_POSITION)
    {
        data = vec3(light.origin,0);
    }
    
    if(slot == SLOT_COLOR)
    {
    	data = light.brightness * light.color;   
    }
    
    float dist = MarchShadow(light.origin, dir);
     
	fragColor = vec4(dist, data);
} 
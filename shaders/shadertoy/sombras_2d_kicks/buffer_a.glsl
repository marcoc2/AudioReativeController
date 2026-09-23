// title: 2D distance field shadows — uma luz por kick — Buffer A
// url: https://www.shadertoy.com/view/XsK3RR (derivado)
// author: Flyguy (original, 2016-01-17); mudanças: ARC
// license: CC BY-NC-SA 3.0 (a do original; o derivado herda)
// notes: quatro luzes (vermelha, azul, amarela, roxa) acendem uma a cada kick do compasso e apagam no compasso seguinte; a mais nova pisca (u_flash); do 5º kick em diante piscam todas

uniform float u_count;
uniform float u_flash;

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
    
    d = opS(sdRect(vec2(0.11,0.03), uv * Rotate(iTime)), d);
    
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
    
    // ARC: four lights; each kick of the bar turns the next one on (u_count: kicks so far in
    // this bar), the newest flashing (u_flash); from the fifth kick on all four flash; a new bar
    // starts dark again
    int order = -1;                               // which kick turns this row's light on
    if(id == 0)                                   // 1st kick: the red beacon inside the ring
    {
        order = 0;
    	light.origin = vec2(0);
        light.color = vec3(1.0, 0.2, 0.2);
        light.brightness = 3.5;
    }
    if(id == 3)                                   // 2nd kick: blue, circling the ring
    {
        order = 1;
        float a = -iTime * 0.3;
    	light.origin = vec2(cos(a), sin(a)) * 0.2;
        light.color = vec3(0.4, 0.4, 1.0);
        light.brightness = 4.0;
    }
    if(id == 4)                                   // 3rd kick: yellow, right
    {
        order = 2;
    	light.origin = vec2(0.4, sin(3.0*iTime-tau/4.0)*0.2);
        light.color = vec3(1.0, 1.0, 0.4);
        light.brightness = 1.6;
    }
    if(id == 5)                                   // 4th kick: purple, left
    {
        order = 3;
    	light.origin = vec2(-0.4, sin(3.0*iTime)*0.2);
        light.color = vec3(1.0, 0.4, 1.0);
        light.brightness = 1.6;
    }
    int lit = int(u_count + 0.5);
    if(order < 0 || order >= lit)
    {
        light.brightness = 0.0;                   // not yet in this bar (rows 1 and 2 never)
    }
    else
    {
        bool newest = order == lit - 1 || lit > 4;
        light.brightness *= newest ? 1.0 + 2.5 * u_flash : 1.0;
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
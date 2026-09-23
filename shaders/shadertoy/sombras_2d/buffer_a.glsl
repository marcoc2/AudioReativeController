// title: 2D distance field shadows (6 luzes) — Buffer A
// url: (preencher)
// author: (preencher)
// license: CC BY-NC-SA 3.0 (padrão do Shadertoy; confirmar)
// notes: mapa de sombras 1D: cada linha é uma luz; r = distância, gba = posição/cor

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
    
    if(id == 0)
    {
    	light.origin = vec2(0);
        light.color = vec3(1.0, 0.2, 0.2);
        light.brightness = sin(iTime * 4.0) * 2.0 + 2.0;
    }
    
    if(id == 1)
    {
    	light.origin = mouse;
        light.color = vec3(1.0, 0.6, 0.6);
        light.brightness = 2.0;
    }
    if(id == 2)
    {
    	light.origin = -mouse;
        light.color = vec3(0.6, 1.0, 0.6);
        light.brightness = 2.0;
    }
    if(id == 3)
    {
        float a = -iTime * 0.3;
    	light.origin = vec2(cos(a), sin(a)) * 0.2; 
        light.color = vec3(0.4, 0.4, 1.0);
        light.brightness = 4.0;
    }
    if(id == 4)
    {
    	light.origin = vec2(0.4, sin(3.0*iTime-tau/4.0)*0.2); 
        light.color = vec3(1.0, 1.0, 0.4);
        light.brightness = 1.0;
    }
    if(id == 5)
    {
    	light.origin = vec2(-0.4, sin(3.0*iTime)*0.2); 
        light.color = vec3(1.0, 0.4, 1.0);
        light.brightness = 1.0;
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
// title: Diamond Dust — Common
// url: https://www.shadertoy.com/view/ffKXRw (derivado; só o Buffer A muda)
// author: Tonny Espeset (2026)
// license: livre com crédito ("Feel free to use this code, but please keep this credit and link.")
// notes: estilos sorteados a cada 11 s (cor, padrão, câmera em 3 famílias), transição nos 3 s finais; câmera e pós (foco, bloom) por estilo

/*

Created by Tonny Espeset, 2026
https://www.shadertoy.com/view/ffKXRw

Diamond Dust

Feel free to use this code, but please keep this credit and link.

Check my other shaders at

https://www.shadertoy.com/user/Espeset

*/

const float DD_PI=3.141592653589793;
const float DD_TAU=6.283185307179586;
const float DD_HALF=5.;
const float DD_CELL=10./512.;

float ddSaturate(float x){return clamp(x,0.,1.);}
float ddQuintic(float x){x=ddSaturate(x);return x*x*x*(x*(x*6.-15.)+10.);}

uint ddHash(uint v){
    v^=0xa3c59ac3u;v*=0x9e3779b9u;
    v^=v>>16u;v*=0x9e3779b9u;
    v^=v>>16u;v*=0x9e3779b9u;
    return v;
}

float ddRandom(uint seed,uint lane){return float(ddHash(seed+lane))*(1./4294967296.);}

uint ddCellHash(ivec2 p){
    uint x=uint(p.x),y=uint(p.y);
    return ddHash(x*0x8da6b343u^y*0xd8163841u^0xcb1ab31fu);
}

float ddLattice(ivec2 p,uint salt){return ddRandom(ddCellHash(p)^ddHash(salt),0u);}

float ddNoise(vec2 p,uint salt){
    ivec2 i=ivec2(floor(p));
    vec2 f=fract(p);f=f*f*(3.-2.*f);
    float a=ddLattice(i,salt),b=ddLattice(i+ivec2(1,0),salt);
    float c=ddLattice(i+ivec2(0,1),salt),d=ddLattice(i+ivec2(1),salt);
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}

vec3 ddHsv(vec3 c){
    vec3 p=abs(fract(c.xxx+vec3(0.,2./3.,1./3.))*6.-3.);
    return c.z*mix(vec3(1.),clamp(p-1.,0.,1.),c.y);
}

float ddHueMix(float a,float b,float t){
    float d=fract(b-a+.5)-.5;
    return fract(a+d*t);
}

struct DDStyle{
    float id;
    float family;
    float hue;
    float accentHue;
    float saturation;
    float angle;
    float scale;
    float aspect;
    float warp;
    float persistence;
    float lemniscate;
    vec2 offset;
    vec2 morphOffset;
};

float ddStyleRandom(float id,uint lane){
    uint n=uint(max(id,0.));
    return ddRandom(ddHash(n*0x9e3779b9u^0x6d2b79f5u),lane*0x85ebca6bu);
}

DDStyle ddMakeStyle(float id){
    DDStyle s;
    float r0=ddStyleRandom(id,0u),r1=ddStyleRandom(id,1u),r2=ddStyleRandom(id,2u);
    float r3=ddStyleRandom(id,3u),r4=ddStyleRandom(id,4u),r5=ddStyleRandom(id,5u);
    float r6=ddStyleRandom(id,6u),r7=ddStyleRandom(id,7u),r8=ddStyleRandom(id,8u);
    s.id=id;
    s.family=id<.5?0.:mod(id,3.);
    s.hue=id<.5?.075:r0;
    float split=r1<.5?mix(.045,.12,r2)*(r3<.5?-1.:1.):mix(.28,.48,r2)*(r3<.5?-1.:1.);
    s.accentHue=id<.5?.105:fract(s.hue+split);
    s.saturation=id<.5?.9:mix(.55,1.,r4);
    s.angle=id<.5?.25:DD_TAU*r5;
    s.scale=id<.5?1.:mix(.82,1.22,r6);
    s.aspect=id<.5?1.2:mix(.78,1.28,r7);
    s.warp=mix(.2,.38,r8);
    s.persistence=mix(.57,.7,ddStyleRandom(id,9u));
    s.lemniscate=id<.5?1.:0.;
    s.offset=id<.5?vec2(.07,-.055):vec2(ddStyleRandom(id,10u)-.5,ddStyleRandom(id,11u)-.5);
    s.morphOffset=31.*vec2(ddStyleRandom(id,12u),ddStyleRandom(id,13u));
    return s;
}

void ddStylePhase(float time,out float id,out float transition){
    float t=max(time,0.);
    id=floor(t/11.);
    transition=ddQuintic((t-id*11.-8.)/3.);
}

vec4 ddLayerGeom(DDStyle s,int layer){
    float jitter=s.id<.5?0.:(ddStyleRandom(s.id,uint(20+layer))-.5)*.002;
    if(layer==0){
        float y=mix(.0358276367,.0315,step(.5,s.family)*step(s.family,1.5))+jitter;
        return vec4(.049987793,y,.0392156877,.0196078438);
    }
    return vec4(.049987793,.0374450684+jitter,.0313725509,.0156862754);
}

vec3 ddLayerSpin(int layer){
    return layer==0?vec3(.0196078438,.0078431377,1.):vec3(.0078431377,.0039215689,1.);
}

struct DDCamera{
    vec3 eye;
    vec3 right;
    vec3 up;
    vec3 forward;
};

float ddSourceTime(DDStyle s,float time){
    float amount=ddSaturate((time-(s.id*11.-3.))/14.);
    float start=s.family<.5?60.9708333333:(s.family<1.5?68.5708333333:76.2041666667);
    float end=s.family<.5?68.5583333333:(s.family<1.5?76.1916666667:82.8583333333);
    return mix(start,end,amount);
}

DDCamera ddStyleCamera(DDStyle s,float time){
    DDCamera c;
    float source=ddSourceTime(s,time);
    vec3 eye,rate,right,anchorUp,anchorForward;
    float anchor,angleRate;
    if(s.family<.5){
        eye=vec3(-.0844116211,2.259765625,-3.93310115);
        rate=vec3(0.,0.,.19489053);anchor=64.;angleRate=-.0064827;
        right=vec3(.9999419451,.0000004396,-.0107735042);
        anchorUp=vec3(.007421071,.7248959541,.6888220906);
        anchorForward=vec3(.0078100003,-.6888620257,.7248538733);
    }else if(s.family<1.5){
        eye=vec3(-1.26171875,2.259765625,1.330246925);
        rate=vec3(0.,0.,-.23956955);anchor=73.8;angleRate=0.;
        right=vec3(-.3833194971,.0000357628,-.9237259626);
        anchorUp=vec3(.7119913101,.6370837688,-.2953743935);
        anchorForward=vec3(.5885049105,-.7708292007,-.2442690134);
    }else{
        eye=vec3(-2.22008,1.08691,1.87843);
        rate=vec3(-.14209,0.,-.05918);anchor=79.5;angleRate=0.;
        right=vec3(-.1682394748,0.,-.9857461535);
        anchorUp=vec3(.5283690578,.8442121075,-.090177915);
        anchorForward=vec3(.8321788377,-.5360092514,-.1420298016);
    }
    if(s.id>.5){
        eye.x+=(ddStyleRandom(s.id,30u)-.5)*.6;
        eye.y*=mix(.92,1.08,ddStyleRandom(s.id,31u));
        eye.z+=(ddStyleRandom(s.id,32u)-.5)*.6;
        rate*=mix(.86,1.14,ddStyleRandom(s.id,33u));
        angleRate+=mix(-.0012,.0012,ddStyleRandom(s.id,34u));
    }
    float delta=angleRate*(source-anchor),co=cos(delta),si=sin(delta);
    c.eye=eye+rate*(source-anchor);
    c.right=right;
    c.up=anchorUp*co+anchorForward*si;
    c.forward=-anchorUp*si+anchorForward*co;
    return c;
}

DDCamera ddCameraAt(float time){
    float id,t;ddStylePhase(time,id,t);
    DDCamera a=ddStyleCamera(ddMakeStyle(id),time);
    if(t<=0.)return a;
    DDCamera b=ddStyleCamera(ddMakeStyle(id+1.),time),c;
    c.eye=mix(a.eye,b.eye,t);
    c.right=normalize(mix(a.right,b.right,t));
    c.forward=normalize(mix(a.forward,b.forward,t));
    c.up=normalize(cross(c.forward,c.right));
    c.right=normalize(cross(c.up,c.forward));
    return c;
}

struct DDPost{
    float focusControl;
    float apertureControl;
    float strengthControl;
    vec3 bloomTint;
};

DDPost ddPostForStyle(DDStyle s,float time){
    DDPost p;
    float source=ddSourceTime(s,time),progress;
    if(s.family<.5)p.focusControl=.445068359375;
    else if(s.family<1.5){progress=ddSaturate((source-68.5708333333)/(76.1708333333-68.5708333333));p.focusControl=mix(.3770463467,.4620649815,progress);}
    else{progress=ddSaturate((source-76.2041666667)/(82.8333333333-76.2041666667));p.focusControl=mix(.3938723803,.2838296592,progress);}
    if(s.id>.5)p.focusControl*=mix(.96,1.04,ddStyleRandom(s.id,41u));
    p.apertureControl=s.family<.5?.7882353067:(s.family<1.5?1.:.7882353067);
    if(s.id>.5)p.apertureControl*=mix(.94,1.06,ddStyleRandom(s.id,42u));
    p.strengthControl=1.;
    vec3 tint=s.id<.5?vec3(1.,.5960784554,.4235294163):ddHsv(vec3(s.accentHue,mix(.1,.3,ddStyleRandom(s.id,44u)),1.));
    p.bloomTint=tint;
    return p;
}

DDPost ddPostAt(float time){
    float id,t;ddStylePhase(time,id,t);
    DDPost a=ddPostForStyle(ddMakeStyle(id),time);
    if(t<=0.)return a;
    DDPost b=ddPostForStyle(ddMakeStyle(id+1.),time),p;
    p.focusControl=mix(a.focusControl,b.focusControl,t);
    p.apertureControl=mix(a.apertureControl,b.apertureControl,t);
    p.strengthControl=mix(a.strengthControl,b.strengthControl,t);
    p.bloomTint=mix(a.bloomTint,b.bloomTint,t);
    return p;
}

float ddEncodeDepth(float viewDepth){return .01/max(viewDepth,1e-6);}
float ddDepthDistance(float encoded){return encoded>0.?1.+.01/encoded:1e6;}
float ddSceneDistance(sampler2D scene,vec2 uv){return ddDepthDistance(textureLod(scene,uv,0.).a);}

float ddDofWeight(sampler2D scene,vec2 uv,float centerCoc,float centerDistance,float neighbourCoc){
    float weight=ddSceneDistance(scene,uv)<centerDistance?neighbourCoc*(.16078431904315948*255.):1.;
    if(!(centerCoc>neighbourCoc+.062745101749897/10.))weight=1.;
    return clamp(weight,0.,1.);
}

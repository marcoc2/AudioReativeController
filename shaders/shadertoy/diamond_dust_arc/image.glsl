// title: Diamond Dust — Image
// url: https://www.shadertoy.com/view/ffKXRw (derivado; só o Buffer A muda)
// author: Tonny Espeset (2026)
// license: livre com crédito ("Feel free to use this code, but please keep this credit and link.")
// iChannel0: buffer_c mipmap
// notes: aberração cromática, bloom (mipmap nível 5), ACES, grão, curva de cor, sRGB

/*

Created by Tonny Espeset, 2026
https://www.shadertoy.com/view/ffKXRw

Diamond Dust v2

Inspired by Empires (Conspiracy)

Feel free to use this code, but please keep this credit and link.

Check my other shaders at

https://www.shadertoy.com/user/Espeset

*/

vec4 ddChromatic(vec2 uv){
    const int sampleCount=18;
    const float radiusControl=.0470588244497776;
    vec2 towardCenter=vec2(.5)-uv;
    vec4 result=textureLod(iChannel0,uv,0.);
    float red=result.r,blue=result.b;
    for(int index=1;index<sampleCount;index++){
        float displacement=float(index)/200.*radiusControl;
        red+=textureLod(iChannel0,uv-towardCenter*displacement,0.).r;
        blue+=textureLod(iChannel0,uv+towardCenter*displacement,0.).b;
    }
    result.r=red/float(sampleCount);
    result.b=blue/float(sampleCount);
    return result;
}

vec4 ddRandom4(vec2 coordinate,float seed){
    float value=sin(dot(coordinate+vec2(seed),vec2(12.9898,78.233)))*43758.5453;
    return fract(value*vec4(1.,1.2154,1.3453,1.3647))*2.-1.;
}

float ddNoise3(vec3 coordinate,float seed){
    const float cell=1./256.;
    float corner[8];
    vec3 grid=cell*floor(coordinate)+vec3(.5/256.);
    vec3 local=fract(coordinate);
    for(int x=0;x<2;x++) for(int y=0;y<2;y++) for(int z=0;z<2;z++){
        vec4 first=ddRandom4(grid.xy+cell*vec2(float(x),float(y)),seed);
        vec3 gradient=ddRandom4(vec2(first.w,grid.z+cell*float(z)),seed).xyz*4.-1.;
        corner[x*4+y*2+z]=dot(gradient,local-vec3(float(x),float(y),float(z)));
    }
    vec4 alongX=mix(vec4(corner[0],corner[1],corner[2],corner[3]),vec4(corner[4],corner[5],corner[6],corner[7]),ddQuintic(local.x));
    vec2 alongY=mix(alongX.xy,alongX.zw,ddQuintic(local.y));
    return mix(alongY.x,alongY.y,ddQuintic(local.z));
}

vec2 ddRotateUv(vec2 coordinate,float angle){
    float aspect=iResolution.x/iResolution.y;
    coordinate.x*=aspect;
    float c=cos(angle),s=sin(angle);
    coordinate=vec2(c*coordinate.x-s*coordinate.y,s*coordinate.x+c*coordinate.y);
    coordinate.x/=aspect;
    return coordinate;
}

vec3 ddAces(vec3 color){
    color=max(color,vec3(0.));
    vec3 aces=vec3(dot(vec3(.59719,.35458,.04823),color),dot(vec3(.076,.90834,.01566),color),dot(vec3(.0284,.13383,.83777),color));
    vec3 mapped=(aces*(aces+.0245786)-.000090537)/(aces*(.983729*aces+.432951)+.238081);
    return clamp(vec3(dot(vec3(1.60475,-.53108,-.07367),mapped),dot(vec3(-.10208,1.10813,-.00605),mapped),dot(vec3(-.00327,-.07276,1.07602),mapped)),0.,1.);
}

vec3 ddGradeCurve(vec3 color){
    vec3 value=vec3(28.2009178,-4.47737138,20.4365427);
    value=value*color+vec3(-101.437163,13.6101229,-63.9569408);
    value=value*color+vec3(136.645712,-16.442739,79.084147);
    value=value*color+vec3(-88.0052381,9.97617983,-48.5028081);
    value=value*color+vec3(29.3367147,-3.05535481,15.0955054);
    value=value*color+vec3(-4.0966274,1.35431496,-1.50459414);
    value=value*color+vec3(.351631336,.0282354678,.293262509);
    return clamp(value*color+vec3(-.00198259751,.00569606752,.0057168624),0.,1.);
}

vec3 ddToSrgb(vec3 color){
    return mix(1.055*pow(color,vec3(1./2.4))-.055,color*12.92,lessThanEqual(color,vec3(.0031308)));
}

vec3 ddGrain(vec3 color,vec2 uv,float seed){
    const float channelMix=.501960813999176;
    const float luminanceMix=.556862771511078;
    const float grainScale=.290196090936661;
    const float grainAmount=.372549027204514;
    vec2 frequency=iResolution.xy/mix(1.5,2.5,grainScale);
    vec2 nativeUv=vec2(uv.x,1.-uv.y);
    vec3 noise=vec3(ddNoise3(vec3(ddRotateUv(nativeUv,seed+1.425)*frequency,0.),seed),ddNoise3(vec3(ddRotateUv(nativeUv,seed+3.892)*frequency,1.),seed),ddNoise3(vec3(ddRotateUv(nativeUv,seed+5.835)*frequency,2.),seed));
    noise=mix(vec3(noise.r),noise,channelMix);
    float luminance=mix(0.,dot(color,vec3(.3,.587,.114)),luminanceMix);
    float reverse=clamp((luminance-.2)/-.2,0.,1.);
    reverse=reverse*reverse*(3.-2.*reverse);
    return color+noise*(1.-pow(reverse+luminance,4.))*grainAmount*.1;
}

vec3 ddBloom(vec2 uv){
    float aspect=iResolution.x/iResolution.y;
    vec2 stepUv=.2352941185/10.*vec2(1.,aspect);
    const float kernel[7]=float[7](1.,6.,15.,20.,15.,6.,1.);
    vec3 total=vec3(0.);
    for(int y=-3;y<=3;y++) for(int x=-3;x<=3;x++){
        float weight=kernel[x+3]*kernel[y+3];
        total+=max(textureLod(iChannel0,uv+vec2(float(x),float(y))*stepUv,5.).rgb,vec3(0.))*weight;
    }
    return total/4096.;
}

void mainImage(out vec4 fragColor, in vec2 fragCoord){
    vec2 uv=fragCoord/iResolution.xy;
    DDPost post=ddPostAt(iTime);
    vec3 linear=ddChromatic(uv).rgb+ddBloom(uv)*post.bloomTint;
    vec3 color=ddGrain(ddAces(linear),uv,0.);
    fragColor=vec4(ddToSrgb(ddGradeCurve(color)),1.);
}

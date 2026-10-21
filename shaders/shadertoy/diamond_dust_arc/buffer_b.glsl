// title: Diamond Dust — Buffer B (foco 1)
// url: https://www.shadertoy.com/view/ffKXRw (derivado; só o Buffer A muda)
// author: Tonny Espeset (2026)
// license: livre com crédito ("Feel free to use this code, but please keep this credit and link.")
// iChannel0: buffer_a
// notes: profundidade de campo (bokeh 8x8) sobre a cena

/*

Created by Tonny Espeset, 2026
https://www.shadertoy.com/view/ffKXRw

Diamond Dust

Feel free to use this code, but please keep this credit and link.

Check my other shaders at

https://www.shadertoy.com/user/Espeset

*/

const float DD_PI_DOF=3.14152965;

float ddCoc(float encodedDepth,DDPost post){
    float focus=post.focusControl*10.,aperture=post.apertureControl/10.,distance=ddDepthDistance(encodedDepth);
    float safeDistance=abs(distance)<1e-8?(distance<0.?-1e-8:1e-8):distance;
    float denominator=focus-aperture;
    denominator=abs(denominator)<1e-8?(denominator<0.?-1e-8:1e-8):denominator;
    return clamp(((post.strengthControl/5.)*abs(distance-focus)/safeDistance*(aperture/denominator))/.024,.0001,.12)*.3;
}

float ddCocTexel(ivec2 coordinate,DDPost post){
    ivec2 size=textureSize(iChannel0,0);
    return ddCoc(texelFetch(iChannel0,clamp(coordinate,ivec2(0),size-1),0).a,post);
}

float ddFilteredCoc(vec2 uv,DDPost post){
    vec2 pixel=uv*vec2(textureSize(iChannel0,0))-.5,f=fract(pixel);
    ivec2 p=ivec2(floor(pixel));
    return mix(mix(ddCocTexel(p,post),ddCocTexel(p+ivec2(1,0),post),f.x),mix(ddCocTexel(p+ivec2(0,1),post),ddCocTexel(p+ivec2(1),post),f.x),f.y);
}

vec2 ddDisk(vec2 coordinate){
    float x=2.*coordinate.x-1.,y=2.*coordinate.y-1.,radius=y,angle;
    if(x*x>y*y){radius=x;angle=DD_PI_DOF*.25*y/x;}
    else angle=DD_PI_DOF*.25*x/y+DD_PI_DOF*.5;
    return vec2(cos(angle)/(iResolution.x/iResolution.y),sin(angle))*radius;
}

void mainImage(out vec4 fragColor,in vec2 fragCoord){
    vec2 uv=fragCoord/iResolution.xy;
    DDPost post=ddPostAt(iTime);
    vec4 total=vec4(0.);
    float centerCoc=ddFilteredCoc(uv,post),centerDistance=ddSceneDistance(iChannel0,uv),totalWeight=0.;
    for(int row=0;row<8;row++)for(int column=0;column<8;column++){
        vec2 coordinate=uv+ddDisk((vec2(float(column),float(row))+.5)/8.)*centerCoc;
        if(any(lessThan(coordinate,vec2(0.)))||any(greaterThan(coordinate,vec2(1.))))continue;
        float neighbourCoc=ddFilteredCoc(coordinate,post);
        vec4 neighbour=vec4(textureLod(iChannel0,coordinate,.9).rgb,neighbourCoc);
        float weight=ddDofWeight(iChannel0,coordinate,centerCoc,centerDistance,neighbourCoc);
        total+=neighbour*weight;totalWeight+=weight;
    }
    fragColor=total/max(totalWeight,1e-8);
}

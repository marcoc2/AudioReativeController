// title: Diamond Dust — Buffer A (cena), padrão movido pela textura do som
// url: https://www.shadertoy.com/view/ffKXRw (derivado)
// author: Tonny Espeset (original, 2026); mudanças: ARC
// license: livre com crédito ("Feel free to use this code, but please keep this credit and link.")
// notes: chão 10x10 com padrão procedural + flocos quadrados girando (grade 512x512, até 8 por célula), PBR GGX; alfa = profundidade.
//        ARC: botões no padrão e nos flocos (0 = o original): u_warp (torção do padrão), u_rough
//        (aspereza do FBM), u_flow (o padrão escorre), u_breath (o padrão amplia/encolhe),
//        u_ripple + u_trem_phase (ondulação no ritmo do tremolo), u_glint (brilho dos flocos),
//        u_spin (giro extra dos flocos, em segundos de relógio)

/*

Created by Tonny Espeset, 2026
https://www.shadertoy.com/view/ffKXRw

Diamond Dust

Feel free to use this code, but please keep this credit and link.

Check my other shaders at

https://www.shadertoy.com/user/Espeset

*/

vec3 ddQ16(vec3 x){return floor(clamp(x,0.,1.)*65535.+.5)/65535.;}

// ARC: the sound's texture moves the pattern (all 0: the original)
uniform float u_warp, u_rough, u_flow, u_breath, u_ripple, u_trem_phase, u_glint, u_spin;

const uint DD_M0[16]=uint[16](13583506u,4302259u,9375848u,3760279u,1293914u,12796088u,7621060u,4619105u,11796310u,14267733u,16736504u,3913519u,12221407u,7709230u,4784939u,7637972u);
const uint DD_M1[64]=uint[64](894534u,3691408u,16212986u,12289726u,3917870u,3482626u,10361558u,12896808u,11163277u,11311954u,10534552u,4022337u,15249849u,3984717u,14332226u,15388633u,5467076u,8041620u,1425578u,8748189u,11736132u,6114724u,3925316u,9649554u,5913604u,13814163u,14262870u,13382240u,735953u,13127685u,8811692u,3420810u,1338139u,5388781u,1619973u,10020941u,10414472u,2942636u,6091787u,2632613u,4851227u,6131202u,13303864u,13312501u,1750933u,12438069u,6922519u,7844630u,9773712u,266405u,8337675u,13932559u,747613u,12470608u,8688788u,5080459u,13020988u,15701960u,3041234u,14720637u,10956200u,9029648u,8267144u,327074u);
const uint DD_M2[256]=uint[256](13238908u,12819089u,2679431u,11320207u,1302547u,13007019u,14407461u,15461898u,15634816u,8026035u,4338339u,11418224u,8292762u,1608318u,14529306u,11299905u,13783475u,1487149u,1094910u,9764723u,5098967u,12899896u,107080u,3944117u,4491016u,2842420u,10928345u,10373085u,14888901u,6277901u,4101801u,13255072u,13978039u,15143947u,2566582u,8465729u,12115505u,10588789u,8268871u,13922544u,4823278u,1878543u,14386713u,14572689u,7322779u,5879803u,850671u,6120017u,14832283u,1324043u,2994778u,15164014u,15800098u,9800194u,4599098u,13831190u,14817460u,9688078u,14331876u,10398198u,9362428u,14888114u,3889152u,5834822u,9885689u,12423033u,1174646u,1441437u,8982420u,11550771u,12697550u,2448049u,14542659u,6883581u,6422230u,12384131u,9903746u,13595181u,11207565u,6395198u,13965172u,7860313u,10889733u,15977841u,529931u,16575171u,14944129u,13656860u,5675516u,12560490u,11583500u,9473077u,6626734u,7662167u,14839327u,5343024u,7590162u,8033728u,13050257u,12332392u,12866415u,13464807u,15697172u,8068479u,845697u,4782132u,12364804u,9466382u,3363300u,2727828u,9395464u,1394758u,10772564u,1129273u,14859660u,11359992u,6503512u,8831185u,6000763u,14271530u,7967314u,2632411u,2301180u,12215421u,7673675u,5552956u,8684604u,14485533u,10877370u,15471718u,6820994u,15642379u,13432377u,10652720u,10106172u,12941269u,16051525u,12528637u,5722158u,11974330u,16280350u,13591573u,15186626u,4192542u,5486653u,3995794u,3086677u,13879450u,4212491u,2394298u,9128688u,10905238u,13175085u,12646739u,11248647u,14947086u,7549910u,16243124u,1488091u,14802982u,7267909u,9667712u,9959208u,1255388u,13021473u,13050734u,6085616u,6371322u,15801500u,8359497u,10936887u,12215352u,11975705u,1062737u,8720686u,16740681u,3045851u,4570589u,12421992u,1072558u,10885749u,7798146u,15749240u,12847081u,6859354u,15705309u,12551389u,15008893u,9316938u,4039068u,2708272u,16504275u,11701989u,3974605u,6862336u,5358565u,8889880u,10377319u,6390285u,9342826u,8989395u,4284862u,13936994u,1227106u,2326787u,9311452u,34559u,7369169u,6150497u,22388u,16120188u,14764340u,16489766u,14798584u,12156622u,6362027u,16524238u,6734998u,2109640u,1516572u,8132493u,13318614u,6150137u,906231u,1937172u,7260034u,3343202u,15706879u,13232103u,11868066u,13817169u,9278774u,9040873u,11633662u,83726u,1550819u,15563212u,15148194u,12113533u,3470612u,3036977u,15540128u,6181362u,5750274u,15683096u,9217750u,8749927u,5264526u,9828054u,11147378u,11101611u,14587791u,241289u,14048590u,13390254u,5729833u);
const uint DD_W0[16]=uint[16](25u,83u,24u,159u,253u,90u,198u,86u,247u,223u,131u,10u,16u,252u,132u,146u);

vec3 ddBytes(uint v){return vec3(v&255u,(v>>8u)&255u,(v>>16u)&255u)/255.;}

vec2 ddPatternUv(DDStyle s,vec2 xz){
    vec2 uv=vec2(xz.x*.1+.5,.5-xz.y*.1),q=uv-.5;
    if(s.lemniscate>.5){
        q-=vec2(0.,.14);
        return vec2(.245,.615)+vec2(q.x*q.x-q.y*q.y-.0169,2.*q.x*q.y)*vec2(1.18,3.4);
    }
    float root=sqrt(s.aspect),co=cos(s.angle),si=sin(s.angle);
    q*=vec2(s.scale*root,s.scale/root);
    q=vec2(co*q.x-si*q.y,si*q.x+co*q.y);
    return q+.5+s.offset;
}

ivec2 ddGraphWrap(ivec2 p,int period){
    return p&ivec2(period-1);
}

uint ddGraphSeed(DDStyle s,uint stream,uint block){
    uint familySeed=stream==0u?8u:(stream==1u?34u:2u);
    uint sourceBlock=stream==2u?block+9u:block;
    if(s.id>.5) familySeed=ddHash(familySeed^uint(s.id)*0x9e3779b9u);
    return ddHash(familySeed*0x85ebca6bu^sourceBlock*0xc2b2ae35u^stream*0x27d4eb2fu);
}

uvec3 ddHash3(uvec3 v){
    v^=uvec3(0xa3c59ac3u);v*=uvec3(0x9e3779b9u);
    v^=v>>uvec3(16u);v*=uvec3(0x9e3779b9u);
    v^=v>>uvec3(16u);v*=uvec3(0x9e3779b9u);
    return v;
}

vec3 ddRandom3(uint seed,uint lane){return vec3(ddHash3(uvec3(seed)+uvec3(lane,lane+1u,lane+2u)))*(1./4294967296.);}

vec3 ddGraphTexel(ivec2 p,uint salt){
    uint h=ddCellHash(p)^salt;
    return ddRandom3(h,0u);
}

vec3 ddGraphNoise(vec2 p,float frequency,uint salt){
    vec2 q=p*frequency,f=fract(q);f=f*f*(3.-2.*f);
    ivec2 i=ivec2(floor(q));int period=int(frequency);i=ddGraphWrap(i,period);
    ivec2 n=(i+1)&ivec2(period-1);
    vec3 a=ddGraphTexel(i,salt),b=ddGraphTexel(ivec2(n.x,i.y),salt);
    vec3 c=ddGraphTexel(ivec2(i.x,n.y),salt),d=ddGraphTexel(n,salt);
    return mix(mix(a,b,f.x),mix(c,d,f.x),f.y);
}

vec3 ddOpeningTexel(ivec2 p,int octave){int period=4<<octave,index=p.y*period+p.x;return ddBytes(octave==0?DD_M0[index]:(octave==1?DD_M1[index]:DD_M2[index]));}

vec3 ddOpeningNoise(vec2 p,int octave){
    int period=4<<octave;vec2 q=p*float(period),f=fract(q);f=f*f*(3.-2.*f);ivec2 i=ddGraphWrap(ivec2(floor(q)),period);
    ivec2 n=(i+1)&ivec2(period-1);
    vec3 A=ddOpeningTexel(i,octave),B=ddOpeningTexel(ivec2(n.x,i.y),octave),C=ddOpeningTexel(ivec2(i.x,n.y),octave),D=ddOpeningTexel(n,octave);
    return mix(mix(A,B,f.x),mix(C,D,f.x),f.y);
}

vec3 ddGraphFbm(vec2 p,DDStyle s,float persistence){
    vec3 value=vec3(.5);float frequency=4.,weight=1.;
    for(int octave=0;octave<11;octave++){
        vec3 noise=s.id<.5&&octave<3?ddOpeningNoise(p,octave):ddGraphNoise(p,frequency,ddGraphSeed(s,0u,uint(octave)));
        value=ddQ16(octave==0?noise:value+(noise-.5)*weight);
        frequency*=2.;weight*=persistence;
    }
    return value;
}

vec3 ddGraphCubic(vec3 p0,vec3 p1,vec3 p2,vec3 p3,float amount){
    vec3 slope=p3-p2-p0+p1;
    return ((slope*amount+(p0-p1-slope))*amount+(p2-p0))*amount+p1;
}

vec3 ddGraphWarpRow(vec2 p,DDStyle s,int row){
    vec2 q=p*4.;ivec2 i=ivec2(floor(q))+ivec2(0,row);float amount=fract(q.x);
    uint salt=ddGraphSeed(s,1u,0u);
    ivec2 a=ddGraphWrap(i,4),b=ivec2((a.x+1)&3,a.y),c=ivec2((a.x+2)&3,a.y),d=ivec2((a.x+3)&3,a.y);
    vec3 A,B,C,D;
    if(s.id<.5){A=ddBytes(DD_W0[a.y*4+a.x]);B=ddBytes(DD_W0[b.y*4+b.x]);C=ddBytes(DD_W0[c.y*4+c.x]);D=ddBytes(DD_W0[d.y*4+d.x]);}
    else{A=ddGraphTexel(a,salt);B=ddGraphTexel(b,salt);C=ddGraphTexel(c,salt);D=ddGraphTexel(d,salt);}
    return ddQ16(ddGraphCubic(A,B,C,D,amount));
}

vec3 ddGraphWarp(vec2 p,DDStyle s){
    float amount=fract(p.y*4.);
    return ddQ16(ddGraphCubic(ddGraphWarpRow(p,s,0),ddGraphWarpRow(p,s,1),ddGraphWarpRow(p,s,2),ddGraphWarpRow(p,s,3),amount));
}

vec3 ddGraphRgbToHsv(vec3 c){
    float value=max(c.r,max(c.g,c.b)),low=min(c.r,min(c.g,c.b));
    float delta=value-low;
    float saturation=value!=0.?delta/value:0.;
    vec3 dist=value-c;
    float hue=(dist.z-dist.y)/delta;
    if(c.g==value)hue=(dist.x-dist.z)/delta+2.;
    if(c.b==value)hue=(dist.y-dist.x)/delta+4.;
    if(delta==0.)hue=-1.;
    return vec3(clamp(hue/6.,0.,1.),saturation,value);
}

float ddGraphBezier(float t,float a,float b,float c,float d){
    float u=1.-t;
    return u*u*u*a+3.*u*u*t*b+3.*u*t*t*c+t*t*t*d;
}

float ddGraphCurveSegment(float x,float x0,float x1,float x2,float x3,float y0,float y1,float y2,float y3){
    float lo=0.,hi=1.,t=.5;
    for(int i=0;i<4;i++){t=(lo+hi)*.5;if(ddGraphBezier(t,x0,x1,x2,x3)<x)lo=t;else hi=t;}
    for(int i=0;i<10;i++){
        float u=1.-t;
        float derivative=3.*u*u*(x1-x0)+6.*u*t*(x2-x1)+3.*t*t*(x3-x2);
        t-=(ddGraphBezier(t,x0,x1,x2,x3)-x)/derivative;
    }
    return ddGraphBezier(t,y0,y1,y2,y3);
}

float ddGraphBaseValue(float x,float first,float middle,float last){
    if(x<=0.)return first;if(x>=1.)return last;
    if(x<.5625)return ddGraphCurveSegment(x,0.,42./255.,.5625-16./255.,.5625,first,first,middle+.033416748046875,middle);
    return ddGraphCurveSegment(x,.5625,.5625+16./255.,1.-28./255.,1.,middle,middle-.033416748046875,last-.0267486572265625,last);
}

vec3 ddGraphBaseCurve(DDStyle s,vec3 rgb){
    vec3 hsv=ddGraphRgbToHsv(rgb);float anchor,first,middle,last;
    if(s.family<.5){hsv.x=mix(1.125,1.0380859375,hsv.x);hsv.y*=.82421875;first=.625;middle=.0758056640625;last=.0712890625;anchor=1.08154296875;}
    else if(s.family<1.5){hsv.x=mix(.375,.625,hsv.x);hsv.y*=.66845703125;first=.58251953125;middle=.0758056640625;last=.0625;anchor=.5;}
    else{hsv.x=mix(.11553955078125,.83642578125,hsv.x);hsv.y=0.;first=.625;middle=.0758056640625;last=.0712890625;anchor=.475982666015625;}
    hsv.z=ddGraphBaseValue(hsv.z,first,middle,last);
    if(s.id>.5){hsv.x+=s.hue-anchor;hsv.y=clamp(hsv.y*s.saturation+.08*s.saturation,0.,1.);}
    return ddQ16(ddHsv(hsv));
}

vec3 ddGraphEmitterCurve(DDStyle s,vec3 rgb,int layer){
    vec3 hsv=ddGraphRgbToHsv(rgb);
    float threshold=layer==0?.01953125:.04296875;
    float peak=layer==0?2.3515625:1.2001953125;
    if(s.family>.5&&s.family<1.5){threshold=layer==0?.01953125:.05078125;peak=layer==0?1.3681640625:1.2001953125;}
    peak*=max(0.,1.+u_glint);                                      // ARC: the flakes glint with the loudness
    hsv.z=max(hsv.z-threshold,0.)*peak/(1.-threshold);
    if(s.id>.5)hsv.x=ddHueMix(hsv.x,layer==0?s.hue:s.accentHue,layer==0?.25:.45);
    return ddQ16(ddHsv(hsv));
}

vec3 ddStyleBase(DDStyle s,vec2 xz){
    // ARC: the pattern breathes (u_breath) and drifts (u_flow) over the ground
    vec2 p=fract((ddPatternUv(s,xz)-.5)*(1.+u_breath)+.5+vec2(u_flow,u_flow*.37));
    float persistence=clamp((s.id<.5?164./255.:s.persistence)+u_rough,.3,.85);   // ARC: rougher when noisy
    vec2 repeat=s.family<.5?vec2(3.,1.):(s.family<1.5?vec2(6.,1.):vec2(3.,1.));
    vec3 warp=ddQ16(ddGraphWarp(fract(p*repeat),s));
    float angle=warp.r*6.28+u_ripple*sin(u_trem_phase),amount=(s.id<.5?78./255.:s.warp)*max(0.,1.+u_warp);   // ARC: twists
    vec3 distorted=ddQ16(ddGraphFbm(p+vec2(cos(angle),sin(angle))*amount,s,persistence));
    vec3 curved=ddGraphBaseCurve(s,distorted);
    float detailPersistence=s.id<.5?102./255.:mix(.3,.52,ddStyleRandom(s.id,14u));
    vec3 detail=ddQ16(ddGraphNoise(p,2048.,ddGraphSeed(s,2u,0u)));
    detail=ddQ16(detail+(ddGraphNoise(p,4096.,ddGraphSeed(s,2u,1u))-.5)*detailPersistence);
    float contrast=s.id<.5?21./255.:mix(17.,25.,ddStyleRandom(s.id,15u))/255.;
    detail=ddQ16(mix(vec3(.5),detail,contrast*contrast*45.));
    vec3 base=ddQ16(mix(1.-2.*(1.-curved)*(1.-detail),2.*curved*detail,lessThan(curved,vec3(.5))));
    return base;
}

float ddLocalMorph(vec2 xz,vec2 morphOffset,float transition){
    if(transition<=0.)return 0.;
    float envelope=4.*transition*(1.-transition);
    float organic=ddNoise(xz*.38+morphOffset+vec2(transition,-transition)*.35,0x45d9f3bu)-.5;
    return ddQuintic(transition+organic*.46*envelope);
}

// Base pattern (layer<0) or emitter pigment of one point, current and next style through one copy of ddStyleBase; the trip count depends on the transition so the compiler keeps the loop rolled.
vec3 ddSurface(vec2 xz,float time,int layer,float local){
    float id,t;ddStylePhase(time,id,t);
    vec3 v0=vec3(0.),v1=vec3(0.);vec2 morphOffset=vec2(0.);
    int count=t<=0.?1:2;
    for(int k=0;k<count;k++){
        DDStyle s=ddMakeStyle(id+float(k));
        vec3 value=ddStyleBase(s,xz);
        if(layer>=0)value=ddGraphEmitterCurve(s,value,layer);
        if(k==0)v0=value;else{v1=value;morphOffset=s.morphOffset;}
    }
    if(t<=0.)return v0;
    if(layer<0)local=ddLocalMorph(xz,morphOffset,t);
    return mix(v0,v1,local);
}

float ddEdgeFade(vec2 xz){return smoothstep(0.,.6,DD_HALF-max(abs(xz.x),abs(xz.y)));}

mat3 ddRotation(float angle,vec3 axis){
    float c=cos(angle),s=sin(angle),t=1.-c;
    vec3 n=normalize(axis);
    return mat3(t*n.x*n.x+c,t*n.x*n.y+s*n.z,t*n.x*n.z-s*n.y,
                t*n.x*n.y-s*n.z,t*n.y*n.y+c,t*n.y*n.z+s*n.x,
                t*n.x*n.z+s*n.y,t*n.y*n.z-s*n.x,t*n.z*n.z+c);
}

vec3 ddFresnel(float cosine,vec3 f0){return f0+(1.-f0)*pow(max(1.-cosine,0.),5.);}

float ddGgx(float NoH,float roughness){
    float a2=roughness*roughness,d=NoH*NoH*(a2-1.)+1.;
    return a2/(DD_PI*d*d);
}

float ddSmith(float NoV,float NoL,float roughness){
    float a2=roughness*roughness;
    float gv=NoL*sqrt(NoV*NoV*(1.-a2)+a2);
    float gl=NoV*sqrt(NoL*NoL*(1.-a2)+a2);
    return .5/max(gv+gl,.0001);
}

const float DD_BRDF_LOW[32]=float[32](.963485579,.927107471,.900337964,.883154088,.874827313,.872618822,.874337569,.878454942,.884102723,.890457196,.897530086,.904364995,.911172523,.917560591,.92333964,.928835802,.933787648,.93809043,.942026013,.945628572,.94875777,.951823748,.954590956,.956919742,.959003824,.960959041,.962808697,.964451634,.965854815,.967271737,.968507921,.969594822);
const float DD_BRDF_HIGH[32]=float[32](.935543537,.856176674,.797831237,.749926031,.709445238,.674226046,.643178701,.61544776,.59042716,.567594469,.546785951,.527685225,.509999931,.49359557,.478318483,.464052081,.45069164,.438145876,.426344514,.415208369,.40469411,.394740194,.385292202,.37631771,.367791981,.359659016,.351913631,.344512403,.337443024,.330670536,.324181944,.317962229);

float ddBrdfRow(float NoV,bool high){
    float x=clamp(NoV*32.-.5,0.,31.);int a=int(floor(x)),b=min(a+1,31);
    return mix(high?DD_BRDF_HIGH[a]:DD_BRDF_LOW[a],high?DD_BRDF_HIGH[b]:DD_BRDF_LOW[b],fract(x));
}

float ddSplitEnergy(float NoV,float roughness){
    const float low=(100./255.)*(100./255.);
    return mix(ddBrdfRow(NoV,false),ddBrdfRow(NoV,true),clamp((roughness-low)/(1.-low),0.,1.));
}

vec3 ddLight(vec3 lightDirection,vec3 lightColour,vec3 viewDirection,vec3 normal,vec3 albedo,float roughness){
    vec3 halfDirection=normalize(lightDirection+viewDirection);
    float VoH=dot(halfDirection,viewDirection);
    float NoL=clamp(dot(normal,lightDirection),0.,1.);
    float NoV=clamp(dot(normal,viewDirection),0.,1.);
    float NoH=dot(normal,halfDirection);
    vec3 f=ddFresnel(VoH,albedo);
    float energy=ddSplitEnergy(NoV,roughness);
    vec3 multiple=f*(1.+albedo*(1.-energy)/energy);
    return multiple*ddGgx(NoH,roughness)*ddSmith(NoV,NoL,roughness)*NoL*lightColour;
}

vec3 ddLayerTint(DDStyle s,int layer,vec3 randomTint){
    vec3 base,spread;
    if(s.family<.5){base=vec3(1.);spread=layer==0?vec3(0.):vec3(8.);}
    else if(s.family<1.5){base=layer==0?vec3(3.):vec3(6.,2.353515625,.49853515625);spread=layer==0?vec3(0.):vec3(8.,5.98828125,12.);}
    else{base=layer==0?vec3(1.):vec3(2.);spread=layer==0?vec3(0.):vec3(2.337890625,4.78515625,6.);}
    if(s.id>.5){
        vec3 target=ddHsv(vec3(layer==0?s.hue:s.accentHue,layer==0?mix(.28,.62,ddStyleRandom(s.id,46u)):mix(.4,.82,ddStyleRandom(s.id,47u)),1.));
        float amount=layer==0?.1:.2,mean=(base.r+base.g+base.b)/3.,targetMean=(target.r+target.g+target.b)/3.,exposure=mix(layer==0?.97:.94,layer==0?1.03:1.06,ddStyleRandom(s.id,uint(48+layer)));
        base=mean*mix(base/max(mean,1e-4),target/max(targetMean,1e-4),amount)*exposure;
        amount=layer==0?.1:.28;mean=(spread.r+spread.g+spread.b)/3.;
        if(mean>1e-4)spread=mean*mix(spread/mean,target/max(targetMean,1e-4),amount)*exposure;
    }
    return base+randomTint*spread;
}

struct DDHit{
    float t;
    float limit;
    vec3 normal;
    vec3 centre;
    uint seed;
    int layer;
};

void ddParticle(ivec2 cell,uint cellHash,int slot,int layer,vec3 ro,vec3 rd,vec4 currentGeom,vec4 nextGeom,vec2 morphOffset,float transition,inout DDHit hit){
    uint seed=ddHash(cellHash^uint(slot)*0x9e3779b9u^uint(layer+1)*0x85ebca6bu);
    if(ddRandom(seed,31u)>=.5)return;
    vec2 centreXZ=-vec2(DD_HALF)+(vec2(cell)+vec2(ddRandom(seed,0u),ddRandom(seed,2u)))*DD_CELL;
    float local=ddLocalMorph(centreXZ,morphOffset,transition);
    vec4 geom=mix(currentGeom,nextGeom,local);
    vec3 centre=vec3(centreXZ.x,(ddRandom(seed,1u)-.5)*geom.x+geom.y,centreXZ.y);
    if(ddRandom(seed,22u)>=ddEdgeFade(centre.xz))return;
    float size=ddRandom(seed,15u)*geom.w+geom.z,halfSize=size*.1;
    vec3 axis=ddRandom3(seed,19u);
    vec3 spinv=ddLayerSpin(layer);
    float spin=(ddRandom(seed,17u)*spinv.y+spinv.x)*.5;
    if(ddRandom(seed,18u)<.5)spin=-spin;
    float angle=(ddRandom(seed,16u)*2.-1.)*spinv.z*(4.*DD_PI);
    angle+=spin*(84.+(iTime+u_spin)*30.-43.5202143);             // ARC: u_spin whirls them more
    mat3 rotation=ddRotation(mod(angle,DD_TAU),axis);
    vec3 n=rotation[2];
    float denominator=dot(n,rd);
    if(abs(denominator)<1e-9)return;
    float distance=dot(n,centre-ro)/denominator;
    if(distance<=0.||distance>=hit.limit)return;
    vec3 localPoint=ro+distance*rd-centre;
    vec2 square=vec2(dot(localPoint,rotation[0]),dot(localPoint,rotation[1]))/halfSize;
    if(any(greaterThan(abs(square),vec2(1.))))return;
    if(hit.layer>=0&&distance>=hit.t)return;
    if(dot(normalize(centre-ro),n)>0.)n=-n;
    hit.t=distance;hit.normal=normalize(n);hit.centre=centre;hit.seed=seed;hit.layer=layer;
}

void ddTraceParticles(vec3 ro,vec3 rd,DDStyle current,DDStyle next,float transition,inout DDHit hit){
    if(abs(rd.y)<1e-8)return;
    float tHigh=(.08-ro.y)/rd.y,tLow=(-.02-ro.y)/rd.y;
    float enter=max(min(tHigh,tLow),0.),leave=min(max(tHigh,tLow),hit.limit);
    if(leave<=enter)return;
    vec4 currentGeom0=ddLayerGeom(current,0),currentGeom1=ddLayerGeom(current,1),nextGeom0=ddLayerGeom(next,0),nextGeom1=ddLayerGeom(next,1);
    vec2 p0=(ro.xz+rd.xz*enter+vec2(DD_HALF))/DD_CELL;
    ivec2 cell=ivec2(floor(p0));
    vec2 direction=sign(rd.xz);
    vec2 safeRay=vec2(abs(rd.x)<1e-9?(rd.x<0.?-1e-9:1e-9):rd.x,
                      abs(rd.z)<1e-9?(rd.z<0.?-1e-9:1e-9):rd.z);
    vec2 delta=abs(vec2(DD_CELL)/safeRay);
    vec2 nextDistance=(vec2(cell)+max(direction,vec2(0.))-p0)*DD_CELL/safeRay;
    nextDistance=mix(nextDistance,vec2(1e30),equal(direction,vec2(0.)));
    int stepCount=min(48,max(0,int(iResolution.x))),particleCount=min(8,max(0,int(iResolution.y)));
    for(int stepIndex=0;stepIndex<stepCount;stepIndex++){
        if(cell.x<0||cell.y<0||cell.x>=512||cell.y>=512)break;
        uint cellHash=ddCellHash(cell);
        for(int particle=0;particle<particleCount;particle++)
            ddParticle(cell,cellHash,particle&3,particle>>2,ro,rd,particle<4?currentGeom0:currentGeom1,particle<4?nextGeom0:nextGeom1,next.morphOffset,transition,hit);
        float boundary=enter+min(nextDistance.x,nextDistance.y);
        if(boundary>=leave||(hit.layer>=0&&hit.t<=boundary))break;
        if(nextDistance.x<nextDistance.y){nextDistance.x+=delta.x;cell.x+=int(direction.x);}
        else{nextDistance.y+=delta.y;cell.y+=int(direction.y);}
    }
}

vec3 ddRay(DDCamera camera,vec2 fragCoord){
    vec2 ndc=fragCoord/iResolution.xy*2.-1.;
    float projectionY=2.414213419,projectionX=projectionY/(iResolution.x/iResolution.y);
    return normalize(camera.right*(ndc.x/projectionX)+camera.up*(ndc.y/projectionY)+camera.forward);
}

float ddGroundHit(vec3 ro,vec3 rd){
    if(rd.y>=0.)return 1e30;
    float distance=-ro.y/rd.y;vec3 point=ro+distance*rd;
    return distance>0.&&abs(point.x)<=DD_HALF&&abs(point.z)<=DD_HALF?distance:1e30;
}

vec3 ddParticleLight(DDHit hit,vec3 ro,vec3 rd,DDStyle current,DDStyle next,float local,vec3 pigment){
    vec3 randomTint=ddRandom3(hit.seed,11u);
    vec3 tint=mix(ddLayerTint(current,hit.layer,randomTint),ddLayerTint(next,hit.layer,randomTint),local);
    vec3 albedo=(127./255.)*pigment*tint,point=ro+hit.t*rd,viewDirection=normalize(ro-point);
    float roughness=hit.layer==0?1.:100./255.;roughness*=roughness;
    float gainA=current.id<.5?1.:(current.family<.5?2.:(current.family<1.5?1.:1.2));
    float gainB=next.family<.5?2.:(next.family<1.5?1.:1.2),particleGain=mix(gainA,gainB,local);
    vec3 lit=ddLight(normalize(vec3(.50373936,.69598544,-.51169336)),vec3(10.),viewDirection,hit.normal,albedo,roughness);
    lit+=ddLight(normalize(vec3(.04815805,-.39421037,-.91776264)),vec3(4.5),viewDirection,hit.normal,albedo,roughness);
    return lit*particleGain;
}

void mainImage(out vec4 fragColor,in vec2 fragCoord){
    float styleId,transition;ddStylePhase(iTime,styleId,transition);
    DDStyle current=ddMakeStyle(styleId),next=ddMakeStyle(styleId+1.);
    DDCamera camera=ddCameraAt(iTime);vec3 ro=camera.eye,centerRay=ddRay(camera,fragCoord);
    float centerGround=ddGroundHit(ro,centerRay);vec3 colour=vec3(0.),sparkle=vec3(0.);
    DDHit centerHit;
    int sampleCount=min(5,max(0,int(iResolution.x)));
    // Sample 0 is the centre ray: its hit sets the depth and its ground point the base colour; samples 1-4 are the corner rays whose particle hits add sparkle. Every surface lookup goes through the single ddSurface call below.
    for(int sampleIndex=0;sampleIndex<sampleCount;sampleIndex++){
        bool center=sampleIndex==0;int corner=max(sampleIndex-1,0);
        vec2 offset=center?vec2(0.):(vec2(float(corner&1),float(corner>>1))-.5)*.5;
        vec3 rd=center?centerRay:ddRay(camera,fragCoord+offset);
        float groundDistance=center?centerGround:ddGroundHit(ro,rd);DDHit hit;
        hit.t=groundDistance;hit.limit=groundDistance;hit.normal=vec3(0.,1.,0.);hit.centre=vec3(0.);hit.seed=0u;hit.layer=-1;
        ddTraceParticles(ro,rd,current,next,transition,hit);
        if(center)centerHit=hit;
        bool ground=center&&centerGround<1e29,particle=!center&&hit.layer>=0;
        if(!(ground||particle))continue;
        vec2 xz=ground?(ro+centerGround*centerRay).xz:hit.centre.xz;
        float local=particle?ddLocalMorph(hit.centre.xz,next.morphOffset,transition):0.;
        vec3 surface=ddSurface(xz,iTime,ground?-1:hit.layer,local);
        if(ground)colour=surface*ddEdgeFade(xz);
        else sparkle+=ddParticleLight(hit,ro,rd,current,next,local,surface)*.25;
    }
    colour+=sparkle;float depth=centerHit.t;
    if(depth<1e29){vec3 point=ro+depth*centerRay;fragColor=vec4(colour,ddEncodeDepth(dot(point-ro,camera.forward)));}else fragColor=vec4(0.);
}

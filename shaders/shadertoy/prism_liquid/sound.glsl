// title: prism liquid — Sound (não roda no ARC)
// url: (preencher)
// author: (preencher)
// license: CC BY-NC-SA 3.0 (padrão do Shadertoy; confirmar)
// notes: a trilha do próprio shader (mainSound), guardada só como referência

float rd(float t) { return fract(sin(dot(floor(fract(t*0.05)*20.),84.259))*7846.236);}
float no (float t){return mix(rd(t),rd(t+1.),smoothstep(0.,1.,fract(t)));}
float it(float t){float r=0.;float a=0.5;for(int i =0; i<5;i++){
r +=no(t/a)*a;a*=0.5;
}return r;}
float hash(float x){return fract(sin(x) * 897612.531);}
float voc(float t, float f, float ft,float t2){float x = fract(t * f) / f;
float a=(sin(x*6.5*ft)*.4+sin(x*13.*ft)+sin(x*24.*ft)*.2);
   return a* min(x * 1000., 1.) * exp(x * -200.);}
vec2 inst2(float t, float var,float t2){
    vec2 v = vec2(0., 0.);
    for(int i = 0; i < 16; ++i){
        float h = float(i);
       	float m = voc(t + h / 3., 50. + pow(2.01, (h - 8.) * .2), var,t2);
        float pan = hash(h);
        v.x += m * pan;
        v.y += m * (1. - pan);
    }
    return v * .1;
}
vec2 sons ( float time) {float tt = time*1.1;
    float vrt = smoothstep(0.2,1.,pow(it(time*0.7),2.))*30.+1.;
    float bt = pow(fract(vrt/3.14),0.2);
    float v = sin(bt*50.)*it(time*24.);
    
    float v2 =  (fract(sin(dot(time,84.259))*7846.236)-0.5);
    float v3 = v2 * smoothstep(0.,1.,(1.-pow(fract(vrt/3.14),0.1))); 
    float v4 = sin(time*250.)*(smoothstep(1.,0.,sin(vrt)*0.5+0.5));
    float f1 = sin(time*200.+it(tt*0.1)*300.)*it(tt*0.3)  ;
    float va = 50.+50.*no(time*0.5);
    return vec2((v4+v*0.05+v3*0.05)+f1*0.5)+ inst2(time,va,time)*0.25;}
vec2 mainSound( in int samp, float time )
{
    
     float ta = 0.01;    
    vec2 rev = vec2(0.);
    float sum = 0.;
    for(float t = 0.; t<2.;t +=ta){      
    float rand = fract(120.*sin(t*1000.));
    float t2 = t + ta*rand*5.;
    float amp = exp2(-t2);
    rev += sons( time - t2 ) * amp;
    sum += amp;}
    rev /= sum;    
    rev *= 10.; 
    vec2 s2 = sons(time);
    vec2 f =  clamp(mix(vec2(s2.x,rev.x),vec2(rev.y,s2.y),it(time*18.)),-1.,1.);
    return clamp(f*1.5,0.,1.);
}
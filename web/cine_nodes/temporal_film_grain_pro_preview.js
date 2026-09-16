// SPDX-License-Identifier: GPL-3.0-or-later
// GPU preview shared by the compact and fullscreen Film Grain PRO monitors.
const VS=`
attribute vec2 a_position; varying vec2 v_uv;
void main(){v_uv=a_position*.5+.5;gl_Position=vec4(a_position,0.,1.);}
`;
const FS=`
precision highp float;
varying vec2 v_uv;
uniform sampler2D u_source,u_plate;
uniform vec2 u_resolution;
uniform float u_frame,u_seed,u_strength,u_size,u_softness,u_roughness,u_complexity,u_temporal,u_chroma;
uniform vec3 u_tone,u_rgb;
uniform int u_engine,u_transfer,u_blend,u_view,u_has_plate;
float hash21(vec2 p,float f,float s){vec3 p3=fract(vec3(p.xyx)*.1031+vec3(f*.013,s*.017,f*.019));p3+=dot(p3,p3.yzx+33.33);return (fract((p3.x+p3.y)*p3.z)*2.-1.)*1.7320508;}
float field(vec2 pixel,float f,float salt){
  float px=max(.35,u_size*max(u_resolution.x,u_resolution.y)/4096.);
  vec2 p=pixel/px; float total=0.; float norm=0.;
  for(int o=0;o<4;o++){if(float(o)>=u_complexity)break;float scale=pow(2.15,float(o));float weight=o==0?1.:(.16+u_roughness*.28)/float(o);vec2 cell=floor(p/scale);vec2 fracp=fract(p/scale);vec2 blendp=mix(step(vec2(.5),fracp),smoothstep(vec2(0.),vec2(1.),fracp),u_softness);
    float a=hash21(cell,f,u_seed+salt),b=hash21(cell+vec2(1.,0.),f,u_seed+salt),c=hash21(cell+vec2(0.,1.),f,u_seed+salt),d=hash21(cell+vec2(1.),f,u_seed+salt);
    total+=mix(mix(a,b,blendp.x),mix(c,d,blendp.x),blendp.y)*weight;norm+=weight;
  }return total/max(norm,.001);
}
vec3 decodeTransfer(vec3 c){if(u_transfer==2)return c;if(u_transfer==1){vec3 lo=c/4.5,hi=pow((c+.099)/1.099,vec3(1./.45));return mix(hi,lo,step(c,vec3(.081)));}vec3 lo=c/12.92,hi=pow((c+.055)/1.055,vec3(2.4));return mix(hi,lo,step(c,vec3(.04045)));}
vec3 encodeTransfer(vec3 c){c=max(c,vec3(0.));if(u_transfer==2)return c;if(u_transfer==1){vec3 lo=c*4.5,hi=1.099*pow(c,vec3(.45))-.099;return mix(hi,lo,step(c,vec3(.018)));}vec3 lo=c*12.92,hi=1.055*pow(c,vec3(1./2.4))-.055;return mix(hi,lo,step(c,vec3(.0031308)));}
float softLight(float b,float x){if(x<=.5)return b-(1.-2.*x)*b*(1.-b);float d=b<=.25?((16.*b-12.)*b+4.)*b:sqrt(b);return b+(2.*x-1.)*(d-b);}
void main(){
 vec4 src=texture2D(u_source,v_uv);if(u_view==2||(u_view==1&&v_uv.x<.5)){gl_FragColor=src;return;}
 float fresh=sqrt(max(0.,1.-u_temporal*u_temporal));float common=field(gl_FragCoord.xy,u_frame,0.)*fresh+field(gl_FragCoord.xy,max(0.,u_frame-1.),0.)*u_temporal;
 if(u_engine==2&&u_has_plate==1){vec3 p=texture2D(u_plate,fract(v_uv+vec2(u_frame*.0037,u_frame*.0053))).rgb;common=(dot(p,vec3(.3333))-.5)*3.;}
 vec3 independent=vec3(field(gl_FragCoord.xy,u_frame,17.),field(gl_FragCoord.xy,u_frame,29.),field(gl_FragCoord.xy,u_frame,43.));
 vec3 noise=mix(vec3(common),independent,u_chroma)*u_rgb;
 if(u_view==3){gl_FragColor=vec4(vec3(clamp(.5+common/6.,0.,1.)),1.);return;}
 vec3 linear=decodeTransfer(src.rgb);float luma=dot(linear,vec3(.2126,.7152,.0722));float sw=clamp((.5-luma)/.5,0.,1.),hw=clamp((luma-.5)/.5,0.,1.),mw=clamp(1.-sw-hw,0.,1.);float sigma=u_strength*.16*clamp(sw*u_tone.x+mw*u_tone.y+hw*u_tone.z,0.,2.);vec3 result;
 if(u_engine==1){vec3 shot=sqrt(max(linear,vec3(.0001)));result=encodeTransfer(linear+noise*sigma*(shot+vec3(.035+.12*u_roughness))*.42);}
 else if(u_blend==0){vec3 density=-log2(max(linear,vec3(.00001)));result=encodeTransfer(exp2(-(density-noise*sigma*.55)));}
 else if(u_blend==1)result=encodeTransfer(linear*exp(noise*sigma-.5*sigma*sigma));
 else if(u_blend==2)result=encodeTransfer(linear+noise*sigma*.32);
 else{vec3 b=clamp(vec3(.5)+noise*sigma*1.9,0.,1.);result=vec3(softLight(src.r,b.r),softLight(src.g,b.g),softLight(src.b,b.b));}
 gl_FragColor=vec4(clamp(result,0.,1.),src.a);
}`;
function shader(gl,type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw new Error(gl.getShaderInfoLog(s)||"Shader failed");return s;}
function dimensions(source){return [source?.videoWidth||source?.naturalWidth||source?.width||640,source?.videoHeight||source?.naturalHeight||source?.height||360];}
export function createProGrainRenderer(canvas,{maxDimension=2048}={}){
 const gl=canvas.getContext("webgl",{alpha:false,antialias:false,preserveDrawingBuffer:false});if(!gl)throw new Error("WebGL unavailable");
 const program=gl.createProgram();gl.attachShader(program,shader(gl,gl.VERTEX_SHADER,VS));gl.attachShader(program,shader(gl,gl.FRAGMENT_SHADER,FS));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))throw new Error(gl.getProgramInfoLog(program)||"WebGL link failed");gl.useProgram(program);
 const buffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,buffer);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array([-1,-1,1,-1,-1,1,-1,1,1,-1,1,1]),gl.STATIC_DRAW);const pos=gl.getAttribLocation(program,"a_position");gl.enableVertexAttribArray(pos);gl.vertexAttribPointer(pos,2,gl.FLOAT,false,0,0);
 const texture=unit=>{const t=gl.createTexture();gl.activeTexture(gl.TEXTURE0+unit);gl.bindTexture(gl.TEXTURE_2D,t);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.LINEAR);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_S,gl.CLAMP_TO_EDGE);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_WRAP_T,gl.CLAMP_TO_EDGE);return t;}, sourceTexture=texture(0),plateTexture=texture(1);
 gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,plateTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,1,1,0,gl.RGBA,gl.UNSIGNED_BYTE,new Uint8Array([128,128,128,255]));
 const loc={};["source","plate","resolution","frame","seed","strength","size","softness","roughness","complexity","temporal","chroma","tone","rgb","engine","transfer","blend","view","has_plate"].forEach(k=>loc[k]=gl.getUniformLocation(program,"u_"+k));gl.uniform1i(loc.source,0);gl.uniform1i(loc.plate,1);gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL,true);
 return {draw(source,plate,values={},view="grain",frame=0){if(!source)return;const [sw,sh]=dimensions(source),scale=Math.min(1,maxDimension/Math.max(sw,sh)),W=Math.max(2,Math.round(sw*scale)),H=Math.max(2,Math.round(sh*scale));if(canvas.width!==W||canvas.height!==H){canvas.width=W;canvas.height=H;}gl.viewport(0,0,W,H);gl.activeTexture(gl.TEXTURE0);gl.bindTexture(gl.TEXTURE_2D,sourceTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,source);let hasPlate=0;if(plate){gl.activeTexture(gl.TEXTURE1);gl.bindTexture(gl.TEXTURE_2D,plateTexture);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA,gl.RGBA,gl.UNSIGNED_BYTE,plate);hasPlate=1;}const f=(k,d=0)=>Number(values[k]??d);gl.uniform2f(loc.resolution,W,H);gl.uniform1f(loc.frame,frame+f("frame_start"));gl.uniform1f(loc.seed,f("seed",1));gl.uniform1f(loc.strength,f("strength",.1));gl.uniform1f(loc.size,f("grain_size_4k_px",1));gl.uniform1f(loc.softness,f("softness",.18));gl.uniform1f(loc.roughness,f("roughness",.24));gl.uniform1f(loc.complexity,f("complexity",3));gl.uniform1f(loc.temporal,f("temporal_correlation"));gl.uniform1f(loc.chroma,f("chroma_amount",.055));gl.uniform3f(loc.tone,f("shadow_response",.62),f("midtone_response",1),f("highlight_response",.38));gl.uniform3f(loc.rgb,f("red_response",1),f("green_response",.96),f("blue_response",1.06));gl.uniform1i(loc.engine,values.engine==="digital_sensor"?1:values.engine==="scanned_grain_plate"?2:0);gl.uniform1i(loc.transfer,values.input_transfer==="rec709"?1:values.input_transfer==="linear"?2:0);gl.uniform1i(loc.blend,values.blend_method==="density_exposure"?1:values.blend_method==="linear_additive"?2:values.blend_method==="soft_light_luma"?3:0);gl.uniform1i(loc.view,view==="split"?1:view==="original"?2:view==="map"?3:0);gl.uniform1i(loc.has_plate,hasPlate);gl.drawArrays(gl.TRIANGLES,0,6);},destroy(){gl.deleteTexture(sourceTexture);gl.deleteTexture(plateTexture);gl.deleteBuffer(buffer);gl.deleteProgram(program);}};
}

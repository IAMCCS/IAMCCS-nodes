const fs=require('node:fs'),path=require('node:path'),vm=require('node:vm'),assert=require('node:assert/strict');
const source=fs.readFileSync(path.join(__dirname,'../web/iamccs_h3_settings_pro_ui.js'),'utf8');
class Element {
  constructor(tag){this.tag=tag;this.style={};this.children=[];this.options=[];this.value='';}
  add(option){this.options.push(option);}
  append(...children){this.children.push(...children);}
  setAttribute(key,value){this[key]=value;}
  after(child){this.following=child;}
}
const classes=['IAMCCS_MiniMaxH3UniversalFastR42','IAMCCS_MiniMaxH3LTXPhaseBLoaderV3','IAMCCS_MiniMaxH3UniversalRTXFinalR42'];
const keys=['off','rtx_final','pixel_tiled_low_vram','h3_fast_latent_2pass','h3_pixel_refine','h3_ultimate_tiled','h3_latent_upres','ltx23','ltx23_per_chunk','wan22_5b'];
const values={width:960,height:576,image_width:960,image_height:576,upscale_width:1920,upscale_height:1152,upscale_enabled:true,upscale_mode:'h3_fast_latent_2pass',h3_upres_model_name:'minimax_h3_latent_upscaler_3d_fp16.safetensors',h3_upres_tile_width:1920,h3_upres_tile_height:1152};
const node={properties:{},widgets:Object.entries(values).map(([name,value])=>({name,value,options:{values:name==='upscale_mode'?keys:[]}}))};
const widget=(owner,key)=>owner.widgets.find(w=>w.name===key);
const get=key=>widget(node,key).value,set=(key,value)=>widget(node,key).value=value;
const recipes={'native-delivery':'off',upres:'h3_fast_latent_2pass','ultimate-tiled':'h3_ultimate_tiled','h3-pixel':'h3_pixel_refine','pixel-safe':'pixel_tiled_low_vram','ltx-master':'ltx23','ltx-per-shot':'ltx23_per_chunk','rtx-final':'rtx_final'};
const format=new Element('select');
let refreshes=0;
const context=vm.createContext({node,widget,setValue:(owner,key,value)=>{const w=widget(owner,key);if(w)w.value=value;},
  firstChoice:()=> 'minimax_h3_latent_upscaler_3d_fp16.safetensors',nodeClass:n=>n.type,shotboardMode:()=> 'v2va_face_swap_lora',
  choices:item=>item?.options?.values||[],connectedNodes:()=>classes.map(type=>({type})),
  applyRecipe:(owner,recipe)=>{set('upscale_mode',recipes[recipe]);set('upscale_enabled',recipe!=='native-delivery');},
  document:{createElement:tag=>new Element(tag),dispatchEvent(){}},CustomEvent:class{},
  Option:function(label,value){return {text:label,value};},format,refresh:()=>refreshes++,alert:message=>{throw Error(message);},
});
const constants=source.slice(source.indexOf('const H3_NATIVE_RESOLUTION_PRESETS ='),source.indexOf('const widget ='));
const branches=source.slice(source.indexOf('function branchGate('),source.indexOf('function modeAvailability('));
const gates=source.slice(source.indexOf('function deliveryBranchGate('),source.indexOf('function applyRecipe('));
vm.runInContext(constants+branches+gates+'globalThis.pairs=H3_FAST_2PASS_PAIRS;globalThis.choose=selectDeliveryMode;globalThis.normalize=normalizeDeliveryResolution;',context);
assert.equal(context.pairs.length,49);
for(const [,lw,lh,hw,hh] of context.pairs){assert.ok([lw,lh,hw,hh].every(n=>n%32===0));assert.ok(hw>=lw&&hh>=lh);}
assert.ok(context.pairs.some(([,lw,lh,hw,hh])=>lw===960&&lh===576&&hw===1920&&hh===1152));
assert.ok(context.pairs.some(([,lw,lh,hw,hh])=>lw===576&&lh===960&&hw===1152&&hh===1920));
assert.ok(context.pairs.some(([,lw,lh,hw,hh])=>lw===480&&lh===1152&&hw===800&&hh===1920));
function render(){
  const start=source.indexOf('      const selectedRoute = Boolean(',source.indexOf('    if (active === "overview" || active === "swap_lora")'));
  const end=source.indexOf('format.after(stage);',start)+'format.after(stage);'.length;
  assert.ok(start>0&&end>start);
  vm.runInContext('{'+source.slice(start,end)+'}',context);
  return format.following.children;
}
let [toggle,route,pair]=render();
assert.equal(toggle.children[0]['aria-label'],'Enable upscale');
assert.equal(route.options.length,10);
assert.equal(pair.options.length,49);
assert.ok(route.options.find(o=>o.value==='wan22_5b').disabled);
assert.ok(route.options.find(o=>o.value==='h3_latent_upres').disabled);
for(const mode of ['ltx23','ltx23_per_chunk','rtx_final','pixel_tiled_low_vram','h3_pixel_refine','h3_ultimate_tiled'])assert.equal(route.options.find(o=>o.value===mode).disabled,false,mode);
// Explicit linked pair changes both LOW and HIGH, with a direct high canvas for Fast.
pair.value='576x960->1152x1920';pair.onchange();
assert.equal(get('width'),576);assert.equal(get('height'),960);
assert.equal(get('upscale_width'),1152);assert.equal(get('upscale_height'),1920);
assert.equal(get('h3_upres_tile_width'),1152);assert.equal(get('h3_upres_tile_height'),1920);
// Switching to a different engine preserves the native generation canvas.
context.choose(node,'ltx23');assert.equal(get('width'),576);assert.equal(get('height'),960);
[toggle,route,pair]=render();
assert.ok(pair.options.every(o=>!o.value.includes('->')));
pair.value='1152x2048';pair.onchange();assert.equal(get('height'),960);assert.equal(get('upscale_height'),2048);
// Pixel/RTX permit exact FHD; latent targets use the stricter 32px grid.
context.choose(node,'rtx_final');set('upscale_width',1080);set('upscale_height',1920);context.normalize(node);
assert.equal(get('upscale_width'),1080);
context.choose(node,'ltx23');assert.equal(get('upscale_width'),1088);
context.choose(node,'h3_ultimate_tiled');assert.equal(get('upscale_mode'),'h3_ultimate_tiled');
context.choose(node,'off');assert.equal(get('upscale_enabled'),false);
[toggle,route,pair]=render();assert.equal(pair.disabled,true);
toggle.children[0].checked=true;toggle.children[0].onchange();assert.equal(get('upscale_mode'),'h3_ultimate_tiled');
assert.ok(refreshes>0);
console.log('PASS: 49 H/V/5:3/scope pairs, delivery engines, missing-branch gates, model grids, output isolation and toggle restore');

const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict'),path=require('node:path');
const ui=fs.readFileSync(path.join(__dirname,'../web/iamccs_h3_settings_pro_ui.js'),'utf8');
const slice=(s,a,b)=>s.slice(s.indexOf(a),s.indexOf(b,s.indexOf(a)));
const state={h3_omni_enabled:true};
const context=vm.createContext({widget:(_,name)=>({value:state[name]}),shotboardMode:()=> 'ref2va'});
vm.runInContext(slice(ui,'function omniActive(','function vedaAvailable('),context);
for(const name of ['steps','turbo_mode','pdd_lora_name','h3_veda_enabled','scheduler','shift_audio'])
  assert.equal(vm.runInContext(`omniLockedField({}, ${JSON.stringify(name)})`,context),true,name);
for(const name of ['width','duration_seconds','h3_clipproj_profile','upscale_enabled','secondary_lora_enabled'])
  assert.equal(vm.runInContext(`omniLockedField({}, ${JSON.stringify(name)})`,context),false,name);
for(const id of ['native','pdd','fasth3','sla','fused','turbo:model.safetensors'])
  assert.equal(vm.runInContext(`omniLockedRecipe({}, ${JSON.stringify(id)})`,context),true,id);
state.h3_omni_enabled=false;
assert.equal(vm.runInContext('omniLockedRecipe({},"turbo:model")',context),false);
const pr=fs.readFileSync(path.join(__dirname,'../web/iamccs_prompter_ui.js'),'utf8');
vm.runInContext(slice(pr,'function omniVoicePrompt(','function composePrompt('),context);
context.project={omni_voice_bindings:[{enabled:true,shot:2,picture:3,audio:1,character:'Knight'},{enabled:false,shot:1,picture:1,audio:2}]};
const prompt=vm.runInContext('omniRequestDirection(project)',context);
assert.match(prompt,/\[Shot 2\].*<Picture 3>.*<Audio 1>/);
assert.doesNotMatch(prompt,/<Audio 2>/);
assert.match(prompt,/six lowercase H3 sections/);
assert.match(ui,/if \(omniLockedRecipe\(node, id\)\) enabled = false/);
assert.match(ui,/if \(omniLockedField\(node, name\)\)/);
console.log('PASS: Omni locked controls and selected per-shot voice bindings');

class Element {
  constructor(){this.children=[];this.dataset={};this.classList={add(){}};}
  append(...items){this.children.push(...items);}
}
state.h3_omni_enabled=true;
Object.assign(context,{node:{},active:'assistant',document:{createElement:()=>new Element()},
  recipeGroup:()=> 'speed',rememberedRecipe:()=>'',selectedAcceleration:()=> 'omni',
  human:x=>x,choices:()=>[],branchGate:()=>({available:true}),integratedAttention:()=>false,
  vedaAvailable:()=>false});
vm.runInContext(slice(ui,'  function addRecipe(','  function renderAssistant('),context);
context.parent=new Element();
vm.runInContext('addRecipe(parent,"Turbo 4","turbo:model",true,false,()=>{throw Error("Disabled action ran")})',context);
assert.equal(context.parent.children[0].disabled,true);
context.parent.children[0].onclick();
vm.runInContext(slice(ui,'  function makeControl(','  function warnings('),context);
const field=vm.runInContext('makeControl("steps")',context);
assert.equal(field.children[1].disabled,true);
console.log('PASS: actual recipe button and sampling input render disabled');

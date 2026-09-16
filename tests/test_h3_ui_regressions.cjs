const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.join(__dirname, '..');
const read = name => fs.readFileSync(path.join(root, 'web', name), 'utf8');
const settings = read('iamccs_h3_settings_pro_ui.js');
const recipe = settings.slice(settings.indexOf('function applyRecipe('), settings.indexOf('function applyAccelerationChoice('));
for (const [name, steps, sampler] of [['native',20,'res_multistep'],['pdd',8,'euler'],['fasth3',6,'euler'],['sla',4,'euler'],['fused',4,'res_multistep']]) {
  const values = {turbo_mode:'old', pdd_lora_name:'old', fused_turbo_model_name:'minimax_fused_convrot.safetensors', shift_video:5, steps:99};
  const context = vm.createContext({node:{}, widget:(_, key)=>({value:values[key]}), setValue:(_,key,value)=>{values[key]=value;}, shotboardMode:()=> 'i2va', modeFamily:()=> 'fl2va', firstChoice:()=> 'asset', isPddLoRA:()=>true,isFastH3LoRA:()=>true,resetAcceleration:set=>{set('turbo_mode','off');set('turbo_lora_name','');set('pdd_lora_name','');set('fused_turbo_model_name','');}, document:{dispatchEvent(){}}, CustomEvent:class{}, alert:message=>{throw Error(message)}});
  vm.runInContext(recipe + `applyRecipe(node, '${name}')`, context);
  assert.equal(values.steps, steps, name); assert.equal(values.sampler_name, sampler, name);
  assert.equal(values.denoise, 1, name); assert.equal(values.shift_audio,3,name);
  assert.equal(values.shift_video,['native','sla'].includes(name)?6:12,name);
}
const declared = settings.slice(settings.indexOf('function applyDeclaredTurboContract('), settings.indexOf('function resetAcceleration('));
for (const steps of [3,4,8]) {
  let result;
  vm.runInNewContext(declared + 'applyDeclaredTurboContract({})', {widget:(_,key)=>({value:key==='turbo_mode'?'early_8_10':'native'}),declaredTurboSteps:()=>steps,setValue:(_,key,value)=>{result=value;}});
  assert.equal(result,steps);
}

const editor = read('iamccs_shotboard_video_editor_v1_ui.js');
const drag = editor.slice(editor.indexOf('  function startDrag('), editor.indexOf('  function startTrim('));
const clip = {id:'target',trackId:'V1',startTime:10,duration:3};
const linked = {id:'audio',trackId:'A1',startTime:11,duration:3};
const manifest = {clips:[{id:'left',trackId:'V1',startTime:5,duration:2},clip,{id:'right',trackId:'V1',startTime:15,duration:2},linked,{id:'audio-right',trackId:'A1',startTime:15,duration:2}]};
const handlers = {};
vm.runInNewContext(drag+'startDrag(event,clip)',{clip,event:{target:{},clientX:0,preventDefault(){},stopPropagation(){}},manifest,selectedClipId:null,selectedTrackId:null,trackIdForClip:c=>c.trackId,linkedClipsFor:()=>[linked],clientDeltaToTimelineCssPx:x=>x,pxPerSecond:()=>1,snapTimelineTime:x=>x,updateClipElementPreview(){},updateMonitor(){},status:null,window:{addEventListener:(name,fn)=>{handlers[name]=fn;}},fmtTime:String});
handlers.pointermove({clientX:100}); assert.equal(clip.startTime,11); assert.equal(linked.startTime,12);
handlers.pointermove({clientX:-100}); assert.equal(clip.startTime,7); assert.equal(linked.startTime,8);

const prompter = read('iamccs_prompter_ui.js');
const inject = prompter.slice(prompter.indexOf('    injectBtn.onclick = () => {'),prompter.indexOf('    writingButtons.forEach(',prompter.indexOf('    injectBtn.onclick = () => {')));
for (const target of ['global','local_1','local_auto']) {
  const calls=[]; const project={injection_target:target,task_mode:'t2va',sections:{scene:'scene only'},local_prompts:[],merge_policy:'replace'};
  const shotboard={_iamccsMiniMaxInjectPrompt:args=>{calls.push(args);return{actualTarget:args.target}}};
  vm.runInNewContext(inject+'injectBtn.onclick()',{injectBtn:{},injectStatus:{},project,commit(){},composePrompt:()=> 'scene only',shotboardsForPrompter:()=>[shotboard],node:{},aiVisualFiles:[],setTimeout(){}});
  assert.equal(calls.length,1,target); assert.equal(calls[0].target,target); assert.equal(calls[0].prompt,'scene only');
}
console.log('UI regression tests OK: 5 recipes, 3 Turbo step contracts, linked collision both directions, 3 injection targets');

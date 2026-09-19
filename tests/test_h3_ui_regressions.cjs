const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const root = path.join(__dirname, '..');
const read = name => fs.readFileSync(path.join(root, 'web', name), 'utf8');
const settings = read('iamccs_h3_settings_pro_ui.js');
assert.match(settings, /document\.createTextNode\('2 STAGE'\)/);
assert.match(settings, /setValue\(node,'longvid_pianosequenza_2stage_enabled',stageInput\.checked\)/);
assert.match(settings, /DIRECT controls the endpoint\/profile; NATIVE controls the linked LOW\/HIGH resolution pair/);
assert.match(settings, /if \(\["upscale_link_to_native", "upscale_link_factor"\]\.includes\(name\)/);
assert.match(settings, /Asset compatibility/);
assert.match(settings, /selectedH3AssetCompatibility\(node, mode\)/);
assert.match(settings, /AUTO · READ CURRENT SHOTBOARD/);
assert.match(settings, /function importSettingsFromShotboard\(node\)/);
assert.match(settings, /iamccs_settings_master = true/);
assert.match(settings, /AUTO_IMPORT_BLOCKED[\s\S]*"global_prompt"[\s\S]*"timeline_data"[\s\S]*"image_paths"/);
assert.match(settings, /AUTO imports once; it is not a live two-way sync/);
const autoImportCode = settings.slice(settings.indexOf('function importSettingsFromShotboard('), settings.indexOf('function assistantModeKey('));
const autoNodeValues = {task_mode:'auto_from_timeline',width:640,steps:20,global_prompt:'keep local truth',audio_mode:'h3_native'};
const autoBoardValues = {
  task_mode:'longvid_guides', width:1344,
  timeline_data:JSON.stringify({h3_truth_revision:17,h3_saved_settings:{steps:8,width:768,global_prompt:'must not import',audio_mode:'h3_custom_audio_drive'}})
};
const autoNode = {id:1,properties:{},widgets:Object.keys(autoNodeValues).map(name=>({name,get value(){return autoNodeValues[name]},set value(value){autoNodeValues[name]=value}})),setDirtyCanvas(){}};
const autoBoard = {id:2,widgets:Object.keys(autoBoardValues).map(name=>({name,get value(){return autoBoardValues[name]},set value(value){autoBoardValues[name]=value}}))};
const autoContext = vm.createContext({
  node:autoNode,AUTO_IMPORT_BLOCKED:new Set(['global_prompt','timeline_data','image_paths','audio_mode']),INTERNAL:new Set(),COMPATIBILITY_ONLY:new Set(),
  linkedShotboard:()=>autoBoard,widget:(owner,key)=>(owner.widgets||[]).find(item=>item.name===key),
  setValue:(owner,key,value)=>{const item=(owner.widgets||[]).find(entry=>entry.name===key);if(item)item.value=value;},
  shotboardMode:()=> 'longvid_guides',document:{dispatchEvent(){}},CustomEvent:class{},app:{graph:{change(){}}},alert:message=>{throw Error(message)},Date
});
vm.runInContext(autoImportCode+'importSettingsFromShotboard(node)',autoContext);
assert.equal(autoNodeValues.width,1344);
assert.equal(autoNodeValues.steps,8);
assert.equal(autoNodeValues.global_prompt,'keep local truth');
assert.equal(autoNodeValues.audio_mode,'h3_native');
assert.equal(autoNodeValues.task_mode,'longvid_guides');
assert.equal(autoNode.properties.iamccs_settings_master,true);
const compatCode = settings.slice(settings.indexOf('function modeFamily('), settings.indexOf('function firstChoice('));
const compatValues = {
  acceleration:'h3_sla', turbo_mode:'early_8_10',
  turbo_lora_name:'minimax_h3_fl2v_lightx2v_turbo_4step.safetensors',
  secondary_lora_enabled:false, secondary_lora_name:'', pdd_lora_name:'', fused_turbo_model_name:''
};
const compatNode = {};
const compatContext = {widget:(_,key)=>({value:compatValues[key],options:{iamccs_h3_assets:[]}}), node:compatNode};
vm.createContext(compatContext);
vm.runInContext(compatCode, compatContext);
const mismatch = vm.runInContext("selectedH3AssetCompatibility(node, 'ref2va')", compatContext);
assert.equal(mismatch.some(([,message])=>message.includes('requires REF2')), true);
const match = vm.runInContext("selectedH3AssetCompatibility(node, 'fl2va')", compatContext);
assert.equal(match.some(([kind])=>kind==='error'), false);
assert.equal(vm.runInContext("modeFamily('v2va_face_swap')", compatContext), 'ref2');
compatValues.acceleration='pdd_native_8step'; compatValues.turbo_mode='off'; compatValues.pdd_lora_name='MiniMax-H3-FL2V-Acc-8Step-comfy.safetensors';
assert.equal(vm.runInContext("selectedH3AssetCompatibility(node, 'ref2va')", compatContext).some(([kind])=>kind==='error'), true);
compatValues.acceleration='native'; compatValues.pdd_lora_name=''; compatValues.secondary_lora_enabled=true; compatValues.secondary_lora_name='creative_fl2v_style.safetensors';
assert.equal(vm.runInContext("selectedH3AssetCompatibility(node, 'v2va_face_swap')", compatContext).some(([kind])=>kind==='error'), true);
compatValues.secondary_lora_enabled=false; compatValues.acceleration='matlowai_fused_turbo_manual_sigma'; compatValues.fused_turbo_model_name='fastvideo_fasth3_int8_convrot.safetensors';
assert.equal(vm.runInContext("selectedH3AssetCompatibility(node, 'longvid_guides')", compatContext).some(([,message])=>message.includes('not compatible')), true);
const recipe = settings.slice(settings.indexOf('function applyRecipe('), settings.indexOf('function applyAccelerationChoice('));
for (const [name, steps, sampler] of [['native',20,'res_multistep'],['pdd',8,'euler'],['fasth3',6,'euler'],['sla',4,'euler'],['fused',4,'res_multistep']]) {
  const values = {turbo_mode:'old', pdd_lora_name:'old', fused_turbo_model_name:'minimax_fused_convrot.safetensors', shift_video:5, steps:99};
  const context = vm.createContext({node:{}, widget:(_, key)=>({value:values[key]}), setValue:(_,key,value)=>{values[key]=value;}, shotboardMode:()=> 'i2va', modeFamily:()=> 'fl2va', firstChoice:()=> 'asset', isPddLoRA:()=>true,isFastH3LoRA:()=>true,resetAcceleration:set=>{set('turbo_mode','off');set('turbo_lora_name','');set('pdd_lora_name','');set('fused_turbo_model_name','');}, document:{dispatchEvent(){}}, CustomEvent:class{}, alert:message=>{throw Error(message)}});
  vm.runInContext(recipe + `applyRecipe(node, '${name}')`, context);
  assert.equal(values.steps, steps, name); assert.equal(values.sampler_name, sampler, name);
  assert.equal(values.denoise, 1, name); assert.equal(values.shift_audio,3,name);
  assert.equal(values.shift_video,name==='sla'?6:12,name);
}
const declared = settings.slice(settings.indexOf('function applyDeclaredTurboContract('), settings.indexOf('function resetAcceleration('));
for (const steps of [3,4,6,8]) {
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
console.log('UI regression tests OK: AUTO Shotboard import, asset-family guard, 5 recipes, 4 Turbo step contracts, linked collision both directions, 3 injection targets');

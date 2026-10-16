const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_prompter_ui.js'), 'utf8');
const start = source.indexOf('    injectBtn.onclick =');
const handler = source.slice(start, source.indexOf('    writingButtons.forEach(', start));
assert.ok(start > 0);
assert.doesNotMatch(handler, /runAIRewrite|fetchApi|queuePrompt\s*\(|editMediaConfig\(/);

function inject(mode, options = {}) {
  const calls = [];
  const project = {
    task_mode:mode, injection_target:'global', extended_conditioning_policy:'default',
    merge_policy:'replace', request:'Analyze my new image', sections:{},
    swapchar_ai_signature:'stale signature', swapchar_variant:options.variant || 'identity',
    edit_video:{path:'new-source.mp4'}, local_prompts:options.locals || [],
  };
  const board = {
    _iamccsMiniMaxInjectPrompt(payload) { calls.push(payload); return {actualTarget:payload.target}; },
    _iamccsMiniMaxSetConditioningSchedule() { return null; },
    _iamccsMiniMaxSetImageReferences() {},
    _iamccsMiniMaxSetI2VHardCutSlots() {},
  };
  const forbidden = () => { throw Error('Inject must never invoke AI or Queue'); };
  const context = vm.createContext({
    project, node:{_iamccsPromptAiBusy:false}, injectBtn:{}, injectStatus:{},
    aiVisualFiles:[{path:'new-reference.png'}],
    canonicalMode:value=>value,
    isEditMode:value=>['v2va_face_swap_lora','ref2va_vfx_edit'].includes(value),
    effectiveGlobalPrompt:()=>options.global ?? 'CURRENT GLOBAL',
    effectiveLocalPrompt:()=>options.local ?? '',
    commit(){}, shotboardsForPrompter:()=>[board],
    widget:(_,key)=>({value:key === 'task_mode' ? mode : key === 'timeline_data' ? JSON.stringify({segments:[{id:'s1',start:0},{id:'s2',start:24}]}) : 2}),
    setWidget(){}, setTimeout(){}, runAIRewrite:forbidden,
    api:{fetchApi:forbidden}, app:{queuePrompt:forbidden},
  });
  vm.runInContext(handler, context);
  context.injectBtn.onclick();
  return {calls,status:context.injectStatus,project};
}
for (const [mode,variant] of [['v2va_face_swap_lora','identity'],['v2va_face_swap_lora','environment'],['ref2va_vfx_edit','vfx'],['reference_image'],['t2va'],['ref2va'],['i2va'],['fl2va'],['multi_shot_lipsync'],['audio_driven'],['v2va_object_swap']]) {
  const result = inject(mode,{variant});
  assert.equal(result.status.className,'iamccs-pr-inject-status ok', `${mode}: ${result.status.textContent}`);
  assert.equal(result.calls.length,1, mode);
  assert.equal(result.calls[0].prompt,'CURRENT GLOBAL', mode);
  assert.equal(result.calls[0].target,'global', mode);
  assert.deepEqual(result.project.sections,{});
}
for (const mode of ['t2va','i2va','ref2va','fl2va','multi_shot_lipsync']) {
  const result = inject(mode,{locals:[{slot:1,slot_id:'s1',prompt:'CURRENT A'},{slot:2,slot_id:'s2',prompt:'CURRENT B'},{slot:3,prompt:'disabled',enabled:false}]});
  assert.equal(result.status.className,'iamccs-pr-inject-status ok', result.status.textContent);
  assert.deepEqual(result.calls.map(call=>[call.target,call.prompt]),[['global','CURRENT GLOBAL'],['local_1','CURRENT A'],['local_2','CURRENT B']]);
}
const empty = inject('reference_image',{global:''});
assert.equal(empty.calls.length,0);
assert.equal(empty.status.textContent,'Nothing injected: fill at least one prompt section.');
const emptySwap = inject('v2va_face_swap_lora',{global:''});
assert.equal(emptySwap.calls.length,0);
assert.deepEqual(emptySwap.project.sections,{});
const localOnly = inject('t2va',{global:'',locals:[{slot:2,prompt:'Only B'}]});
assert.deepEqual(localOnly.calls.map(call=>[call.target,call.prompt]),[['local_2','Only B']]);
console.log('PASS: Inject transfers current global/requested locals across presets without AI, media analysis or Queue');

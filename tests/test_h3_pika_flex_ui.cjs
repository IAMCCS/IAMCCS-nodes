const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_h3_settings_pro_ui.js'), 'utf8').replace(/\r\n/g, '\n');
const extract = name => {
  const start = source.indexOf(`function ${name}(`);
  return source.slice(start, source.indexOf('\n}\n', start) + 2);
};
for (const mode of ['t2va', 'ref2va']) {
  const values = {acceleration:'fasth3_dense_6step', turbo_mode:'lightx2v', h3_omni_enabled:true,
    turbo_lora_name:'old-speed', pdd_lora_name:'old-pdd', steps:8, sampler_name:'euler',
    pika_flex_fl2va_lora:'', pika_flex_ref2va_lora:''};
  const node = {id:811, widgets:Object.entries(values).map(([name,value]) => ({name,value})), properties:{}};
  const widget = (n, name) => n.widgets.find(w => w.name === name);
  const context = vm.createContext({
    widget, shotboardMode:() => mode, modeFamily:() => mode === 'ref2va' ? 'ref2' : 'fl2',
    setValue:(n,name,value) => { const w=widget(n,name); if(w) w.value=value; else n.widgets.push({name,value}); },
    firstChoice:(n,key) => key.includes('ref2va') ? 'Pika-H3-Flex/pika_h3_flex_ref2va_comfy_bf16.safetensors' : 'Pika-H3-Flex/pika_h3_flex_fl2va_comfy_bf16.safetensors',
    document:{dispatchEvent(){}}, CustomEvent:class {constructor(name,options){this.detail=options.detail;}},
    rememberAttentionChoice(){},
  });
  vm.runInContext(['resetAcceleration','selectedAcceleration','accelerationAvailable','applyRecipe'].map(extract).join('\n') + '\nglobalThis.apply=applyRecipe;globalThis.selected=selectedAcceleration;', context);
  assert.equal(context.apply(node,'pika'),true);
  for(const [name,value] of Object.entries({acceleration:'pika_h3_flex_6step',turbo_mode:'off',h3_omni_enabled:false,
    steps:6,sampler_name:'lcm',scheduler:'simple',denoise:1,shift_video:12,shift_audio:3,h3_veda_enabled:false})) {
    assert.equal(widget(node,name).value,value,`${mode}: ${name}`);
  }
  assert.equal(context.selected(node),'pika');
  assert.match(widget(node, mode === 'ref2va' ? 'pika_flex_ref2va_lora' : 'pika_flex_fl2va_lora').value,/pika_h3_flex/);
  assert.equal(context.apply(node,'native'),true);
  assert.equal(context.selected(node),'native');
  assert.equal(widget(node,'steps').value,20);
}
console.log('PASS: Pika selection replaces old speed/Omni recipes, synchronizes six-NFE fields and returns to Native in both model families.');

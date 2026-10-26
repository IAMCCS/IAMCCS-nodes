const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_h3_settings_pro_ui.js'), 'utf8');
const recipe = source.slice(source.indexOf('function applyRecipe('), source.indexOf('const FACE_REFINE_PRESET_HELP'));
const reset = source.slice(source.indexOf('function resetAcceleration('), source.indexOf('function selectedAcceleration('));
const values = {turbo_mode:'early_8_10',turbo_lora_name:'old',pdd_lora_name:'old',h3_veda_enabled:true};
vm.runInNewContext(reset + recipe + 'applyRecipe(node, "omni")', {
  node:{id:1}, shotboardMode:()=> 'ref2va', modeFamily:()=> 'ref2',
  setValue:(_,key,value)=> {values[key]=value;}, document:{dispatchEvent:()=>{}},
  CustomEvent:class {},
});
assert.equal(values.h3_omni_enabled,true);
assert.equal(values.steps,8);
assert.equal(values.sampler_name,'euler');
assert.equal(values.scheduler,'simple');
assert.equal(values.shift_video,12);
assert.equal(values.shift_audio,3);
assert.equal(values.h3_veda_enabled,false);
assert.equal(values.turbo_mode,'off');
assert.equal(values.pdd_lora_name,'');
assert.match(source,/key === "ref2va_omni" \? "ref2va"/);
assert.match(source,/setValue\(node, "h3_omni_enabled", key === "ref2va_omni", false\)/);
assert.match(source,/FASTH3 OMNI · 8 STEP/);
console.log('Omni UI: atomic recipe, model mode alias and adapter isolation passed.');

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_h3_settings_pro_ui.js'), 'utf8');
const code = source.slice(source.indexOf('function installAssetOptions('), source.indexOf('function serializeNamedValues('));
const context = vm.createContext({widget:(node,name)=>node.widgets.find(item=>item.name===name)});
vm.runInContext(code + 'globalThis.install=installAssetOptions;', context);

// Use the installed frontend's actual preflight scanner when its path is given.
let missing = item => !item.options.values.includes(item.value);
if (process.argv[2]) {
  const frontend = fs.readFileSync(process.argv[2], 'utf8');
  const functions = ['resolveComboValues', 'isModelFileName', 'scanComboWidget'].map(name => {
    const start = frontend.indexOf(`function ${name}(`);
    assert.ok(start >= 0, name);
    return frontend.slice(start, frontend.indexOf('function ', start + 9));
  });
  const scanContext = vm.createContext({CR:new Set(['.safetensors','.ckpt','.pt','.pth','.bin','.sft','.onnx','.gguf'])});
  vm.runInContext(functions.join('\n') + '\nglobalThis.scan=scanComboWidget;', scanContext);
  missing = item => scanContext.scan({
    valueWidget:item,definitionWidget:item,executionId:'811',
    nodeType:'IAMCCS_ShotboardH3SettingsPro',definitionWidgetName:item.name,candidateWidgetName:item.name,
  },()=>false,()=>undefined)?.isMissing;
}

const birefnet = {name:'h3_faceswap_birefnet_model',type:'combo',value:'birefnet.safetensors',options:{values:[''],tooltip:'preserve'}};
const sam = {name:'h3_faceswap_sam_model',type:'combo',value:'missing_sam.safetensors',options:{values:[]}};
const lora = {name:'turbo_lora_name',value:'turbo.safetensors',options:{values:()=>[]}};
const resolution = {name:'upscale_link_factor',value:'576x960 -> 1152x1920'};
const numeric = {name:'steps',value:20,options:{min:1,max:100}};
const node = {widgets:[birefnet,sam,lora,resolution,numeric]};
const specs = {input:{required:{
  turbo_lora_name:[['','turbo.safetensors'],{iamccs_h3_assets:[{name:'turbo.safetensors'}],iamccs_sla_available:true}],
  upscale_link_factor:[['2x','576x960 -> 1152x1920'],{default:'2x'}],
  steps:['INT',{default:20,min:1,max:100}],
},optional:{
  h3_faceswap_birefnet_model:[['','birefnet.safetensors'],{default:'birefnet.safetensors'}],
  h3_faceswap_sam_model:[['','installed_sam.safetensors'],{default:''}],
}}};
assert.equal(missing(birefnet),true,'reproduce installed model falsely marked missing by stale widget choices');
context.install(node,specs);
assert.equal(missing(birefnet),false,'installed BiRefNet passes the frontend check');
assert.equal(missing(lora),false,'LoRA choices are restored too');
assert.equal(missing(sam),true,'a genuinely missing selected model remains visible to validation');
assert.equal(sam.value,'missing_sam.safetensors','do not replace another mode\'s selected model');
assert.equal(birefnet.value,'birefnet.safetensors');
assert.equal(birefnet.options.tooltip,'preserve');
assert.equal(lora.options.iamccs_sla_available,true);
assert.equal(lora.options.iamccs_h3_assets[0].name,'turbo.safetensors');
assert.ok(resolution.options.values.includes(resolution.value));
assert.equal(numeric.options.max,100);
assert.equal(numeric.options.values,undefined);
assert.notEqual(birefnet.options.values,specs.input.optional.h3_faceswap_birefnet_model[0]);
// A subsequent schema refresh removes stale files without legitimizing them.
specs.input.optional.h3_faceswap_birefnet_model[0]=[''];
context.install(node,specs);
assert.equal(missing(birefnet),true);
assert.equal(birefnet.value,'birefnet.safetensors');
console.log('PASS: Settings PRO restores server combo lists; installed models pass preflight, missing models remain detectable, selections preserved');

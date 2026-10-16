const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname,'../web/iamccs_h3_settings_pro_ui.js'),'utf8');
// Check the entire browser module as ESM, including UI code outside the extracted helpers.
const syntax = require('node:child_process').spawnSync(process.execPath, ['--input-type=module', '--check'], {input:source,encoding:'utf8'});
assert.equal(syntax.status,0,syntax.stderr);
const helpers = source.slice(source.indexOf('function restoreNamedValues('),source.indexOf('function normalizePianosequenzaStageControls('));
const context = vm.createContext({widget:(node,name)=>node.widgets.find(w=>w.name===name)});
vm.runInContext(helpers+'globalThis.restore=restoreNamedValues;globalThis.save=serializeNamedValues;',context);
const input = process.argv[2] ? JSON.parse(fs.readFileSync(process.argv[2],'utf8')) : {
  nodeData:{name:'IAMCCS_ShotboardH3SettingsPro',input:{required:{
    seed:['INT',{default:42}],seed_control_after_generate_compat:['STRING',{default:'fixed'}],
    steps:['INT',{default:20,min:1,max:100}],
    h3_upres_model_name:[['','upscaler.safetensors'],{default:'upscaler.safetensors'}],
    h3_upres_precision:[['fp16','bf16'],{default:'fp16'}],
  },optional:{
    h3_faceswap_sam_model:[['','sam3.safetensors'],{default:''}],
    h3_faceswap_birefnet_model:[['','birefnet.safetensors'],{default:'birefnet.safetensors'}],
    h3_faceswap_mask_prompt:['STRING',{default:'head'}],
  }}},
};
const nodeData = input.nodeData;
const specs = {...nodeData.input.required,...nodeData.input.optional};
const names = Object.keys(specs);
const values = names.map(name=>input.saved?.widgets_values_named?.[name] ?? specs[name][1]?.default ?? (Array.isArray(specs[name][0]) ? specs[name][0][0] : ''));
const expected = Object.fromEntries(names.map((name,i)=>[name,values[i]]));

// Exercise the actual installed INT constructor: absent means an automatic
// seed control, while explicit false means no auxiliary widget at all.
let constructInt;
if(process.argv[3]) {
  const frontend = fs.readFileSync(process.argv[3],'utf8');
  const start=frontend.indexOf('useIntWidget=()=>');
  const end=frontend.indexOf(';function addMarkdownWidget',start);
  assert.ok(start>0 && end>start);
  const native=vm.createContext({
    isIntInputSpec:()=>true,$:()=>({get:()=>false}),onValueChange(){},
    transformInputSpecV2ToV1:spec=>['INT',spec],
    addValueControlWidget(node){return node.addWidget('combo','control_after_generate','randomize',()=>{},{serialize:false});},
  });
  vm.runInContext('var '+frontend.slice(start,end)+';globalThis.construct=useIntWidget();',native);
  constructInt=native.construct;
} else {
  constructInt=(node,spec)=>{
    const item=node.addWidget('number',spec.name,spec.default);
    if(spec.control_after_generate ?? ['seed','noise_seed'].includes(spec.name)) node.addWidget('combo','control_after_generate','randomize',null,{serialize:false});
    return item;
  };
}
function makeNode() {
  const node={widgets:[],addWidget(type,name,value,callback,options={}){
    const item={type,name,value,callback,options};this.widgets.push(item);return item;
  }};
  for(const [name,[kind,options]] of Object.entries(specs)) {
    if(kind==='INT')constructInt(node,{type:kind,name,...options});
    else node.addWidget(Array.isArray(kind)?'combo':kind.toLowerCase(),name,options.default,null,Array.isArray(kind)?{values:kind.slice()}:{});
  }
  return node;
}
function applyNativePositions(node,info) {
  let index=0;
  for(const item of node.widgets) {
    // LGraphNode.configure counts widget.serialize, not options.serialize.
    if(item.serialize===false)continue;
    if(index<info.widgets_values.length)item.value=info.widgets_values[index++];
  }
}
// Reproduce the historical frontend behavior even when the live server already disables it.
delete specs.seed[1].control_after_generate;
const before=makeNode();
assert.equal(before.widgets.filter(w=>w.name==='control_after_generate').length,1);
applyNativePositions(before,{widgets_values:values});
assert.equal(before.widgets.find(w=>w.name==='h3_faceswap_sam_model').value,expected.h3_faceswap_birefnet_model);
assert.equal(before.widgets.find(w=>w.name==='h3_upres_model_name').value,expected.h3_upres_precision);

// Run the registration hook up to the seed fix, before widget construction.
const hookStart=source.indexOf('  async beforeRegisterNodeDef(nodeType, nodeData)');
const seedFix=source.slice(source.indexOf('    const seedOptions',hookStart),source.indexOf('    const created =',hookStart));
vm.runInNewContext(seedFix,{nodeData});
assert.equal(specs.seed[1].control_after_generate,false);
const node=makeNode();
assert.equal(node.widgets.filter(w=>w.name==='control_after_generate').length,0);
const positional={widgets_values:values.slice(),properties:input.saved?.properties ?? {iamccs_h3_settings_schema:names.slice()}};
applyNativePositions(node,positional);
context.restore(node,positional,nodeData);
for(const name of names)assert.deepEqual(node.widgets.find(w=>w.name===name).value,expected[name],name);

// Repeated save/reload, including reordered frontend widgets and a UI widget.
// Named restoration must win over the positional layout in every cycle.
for(let cycle=0;cycle<5;cycle++) {
  const saved={};context.save(node,saved,nodeData);
  assert.deepEqual(Array.from(saved.properties.iamccs_h3_settings_schema),names);
  assert.deepEqual(Array.from(saved.widgets_values),values);
  node.widgets.reverse();
  if(cycle===0)node.widgets.push({name:'H3 Settings PRO',value:'',options:{serialize:false}});
  applyNativePositions(node,saved);
  context.restore(node,saved,nodeData);
  for(const name of names)assert.deepEqual(node.widgets.find(w=>w.name===name).value,expected[name],`${cycle}: ${name}`);
}
// A saved older schema maps positions by its own names, even after additions.
const oldNames=names.filter(name=>name!=='h3_upres_precision');
const old={widgets_values:oldNames.map(name=>expected[name]),properties:{iamccs_h3_settings_schema:oldNames}};
context.restore(node,old,nodeData);
assert.equal(node.widgets.find(w=>w.name==='h3_upres_precision').value,specs.h3_upres_precision[1].default);
assert.equal(node.widgets.find(w=>w.name==='h3_faceswap_birefnet_model').value,expected.h3_faceswap_birefnet_model);
// Imported values can become invalid after onConfigure, before Queue.
const corruptSpecs = {input:{required:{
  fused_turbo_model_name:[['','fused.safetensors'],{default:''}],
  h3_upres_model_name:[['','upscaler.safetensors'],{default:'upscaler.safetensors'}],
  motion_context_window_frames:['INT',{default:362,min:56,max:362}],
  vram_clean_before_decode:['BOOLEAN',{default:true}],
  steps:['INT',{default:20,min:1,max:100}],
}}};
const corruptNode = {widgets:Object.entries({fused_turbo_model_name:'strict_match',h3_upres_model_name:false,
  motion_context_window_frames:2,vram_clean_before_decode:'low_vram',steps:27}).map(([name,value])=>({name,value}))};
const repaired={};context.save(corruptNode,repaired,corruptSpecs);
assert.deepEqual(Array.from(repaired.widgets_values),['','upscaler.safetensors',362,true,27]);
assert.equal(corruptNode.widgets.find(item=>item.name==='motion_context_window_frames').value,362);
// Exercise AUTO FROM SHOTBOARD itself, which runs after loading the node.
const importCode=source.slice(source.indexOf('function importSettingsFromShotboard('),source.indexOf('function assistantModeKey('));
const importedNode={widgets:[],_iamccsH3SettingsNodeData:corruptSpecs};
for(const [name,spec] of Object.entries(corruptSpecs.input.required)) importedNode.widgets.push({name,value:spec[1].default});
Object.assign(context,{
  linkedShotboard:()=>corruptNode,
  AUTO_IMPORT_BLOCKED:new Set(),INTERNAL:new Set(),COMPATIBILITY_ONLY:new Set(),HIDDEN_MODES:new Set(),
  setValue:(owner,name,value)=>{const item=owner.widgets.find(item=>item.name===name);if(item)item.value=value;},
  shotboardMode:()=> 'auto_from_timeline',
  document:{dispatchEvent(){}},CustomEvent:class {},app:{graph:{change(){}}},
});
corruptNode.widgets.find(item=>item.name==='fused_turbo_model_name').value='strict_match';
corruptNode.widgets.find(item=>item.name==='motion_context_window_frames').value=2;
vm.runInContext(importCode+'globalThis.importBoard=importSettingsFromShotboard;',context);
assert.equal(context.importBoard(importedNode),true);
assert.equal(importedNode.widgets.find(item=>item.name==='fused_turbo_model_name').value,'');
assert.equal(importedNode.widgets.find(item=>item.name==='motion_context_window_frames').value,362);
assert.equal(importedNode.widgets.find(item=>item.name==='steps').value,27);
const boardSource=fs.readFileSync(path.join(__dirname,'../web/iamccs_minimax_h3_shotboard_ui.js'),'utf8');
const boardNamesCode=boardSource.slice(boardSource.indexOf('const MINIMAX_H3_CANONICAL_WIDGET_NAMES ='),boardSource.indexOf('function applyMiniMaxH3CanonicalValues('));
const boardContext=vm.createContext({});vm.runInContext(boardNamesCode+'globalThis.names=MINIMAX_H3_CANONICAL_WIDGET_NAMES;',boardContext);
assert.equal(boardContext.names.length,112,'all appended backend fields must survive saving');
assert.equal(boardContext.names[100],'fused_turbo_model_name');
assert.equal(boardContext.names[111],'extended_av_joint_window_overlap');
console.log('PASS: corrupt imported model/window/boolean values repaired before save; valid steps preserved; Shotboard persists all 112 fields');
console.log(`PASS: reproduced one-slot model shift with native seed helper; ${names.length} fields restored correctly across 5 reordered round trips and legacy schema`);
if(process.argv[3]) {
  const frontend=fs.readFileSync(process.argv[3],'utf8');
  const start=frontend.indexOf('var graphToPrompt=async');
  const end=frontend.indexOf(';function scanMissingNodes',start);
  assert.ok(start>0 && end>start);
  node.id=811;node.comfyClass='IAMCCS_ShotboardH3SettingsPro';node.title='Settings PRO';node.mode=0;node.inputs=[];
  const graph={
    computeExecutionOrder:()=>[node],
    serialize(){const saved={};context.save(node,saved,nodeData);return {nodes:[saved]};},
  };
  const queue=vm.createContext({
    Rr:{NEVER:2,BYPASS:4},compressWidgetInputSlots(){},
    zl:class {constructor(node){Object.assign(this,node);}getInnerNodes(){return[this];}},
  });
  vm.runInContext(frontend.slice(start,end)+';globalThis.build=graphToPrompt;',queue);
  queue.build(graph).then(prompt=>{
    assert.equal(Object.keys(prompt.output['811'].inputs).length,names.length);
    for(const name of names)assert.deepEqual(prompt.output['811'].inputs[name],expected[name],`Queue: ${name}`);
    console.log(`PASS: installed frontend graphToPrompt emits all ${names.length} Settings PRO inputs under their correct names, without queuing a render`);
  }).catch(error=>{console.error(error);process.exitCode=1;});
}

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');

const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_minimax_h3_shotboard_ui.js'), 'utf8');
const migration = source.slice(source.indexOf('const MINIMAX_H3_LEGACY_WIDGET_COUNT ='), source.indexOf('function repairMiniMaxH3WidgetState('));
const context = vm.createContext({getWidget:(node,name)=>node.widgets.find(item=>item.name===name), console});
vm.runInContext(migration + 'globalThis.names=MINIMAX_H3_CANONICAL_WIDGET_NAMES;globalThis.migrate=migrateMiniMaxH3WidgetSchema;', context);
const names = Array.from(context.names);
assert.equal(names.length, 112);
assert.equal(new Set(names).size, names.length);

for (const count of [100, 112]) {
  const values = names.slice(0, count).map(name=>`saved:${name}`);
  const widgets = names.map(name=>({name,value:`default:${name}`}));
  widgets.splice(names.indexOf('seed') + 1, 0, {name:'control_after_generate',value:'fixed',options:{serialize:false}});
  const node = {widgets};
  const saved = {widgets_values:values};
  // Native positional loading can consume a value for the UI seed helper.
  widgets.forEach((item,index)=>{if(index<values.length)item.value=values[index];});
  assert.notEqual(widgets.find(item=>item.name==='h3_controlnet_end_policy').value, 'saved:h3_controlnet_end_policy');
  context.migrate(node, saved);
  for (const name of names.slice(0,count)) {
    assert.equal(widgets.find(item=>item.name===name).value, `saved:${name}`, `${count}: ${name}`);
  }
  assert.equal(saved.widgets_values.length,112);
  for (let cycle=0;cycle<5;cycle++) {
    context.migrate(node,saved);
    assert.equal(widgets.find(item=>item.name==='fused_turbo_model_name').value,
      count===112 ? 'saved:fused_turbo_model_name' : 'default:fused_turbo_model_name');
    assert.equal(widgets.find(item=>item.name==='extended_av_joint_window_overlap').value,
      count===112 ? 'saved:extended_av_joint_window_overlap' : 'default:extended_av_joint_window_overlap');
  }
  widgets.find(item=>item.name==='steps').value=27;
  context.migrate(node);
  assert.equal(widgets.find(item=>item.name==='steps').value,27,'a live repair must preserve unsaved edits');
}
console.log('PASS: old 100-field and current 112-field Shotboards restore by backend names despite native seed controls; appended fields survive five cycles and unsaved edits remain intact');

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname,'../web/iamccs_h3_settings_pro_ui.js'),'utf8');
const start = source.indexOf('  let dropdownRefreshPending = false;');
const code = source.slice(start,source.indexOf('  node._iamccsSettingsProRefresh = refresh;',start));
assert.ok(start > 0);
const events = {}, tasks = [];
const controls = new Set();
const select = name => {
  const control = {name,matches:selector=>selector === 'select',closest:selector=>selector === 'select' ? control : null};
  controls.add(control);
  return control;
};
const main = {scrollTop:37};
let renders = 0, saved = null, renderedValue = null;
const root = {
  contains:target=>controls.has(target),isConnected:true,
  addEventListener:(name,handler,capture)=>{events[name]={handler,capture};},
};
const context = vm.createContext({
  root, document:{activeElement:null}, window:{setTimeout:fn=>tasks.push(fn)},
  node:{},lastMode:'v2va_face_swap_lora',
  q:selector=>selector === '.h3p-main' ? main : {},
  shotboardMode:()=> 'v2va_face_swap_lora', mirrorShotboardAuthority(){},
  connectedNodes:()=>[],nodeClass:()=>'',renderRail(){},
  renderMain(){renders++;renderedValue=saved;},renderTruth(){},
});
vm.runInContext(code+'globalThis.redraw=refresh;',context);
const flush = () => { while(tasks.length)tasks.shift()(); };
assert.equal(events.change.capture,true);
for(const name of ['resolution','upscale route','linked pair','model','sampler','scheduler','attention','memory','audio','swap duration']) {
  const control = select(name);
  context.document.activeElement=control;
  const before=renders;
  // Repeated background updates leave every focused native select mounted.
  for(let i=0;i<5;i++)context.redraw();
  assert.equal(renders,before,name);
  // Capture runs before onchange. Save and request redraw from that handler,
  // as makeControl and the Swap duration controller do.
  events.change.handler({target:control});
  saved=name;context.redraw();
  assert.equal(renders,before,name);
  flush();
  assert.equal(renders,before+1,name);
  assert.equal(renderedValue,name);
  assert.equal(main.scrollTop,37);
  // Existing explicit selection handlers still redraw once, immediately.
  events.change.handler({target:control});
  context.redraw(true);flush();
  assert.equal(renders,before+2,name);
}
// Escape/cancel without changing the value: flush when focus leaves the node.
context.document.activeElement=select('cancelled');context.redraw();
let before=renders;
events.focusout.handler();context.document.activeElement=null;flush();
assert.equal(renders,before+1);
// Moving to another picker must not close that picker to apply an old refresh.
context.document.activeElement=select('first');context.redraw();
before=renders;events.focusout.handler();context.document.activeElement=select('next');flush();
assert.equal(renders,before);
events.focusout.handler();context.document.activeElement=null;flush();
assert.equal(renders,before+1);
// Prevent graph pointer handlers stealing native focus, without preventDefault.
for(const name of ['pointerdown','mousedown']) {
  let stopped=0;
  events[name].handler({target:select('pointer'),stopPropagation(){stopped++;}});
  assert.equal(stopped,1);
  events[name].handler({target:{closest:()=>null},stopPropagation(){stopped++;}});
  assert.equal(stopped,1);
}
// Non-select input changes and selects outside this Settings node do not lock it.
context.document.activeElement={matches:()=>false};before=renders;context.redraw();
assert.equal(renders,before+1);
context.document.activeElement={matches:()=>true};context.redraw();
assert.equal(renders,before+2);
events.change.handler({target:{matches:()=>false}});assert.equal(tasks.length,0);
// Scheduled events cannot rebuild a removed node.
context.document.activeElement=select('removed');context.redraw();
events.change.handler({target:context.document.activeElement});
root.isConnected=false;before=renders;flush();assert.equal(renders,before);
assert.match(source,/format\.onchange[^\n]+refresh\(true\)/);
assert.match(source,/stagePair\.onchange = \(\) => \{[\s\S]{0,1100}refresh\(true\)/);
console.log('PASS: all Settings PRO dropdowns survive external refreshes; committed values, blur, pointer focus and removal handled');

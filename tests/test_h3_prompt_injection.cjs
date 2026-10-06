const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[2] || path.join(__dirname, '../web/iamccs_minimax_h3_shotboard_ui.js'), 'utf8');
const start = source.indexOf('    node._iamccsMiniMaxInjectPrompt = (');
const end = source.indexOf('\n    // Explicit bridge used by IAMCCS_Prompter.', start);
assert.ok(start >= 0 && end > start);

function board() {
  const timeline = {segments: [1, 2].map(i => ({id: `s${i}`, type: 'image', start: i * 5, prompt: `old${i}`, use_prompt: false})), globalPromptOnly: true};
  const editors = new Map(timeline.segments.map(s => [s.id, [s.prompt, s.prompt]]));
  const node = {widgets: [{name:'task_mode', value:'i2va'}]};
  const promptArea = {value:'old global'};
  let saved;
  const sync = () => timeline.segments.forEach(s => {
    if (editors.has(s.id)) s.prompt = s.local_prompt = s.relay_prompt = editors.get(s.id)[0];
  });
  const context = {node, timeline, promptArea, promptWidget: promptArea,
    syncTimelineTextFromDom: sync,
    syncSegmentTextPeers: (id, key, value) => { if (editors.has(id)) editors.set(id, [value, value]); },
    syncSegmentRelayPeers: () => {},
    setWidgetValue: (n, key, value) => { n[key] = value; },
    writeTimeline: (options = {}) => { if (!options.skipDomSync) sync(); saved = JSON.parse(JSON.stringify(timeline)); },
    draw: () => { context.writeTimeline(); timeline.segments.forEach(s => editors.set(s.id, [s.prompt, s.prompt])); },
    refPaths: () => [], newId: () => `s${timeline.segments.length + 1}`,
    defaultLen: () => 5, endOfSegments: rows => rows.length * 5};
  vm.runInNewContext(source.slice(start, end), context);
  return {node, timeline, editors, promptArea, saved: () => saved};
}

for (const strictSlot of [false, true]) {
  const b = board();
  const inject = b.node._iamccsMiniMaxInjectPrompt;
  inject({target:'global', prompt:'new global'});
  inject({target:'local_1', prompt:'action one', strictSlot});
  inject({target:'local_2', prompt:'action two', strictSlot});
  assert.equal(b.promptArea.value, 'new global');
  assert.deepEqual(b.saved().segments.map(s => s.prompt), ['action one','action two']);
  assert.equal(b.saved().globalPromptOnly, false);
  assert.ok(b.saved().segments.every(s => s.use_prompt));
  inject({target:'local_1', prompt:'extra', mergePolicy:'append', strictSlot});
  assert.equal(b.saved().segments[0].prompt, 'action one\n\nextra');
  // Simulate subsequent manual edit and redraw/Queue flush.
  b.editors.set('s1', ['manual edit','manual edit']);
  inject({target:'global', prompt:'updated global'});
  assert.equal(b.saved().segments[0].prompt, 'manual edit');
}
const b = board();
b.node._iamccsMiniMaxInjectPrompt({target:'local_2', slotId:'stale-id', strictSlot:true, createMissing:true, prompt:'bound fallback'});
assert.equal(b.saved().segments[1].prompt, 'bound fallback');
b.node._iamccsMiniMaxInjectPrompt({target:'local_3', strictSlot:true, createMissing:true, prompt:'new slot'});
assert.equal(b.saved().segments[2].prompt, 'new slot');
assert.throws(() => b.node._iamccsMiniMaxInjectPrompt({target:'local_9',strictSlot:true,prompt:'invalid'}), /no longer exists/);

const i2vStart = source.indexOf('    node._iamccsMiniMaxSetI2VHardCutSlots = (');
const i2vEnd = source.indexOf('\n    node._iamccsMiniMaxSetConditioningSchedule = (', i2vStart);
assert.ok(i2vStart >= 0 && i2vEnd > i2vStart);
function i2vBoard(mode = 'i2va') {
  const timeline = {segments:[{id:'s1',type:'image',start:0,length:5,prompt:'one',transition:'continuous'}],globalPromptOnly:true};
  const node = {widgets:[{name:'task_mode',value:mode}],properties:{}};
  let paths = [], saved = null, draws = 0;
  const context = {node,timeline,
    getWidget:(n,name)=>n.widgets.find(w=>w.name===name),
    syncTimelineTextFromDom(){},
    setOwnReferencePaths:(n,next)=>{paths=next.slice();},
    newId:()=>`s${timeline.segments.length+1}`,
    endOfSegments:rows=>rows.reduce((max,row)=>Math.max(max,Number(row.start||0)+Number(row.length||1)),0),
    defaultLen:()=>5,
    defaultForceWidget:{value:1},
    ensureDurationForFrames(){},
    writeTimeline(){saved=JSON.parse(JSON.stringify(timeline));},
    draw(){draws+=1;},
  };
  vm.runInNewContext(source.slice(i2vStart,i2vEnd),context);
  return {node,timeline,paths:()=>paths,saved:()=>saved,draws:()=>draws};
}
const i2v = i2vBoard();
const i2vResult = i2v.node._iamccsMiniMaxSetI2VHardCutSlots({paths:['one.png','two.png','three.png']});
assert.equal(i2vResult.slotCount,3);
assert.deepEqual(i2v.paths(),['one.png','two.png','three.png']);
assert.deepEqual(i2v.saved().segments.map(s=>s.imageFile),['one.png','two.png','three.png']);
assert.deepEqual(i2v.saved().segments.map(s=>s.transition),['continuous','hard_cut','hard_cut']);
assert.deepEqual(i2v.saved().segments.map(s=>s.imageTruthSource),['prompter_i2va_inject','prompter_i2va_inject','prompter_i2va_inject']);
assert.ok(i2v.saved().segments.every(s=>s.use_guide && s.imageTruthPinned));
assert.equal(i2v.draws(),1);
assert.throws(() => i2vBoard('ref2va').node._iamccsMiniMaxSetI2VHardCutSlots({paths:['identity.png']}), /Select I2VA/);
console.log('PASS: prompt injection survives DOM flush and ordered Prompter images become isolated I2VA hard-cut slots only');

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_h3_settings_pro_ui.js'), 'utf8');
const start = source.indexOf('    if (active === "engine_multishot") {');
const end = source.indexOf('    if (["engine_ref2va", "engine_ref2vid"].includes(active))', start);
assert.ok(start > 0 && end > start);
for (const initial of ['joint', 'isolated_shots']) {
  const values = {h3_multishot_boundary_mode: initial, flf_overlap_frames: 12, steps: 4};
  const buttons = new Map();
  const context = {
    active: 'engine_multishot', node: {}, recipes: {}, refresh: () => {},
    widget: (_, key) => ({value: values[key]}),
    setValue: (_, key, value) => { values[key] = value; },
    addRecipe: (_, label, id, enabled, selected, click) => buttons.set(id, {selected, click}),
    setShotboardAudio: () => {},
  };
  vm.runInNewContext(source.slice(start, end), context);
  assert.equal(buttons.get('multishot_clean').selected, initial === 'isolated_shots');
  assert.equal(buttons.get('multishot_joint').selected, initial === 'joint');
  buttons.get('multishot_clean').click();
  assert.equal(values.h3_multishot_boundary_mode, 'isolated_shots');
  assert.equal(values.flf_overlap_frames, 0);
  assert.equal(values.flf_join_mode, 'h3_keyframe_cut');
  buttons.get('multishot_joint').click();
  assert.equal(values.h3_multishot_boundary_mode, 'joint');
  assert.equal(values.steps, 4);
}
console.log('MultiShot presets: selection, persistence key and independent sampling controls passed.');

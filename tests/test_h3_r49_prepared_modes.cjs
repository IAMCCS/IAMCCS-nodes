const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');
const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_h3_settings_pro_ui.js'), 'utf8');
const events = [];
const values = {};
const preview = { mode: 2, properties: { iamccs_r49_preview_modes: ['v2va_controlnet'] } };
const context = {
  setValue: (_, name, value) => { values[name] = value; },
  widget: (_, name) => ({ value: values[name] }),
  shotboardMode: () => values.task_mode || 'i2va',
  linkedShotboard: () => null,
  setShotboardMode: (_, mode) => { values.task_mode = mode; },
  setShotboardAudio: (_, mode) => { values.audio_mode = mode; },
  connectedNodes: () => [preview], applyRecipe: () => {},
  document: { dispatchEvent: event => events.push(event.detail) },
  CustomEvent: class { constructor(_, options) { this.detail = options.detail; } },
};
vm.createContext(context);
vm.runInContext(source.slice(source.indexOf('function setSubjectReferenceCount('), source.indexOf('function choices(')), context);
for (const count of [1, 2, 3, 4]) {
  context.setSubjectReferenceCount({ id: 811 }, count);
  assert.deepStrictEqual([1, 2, 3, 4].map(i => values[`reference_role_${i}`]),
    [1, 2, 3, 4].map(i => i <= count ? 'subject_identity' : 'disabled'));
}
assert.strictEqual(events.length, 4);
assert.strictEqual(events[2].field, 'reference_roles');
vm.runInContext(source.slice(source.indexOf('const GROUPS ='), source.indexOf('const ENGINE_IDS =')) + '\nthis.groups = GROUPS;', context);
for (const name of ['engine_ref2va', 'engine_ref2vid']) {
  const fields = context.groups.find(group => group.id === name).fields;
  for (const field of ['reference_role_1', 'reference_role_2', 'reference_role_3', 'reference_role_4', 'reference_video_role']) {
    assert(fields.includes(field), `${name} lacks ${field}`);
  }
}
vm.runInContext(source.slice(source.indexOf('function setAssistantMode('), source.indexOf('function modeContext(')), context);
const prepared = { properties: { iamccs_r49_all_modes: true } };
context.setAssistantMode(prepared, 'v2va_controlnet');
assert.strictEqual(preview.mode, 0);
context.setAssistantMode(prepared, 'ref2vid_lipsync');
assert.strictEqual(preview.mode, 2);
preview.mode = 0;
context.setAssistantMode({ properties: {} }, 'ref2va');
assert.strictEqual(preview.mode, 0, 'Existing workflows retain their preview behavior');
console.log('R49 reference role presets, engine fields and mode-selected preview PASS');

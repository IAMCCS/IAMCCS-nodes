const fs = require('fs');
const path = require('path');
const assert = require('assert');
const vm = require('vm');
const root = path.resolve(__dirname, '..');
const src = fs.readFileSync(path.join(root, 'web/iamccs_h3_settings_pro_ui.js'), 'utf8');
const block = src.slice(src.indexOf('function setAssistantMode('), src.indexOf('function modeContext('));
const calls = [];
const values = { audio_mode: 'h3_custom_audio_drive', extended_av_profile: 'balanced_12_16gb' };
const context = {
  linkedShotboard: () => null, setShotboardMode: (_, mode) => { values.task_mode = mode; },
  setShotboardAudio: (_, value) => { calls.push(value); values.audio_mode = value; },
  widget: (_, name) => ({value: values[name]}), setValue: (_, name, value) => { values[name] = value; },
  normalizeExtendedAvProductionContract: () => {}, importSettingsFromShotboard: () => {},
  normalize2StageResolutionLink: () => {},
  document: {dispatchEvent: () => {}}, CustomEvent: class {},
};
vm.createContext(context);
vm.runInContext(block, context);
for (const mode of ['t2va', 'i2va', 't2va_continuous', 'fl2va_extended_av', 'fl2va_stable', 'fl2va_continuous', 'keyframe_joint_native', 'longvid_guides']) {
  context.setAssistantMode({}, mode);
  assert.equal(values.audio_mode, 'h3_custom_audio_drive', `${mode} replaced custom audio`);
}
assert.equal(values.keyframe_joint_latent_new, true);
assert.deepEqual(calls, []);
const registry = src.slice(src.indexOf('const MODE_CHOICES ='), src.indexOf('const FRIENDLY_VALUES ='));
vm.runInContext(registry + '\nthis.publicModes = MODE_CHOICES.filter(([,v]) => !HIDDEN_MODES.has(v)).map(([,v]) => v);', context);
assert(context.publicModes.includes('t2va_continuous'));
assert(!context.publicModes.includes('latent_go_ahead'));
assert(!context.publicModes.includes('longvid_motion_context'));
const autoCode = src.slice(src.indexOf('function importSettingsFromShotboard('), src.indexOf('function assistantModeKey('));
context.AUTO_IMPORT_BLOCKED = new Set();
context.INTERNAL = new Set();
context.COMPATIBILITY_ONLY = new Set();
context.app = {graph: {change() {}}};
context.linkedShotboard = () => ({id: 2});
context.shotboardMode = () => 'latent_go_ahead';
vm.runInContext(autoCode, context);
const autoNode = {};
context.importSettingsFromShotboard(autoNode);
assert.equal(values.task_mode, 'auto_from_timeline');
assert.equal(autoNode.properties.iamccs_auto_import_snapshot.effective_mode, 'auto_from_timeline');
const board = fs.readFileSync(path.join(root, 'web/iamccs_minimax_h3_shotboard_ui.js'), 'utf8');
const canonical = board.slice(board.indexOf('function canonicalH3TaskMode('), board.indexOf('function h3LipsyncTask('));
vm.runInContext(canonical, context);
assert.equal(context.canonicalH3TaskMode('t2va_continuous'), 't2va_continuous');
assert.equal(context.canonicalH3TaskMode('fl2va_extended_av'), 'fl2va_extended_av');
console.log('B1bis UI: audio authority, public modes, Latent New selection and task serialization PASS');

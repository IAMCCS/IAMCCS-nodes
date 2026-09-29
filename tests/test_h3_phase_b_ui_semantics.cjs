const fs = require('fs');
const path = require('path');
const root = path.resolve(__dirname, '..');
const ui = fs.readFileSync(path.join(root, 'web', 'iamccs_h3_settings_pro_ui.js'), 'utf8');
function assert(cond, msg) { if (!cond) throw new Error(msg); }
assert(ui.includes('const ENGINE_IDS'), 'ENGINES child registry missing');
for (const id of ['ahead','extension','continuation','refmod','control','face','face_refine','scout']) {
  assert(ui.includes(`"${id}"`), `ENGINES missing ${id}`);
}
assert(ui.includes('folder.innerHTML = `ENGINES'), 'ENGINES folder label missing');
assert(ui.includes('group.id !== "finish"'), 'ENGINES is not placed after numbered OUTPUT');
assert(ui.includes('extended_av_boundary_polish_ms'), 'Audio boundary polish control missing');
assert(ui.includes('extended_av_boundary_polish_strength'), 'Audio boundary polish strength missing');
assert(ui.includes('extended_av_soft_audio_handover_ms'), 'Soft AV handover control missing');
console.log('Phase B Settings PRO UI semantics: PASS');

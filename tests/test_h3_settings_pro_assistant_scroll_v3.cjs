const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');

const root = path.join(__dirname, '..');
const scroll = fs.readFileSync(path.join(root, 'web', 'iamccs_h3_settings_pro_assistant_scroll_v3.js'), 'utf8');

for (const token of [
  'IAMCCS.H3SettingsPro.AssistantScrollV3',
  'data-section',
  '[data-grid]',
  'height: 0 !important',
  'min-height: 0 !important',
  'overflow-y: auto !important',
  'overflow: hidden !important',
  'requestAnimationFrame(() => restoreScroll',
  'event.preventDefault()',
  'event.stopPropagation()',
  '{ passive: false }',
]) {
  assert.ok(scroll.includes(token), `missing scroll contract token: ${token}`);
}

assert.match(scroll, /state\.scrollTop\s*=\s*scroller\.scrollTop/);
assert.match(scroll, /grid\.classList\.toggle\(SCROLLER_CLASS, assistantActive\)/);
assert.match(scroll, /main\.classList\.toggle\(MAIN_CLASS, assistantActive\)/);
assert.match(scroll, /new MutationObserver\(\(\) => syncPanel\(root\)\)/);
console.log('H3 Settings PRO Assistant Scroll V3 contract: OK');

const fs = require('fs');
const assert = require('assert');
const source = fs.readFileSync(require('path').join(__dirname, '..', 'web', 'iamccs_prompter_ui.js'), 'utf8');
const settings = fs.readFileSync(require('path').join(__dirname, '..', 'web', 'iamccs_h3_settings_pro_ui.js'), 'utf8');
const shotboard = fs.readFileSync(require('path').join(__dirname, '..', 'web', 'iamccs_minimax_h3_shotboard_ui.js'), 'utf8');

assert.match(source, /new Option\("DEFAULT", "default"\)[\s\S]{0,180}new Option\("CONTINUOUS", "continuous"\)[\s\S]{0,180}new Option\("EVOLVING", "evolving"\)/,
  'DEFAULT, CONTINUOUS and EVOLVING must share one compact dropdown');
assert.match(source, /if \(policyName !== "default"\) project\.task_mode = "fl2va";/,
  'Continuous and Evolving must select FL2VA automatically while Default preserves normal mode selection');
assert.match(source, /actions\.append\(conditioningMode, restoreBtn, loadBtn, saveBtn, fileInput\)/,
  'The compact conditioning dropdown must replace the two toolbar buttons');
assert.doesNotMatch(source, /const continuousBtn = button|const evolvingBtn = button/,
  'Separate Continuous/Evolving buttons must stay removed');
assert.doesNotMatch(source, /actions\.append\(exampleSelect, exampleBtn/,
  'Stock example controls must stay out of the production toolbar');
assert.match(source, /const zoom = button\("\+", "iamccs-pr-zoom-btn"\)/,
  'Textarea expander must be a plus button');
assert.match(source, /padding-right:54px!important/,
  'Textarea must reserve room for the expander');
assert.match(source, /right:19px!important/,
  'Expander must stay left of the native scrollbar');
assert.match(source, /\.iamccs-pr-ai \.iamccs-pr-zoom-btn\{[^}]*width:25px!important[^}]*min-width:25px!important[^}]*padding:0!important/s,
  'AI panel expanders must override the full-width AI button rule');
assert.match(source, /Use positive language; do not write negative prompt lists|extractTimedEvolvingLines/,
  'Extended evolving UI contract must remain present');
assert.doesNotMatch(source, /using “0-8 seconds: action” for the first range and “At 8 seconds action” for each later change/,
  'Extended evolving AI rewrite must not hardcode 8-second phase templates');
assert.match(source, /const rewrittenSections = \{ \.\.\.project\.sections \};[\s\S]{0,2200}project\.sections = rewrittenSections;/,
  'AI rewrites must validate timed output before replacing the visible project fields');
assert.ok(source.includes('const second = "\\\\d+(?::\\\\d+(?:[.,]\\\\d+)?)?|\\\\d+(?:[.,]\\\\d+)?";'),
  'Narrative timed requests must recognise clock-style tokens while splitting inline phases');
assert.match(source, /const aiModelPicker = el\("select"\)/,
  'Local provider models must use an explicit select instead of a value-filtered datalist');
assert.match(source, /aiModelPicker\.replaceChildren\([\s\S]{0,180}new Option\(name, name\)/,
  'Every model returned by Ollama must be mounted as a selectable option');
assert.match(source, /aiModelPicker\.onchange[\s\S]{0,260}model selected:/,
  'Changing a local model must immediately update the visible selected-model status');
assert.doesNotMatch(source, /const aiModelList = el\("datalist"\)/,
  'The native datalist must not hide non-matching models behind the previously selected model text');


assert.match(source, /const preview = el\("textarea", "iamccs-pr-preview"\)/,
  'Final prompt must be directly editable');
assert.match(source, /const localPreview = el\("textarea", "iamccs-pr-preview"\)/,
  'FL2VA local\/timeline queue truth must be directly editable');
assert.match(source, /project\.final_prompt_override_enabled = true/,
  'Manual final global edits must become authoritative');
assert.match(source, /project\.final_local_prompt_override_enabled = true/,
  'Manual final local edits must become authoritative');
assert.match(source, /composeFl2vaGlobalPrompt/,
  'FL2VA must have a dedicated static global prompt composer');
assert.match(source, /composeFl2vaLocalPrompt/,
  'FL2VA must have a dedicated action\/event local prompt composer');
assert.match(source, /project\?\.extended_conditioning_policy === "continuous"[\s\S]{0,120}return "";/,
  'Continuous mode must not compose a local prompt');
assert.match(source, /continuous_action:[\s\S]{0,500}Do not divide it into timed phases or local prompts/,
  'Continuous mode must express one uninterrupted user action in GLOBAL');
assert.match(source, /extendedContinuous\s*\? \["action"\]/,
  'Continuous REQUEST AI must ask the model for one action field only');
assert.match(source, /isExtendedContinuousRequest\) \{[\s\S]{0,350}project\.sections\.shot_list = "";[\s\S]{0,350}project\.local_prompts = \[\];/,
  'Continuous AI success must clear stale timed and local prompt divisions');
assert.match(shotboard, /selectedPolicy === "default"[\s\S]{0,500}delete timeline\.pan_h3_conditioning_v1/,
  'Returning to Default must clear any previous Extended conditioning schedule');
assert.doesNotMatch(shotboard, /extended_single_slot_right_resize/,
  'Slot drag must not alter the separately authored take duration');
assert.doesNotMatch(shotboard, /extendedSourceSlots\[0\]\.start = 0|extendedSourceSlots\[0\]\.length = durationFrames/,
  'Saving must preserve the authored extended slot bounds');
assert.match(source, /narrativeRequest === null && activePromptKey === "request"/,
  'AI on the active REQUEST must translate narrative into H3 fields');
const dragCode = shotboard.slice(shotboard.indexOf('    function edgeDragPreview('), shotboard.indexOf('    function audioDragPreview('));
const runDrag = new Function('node', 'timeline', 'cloneSegments', 'clampSegment', 'normalizeTimelineDragPreviewItems', dragCode + '\nreturn edgeDragPreview;');
for (const mode of ['fl2va_extended_av', 't2va_continuous', 'i2va']) {
  const drag = runDrag({widgets:[{name:'task_mode',value:mode}]}, {}, items=>items.map(x=>({...x})), x=>x, x=>x);
  const rows = drag([{id:'a',type:'image',start:0,length:720}], 'a', -24, 'right', 960);
  assert.equal(rows[0].length, mode === 'i2va' ? 362 : 696, mode + ' right resize');
  const left = drag([{id:'a',type:'image',start:0,length:720}], 'a', 24, 'left', 960);
  assert.equal(left[0].start, mode === 'i2va' ? 358 : 24, mode + ' left resize');
}
assert.match(source, /validateEvolvingTimelineAgainstRequest\(narrativeRequest, timed\)/,
  'Evolving AI rewrite must validate immutable user timestamps');
assert.doesNotMatch(source, /using [“"]0-8 seconds: action[”"]/i,
  'Evolving AI rewrite must not hardcode an 8-second schedule');


assert.doesNotMatch(source, /fl2va:\s*["']action["']/,
  'Visual Story / Global+Locals must never place an FL2VA global prompt into ACTION');
assert.match(source, /final_prompt_override_enabled = Boolean\(globalText\)/,
  'FL2VA Visual Story global text must feed the editable final GLOBAL prompt');
assert.match(source, /const sourceText = project\.extended_conditioning_policy === "evolving" \? finalLocalPrompt : ""/,
  'Extended Evolving must parse the editable FINAL LOCAL timeline as queue truth');
assert.match(source, /function canonicalizeEvolvingTimeline\(/,
  'Extended Evolving must canonicalize malformed AI timelines before queue');
assert.match(source, /function validateCanonicalEvolvingTimeline\(/,
  'Extended Evolving must reject non-canonical carried-state prompting');
assert.match(source, /function validateEvolvingGlobalPrompt\(/,
  'Extended Evolving must reject timed or negative action syntax in GLOBAL');
assert.match(source, /EVERY LINE MUST START WITH ITS TIMESTAMP OR RANGE/,
  'AI authoring contract must require timestamp-first phase grammar');
assert.match(source, /Never output \[ONSET_once\], \[SUSTAIN\]/,
  'AI authoring contract must prohibit legacy tag aliases');


assert.match(source, /iamccs-pr-field-tools/,
  'Per-field AI and AUDIO LINE controls must live in a dedicated compact toolbar');
assert.match(source, /\.iamccs-pr-field-ai\{[^}]*width:auto!important[^}]*height:25px!important[^}]*max-height:25px!important[^}]*flex:0 0 auto!important/s,
  'Per-field action buttons must remain compact and must not stretch over textareas');
assert.match(source, /iamccs-pr-request-actions/,
  'REQUEST actions must be isolated in their own compact toolbar');

for (const hidden of ['latent_go_ahead', 'longvid_motion_context', 'longvid_continuous_guided', 'longvid_masked_loop_guided', 'guided_av_loop_experimental']) {
  assert.match(settings, new RegExp(`HIDDEN_MODES[^;]+${hidden}`), `${hidden} must stay hidden in Settings PRO`);
}
assert.match(settings, /LONGVID POSITIONED GUIDED · 2 HIGH PASS/);
assert.match(settings, /LONGVID POSITIONED GUIDED · AUDIOCUSTOM/);
assert.match(settings, /key === "longvid_guides"[\s\S]{0,300}longvid_pianosequenza_2stage_enabled", true/,
  'LongVid positioned guided must activate the two-stage path');
console.log('Prompter + Settings PRO Extended UI regression tests OK');

const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_dialogue_tag_editor_ui.js'), 'utf8');

(async () => {
  const listeners = {}, requests = [], statuses = [];
  const makeNode = (id, type, values = {}) => ({
    id, type, properties: {}, widgets: Object.entries(values).map(([name, value]) => ({name, value})),
    setDirtyCanvas() {},
  });
  const editor = makeNode(1, 'IAMCCS_DialogueTagEditor');
  editor._iamccsDialogueAudioStatus = (text, busy) => statuses.push({text, busy});
  const board = makeNode(6, 'IAMCCS_AudioBoardArranger', {arranger_data: '{}'});
  const shot = makeNode(7, 'IAMCCS_MiniMaxH3ShotPlanner', {timeline_data: '{"audioSegments":[],"segments":[]}', global_prompt: 'Keep current', duration_seconds: 14});
  const nodes = [editor, board, shot];
  const graph = {_nodes: nodes, change() {}, getNodeById: id => nodes.find(n => String(n.id) === String(id))};
  const full = {output: {
    '1': {class_type: editor.type, inputs: {}},
    '2': {class_type: 'IAMCCS_CineAudioInfo', inputs: {cine_linx: ['1', 0]}},
    '3': {class_type: 'OmniVoiceEngineNode', inputs: {instruct: 'female, low pitch'}},
    '4': {class_type: 'UnifiedTTSSRTNode', inputs: {engine: ['3', 0], srt_content: ['2', 5]}},
    '5': {class_type: 'IAMCCS_CineAudioInfo', inputs: {cine_linx: ['1', 0], generated_audio: ['4', 0]}},
    '6': {class_type: board.type, inputs: {cine_linx: ['5', 0], arranger_data: '{}'}},
    '7': {class_type: shot.type, inputs: {cine_linx: ['6', 0]}},
    '8': {class_type: 'IAMCCS_MiniMaxH3AtomicGeneration', inputs: {cine_linx: ['7', 0]}},
  }, workflow: {nodes: []}};
  let job = 0, published = 0;
  const audio = JSON.stringify({audioSegments: [{audioFile: 'rendered.wav', start: 0, length: 120}]});
  const api = {
    queuePrompt: async (_, request) => { requests.push(structuredClone(request)); return {prompt_id: `job-${++job}`}; },
    fetchApi: async () => ({ok: true, json: async () => ({[`job-${job}`]: {outputs: {'6': {iamccs_audio_board: [audio]}}}})}),
    addEventListener: (key, callback) => { listeners[key] = callback; },
  };
  const ctx = vm.createContext({app: {registerExtension() {}, graph, graphToPrompt: async () => structuredClone(full)}, api,
    console, window: {setTimeout, clearTimeout}, setTimeout, clearTimeout,
  });
  vm.runInContext(source.replace(/^import .*;\r?$/gm, ''), ctx);
  ctx.repairExistingDialogueRigLinks = () => 0;
  ctx.downstream = (_, predicate) => [board, shot].filter(predicate);
  const preset = ctx.applyMultiShotLipSyncDemo(ctx.defaultData());
  assert.equal(preset.lines[3].text, "Because even the universe doesn't know how to say goodbye.");
  const engine = makeNode(3, 'OmniVoiceEngineNode', {instruct: preset.lines[0].text});
  ctx.configureRigTTSNode(engine, engine.type, 0, preset);
  assert.equal(engine.widgets[0].value, 'female, middle-aged, low pitch');
  preset.speakers[0].voice_design.instruct = 'female, young adult, british accent';
  ctx.configureRigTTSNode(engine, engine.type, 0, preset);
  assert.equal(engine.widgets[0].value, 'female, young adult, british accent');
  board._iamccsAudioBoardApplyRuntimeTimeline = raw => { board.widgets[0].value = raw; return true; };
  board._iamccsAudioBoardTransport = {publish: async () => { published++; return true; }};
  ctx.installDialogueAudioQueue();
  await ctx.renderDialogueAudio(editor);
  assert.deepEqual(Object.keys(requests[0].output).sort(), ['1','2','3','4','5','6']);
  assert.deepEqual(requests[0].output['6'].inputs.cine_linx, ['5',0]);
  assert.equal(editor.properties.iamccs_dialogue_audio_render.stage, 'rendering');
  assert.throws(() => ctx.api.queuePrompt(0, full), /Wait for/);
  await ctx.finishDialogueAudio(editor);
  assert.equal(published, 1);
  assert.equal(board.widgets[0].value, audio);
  assert.equal(editor.properties.iamccs_dialogue_audio_render.stage, 'shotboard');
  assert.equal(requests.length, 1, 'Finishing audio must not queue video');
  await ctx.api.queuePrompt(0, full);
  assert.equal(requests[1].output['6'].inputs.cine_linx, undefined);
  assert.deepEqual(full.output['6'].inputs.cine_linx, ['5', 0], 'Do not disconnect the visible rig');
  const videoPath = {...requests[1].output, '7': {...requests[1].output['7'], class_type: 'PlannerFixture'}};
  assert.deepEqual(Object.keys(ctx.dialogueAudioPrompt(videoPath, ['8'])).sort(), ['6','7','8']);
  const partial = {output: {'40': {class_type: 'IAMCCS_Prompter', inputs: {}}}};
  await ctx.api.queuePrompt(0, partial);
  assert.deepEqual(requests[2], partial, 'Prompter partial execution must stay available');
  await ctx.renderDialogueAudio(editor);
  assert.deepEqual(requests[3].output['6'].inputs.cine_linx, ['5',0], 'Explicit Render must reconnect the execution path');
  shot.widgets.find(w => w.name === 'duration_seconds').value = 11.5;
  await ctx.finishDialogueAudio(editor);
  assert.equal(published, 1, 'Do not overwrite a Shotboard edited during TTS rendering');
  assert.equal(editor.properties.iamccs_dialogue_audio_render.stage, 'shotboard');
  assert.match(statuses.at(-1).text, /Shotboard changed/);
  await ctx.renderDialogueAudio(editor);
  listeners.execution_error({detail: {prompt_id: `job-${job}`, exception_message: 'TTS failed'}});
  assert.equal(editor.properties.iamccs_dialogue_audio_render.stage, 'shotboard');
  assert.match(statuses.at(-1).text, /TTS failed/);
  assert.throws(() => ctx.dialogueAudioPrompt(full.output, ['8']), /depends on Shotboard/);
  const boardSource = fs.readFileSync(path.join(__dirname, '../web/iamccs_minimax_h3_shotboard_ui.js'), 'utf8');
  const repair = boardSource.slice(boardSource.indexOf('    const repairInitialMultiAudioView ='), boardSource.indexOf('    repairInitialMultiAudioView();'));
  for (const current of [[], [{id: 'edited', audioFile: 'current.wav', start: 12, length: 71, trimStart: 18}]]) {
    const timeline = {truth_revision: 42, duration_seconds: 9, audioSegments: current,
      multiGeneration: {activeTimelineId: 'T01', audioByTimeline: {T01: [{audioFile: 'old.wav', length: 360}]}}};
    vm.runInNewContext(repair + 'repairInitialMultiAudioView();', {
      timeline, timelineWidget: {value: JSON.stringify(timeline)}, node: {},
    });
    assert.deepEqual(timeline.audioSegments, current);
    assert.equal(timeline.duration_seconds, 9);
  }
  console.log('Dialogue Render: audio-only dependencies, handoff, regeneration, concurrent edits, failures and OmniVoice instructions PASS');
})().catch(error => { console.error(error); process.exitCode = 1; });

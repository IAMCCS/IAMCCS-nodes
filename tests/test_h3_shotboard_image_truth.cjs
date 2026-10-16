const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_minimax_h3_shotboard_ui.js'), 'utf8');
const ctx = vm.createContext({app: {registerExtension() {}}, api: {}, console,
  document: {addEventListener() {}}, window: {}});
vm.runInContext(source.replace(/^import .*;\r?$/gm, ''), ctx);

const edited = {id: 'B', type: 'image', ref: 1, start: 96, length: 72,
  imageFile: 'current/B.png', imageTruthPath: 'imported/A.png', image_truth_path: 'imported/A.png',
  prompt: 'Current performance', local_prompt: 'Imported performance', relay_prompt: 'Old injection'};
const deleted = {...edited, imageFile: '', prompt: ''};
assert.equal(ctx.currentShotboardImagePath(edited), 'current/B.png');
assert.equal(ctx.currentShotboardImagePath(deleted), '');
assert.equal(ctx.currentShotboardImagePath({imageTruthPath: 'legacy.png'}), 'legacy.png');
ctx.syncSegmentImageFileForReference([edited], 0, 'edited/B.png');
assert.equal(edited.imageTruthPath, 'edited/B.png');
assert.equal(edited.imageTruthName, 'B.png');

for (const kind of ['Segments', 'Rows']) {
  const item = {...edited, imageFile: 'current/B.png', imageTruthPath: 'imported/A.png'};
  const map = {'current/B.png': 'package/B.png', 'imported/A.png': 'package/A.png'};
  if (kind === 'Segments') ctx.rewritePackagedSegments([item], map, {}, ['imported/A.png']);
  else ctx.rewritePackagedRows([item], {}, ['imported/A.png'], map);
  assert.equal(item.imageFile, 'package/B.png', `${kind}: remap the edited image`);
  const cleared = {...deleted};
  if (kind === 'Segments') ctx.rewritePackagedSegments([cleared], map, {}, ['imported/A.png']);
  else ctx.rewritePackagedRows([cleared], {}, ['imported/A.png'], map);
  assert.equal(cleared.imageFile, '', `${kind}: do not restore a deleted image through ref`);
}

const localSource = (start, stop) => source.slice(source.indexOf(start), source.indexOf(stop, source.indexOf(start)));
const normalize = localSource('    const normalizeTimelineSegmentReferences =', '    const endOfSegments =');
const segmentToRow = localSource('    const segmentToRow =', '    let pendingTimelineCommit =');
assert.ok(segmentToRow.length < 20000 && segmentToRow.length > 100, 'Keep the test limited to row serialization');
const timeline = {segments: [edited, deleted]};
ctx.timeline = timeline;
ctx.refPaths = () => ['imported/A.png', 'edited/B.png'];
ctx.isTimelineImageSegment = seg => seg.type === 'image';
ctx.referenceIndexForPath = (p, refs) => refs.indexOf(p) + 1;
ctx.getFps = () => 24;
ctx.videoPathForSegment = () => '';
ctx.defaultForceWidget = {value: 0.25};
vm.runInContext(normalize + segmentToRow + '\nnormalizeTimelineSegmentReferences(); globalThis.rows = timeline.segments.map(segmentToRow);', ctx);
assert.equal(ctx.rows[0].imageFile, 'edited/B.png');
assert.equal(ctx.rows[0].relay_prompt, 'Current performance');
assert.equal(ctx.rows[1].imageFile, '');
assert.equal(ctx.rows[1].imageTruthPath, '');
assert.equal(ctx.rows[1].relay_prompt, '');

const importRows = localSource('    function rowsToSegments(', '    async function applyImportedBoard(');
assert.ok(importRows.length < 14000 && importRows.length > 100, 'Keep the test limited to row import');
ctx.newId = () => 'imported';
ctx.getTotalFrames = () => 240;
vm.runInContext(importRows + '\nglobalThis.imported = rowsToSegments(rows);', ctx);
assert.equal(ctx.imported[0].imageFile, 'edited/B.png');
assert.equal(ctx.imported[0].prompt, 'Current performance');
assert.equal(ctx.imported[1].imageFile, '');
assert.equal(ctx.imported[1].prompt, '');
const conflictingRows = [{...edited, prompt: 'Edited prompt', relay_prompt: 'Old injection'}, {...deleted, relay_prompt: 'Old injection'}];
ctx.conflictingRows = conflictingRows;
vm.runInContext('globalThis.imported = rowsToSegments(conflictingRows);', ctx);
assert.equal(ctx.imported[0].prompt, 'Edited prompt');
assert.equal(ctx.imported[1].prompt, '');
console.log('Shotboard image truth: replacement, deletion, reference editing, package remap, queue and reimport PASS');

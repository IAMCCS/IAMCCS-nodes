const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(path.join(__dirname, '../web/iamccs_minimax_h3_shotboard_ui.js'), 'utf8');

// ComfyUI applies DOM-widget dimensions after the node's setSize call.
const layout = vm.createContext({
  solo:false, timelineViewport:{clientWidth:420,scrollLeft:500},
  timelineCanvas:{style:{}}, timelineMeterReadout:{},
  timelineViewMode:'fit', timelinePixelsPerSecond:200, timelineMeterSeconds:15,
  getDuration:()=>15, activeVisualSegments:()=>[],
});
vm.runInContext('const timelineViewportWidth = () => timelineViewport.clientWidth;' +
  source.slice(source.indexOf('    function computeTimelineCanvasWidth('), source.indexOf('    const isTimelineVideoSegment =')), layout);
vm.runInContext('resizeTimelineCanvas()', layout);
assert.equal(layout.timelineCanvas.style.width, '100%');
layout.timelineViewport.clientWidth = 1980;
vm.runInContext('resizeTimelineCanvas()', layout);
assert.equal(layout.timelineCanvas.style.width, '100%');
assert.equal(layout.timelineViewport.scrollLeft, 0);
layout.timelineViewMode = 'manual';
layout.timelineViewport.scrollLeft = 400;
vm.runInContext('resizeTimelineCanvas()', layout);
assert.equal(layout.timelineCanvas.style.width, '3000px');
assert.equal(layout.timelineViewport.scrollLeft, 400);
layout.solo = true;
layout.timelineViewport.clientWidth = 0;
vm.runInContext('resizeTimelineCanvas()', layout);
assert.equal(layout.timelineCanvas.style.width, '3000px');
layout.solo = false;
layout.timelineViewport.clientWidth = 3600;
vm.runInContext('resizeTimelineCanvas()', layout);
assert.equal(layout.timelineCanvas.style.width, '3600px');
assert.match(source, /new ResizeObserver\(resizeTimelineCanvas\)/);

class Element {
  constructor(tag) {
    this.tag = tag; this.children = []; this.dataset = {}; this.style = {};
    this.attributes = {}; this.value = ''; this.selectionStart = 0; this.selectionEnd = 0;
  }
  append(...children) { children.forEach(child => { child.parentNode = child.parentElement = this; this.children.push(child); }); }
  insertBefore(child, before) { child.parentNode = child.parentElement = this; this.children.splice(this.children.indexOf(before), 0, child); }
  setAttribute(key, value) { this.attributes[key] = value; }
  getAttribute(key) { return this.attributes[key]; }
  querySelectorAll(tag) { return this.children.flatMap(child => [...(child.tag === tag ? [child] : []), ...child.querySelectorAll(tag)]); }
  dispatchEvent(event) { this['on' + event.type]?.(event); }
  remove() { this.parentNode.children = this.parentNode.children.filter(child => child !== this); }
  focus(options) { this.focusOptions = options; }
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
}
const root = new Element('div');
const body = new Element('body');
let mutate;
const magnifiers = vm.createContext({
  document:{body,createElement:tag=>new Element(tag)},
  MutationObserver:class { constructor(callback) { mutate = callback; } observe() {} },
  Event:class { constructor(type) { this.type = type; } },
});
const magnifierCode = source.slice(source.indexOf('function installShotboardPromptMagnifiers('), source.indexOf('function renderShotboardLite('));
vm.runInContext(magnifierCode, magnifiers);
const global = new Element('textarea'); global.setAttribute('aria-label', 'Global prompt');
const note = new Element('textarea'); note.title = 'Private note, not sent to PromptRelay';
root.append(global, note);
magnifiers.root = root;
vm.runInContext('installShotboardPromptMagnifiers(root)', magnifiers);
assert.equal(root.querySelectorAll('button').length, 1);
let saved = '', committed = 0;
const local = new Element('textarea');
local.value = 'slot B4'; local.dataset.iamccsV3Key = 'prompt';
// Surrounding UI text must never exclude an explicitly marked local prompt.
local.title = 'Private note controls nearby';
local.style = {position:'absolute',left:'28px',right:'8px',top:'160px',height:'70px',zIndex:'40'};
local.oninput = () => { saved = local.value; };
local.onchange = () => { committed += 1; };
root.append(local);
mutate();
const wrap = local.parentNode;
const lens = wrap.children.find(child => child.tag === 'button');
assert.ok(lens);
assert.equal(wrap.style.left, 'min(28px, 20%)');
assert.equal(wrap.style.right, '8px');
assert.equal(local.style.height, '100%');
assert.equal(local.style.paddingRight, '42px');
const count = root.querySelectorAll('button').length;
mutate(); assert.equal(root.querySelectorAll('button').length, count);
lens.onclick({preventDefault(){},stopPropagation(){}});
const overlay = body.children[0], panel = overlay.children[0];
const editor = panel.children.find(child => child.tag === 'textarea');
assert.equal(editor.value, 'slot B4');
assert.match(panel.style.cssText, /box-sizing:border-box/);
assert.match(panel.style.cssText, /max-height:100%/);
assert.match(editor.style.cssText, /min-width:0/);
assert.match(editor.style.cssText, /overflow:auto/);
editor.value = 'updated B4'; editor.oninput();
assert.equal(saved, 'updated B4');
overlay.onkeydown({key:'Escape',preventDefault(){},stopPropagation(){}});
assert.equal(committed, 1);
assert.equal(body.children.length, 0);
assert.equal(lens.getAttribute('aria-pressed'), 'false');
assert.equal(local.focusOptions.preventScroll, true);
console.log('PASS: Solo layout restoration, manual zoom, dynamic local prompt overlay and persistence');

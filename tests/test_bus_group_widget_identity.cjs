const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const sourcePath = process.argv[2] || path.join(__dirname, '..', 'web', 'iamccs_bus_group.js');
const source = fs.readFileSync(sourcePath, 'utf8').replace(
  'import { app } from "../../scripts/app.js";',
  'const app = globalThis.app;'
);

const member = { id: 91, mode: 0, pos: [20, 20] };
const group = {
  id: 7,
  title: 'Group A',
  pos: [0, 0],
  size: [200, 200],
  _nodes: [member]
};
const graph = {
  _groups: [group],
  groups: [group],
  _nodes: [member],
  getNodeById: id => graph._nodes.find(node => node.id === id),
  setDirtyCanvas() {}
};
group.graph = graph;

let extension;
const app = {
  graph,
  canvas: {
    getCurrentGraph: () => graph,
    setDirty() {}
  },
  registerExtension(value) {
    extension = value;
  }
};

const context = vm.createContext({
  app,
  console,
  document: {
    createElement: () => ({
      getContext: () => ({
        font: '',
        measureText: value => ({ width: String(value).length * 8 })
      })
    })
  },
  setTimeout: callback => {
    callback();
    return 1;
  },
  clearTimeout() {},
  window: {
    LiteGraph: {
      ALWAYS: 0,
      NEVER: 2,
      NODE_TITLE_HEIGHT: 30,
      NODE_WIDGET_HEIGHT: 20
    }
  }
});
vm.runInContext(source, context, { filename: sourcePath });

class TestNode {
  constructor() {
    this.widgets = [];
    this.properties = {};
    this.inputs = [];
    this.outputs = [];
    this.size = [520, 120];
    this.pos = [0, 0];
    this.title = 'Bus Group';
    this.graph = graph;
  }

  addWidget(type, name, value, callback, options = {}) {
    const widget = {
      type,
      name,
      value,
      callback,
      options,
      computeSize: width => [width || 520, 40]
    };
    this.widgets.push(widget);
    return widget;
  }

  addCustomWidget(widget) {
    this.widgets.push(widget);
    return widget;
  }

  computeSize() {
    return this.size;
  }

  removeInput(index) {
    this.inputs.splice(index, 1);
  }

  removeOutput(index) {
    this.outputs.splice(index, 1);
  }
}

function adoptCustomWidgets(node) {
  for (const widget of node.widgets.filter(({ type }) => type === 'custom')) {
    const descriptors = {};
    let current = widget;
    while (current && current !== Object.prototype) {
      for (const key of Reflect.ownKeys(current)) {
        if (key === 'constructor' || Object.hasOwn(descriptors, key)) continue;
        descriptors[key] = Object.getOwnPropertyDescriptor(current, key);
      }
      current = Object.getPrototypeOf(current);
    }
    Object.setPrototypeOf(widget, { constructor: class LegacyWidget {} });
    Object.defineProperties(widget, descriptors);
  }
}

extension.beforeRegisterNodeDef(TestNode, { name: 'IAMCCS_bus_group' });
const node = new TestNode();
node.onNodeCreated();
adoptCustomWidgets(node);

const muteAll = node.widgets.find(widget => widget.name === 'Mute all');
muteAll.callback();
assert.equal(member.mode, 2, 'Mute all must apply to an adopted group row');
assert.equal(node.properties.bus_group_state['id:7'].mute, true);
assert.equal(node.properties.bus_group_state['id:7'].solo, false);

node.widgets.find(widget => widget.name === 'Enable all').callback();
assert.equal(member.mode, 0, 'Enable all must apply to an adopted group row');

node.properties.iamccs_bus_group_macros = [
  { name: 'Macro A', keys: ['id:7'] }
];
node._iamccsRefresh();
adoptCustomWidgets(node);
node.widgets
  .find(widget => widget.name === 'iamccs_bus_group_macro_row')
  ._toggleMuteAll();
assert.equal(member.mode, 2, 'Macros must resolve adopted group rows');

graph._groups = [];
graph.groups = graph._groups;
node.properties.iamccs_bus_group_macros = [];
node._iamccsRefresh();
for (const name of [
  'iamccs_bus_group_row',
  'iamccs_bus_group_macro_row',
  'iamccs_bus_group_divider'
]) {
  assert.equal(
    node.widgets.filter(widget => widget.name === name).length,
    0,
    `Refresh must remove stale adopted ${name} widgets`
  );
}

console.log('Bus Group adopted widget identity: OK');

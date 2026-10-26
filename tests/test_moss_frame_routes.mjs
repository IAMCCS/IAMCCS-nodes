import test from "node:test";
import assert from "node:assert/strict";
import { upstreamDesigners, layoutSignature, latestSavedRender } from "../web/moss_frame_routes.js";
import { LIGHT_PRESETS, CAMERA_PRESETS, STYLE_PRESETS } from "../web/moss_frame_presets.js";

const design = { canvas: { width: 1280, height: 720 }, scene: { high_level_description: "Two people" }, items: [
  { kind: "obj", role: "character", desc: "Red coat", x: 80, y: 100, w: 300, h: 820 },
  { kind: "obj", role: "character", desc: "Blue jacket", x: 620, y: 100, w: 300, h: 820 },
] };
const node = (id, type, origins = []) => ({ id, type, inputs: origins, getInputLink: slot => ({ origin_id: origins[slot] }) });

test("Recovery distinguishes local style and exact lettering mode", () => {
  const changed = structuredClone(design); changed.items[0].style = "2D anime";
  assert.notEqual(layoutSignature(design), layoutSignature(changed));
  changed.items[0].style = ""; changed.items[0].text_mode = "overlay";
  assert.notEqual(layoutSignature(design), layoutSignature(changed));
});

test("Presets contain plain editable scene text", () => {
  for (const group of [LIGHT_PRESETS, CAMERA_PRESETS, STYLE_PRESETS]) {
    assert.ok(group.length >= 10);
    assert.equal(new Set(group.map(row => row[0])).size, group.length);
    for (const [label, prompt] of group) {
      assert.ok(label && prompt.length > 30);
      assert.doesNotMatch(prompt, /\{prompt\}|<lora:|https?:\/\//);
    }
  }
});

test("SaveImage routes only to its own upstream MOSS designer", () => {
  const nodes = [node(1, "IAMCCS_MOSSFrameDesigner"), node(2, "CLIPTextEncode", [1]), node(3, "KSampler", [2]),
    node(4, "VAEDecode", [3]), node(5, "SaveImage", [4]), node(6, "IAMCCS_MOSSFrameDesigner"), node(7, "SaveImage", [6])];
  const graph = { getNodeById: id => nodes.find(n => String(n.id) === String(id)) };
  assert.deepEqual(upstreamDesigners(graph, 5).map(n => n.id), [1]);
  assert.deepEqual(upstreamDesigners(graph, 7).map(n => n.id), [6]);
});

test("Graph traversal tolerates missing nodes and cycles", () => {
  const nodes = [node(1, "IAMCCS_MOSSFrameDesigner", [2]), node(2, "Other", [1, 99])];
  assert.deepEqual(upstreamDesigners({ getNodeById: id => nodes.find(n => n.id === id) }, 2).map(n => n.id), [1]);
});

test("Legacy render recovery ignores project ids but requires matching layout", () => {
  const record = (order, value, filename) => ({ status: { status_str: "success" }, prompt: [order, "job", {
    "1": { class_type: "IAMCCS_MOSSFrameDesigner", inputs: { design_data: JSON.stringify(value) } },
    "2": { class_type: "SaveImage", inputs: { images: ["1", 0] } },
  }], outputs: { "2": { images: [{ filename, type: "output", subfolder: "" }] } } });
  const different = structuredClone(design); different.items[0].x = 600;
  const history = { old: record(1, design, "correct.png"), unrelated: record(2, different, "wrong.png") };
  assert.equal(latestSavedRender(history, { ...design, project_id: "new browser id" }).images[0].filename, "correct.png");
  assert.notEqual(layoutSignature(design), layoutSignature(different));
});

test("Recovery ignores an image from a disconnected save branch", () => {
  const history = { unrelated: { status: { status_str: "success" }, prompt: [1, "job", {
    "1": { class_type: "IAMCCS_MOSSFrameDesigner", inputs: { design_data: JSON.stringify(design) } },
    "2": { class_type: "SaveImage", inputs: { images: ["3", 0] } },
    "3": { class_type: "LoadImage", inputs: { image: "other.png" } },
  }], outputs: { "2": { images: [{ filename: "unrelated.png" }] } } } };
  assert.equal(latestSavedRender(history, design), null);
});

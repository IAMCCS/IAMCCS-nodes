import test from "node:test";
import assert from "node:assert/strict";
import { importedMboards, portableDesign, cloneProjectData } from "../web/moss_mboard_library.js";

const normalize = raw => {
  const value = raw.format === "IAMCCS.Mboard" ? raw.design : raw;
  if (!value?.canvas || !Array.isArray(value.items)) throw new Error("Invalid design");
  return structuredClone(value);
};
const design = { canvas: { width: 1024, height: 1536 }, items: [{ id: "actor", desc: "Original performer" }] };
const board = (name, value = design) => ({ name, project: { format: "IAMCCS.Mboard", version: 1, edition: "MOSS", design: value } });

function reactiveProxy(value) {
  return value && typeof value === "object" ? new Proxy(value, {
    get(target, key, receiver) { return reactiveProxy(Reflect.get(target, key, receiver)); },
  }) : value;
}

test("Reactive node properties reproduce the old DataCloneError and load through the JSON boundary", () => {
  const properties = reactiveProxy({ mossImportedBoards: Array.from({length:12}, (_, i) => ({
    key: String(i), name: `Scene ${i + 1}`, design: {
      ...design, scene: { lighting: "Window light", photo: "85mm" },
      generated: { files: [{ filename: "frame.png", type: "output" }], data_url: "data:image/png;base64,test" },
    },
  })) });
  assert.throws(() => structuredClone(properties.mossImportedBoards), { name: "DataCloneError" });
  const snapshot = cloneProjectData(properties.mossImportedBoards);
  assert.equal(snapshot.length, 12);
  assert.deepEqual(snapshot, JSON.parse(JSON.stringify(properties.mossImportedBoards)));
  assert.doesNotThrow(() => structuredClone(snapshot));
  snapshot[0].design.items[0].desc = "Changed clone";
  assert.equal(properties.mossImportedBoards[0].design.items[0].desc, "Original performer");
});

test("Even an empty reactive Mboard list can be restored without discarding it", () => {
  const list = reactiveProxy([]);
  assert.throws(() => structuredClone(list), { name: "DataCloneError" });
  assert.deepEqual(cloneProjectData(list), []);
});

test("Portable designs accept nested proxies and preserve Unicode, boxes and saved image references", () => {
  const source = reactiveProxy({ ...design, scene: { high_level_description: "The jester’s room — 1520" },
    generated: { files: [{ filename: "actor.png", subfolder: "MOSS", type: "output" }], width:1024, height:1536 },
    items:[{id:"title",kind:"text",text:'IAMCCS "MOSS"',x:80,y:25,w:800,h:100}] });
  const result = portableDesign(source);
  assert.deepEqual(result, JSON.parse(JSON.stringify(source)));
  assert.doesNotThrow(() => structuredClone(result));
});

test("A JSON pack returns every named board without modifying its source", () => {
  const pack = { format: "IAMCCS.Mboard.pack", edition: "MOSS", version: 1,
    boards: Array.from({ length: 12 }, (_, i) => board(`Scene ${i + 1}`)) };
  const before = JSON.stringify(pack);
  const entries = importedMboards(pack, normalize);
  assert.equal(entries.length, 12);
  assert.equal(entries.at(-1).name, "Scene 12");
  assert.equal(new Set(entries.map(entry => entry.key)).size, 12);
  entries[0].design.items[0].desc = "An edited scene";
  assert.equal(JSON.stringify(pack), before);
  assert.equal(entries[1].design.items[0].desc, "Original performer");
});

test("One malformed board rejects the entire pack before editor state is replaced", () => {
  const pack = { format: "IAMCCS.Mboard.pack", edition: "MOSS", version: 1,
    boards: [board("Valid"), board("Broken", { items: [] })] };
  assert.throws(() => importedMboards(pack, normalize), /Invalid design/);
  assert.throws(() => importedMboards({ ...pack, edition: "IDEOGRAM" }, normalize), /Unsupported/);
});

test("Single Mboards and legacy preset collections remain importable", () => {
  assert.equal(importedMboards(board("One").project, normalize, "One.Mboard")[0].name, "One.Mboard");
  assert.equal(importedMboards({ presets: { first: { ...design, label: "First scene" } } }, normalize)[0].name, "First scene");
  assert.equal(importedMboards({ nodes: [] }, normalize), null);
});

test("Workflow snapshots and embedded PNGs do not recursively inflate the dropdown", () => {
  const source = { ...design, project_metadata: { workflow: { nodes: [] }, note: "Keep" },
    generated: { data_url: "data:image/png;base64,large", files: [{ filename: "frame.png" }] } };
  const value = portableDesign(source);
  assert.equal(value.project_metadata.workflow, undefined);
  assert.equal(value.generated.data_url, undefined);
  assert.equal(value.generated.files[0].filename, "frame.png");
  assert.equal(value.project_metadata.note, "Keep");
  assert.ok(source.generated.data_url);
});

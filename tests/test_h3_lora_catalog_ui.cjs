const assert = require("node:assert/strict");
const { readFileSync } = require("node:fs");
const { join } = require("node:path");

async function main() {
  const source = readFileSync(join(__dirname, "..", "web", "iamccs_h3_advisor.js"), "utf8");
  const ui = await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
  const data = {
    choices: ["renamed.safetensors", "h3_sla.safetensors"],
    pdd: [],
    fasth3: ["renamed.safetensors"],
    assets: [],
    sla_defaults: { turbo_lora_name: "h3_sla.safetensors", steps: 4 },
  };
  let requests = 0;
  const api = {
    async fetchApi() {
      requests++;
      return { ok: true, json: async () => data };
    },
  };
  await Promise.all([ui.fetchH3LoraCatalog(api), ui.fetchH3LoraCatalog(api, { force: true })]);
  assert.equal(requests, 1);

  const turbo = { name: "turbo_lora_name", value: "saved.safetensors" };
  const steps = { name: "steps", value: 16 };
  const node = { widgets: [turbo, steps], _iamccsH3LoraConfigured: true };
  const initial = ui.captureH3LoraDefaults(node);
  ui.applyH3LoraCatalog(node, data);
  ui.applyNewH3LoraDefaults(node, data, initial);
  assert.equal(turbo.value, "saved.safetensors");
  assert.ok(turbo.options.values.includes(turbo.value));
  assert.deepEqual(turbo.options.iamccs_fasth3_values, ["renamed.safetensors"]);

  node._iamccsH3LoraConfigured = false;
  steps.value = 8;
  ui.applyNewH3LoraDefaults(node, data, initial);
  assert.equal(turbo.value, "saved.safetensors");
  assert.equal(steps.value, 8);

  steps.value = 16;
  ui.applyNewH3LoraDefaults(node, data, initial);
  assert.equal(turbo.value, "h3_sla.safetensors");
  assert.equal(steps.value, 4);
  console.log("PASS: shared catalog request, saved selection, and untouched-node defaults");
}

main().catch(error => { console.error(error); process.exitCode = 1; });

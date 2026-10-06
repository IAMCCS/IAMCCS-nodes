const fs = require("fs");
const path = require("path");

const settings = fs.readFileSync(
  path.join(__dirname, "..", "web", "iamccs_h3_settings_pro_ui.js"),
  "utf8",
);
const deno = fs.readFileSync(
  path.join(__dirname, "..", "..", "comfyui-deno-custom-nodes", "web", "js", "deno_floating_tools.js"),
  "utf8",
);

for (const contract of [
  "function deliveryBranchGate(node, mode)",
  "if (!deliveryBranchGate(node, mode).available) return \"native-delivery\";",
  "deliveryToggle._iamccsOptionalBranchGuard",
  "deliveryToggle.value = false;",
  "iamccs_optional_delivery_muted",
]) {
  if (!settings.includes(contract)) throw new Error(`Missing Settings PRO optional-branch guard: ${contract}`);
}

for (const contract of [
  "function uninstallSosPromptFailureHooks()",
  "if (isEnabled()) installSosRuntimeHooks();",
  "uninstallSosPromptFailureHooks();",
]) {
  if (!deno.includes(contract)) throw new Error(`Missing DENO opt-in hook contract: ${contract}`);
}

console.log("PASS: unavailable delivery mutes at queue and disabled DENO tools do not wrap /prompt");

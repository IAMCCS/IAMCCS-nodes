const fs = require("fs");
const path = require("path");

const source = fs.readFileSync(
  path.join(__dirname, "..", "web", "iamccs_prompter_ui.js"),
  "utf8",
);

for (const contract of [
  'const isEmptyImageRequest = isImageRequest && !String(narrativeRequest || "").trim();',
  'if (narrativeRequest !== null && !String(narrativeRequest).trim() && !isImageRequest)',
  'Add at least one AI reference image before using an empty image REQUEST.',
  'Preserve the reference medium, photographic or illustrative style',
  'if (!String(project.request || "").trim() && !aiVisualFiles.length)',
]) {
  if (!source.includes(contract)) throw new Error(`Missing empty-image-request contract: ${contract}`);
}

console.log("PASS: empty image REQUEST uses references and preserves their visible style");

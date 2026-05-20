import { app } from "../../scripts/app.js";

// IAMCCS intentionally does not touch ComfyUI workflow persistence.
//
// ComfyUI frontend v1.43+ restores open workflow tabs through its own browser
// storage keys. Older revisions of this extension tried to clean or recover
// those keys when localStorage quota was exceeded, but that can make ComfyUI
// start with a blank workspace or only one restored workflow.
//
// Keep this extension as a harmless no-op so cached extension lists and older
// installs do not fail to import it, while leaving all workflow/tab persistence
// behavior to ComfyUI itself.
app.registerExtension({
    name: "iamccs.workflow_persist_cleanup",
    async setup() {},
});

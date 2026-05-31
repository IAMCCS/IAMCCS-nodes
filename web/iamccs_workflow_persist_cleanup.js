import { app } from "../../scripts/app.js";

// IAMCCS: targeted ComfyUI frontend draft cleanup.
//
// ComfyUI frontend v1.46 stores workflow drafts in browser localStorage under:
// - Comfy.Workflow.Draft.v2:
// - Comfy.Workflow.DraftIndex.v2:
//
// Large audio-board / multigeneration workflows can exhaust browser draft
// storage and trigger "Failed to save workflow draft" on every edit, even for
// unrelated workflows. IAMCCS stores important workflows as physical JSON files,
// so draft payloads can be safely purged once when the browser gets stuck.
function clearWorkflowDraftStorage(reason = "manual") {
    try {
        const payloadPrefix = "Comfy.Workflow.Draft.v2:";
        const indexPrefix = "Comfy.Workflow.DraftIndex.v2:";
        const legacyPayloadPrefix = "Comfy.Workflow.Draft:";
        const legacyIndexPrefix = "Comfy.Workflow.DraftIndex:";
        let removed = 0;
        let chars = 0;
        for (let i = localStorage.length - 1; i >= 0; i -= 1) {
            const key = localStorage.key(i);
            if (!key) continue;
            const shouldPurge =
                key.startsWith(payloadPrefix) ||
                key.startsWith(indexPrefix) ||
                key.startsWith(legacyPayloadPrefix) ||
                key.startsWith(legacyIndexPrefix);
            if (!shouldPurge) continue;
            const value = localStorage.getItem(key) || "";
            chars += value.length;
            localStorage.removeItem(key);
            removed += 1;
        }
        if (removed) {
            console.info(`[IAMCCS] Manually purged ${removed} ComfyUI workflow draft storage entries (${chars} chars).`, { reason });
        } else {
            console.info("[IAMCCS] Workflow draft storage manual purge requested; no entries found.", { reason });
        }
    } catch (err) {
        console.warn("[IAMCCS] Workflow draft cleanup failed", err);
    }
}

function installWorkflowDraftRescue() {
    try {
        window.IAMCCS_clearWorkflowDraftStorage = clearWorkflowDraftStorage;
        installLiteDraftFallback();
        console.info("[IAMCCS] Workflow draft restore mode active. Drafts are preserved; manual rescue available as window.IAMCCS_clearWorkflowDraftStorage().");
    } catch (err) {
        console.warn("[IAMCCS] Workflow draft rescue install failed", err);
    }
}

function isWorkflowDraftKey(key) {
    const text = String(key || "");
    return text.startsWith("Comfy.Workflow.Draft.v2:") || text.startsWith("Comfy.Workflow.Draft:");
}

function looksLikeHeavyMediaString(value) {
    const text = String(value || "");
    return text.length > 180_000 || text.includes("data:audio") || text.includes("data:video") || text.includes(";base64,");
}

function makeLiteDraftPayload(payload) {
    try {
        const parsed = JSON.parse(String(payload || "{}"));
        let stripped = 0;
        const visit = (value, key = "") => {
            if (typeof value === "string") {
                const lowerKey = String(key || "").toLowerCase();
                if (
                    lowerKey.includes("b64") ||
                    lowerKey.includes("base64") ||
                    looksLikeHeavyMediaString(value)
                ) {
                    stripped += value.length;
                    return `[IAMCCS stripped heavy draft payload: ${value.length} chars]`;
                }
                return value;
            }
            if (Array.isArray(value)) return value.map((item) => visit(item, key));
            if (value && typeof value === "object") {
                for (const objectKey of Object.keys(value)) {
                    value[objectKey] = visit(value[objectKey], objectKey);
                }
            }
            return value;
        };
        const lite = visit(parsed);
        return { payload: JSON.stringify(lite), stripped };
    } catch {
        const text = String(payload || "");
        if (!looksLikeHeavyMediaString(text)) return { payload: text, stripped: 0 };
        return {
            payload: JSON.stringify({
                iamccs_lite_draft: true,
                note: "Original Comfy draft was too heavy for browser storage.",
                stripped_chars: text.length,
            }),
            stripped: text.length,
        };
    }
}

function pruneOtherWorkflowDrafts(currentKey) {
    let removed = 0;
    let chars = 0;
    try {
        const keys = [];
        for (let i = 0; i < localStorage.length; i += 1) {
            const key = localStorage.key(i);
            if (!key || key === currentKey) continue;
            if (isWorkflowDraftKey(key)) {
                const value = localStorage.getItem(key) || "";
                keys.push({ key, length: value.length });
            }
        }
        keys.sort((a, b) => b.length - a.length);
        for (const item of keys.slice(0, 6)) {
            const value = localStorage.getItem(item.key) || "";
            chars += value.length;
            localStorage.removeItem(item.key);
            removed += 1;
        }
    } catch (err) {
        console.warn("[IAMCCS] Draft pruning during quota rescue failed", err);
    }
    return { removed, chars };
}

function installLiteDraftFallback() {
    if (window.__IAMCCS_LITE_DRAFT_FALLBACK_INSTALLED__) return;
    window.__IAMCCS_LITE_DRAFT_FALLBACK_INSTALLED__ = true;
    const nativeSetItem = Storage.prototype.setItem;
    Storage.prototype.setItem = function iamccsSetItemWithDraftFallback(key, value) {
        try {
            return nativeSetItem.call(this, key, value);
        } catch (err) {
            if (!isWorkflowDraftKey(key)) throw err;
            const pruned = pruneOtherWorkflowDrafts(String(key || ""));
            try {
                return nativeSetItem.call(this, key, value);
            } catch (secondErr) {
                const lite = makeLiteDraftPayload(value);
                if (!lite.stripped) throw secondErr;
                try {
                    nativeSetItem.call(this, key, lite.payload);
                    console.warn("[IAMCCS] Saved lite workflow draft after browser quota error.", {
                        key,
                        stripped: lite.stripped,
                        pruned,
                    });
                    return undefined;
                } catch (thirdErr) {
                    console.error("[IAMCCS] Lite workflow draft fallback failed.", { key, pruned, thirdErr });
                    throw thirdErr;
                }
            }
        }
    };
}

app.registerExtension({
    name: "iamccs.workflow_persist_cleanup",
    async setup() {
        installWorkflowDraftRescue();
    },
});

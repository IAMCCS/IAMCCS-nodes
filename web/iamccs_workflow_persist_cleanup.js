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
    return text.includes("data:audio") || text.includes("data:video") || text.includes(";base64,");
}

function isRegenerableDraftCacheKey(key) {
    const normalized = String(key || "").toLowerCase().replace(/[^a-z0-9]/g, "");
    return normalized.includes("waveformpeaks") ||
        normalized.includes("waveformcache") ||
        normalized.includes("decodedwaveform") ||
        normalized.includes("audiobuffercache");
}

function makeLiteDraftPayload(payload) {
    try {
        const parsed = JSON.parse(String(payload || "{}"));
        let stripped = 0;
        const visit = (value, key = "", depth = 0) => {
            const lowerKey = String(key || "").toLowerCase();
            if (Array.isArray(value)) {
                if (isRegenerableDraftCacheKey(key)) {
                    stripped += JSON.stringify(value).length;
                    return [];
                }
                if (lowerKey === "edits" && value.length > 80) {
                    const kept = value.slice(-80);
                    stripped += Math.max(0, JSON.stringify(value).length - JSON.stringify(kept).length);
                    return kept.map((item) => visit(item, key, depth + 1));
                }
                return value.map((item) => visit(item, key, depth + 1));
            }
            if (typeof value === "string") {
                if (
                    lowerKey.includes("b64") ||
                    lowerKey.includes("base64") ||
                    looksLikeHeavyMediaString(value)
                ) {
                    stripped += value.length;
                    return `[IAMCCS stripped embedded media: ${value.length} chars]`;
                }
                const trimmed = value.trim();
                if (depth < 10 && value.length > 2048 && (trimmed.startsWith("{") || trimmed.startsWith("["))) {
                    try {
                        const embedded = JSON.parse(value);
                        const compacted = JSON.stringify(visit(embedded, `${key}_embedded`, depth + 1));
                        stripped += Math.max(0, value.length - compacted.length);
                        return compacted;
                    } catch {}
                }
                return value;
            }
            if (value && typeof value === "object") {
                for (const objectKey of Object.keys(value)) {
                    value[objectKey] = visit(value[objectKey], objectKey, depth + 1);
                }
            }
            return value;
        };
        const lite = visit(parsed);
        return { payload: JSON.stringify(lite), stripped };
    } catch {
        // A malformed draft must never be replaced with a placeholder: ComfyUI
        // would remove it on the next refresh because it is no longer a workflow.
        return { payload: String(payload || ""), stripped: 0 };
    }
}

function compactStoredWorkflowDrafts(currentKey, nativeSetItem) {
    let compacted = 0;
    let chars = 0;
    try {
        for (let i = 0; i < localStorage.length; i += 1) {
            const key = localStorage.key(i);
            if (!key || key === currentKey || !isWorkflowDraftKey(key)) continue;
            const value = localStorage.getItem(key) || "";
            const lite = makeLiteDraftPayload(value);
            if (!lite.stripped || lite.payload.length >= value.length) continue;
            nativeSetItem.call(localStorage, key, lite.payload);
            compacted += 1;
            chars += value.length - lite.payload.length;
        }
    } catch (err) {
        console.warn("[IAMCCS] Existing draft compaction during quota rescue failed", err);
    }
    return { compacted, chars };
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
            const compactedExisting = compactStoredWorkflowDrafts(String(key || ""), nativeSetItem);
            try {
                return nativeSetItem.call(this, key, value);
            } catch (secondErr) {
                const lite = makeLiteDraftPayload(value);
                if (!lite.stripped || lite.payload.length >= String(value || "").length) throw secondErr;
                try {
                    nativeSetItem.call(this, key, lite.payload);
                    console.warn("[IAMCCS] Saved valid compact workflow draft after browser quota error.", {
                        key,
                        stripped: lite.stripped,
                        compactedExisting,
                    });
                    return undefined;
                } catch (thirdErr) {
                    console.error("[IAMCCS] Compact workflow draft fallback failed without deleting other drafts.", {
                        key,
                        compactedExisting,
                        thirdErr,
                    });
                    throw thirdErr;
                }
            }
        }
    };
    const startupCompaction = compactStoredWorkflowDrafts("", nativeSetItem);
    if (startupCompaction.compacted) {
        console.info("[IAMCCS] Compacted existing valid workflow drafts without deleting them.", startupCompaction);
    }
}

app.registerExtension({
    name: "iamccs.workflow_persist_cleanup",
    async setup() {
        installWorkflowDraftRescue();
    },
});

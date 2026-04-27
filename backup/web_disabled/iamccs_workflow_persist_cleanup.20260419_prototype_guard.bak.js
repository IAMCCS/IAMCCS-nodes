import { app } from "../../scripts/app.js";

const LEGACY_LOCAL_KEYS = [
    "workflow",
    "Comfy.OpenWorkflowsPaths",
    "Comfy.ActiveWorkflowIndex",
];

const LEGACY_PREFIXES = [
    "Comfy.Workflow.Drafts",
    "Comfy.Workflow.DraftOrder",
];

const V2_PREFIX = {
    draftIndex: "Comfy.Workflow.DraftIndex.v2:",
    draftPayload: "Comfy.Workflow.Draft.v2:",
    lastActivePath: "Comfy.Workflow.LastActivePath:",
    lastOpenPaths: "Comfy.Workflow.LastOpenPaths:",
};

const MAX_DRAFTS_PER_WORKSPACE = 4;
const MAX_DRAFT_CHARS_PER_WORKSPACE = 2_500_000;
const STORAGE_PATCH_FLAG = "__iamccsWorkflowDraftQuotaPatchApplied";

function safeParseJson(json) {
    if (!json) return null;
    try {
        return JSON.parse(json);
    } catch {
        return null;
    }
}

function listStorageKeys(storage) {
    const keys = [];
    try {
        for (let index = 0; index < storage.length; index += 1) {
            const key = storage.key(index);
            if (key) keys.push(key);
        }
    } catch {
        return [];
    }
    return keys;
}

function removeKey(storage, key) {
    try {
        storage.removeItem(key);
        return true;
    } catch {
        return false;
    }
}

function getPayloadKey(workspaceId, draftKey) {
    return `${V2_PREFIX.draftPayload}${workspaceId}:${draftKey}`;
}

function extractWorkspaceIdFromKey(key) {
    if (typeof key !== "string") return null;
    if (key.startsWith(V2_PREFIX.draftIndex)) {
        return key.slice(V2_PREFIX.draftIndex.length);
    }
    if (key.startsWith(V2_PREFIX.draftPayload)) {
        const rest = key.slice(V2_PREFIX.draftPayload.length);
        const separatorIndex = rest.indexOf(":");
        return separatorIndex === -1 ? null : rest.slice(0, separatorIndex);
    }
    if (key.startsWith(V2_PREFIX.lastActivePath)) {
        return key.slice(V2_PREFIX.lastActivePath.length);
    }
    if (key.startsWith(V2_PREFIX.lastOpenPaths)) {
        return key.slice(V2_PREFIX.lastOpenPaths.length);
    }
    return null;
}

function isDraftStorageKey(key) {
    return typeof key === "string" && (
        key.startsWith(V2_PREFIX.draftIndex)
        || key.startsWith(V2_PREFIX.draftPayload)
        || key.startsWith(V2_PREFIX.lastActivePath)
        || key.startsWith(V2_PREFIX.lastOpenPaths)
        || LEGACY_LOCAL_KEYS.includes(key)
        || LEGACY_PREFIXES.some((prefix) => key.startsWith(prefix))
    );
}

function isQuotaExceededError(error) {
    if (!error) return false;
    const name = String(error.name || "");
    const message = String(error.message || "").toLowerCase();
    return name === "QuotaExceededError"
        || name === "NS_ERROR_DOM_QUOTA_REACHED"
        || message.includes("quota")
        || message.includes("storage is full");
}

function cleanupWorkspaceTrackingStorage(localStorageRef, workspaceId = null) {
    let removed = 0;
    for (const key of listStorageKeys(localStorageRef)) {
        const isTrackingKey = key.startsWith(V2_PREFIX.lastActivePath)
            || key.startsWith(V2_PREFIX.lastOpenPaths);
        if (!isTrackingKey) continue;

        if (workspaceId !== null && extractWorkspaceIdFromKey(key) !== workspaceId) {
            continue;
        }

        if (removeKey(localStorageRef, key)) removed += 1;
    }
    return removed;
}

function cleanupLegacyWorkflowStorage(localStorageRef) {
    let removed = 0;
    for (const key of LEGACY_LOCAL_KEYS) {
        if (removeKey(localStorageRef, key)) removed += 1;
    }

    for (const key of listStorageKeys(localStorageRef)) {
        if (LEGACY_PREFIXES.some((prefix) => key.startsWith(prefix))) {
            if (removeKey(localStorageRef, key)) removed += 1;
        }
    }

    return removed;
}

function cleanupWorkspaceDrafts(localStorageRef, workspaceId) {
    const indexKey = `${V2_PREFIX.draftIndex}${workspaceId}`;
    const rawIndex = localStorageRef.getItem(indexKey);
    if (!rawIndex) return { removed: 0, rewritten: false };

    const index = safeParseJson(rawIndex);
    if (!index || typeof index !== "object" || !Array.isArray(index.order) || typeof index.entries !== "object" || index.entries == null) {
        let removed = removeKey(localStorageRef, indexKey) ? 1 : 0;
        const payloadPrefix = `${V2_PREFIX.draftPayload}${workspaceId}:`;
        for (const key of listStorageKeys(localStorageRef)) {
            if (key.startsWith(payloadPrefix) && removeKey(localStorageRef, key)) {
                removed += 1;
            }
        }
        return { removed, rewritten: false };
    }

    const payloadPrefix = `${V2_PREFIX.draftPayload}${workspaceId}:`;
    const seenPayloads = new Set();
    const keptEntries = {};
    const keptOrder = [];
    let keptChars = 0;
    let keptCount = 0;
    let removed = 0;

    const newestFirst = [...index.order].reverse();
    for (const draftKey of newestFirst) {
        const entry = index.entries[draftKey];
        if (!entry || typeof entry.path !== "string") continue;

        const payloadKey = getPayloadKey(workspaceId, draftKey);
        const payloadJson = localStorageRef.getItem(payloadKey);
        if (!payloadJson) continue;

        const payloadChars = payloadJson.length;
        const canKeep = keptCount < MAX_DRAFTS_PER_WORKSPACE && (
            keptCount === 0 || keptChars + payloadChars <= MAX_DRAFT_CHARS_PER_WORKSPACE
        );

        if (canKeep) {
            keptEntries[draftKey] = entry;
            keptOrder.unshift(draftKey);
            keptChars += payloadChars;
            keptCount += 1;
            seenPayloads.add(payloadKey);
        } else if (removeKey(localStorageRef, payloadKey)) {
            removed += 1;
        }
    }

    for (const key of listStorageKeys(localStorageRef)) {
        if (key.startsWith(payloadPrefix) && !seenPayloads.has(key)) {
            if (removeKey(localStorageRef, key)) removed += 1;
        }
    }

    const nextIndex = {
        ...index,
        updatedAt: Date.now(),
        order: keptOrder,
        entries: keptEntries,
    };

    const nextIndexJson = JSON.stringify(nextIndex);
    const rewritten = nextIndexJson !== rawIndex;
    if (rewritten) {
        try {
            localStorageRef.setItem(indexKey, nextIndexJson);
        } catch {
            // If index rewrite fails, leaving the trimmed payload set is still better
            // than keeping localStorage saturated with stale drafts.
        }
    }

    return { removed, rewritten };
}

function cleanupWorkflowPersistenceStorage() {
    try {
        const localStorageRef = window.localStorage;
        if (!localStorageRef) return;

        let removed = cleanupLegacyWorkflowStorage(localStorageRef);
        removed += cleanupWorkspaceTrackingStorage(localStorageRef);
        let rewritten = 0;

        const workspaceIds = new Set();
        for (const key of listStorageKeys(localStorageRef)) {
            if (key.startsWith(V2_PREFIX.draftIndex)) {
                workspaceIds.add(key.slice(V2_PREFIX.draftIndex.length));
            }
        }

        for (const workspaceId of workspaceIds) {
            const result = cleanupWorkspaceDrafts(localStorageRef, workspaceId);
            removed += result.removed;
            if (result.rewritten) rewritten += 1;
        }

        if (removed || rewritten) {
            console.info("[IAMCCS] Workflow draft storage cleanup applied", {
                removedKeys: removed,
                rewrittenIndexes: rewritten,
            });
        }
    } catch (error) {
        console.warn("[IAMCCS] Workflow draft storage cleanup skipped", error);
    }
}

function recoverDraftStorage(localStorageRef, key) {
    let removed = cleanupLegacyWorkflowStorage(localStorageRef);
    let rewritten = 0;

    const workspaceIds = new Set();
    const workspaceId = extractWorkspaceIdFromKey(key);
    if (workspaceId) workspaceIds.add(workspaceId);

    for (const storageKey of listStorageKeys(localStorageRef)) {
        if (storageKey.startsWith(V2_PREFIX.draftIndex)) {
            workspaceIds.add(storageKey.slice(V2_PREFIX.draftIndex.length));
        }
    }

    for (const currentWorkspaceId of workspaceIds) {
        removed += cleanupWorkspaceTrackingStorage(localStorageRef, currentWorkspaceId);
        const result = cleanupWorkspaceDrafts(localStorageRef, currentWorkspaceId);
        removed += result.removed;
        if (result.rewritten) rewritten += 1;
    }

    return { removed, rewritten };
}

function aggressiveRecoverDraftStorage(localStorageRef) {
    let removed = cleanupLegacyWorkflowStorage(localStorageRef);
    removed += cleanupWorkspaceTrackingStorage(localStorageRef);

    for (const storageKey of listStorageKeys(localStorageRef)) {
        if (
            storageKey.startsWith(V2_PREFIX.draftIndex)
            || storageKey.startsWith(V2_PREFIX.draftPayload)
        ) {
            if (removeKey(localStorageRef, storageKey)) removed += 1;
        }
    }

    return { removed, rewritten: 0 };
}

function installDraftStorageQuotaGuard() {
    try {
        const localStorageRef = window.localStorage;
        if (!localStorageRef || localStorageRef[STORAGE_PATCH_FLAG]) return;

        const originalSetItem = localStorageRef.setItem.bind(localStorageRef);

        localStorageRef.setItem = function patchedSetItem(key, value) {
            try {
                return originalSetItem(key, value);
            } catch (error) {
                if (!isDraftStorageKey(key) || !isQuotaExceededError(error)) {
                    throw error;
                }

                const recovery = recoverDraftStorage(localStorageRef, key);
                console.warn("[IAMCCS] Workflow draft quota reached, attempting recovery", {
                    key,
                    removedKeys: recovery.removed,
                    rewrittenIndexes: recovery.rewritten,
                });

                try {
                    const result = originalSetItem(key, value);
                    console.info("[IAMCCS] Workflow draft save recovered after storage cleanup", {
                        key,
                        removedKeys: recovery.removed,
                        rewrittenIndexes: recovery.rewritten,
                    });
                    return result;
                } catch (retryError) {
                    if (isQuotaExceededError(retryError)) {
                        const aggressiveRecovery = aggressiveRecoverDraftStorage(localStorageRef);
                        console.warn("[IAMCCS] Workflow draft quota still exceeded, applying aggressive recovery", {
                            key,
                            removedKeys: aggressiveRecovery.removed,
                        });

                        try {
                            const result = originalSetItem(key, value);
                            console.info("[IAMCCS] Workflow draft save recovered after aggressive cleanup", {
                                key,
                                removedKeys: recovery.removed + aggressiveRecovery.removed,
                                rewrittenIndexes: recovery.rewritten,
                            });
                            return result;
                        } catch (finalRetryError) {
                            console.error("[IAMCCS] Workflow draft save still exceeds storage quota after aggressive cleanup", {
                                key,
                                removedKeys: recovery.removed + aggressiveRecovery.removed,
                                rewrittenIndexes: recovery.rewritten,
                                retryError: finalRetryError,
                            });
                            throw finalRetryError;
                        }
                    }

                    console.error("[IAMCCS] Workflow draft save still exceeds storage quota after cleanup", {
                        key,
                        removedKeys: recovery.removed,
                        rewrittenIndexes: recovery.rewritten,
                        retryError,
                    });
                    throw retryError;
                }
            }
        };

        Object.defineProperty(localStorageRef, STORAGE_PATCH_FLAG, {
            value: true,
            configurable: false,
            enumerable: false,
            writable: false,
        });
    } catch (error) {
        console.warn("[IAMCCS] Workflow draft quota guard skipped", error);
    }
}

app.registerExtension({
    name: "iamccs.workflow_persist_cleanup",
    async setup() {
        cleanupWorkflowPersistenceStorage();
        installDraftStorageQuotaGuard();
    },
});

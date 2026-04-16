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

app.registerExtension({
    name: "iamccs.workflow_persist_cleanup",
    async setup() {
        cleanupWorkflowPersistenceStorage();
    },
});

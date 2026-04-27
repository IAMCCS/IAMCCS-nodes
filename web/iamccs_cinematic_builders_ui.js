import { app } from "../../scripts/app.js";

const CINEMATIC_BUILDERS = new Set([
    "IAMCCS_LTX2_CinematicPromptComposer",
    "IAMCCS_LTX2_CinematicShotLineBuilder",
    "IAMCCS_LTX2_CinematicV2VTimelineLineBuilder",
    "IAMCCS_LTX2_CinematicLineStacker",
]);

const SIZE_PRESETS = {
    IAMCCS_LTX2_CinematicPromptComposer: [520, 620],
    IAMCCS_LTX2_CinematicShotLineBuilder: [560, 760],
    IAMCCS_LTX2_CinematicV2VTimelineLineBuilder: [590, 840],
    IAMCCS_LTX2_CinematicLineStacker: [520, 520],
};

function tuneWidgets(node) {
    if (!node?.widgets) return;
    for (const widget of node.widgets) {
        if (!widget) continue;
        const name = String(widget.name || "").toLowerCase();
        if (
            name.includes("prompt") ||
            name.includes("dialogue") ||
            name.includes("voice") ||
            name.includes("continuity") ||
            name.includes("comment") ||
            name.startsWith("line_")
        ) {
            widget.computeSize = function(width) {
                return [width, name.startsWith("line_") ? 70 : 96];
            };
        }
    }
}

app.registerExtension({
    name: "iamccs.cinematic.builders.ui",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        const name = nodeData?.name || "";
        if (!CINEMATIC_BUILDERS.has(name)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = onNodeCreated?.apply(this, arguments);
            try {
                this.color = "#243447";
                this.bgcolor = "#18232f";
                this.boxcolor = "#4f9fd8";
                this.shape = typeof LiteGraph !== "undefined" ? LiteGraph.BOX_SHAPE : 0;

                const preset = SIZE_PRESETS[name];
                if (preset && (!this.size || this.size[0] < preset[0])) {
                    this.size = [...preset];
                }

                tuneWidgets(this);
                this.setDirtyCanvas(true, true);
            } catch {
                // UI-only enhancement; never block node loading.
            }
            return result;
        };
    },
});

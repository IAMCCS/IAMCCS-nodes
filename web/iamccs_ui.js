import { app } from "../../scripts/app.js";

function isIamccsNode(nodeData) {
    const name = nodeData?.name || "";
    return name.startsWith("IAMCCS_") || name.startsWith("iamccs_");
}

app.registerExtension({
    name: "iamccs.ui.box_shape",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isIamccsNode(nodeData)) return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            try {
                // Force standard rectangular nodes
                if (typeof LiteGraph !== "undefined" && LiteGraph?.BOX_SHAPE != null) {
                    this.shape = LiteGraph.BOX_SHAPE;
                } else {
                    // Fallback: 0 is BOX in most LiteGraph builds
                    this.shape = 0;
                }
                this.setDirtyCanvas(true, true);
            } catch {
                // ignore
            }
            return r;
        };
    },
});

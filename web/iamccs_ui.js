import { app } from "../../scripts/app.js";

function isIamccsNode(nodeData) {
    const name = nodeData?.name || "";
    return name.startsWith("IAMCCS_") || name.startsWith("iamccs_");
}

app.registerExtension({
    name: "iamccs.ui.box_shape",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!isIamccsNode(nodeData)) return;

        const addDOMWidget = nodeType.prototype.addDOMWidget;
        if (typeof addDOMWidget === "function" && !nodeType.prototype._iamccsStableDOMWidgetSelection) {
            Object.defineProperty(nodeType.prototype, "_iamccsStableDOMWidgetSelection", {
                value: true,
            });
            nodeType.prototype.addDOMWidget = function (name, type, element, options = {}) {
                const stableOptions = Object.prototype.hasOwnProperty.call(options, "selectOn")
                    ? options
                    : { ...options, selectOn: [] };
                const widget = addDOMWidget.call(this, name, type, element, stableOptions);

                // ComfyUI's Parameters panel renders the same legacy widget in a
                // narrow preview and writes that preview width back to the widget.
                // DOM widgets are full-node UIs, so keep width unset and let
                // DomWidgets.vue use the owning node width instead.
                if (widget) {
                    Object.defineProperty(widget, "width", {
                        configurable: true,
                        get: () => undefined,
                        set: () => {},
                    });
                }

                return widget;
            };
        }

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

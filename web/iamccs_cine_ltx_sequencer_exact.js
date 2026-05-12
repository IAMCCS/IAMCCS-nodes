import { app } from "../../scripts/app.js";

const CINE_FILM_LAB = {
    header: "#2E2A24",
    nodeBg: "#171512",
    guide: "#D6A85A",
};

function hideWidget(widget, hidden) {
    if (!widget) return;
    if (hidden) {
        if (widget.type !== "hidden") {
            widget._iamccsOrigType = widget.type;
            widget._iamccsOrigComputeSize = widget.computeSize;
            widget.type = "hidden";
            widget.computeSize = () => [0, -4];
        }
    } else if (widget._iamccsOrigType !== undefined) {
        widget.type = widget._iamccsOrigType;
        widget.computeSize = widget._iamccsOrigComputeSize;
        delete widget._iamccsOrigType;
        delete widget._iamccsOrigComputeSize;
    }
}

function getCount(node) {
    const widget = node.widgets?.find((w) => w.name === "num_images");
    const raw = widget?.value ?? node.properties?.num_images ?? 1;
    const count = Number.parseInt(raw, 10);
    return Number.isFinite(count) ? Math.max(0, Math.min(50, count)) : 1;
}

function applyVisibleSlots(node) {
    const count = getCount(node);
    for (const widget of node.widgets || []) {
        const match = /^(insert_frame|insert_second|strength)_(\d+)$/.exec(widget.name || "");
        if (!match) continue;
        const index = Number.parseInt(match[2], 10);
        hideWidget(widget, index > count);
    }
    node.setDirtyCanvas(true, true);
    requestAnimationFrame(() => {
        if (!node.graph || !node.computeSize) return;
        const width = Math.max(node.size?.[0] || 320, 340);
        const size = node.computeSize();
        node.setSize([width, Math.max(160, size[1])]);
    });
}

app.registerExtension({
    name: "IAMCCS.CineLTXSequencerExact.CleanUI",
    async nodeCreated(node) {
        if (node.comfyClass !== "IAMCCS_CineLTXSequencerExact") return;

        node.properties = node.properties || {};
        node.color = CINE_FILM_LAB.header;
        node.bgcolor = CINE_FILM_LAB.nodeBg;
        node.boxcolor = CINE_FILM_LAB.guide;

        const originalOnConfigure = node.onConfigure;
        node.onConfigure = function(info) {
            originalOnConfigure?.apply(this, arguments);
            setTimeout(() => applyVisibleSlots(this), 50);
        };

        const originalOnAdded = node.onAdded;
        node.onAdded = function() {
            originalOnAdded?.apply(this, arguments);
            setTimeout(() => applyVisibleSlots(this), 50);
        };

        setTimeout(() => {
            const num = node.widgets?.find((w) => w.name === "num_images");
            if (num) {
                const oldCallback = num.callback;
                num.callback = (value, canvas, n, pos, event) => {
                    node.properties.num_images = value;
                    oldCallback?.(value, canvas, n, pos, event);
                    applyVisibleSlots(node);
                };
            }
            applyVisibleSlots(node);
        }, 100);
    },
});

export function upstreamDesigners(graph, startId) {
  const found = [], visited = new Set(), pending = [startId];
  while (pending.length) {
    const id = pending.pop();
    if (visited.has(String(id))) continue;
    visited.add(String(id));
    const node = graph.getNodeById(id);
    if (!node) continue;
    if ((node.comfyClass || node.type) === "IAMCCS_MOSSFrameDesigner") found.push(node);
    for (let slot = 0; slot < (node.inputs?.length || 0); slot++) {
      const link = node.getInputLink(slot);
      if (link) pending.push(link.origin_id);
    }
  }
  return found;
}

export function layoutSignature(design) {
  const scene = design.scene || {};
  return JSON.stringify([
    +design.canvas.width, +design.canvas.height,
    ["high_level_description", "background", "aesthetics", "lighting", "photo", "medium", "art_style"].map(key => scene[key] || ""),
    design.items.filter(item => !["image", "mask"].includes(item.kind)).map(item => [
      item.kind === "text" ? "text" : item.role || "object", item.desc || "", item.text || "",
      +item.x, +item.y, +item.w, +item.h, item.color_palette || [],
      item.style || "", item.text_mode || "native", +item.font_size || 0,
    ]),
  ]);
}

export function latestSavedRender(history, design) {
  const signature = layoutSignature(design);
  const records = Object.entries(history).sort((a, b) => (b[1].prompt?.[0] || 0) - (a[1].prompt?.[0] || 0));
  for (const [promptId, record] of records) {
    if (record.status?.status_str !== "success") continue;
    const prompt = record.prompt?.[2] || {};
    const graph = { getNodeById(id) {
      const entry = prompt[String(id)];
      if (!entry) return null;
      const links = Object.values(entry.inputs).filter(value => Array.isArray(value) && value.length === 2 && prompt[String(value[0])]);
      return { id: String(id), type: entry.class_type, inputs: links, getInputLink: slot => ({ origin_id: links[slot][0] }) };
    } };
    for (const [outputId, output] of Object.entries(record.outputs || {})) {
      if (!output.images?.length) continue;
      for (const node of upstreamDesigners(graph, outputId)) {
        const raw = prompt[node.id]?.inputs.design_data;
        if (typeof raw === "string" && layoutSignature(JSON.parse(raw)) === signature) return { images: output.images, prompt_id: promptId };
      }
    }
  }
  return null;
}

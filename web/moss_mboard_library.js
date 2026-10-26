// Mboard projects are JSON documents. ComfyUI may wrap node properties in
// reactive proxies, which structuredClone cannot accept. Snapshot their JSON
// representation before editing or passing values to IndexedDB.
export function cloneProjectData(value) {
  const encoded = JSON.stringify(value);
  if (encoded === undefined) throw new TypeError("MOSS project data must be a JSON document.");
  return JSON.parse(encoded);
}

// Validate the entire import before the editor changes its current project.
export function importedMboards(raw, normalize, filename = "Imported Mboard") {
  let entries;
  if (raw?.format === "IAMCCS.Mboard.pack") {
    if (raw.edition !== "MOSS" || raw.version !== 1 || !Array.isArray(raw.boards) || !raw.boards.length) {
      throw new Error("Unsupported MOSS Mboard pack.");
    }
    entries = raw.boards.map((board, index) => ({ name: board.name || `Mboard ${index + 1}`, project: board.project }));
  } else if (raw?.presets && typeof raw.presets === "object") {
    entries = Object.entries(raw.presets).map(([key, project]) => ({ name: project.label || key, project }));
  } else if (raw?.format === "IAMCCS.Mboard") {
    entries = [{ name: filename, project: raw }];
  } else return null;
  if (!entries.length) throw new Error("The imported collection has no Mboards.");
  return entries.map(({ name, project }, index) => ({
    key: `${index}:${String(name)}`,
    name: String(name),
    design: portableDesign(normalize(project)),
  }));
}

export function portableDesign(design) {
  const clean = cloneProjectData(design);
  if (clean.project_metadata) delete clean.project_metadata.workflow;
  if (clean.generated) delete clean.generated.data_url;
  return clean;
}

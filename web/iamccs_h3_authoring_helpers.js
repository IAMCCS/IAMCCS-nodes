// Shared authoring-only controls. Never queues a ComfyUI prompt.
const HELP = {
    t2va: ["Write the scene, subjects and one camera intention in GLOBAL.", "Use local text slots for successive beats. No reference image is required.", "For H3-generated speech, supply exact words with <Subject 1> (S1): <d>[Language] words</d>.", "Select Native Audio in Settings; review and inject before Queue."],
    i2va: ["Place each opening image in a main Shotboard slot and set its duration.", "GLOBAL defines identity/style; each LOCAL prompt names the active speaker, action and camera for that slot.", "For lipsync, align the matching AudioBoard clip to the slot and choose Audio Drive. Native Audio instead generates new sound.", "Independent slots are cuts; selecting I2VA does not promise a continuous camera bridge."],
    fl2va: ["Place first/last images on the Shotboard boundary slots.", "Describe the connecting action between the two images, not separate unrelated shots.", "Choose Stable Keyframes or Native AV Context in Settings; these are different handoff strategies.", "Audio Drive follows AudioBoard timing; exact speech across a boundary needs aligned audio, not only prompt tags."],
    ref2va: ["Connect reference images to the backend reference-image inputs, not as I2V opening frames.", "Define <Subject N> and each <Picture N> role in GLOBAL; reference pictures are identities/objects/style, not mandatory camera frames.", "Use LOCAL text slots for successive actions and the speaker visible in each shot.", "For exact lipsync use Ref2Vid LipSync + aligned AudioBoard. Voice-timbre reference alone is not locked lipsync."],
    longvid_guides: ["Put positioned image guides in the main Shotboard timeline; their timing is guide timing, not guaranteed hard editing cuts.", "Write the shared sequence in GLOBAL and the action/speaker in LOCAL prompts.", "Choose generated audio or the explicit guided Audio Drive route in Settings. Prompt tags alone cannot lock audio.", "Check guide positions, slot durations and audio before Queue; this is not REF2VA identity-reference mode."],
    multi_shot_lipsync: ["Put one opening guide per intended shot in the main timeline; keep each speaking face readable.", "Place the continuous performance in AudioBoard and align clips with the slot magnets.", "GLOBAL fixes identities and world; LOCAL names the active speaker and silent listener for each shot.", "Select Multi-Shot LipSync / R37. Technical H3 chunks can exceed the image count; their audio positions remain on the same clock.", "Review and inject prompts. Motion Context does not guarantee a visual morph between different images."],
    audio_driven: ["Choose the visual mode first: I2VA opening image, REF2VA external identity references or LongVid positioned guides.", "Place performance audio in AudioBoard and align it to its intended visual slot.", "Write active speaker, visible mouth and actions; never invent a transcript that differs from the recording.", "Select Audio Drive. A final audio mux or voice-timbre reference does not by itself drive lipsync."],
    v2va_object_swap: ["Connect the source video to the backend video input; connect replacement identity images to reference inputs.", "GLOBAL separates source motion/camera to retain from the person, environment or style to replace.", "LOCAL text describes each source interval. Keep source offsets and duration explicit.", "Choose source-audio retention or aligned AudioBoard drive in Settings; a reference image alone does not supply motion."],
};
export function modeHelpKey(mode) {
    return ({longvid_motion_context: "multi_shot_lipsync", longvid_guided_lipsync: "longvid_guides",
        longvid_ref2vid_lipsync: "longvid_guides", ref2vid_lipsync: "ref2va", auto_from_timeline: "i2va"})[mode] || mode;
}
export function modeHelp(mode) { return HELP[modeHelpKey(mode)] || HELP.i2va; }
export function createModeHelper(mode) {
    const details = document.createElement("details"); details.open = true;
    details.style.cssText = "border:1px solid #4c9f9b;background:#112b2e;color:#d9f3ef;padding:10px;margin:8px 0;line-height:1.5;font-size:12px;overflow-wrap:anywhere;grid-column:1/-1";
    const summary = document.createElement("summary"); summary.textContent = "QUICK START · " + String(mode).replaceAll("_", " ").toUpperCase();
    summary.style.cssText = "font-weight:800;cursor:pointer";
    const list = document.createElement("ol"); list.style.cssText = "padding-left:22px;margin:8px 0";
    const helpSelect = document.createElement("select");
    helpSelect.setAttribute("aria-label", "Workflow help · instructions only, does not change generation mode");
    helpSelect.title="Instructions only. Select the real generation mode in IAMCCS H3 Settings.";
    helpSelect.style.cssText="max-width:100%;margin-top:8px;padding:5px;background:#183c40;color:#d9f3ef;border:1px solid #4c9f9b";
    for (const [key,label] of Object.entries({t2va:"T2VA",i2va:"I2VA",fl2va:"FL2VA",ref2va:"REF2VA / LipSync",longvid_guides:"LongVid positioned guides",multi_shot_lipsync:"Multi-Shot LipSync",audio_driven:"Audio Driven",v2va_object_swap:"V2VA"})) helpSelect.append(new Option(label,key));
    helpSelect.value=modeHelpKey(mode);
    const refresh = () => { summary.textContent="QUICK START · " + helpSelect.options[helpSelect.selectedIndex].text; list.replaceChildren(); for (const step of modeHelp(helpSelect.value)) { const li = document.createElement("li"); li.textContent = step; list.append(li); } };
    helpSelect.onchange=refresh; refresh();
    const note = document.createElement("div");
    note.textContent = "AI reference pictures describe the prompt only. They are not automatically connected to the generation backend. AI buttons edit text; they never start Queue.";
    note.style.color = "#9bd0cb";
    details.append(summary, helpSelect, list, note); return details;
}
export function beginAiBusy(button) {
    if (!document.getElementById("iamccs-ai-progress-style")) {
        const style = document.createElement("style"); style.id = "iamccs-ai-progress-style";
        style.textContent = "@keyframes iamccs-ai-spin{to{transform:rotate(360deg)}} .iamccs-ai-spinner{display:inline-block;width:12px;height:12px;border:2px solid #ffffff55;border-top-color:#ffe19a;border-radius:50%;animation:iamccs-ai-spin .8s linear infinite;flex:none} button[aria-busy=true]{display:inline-flex!important;align-items:center;justify-content:center;gap:6px;white-space:nowrap!important;opacity:1!important;color:#ffe19a!important}";
        document.head.append(style);
    }
    const original = [...button.childNodes], started = Date.now();
    const spinner = document.createElement("span"); spinner.className = "iamccs-ai-spinner"; spinner.setAttribute("aria-hidden", "true");
    const text = document.createElement("span"); text.textContent = "AI · 0s";
    button.replaceChildren(spinner, text); button.disabled = true; button.setAttribute("aria-busy", "true");
    const timer = setInterval(() => { text.textContent = `AI · ${Math.floor((Date.now() - started) / 1000)}s`; }, 1000);
    return () => { clearInterval(timer); button.replaceChildren(...original); button.disabled = false; button.removeAttribute("aria-busy"); };
}

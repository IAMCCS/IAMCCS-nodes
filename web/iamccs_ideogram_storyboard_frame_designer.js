import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
// By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com

const TYPE = "IAMCCS_StoryboardFrameDesigner";
const TYPE_V2 = "IAMCCS_StoryboardFrameDesignerV2";
const TYPE_JSON_PASS = "IAMCCS_IdeogramJSONPreviewPass";
const STYLE_ID = "iamccs-ideogram-storyboard-frame-style";
const IDEOBOARD_SCHEMA = "iamccs.ideoboard.package";
const JS_BUILD = "IAMCCS-INTERNAL-GEMMA4-ASSISTANT-APPLY-20260622";
console.info("[IAMCCS FrameDesigner] loaded", JS_BUILD, import.meta.url);

let lastIamccsResultImage = null;
let lastIamccsStoryboardSourceImage = null;
let lastIamccsPrimaryResultImage = null;
let iamccsActivePromptKey = "";
const iamccsLiveFrameNodes = new Set();
const iamccsFrameDesignerNodes = new Set();

function iamccsResultViewUrl(img) {
  if (!img) return "";
  const params = new URLSearchParams({
    filename: img.filename || "",
    subfolder: img.subfolder || "",
    type: img.type || "output",
  });
  return `/view?${params.toString()}`;
}

function iamccsIsStoryboardSourceImage(img) {
  const text = `${img?.subfolder || ""}/${img?.filename || ""}`.toLowerCase();
  return text.includes("sheet_framev2_ideogram") && text.includes("source_sheet");
}

function iamccsShouldIgnoreAutoCompareImage(img) {
  const text = `${img?.subfolder || ""}/${img?.filename || ""}`.toLowerCase();
  return (
    text.includes("inpaint_source") ||
    text.includes("_debug/") ||
    text.includes("/debug/") ||
    text.includes("inpaint_debug") ||
    text.includes("masked_pixels") ||
    text.includes("mask_preview")
  );
}

function iamccsEventPromptKey(event) {
  return String(
    event?.detail?.prompt_id ||
    event?.detail?.promptId ||
    event?.detail?.prompt?.id ||
    iamccsActivePromptKey ||
    "default"
  );
}

function iamccsEventNode(event) {
  const rawId = event?.detail?.node ?? event?.detail?.node_id ?? event?.detail?.nodeId;
  if (rawId == null) return null;
  const id = Number(rawId);
  const graph = app?.graph;
  return graph?.getNodeById?.(id) || graph?._nodes_by_id?.[id] || graph?._nodes_by_id?.[String(rawId)] || null;
}

function iamccsIsSaveImageEvent(event) {
  const graphNode = iamccsEventNode(event);
  const type = String(
    graphNode?.type ||
    graphNode?.comfyClass ||
    graphNode?.properties?.["Node name for S&R"] ||
    ""
  ).toLowerCase();
  return type === "saveimage" || type.endsWith(".saveimage");
}

try {
  api?.addEventListener?.("execution_start", (event) => {
    iamccsActivePromptKey = String(event?.detail?.prompt_id || event?.detail?.promptId || Date.now());
    lastIamccsPrimaryResultImage = null;
    lastIamccsStoryboardSourceImage = null;
  });
  api?.addEventListener?.("executed", (event) => {
    const images = event?.detail?.output?.images;
    if (!Array.isArray(images) || !images.length) return;
    if (!iamccsIsSaveImageEvent(event)) return;
    const img = images[0];
    if (iamccsShouldIgnoreAutoCompareImage(img)) return;
    const promptKey = iamccsEventPromptKey(event);
    if (lastIamccsPrimaryResultImage && promptKey === iamccsActivePromptKey) return;
    iamccsActivePromptKey = promptKey;
    lastIamccsPrimaryResultImage = img;
    lastIamccsResultImage = img;
    if (iamccsIsStoryboardSourceImage(img)) lastIamccsStoryboardSourceImage = img;
    const url = iamccsResultViewUrl(img);
    iamccsLiveFrameNodes.forEach((node) => node?._iamccsSetResultBackground?.(url, true));
    // FrameDesigner compare overlays are intentionally disabled: results must not
    // inject extra UI/image layers into the editable canvas.
  });
  api?.addEventListener?.("b_preview", (event) => {
    // Do not use live sampler previews as the under-box image. They can be
    // sigma/debug previews and would replace the real first SaveImage result.
  });
} catch {}


const DENOISE_PRESETS = [
  {
    key: "preserve",
    label: "Preserve",
    value: 0.22,
    hint: "Clean pixels and texture while holding face, costume, pose, and framing.",
  },
  {
    key: "refine",
    label: "Refine",
    value: 0.28,
    hint: "Storyboard default: sharper hi-res version with strong consistency.",
  },
  {
    key: "cinema",
    label: "Cinema",
    value: 0.38,
    hint: "More lighting and material improvement; still keeps the shot readable.",
  },
  {
    key: "reimagine",
    label: "Reimagine",
    value: 0.55,
    hint: "Creative rebuild; useful when the crop is weak, less consistent.",
  },
];

const PRESETS = {
  storyboard: {
  "label": "Storyboard - Frozen Planet Astronaut 2x3",
  "summary": "Six varied photoreal sci-fi survival beats on a frozen exoplanet.",
  "workflow_mode": "storyboard_grid",
  "grid_key": "story_2x3",
  "target_resolution_key": "hd_720",
  "preview": {
    "kind": "storyboard_2x3",
    "label": "FROZEN 2x3"
  },
  "canvas": {
    "width": 2560,
    "height": 2160,
    "aspect_label": "1280 x 720 - HD 16:9 target / Storyboard 2x3 canvas 2560x2160",
    "target_resolution_key": "hd_720",
    "target_width": 1280,
    "target_height": 720
  },
  "scene": {
    "high_level_description": "A six-panel 2x3 cinematic storyboard contact sheet set on a frozen exoplanet, following one exhausted astronaut in a strange partly unsealed expedition suit crossing black ice plains, ruined research vehicles, and distant alien towers under two visible planets in the sky.",
    "aesthetics": "extremely photoreal live-action science fiction cinema, tactile frozen surfaces, strange worn astronaut suit, partly open asymmetric suit collar, visible inner thermal layers, practical space gear, believable human fatigue, cinematic continuity",
    "lighting": "cold blue planetary daylight, low amber rim light on ice crystals, breath vapor, long shadows, readable eyes, crisp silhouettes, clean subject separation in every panel",
    "photo": "35mm anamorphic film stills, varied shot scale, wide exoplanet landscape, tight medium portrait, top-down equipment detail, medium action frame, long telephoto silhouette, macro visor reflection, fine analog grain",
    "medium": "photograph",
    "art_style": "",
    "color_palette": [
      "#06111A",
      "#557C8A",
      "#A8D8E8",
      "#E6E1C6",
      "#E87C45",
      "#07080A"
    ],
    "background": "Clean storyboard contact sheet with exactly two columns and three rows. Six separate 16:9 cinematic stills show different beats of the same frozen-planet sequence with thin panel borders, readable horizon lines, alien planets in the sky, and clear physical action."
  },
  "i2i": {
    "enabled": false,
    "denoise": 0.28,
    "low_sigma_start_step": 12,
    "scheduler_hint": "Storyboard grid generation. Use AutoCropGrid with 2 columns and 3 rows.",
    "source_mode": "canvas_composite"
  },
  "items": [
    {
      "id": "panel_01",
      "kind": "obj",
      "label": "Panel 1 - Ice Plain Arrival",
      "text": "",
      "x": 0,
      "y": 0,
      "w": 500,
      "h": 333,
      "desc": "ICE PLAIN ARRIVAL: wide establishing shot of a frozen exoplanet plain at sunrise, black ice road crossing toward ruined alien towers, one small astronaut figure walking alone, two planets hanging in the pale sky, long shadow, cold blue atmosphere.",
      "color_palette": [
        "#557C8A",
        "#A8D8E8"
      ]
    },
    {
      "id": "panel_02",
      "kind": "obj",
      "label": "Panel 2 - Broken Suit Collar",
      "text": "",
      "x": 500,
      "y": 0,
      "w": 500,
      "h": 333,
      "desc": "BROKEN SUIT COLLAR: tight medium portrait of the same exhausted astronaut pulling a frost-covered breathing scarf across the mouth, strange expedition suit partly unsealed at the collar, inner thermal fabric visible, breath vapor around focused eyes.",
      "color_palette": [
        "#A8D8E8",
        "#E87C45"
      ]
    },
    {
      "id": "panel_03",
      "kind": "obj",
      "label": "Panel 3 - Frozen Nav Module",
      "text": "",
      "x": 0,
      "y": 333,
      "w": 500,
      "h": 334,
      "desc": "FROZEN NAV MODULE: top-down close detail of a cracked wrist computer, circular oxygen gauge, folded star map, metal sample case, and frost crystals on white-blue ice, precise object clarity.",
      "color_palette": [
        "#06111A",
        "#557C8A"
      ]
    },
    {
      "id": "panel_04",
      "kind": "obj",
      "label": "Panel 4 - Rover Drag",
      "text": "",
      "x": 500,
      "y": 333,
      "w": 500,
      "h": 334,
      "desc": "ROVER DRAG: medium wide action frame of the astronaut dragging a broken rover battery sled across an ice trench, bent posture, boots cutting into snow crust, torn suit panels and loose cables moving in the wind.",
      "color_palette": [
        "#557C8A",
        "#E6E1C6"
      ]
    },
    {
      "id": "panel_05",
      "kind": "obj",
      "label": "Panel 5 - Footprints To Antenna",
      "text": "",
      "x": 0,
      "y": 667,
      "w": 500,
      "h": 333,
      "desc": "FOOTPRINTS TO ANTENNA: long telephoto silhouette of the astronaut walking along a frozen ridge toward a collapsed research antenna, clear footprints through powder snow, giant ringed planet low over the horizon.",
      "color_palette": [
        "#A8D8E8",
        "#07080A"
      ]
    },
    {
      "id": "panel_06",
      "kind": "obj",
      "label": "Panel 6 - Visor Planet Reflection",
      "text": "",
      "x": 500,
      "y": 667,
      "w": 500,
      "h": 333,
      "desc": "VISOR PLANET REFLECTION: extreme close-up of a cracked astronaut visor, twin planets and broken towers reflected in the glass, frost on eyelashes, skin dust, sweat, sharp helmet texture, cinematic macro detail.",
      "color_palette": [
        "#07080A",
        "#A8D8E8"
      ]
    }
  ]
},
  auteur_closeup_storyboard_2x3: {
    label: "Auteur Close-Up Storyboard 2x3",
    summary: "Six clean photoreal auteur close-ups for testing cinematic continuity across a 2x3 storyboard.",
    workflow_mode: "storyboard_grid",
    grid_key: "story_2x3",
    target_resolution_key: "hd_720",
    preview: { kind: "storyboard_2x3", label: "2x3 CLOSE-UP" },
    canvas: {
      width: 2560,
      height: 2160,
      aspect_label: "Storyboard 2x3 / 1280x720 panels",
      target_width: 1280,
      target_height: 720,
      target_resolution_key: "hd_720",
    },
    scene: {
      high_level_description: "A clean six-panel 2x3 storyboard of restrained auteur cinema close-ups, extremely photoreal live-action frames, each panel an intact photographic still with crisp faces, readable hands, muted melancholy, and precise continuity.",
      aesthetics: "photoreal European art-film close-ups, clean camera-captured surfaces, lived-in faces, practical costumes, crisp local facial detail, restrained production design",
      lighting: "soft window light, dim practical lamps, gentle falloff, controlled contrast, readable eyes and hands, clean subject separation",
      photo: "35mm film stills, natural lens perspective, controlled fine grain, clean lens rendering, medium close-up and close-up framing",
      medium: "photograph",
      art_style: "",
      color_palette: ["#111820", "#6D5D49", "#A87A55", "#D6C7AA", "#374A55"],
      background: "Quiet interiors and empty streets stay inside each individual panel. The sheet has thin clean borders, separate photographic frames, crisp faces, readable hands, and no global overlay texture.",
    },
    items: [
      { kind: "obj", label: "Panel 1 - Window Breath", x: 0, y: 0, w: 500, h: 333, desc: "Panel 1: extreme clean photographic close-up of a tired actor near a small window, eyes turned toward pale morning light, quiet melancholy, crisp skin and eye detail.", color_palette: ["#D6C7AA", "#6D5D49"] },
      { kind: "obj", label: "Panel 2 - Hand At Collar", x: 500, y: 0, w: 500, h: 333, desc: "Panel 2: close-up profile as the same actor tightens a simple collar with one hand, candlelight grazing cheek and knuckles, clean practical interior background.", color_palette: ["#A87A55", "#111820"] },
      { kind: "obj", label: "Panel 3 - Corridor Pause", x: 0, y: 333, w: 500, h: 334, desc: "Panel 3: medium close shot in a narrow corridor, the actor pauses mid-step beside a wall mirror, shoulders tense, practical costume layers, sparse auteur composition.", color_palette: ["#374A55", "#D6C7AA"] },
      { kind: "obj", label: "Panel 4 - Brass Key", x: 500, y: 333, w: 500, h: 334, desc: "Panel 4: clean close-up of both hands holding a brass key over a dark wooden table, face soft in the background, crisp hand detail, cinematic silence.", color_palette: ["#A87A55", "#6D5D49"] },
      { kind: "obj", label: "Panel 5 - Courtyard Glance", x: 0, y: 667, w: 500, h: 333, desc: "Panel 5: tight over-shoulder close-up as the actor turns in a quiet courtyard, natural hair detail, simple coat, distant stone arch, restrained emotional performance.", color_palette: ["#374A55", "#111820"] },
      { kind: "obj", label: "Panel 6 - Stage Stillness", x: 500, y: 667, w: 500, h: 333, desc: "Panel 6: final clean close-up of the same actor seated on an empty theater stage, candle reflections, pale face, quiet unresolved expression, crisp photographic detail.", color_palette: ["#D6C7AA", "#A87A55"] },
    ],
  },
  poster: {
    label: "Poster",
    summary: "Advanced theatrical key art with hero, title, billing, and background hierarchy.",
    workflow_mode: "single_image",
    grid_key: "single_frame_1x1",
    target_resolution_key: "portrait_720x1280",
    preview: { kind: "poster", label: "POSTER" },
    canvas: { width: 1024, height: 1536, aspect_label: "2:3 Poster" },
    scene: {
      high_level_description: "A premium theatrical poster with a dominant live-action hero subject, clear title area, readable billing zone, and strong cinematic hierarchy.",
      aesthetics: "photoreal theatrical key art, premium cinema poster design, iconic hero silhouette, practical costume detail, polished art direction",
      lighting: "dramatic poster key light with controlled rim light, deep atmospheric background, and readable face and costume detail",
      photo: "studio-quality cinematic portrait photography, long-lens compression, refined film grain, sharp hero detail",
      medium: "photograph",
      color_palette: ["#101014", "#7A1F1F", "#D8B46A", "#EAE0CF", "#253E58"],
      background: "Layered cinematic background with atmosphere, scale, and negative space reserved for title and billing typography.",
    },
    items: [
      { kind: "obj", label: "Hero subject", x: 230, y: 150, w: 540, h: 650, desc: "dominant hero figure facing camera or three-quarter profile, iconic silhouette, expressive face, detailed costume, premium theatrical key art presence", color_palette: ["#EAE0CF", "#253E58"] },
      { kind: "text", label: "Title zone", text: "TITLE", x: 120, y: 760, w: 760, h: 120, desc: "large readable film title typography integrated into the poster, sharp edges, strong contrast, centered hierarchy", color_palette: ["#D8B46A"] },
      { kind: "obj", label: "Cinematic world", x: 80, y: 40, w: 840, h: 1080, desc: "atmospheric background world behind the hero, layered depth, symbolic shapes, cinematic scale, carefully separated from the title area", color_palette: ["#101014", "#253E58"] },
      { kind: "text", label: "Billing block", text: "BILLING", x: 140, y: 910, w: 720, h: 70, desc: "small readable billing block with clean alignment and high-end poster spacing", color_palette: ["#EAE0CF"] },
    ],
  },
  signage: {
    label: "Signage",
    summary: "Advanced readable in-world signage with surface, text, and environment boxes.",
    workflow_mode: "single_image",
    grid_key: "single_frame_1x1",
    target_resolution_key: "hd_720",
    preview: { kind: "signage", label: "SIGN" },
    canvas: { width: 1536, height: 864, aspect_label: "16:9 Signage" },
    scene: {
      high_level_description: "A cinematic environment built around readable in-world signage, practical surfaces, and believable production design.",
      aesthetics: "photoreal environmental graphics, readable typography, practical set dressing, premium prop design",
      lighting: "motivated practical lighting that keeps sign text legible while preserving cinematic contrast and atmosphere",
      photo: "35mm location photograph, realistic reflections, surface wear, shallow practical light falloff",
      medium: "photograph",
      color_palette: ["#0E1A24", "#19A7CE", "#F6F1D1", "#D65A31", "#2A2F35"],
      background: "Architectural setting with believable surfaces, depth, and supporting props that frame the sign as the hero object.",
    },
    items: [
      { kind: "text", label: "Hero sign text", text: "OPEN", x: 250, y: 270, w: 500, h: 170, desc: "large readable in-world sign text on a physical surface, crisp letters, correct perspective, visible material wear", color_palette: ["#F6F1D1", "#19A7CE"] },
      { kind: "obj", label: "Sign surface", x: 210, y: 220, w: 580, h: 250, desc: "physical sign board or glass surface with realistic grime, screws, reflections, and believable mounting hardware", color_palette: ["#2A2F35", "#D65A31"] },
      { kind: "obj", label: "Set context", x: 40, y: 90, w: 900, h: 760, desc: "surrounding cinematic environment with architectural context, practical lamps, weathered materials, and depth behind the sign", color_palette: ["#0E1A24"] },
    ],
  },
  screen_ui: {
    label: "Screen UI",
    summary: "Advanced diegetic monitor UI with multiple readable interface regions.",
    workflow_mode: "single_image",
    grid_key: "single_frame_1x1",
    target_resolution_key: "hd_720",
    preview: { kind: "screen_ui", label: "SCREEN" },
    canvas: { width: 1536, height: 864, aspect_label: "16:9 Screen UI" },
    scene: {
      high_level_description: "A believable diegetic monitor screen with readable interface panels, cinematic reflections, and practical hardware context.",
      aesthetics: "premium diegetic screen graphics, clean information hierarchy, realistic monitor glow, cinematic interface design",
      lighting: "monitor-emissive cyan and amber light with controlled reflections, visible glass surface, and readable panel contrast",
      photo: "macro photograph of a practical monitor prop, slight glass reflections, sharp UI detail, dark surrounding set",
      medium: "photograph",
      color_palette: ["#08121C", "#37D5D6", "#D9F3FF", "#F2A65A", "#0B2E3A"],
      background: "Physical monitor housing and dark set environment that support the screen graphics without hiding readable text zones.",
    },
    items: [
      { kind: "text", label: "Header readout", text: "SYSTEM READY", x: 90, y: 90, w: 820, h: 95, desc: "sharp readable interface header text, believable screen glow, clean spacing, cinematic technical design", color_palette: ["#D9F3FF"] },
      { kind: "obj", label: "Main data panel", x: 90, y: 210, w: 520, h: 520, desc: "large interface panel with charts, scan lines, map-like geometry, and crisp hierarchy inside a real monitor display", color_palette: ["#37D5D6", "#0B2E3A"] },
      { kind: "obj", label: "Status column", x: 650, y: 210, w: 260, h: 520, desc: "vertical column of compact readable status widgets, icons, numbers, and warning accents", color_palette: ["#F2A65A", "#D9F3FF"] },
    ],
  },
  title_card: {
    label: "Title Card",
    summary: "Advanced title frame with cinematic background, main title, and subtitle hierarchy.",
    workflow_mode: "single_image",
    grid_key: "single_frame_1x1",
    target_resolution_key: "hd_720",
    preview: { kind: "title_card", label: "TITLE" },
    canvas: { width: 1536, height: 864, aspect_label: "16:9 Title Card" },
    scene: {
      high_level_description: "A cinematic title card with elegant typography, atmospheric background, and carefully controlled negative space.",
      aesthetics: "premium opening title card, refined typography, cinematic restraint, strong graphic hierarchy",
      lighting: "low-key atmospheric light with subtle texture and contrast that preserves crisp title readability",
      photo: "film title design over photographed atmosphere, controlled grain, subtle optical diffusion",
      medium: "photograph",
      color_palette: ["#141414", "#8C1C13", "#F6E7CB", "#6C8EA3", "#0F2230"],
      background: "Minimal atmospheric background with texture, depth, and open space that elevates the title without visual clutter.",
    },
    items: [
      { kind: "text", label: "Main title", text: "TITLE", x: 160, y: 350, w: 680, h: 130, desc: "large elegant title typography with crisp readable letters, premium spacing, and cinematic restraint", color_palette: ["#F6E7CB"] },
      { kind: "text", label: "Subtitle", text: "A FILM BY", x: 250, y: 500, w: 500, h: 60, desc: "small subtitle line with clean tracking and careful alignment beneath the main title", color_palette: ["#6C8EA3"] },
      { kind: "obj", label: "Atmospheric field", x: 0, y: 0, w: 1000, h: 1000, desc: "subtle photographed atmosphere, texture, smoke, grain, and tonal depth behind the typography", color_palette: ["#141414", "#0F2230"] },
    ],
  },
};


const IMPORTED_PRESETS = {};

const PERSISTED_PRESET_KEYS = {
  gallery: "iamccs.frame_designer.imported_preset_gallery.v1",
  styles: "iamccs.frame_designer.imported_style_presets.v1",
  grids: "iamccs.frame_designer.imported_grid_presets.v1",
  hiddenGallery: "iamccs.frame_designer.hidden_preset_gallery.v1",
};

const MEMORY_PRESET_STORE = window.__IAMCCS_FRAME_DESIGNER_PRESETS__ ||= {};

function readPersistedObject(key) {
  try {
    const raw = window.localStorage?.getItem(key);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        MEMORY_PRESET_STORE[key] = parsed;
        return parsed;
      }
    }
  } catch (error) {
    console.warn("[IAMCCS FrameDesigner] preset read failed", error);
  }
  const fallback = MEMORY_PRESET_STORE[key];
  return fallback && typeof fallback === "object" && !Array.isArray(fallback) ? fallback : {};
}

function writePersistedObject(key, value) {
  const payload = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  MEMORY_PRESET_STORE[key] = payload;
  try {
    window.localStorage?.setItem(key, JSON.stringify(payload));
    return true;
  } catch (error) {
    console.warn("[IAMCCS FrameDesigner] preset persistence failed", error);
    return false;
  }
}

function assignImportedPresets(target, entries) {
  Object.entries(entries || {}).forEach(([key, value]) => {
    if (key && value && typeof value === "object") target[key] = value;
  });
}

function migrateImportedGalleryKeyCollisions() {
  Object.entries({ ...IMPORTED_PRESETS }).forEach(([key, preset], index) => {
    if (!Object.prototype.hasOwnProperty.call(PRESETS, key)) return;
    const nextKey = uniqueImportedGalleryKey("", preset, index);
    if (!nextKey || nextKey === key) return;
    IMPORTED_PRESETS[nextKey] = preset;
    delete IMPORTED_PRESETS[key];
    console.info("[IAMCCS FrameDesigner] migrated imported gallery key collision", { from: key, to: nextKey });
  });
}

const STYLE_PRESETS = {
  default_photoreal_cinema: {
    label: "Default Photorealistic Cinema",
    summary: "Grounded photoreal live-action cinema language for general Ideogram boards.",
    high: "A cinematic photoreal live-action frame with real actors, practical locations, believable costume detail, tactile materials, and clear dramatic composition.",
    background: "A production-aware cinematic environment with natural depth, realistic props, practical surfaces, readable blocking, and continuity details that support the action.",
    aesthetics: "photoreal live-action cinema, practical production design, real actors, natural skin texture, tactile costumes, grounded cinematic realism",
    lighting: "motivated cinematic light, soft key with controlled contrast, realistic bounce, practical highlights, clear subject separation",
    photo: "35mm anamorphic film still, natural lens perspective, controlled grain, shallow depth of field when useful, high-detail practical set photography",
    medium: "photograph",
    palette: ["#101014", "#2E3438", "#7A6A58", "#D6C7AA", "#8C3F2E"],
  },
};

const GRID_PRESETS = {
  free_canvas: { label: "Free Canvas", width: 1280, height: 720, boxes: null, order: "free", summary: "Empty canvas. Add object, text, or image layers manually." },
  single_frame_1x1: { label: "Single Frame", width: 1280, height: 720, boxes: [1, 1], order: "row_major", summary: "One full-frame image." },
  story_2x3: { label: "Storyboard 2x3", width: 2560, height: 2160, boxes: [2, 3], order: "row_major", summary: "Six storyboard panels; import other grids from JSON preset files." },
};


const TARGET_RESOLUTION_PRESETS = {
  hd_720: { label: "1280 x 720 - HD 16:9", width: 1280, height: 720, note: "Safe default for reference target panels." },
  fhd_1080: { label: "1920 x 1080 - Full HD 16:9", width: 1920, height: 1080, note: "Large target; heavier generation." },
  two_k_1152: { label: "2048 x 1152 - 2K 16:9", width: 2048, height: 1152, note: "2K wide target." },
  qhd_1440: { label: "2560 x 1440 - QHD 16:9", width: 2560, height: 1440, note: "High detail target; very heavy in reference modes." },
  three_k_1728: { label: "3072 x 1728 - 3K 16:9", width: 3072, height: 1728, note: "Extreme target; requires serious VRAM/time." },
  uhd_2160: { label: "3840 x 2160 - 4K 16:9", width: 3840, height: 2160, note: "Maximum experimental target; may fail on local VRAM." },
  square_1024: { label: "1024 x 1024 - Square", width: 1024, height: 1024, note: "Square image target." },
  square_2048: { label: "2048 x 2048 - 2K Square", width: 2048, height: 2048, note: "Large square target." },
  portrait_720x1280: { label: "720 x 1280 - Vertical 9:16", width: 720, height: 1280, note: "Vertical target." },
  portrait_1080x1920: { label: "1080 x 1920 - Full HD Vertical", width: 1080, height: 1920, note: "Large vertical target." },
  scope_1920x804: { label: "1920 x 804 - Scope 2.39:1", width: 1920, height: 804, note: "Cinemascope target." },
  scope_3072x1286: { label: "3072 x 1286 - 3K Scope", width: 3072, height: 1286, note: "Extreme cinemascope target." },
  custom: { label: "Custom - use Width / Height", width: 0, height: 0, note: "Manual canvas values stay editable." },
};

const PALETTE_PRESETS = {
  custom: {
    label: "Custom / Current",
    colors: [],
    summary: "Keep the palette currently typed in the field.",
  },
  photoreal_neutral_color: {
    label: "Photoreal Neutral Color",
    colors: ["#101014", "#2E3438", "#7A6A58", "#D6C7AA", "#8C3F2E"],
    summary: "Grounded cinematic color for realistic faces, costume, and natural interiors.",
  },
  silver_halide_bw: {
    label: "Silver Halide B/W",
    colors: ["#050505", "#1A1A1A", "#444444", "#8A8A8A", "#D8D8D8", "#F5F5F5"],
    summary: "True black-and-white documentary print values.",
  },
  faded_eastmancolor: {
    label: "Faded Eastmancolor",
    colors: ["#151718", "#47565B", "#8D7652", "#C7B383", "#B4563D", "#E4D6B5"],
    summary: "Vintage 1960s-1970s film color with muted cyan, amber, and aged skin tones.",
  },
  cold_scifi_ice: {
    label: "Cold Sci-Fi Ice",
    colors: ["#06111A", "#244454", "#557C8A", "#A8D8E8", "#E6E1C6", "#E87C45"],
    summary: "Frozen planet / space-survival palette with cold blues and small warm accents.",
  },
  candle_gothic: {
    label: "Candle Gothic",
    colors: ["#090706", "#241813", "#5E3B24", "#A26F3A", "#D9B978", "#F2E3C2"],
    summary: "Low-key candlelit interiors, aged wood, stone, and warm practical light.",
  },
  desert_epic: {
    label: "Desert Epic",
    colors: ["#100D0A", "#4B3828", "#8F7048", "#C6A46A", "#E5D0A1", "#48646B"],
    summary: "Dry sand, sun-baked fabric, dust, and restrained blue-green shadows.",
  },
  muted_fairytale_horror: {
    label: "Muted Fairytale Horror",
    colors: ["#0B0E10", "#273135", "#5B5245", "#91785D", "#BCA683", "#6E2328"],
    summary: "Dirty storybook realism, sickly fabrics, old rooms, and restrained blood-red accents.",
  },
  neon_night_city: {
    label: "Neon Night City",
    colors: ["#05070A", "#102035", "#00A6B8", "#9D4EDD", "#F72585", "#F5F0D8"],
    summary: "Wet night exteriors, signs, reflections, and graphic cyberpunk contrast.",
  },
  oz_sickly_green_gold: {
    label: "Sickly Green / Gold",
    colors: ["#08100A", "#27351E", "#5F6F35", "#A58D3D", "#D9C071", "#D8C8A8"],
    summary: "Grotesque fairytale golds, green shadows, antique costume texture.",
  },
};

const WORKFLOW_MODES = {
  single_image: {
    label: "Single Image",
    summary: "One full-canvas image. Canvas equals the selected target resolution.",
    ref_mode: "single",
    i2i: false,
    source_mode: "canvas_composite",
    columns: 1,
    rows: 1,
  },
  image_refine: {
    label: "Image Refine / i2i",
    summary: "One full-canvas image guide. Canvas equals the selected target resolution.",
    ref_mode: "single",
    i2i: true,
    source_mode: "refine_source_image",
    columns: 1,
    rows: 1,
  },
  storyboard_grid: {
    label: "Storyboard Grid",
    summary: "Uses the selected grid preset. Canvas = target panel size multiplied by grid columns and rows.",
    ref_mode: "single",
    i2i: false,
    source_mode: "canvas_composite",
    grid: true,
  },
  character_diptych: {
    label: "Character Ref Diptych",
    summary: "Two panels: left reference, right target. Canvas = target width x 2.",
    ref_mode: "character_diptych",
    i2i: true,
    source_mode: "reference_diptych",
    columns: 2,
    rows: 1,
  },
  multi_ref_triptych: {
    label: "Multi Ref Triptych",
    summary: "Three panels: two references, right target. Canvas = target width x 3.",
    ref_mode: "multi_ref_triptych",
    i2i: true,
    source_mode: "reference_triptych",
    columns: 3,
    rows: 1,
  },
};

const IMPORTED_STYLE_PRESETS = {};
const IMPORTED_GRID_PRESETS = {};
assignImportedPresets(IMPORTED_PRESETS, normalizePresetGallery(readPersistedObject(PERSISTED_PRESET_KEYS.gallery)));
assignImportedPresets(IMPORTED_STYLE_PRESETS, normalizeStyleGallery(readPersistedObject(PERSISTED_PRESET_KEYS.styles)));
assignImportedPresets(IMPORTED_GRID_PRESETS, normalizeGridGallery(readPersistedObject(PERSISTED_PRESET_KEYS.grids)));
migrateImportedGalleryKeyCollisions();
writePersistedObject(PERSISTED_PRESET_KEYS.gallery, IMPORTED_PRESETS);
writePersistedObject(PERSISTED_PRESET_KEYS.styles, IMPORTED_STYLE_PRESETS);
writePersistedObject(PERSISTED_PRESET_KEYS.grids, IMPORTED_GRID_PRESETS);

function styleEntries() {
  return { ...STYLE_PRESETS, ...IMPORTED_STYLE_PRESETS };
}

function gridEntries() {
  return { ...GRID_PRESETS, ...IMPORTED_GRID_PRESETS };
}

function hiddenGalleryPresetKeys() {
  return readPersistedObject(PERSISTED_PRESET_KEYS.hiddenGallery);
}

function writeHiddenGalleryPresetKeys(value) {
  return writePersistedObject(PERSISTED_PRESET_KEYS.hiddenGallery, value);
}


function targetResolutionEntries() {
  return { ...TARGET_RESOLUTION_PRESETS };
}

function paletteEntries() {
  return { ...PALETTE_PRESETS };
}

function paletteKeyForColors(colors) {
  const normalized = paletteList(colors, []).map((color) => color.toUpperCase()).join(",");
  if (!normalized) return "custom";
  for (const [key, preset] of Object.entries(PALETTE_PRESETS)) {
    const value = paletteList(preset.colors, []).map((color) => color.toUpperCase()).join(",");
    if (value && value === normalized) return key;
  }
  return "custom";
}

function applyPalettePresetToList(key, fallback = []) {
  const preset = paletteEntries()[key];
  if (!preset || key === "custom") return paletteList(fallback, []);
  return paletteList(preset.colors, fallback);
}

function workflowModeEntries() {
  return { ...WORKFLOW_MODES };
}

function workflowModeDropdownEntries() {
  const entries = { ...WORKFLOW_MODES };
  delete entries.character_diptych;
  delete entries.multi_ref_triptych;
  return entries;
}


function normalizePresetMap(entries, normalizer, fallbackPrefix = "preset") {
  const out = {};
  if (Array.isArray(entries)) {
    entries.forEach((entry, index) => {
      const preset = normalizer(entry, entry?.key || `${fallbackPrefix}_${index + 1}`);
      if (!preset) return;
      const key = cleanText(entry?.key) || importablePresetKey(preset.label || fallbackPrefix, index);
      out[key] = preset;
    });
  } else if (entries && typeof entries === "object") {
    Object.entries(entries).forEach(([key, entry], index) => {
      const preset = normalizer(entry, key);
      if (!preset) return;
      out[cleanText(entry?.key) || cleanText(key) || importablePresetKey(preset.label || fallbackPrefix, index)] = preset;
    });
  }
  return out;
}

function directStylePresetsFromJson(raw) {
  if (!raw || typeof raw !== "object") return {};
  const schema = cleanText(raw.schema);
  if (
    schema === "iamccs.frame_v2.style_presets" ||
    schema === "iamccs.frame_designer.style_presets" ||
    raw.styles || raw.style_presets || raw.stylePresets
  ) {
    return normalizePresetMap(raw.styles || raw.presets || raw.style_presets || raw.stylePresets, normalizeStylePreset, "style");
  }
  const single = normalizeStylePreset(raw, raw.key || raw.label || raw.name || "style_single");
  if (single) {
    const key = cleanText(raw.key) || importablePresetKey(single.label || raw.label || "style_single", 0);
    return { [key]: single };
  }
  return {};
}


function directGridPresetsFromJson(raw) {
  if (!raw || typeof raw !== "object") return {};
  const schema = cleanText(raw.schema);
  if (
    schema === "iamccs.frame_v2.grid_presets" ||
    schema === "iamccs.frame_designer.grid_presets" ||
    raw.grids || raw.grid_presets || raw.gridPresets
  ) {
    return normalizePresetMap(raw.grids || raw.presets || raw.grid_presets || raw.gridPresets, normalizeGridPreset, "grid");
  }
  const single = normalizeGridPreset(raw, raw.key || raw.label || raw.name || "grid_single");
  if (single) {
    const key = cleanText(raw.key) || importablePresetKey(single.label || raw.label || "grid_single", 0);
    return { [key]: single };
  }
  return {};
}


function directGalleryPresetsFromJson(raw) {
  if (!raw || typeof raw !== "object") return {};
  const schema = cleanText(raw.schema);
  if (
    schema === "iamccs.frame_v2.preset_gallery" ||
    schema === "iamccs.frame_designer.preset_gallery" ||
    raw.preset_gallery || raw.frame_presets || raw.gallery ||
    (raw.presets && Object.values(raw.presets).some((entry) => normalizePresetEntry(entry, "", 0)))
  ) {
    return normalizePresetMap(raw.presets || raw.gallery || raw.preset_gallery || raw.frame_presets, normalizePresetEntry, "gallery");
  }
  return {};
}


function normalizeStylePreset(entry, key = "") {
  if (!entry || typeof entry !== "object") return null;
  const scene = entry.scene && typeof entry.scene === "object" ? entry.scene : entry;
  const hasStyleSignal = Boolean(
    entry.key || entry.label || entry.name || entry.high || entry.background ||
    entry.aesthetics || entry.lighting || entry.photo || entry.medium ||
    scene.high || scene.high_level_description || scene.description ||
    scene.aesthetics || scene.style || scene.look || scene.lighting
  );
  if (!hasStyleSignal) return null;
  return {
    label: cleanText(entry.label || entry.name || key || "Style Preset"),
    summary: cleanText(entry.summary || entry.description || entry.hint) || "Imported Frame V2 style preset.",
    high: cleanText(entry.high || scene.high || scene.high_level_description || scene.description),
    background: cleanText(entry.background || scene.background || scene.environment),
    aesthetics: cleanText(entry.aesthetics || scene.aesthetics || scene.style || scene.look),
    lighting: cleanText(entry.lighting || scene.lighting),
    photo: cleanText(entry.photo || scene.photo || scene.lens || scene.camera),
    medium: cleanText(entry.medium || scene.medium) || "photograph",
    palette: paletteList(entry.palette || entry.color_palette || scene.palette || scene.color_palette, ["#1A1A2E", "#FFE4B5"]),
  };
}

function normalizeStyleGallery(raw) {
  let source = raw;
  const out = {};
  if (source && typeof source === "object" && source.schema === "iamccs.ideoboard.package" && source.boards) {
    Object.entries(source.boards).forEach(([key, board], index) => {
      const preset = normalizeStylePreset({
        key,
        label: board?.name || board?.title || board?.canvas?.aspect_label || key,
        summary: `Imported from ideoboard ${source.board_name || ""}`.trim(),
        ...(board?.scene || {}),
      }, key);
      if (preset) out[cleanText(key) || importablePresetKey(preset.label, index)] = preset;
    });
    return out;
  }
  if (source && typeof source === "object" && source.schema === "iamccs.ideogram_storyboard_frame_designer") {
    const preset = normalizeStylePreset({
      key: source.preset_key || "ideoboard_style",
      label: source.canvas?.aspect_label || source.preset_key || "Ideoboard Style",
      summary: "Imported from FrameDesigner ideoboard.",
      ...(source.scene || {}),
    }, source.preset_key || "ideoboard_style");
    if (preset) out[presetKeyFromLabel(source.preset_key || preset.label, "style")] = preset;
    return out;
  }
  if (source && typeof source === "object" && source.styles) source = source.styles;
  else if (source && typeof source === "object" && source.style_presets) source = source.style_presets;
  else if (source && typeof source === "object" && source.schema === "iamccs.frame_v2.style_presets" && source.presets) source = source.presets;
  else if (source && typeof source === "object" && source.presets && Object.values(source.presets).some((entry) => normalizeStylePreset(entry))) source = source.presets;
  else if (source && typeof source === "object" && (source.prompting_guidelines || source.source_basis || source.usage || source.instructions)) source = null;
  if (Array.isArray(source)) {
    source.forEach((entry, index) => {
      const preset = normalizeStylePreset(entry, entry?.key || `style_${index + 1}`);
      if (preset) out[importablePresetKey(entry?.key || preset.label, index)] = preset;
    });
  } else if (source && typeof source === "object") {
    Object.entries(source).forEach(([key, entry], index) => {
      const preset = normalizeStylePreset(entry, key);
      if (preset) out[cleanText(entry?.key) || cleanText(key) || importablePresetKey(preset.label, index)] = preset;
    });
  }
  return out;
}

function normalizeGridPreset(entry, key = "") {
  if (!entry || typeof entry !== "object") return null;
  let boxes = Array.isArray(entry.boxes) ? entry.boxes : (Number(entry.columns) && Number(entry.rows) ? [Number(entry.columns), Number(entry.rows)] : null);
  if (!boxes && Array.isArray(entry.items) && entry.items.length) {
    const xs = new Set();
    const ys = new Set();
    entry.items.forEach((item) => {
      xs.add(Math.round(Number(item?.x || 0)));
      ys.add(Math.round(Number(item?.y || 0)));
    });
    boxes = [Math.max(1, xs.size || 1), Math.max(1, ys.size || 1)];
  }
  const hasGridSignal = Boolean(boxes || entry.width || entry.height || entry.canvas?.width || entry.canvas?.height || Array.isArray(entry.items));
  if (!hasGridSignal) return null;
  return {
    label: cleanText(entry.label || entry.name || key || "Grid Preset"),
    summary: cleanText(entry.summary || entry.description || entry.hint) || "Imported Frame V2 grid preset.",
    width: clampInt(entry.width || entry.canvas?.width, 256, 8192, 1536),
    height: clampInt(entry.height || entry.canvas?.height, 256, 8192, 864),
    aspect_label: cleanText(entry.aspect_label || entry.canvas?.aspect_label || entry.aspect || entry.label || key),
    boxes: boxes && boxes.length >= 2 ? [Math.max(1, Number(boxes[0]) || 1), Math.max(1, Number(boxes[1]) || 1)] : null,
    order: cleanText(entry.order) || "row_major",
    items: Array.isArray(entry.items) ? entry.items.map((item, index) => normalizeItem(item, index)) : null,
  };
}

function normalizeGridGallery(raw) {
  let source = raw;
  const out = {};
  if (source && typeof source === "object" && source.schema === "iamccs.ideoboard.package" && source.boards) {
    Object.entries(source.boards).forEach(([key, board], index) => {
      const preset = normalizeGridPreset({
        key,
        label: board?.canvas?.aspect_label || board?.name || board?.title || key,
        summary: `Imported from ideoboard ${source.board_name || ""}`.trim(),
        width: board?.canvas?.width,
        height: board?.canvas?.height,
        aspect_label: board?.canvas?.aspect_label,
        items: board?.items || [],
      }, key);
      if (preset) out[cleanText(key) || importablePresetKey(preset.label, index)] = preset;
    });
    return out;
  }
  if (source && typeof source === "object" && source.schema === "iamccs.ideogram_storyboard_frame_designer") {
    const preset = normalizeGridPreset({
      key: source.preset_key || "ideoboard_grid",
      label: source.canvas?.aspect_label || source.preset_key || "Ideoboard Grid",
      summary: "Imported from FrameDesigner ideoboard.",
      width: source.canvas?.width,
      height: source.canvas?.height,
      aspect_label: source.canvas?.aspect_label,
      items: source.items || [],
    }, source.preset_key || "ideoboard_grid");
    if (preset) out[presetKeyFromLabel(source.preset_key || preset.label, "grid")] = preset;
    return out;
  }
  if (source && typeof source === "object" && source.grids) source = source.grids;
  else if (source && typeof source === "object" && source.grid_presets) source = source.grid_presets;
  else if (source && typeof source === "object" && source.schema === "iamccs.frame_v2.grid_presets" && source.presets) source = source.presets;
  else if (source && typeof source === "object" && source.presets && Object.values(source.presets).some((entry) => normalizeGridPreset(entry))) source = source.presets;
  else if (source && typeof source === "object" && (source.prompting_guidelines || source.source_basis || source.usage || source.instructions)) source = null;
  if (Array.isArray(source)) {
    source.forEach((entry, index) => {
      const preset = normalizeGridPreset(entry, entry?.key || `grid_${index + 1}`);
      if (preset) out[importablePresetKey(entry?.key || preset.label, index)] = preset;
    });
  } else if (source && typeof source === "object") {
    Object.entries(source).forEach(([key, entry], index) => {
      const preset = normalizeGridPreset(entry, key);
      if (preset) out[cleanText(entry?.key) || cleanText(key) || importablePresetKey(preset.label, index)] = preset;
    });
  }
  return out;
}

function normalizeReferenceMode(value) {
  const raw = value && typeof value === "object" ? value : {};
  const mode = ["single", "character_diptych", "multi_ref_triptych"].includes(String(raw.mode || "").toLowerCase())
    ? String(raw.mode).toLowerCase()
    : "single";
  const defaults = {
    single: { label: "Single / Storyboard", panel_count: 1, target_index: 0, locked_indices: [] },
    character_diptych: { label: "Character Ref Diptych", panel_count: 2, target_index: 1, locked_indices: [0] },
    multi_ref_triptych: { label: "Multi Ref Triptych", panel_count: 3, target_index: 2, locked_indices: [0, 1] },
  }[mode];
  return {
    mode,
    label: cleanText(raw.label) || defaults.label,
    panel_count: clampInt(raw.panel_count, 1, 8, defaults.panel_count),
    target_index: clampInt(raw.target_index, 0, 7, defaults.target_index),
    locked_indices: Array.isArray(raw.locked_indices) ? raw.locked_indices.map((v) => clampInt(v, 0, 7, 0)) : defaults.locked_indices,
  };
}

function makeReferenceItems(mode, palette = []) {
  const p0 = palette[0] || "#2F6F9E";
  const p1 = palette[1] || "#C4A15D";
  const p2 = palette[2] || "#7C6B55";
  if (mode === "character_diptych") {
    return [
      normalizeItem({
        id: "char_ref_01",
        kind: "image",
        label: "LOCKED Character Reference",
        x: 0,
        y: 0,
        w: 500,
        h: 1000,
        desc: "Load the full-body character reference here. This left panel is locked by the reference mask.",
        color_palette: [p0],
        image_path: "",
        fit: "contain",
        opacity: 1,
      }, 0),
      normalizeItem({
        id: "char_target_01",
        kind: "obj",
        label: "GENERATE Target Character",
        x: 500,
        y: 0,
        w: 500,
        h: 1000,
        desc: "the exact same person: repeat the character description verbatim, same face and hair, then describe the new costume, pose, action, and environment.",
        color_palette: [p1],
      }, 1),
    ];
  }
  if (mode === "multi_ref_triptych") {
    return [
      normalizeItem({
        id: "multi_ref_a",
        kind: "image",
        label: "LOCKED Reference A",
        x: 0,
        y: 0,
        w: 333,
        h: 1000,
        desc: "Load PERSON A reference here. This left panel is locked.",
        color_palette: [p0],
        image_path: "",
        fit: "contain",
        opacity: 1,
      }, 0),
      normalizeItem({
        id: "multi_ref_b",
        kind: "image",
        label: "LOCKED Reference B",
        x: 333,
        y: 0,
        w: 334,
        h: 1000,
        desc: "Load PERSON B or location/style reference here. This middle panel is locked.",
        color_palette: [p1],
        image_path: "",
        fit: "contain",
        opacity: 1,
      }, 1),
      normalizeItem({
        id: "multi_target",
        kind: "obj",
        label: "GENERATE Combined Target",
        x: 667,
        y: 0,
        w: 333,
        h: 1000,
        desc: "the exact same referenced subject(s), preserving identity cues, now together in the new target scene, pose, costume, lighting, and environment.",
        color_palette: [p2],
      }, 2),
    ];
  }
  return [];
}

function makeGridItems(columns, rows, palette = []) {
  const cols = Math.max(1, Number(columns) || 1);
  const rws = Math.max(1, Number(rows) || 1);
  const items = [];
  let index = 1;
  for (let r = 0; r < rws; r += 1) {
    for (let c = 0; c < cols; c += 1) {
      const x = Math.round((c / cols) * 1000);
      const y = Math.round((r / rws) * 1000);
      const x2 = Math.round(((c + 1) / cols) * 1000);
      const y2 = Math.round(((r + 1) / rws) * 1000);
      const color = palette[(index - 1) % Math.max(1, palette.length)] || (index % 2 ? "#C4A15D" : "#2F6F9E");
      items.push(normalizeItem({
        id: `panel_${String(index).padStart(2, "0")}`,
        kind: "obj",
        label: `Panel ${index}`,
        x,
        y,
        w: Math.max(20, x2 - x),
        h: Math.max(20, y2 - y),
        desc: `Panel ${index}: write the shot scale, subject action, environment, camera and lens language, and continuity cue.`,
        color_palette: [color],
      }, index - 1));
      index += 1;
    }
  }
  return items;
}

function workflowDefaultScene(modeKey) {
  const base = defaultData().scene;
  const defaults = {
    single_image: {
      high_level_description: "A single hyperreal cinematic image with one clear subject, readable staging, and a strong visual concept.",
      aesthetics: "photoreal live-action cinema, real materials, tactile production design, natural subject detail, cinematic realism",
      lighting: "motivated cinematic lighting with clear subject separation, controlled contrast, atmospheric depth",
      photo: "35mm anamorphic film still, practical set photography, realistic lens character, subtle film grain",
      medium: "photograph",
      color_palette: ["#101418", "#2F4550", "#8C6A45", "#D8C7A0"],
      background: "A coherent environment that supports the subject and camera angle.",
    },
    image_refine: {
      high_level_description: "Use the source image as the composition guide and create a cleaner, more detailed cinematic version of the same shot.",
      aesthetics: "photoreal live-action image restoration, sharper material detail, preserved composition, improved cinematic finish",
      lighting: "retain the existing lighting direction while improving separation, surface detail, and atmosphere",
      photo: "cleaned 35mm film still, realistic texture, refined facial and costume detail, controlled grain",
      medium: "photograph",
      color_palette: ["#101418", "#2F4550", "#8C6A45", "#D8C7A0"],
      background: "The same scene layout with improved clarity and tactile detail.",
    },
    storyboard_grid: {
      high_level_description: "A coherent cinematic storyboard contact sheet with multiple readable panels, each panel showing a different shot beat from the same film sequence.",
      aesthetics: "photoreal cinematic storyboard, consistent film language, live-action framing, clear panel-by-panel staging",
      lighting: "consistent motivated cinematic lighting across panels, readable silhouettes, atmospheric continuity",
      photo: "35mm anamorphic film stills arranged as a clean storyboard sheet, practical production design, film grain",
      medium: "photograph",
      color_palette: ["#123A5A", "#2F6F9E", "#C4A15D", "#7C6B55", "#D8C7A0"],
      background: "A continuous cinematic world with clear continuity cues across the panels.",
    },
    character_diptych: {
      high_level_description: "A two-panel character reference edit: left panel contains the source character reference, right panel generates the same character identity in a new pose and scene.",
      aesthetics: "photoreal character continuity, same facial structure and costume identity cues, cinematic live-action realism",
      lighting: "matched cinematic lighting relationship between reference and target panel, clean facial readability",
      photo: "photoreal film still, practical costume detail, realistic skin and prosthetic texture, consistent lens perspective",
      medium: "photograph",
      color_palette: ["#101418", "#2F4550", "#8C6A45", "#D8C7A0"],
      background: "Left panel is the source reference. Right panel describes the new pose, action, costume variation, and environment.",
    },
    multi_ref_triptych: {
      high_level_description: "A three-panel multi-reference edit: left and middle panels contain source references, right panel generates the target image using their identity, style, or location cues.",
      aesthetics: "photoreal multi-reference continuity, consistent identity cues, cinematic composition, practical material detail",
      lighting: "coherent cinematic lighting across reference and target panels, clear target subject separation",
      photo: "photoreal film still, realistic texture, practical costume and set detail, consistent lens perspective",
      medium: "photograph",
      color_palette: ["#101418", "#2F4550", "#8C6A45", "#D8C7A0"],
      background: "Left and middle panels are references. Right panel describes the new combined scene.",
    },
  };
  return { ...base, ...(defaults[modeKey] || defaults.storyboard_grid) };
}

function workflowDefaultDirectPrompt(modeKey, scene = {}) {
  if (modeKey === "single_image") {
    return "A single hyperreal cinematic film still of a clear subject in a coherent environment, natural camera perspective, practical production design, realistic texture, atmospheric lighting, sharp visual focus.";
  }
  if (modeKey === "image_refine") {
    return "Use the source image as the composition guide and create a cleaner, sharper, more cinematic version of the same shot, preserving subject placement, camera angle, pose, costume identity cues, and environment layout while improving fine detail and material clarity.";
  }
  return directPromptFromScene(scene, []);
}


function presetEntries() {
  const hidden = hiddenGalleryPresetKeys();
  const entries = { ...PRESETS, ...IMPORTED_PRESETS };
  // Keep the official storyboard preset deterministic; imported premium packs
  // can add presets, but they should not silently replace the core storyboard.
  entries.storyboard = PRESETS.storyboard;
  Object.keys(hidden || {}).forEach((key) => {
    if (hidden[key]) delete entries[key];
  });
  return entries;
}

function hasPresetKey(key) {
  return Object.prototype.hasOwnProperty.call(presetEntries(), String(key || ""));
}

function presetByKey(key) {
  return presetEntries()[String(key || "")] || PRESETS.storyboard;
}

function uniqueImportedGalleryKey(preferredKey, preset, index = 0) {
  const preferred = cleanText(preferredKey);
  if (preferred && !Object.prototype.hasOwnProperty.call(PRESETS, preferred)) return preferred;
  const label = cleanText(preset?.label || preset?.name || preferred || `Imported Preset ${index + 1}`);
  return importablePresetKey(label || `imported_preset_${index + 1}`, index);
}

function importablePresetKey(label, index = 0) {
  const base = String(label || `preset_${index + 1}`)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "")
    .slice(0, 64) || `preset_${index + 1}`;
  let key = base;
  let suffix = 2;
  while (hasPresetKey(key)) {
    key = `${base}_${suffix}`;
    suffix += 1;
  }
  return key;
}

function parseJsonLoose(text) {
  const raw = String(text || "").trim();
  if (!raw) throw new Error("empty JSON");
  const candidates = [raw];
  const fenced = raw.match(/```(?:json)?\s*([\s\S]*?)```/i);
  if (fenced?.[1]) candidates.push(fenced[1].trim());
  const first = raw.indexOf("{");
  const last = raw.lastIndexOf("}");
  if (first >= 0 && last > first) candidates.push(raw.slice(first, last + 1));
  const arrFirst = raw.indexOf("[");
  const arrLast = raw.lastIndexOf("]");
  if (arrFirst >= 0 && arrLast > arrFirst) candidates.push(raw.slice(arrFirst, arrLast + 1));
  for (const candidate of candidates) {
    try { return JSON.parse(candidate); } catch {}
  }
  throw new Error("invalid JSON");
}

function normalizePresetEntry(entry, key, index = 0) {
  if (!entry || typeof entry !== "object") return null;
  const scene = entry.scene && typeof entry.scene === "object" ? entry.scene : entry;
  const canvas = entry.canvas && typeof entry.canvas === "object" ? entry.canvas : {};
  const hasFrameSignal = Boolean(
    entry.canvas || entry.scene || Array.isArray(entry.items) ||
    entry.high_level_description || entry.high || entry.description ||
    entry.aesthetics || entry.lighting || entry.background ||
    entry.width || entry.height || canvas.width || canvas.height
  );
  if (!hasFrameSignal) return null;
  const presetKey = cleanText(entry.key || key) || importablePresetKey(entry.label || entry.name, index);
  const fallback = PRESETS.storyboard;
  const workflowMode = cleanText(entry.workflow_mode || entry.workflowMode || entry.mode);
  const gridKey = cleanText(entry.grid_key || entry.gridKey || entry.grid_preset || entry.gridPreset || canvas.grid_key || canvas.gridKey);
  const targetResolutionKey = cleanText(entry.target_resolution_key || entry.targetResolutionKey || canvas.target_resolution_key || canvas.targetResolutionKey);
  return {
    label: cleanText(entry.label || entry.name || presetKey.replace(/_/g, " ")) || fallback.label,
    summary: cleanText(entry.summary || entry.description || entry.hint) || "Imported IAMCCS frame preset.",
    workflow_mode: workflowMode,
    grid_key: gridKey,
    target_resolution_key: targetResolutionKey,
    preview: entry.preview && typeof entry.preview === "object" ? cloneValue(entry.preview, {}) : undefined,
    canvas: {
      width: clampInt(canvas.width ?? entry.width, 256, 8192, fallback.canvas.width),
      height: clampInt(canvas.height ?? entry.height, 256, 8192, fallback.canvas.height),
      aspect_label: cleanText(canvas.aspect_label || entry.aspect_label || entry.aspect || fallback.canvas.aspect_label),
      target_resolution_key: targetResolutionKey || cleanText(canvas.target_resolution_key || canvas.targetResolutionKey),
      target_width: clampInt(canvas.target_width ?? canvas.targetWidth ?? entry.target_width, 0, 8192, canvas.width || fallback.canvas.target_width || fallback.canvas.width),
      target_height: clampInt(canvas.target_height ?? canvas.targetHeight ?? entry.target_height, 0, 8192, canvas.height || fallback.canvas.target_height || fallback.canvas.height),
    },
    scene: {
      high_level_description: cleanText(scene.high_level_description || scene.high || scene.description) || fallback.scene.high_level_description,
      aesthetics: cleanText(scene.aesthetics || scene.style || scene.look) || fallback.scene.aesthetics,
      lighting: cleanText(scene.lighting) || fallback.scene.lighting,
      photo: cleanText(scene.photo || scene.lens || scene.camera),
      medium: cleanText(scene.medium) || fallback.scene.medium,
      color_palette: paletteList(scene.color_palette || entry.color_palette || scene.palette, fallback.scene.color_palette),
      background: cleanText(scene.background || scene.environment) || fallback.scene.background,
    },
    i2i: entry.i2i && typeof entry.i2i === "object" ? normalizeI2I(entry.i2i) : undefined,
    items: Array.isArray(entry.items) ? entry.items.map(normalizeItem) : undefined,
  };
}

function normalizePresetGallery(raw) {
  let source = raw;
  const out = {};
  if (source && typeof source === "object" && source.schema === "iamccs.ideogram_storyboard_frame_designer") {
    const normalized = normalizePresetEntry(source, source.key || source.preset_key || source.label || source.name, 0);
    if (normalized) out[cleanText(source.key || source.preset_key) || importablePresetKey(normalized.label, 0)] = normalized;
    return out;
  }
  if (source && typeof source === "object" && source.boards && typeof source.boards === "object") source = source.boards;
  else if (source && typeof source === "object" && source.preset_gallery) source = source.preset_gallery;
  else if (source && typeof source === "object" && source.gallery) source = source.gallery;
  else if (source && typeof source === "object" && source.frame_presets) source = source.frame_presets;
  else if (source && typeof source === "object" && source.presets && Object.values(source.presets).some((entry) => normalizePresetEntry(entry, "", 0))) source = source.presets;
  else if (source && typeof source === "object" && (source.prompting_guidelines || source.source_basis || source.usage || source.instructions)) source = null;
  if (Array.isArray(source)) {
    source.forEach((entry, index) => {
      const normalized = normalizePresetEntry(entry, entry?.key, index);
      if (normalized) out[cleanText(entry?.key) || importablePresetKey(normalized.label, index)] = normalized;
    });
  } else if (source && typeof source === "object" && (source.canvas || source.scene || source.items || source.high_level_description)) {
    const normalized = normalizePresetEntry(source, source.key || source.preset_key || source.label || source.name, 0);
    if (normalized) out[cleanText(source.key || source.preset_key) || importablePresetKey(normalized.label, 0)] = normalized;
  } else if (source && typeof source === "object") {
    Object.entries(source).forEach(([key, entry], index) => {
      const normalized = normalizePresetEntry(entry, key, index);
      if (normalized) out[cleanText(entry?.key || entry?.preset_key) || cleanText(key) || importablePresetKey(normalized.label, index)] = normalized;
    });
  }
  return out;
}


function widget(node, name) {
  return (node.widgets || []).find((entry) => entry?.name === name);
}

function setWidget(node, name, value) {
  const entry = widget(node, name);
  if (!entry) return;
  entry.value = value;
  try { entry.callback?.(value); } catch {}
  node.setDirtyCanvas?.(true, true);
  app.graph?.setDirtyCanvas?.(true, true);
}

function hideWidget(entry) {
  if (!entry) return;
  entry.hidden = true;
  entry.type = "hidden";
  entry.computeSize = () => [0, -4];
  entry.draw = () => {};
  entry.options = { ...(entry.options || {}), hidden: true };
}

function cloneValue(value, fallback = null) {
  try {
    return JSON.parse(JSON.stringify(value));
  } catch {
    return fallback;
  }
}

function defaultData() {
  return {
  "schema": "iamccs.ideogram_storyboard_frame_designer",
  "schema_version": 1,
  "preset_key": "",
  "workflow_mode": "single_image",
  "grid_key": "free_canvas",
  "__iamccs_user_applied_preset": false,
  "target_resolution_key": "hd_720",
  "json_export_mode": "json_perfect",
  "brief_to_json": {
    "brief": "",
    "instruction": "Enhance the current Ideogram JSON without changing layout, bbox coordinates, visible text, or panel count."
  },
  "gemma_assistant": {
    "enabled": false,
    "provider": "local_gemma",
    "mode": "full_json_enhance",
    "speed": "fast",
    "model": "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors",
    "selected_id": "",
    "target_field": "",
    "current_text": "",
    "brief": "",
    "request_ready": false
  },
  "mask_paint": {
    "brush_size": 48,
    "strokes": []
  },
  "canvas": {
    "width": 1280,
    "height": 720,
    "aspect_label": "Empty single canvas / 1280 x 720 - HD 16:9",
    "target_resolution_key": "hd_720",
    "target_width": 1280,
    "target_height": 720
  },
  "scene": {
    "high_level_description": "",
    "aesthetics": "",
    "lighting": "",
    "photo": "",
    "medium": "photograph",
    "art_style": "",
    "color_palette": [
      "#101014",
      "#2E3438",
      "#7A6A58",
      "#D6C7AA",
      "#8C3F2E"
    ],
    "background": ""
  },
  "i2i": {
    "enabled": false,
    "denoise": 0.28,
    "low_sigma_start_step": 12,
    "scheduler_hint": "",
    "source_mode": "canvas_composite"
  },
  "reference_mode": {
    "mode": "single",
    "label": "Single",
    "panel_count": 1,
    "target_index": 0,
    "locked_indices": []
  },
  "direct_prompt": {
    "enabled": false,
    "text": ""
  },
  "json_override": {
    "enabled": false,
    "text": ""
  },
  "items": []
};
}



function presetData(key) {
  const preset = presetByKey(key);
  const base = defaultData();
  return normalizeDesignObject({
    ...base,
    preset_key: hasPresetKey(key) ? key : "storyboard",
    workflow_mode: preset.workflow_mode || base.workflow_mode,
    grid_key: preset.grid_key || base.grid_key,
    target_resolution_key: preset.target_resolution_key || preset.canvas?.target_resolution_key || base.target_resolution_key,
    preview: preset.preview,
    canvas: { ...base.canvas, ...(preset.canvas || {}) },
    scene: { ...base.scene, ...(preset.scene || {}), color_palette: [...(preset.scene?.color_palette || base.scene.color_palette)] },
    i2i: normalizeI2I(preset.i2i || base.i2i),
    items: Array.isArray(preset.items) && preset.items.length ? preset.items : base.items,
  }, base);
}

function cleanText(value) {
  return String(value || "").trim();
}

function clampInt(value, min, max, fallback) {
  const number = Number.isFinite(Number(value)) ? Math.round(Number(value)) : fallback;
  return Math.max(min, Math.min(max, number));
}

function normalizeHex(value) {
  let text = String(value || "").trim().toUpperCase();
  if (!text) return "";
  if (!text.startsWith("#")) text = `#${text}`;
  if (text.length === 4) text = `#${text[1]}${text[1]}${text[2]}${text[2]}${text[3]}${text[3]}`;
  if (!/^#[0-9A-F]{6}$/.test(text)) return "";
  return text;
}

function paletteList(value, fallback = []) {
  const source = Array.isArray(value) ? value : String(value || "").replace(/;/g, ",").split(",");
  const out = [];
  source.forEach((entry) => {
    const color = normalizeHex(entry);
    if (color && !out.includes(color)) out.push(color);
  });
  return out.length ? out : [...fallback];
}

function itemFromPrompt(entry, index) {
  const bbox = Array.isArray(entry?.bbox) && entry.bbox.length === 4 ? entry.bbox : [120, 120, 760, 760];
  const y = clampInt(bbox[0], 0, 999, 120);
  const x = clampInt(bbox[1], 0, 999, 120);
  const y2 = clampInt(bbox[2], y + 1, 1000, 760);
  const x2 = clampInt(bbox[3], x + 1, 1000, 760);
  const kind = String(entry?.type || "obj").toLowerCase() === "text" ? "text" : "obj";
  return {
    id: `item_${String(index + 1).padStart(3, "0")}`,
    kind,
    label: cleanText(entry?.label || entry?.text || entry?.type || `Element ${index + 1}`),
    text: kind === "text" ? cleanText(entry?.text) : "",
    x,
    y,
    w: Math.max(20, x2 - x),
    h: Math.max(20, y2 - y),
    desc: cleanText(entry?.desc || entry?.description || `Element ${index + 1}`),
    color_palette: paletteList(entry?.color_palette, ["#FFE4B5", "#1A1A2E"]),
  };
}

function promptToDesign(raw) {
  const base = defaultData();
  const style = raw?.style_description || {};
  const rawComp = raw?.compositional_deconstruction;
  const comp = rawComp && !Array.isArray(rawComp) && typeof rawComp === "object" ? rawComp : {};
  const elements = Array.isArray(rawComp) ? rawComp : (Array.isArray(comp?.elements) ? comp.elements : []);
  return {
    ...base,
    scene: {
      high_level_description: cleanText(raw?.high_level_description) || base.scene.high_level_description,
      aesthetics: cleanText(style?.aesthetics) || base.scene.aesthetics,
      lighting: cleanText(style?.lighting) || base.scene.lighting,
      photo: cleanText(style?.photo),
      medium: cleanText(style?.medium) || base.scene.medium,
      art_style: cleanText(style?.art_style),
      color_palette: paletteList(style?.color_palette, base.scene.color_palette),
      background: cleanText(comp?.background) || base.scene.background,
    },
    items: elements.length ? elements.filter((item) => item && typeof item === "object").map(itemFromPrompt) : cloneValue(base.items, []),
  };
}

function normalizeItem(entry, index) {
  const rawKind = String(entry?.kind || entry?.type || "obj").toLowerCase();
  const kind = rawKind === "text" || rawKind === "txt"
    ? "text"
    : (["image", "img", "reference", "source"].includes(rawKind)
      ? "image"
      : (["mask", "paint_mask", "inpaint_mask"].includes(rawKind) ? "mask" : "obj"));
  let x = clampInt(entry?.x, 0, 980, 120 + index * 40);
  let y = clampInt(entry?.y, 0, 980, 120 + index * 30);
  let bw = entry?.w;
  let bh = entry?.h;
  if (Array.isArray(entry?.bbox) && entry.bbox.length === 4) {
    const ymin = clampInt(entry.bbox[0], 0, 999, y);
    const xmin = clampInt(entry.bbox[1], 0, 999, x);
    const ymax = clampInt(entry.bbox[2], ymin + 1, 1000, ymin + 180);
    const xmax = clampInt(entry.bbox[3], xmin + 1, 1000, xmin + 260);
    x = xmin;
    y = ymin;
    bw = xmax - xmin;
    bh = ymax - ymin;
  }
  const item = {
    id: cleanText(entry?.id) || `item_${String(index + 1).padStart(3, "0")}`,
    kind,
    label: cleanText(entry?.label || entry?.name || `Element ${index + 1}`),
    text: kind === "text" ? cleanText(entry?.text) : "",
    x,
    y,
    w: Math.min(1000 - x, clampInt(bw, 20, 1000, 260)),
    h: Math.min(1000 - y, clampInt(bh, 20, 1000, 180)),
    desc: cleanText(entry?.desc || entry?.description || `Element ${index + 1}`),
    color_palette: paletteList(entry?.color_palette, ["#FFE4B5", "#1A1A2E"]),
  };
  if (kind === "image") {
    item.image_path = cleanText(entry?.image_path || entry?.imagePath || entry?.imageFile || entry?.path || entry?.file);
    item.fit = ["cover", "contain", "stretch"].includes(String(entry?.fit || entry?.resize_mode || "cover").toLowerCase()) ? String(entry?.fit || entry?.resize_mode || "cover").toLowerCase() : "cover";
    item.opacity = Math.max(0, Math.min(1, Number(entry?.opacity ?? 1) || 1));
  } else if (kind === "mask") {
    item.shape = ["rect", "ellipse"].includes(String(entry?.shape || entry?.mask_shape || "rect").toLowerCase())
      ? String(entry?.shape || entry?.mask_shape || "rect").toLowerCase()
      : "rect";
    item.color_palette = paletteList(entry?.color_palette, ["#FFFFFF", "#FF4D6D"]);
  }
  return item;
}

function normalizeI2I(value) {
  const raw = value && typeof value === "object" ? value : {};
  return {
    enabled: Boolean(raw.enabled),
    denoise: Math.max(0, Math.min(1, Number(raw.denoise ?? 0.28) || 0.28)),
    low_sigma_start_step: clampInt(raw.low_sigma_start_step, 0, 1000, 12),
    scheduler_hint: cleanText(raw.scheduler_hint) || "Use SplitSigmasDenoise for denoise or split high/low sigmas by step for advanced i2i.",
    source_mode: cleanText(raw.source_mode) || "canvas_composite",
  };
}

function directPromptFromScene(scene = {}, items = []) {
  const parts = [
    scene.high_level_description,
    scene.aesthetics,
    scene.lighting,
    scene.photo,
    scene.medium,
    scene.background,
    ...(Array.isArray(items) ? items.map((item) => item?.desc || item?.text || item?.label) : []),
  ].map(cleanText).filter(Boolean);
  return parts.join("\n\n");
}

function normalizeDirectPrompt(value, scene = {}, items = []) {
  const raw = value && typeof value === "object" ? value : {};
  return {
    enabled: Boolean(raw.enabled),
    text: cleanText(raw.text) || directPromptFromScene(scene, items),
  };
}

function normalizeJsonOverride(value) {
  const raw = value && typeof value === "object" ? value : {};
  return {
    enabled: Boolean(raw.enabled),
    text: cleanText(raw.text),
  };
}

function normalizeMaskStrokeTarget(value) {
  const raw = value && typeof value === "object" ? value : {};
  const kind = cleanText(raw.kind || raw.type).toLowerCase();
  if (kind === "canvas_full") {
    return { kind: "canvas_full", x: 0, y: 0, w: 1000, h: 1000 };
  }
  if (kind !== "image_content") return null;
  const x = clampInt(raw.x, 0, 999, 0);
  const y = clampInt(raw.y, 0, 999, 0);
  const w = clampInt(raw.w, 1, Math.max(1, 1000 - x), Math.max(1, 1000 - x));
  const h = clampInt(raw.h, 1, Math.max(1, 1000 - y), Math.max(1, 1000 - y));
  return {
    kind: "image_content",
    item_id: cleanText(raw.item_id || raw.itemId || raw.id),
    x,
    y,
    w,
    h,
    fit: cleanText(raw.fit || "contain") || "contain",
  };
}

function normalizeMaskPaint(value, fallback = { brush_size: 48, strokes: [] }) {
  const raw = value && typeof value === "object" ? value : {};
  const brushSize = clampInt(raw.brush_size, 1, 240, fallback.brush_size || 48);
  const strokes = [];
  const source = Array.isArray(raw.strokes) ? raw.strokes : [];
  source.forEach((stroke) => {
    if (!stroke || typeof stroke !== "object") return;
    const points = [];
    (Array.isArray(stroke.points) ? stroke.points : []).forEach((point) => {
      if (!Array.isArray(point) || point.length < 2) return;
      points.push([
        clampInt(point[0], 0, 1000, 0),
        clampInt(point[1], 0, 1000, 0),
      ]);
    });
    if (!points.length) return;
    const mode = String(stroke.mode || "paint").toLowerCase() === "erase" ? "erase" : "paint";
    const shape = ["lasso", "rect", "stroke"].includes(String(stroke.shape || "stroke").toLowerCase())
      ? String(stroke.shape || "stroke").toLowerCase()
      : "stroke";
    const nextStroke = {
      mode,
      shape,
      size: clampInt(stroke.size, 1, 240, brushSize),
      points,
    };
    const target = normalizeMaskStrokeTarget(stroke.target);
    if (target) nextStroke.target = target;
    strokes.push(nextStroke);
  });
  return { brush_size: brushSize, strokes };
}

function isIdeogramPromptJson(value) {
  return Boolean(
    value && typeof value === "object" && !Array.isArray(value) &&
    value.compositional_deconstruction && typeof value.compositional_deconstruction === "object"
  );
}

function imageViewUrl(path) {
  const clean = cleanText(path);
  if (!clean) return "";
  if (/^[a-zA-Z]:[\\/]/.test(clean) || clean.startsWith("/") || clean.startsWith("\\\\")) {
    return `/api/iamccs/cine/view_image?path=${encodeURIComponent(clean)}`;
  }
  const normalized = clean.replace(/\\/g, "/");
  const parts = normalized.split("/").filter(Boolean);
  const filename = parts.pop() || normalized;
  const subfolder = parts.join("/");
  const params = new URLSearchParams({ filename, type: "input" });
  if (subfolder) params.set("subfolder", subfolder);
  return `/view?${params.toString()}`;
}

function normalizeDesignObject(raw, fallback = defaultData()) {
  if (!raw || typeof raw !== "object") return cloneValue(fallback, fallback);
  return {
    ...fallback,
    __iamccs_user_applied_preset: Boolean(raw.__iamccs_user_applied_preset || fallback.__iamccs_user_applied_preset),
    preset_key: hasPresetKey(raw.preset_key) ? raw.preset_key : fallback.preset_key,
    workflow_mode: workflowModeEntries()[raw.workflow_mode] ? raw.workflow_mode : (workflowModeEntries()[fallback.workflow_mode] ? fallback.workflow_mode : "single_image"),
    grid_key: cleanText(raw.grid_key || raw.gridKey || fallback.grid_key),
    target_resolution_key: cleanText(raw.target_resolution_key || raw.targetResolutionKey || raw.canvas?.target_resolution_key || fallback.target_resolution_key),
    preview: raw.preview && typeof raw.preview === "object" ? cloneValue(raw.preview, {}) : (fallback.preview && typeof fallback.preview === "object" ? cloneValue(fallback.preview, {}) : undefined),
    canvas: { ...fallback.canvas, ...(raw.canvas || {}) },
    scene: {
      ...fallback.scene,
      ...(raw.scene || {}),
      color_palette: paletteList(raw?.scene?.color_palette, fallback.scene.color_palette),
    },
    i2i: normalizeI2I(raw.i2i || fallback.i2i),
    reference_mode: normalizeReferenceMode(raw.reference_mode || fallback.reference_mode),
    json_override: normalizeJsonOverride(raw.json_override || fallback.json_override),
    direct_prompt: normalizeDirectPrompt(raw.direct_prompt || fallback.direct_prompt, { ...fallback.scene, ...(raw.scene || {}) }, Array.isArray(raw.items) ? raw.items : fallback.items),
    mask_paint: normalizeMaskPaint(raw.mask_paint || fallback.mask_paint, fallback.mask_paint),
    items: Array.isArray(raw.items) ? raw.items.map(normalizeItem) : cloneValue(fallback.items, fallback.items),
  };
}

function isLegacyDefaultStoryboardSeed(raw) {
  if (!raw || typeof raw !== "object" || raw.__iamccs_user_applied_preset) return false;
  const scene = raw.scene || {};
  const items = Array.isArray(raw.items) ? raw.items : [];
  const high = cleanText(scene.high_level_description).toLowerCase();
  const labels = items.map((item) => cleanText(item?.label).toLowerCase()).join(" | ");
  return (
    raw.workflow_mode === "storyboard_grid" &&
    raw.grid_key === "story_2x3" &&
    items.length === 6 &&
    high.includes("frozen exoplanet") &&
    labels.includes("ice plain arrival") &&
    labels.includes("visor planet reflection")
  );
}

function parseData(node) {
  const fallback = defaultData();
  try {
    const parsed = JSON.parse(String(widget(node, "design_data")?.value || ""));
    if (parsed && typeof parsed === "object") {
      if (isLegacyDefaultStoryboardSeed(parsed)) {
        console.info("[IAMCCS FrameDesigner] ignored legacy default storyboard seed", { build: JS_BUILD });
        return fallback;
      }
      if (parsed.boards && typeof parsed.boards === "object") {
        const activeKey = hasPresetKey(parsed.active_preset_key) ? parsed.active_preset_key : (hasPresetKey(parsed.preset_key) ? parsed.preset_key : fallback.preset_key);
        const activeBoard = parsed.boards[activeKey];
        if (activeBoard && typeof activeBoard === "object") {
          if (isLegacyDefaultStoryboardSeed(activeBoard)) {
            console.info("[IAMCCS FrameDesigner] ignored legacy default storyboard seed package", { build: JS_BUILD });
            return fallback;
          }
          return normalizeDesignObject({ ...activeBoard, preset_key: activeKey }, fallback);
        }
      }
      if (parsed.high_level_description && parsed.style_description && parsed.compositional_deconstruction) {
        return promptToDesign(parsed);
      }
      return normalizeDesignObject(parsed, fallback);
    }
  } catch {}
  return fallback;
}

function toPrompt(data) {
  const override = normalizeJsonOverride(data?.json_override);
  if (override.enabled) {
    try {
      const parsed = parseJsonLoose(override.text);
      if (isIdeogramPromptJson(parsed)) return parsed;
    } catch {}
  }
  const items = Array.isArray(data?.items) ? data.items.map(normalizeItem).filter((item) => item.kind !== "image" && item.kind !== "mask") : [];
  const monochrome = isMonochromeDesign(data);
  const scenePalette = paletteList(data?.scene?.color_palette, []);
  const fallbackMonoPalette = ["#050505", "#1A1A1A", "#777777", "#F2F2F2"];
  const elementForItem = (item) => {
    const palette = monochrome
      ? (scenePalette.length ? scenePalette : fallbackMonoPalette)
      : paletteList(item.color_palette, []);
    const entry = {
      type: item.kind === "text" ? "text" : "obj",
      bbox: [item.y, item.x, Math.min(1000, item.y + item.h), Math.min(1000, item.x + item.w)],
      desc: item.desc,
      ...(item.kind === "text" ? { text: item.text || item.label } : {}),
    };
    if (palette.length) entry.color_palette = palette;
    return entry;
  };
  return {
    high_level_description: cleanText(data?.scene?.high_level_description),
    style_description: {
      aesthetics: cleanText(data?.scene?.aesthetics),
      lighting: cleanText(data?.scene?.lighting),
      photo: cleanText(data?.scene?.photo),
      medium: cleanText(data?.scene?.medium),
      art_style: cleanText(data?.scene?.art_style),
      color_palette: paletteList(data?.scene?.color_palette, ["#1A1A2E", "#FFE4B5"]),
    },
    compositional_deconstruction: {
      background: cleanText(data?.scene?.background),
      elements: items.map(elementForItem),
    },
  };
}

function isMonochromeDesign(data) {
  const scene = data?.scene || {};
  const text = [
    scene.high_level_description,
    scene.aesthetics,
    scene.lighting,
    scene.photo,
    scene.medium,
    scene.art_style,
    scene.background,
  ].map(cleanText).join(" ").toLowerCase();
  return /\b(black and white|black-and-white|monochrome|grayscale|grey scale|silver halide|silver gelatin)\b/.test(text);
}

function safeFilename(value, fallback = "IAMCCS_Ideoboard") {
  const cleaned = cleanText(value).replace(/[\\/:*?"<>|]+/g, "_").replace(/\s+/g, " ").trim();
  return cleaned || fallback;
}

function escapeXml(value) {
  return String(value || "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&apos;");
}

function boardPreviewSvgDataUri(data) {
  const palette = paletteList(data?.scene?.color_palette, ["#18202A", "#0D1118", "#F6E7CB"]);
  const items = Array.isArray(data?.items) ? data.items.map(normalizeItem) : [];
  const canvasW = Math.max(256, Number(data?.canvas?.width || 1536) || 1536);
  const canvasH = Math.max(256, Number(data?.canvas?.height || 864) || 864);
  const aspect = Math.max(0.32, Math.min(3.6, canvasW / canvasH));
  const frameW = aspect >= 1 ? 860 : Math.max(380, Math.round(680 * aspect));
  const frameH = aspect >= 1 ? Math.max(280, Math.round(860 / aspect)) : 680;
  const frameX = Math.round((1000 - frameW) / 2);
  const frameY = Math.round((1000 - frameH) / 2) + 24;
  const kind = cleanText(data?.preview?.kind).toLowerCase()
    || (items.length >= 6 ? "storyboard_2x3" : items.some((item) => item.kind === "text") ? "title_card" : "frame");
  const label = escapeXml(cleanText(data?.preview?.label || data?.label || data?.preset_key || "IDEOBOARD").toUpperCase());
  const p0 = palette[0] || "#18202A";
  const p1 = palette[1] || "#0D1118";
  const p2 = palette[2] || "#F6E7CB";
  const p3 = palette[3] || "#48C2BA";
  const cell = (c, r, cols = 2, rows = 3) => {
    const gap = 10;
    const x = frameX + gap + c * ((frameW - gap * 2) / cols);
    const y = frameY + gap + r * ((frameH - gap * 2) / rows);
    const w = ((frameW - gap * 2) / cols) - gap;
    const h = ((frameH - gap * 2) / rows) - gap;
    return { x, y, w, h };
  };
  const itemOverlay = items.slice(0, 8).map((item) => {
    const x = frameX + (item.x / 1000) * frameW;
    const y = frameY + (item.y / 1000) * frameH;
    const w = Math.max(12, (item.w / 1000) * frameW);
    const h = Math.max(12, (item.h / 1000) * frameH);
    const fill = item.kind === "text" ? (item.color_palette?.[0] || "#FFC078") : (item.color_palette?.[0] || p3);
    return `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${w.toFixed(1)}" height="${h.toFixed(1)}" rx="10" fill="${fill}" fill-opacity="0.16" stroke="${fill}" stroke-width="3"/>`;
  }).join("");
  let composition = "";
  if (kind.includes("storyboard")) {
    const closeup = label.includes("CLOSE");
    const cells = [];
    for (let r = 0; r < 3; r += 1) {
      for (let c = 0; c < 2; c += 1) {
        const b = cell(c, r, 2, 3);
        const cx = b.x + b.w * 0.5;
        const cy = b.y + b.h * 0.55;
        cells.push(`<g><rect x="${b.x}" y="${b.y}" width="${b.w}" height="${b.h}" rx="12" fill="${p0}" fill-opacity="0.58" stroke="${p2}" stroke-opacity="0.7" stroke-width="2"/>${closeup ? `<ellipse cx="${cx}" cy="${cy}" rx="${b.w * 0.18}" ry="${b.h * 0.28}" fill="${p2}" fill-opacity="0.55"/><circle cx="${cx - b.w * 0.06}" cy="${cy - b.h * 0.04}" r="5" fill="#101014"/><circle cx="${cx + b.w * 0.06}" cy="${cy - b.h * 0.04}" r="5" fill="#101014"/>` : `<path d="M${b.x + 12} ${b.y + b.h - 26} C${b.x + b.w * 0.35} ${b.y + b.h * 0.55}, ${b.x + b.w * 0.7} ${b.y + b.h * 0.65}, ${b.x + b.w - 12} ${b.y + 22}" fill="none" stroke="${p3}" stroke-width="4" stroke-opacity="0.75"/><circle cx="${cx}" cy="${cy}" r="${Math.min(b.w, b.h) * 0.12}" fill="${p2}" fill-opacity="0.55"/>`}</g>`);
      }
    }
    composition = cells.join("");
  } else if (kind === "poster") {
    composition = `<rect x="${frameX + frameW * 0.1}" y="${frameY + frameH * 0.08}" width="${frameW * 0.8}" height="${frameH * 0.84}" rx="18" fill="${p0}" fill-opacity="0.68" stroke="${p2}" stroke-width="4"/><circle cx="${frameX + frameW * 0.5}" cy="${frameY + frameH * 0.35}" r="${Math.min(frameW, frameH) * 0.18}" fill="${p2}" fill-opacity="0.55"/><rect x="${frameX + frameW * 0.22}" y="${frameY + frameH * 0.68}" width="${frameW * 0.56}" height="${frameH * 0.08}" rx="10" fill="${p2}" fill-opacity="0.85"/><rect x="${frameX + frameW * 0.3}" y="${frameY + frameH * 0.8}" width="${frameW * 0.4}" height="${frameH * 0.035}" rx="6" fill="${p3}" fill-opacity="0.7"/>`;
  } else if (kind === "signage") {
    composition = `<rect x="${frameX + frameW * 0.14}" y="${frameY + frameH * 0.3}" width="${frameW * 0.72}" height="${frameH * 0.28}" rx="18" fill="${p2}" fill-opacity="0.22" stroke="${p2}" stroke-width="6"/><text x="${frameX + frameW * 0.5}" y="${frameY + frameH * 0.48}" fill="${p2}" font-size="62" text-anchor="middle" font-family="Segoe UI, Arial, sans-serif" font-weight="800">TEXT</text><path d="M${frameX + 40} ${frameY + frameH - 55} L${frameX + frameW - 40} ${frameY + 70}" stroke="${p3}" stroke-width="5" stroke-opacity="0.45"/>`;
  } else if (kind === "screen_ui") {
    composition = `<rect x="${frameX + 28}" y="${frameY + 28}" width="${frameW - 56}" height="${frameH - 56}" rx="18" fill="${p0}" stroke="${p3}" stroke-width="4"/><rect x="${frameX + 60}" y="${frameY + 75}" width="${frameW * 0.5}" height="${frameH * 0.5}" rx="10" fill="${p3}" fill-opacity="0.22"/><rect x="${frameX + frameW * 0.66}" y="${frameY + 75}" width="${frameW * 0.22}" height="${frameH * 0.5}" rx="10" fill="${p2}" fill-opacity="0.18"/><path d="M${frameX + 80} ${frameY + frameH * 0.68} H${frameX + frameW - 80} M${frameX + 80} ${frameY + frameH * 0.78} H${frameX + frameW - 130}" stroke="${p2}" stroke-width="8" stroke-linecap="round" opacity="0.75"/>`;
  } else if (kind === "title_card") {
    composition = `<rect x="${frameX}" y="${frameY}" width="${frameW}" height="${frameH}" rx="20" fill="${p0}" fill-opacity="0.7" stroke="${p2}" stroke-opacity="0.5" stroke-width="3"/><text x="${frameX + frameW * 0.5}" y="${frameY + frameH * 0.48}" fill="${p2}" font-size="78" text-anchor="middle" font-family="Georgia, serif" font-weight="700">TITLE</text><rect x="${frameX + frameW * 0.32}" y="${frameY + frameH * 0.57}" width="${frameW * 0.36}" height="10" rx="5" fill="${p3}" fill-opacity="0.65"/>`;
  } else {
    composition = `<rect x="${frameX}" y="${frameY}" width="${frameW}" height="${frameH}" rx="20" fill="${p0}" fill-opacity="0.7" stroke="${p2}" stroke-opacity="0.5" stroke-width="3"/>${itemOverlay}`;
  }
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 1000"><defs><linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="${p0}"/><stop offset="100%" stop-color="${p1}"/></linearGradient><radialGradient id="glow" cx="50%" cy="28%" r="70%"><stop offset="0%" stop-color="${p2}" stop-opacity="0.24"/><stop offset="100%" stop-color="${p2}" stop-opacity="0"/></radialGradient></defs><rect width="1000" height="1000" fill="url(#bg)"/><rect width="1000" height="1000" fill="url(#glow)"/><text x="44" y="70" fill="#FFF6E7" font-size="42" font-weight="800" font-family="Segoe UI, Arial, sans-serif">${label}</text><text x="46" y="112" fill="#BFD7D6" font-size="24" font-family="Segoe UI, Arial, sans-serif">${escapeXml(data?.canvas?.aspect_label || `${canvasW}x${canvasH}`)}</text><rect x="${frameX - 8}" y="${frameY - 8}" width="${frameW + 16}" height="${frameH + 16}" rx="26" fill="#02070A" fill-opacity="0.54" stroke="${p3}" stroke-opacity="0.45" stroke-width="3"/>${composition}${kind.includes("storyboard") ? "" : itemOverlay}\n</svg>`;
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}

function saveDialogWindow() {
  try {
    if (window.top && typeof window.top.showSaveFilePicker === 'function') return window.top;
  } catch {}
  return typeof window.showSaveFilePicker === 'function' ? window : null;
}

function buildIdeoboardPackage(data, boardName = "IAMCCS_Ideoboard") {
  const normalized = normalizeDesignObject(data, defaultData());
  const presetKey = hasPresetKey(normalized.preset_key) ? normalized.preset_key : "storyboard";
  return {
    schema: IDEOBOARD_SCHEMA,
    schema_version: 1,
    board_name: cleanText(boardName) || "IAMCCS_Ideoboard",
    active_preset_key: presetKey,
    boards: {
      [presetKey]: cloneValue({ ...normalized, preset_key: presetKey }, normalized),
    },
    metadata: {
      exported_at: new Date().toISOString(),
      source_node: TYPE,
      width: normalized.canvas?.width || 0,
      height: normalized.canvas?.height || 0,
    },
    assets: {
      preview_svg: boardPreviewSvgDataUri(normalized),
      prompt_json: toPrompt(normalized),
    },
  };
}

function designFromImportedBoard(raw) {
  if (raw && typeof raw === "object" && raw.high_level_description && raw.style_description && raw.compositional_deconstruction) {
    return promptToDesign(raw);
  }
  if (raw && typeof raw === "object" && raw.boards && typeof raw.boards === "object") {
    const activeKey = hasPresetKey(raw.active_preset_key) ? raw.active_preset_key : (hasPresetKey(raw.preset_key) ? raw.preset_key : "storyboard");
    const selected = raw.boards[activeKey];
    if (selected && typeof selected === "object") {
      return normalizeDesignObject({ ...selected, preset_key: activeKey }, defaultData());
    }
  }
  return normalizeDesignObject(raw, defaultData());
}

function inferWorkflowModeFromDesign(data, fallbackKey = "storyboard_grid") {
  const modes = workflowModeEntries();
  if (modes[data?.workflow_mode]) return data.workflow_mode;
  const refMode = normalizeReferenceMode(data?.reference_mode || "single");
  if (refMode.mode === "character_diptych") return "character_diptych";
  if (refMode.mode === "multi_ref_triptych") return "multi_ref_triptych";
  const i2i = normalizeI2I(data?.i2i || {});
  if (i2i.enabled && i2i.source_mode === "refine_source_image") return "image_refine";
  if (i2i.enabled && (data?.items || []).some((item) => normalizeItem(item).kind === "image")) return "image_refine";
  const itemCount = Array.isArray(data?.items) ? data.items.filter((item) => normalizeItem(item).kind !== "image").length : 0;
  if (itemCount >= 4) return "storyboard_grid";
  return modes[fallbackKey] ? fallbackKey : "single_image";
}


function adoptExternalDesign(raw, fallbackMode = "storyboard_grid") {
  const next = designFromImportedBoard(raw);
  const modeKey = inferWorkflowModeFromDesign(next, fallbackMode);
  next.workflow_mode = modeKey;
  return { data: next, modeKey };
}


function writeData(node, data) {
  setWidget(node, "design_data", JSON.stringify(data, null, 2));
}

function writeInputSignature(node, signature) {
  setWidget(node, "ideoboard_input_signature", String(signature || ""));
}

function ensureStyles() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
    .iamccs-isf-mount { box-sizing:border-box; width:100%; height:100%; max-height:100%; min-width:0; min-height:0; padding:8px; overflow:hidden; }
    .iamccs-isf { box-sizing:border-box; width:100%; min-width:0; height:100%; color:#eef4ef; border:1px solid #1f4648; border-radius:16px; overflow:hidden; background:linear-gradient(180deg,#0d1417,#121f24 48%,#0a1013); box-shadow:0 18px 40px rgba(0,0,0,.35); font:12px/1.35 "Segoe UI",system-ui,sans-serif; display:grid; grid-template-rows:auto minmax(0,1fr) auto; }
    .iamccs-isf * { box-sizing:border-box; }
    .iamccs-isf.iamccs-isf-fullscreen { position:fixed; inset:10px; width:auto; height:auto; z-index:2147483640; border-radius:20px; box-shadow:0 30px 80px rgba(0,0,0,.55); }
    .iamccs-isf-head { display:grid; grid-template-columns:minmax(240px,340px) minmax(0,1fr); align-items:start; gap:10px; padding:12px; border-bottom:1px solid rgba(120,200,198,.18); background:radial-gradient(circle at top left,rgba(255,107,53,.15),transparent 28%),linear-gradient(180deg,#122026,#0c1418); }
    .iamccs-isf-title { font-size:16px; font-weight:800; letter-spacing:.02em; color:#fff6e7; }
    .iamccs-isf-sub { color:#9cb8b5; margin-top:2px; max-width:520px; }
    .iamccs-isf-toolbar { display:grid; grid-template-columns:repeat(6,minmax(96px,1fr)); gap:6px; align-items:stretch; min-width:0; }
    .iamccs-isf-btn { border:1px solid #2f676b; background:#122b2f; color:#f0fbfa; border-radius:8px; padding:7px 8px; font-weight:800; cursor:pointer; font-size:10.5px; min-height:30px; text-align:center; }
    .iamccs-isf-btn.primary { background:linear-gradient(180deg,#ffb65d,#f28b45); color:#24170b; border-color:#ffcb86; }
    .iamccs-isf-btn[data-action="toggle-fullscreen"] { background:linear-gradient(180deg,#9ad7ff,#3a8fd6); border-color:#b8e3ff; color:#071625; }
    .iamccs-isf.iamccs-isf-fullscreen .iamccs-isf-btn[data-action="toggle-fullscreen"] { background:linear-gradient(180deg,#ffd6a0,#e47a36); border-color:#ffe0b8; color:#241207; }
    .iamccs-isf-btn[data-action="toggle-show-img"] { background:linear-gradient(180deg,#bfefff,#4898c8); border-color:#bdeaff; color:#071723; }
    .iamccs-isf-btn[data-action="toggle-show-img"].off { background:#152329; border-color:#36545d; color:#8da7ad; }
    .iamccs-isf-btn[data-action="toggle-paper"] { background:linear-gradient(180deg,#fff7e7,#d9c49a); border-color:#ffe1a8; color:#17110a; }
    .iamccs-isf-btn[data-action="save-ideoboard"], .iamccs-isf-btn[data-action="copy-json"] { background:#173923; border-color:#3b8f5b; }
    .iamccs-isf-btn[data-action="import-ideoboard"], .iamccs-isf-btn[data-action="import-preset-gallery"] { background:#182d4f; border-color:#4d83d5; }
    .iamccs-isf-btn[data-action="add-object"], .iamccs-isf-btn[data-action="add-text"], .iamccs-isf-btn[data-action="add-image"] { background:#2d244d; border-color:#8c78d8; }
    .iamccs-isf-btn[data-action="clear-boxes"] { background:#3a1010; border-color:#ff8e76; color:#ffe3dc; }
    .iamccs-isf-btn[data-action="grab-result"], .iamccs-isf-btn[data-action="toggle-auto-compare"], .iamccs-isf-btn[data-action="clear-result"] { background:#4b2f15; border-color:#d7964d; }
    .iamccs-isf-btn[data-action="duplicate"] { background:#283a3c; border-color:#58aeb2; }
    .iamccs-isf-btn[data-action="delete"] { background:#4a1717; border-color:#b84a4a; }
    .iamccs-isf-btn[data-action="reset"] { background:linear-gradient(180deg,#ffb2a1,#d95442); border-color:#ffc5bb; color:#240807; }
    .iamccs-isf-body { display:grid; grid-template-columns:minmax(280px,340px) minmax(0,1fr) minmax(280px,340px); min-height:0; height:100%; max-height:100%; overflow:hidden; }
    .iamccs-isf.iamccs-isf-fullscreen .iamccs-isf-body { grid-template-columns:minmax(280px,18vw) minmax(0,1fr) minmax(280px,20vw); height:calc(100vh - 138px); min-height:0; width:100%; }
    .iamccs-isf-pane { padding:12px; border-right:1px solid rgba(120,200,198,.14); background:rgba(8,14,16,.52); overflow-y:auto; min-height:0; }
    .iamccs-isf-pane:last-child { border-right:none; border-left:1px solid rgba(120,200,198,.14); }
    .iamccs-isf-panel { border:1px solid rgba(100,188,184,.22); border-radius:12px; background:linear-gradient(180deg,rgba(18,30,34,.96),rgba(10,17,19,.96)); padding:12px; margin-bottom:12px; box-shadow:inset 0 1px 0 rgba(255,255,255,.035); }
    .iamccs-isf-panel h4 { margin:0 0 10px; font-size:12px; text-transform:uppercase; letter-spacing:.08em; color:#ffcf8f; }
    .iamccs-isf-panel-headline { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:0 0 10px; }
    .iamccs-isf-panel-headline h4 { margin:0; }
    .iamccs-isf-beta { display:inline-flex; align-items:center; margin-left:6px; padding:2px 6px; border-radius:999px; border:1px solid rgba(255,184,196,.46); color:#ffd6de; background:#36121c; font-size:9px; letter-spacing:.08em; vertical-align:middle; }
    .iamccs-isf-gemma-panel { border-color:rgba(151,229,171,.36); background:linear-gradient(180deg,rgba(16,43,28,.96),rgba(8,18,14,.94)); }
    .iamccs-isf-gemma-panel h4 { color:#bff0c8; }
    .iamccs-isf-gemma-body[hidden], .iamccs-isf-refine-body[hidden] { display:none !important; }
    .iamccs-isf-gemma-result { margin-top:10px; border:1px solid rgba(129,230,170,.34); background:rgba(4,14,10,.72); border-radius:10px; padding:9px; }
    .iamccs-isf-gemma-result[hidden] { display:none !important; }
    .iamccs-isf-gemma-result-head { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:7px; }
    .iamccs-isf-gemma-result-head strong { color:#d9ffe6; font-size:12px; }
    .iamccs-isf-gemma-result-head span { color:#9ee7bd; font-size:10px; text-transform:uppercase; letter-spacing:.04em; }
    .iamccs-isf-gemma-result textarea { width:100%; min-height:130px; max-height:260px; resize:vertical; border:1px solid rgba(129,230,170,.36); border-radius:8px; padding:8px; background:#06110d; color:#eafff1; font:11px/1.42 Consolas,monospace; outline:none; }
    .iamccs-isf-scene-panel { border-color:rgba(119,188,255,.30); background:linear-gradient(180deg,rgba(14,30,42,.96),rgba(8,15,22,.94)); }
    .iamccs-isf-scene-panel h4 { color:#b9dcff; }
    .iamccs-isf-art-panel, .iamccs-isf-style-preset-panel { border-color:rgba(255,207,143,.30); background:linear-gradient(180deg,rgba(37,27,16,.96),rgba(14,17,18,.94)); }
    .iamccs-isf-refine-panel { border-color:rgba(255,139,159,.34); background:linear-gradient(180deg,rgba(42,17,25,.96),rgba(14,16,19,.94)); }
    .iamccs-isf-refine-panel h4 { color:#ffc1cc; }
    .iamccs-isf-panel.mode-panel { border-color:rgba(255,207,143,.42); background:linear-gradient(180deg,rgba(42,31,16,.96),rgba(13,18,20,.94)); }
    .iamccs-isf-mode-grid { display:grid; grid-template-columns:1fr; gap:7px; }
    .iamccs-isf-mode-btn { text-align:left; border:1px solid rgba(255,207,143,.24); background:#0b1417; color:#eaf6f4; border-radius:10px; padding:9px; cursor:pointer; }
    .iamccs-isf-mode-btn strong { display:block; color:#fff6e7; margin-bottom:3px; }
    .iamccs-isf-mode-btn span { display:block; color:#a9c2bf; font-size:11px; line-height:1.3; }
    .iamccs-isf-mode-btn.selected { border-color:#ffcf8f; background:linear-gradient(180deg,#2a2115,#10191b); box-shadow:0 0 0 1px rgba(255,207,143,.24); }
    .iamccs-isf-panel.layers-panel { border-color:rgba(107,185,255,.34); background:linear-gradient(180deg,rgba(15,27,43,.96),rgba(8,15,24,.94)); }
    .iamccs-isf-panel.layers-panel h4 { color:#add1ff; }
    .iamccs-isf-panel.editor-panel { border-color:rgba(255,184,100,.34); background:linear-gradient(180deg,rgba(42,28,16,.96),rgba(20,14,10,.94)); }
    .iamccs-isf-panel.editor-panel h4 { color:#ffc77d; }
    .iamccs-isf-panel.export-panel { border-color:rgba(156,224,155,.3); background:linear-gradient(180deg,rgba(16,34,24,.96),rgba(8,18,13,.94)); }
    .iamccs-isf-panel.export-panel h4 { color:#b9ecb4; }
    .iamccs-isf-preset-grid { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; }
    .iamccs-isf-preset { text-align:center; border:1px solid rgba(97,177,173,.18); background:#0b1417; color:#eaf6f4; border-radius:8px; padding:7px 6px; cursor:pointer; min-height:42px; }
    .iamccs-isf-preset.selected { border-color:#f4b868; box-shadow:0 0 0 1px rgba(244,184,104,.35); background:linear-gradient(180deg,#18242a,#0b1417); }
    .iamccs-isf-preset strong { display:block; color:#fff6e7; margin-bottom:0; font-size:11px; line-height:1.1; }
    .iamccs-isf-preset span { display:none; color:#9cb8b5; }
    .iamccs-isf-gallery-grid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; max-height:320px; overflow:auto; padding-right:4px; }
    .iamccs-isf-gallery-card { text-align:left; border:1px solid rgba(97,177,173,.18); background:#0b1417; color:#eaf6f4; border-radius:10px; padding:8px; cursor:pointer; min-height:158px; display:flex; flex-direction:column; gap:7px; }
    .iamccs-isf-gallery-card:hover { border-color:rgba(244,184,104,.55); }
    .iamccs-isf-gallery-card.selected { border-color:#f4b868; box-shadow:0 0 0 2px rgba(244,184,104,.48), 0 0 24px rgba(244,184,104,.14); background:linear-gradient(180deg,#18242a,#0b1417); }
    .iamccs-isf-gallery-card.imported { border-color:rgba(244,184,104,.52); }
    .iamccs-isf-gallery-badges { display:flex; gap:5px; flex-wrap:wrap; align-items:center; min-height:16px; }
    .iamccs-isf-gallery-badge { border:1px solid rgba(120,200,198,.26); background:#091215; color:#cfe8e4; border-radius:999px; padding:2px 6px; font-size:9px; line-height:1; font-weight:900; text-transform:uppercase; }
    .iamccs-isf-gallery-badge.user { border-color:#f4b868; color:#ffe0a5; background:#21170b; }
    .iamccs-isf-gallery-thumb { height:76px; border-radius:8px; overflow:hidden; background:#05090b; border:1px solid rgba(255,255,255,.08); }
    .iamccs-isf-gallery-thumb img { width:100%; height:100%; object-fit:cover; display:block; }
    .iamccs-isf-gallery-name { color:#fff6e7; font-weight:900; font-size:11px; line-height:1.15; }
    .iamccs-isf-gallery-summary { color:#9cb8b5; font-size:10px; line-height:1.25; max-height:38px; overflow:hidden; }
    .iamccs-isf-gallery-actions { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; margin-top:8px; }
    .iamccs-isf-gallery-actions .iamccs-isf-btn { min-height:32px; font-size:10px; }
    .iamccs-isf-gallery-manager-card { width:min(1280px,96vw); height:min(860px,92vh); display:grid; grid-template-rows:auto auto minmax(0,1fr) auto; background:#071014; color:#eaf6f4; border:1px solid #5fb5b3; border-radius:16px; box-shadow:0 30px 90px rgba(0,0,0,.65); overflow:hidden; }
    .iamccs-isf-gallery-manager-head { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; padding:14px 16px; border-bottom:1px solid rgba(120,200,198,.2); background:linear-gradient(180deg,#13262b,#0b1518); }
    .iamccs-isf-gallery-manager-head strong { display:block; color:#fff6e7; font-size:18px; }
    .iamccs-isf-gallery-manager-head span { display:block; color:#9cb8b5; margin-top:3px; max-width:720px; line-height:1.35; }
    .iamccs-isf-gallery-manager-tags { display:flex; flex-wrap:wrap; gap:7px; padding:10px 16px; border-bottom:1px solid rgba(120,200,198,.14); background:#0a1417; }
    .iamccs-isf-gallery-manager-tag { border:1px solid rgba(120,200,198,.28); background:#101d22; color:#d9fffb; border-radius:999px; padding:6px 10px; font-weight:850; cursor:pointer; font-size:11px; }
    .iamccs-isf-gallery-manager-tag.selected { background:#f4b868; color:#1d1308; border-color:#ffd595; }
    .iamccs-isf-gallery-manager-grid { overflow:auto; padding:14px 16px; display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); grid-auto-rows:376px; gap:12px; align-content:start; }
    .iamccs-isf-gallery-manager-item { border:1px solid rgba(120,200,198,.22); background:#0d171b; border-radius:12px; overflow:hidden; display:grid; grid-template-rows:118px minmax(32px,auto) minmax(0,1fr) auto; min-height:376px; height:376px; }
    .iamccs-isf-gallery-manager-item.user { border-color:rgba(244,184,104,.55); }
    .iamccs-isf-gallery-manager-item.selected { border-color:#f4b868; box-shadow:0 0 0 2px rgba(244,184,104,.42), 0 0 28px rgba(244,184,104,.15); }
    .iamccs-isf-gallery-manager-thumb { height:118px; background:#03080a; border-bottom:1px solid rgba(255,255,255,.08); }
    .iamccs-isf-gallery-manager-thumb img { width:100%; height:100%; object-fit:cover; display:block; }
    .iamccs-isf-gallery-manager-title { padding:9px 10px 0; color:#fff6e7; font-weight:950; font-size:12px; line-height:1.2; }
    .iamccs-isf-gallery-manager-meta { padding:7px 10px; color:#b8ceca; font-size:10.5px; line-height:1.35; display:grid; gap:4px; min-height:0; overflow:auto; align-content:start; }
    .iamccs-isf-gallery-manager-chipline { display:flex; flex-wrap:wrap; gap:4px; }
    .iamccs-isf-gallery-manager-chip { border:1px solid rgba(120,200,198,.22); background:#081114; color:#cfe8e4; border-radius:999px; padding:2px 6px; font-size:9.5px; font-weight:800; }
    .iamccs-isf-gallery-manager-chip.user { border-color:#f4b868; color:#ffe0a5; }
    .iamccs-isf-gallery-manager-actions { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; padding:10px; border-top:1px solid rgba(255,255,255,.08); background:#081114; align-self:end; position:relative; z-index:2; }
    .iamccs-isf-gallery-manager-actions.two { grid-template-columns:repeat(2,minmax(0,1fr)); }
    .iamccs-isf-gallery-manager-actions .iamccs-isf-btn[disabled] { opacity:.55; cursor:not-allowed; }
    .iamccs-isf-gallery-manager-empty { padding:26px; color:#cfe8e4; font-weight:800; }
    .iamccs-isf-gallery-manager-foot { padding:10px 16px; border-top:1px solid rgba(120,200,198,.14); color:#9cb8b5; background:#091316; font-size:11px; }
    .iamccs-isf-sheet-controls { display:grid; grid-template-columns:1fr; gap:8px; }
    .iamccs-isf-sheet-controls select { width:100%; border:1px solid #2b4d50; background:#091215; color:#f7fbfb; border-radius:9px; padding:7px 8px; outline:none; }
    .iamccs-isf-spec-head { display:flex; align-items:center; justify-content:space-between; gap:10px; margin:0 0 8px; }
    .iamccs-isf-spec-head h4 { margin:0; }
    .iamccs-isf-btn.danger-lite { background:#3b1519; border-color:#b94a5a; color:#ffd9de; }
    .iamccs-isf-toggle { display:inline-flex; align-items:center; gap:7px; font-size:11px; color:#dcefeb; user-select:none; }
    .iamccs-isf-toggle input { accent-color:#76d0c8; }
    .iamccs-isf-panel.collapsed .iamccs-isf-sheet-controls { display:none; }
    .iamccs-isf-field label[data-drag-number="true"] { cursor:ew-resize; color:#ffe2ac; }
    .iamccs-isf-mini-actions { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:5px; }
    .iamccs-isf-mini-actions .iamccs-isf-btn { padding:6px 5px; font-size:10px; min-height:28px; }
    .iamccs-isf-layer-actions { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:8px; }
    .iamccs-isf-field { margin-bottom:10px; }
    .iamccs-isf-field label { display:block; margin-bottom:5px; color:#dbe9e7; font-weight:700; }
    .iamccs-isf-field input, .iamccs-isf-field textarea, .iamccs-isf-field select { width:100%; border:1px solid #2b4d50; background:#091215; color:#f7fbfb; border-radius:10px; padding:8px 10px; outline:none; }
    .iamccs-isf-field textarea { min-height:86px; resize:vertical; }
    .iamccs-isf-field-head { display:flex; align-items:center; justify-content:space-between; gap:8px; margin-bottom:5px; }
    .iamccs-isf-field-head label { margin:0; }
    .iamccs-isf-field-actions { display:inline-flex; align-items:center; justify-content:flex-end; gap:4px; flex:0 0 auto; min-width:0; }
    .iamccs-isf-zoom-btn { display:inline-grid; place-items:center; width:24px; height:22px; border:1px solid rgba(255,207,143,.45); background:#2a2115; color:#ffe2ac; border-radius:7px; cursor:pointer; font-weight:900; line-height:1; }
    .iamccs-isf-gemma-mini { display:none; width:24px; height:22px; min-width:24px; border-color:rgba(129,230,170,.58); background:#143628; color:#ccffe1; font-size:11px; }
    .iamccs-isf-gemma-mini.critic { border-color:rgba(159,144,255,.58); background:#2a244b; color:#eeeaff; }
    .iamccs-isf.gemma-enabled .iamccs-isf-gemma-mini { display:inline-grid; }
    .iamccs-isf-zoom-modal { position:fixed; inset:0; background:rgba(0,0,0,.72); z-index:2147483646; display:flex; align-items:center; justify-content:center; padding:24px; box-sizing:border-box; }
    .iamccs-isf-zoom-card { width:min(1080px,94vw); max-height:92vh; display:grid; grid-template-rows:auto 1fr auto; background:#f7f2e8; color:#121212; border:1px solid #d5c4a3; border-radius:12px; box-shadow:0 30px 90px rgba(0,0,0,.55); overflow:hidden; }
    .iamccs-isf-zoom-head { display:flex; justify-content:space-between; align-items:center; gap:12px; padding:14px 16px; background:#fff9ec; border-bottom:1px solid #ddcaa8; }
    .iamccs-isf-zoom-head strong { color:#111; font-size:16px; }
    .iamccs-isf-zoom-body { padding:16px; overflow:auto; }
    .iamccs-isf-zoom-body textarea { width:100%; min-height:58vh; border:1px solid #b9a77f; border-radius:10px; padding:12px; font:14px/1.45 "Segoe UI",system-ui,sans-serif; background:#fff; color:#111; resize:vertical; outline:none; }
    .iamccs-isf-zoom-foot { display:flex; justify-content:flex-end; gap:8px; padding:12px 16px; background:#fff9ec; border-top:1px solid #ddcaa8; }
    .iamccs-isf-field.small textarea { min-height:60px; }
    .iamccs-isf-denoise-guide { border:1px solid rgba(255,207,143,.18); border-radius:12px; padding:10px; margin:2px 0 10px; background:rgba(10,18,20,.66); }
    .iamccs-isf-refine-hero { width:100%; border:1px solid #ffd18a; border-radius:12px; background:linear-gradient(180deg,#ffe1a5,#f4a653); color:#211507; cursor:pointer; padding:12px 10px; margin:0 0 10px; text-align:left; box-shadow:0 10px 22px rgba(0,0,0,.28); }
    .iamccs-isf-refine-hero strong { display:block; font-size:14px; letter-spacing:.04em; }
    .iamccs-isf-refine-hero span { display:block; margin-top:4px; font-size:11px; line-height:1.3; font-weight:700; color:#3b250d; }
    .iamccs-isf-denoise-title { display:flex; justify-content:space-between; gap:8px; align-items:center; color:#fff6e7; font-weight:800; margin-bottom:8px; }
    .iamccs-isf-denoise-value { color:#ffcf8f; font-variant-numeric:tabular-nums; }
    .iamccs-isf-denoise-buttons { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:7px; }
    .iamccs-isf-denoise-preset { text-align:left; border:1px solid rgba(97,177,173,.2); background:#091215; color:#eaf6f4; border-radius:10px; padding:8px; cursor:pointer; min-height:70px; }
    .iamccs-isf-denoise-preset strong { display:flex; justify-content:space-between; gap:6px; color:#fff; margin-bottom:4px; }
    .iamccs-isf-denoise-preset span { display:block; color:#9cb8b5; font-size:11px; line-height:1.25; }
    .iamccs-isf-denoise-preset.selected { border-color:#ffcf8f; box-shadow:0 0 0 1px rgba(255,207,143,.3); background:linear-gradient(180deg,#18242a,#0b1417); }
    .iamccs-isf-denoise-help { margin:8px 0 0; color:#b8ceca; font-size:11px; line-height:1.35; }
    .iamccs-isf-stage-wrap { padding:12px; display:grid; grid-template-rows:auto minmax(0,1fr); gap:8px; align-items:stretch; background:#0b1013; overflow:hidden; min-height:0; height:100%; max-height:100%; }
    .iamccs-isf-stage-meta { display:flex; justify-content:space-between; align-items:center; gap:10px; color:#9eb8b7; }
    .iamccs-isf-stage-meta strong { color:#fff; font-size:13px; }
    .iamccs-isf-board-preview { min-height:0; height:100%; overflow:auto; border:1px solid rgba(120,200,198,.22); border-radius:10px; background:#ece9df; padding:12px; box-sizing:border-box; color:#111; }
    .iamccs-isf-board-sheet { min-height:100%; border:1px solid #c4c0b4; background:#f8f7f2; box-shadow:0 12px 28px rgba(0,0,0,.25); }
    .iamccs-isf-board-title { padding:14px 16px; border-bottom:2px solid #c9c3b6; font-size:28px; font-weight:950; line-height:1.05; color:#111; text-transform:uppercase; }
    .iamccs-isf-board-meta { padding:8px 16px; border-bottom:1px solid #d3cec1; color:#4a4a4a; font-size:13px; font-weight:800; }
    .iamccs-isf-board-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:0; border-top:1px solid #d4d0c7; border-left:1px solid #d4d0c7; }
    .iamccs-isf-board-card { min-height:230px; border-right:1px solid #d4d0c7; border-bottom:1px solid #d4d0c7; background:#fff; display:grid; grid-template-rows:auto minmax(96px,1fr) auto; }
    .iamccs-isf-board-card-head { padding:8px 10px; background:#f3f0e8; color:#111; font-weight:950; font-size:17px; line-height:1.12; border-bottom:1px solid #d8d2c4; }
    .iamccs-isf-board-card-body { padding:10px; color:#111; font-size:15px; line-height:1.34; font-weight:650; overflow:auto; }
    .iamccs-isf-board-card-foot { padding:7px 10px; background:#f7f6f1; color:#5a5a5a; font-size:12px; font-weight:800; border-top:1px solid #e1ddd2; }
    .iamccs-isf-board-image { width:100%; height:180px; object-fit:contain; display:block; background:#111; border-bottom:1px solid #d8d2c4; }
    .iamccs-isf-board-empty { padding:22px; color:#333; font-size:16px; line-height:1.45; font-weight:750; }

    .iamccs-isf-board-preview { min-height:0; height:100%; overflow:auto; border:1px solid rgba(120,200,198,.22); border-radius:10px; background:#0b1013; padding:0; box-sizing:border-box; color:#111; display:flex; flex-direction:column; justify-content:flex-start; align-items:stretch; gap:0; }
    .iamccs-isf-board-artboard { position:relative; flex:0 0 auto; width:100%; max-width:none; aspect-ratio:var(--iamccs-artboard-aspect, 16 / 9); min-height:320px; background:linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(0deg,rgba(255,255,255,.035) 1px,transparent 1px),#16282e; background-size:32px 32px; border:0; box-shadow:none; overflow:hidden; isolation:isolate; }
    .iamccs-isf-board-artboard::before { content:""; position:absolute; inset:0; pointer-events:none; background:linear-gradient(180deg,rgba(255,255,255,.06),transparent 18%,rgba(0,0,0,.12)); z-index:0; }
    .iamccs-isf-result-bg { position:absolute; inset:0; z-index:1; pointer-events:none; background-position:center; background-size:cover; background-repeat:no-repeat; opacity:var(--iamccs-result-opacity,.82); filter:saturate(1.02); }
    .iamccs-isf-result-bg::after { content:""; position:absolute; inset:0; background:rgba(0,0,0,.06); pointer-events:none; }
    .iamccs-isf-show-img-panel { border-color:rgba(134,213,255,.38); background:linear-gradient(180deg,rgba(13,35,48,.96),rgba(8,17,23,.94)); }
    .iamccs-isf-show-img-row { display:grid; grid-template-columns:1fr auto; gap:8px; align-items:center; }
    .iamccs-isf-range-row { display:grid; grid-template-columns:1fr 42px; gap:8px; align-items:center; margin-top:10px; }
    .iamccs-isf-range-row input[type="range"] { width:100%; accent-color:#8bdcff; }
    .iamccs-isf-show-img-note { margin:8px 0 0; color:#9ec1c7; font-size:11px; line-height:1.35; }
    .iamccs-isf-board-artboard-meta { flex:0 0 auto; width:100%; max-width:none; color:#d9fffb; background:rgba(2,9,12,.72); border:0; border-bottom:1px solid rgba(184,255,248,.35); border-radius:0; padding:5px 8px; font-size:11px; font-weight:900; pointer-events:none; text-align:right; }
    .iamccs-isf-board-artboard-empty { position:absolute; inset:0; display:grid; place-items:center; text-align:center; padding:28px; color:#d9fffb; font-size:16px; line-height:1.4; font-weight:800; z-index:1; pointer-events:none; }
    .iamccs-isf-mask-panel { border-color:rgba(255,90,115,.36); background:linear-gradient(180deg,rgba(45,16,24,.96),rgba(13,17,20,.94)); }
    .iamccs-isf-mask-panel h4 { color:#ffb8c4; }
    .iamccs-isf-mask-tools { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:6px; margin-bottom:8px; }
    .iamccs-isf-mask-tool { border:1px solid rgba(255,184,196,.34); background:#171014; color:#ffecef; border-radius:8px; min-height:30px; font-weight:900; cursor:pointer; }
    .iamccs-isf-mask-tool.active { background:linear-gradient(180deg,#ff889a,#c83f5b); color:#21070d; border-color:#ffc4ce; }
    .iamccs-isf-mask-row { display:grid; grid-template-columns:minmax(0,1fr) 82px; gap:8px; align-items:end; }
    .iamccs-isf-mask-status { margin:8px 0 0; color:#ffc7d0; font-size:11px; line-height:1.35; }
    .iamccs-isf-mask-svg { position:absolute; left:0; top:0; width:1px; height:1px; z-index:48; pointer-events:none; overflow:visible; }
    .iamccs-isf-mask-svg .mask-fill { fill:rgba(255,28,62,.50); stroke:rgba(255,28,62,.96); stroke-width:8; vector-effect:non-scaling-stroke; }
    .iamccs-isf-mask-svg .mask-stroke { fill:none; stroke:rgba(255,28,62,.92); stroke-width:18; stroke-linecap:round; stroke-linejoin:round; vector-effect:non-scaling-stroke; }
    .iamccs-isf-mask-svg .mask-erase { fill:rgba(30,30,30,.55); stroke:rgba(255,255,255,.78); }
    .iamccs-isf-mask-canvas { position:absolute; left:0; top:0; width:1px; height:1px; z-index:50; opacity:0.01; pointer-events:none; touch-action:none; user-select:none; }
    .iamccs-isf-item.mask-paint-active .iamccs-isf-mask-canvas,
    .iamccs-isf-image-editor-frame.mask-paint-active .iamccs-isf-mask-canvas { pointer-events:auto; cursor:crosshair; }
    .iamccs-isf-image-editor { width:100%; max-width:1480px; display:grid; grid-template-rows:auto minmax(0,1fr); gap:10px; }
    .iamccs-isf-image-editor-meta { display:flex; justify-content:space-between; align-items:center; gap:10px; color:#d9fffb; background:rgba(2,9,12,.78); border:1px solid rgba(184,255,248,.35); border-radius:8px; padding:7px 10px; font-size:12px; font-weight:900; }
    .iamccs-isf-image-editor-frame { position:relative; width:100%; aspect-ratio:var(--iamccs-image-editor-aspect,16/9); min-height:320px; max-height:calc(100vh - 260px); background:#05080a; border:2px solid rgba(119,188,255,.98); border-radius:8px; overflow:hidden; box-shadow:0 16px 42px rgba(0,0,0,.42); }
    .iamccs-isf-image-editor-frame.mask-paint-active { box-shadow:0 0 0 2px rgba(255,70,100,.45),0 0 0 6px rgba(255,70,100,.18),0 16px 42px rgba(0,0,0,.42); }
    .iamccs-isf-image-editor-frame.mask-paint-active::after { content:"MASK ACTIVE"; position:absolute; right:10px; top:10px; z-index:60; pointer-events:none; color:#ffd5dd; background:rgba(42,5,13,.76); border:1px solid rgba(255,88,115,.62); border-radius:7px; padding:4px 7px; font-size:11px; font-weight:900; letter-spacing:.04em; }
    .iamccs-isf-image-editor-frame .iamccs-isf-item-image { position:absolute; inset:0; width:100%; height:100%; z-index:0; opacity:1; }
    .iamccs-isf-image-editor-empty { min-height:360px; display:grid; place-items:center; text-align:center; color:#d9fffb; background:linear-gradient(90deg,rgba(255,255,255,.035) 1px,transparent 1px),linear-gradient(0deg,rgba(255,255,255,.035) 1px,transparent 1px),#101a1f; background-size:32px 32px; border:2px dashed rgba(184,255,248,.4); border-radius:8px; padding:24px; font-size:16px; line-height:1.4; font-weight:850; }
    .iamccs-isf-image-editor-empty button { margin-top:12px; }
    .iamccs-isf-image-editor-frame .iamccs-isf-mask-svg { left:0; top:0; width:100%; height:100%; z-index:48; }
    .iamccs-isf-image-editor-frame .iamccs-isf-mask-canvas { left:0; top:0; width:100%; height:100%; z-index:50; opacity:1; }
    .iamccs-isf-item.mask-paint-active .iamccs-isf-item-head,
    .iamccs-isf-item.mask-paint-active .iamccs-isf-item-body,
    .iamccs-isf-item.mask-paint-active .iamccs-isf-handle { pointer-events:none; }
    .iamccs-isf-item.mask-paint-active { box-shadow:0 0 0 2px rgba(255,70,100,.45),0 0 0 6px rgba(255,70,100,.18),0 14px 28px rgba(0,0,0,.3); cursor:crosshair; }
    .iamccs-isf-board-artboard .iamccs-isf-item { cursor:grab; }
    .iamccs-isf-board-artboard .iamccs-isf-item:active { cursor:grabbing; }
    .iamccs-isf-board-artboard .iamccs-isf-item-head { min-height:34px; }
    .iamccs-isf-board-artboard .iamccs-isf-item-body { max-height:calc(100% - 34px); overflow:auto; }
    .iamccs-isf-zoom-btn { flex:0 0 auto; display:inline-grid; place-items:center; width:26px; height:24px; min-width:26px; border:1px solid rgba(255,207,143,.55); background:#2a2115; color:#ffe2ac; border-radius:7px; cursor:pointer; font-size:15px; font-weight:900; line-height:1; padding:0; text-align:center; }
    .iamccs-isf-field-head { display:grid; grid-template-columns:minmax(0,1fr) auto; align-items:center; gap:8px; margin-bottom:5px; }
    .iamccs-isf-field-head label { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .iamccs-isf-gemma-mini { width:26px; height:24px; min-width:26px; font-size:11px; }
    .iamccs-isf .iamccs-isf-gemma-mini { display:none; }
    .iamccs-isf.gemma-enabled .iamccs-isf-gemma-mini { display:inline-grid; }

    .iamccs-isf-toast { position:fixed; z-index:2147483647; max-width:420px; padding:12px 14px; border:1px solid rgba(154,215,255,.45); border-radius:12px; background:rgba(8,14,18,.94); color:#f4fbff; box-shadow:0 18px 44px rgba(0,0,0,.42); font:12px/1.35 "Segoe UI",system-ui,sans-serif; pointer-events:none; opacity:0; transform:translateY(6px); transition:opacity .18s ease,transform .18s ease; }
    .iamccs-isf-toast.visible { opacity:1; transform:translateY(0); }
    .iamccs-isf-toast.success { border-color:rgba(98,220,146,.6); background:rgba(14,42,28,.95); }
    .iamccs-isf-toast.warn { border-color:rgba(255,198,103,.7); background:rgba(58,39,13,.95); }
    .iamccs-isf-toast.error { border-color:rgba(255,116,116,.72); background:rgba(58,16,16,.95); }
    .iamccs-isf-direct-note { margin:8px 0 0; color:#a9c2bf; font-size:11px; }
    .iamccs-isf-direct-actions { display:grid; grid-template-columns:1fr 1fr; gap:6px; margin-top:8px; }
    .iamccs-isf-item { z-index:2; }
    .iamccs-isf-item { position:absolute; display:block; min-width:48px; min-height:42px; border-radius:4px; border:3px solid rgba(190,255,248,.95); background:rgba(8,22,26,var(--iamccs-box-fill-opacity,.82)); box-shadow:0 0 0 1px rgba(0,0,0,.82),0 8px 22px rgba(0,0,0,.32); cursor:default; overflow:hidden; }
    .iamccs-isf-item.text { border-color:rgba(255,205,128,.96); background:rgba(38,22,8,.82); }
    .iamccs-isf-item.image { border-color:rgba(119,188,255,.98); background:rgba(10,28,52,.58); }
    .iamccs-isf-item.mask { border-color:rgba(255,255,255,.96); background:rgba(255,255,255,.25); box-shadow:0 0 0 2px rgba(255,77,109,.42), inset 0 0 18px rgba(255,255,255,.2), 0 8px 22px rgba(0,0,0,.22); }
    .iamccs-isf-item.mask .iamccs-isf-item-head { background:rgba(76,13,29,.78); color:#fff; }
    .iamccs-isf-item.mask .iamccs-isf-item-kind { color:#fff; }
    .iamccs-isf-item.mask .iamccs-isf-item-body { color:#fff; text-shadow:0 1px 2px #000; }
    .iamccs-isf-item-image { position:absolute; inset:0; width:100%; height:100%; object-fit:cover; opacity:.86; z-index:0; }
    .iamccs-isf-item.image .iamccs-isf-item-head, .iamccs-isf-item.image .iamccs-isf-item-body { position:relative; z-index:1; }
    .iamccs-isf-item.image .iamccs-isf-item-body { background:linear-gradient(180deg,rgba(0,0,0,.15),rgba(0,0,0,.62)); position:absolute; left:0; right:0; bottom:0; max-height:54%; overflow:hidden; }
    .iamccs-isf-item.selected { z-index:900 !important; box-shadow:0 0 0 2px rgba(255,255,255,.18),0 0 0 5px rgba(72,194,186,.28),0 14px 28px rgba(0,0,0,.3); }
    .iamccs-isf-item-head { display:flex; justify-content:space-between; align-items:center; gap:8px; padding:8px 10px; background:rgba(3,7,9,var(--iamccs-box-head-opacity,.86)); font-weight:900; color:#fff8ea; text-shadow:0 1px 0 rgba(0,0,0,.55); cursor:grab; font-size:16px; line-height:1.05; }
    .iamccs-isf-item-kind { font-size:12px; letter-spacing:.05em; text-transform:uppercase; color:#96e8e2; }
    .iamccs-isf-item.text .iamccs-isf-item-kind { color:#ffd39e; }
    .iamccs-isf-item-body { padding:10px; color:#f4fffe; font-size:18px; line-height:1.22; font-weight:750; text-shadow:0 1px 2px rgba(0,0,0,.9); }
    .iamccs-isf.iamccs-isf-fullscreen .iamccs-isf-item-body { font-size:19px; }
    .iamccs-isf.iamccs-isf-paper .iamccs-isf-pane .iamccs-isf-field input,
    .iamccs-isf.iamccs-isf-paper .iamccs-isf-pane .iamccs-isf-field textarea,
    .iamccs-isf.iamccs-isf-paper .iamccs-isf-pane .iamccs-isf-preview {
      background:#fffdf7;
      color:#111;
      border-color:#c8b27f;
      box-shadow:inset 0 0 0 1px rgba(199,157,78,.18);
    }
    .iamccs-isf.iamccs-isf-paper .iamccs-isf-pane .iamccs-isf-field input:focus,
    .iamccs-isf.iamccs-isf-paper .iamccs-isf-pane .iamccs-isf-field textarea:focus,
    .iamccs-isf.iamccs-isf-paper .iamccs-isf-pane .iamccs-isf-preview:focus {
      background:#fff;
      color:#000;
      border-color:#d39a3d;
      box-shadow:0 0 0 2px rgba(211,154,61,.22), inset 0 0 0 1px rgba(0,0,0,.04);
    }
    .iamccs-isf.iamccs-isf-paper .editor-panel .iamccs-isf-field input,
    .iamccs-isf.iamccs-isf-paper .editor-panel .iamccs-isf-field textarea,
    .iamccs-isf.iamccs-isf-paper .editor-panel .iamccs-isf-field input:focus,
    .iamccs-isf.iamccs-isf-paper .editor-panel .iamccs-isf-field textarea:focus {
      background:#061a1d;
      color:#eaf8f5;
      border-color:#24494b;
      box-shadow:none;
    }
    .iamccs-isf-item-body[contenteditable="true"] { cursor:text; outline:none; border-radius:8px; background:rgba(0,0,0,var(--iamccs-box-body-opacity,.22)); min-height:28px; }
    .iamccs-isf-item-body[contenteditable="true"]:focus { box-shadow:inset 0 0 0 1px rgba(255,207,143,.75); background:rgba(0,0,0,.38); color:#fff6e7; }
    .iamccs-isf-handle { position:absolute; z-index:3; width:20px; height:20px; right:8px; bottom:8px; border-radius:50%; border:2px solid rgba(20,30,38,.9); background:#fff; cursor:nwse-resize; box-shadow:0 2px 10px rgba(0,0,0,.42); }
    .iamccs-isf-list { display:flex; flex-direction:column; gap:8px; min-height:260px; max-height:430px; overflow:auto; }
    .iamccs-isf-card { border:1px solid rgba(97,177,173,.18); background:#0b1417; border-radius:12px; padding:10px; cursor:pointer; }
    .iamccs-isf-card.selected { border-color:#f4b868; box-shadow:0 0 0 1px rgba(244,184,104,.35); }
    .iamccs-isf-card-title { display:grid; grid-template-columns:minmax(0,1fr) auto auto; align-items:center; gap:8px; font-weight:800; color:#fff; }
    .iamccs-isf-card-title span:first-child { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
    .iamccs-isf-delete { display:inline-grid; place-items:center; width:22px; height:22px; border:1px solid rgba(255,142,118,.5); border-radius:7px; background:#34120f; color:#ffd1c8; cursor:pointer; font-weight:900; line-height:1; }
    .iamccs-isf-delete:hover { border-color:#ffb3a5; background:#551d17; color:#fff; }
    .iamccs-isf-chip { display:inline-flex; align-items:center; padding:2px 8px; border-radius:999px; background:#143438; color:#9fece5; font-size:10px; text-transform:uppercase; letter-spacing:.08em; }
    .iamccs-isf-chip.text { background:#3a2614; color:#ffd8a5; }
    .iamccs-isf-chip.image { background:#14233a; color:#b7d3ff; }
    .iamccs-isf-card p { margin:6px 0 0; color:#a7c2bf; }
    .iamccs-isf-palette { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
    .iamccs-isf-swatch { width:18px; height:18px; border-radius:50%; border:1px solid rgba(255,255,255,.24); box-shadow:0 2px 6px rgba(0,0,0,.25); }
    .iamccs-isf-posgrid { display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:8px; }
    .iamccs-isf-preview { width:100%; min-height:220px; border:1px solid #2b4d50; background:#091215; color:#d8eeeb; border-radius:10px; padding:8px 10px; resize:vertical; font:11px/1.4 Consolas,monospace; }
    .iamccs-isf-preview.override { border-color:#ffcf8f; background:#171009; color:#ffe8bd; box-shadow:0 0 0 1px rgba(255,207,143,.28); }
    .iamccs-isf-json-tools { display:grid; grid-template-columns:auto 1fr auto; gap:6px; align-items:center; margin:0 0 8px; }
    .iamccs-isf-json-tools .iamccs-isf-btn { min-height:26px; padding:5px 7px; }
    .iamccs-jsonpass { box-sizing:border-box; width:100%; height:100%; padding:8px; color:#eef4ef; background:#10181b; border:1px solid #2f676b; border-radius:10px; display:grid; grid-template-rows:auto 1fr; gap:8px; font:12px/1.35 "Segoe UI",system-ui,sans-serif; overflow:hidden; }
    .iamccs-jsonpass strong { color:#ffcf8f; }
    .iamccs-jsonpass textarea { width:100%; min-height:160px; resize:none; border:1px solid #2b4d50; background:#071013; color:#d8eeeb; border-radius:8px; padding:8px; font:11px/1.4 Consolas,monospace; }
    .iamccs-isf-foot { display:flex; justify-content:space-between; gap:10px; padding:12px 16px; border-top:1px solid rgba(120,200,198,.12); color:#95b3af; background:#0a1012; }
  `;
  document.head.appendChild(style);
}

function install(node) {
  if (!node || typeof node.addDOMWidget !== "function") return;
  if (node._iamccsIdeogramStoryboardReady && node._iamccsIdeogramStoryboardBuild === JS_BUILD) return;
  if (node._iamccsIdeogramStoryboardReady && node._iamccsIdeogramStoryboardBuild !== JS_BUILD && Array.isArray(node.widgets)) {
    node.widgets = node.widgets.filter((widget) => widget?.name !== "IAMCCS StoryboardFrame Director" && widget?.type !== "iamccs_storyboard_frame_designer");
  }
  node._iamccsIdeogramStoryboardReady = true;
  node._iamccsIdeogramStoryboardBuild = JS_BUILD;
  const nodeType = node?.comfyClass || node?.type || node?.constructor?.type || '';
  const isV2 = nodeType === TYPE_V2 || node?.type === TYPE_V2;
  ensureStyles();
  hideWidget(widget(node, "design_data"));
  hideWidget(widget(node, "ideoboard_input_signature"));
  hideWidget(widget(node, "i2i_enabled"));
  hideWidget(widget(node, "i2i_denoise"));
  hideWidget(widget(node, "low_sigma_start_step"));

  const state = {
    data: parseData(node),
    selectedId: null,
    drag: null,
    resultBgUrl: "",
    resultBgFinal: false,
    resultOverlayOpacity: 0.82,
    resultOverlaySplit: 50,
    resultOverlayVisible: true,
    showResultImage: true,
    autoResultCompare: true,
    liveResult: false,
    showSheetSpecs: false,
    showRefineGuide: true,
    paperMode: false,
    gallerySelectedKey: "",
    styleSelectedKey: "default_photoreal_cinema",
    gridSelectedKey: "free_canvas",
    targetResolutionKey: "hd_720",
    workflowModeKey: "single_image",
    paintTool: "select",
    paintDrawing: null,
    maskDebugLog: [],
    gemmaRunning: false,
    gemmaAbortController: null,
    gemmaPending: null,
    gemmaRunId: 0,
    gemmaCanceledRunIds: new Set(),
  };
  state.paperMode = Boolean(state.data?.ui?.paper_mode);
  state.showResultImage = state.data?.ui?.show_img !== false;
  state.showRefineGuide = state.data?.ui?.show_refine_guide !== false;
  state.resultOverlayOpacity = Math.max(0, Math.min(1, Number(state.data?.ui?.show_img_opacity ?? state.resultOverlayOpacity) || state.resultOverlayOpacity));
  state.workflowModeKey = inferWorkflowModeFromDesign(state.data, state.data?.workflow_mode || "single_image");
  state.data.workflow_mode = state.workflowModeKey;
  if (state.data?.grid_key && gridEntries()[state.data.grid_key]) {
    state.gridSelectedKey = state.data.grid_key;
  }
  if (state.data?.preset_key && hasPresetKey(state.data.preset_key)) {
    state.gallerySelectedKey = state.data.preset_key;
  }
  if (state.data?.canvas?.target_resolution_key && targetResolutionEntries()[state.data.canvas.target_resolution_key]) {
    state.targetResolutionKey = state.data.canvas.target_resolution_key;
  }
  state.selectedId = state.data.items?.[0]?.id || null;
  writeData(node, state.data);

  const mountHost = document.createElement("div");
  mountHost.className = "iamccs-isf-mount";

  const root = document.createElement("div");
  root.className = `iamccs-isf ${state.paperMode ? "iamccs-isf-paper" : ""}`.trim();
  root.innerHTML = `
    <div class="iamccs-isf-head">
      <div>
        <div class="iamccs-isf-title">IAMCCS StoryboardFrame + TextInFrame Director</div>
        <div class="iamccs-isf-sub">Visual layout canvas for Ideogram-style shot design, prop text, signage, title zones, and art-directed frame composition.</div>
      </div>
      <div class="iamccs-isf-toolbar">
        <button class="iamccs-isf-btn" type="button" data-action="save-ideoboard">Save Ideoboard</button>
        <button class="iamccs-isf-btn" type="button" data-action="import-ideoboard">Load Ideoboard (Replace)</button>
        <button class="iamccs-isf-btn" type="button" data-action="toggle-fullscreen">Open Editor</button>
        <button class="iamccs-isf-btn" type="button" data-action="toggle-show-img">Show Img</button>
        <button class="iamccs-isf-btn" type="button" data-action="toggle-paper">Paper</button>
        <button class="iamccs-isf-btn" type="button" data-action="copy-json">Copy Prompt JSON</button>
        <button class="iamccs-isf-btn" type="button" data-action="add-object">Add Object</button>
        <button class="iamccs-isf-btn" type="button" data-action="add-text">Add Text</button>
        <button class="iamccs-isf-btn" type="button" data-action="add-image">Add Image</button>
        <button class="iamccs-isf-btn" type="button" data-action="duplicate">Duplicate</button>
        <button class="iamccs-isf-btn" type="button" data-action="delete">Delete</button>
        <button class="iamccs-isf-btn" type="button" data-action="clear-boxes">Clear Boxes</button>
        <button class="iamccs-isf-btn primary" type="button" data-action="reset">Reset Layout</button>
      </div>
    </div>
    <div class="iamccs-isf-body">
      <div class="iamccs-isf-pane" data-pane="scene"></div>
      <div class="iamccs-isf-stage-wrap">
        <div class="iamccs-isf-stage-meta">
          <strong>IAMCCS Frame Preview</strong>
          <span data-stage-size></span>
        </div>
        <div class="iamccs-isf-board-preview" data-board-preview></div>
      </div>
      <div class="iamccs-isf-pane" data-pane="inspector"></div>
    </div>
    <div class="iamccs-isf-foot"><span>patreon.com/IAMCCS</span><span data-foot-status></span></div>
    <input type="file" accept=".ideoboard.json,.json" data-role="ideoboard-import" style="display:none" />
    <input type="file" accept=".json,application/json" data-role="preset-gallery-import" style="display:none" />
    <input type="file" accept=".json,application/json" data-role="style-preset-import" style="display:none" />
    <input type="file" accept=".json,application/json" data-role="grid-preset-import" style="display:none" />
    <input type="file" accept="image/*" multiple data-role="image-import" style="display:none" />
  `;

  const scenePane = root.querySelector('[data-pane="scene"]');
  const inspectorPane = root.querySelector('[data-pane="inspector"]');
  const boardPreview = root.querySelector('[data-board-preview]');
  let artboard = boardPreview;
  const footStatus = root.querySelector('[data-foot-status]');
  const stageSize = root.querySelector('[data-stage-size]');
  const fullscreenButton = root.querySelector('[data-action="toggle-fullscreen"]');
  const showImgButton = root.querySelector('[data-action="toggle-show-img"]');
  const paperButton = root.querySelector('[data-action="toggle-paper"]');
  const importInput = root.querySelector('[data-role="ideoboard-import"]');
  const presetGalleryInput = root.querySelector('[data-role="preset-gallery-import"]');
  const stylePresetInput = root.querySelector('[data-role="style-preset-import"]');
  const gridPresetInput = root.querySelector('[data-role="grid-preset-import"]');
  const imageInput = root.querySelector('[data-role="image-import"]');
  if (!isV2) {
    const imageButton = root.querySelector('[data-action="add-image"]');
    if (imageButton) imageButton.style.display = 'none';
  }

  const sceneFields = {};
  const itemFields = {};
  const i2iFields = {};
  const directPromptFields = {};
  let directPromptPanel = null;
  let previewField = null;
  let layerListHost = null;
  let maskBrushSizeInput = null;
  let maskStatusHost = null;
  let maskDebugHost = null;
  let maskPreviewCanvas = null;
  let fullscreenHost = null;

  const FRAME_NODE_SIZE = [1960, 1340];
  const FRAME_WIDGET_SIZE = [1932, 1248];
  function applyNodeSize() {
    node.resizable = false;
    const current = Array.isArray(node.size) ? node.size : [0, 0];
    if (Math.abs(Number(current[0] || 0) - FRAME_NODE_SIZE[0]) > 1 || Math.abs(Number(current[1] || 0) - FRAME_NODE_SIZE[1]) > 1) {
      node.size = FRAME_NODE_SIZE.slice();
    }
    mountHost.style.width = `${FRAME_WIDGET_SIZE[0]}px`;
    mountHost.style.height = `${FRAME_WIDGET_SIZE[1]}px`;
    mountHost.style.maxHeight = `${FRAME_WIDGET_SIZE[1]}px`;
    root.style.width = "100%";
    root.style.height = "100%";
    root.style.maxHeight = "100%";
    node.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
  }


  function showToast(message, options = {}) {
    const toast = document.createElement('div');
    toast.className = `iamccs-isf-toast ${options.tone || ''}`.trim();
    toast.textContent = String(message || '');
    const ms = Number(options.ms || 3000);
    document.body.appendChild(toast);
    const anchor = options.anchor;
    if (options.center || !anchor?.getBoundingClientRect) {
      toast.style.left = '50%';
      toast.style.top = '50%';
      toast.style.transform = 'translate(-50%, calc(-50% + 6px))';
      requestAnimationFrame(() => {
        toast.classList.add('visible');
        toast.style.transform = 'translate(-50%, -50%)';
      });
    } else {
      const rect = anchor.getBoundingClientRect();
      const left = Math.min(window.innerWidth - 440, Math.max(12, rect.right + 10));
      const top = Math.min(window.innerHeight - 80, Math.max(12, rect.top));
      toast.style.left = `${left}px`;
      toast.style.top = `${top}px`;
      requestAnimationFrame(() => toast.classList.add('visible'));
    }
    window.setTimeout(() => {
      toast.classList.remove('visible');
      window.setTimeout(() => toast.remove(), 220);
    }, ms);
    if (message) footStatus.textContent = String(message);
  }

  function canPaintImageMaskInCurrentWorkflow() {
    const modeKey = state.workflowModeKey || state.data?.workflow_mode || "";
    const i2i = normalizeI2I(state.data?.i2i || {});
    return modeKey === "image_refine" || (i2i.enabled && i2i.source_mode === "refine_source_image");
  }

  function guardImageMaskPaint(anchor = null) {
    if (canPaintImageMaskInCurrentWorkflow()) return true;
    showToast("Please open image in Image Refine/i2i workflow to inpaint.", { anchor, center: !anchor, tone: "warn", ms: 3600 });
    state.paintTool = "select";
    state.paintDrawing = null;
    state.imageEditorDrawing = null;
    syncMaskUi();
    return false;
  }

  function setResultBackground(url, final = true, status = "") {
    state.resultBgUrl = cleanText(url);
    state.resultBgFinal = Boolean(final);
    state.resultOverlayVisible = Boolean(state.resultBgUrl);
    if (status) footStatus.textContent = status;
    if (state.resultBgUrl && state.showResultImage && state.workflowModeKey !== "image_refine") {
      renderArtboard();
    } else {
      applyResultCompareCss();
    }
  }

  node._iamccsSetResultBackground = (url, final = true) => {
    if (!state.showResultImage) {
      state.resultBgUrl = cleanText(url || state.resultBgUrl);
      state.resultBgFinal = Boolean(final);
      state.resultOverlayVisible = Boolean(state.resultBgUrl);
      return;
    }
    setResultBackground(url, final, final ? "Generated image shown under boxes" : "Live preview shown under boxes");
  };

  function grabLastResultBackground(anchor = null) {
    const preferred = state.data?.workflow_mode === "storyboard_grid" ? (lastIamccsStoryboardSourceImage || lastIamccsResultImage) : lastIamccsResultImage;
    const url = iamccsResultViewUrl(preferred);
    if (!url) {
      showToast("No generated result available yet. Run a generation first.", { center: true, tone: "warn", ms: 3000 });
      return;
    }
    setResultBackground(url, true, state.data?.workflow_mode === "storyboard_grid" ? "Grabbed generated source sheet for box compare" : "Grabbed last generated result as visual background");
    showToast("Loaded generated result compare", { anchor, tone: "success" });
  }

  function clearResultBackground() {
    state.liveResult = false;
    iamccsLiveFrameNodes.delete(node);
    state.resultBgUrl = "";
    state.resultBgFinal = false;
    state.resultOverlayVisible = false;
    syncShowImgControls();
    renderArtboard();
  }


  function currentItem() {
    return (state.data.items || []).find((entry) => entry.id === state.selectedId) || null;
  }

  function syncSelectionState() {
    requestAnimationFrame(() => {
      const selectedCard = layerListHost?.querySelector(`[data-item-id="${state.selectedId}"]`);
      selectedCard?.scrollIntoView({ block: 'nearest' });
    });
  }

  function syncPaperModeButton() {
    root.classList.toggle("iamccs-isf-paper", Boolean(state.paperMode));
    if (paperButton) paperButton.textContent = state.paperMode ? "Paper On" : "Paper";
  }

  function persistUiState() {
    state.data.ui = state.data.ui && typeof state.data.ui === "object" ? state.data.ui : {};
    state.data.ui.paper_mode = Boolean(state.paperMode);
    state.data.ui.show_img = Boolean(state.showResultImage);
    state.data.ui.show_refine_guide = Boolean(state.showRefineGuide);
    state.data.ui.show_img_opacity = Math.max(0, Math.min(1, Number(state.resultOverlayOpacity) || 0));
    writeData(node, state.data);
  }

  function syncShowImgControls() {
    if (state.showResultImage) iamccsLiveFrameNodes.add(node);
    else iamccsLiveFrameNodes.delete(node);
    if (showImgButton) {
      showImgButton.textContent = state.showResultImage ? "Show Img On" : "Show Img";
      showImgButton.classList.toggle("off", !state.showResultImage);
      showImgButton.setAttribute("aria-pressed", state.showResultImage ? "true" : "false");
    }
    const toggle = root.querySelector('[data-role="show-img-toggle"]');
    const opacity = root.querySelector('[data-role="show-img-opacity"]');
    const readout = root.querySelector('[data-role="show-img-opacity-readout"]');
    if (toggle) toggle.checked = Boolean(state.showResultImage);
    if (opacity) opacity.value = String(Math.round(Math.max(0, Math.min(1, Number(state.resultOverlayOpacity) || 0)) * 100));
    if (readout) readout.textContent = `${Math.round(Math.max(0, Math.min(1, Number(state.resultOverlayOpacity) || 0)) * 100)}%`;
    applyResultCompareCss();
  }

  function toggleFullscreen(force) {
    const next = typeof force === 'boolean' ? force : !root.classList.contains('iamccs-isf-fullscreen');
    if (next) {
      if (!fullscreenHost) {
        fullscreenHost = document.createElement('div');
        fullscreenHost.style.position = 'fixed';
        fullscreenHost.style.inset = '0';
        fullscreenHost.style.zIndex = '2147483639';
        fullscreenHost.style.background = 'rgba(3, 6, 8, 0.72)';
      }
      document.body.appendChild(fullscreenHost);
      fullscreenHost.appendChild(root);
    } else if (fullscreenHost?.parentNode) {
      fullscreenHost.remove();
      mountHost.appendChild(root);
      root.style.display = '';
      root.getBoundingClientRect();
      requestAnimationFrame(() => {
        applyNodeSize();
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
      });
    }
    root.classList.toggle('iamccs-isf-fullscreen', next);
    if (fullscreenButton) {
      fullscreenButton.textContent = next ? 'Close Editor' : 'Open Editor';
    }
  }

  function downloadJsonFile(filename, payload) {
    try {
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 500);
      return true;
    } catch (error) {
      console.warn("[IAMCCS FrameDesigner] JSON download failed", error);
      return false;
    }
  }

  async function downloadIdeoboard(anchor = null) {
    const suggested = `${safeFilename(`IAMCCS_${state.data.preset_key}_board`)}.ideoboard.json`;
    const payloadData = buildIdeoboardPackage(state.data, suggested.replace(/\.ideoboard\.json$/i, ''));
    const payload = JSON.stringify(payloadData, null, 2);
    showToast('Preparing Ideoboard save...', { anchor });
    const pickerWindow = saveDialogWindow();
    if (pickerWindow) {
      try {
        const handle = await pickerWindow.showSaveFilePicker({
          suggestedName: suggested,
          types: [{
            description: 'IAMCCS Ideoboard JSON',
            accept: { 'application/json': ['.json'] },
          }],
        });
        const writable = await handle.createWritable();
        await writable.write(payload);
        await writable.close();
        showToast(`Ideoboard saved to ${handle.name}`, { anchor, tone: "success" });
        return;
      } catch (error) {
        if (error?.name === 'AbortError') {
          showToast('Ideoboard save cancelled', { anchor, tone: "warn" });
          return;
        }
        showToast(`Save As unavailable, using browser download (${error?.name || 'runtime error'})`, { anchor, tone: "warn" });
      }
    }
    try {
      const filename = suggested;
      const blob = new Blob([payload], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      link.rel = 'noopener';
      link.style.display = 'none';
      (document.body || root).appendChild(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      showToast(`Ideoboard downloaded as ${filename}`, { anchor, tone: "success" });
    } catch (error) {
      showToast(`Ideoboard save failed (${error?.name || 'runtime error'})`, { anchor, tone: "error" });
    }
  }

  function hasBoardContent() {
    const scene = state.data?.scene || {};
    return Boolean((state.data?.items || []).length || cleanText(scene.high_level_description) || cleanText(scene.aesthetics) || cleanText(scene.background));
  }

  function importIdeoboardFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        if (hasBoardContent()) showToast('Current board replaced by imported Ideoboard', { center: true, tone: "warn", ms: 2200 });
        const imported = adoptExternalDesign(parseJsonLoose(String(reader.result || '')), state.workflowModeKey);
        state.data = imported.data;
        state.workflowModeKey = imported.modeKey;
        state.selectedId = state.data.items[0]?.id || null;
        writeInputSignature(node, "__manual_import__");
        persist();
        render();
        footStatus.textContent = `Imported ideoboard ${file.name}`;
      } catch {
        footStatus.textContent = `Import failed for ${file.name}`;
      }
    };
    reader.readAsText(file);
  }


  function importPresetGalleryFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = parseJsonLoose(String(reader.result || ""));
        const schema = cleanText(parsed?.schema);
        let gallery = {};
        let importedType = "Preset Gallery";

        if (schema === "iamccs.ideogram_storyboard_frame_designer") {
          gallery = normalizePresetGallery(parsed);
          importedType = "Ideoboard converted to Preset Gallery";
        } else if (
          schema === "iamccs.frame_v2.preset_gallery" ||
          schema === "iamccs.frame_designer.preset_gallery" ||
          parsed?.preset_gallery ||
          parsed?.frame_presets ||
          parsed?.gallery
        ) {
          gallery = normalizePresetMap(parsed.presets || parsed.gallery || parsed.preset_gallery || parsed.frame_presets, normalizePresetEntry, "gallery");
        } else {
          const styleCount = Object.keys({ ...normalizeStyleGallery(parsed), ...directStylePresetsFromJson(parsed) }).length;
          const gridCount = Object.keys({ ...normalizeGridGallery(parsed), ...directGridPresetsFromJson(parsed) }).length;
          if (styleCount || gridCount) {
            throw new Error(`this is a ${styleCount ? "style" : "grid"} preset file; use Import ${styleCount ? "Style" : "Grid"}, not Import Preset`);
          }
          throw new Error("not a FrameDesigner Preset Gallery file");
        }

        const galleryEntries = Object.entries(gallery);
        if (!galleryEntries.length) throw new Error("empty preset gallery file");

        const importedGalleryKeys = [];
        const persistedGallery = readPersistedObject(PERSISTED_PRESET_KEYS.gallery);
        const hiddenGallery = readPersistedObject(PERSISTED_PRESET_KEYS.hiddenGallery);

        galleryEntries.forEach(([key, preset], index) => {
          const finalKey = uniqueImportedGalleryKey(key, preset, index);
          IMPORTED_PRESETS[finalKey] = preset;
          persistedGallery[finalKey] = preset;
          delete hiddenGallery[finalKey];
          importedGalleryKeys.push(finalKey);
        });

        const galleryOk = writePersistedObject(PERSISTED_PRESET_KEYS.gallery, persistedGallery);
        const hiddenOk = writeHiddenGalleryPresetKeys(hiddenGallery);
        const firstImportedKey = importedGalleryKeys[0];
        state.gallerySelectedKey = firstImportedKey || state.gallerySelectedKey;
        if (firstImportedKey) {
          applyPresetToBoard(firstImportedKey, "replace");
          refreshPresetGalleryViews(firstImportedKey, { revealUser: true });
          window.setTimeout(() => refreshPresetGalleryViews(firstImportedKey, { revealUser: true }), 0);
        } else {
          refreshPresetGalleryViews("", { revealUser: true });
          window.setTimeout(() => refreshPresetGalleryViews("", { revealUser: true }), 0);
        }

        const message = `${importedType}: imported ${galleryEntries.length} preset(s) from ${file.name}. Canvas reset and applied ${presetByKey(firstImportedKey)?.label || firstImportedKey || "preset"}.`;
        footStatus.textContent = message;
        console.info("[IAMCCS FrameDesigner] import preset gallery hard replace", { file: file.name, importedGalleryKeys, appliedToBoard: firstImportedKey || null });
        showToast(`${message}${galleryOk && hiddenOk ? "" : " (session only: browser cache blocked)"}`, { tone: galleryOk && hiddenOk ? "success" : "warn", ms: 4600 });
      } catch (error) {
        const message = `Preset import failed: ${error?.message || file.name}`;
        footStatus.textContent = message;
        showToast(message, { tone: "warn", ms: 5200 });
      }
    };
    reader.readAsText(file);
  }


  function applyRuntimeIdeoboard(payload, signature, usedIncoming) {
    if (!payload) return;
    if (signature) writeInputSignature(node, signature);
    if (!usedIncoming) {
      footStatus.textContent = 'Runtime result received; canvas protected from auto-replace';
      return;
    }
    try {
      const parsed = parseJsonLoose(payload);
      const imported = isIdeogramPromptJson(parsed)
        ? { ...promptToDesign(parsed), canvas: state.data.canvas, i2i: state.data.i2i, reference_mode: state.data.reference_mode, workflow_mode: state.data.workflow_mode, preset_key: state.data.preset_key }
        : parsed;
      const next = normalizeDesignObject(imported, state.data);
      state.data = next;
      state.selectedId = state.data.items?.[0]?.id || null;
      writeData(node, state.data);
      render();
      footStatus.textContent = 'Connected ideoboard input applied as source of truth';
      showToast('Connected ideoboard JSON applied to boxes and canvas', { tone: 'success', ms: 3000 });
    } catch (error) {
      footStatus.textContent = `Connected ideoboard input could not update UI: ${error?.message || error}`;
      showToast('Connected ideoboard used by backend, but UI parse failed', { tone: 'warn', ms: 4200 });
    }
  }

  function buildGemmaFrontendRequest(mode, brief, extra = {}) {
    return {
      mode: mode || "full_json_enhance",
      brief: brief || "",
      selected_id: extra.selectedId || state.selectedId || "",
      target_field: extra.targetField || "",
      current_text: extra.currentText || "",
      speed: extra.speed || state.data.gemma_assistant?.speed || "fast",
      design_data: state.data,
      prompt_json: toPrompt(state.data),
    };
  }

  function applyGemmaFieldPatch(payload, request) {
    const field = cleanText(payload?.field_key || request?.target_field);
    const text = cleanText(payload?.text || payload?.replacement || payload?.desc || payload?.description);
    if (!field || !text) return false;
    const selectedId = cleanText(payload?.selected_id || request?.selected_id || state.selectedId);
    const item = selectedId ? state.data.items?.find((entry) => entry.id === selectedId) : currentItem();
    if (field === "scene.high_level_description") state.data.scene.high_level_description = text;
    else if (field === "scene.background") state.data.scene.background = text;
    else if (field === "scene.aesthetics") state.data.scene.aesthetics = text;
    else if (field === "scene.lighting") state.data.scene.lighting = text;
    else if (field === "scene.medium") state.data.scene.medium = text;
    else if (field === "scene.photo") state.data.scene.photo = text;
    else if (field === "scene.color_palette") state.data.scene.color_palette = paletteList(text.split(","), state.data.scene.color_palette);
    else if (field === "item.desc" && item) item.desc = text;
    else if (field === "item.text" && item) item.text = text;
    else if (field === "item.label" && item) item.label = text;
    else if (field === "item.palette" && item) item.color_palette = paletteList(text.split(","), item.color_palette);
    else return false;
    if (selectedId) state.selectedId = selectedId;
    return true;
  }

  function gemmaPendingPreview(payload, request, notes = "") {
    const rawText = cleanText(payload?.raw_text || "");
    const rawResponse = payload?.raw_response;
    const rawObject = rawResponse && typeof rawResponse === "object" ? rawResponse : null;
    const noteText = cleanText(notes || rawObject?.notes || payload?.notes || "");
    const compact = (mainText) => {
      const main = cleanText(mainText);
      return [main, noteText ? `NOTES: ${noteText}` : ""].filter(Boolean).join("\n");
    };
    if (payload?.field_patch) {
      const patch = payload.field_patch;
      return compact(patch.text || patch.replacement || patch.desc || patch.description || rawObject?.text || rawText);
    }
    if (payload?.design_data) {
      return compact(JSON.stringify(payload.design_data, null, 2));
    }
    if (rawObject) {
      return compact(rawObject.text || rawObject.replacement || rawObject.desc || rawObject.description || JSON.stringify(rawObject, null, 2));
    }
    return compact(rawText || "Gemma returned an empty proposal.");
  }

  function renderGemmaPendingResult() {
    const box = root.querySelector('[data-role="gemma-result"]');
    if (!box) return;
    const pending = state.gemmaPending;
    const area = box.querySelector('[data-role="gemma-result-text"]');
    const label = box.querySelector('[data-role="gemma-result-label"]');
    const apply = box.querySelector('[data-role="gemma-apply"]');
    const copy = box.querySelector('[data-role="gemma-copy-result"]');
    const reject = box.querySelector('[data-role="gemma-reject"]');
    if (!pending) {
      box.hidden = true;
      if (area) area.value = "";
      if (label) label.textContent = "";
      if (apply) apply.disabled = true;
      if (copy) copy.disabled = true;
      if (reject) reject.disabled = true;
      return;
    }
    box.hidden = false;
    if (label) label.textContent = pending.applicable ? "Ready to apply" : "Critic notes";
    if (area) area.value = pending.preview || "";
    if (apply) apply.disabled = !pending.applicable || state.gemmaRunning;
    if (copy) copy.disabled = false;
    if (reject) reject.disabled = false;
  }

  function setGemmaPending(payload, request, notes = "", anchor = null, statusHost = null) {
    const mode = cleanText(request?.mode || payload?.mode || "");
    const applicable = mode !== "prompt_critic" && Boolean(payload?.field_patch || payload?.design_data);
    state.gemmaPending = {
      mode,
      payload,
      request,
      notes,
      applicable,
      preview: gemmaPendingPreview(payload, request, notes),
      createdAt: Date.now(),
    };
    renderGemmaPendingResult();
    const message = applicable
      ? "Gemma proposal ready. Review it, then Apply / Reject / Copy."
      : "Gemma critic ready. Review or Copy the notes.";
    if (statusHost) statusHost.textContent = message;
    showToast(message, { anchor, tone: "success", ms: 3600 });
  }

  function clearGemmaPending(statusHost = null, message = "Gemma proposal rejected.") {
    state.gemmaPending = null;
    renderGemmaPendingResult();
    if (statusHost) statusHost.textContent = message;
  }

  function applyGemmaPending(anchor = null, statusHost = null) {
    const pending = state.gemmaPending;
    if (!pending) {
      showToast("No Gemma proposal to apply.", { anchor, tone: "warn", ms: 2400 });
      return;
    }
    if (!pending.applicable) {
      showToast("This Gemma result is critic notes only; nothing to apply.", { anchor, tone: "warn", ms: 3200 });
      return;
    }
    const payload = pending.payload || {};
    const request = pending.request || {};
    const mode = pending.mode || request.mode || "";
    let applied = false;
    if (payload.field_patch && mode !== "prompt_critic") {
      applied = applyGemmaFieldPatch(payload.field_patch, request);
      if (applied) writeInputSignature(node, `__gemma_field_${Date.now()}__`);
    } else if (payload.design_data && mode !== "prompt_critic") {
      const next = normalizeDesignObject(payload.design_data, state.data);
      state.data = next;
      state.workflowModeKey = inferWorkflowModeFromDesign(state.data, state.data.workflow_mode || state.workflowModeKey);
      state.selectedId = payload.selected_id && state.data.items?.some((item) => item.id === payload.selected_id)
        ? payload.selected_id
        : (state.data.items?.[0]?.id || null);
      writeInputSignature(node, `__gemma_${Date.now()}__`);
      applied = true;
    }
    if (!applied) {
      showToast("Gemma proposal could not be applied to this board.", { anchor, tone: "error", ms: 4200 });
      return;
    }
    state.gemmaPending = null;
    persist();
    render();
    if (statusHost) statusHost.textContent = "Gemma proposal applied.";
    showToast("Gemma proposal applied to FrameDesigner.", { anchor, tone: "success", ms: 3200 });
  }

  async function copyGemmaPending(anchor = null) {
    const pending = state.gemmaPending;
    if (!pending) {
      showToast("No Gemma proposal to copy.", { anchor, tone: "warn", ms: 2400 });
      return;
    }
    const visibleArea = root.querySelector('[data-role="gemma-result-text"]');
    const text = visibleArea?.value || pending.preview || JSON.stringify(pending.payload || {}, null, 2);
    try {
      await navigator.clipboard?.writeText(text);
      showToast("Gemma proposal copied.", { anchor, tone: "success", ms: 2400 });
    } catch (_) {
      showToast("Clipboard copy failed.", { anchor, tone: "error", ms: 3000 });
    }
  }

  function setGemmaRunning(running, statusHost = null, message = "") {
    state.gemmaRunning = Boolean(running);
    root.classList.toggle("gemma-running", state.gemmaRunning);
    root.querySelectorAll('[data-role="gemma-run"], .iamccs-isf-gemma-mini').forEach((button) => {
      button.disabled = state.gemmaRunning;
    });
    root.querySelectorAll('[data-role="gemma-stop"]').forEach((button) => {
      button.disabled = !state.gemmaRunning;
    });
    renderGemmaPendingResult();
    if (statusHost && message) statusHost.textContent = message;
  }

  async function stopGemmaAssistant(anchor = null, statusHost = null) {
    if (!state.gemmaRunning && !state.gemmaAbortController) {
      showToast("Gemma is not running.", { anchor, tone: "warn", ms: 2200 });
      return;
    }
    const activeRunId = state.gemmaRunId;
    if (activeRunId) state.gemmaCanceledRunIds.add(activeRunId);
    state.gemmaAbortController = null;
    setGemmaRunning(false, statusHost, "Gemma stop requested.");
    showToast("Gemma stop requested.", { anchor, tone: "warn", ms: 2600 });
    try {
      await api.fetchApi("/iamccs/framedesigner/gemma_abort", { method: "POST" });
    } catch (error) {
      console.warn("[IAMCCS FrameDesigner] Gemma abort request failed", error);
    }
  }

  async function runGemmaAssistant({ mode, model, brief, targetField = "", currentText = "", selectedId = "", speed = "", statusHost, anchor }) {
    if (!state.data.gemma_assistant?.enabled) {
      showToast("Enable Gemma Assistant first.", { anchor, tone: "warn", ms: 2600 });
      return;
    }
    if (state.gemmaRunning) {
      showToast("Gemma is already working. Press Stop Gemma first if you want to abort it.", { anchor, tone: "warn", ms: 3200 });
      return;
    }
    const speedMode = cleanText(speed || state.data.gemma_assistant?.speed || gemmaPanelValue("gemma-speed", "fast")) || "fast";
    state.data.gemma_assistant = {
      ...(state.data.gemma_assistant || {}),
      enabled: true,
      provider: "local_gemma",
      mode: mode || "full_json_enhance",
      speed: speedMode === "detailed" ? "detailed" : "fast",
      model: cleanText(model || state.data.gemma_assistant?.model) || "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors",
      selected_id: selectedId || state.selectedId || "",
      target_field: targetField || "",
      current_text: currentText || "",
      brief: cleanText(brief) || "",
      request_ready: false,
      updated_at: Date.now(),
    };
    persist();
    const request = buildGemmaFrontendRequest(mode, brief, { targetField, currentText, selectedId, speed: state.data.gemma_assistant.speed });
    request.model = state.data.gemma_assistant.model;
    const runId = Date.now();
    state.gemmaRunId = runId;
    state.gemmaAbortController = new AbortController();
    setGemmaRunning(true, statusHost, `Gemma is working (${state.data.gemma_assistant.speed}) with ${request.model}...`);
    showToast("Gemma assistant started. It updates text/JSON only, not Ideogram.", { anchor, tone: "success", ms: 2600 });
    try {
      const response = await api.fetchApi("/iamccs/framedesigner/gemma_assist", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(request),
        signal: state.gemmaAbortController.signal,
      });
      const payload = await response.json();
      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.error || `HTTP ${response.status}`);
      }
      if (state.gemmaCanceledRunIds.has(runId) || state.gemmaRunId !== runId) {
        state.gemmaCanceledRunIds.delete(runId);
        if (statusHost) statusHost.textContent = "Gemma result ignored after stop.";
        return;
      }
      const notes = cleanText(payload.notes || payload.report || "Gemma assistant finished.");
      setGemmaPending(payload, request, notes, anchor, statusHost);
    } catch (error) {
      if (error?.name === "AbortError") {
        const message = "Gemma assistant stopped.";
        if (statusHost) statusHost.textContent = message;
        showToast(message, { anchor, tone: "warn", ms: 2600 });
        return;
      }
      const message = `Gemma assistant failed: ${error?.message || error}`;
      if (statusHost) statusHost.textContent = message;
      showToast(message, { anchor, tone: "error", ms: 5200 });
      console.error("[IAMCCS FrameDesigner] Gemma assistant failed", error);
    } finally {
      if (state.gemmaRunId === runId) {
        state.gemmaRunId = 0;
        state.gemmaAbortController = null;
        setGemmaRunning(false);
      }
    }
  }

  function gemmaPanelValue(role, fallback = "") {
    const el = root.querySelector(`[data-role="${role}"]`);
    return el?.value ?? fallback;
  }

  async function loadGemmaModelOptions(select, statusHost = null) {
    if (!select) return;
    const current = cleanText(select.value || state.data.gemma_assistant?.model || "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors");
    select.disabled = true;
    select.innerHTML = '';
    const loading = document.createElement('option');
    loading.value = current;
    loading.textContent = current ? `Loading models... (${current})` : 'Loading models...';
    select.appendChild(loading);
    try {
      const response = await api.fetchApi("/iamccs/framedesigner/gemma_models", { method: "GET" });
      const payload = await response.json();
      if (!response.ok || !payload?.ok) throw new Error(payload?.error || `HTTP ${response.status}`);
      const models = Array.isArray(payload.models) ? payload.models.filter(Boolean) : [];
      const selected = models.includes(current) ? current : (models.includes(payload.default) ? payload.default : (models[0] || current));
      select.innerHTML = '';
      models.forEach((name) => {
        const option = document.createElement('option');
        option.value = name;
        option.textContent = name;
        select.appendChild(option);
      });
      if (!models.length && current) {
        const option = document.createElement('option');
        option.value = current;
        option.textContent = current;
        select.appendChild(option);
      }
      select.value = selected;
      state.data.gemma_assistant = {
        ...(state.data.gemma_assistant || {}),
        provider: "local_gemma",
        model: selected || current,
      };
      persist();
      if (statusHost) statusHost.textContent = models.length
        ? `Gemma model list loaded: ${models.length} text encoder(s).`
        : 'No text encoder model found by ComfyUI.';
    } catch (error) {
      select.innerHTML = '';
      const option = document.createElement('option');
      option.value = current;
      option.textContent = current || 'text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors';
      select.appendChild(option);
      select.value = option.value;
      if (statusHost) statusHost.textContent = `Gemma model list failed: ${error?.message || error}`;
      showToast(`Gemma model list failed: ${error?.message || error}`, { tone: "error", ms: 4200 });
    } finally {
      select.disabled = false;
    }
  }

  function runOrPrepareGemma({ mode = "full_json_enhance", brief = "", targetField = "", currentText = "", selectedId = "", speed = "", anchor = null, statusHost = null } = {}) {
    runGemmaAssistant({
      mode,
      model: gemmaPanelValue("gemma-model", state.data.gemma_assistant?.model || "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors"),
      brief,
      targetField,
      currentText,
      selectedId,
      speed: speed || gemmaPanelValue("gemma-speed", state.data.gemma_assistant?.speed || "fast"),
      statusHost: statusHost || root.querySelector('[data-role="gemma-status"]'),
      anchor,
    });
  }

  function updatePromptPreview() {
    if (!previewField) return;
    try {
      state.data.direct_prompt = { enabled: false, text: "" };
      state.data.json_override = normalizeJsonOverride(state.data.json_override);
      const override = state.data.json_override;
      previewField.classList.toggle("override", Boolean(override.enabled));
      previewField.readOnly = false;
      const toggle = root.querySelector('[data-role="json-override-toggle"]');
      if (toggle) toggle.checked = Boolean(override.enabled);
      if (override.enabled) {
        if (document.activeElement !== previewField) previewField.value = override.text || JSON.stringify(toPrompt({ ...state.data, json_override: { enabled: false, text: "" } }), null, 2);
        return;
      }
      previewField.value = JSON.stringify(toPrompt({ ...state.data, json_override: { enabled: false, text: "" } }), null, 2);
    } catch (error) {
      previewField.value = JSON.stringify({ error: "Prompt JSON preview failed", detail: String(error?.message || error || "") }, null, 2);
    }
  }

  function applyJsonOverrideFromPreview(anchor = null) {
    const text = cleanText(previewField?.value);
    if (!text) {
      showToast("Prompt JSON override is empty", { anchor, tone: "warn" });
      return false;
    }
    let parsed;
    try {
      parsed = parseJsonLoose(text);
    } catch (error) {
      showToast(`Invalid JSON override: ${error?.message || error}`, { anchor, tone: "error", ms: 4200 });
      return false;
    }
    if (!isIdeogramPromptJson(parsed)) {
      showToast("JSON override must be an Ideogram structured prompt with compositional_deconstruction", { anchor, tone: "error", ms: 4200 });
      return false;
    }
    const previous = state.data;
    const next = promptToDesign(parsed);
    state.data = normalizeDesignObject({
      ...next,
      preset_key: previous.preset_key,
      workflow_mode: previous.workflow_mode,
      canvas: previous.canvas,
      i2i: previous.i2i,
      reference_mode: previous.reference_mode,
      json_override: { enabled: true, text: JSON.stringify(parsed, null, 2) },
    }, previous);
    state.data.json_override = { enabled: true, text: JSON.stringify(parsed, null, 2) };
    state.selectedId = state.data.items?.[0]?.id || null;
    writeInputSignature(node, "__manual_json_override__");
    persist();
    render();
    showToast("JSON override applied as source of truth", { anchor, tone: "success" });
    return true;
  }

  function persist() {
    state.data.workflow_mode = currentWorkflowMode().key;
    state.data.mask_paint = normalizeMaskPaint(state.data.mask_paint || defaultData().mask_paint, defaultData().mask_paint);
    writeData(node, state.data);
    footStatus.textContent = `Saved ${state.data.items.length} layers to design_data`;
    updatePromptPreview();
  }

  function denoiseLabel(value) {
    const current = Number(value) || 0;
    let best = DENOISE_PRESETS[0];
    let bestDistance = Math.abs(current - best.value);
    DENOISE_PRESETS.forEach((preset) => {
      const distance = Math.abs(current - preset.value);
      if (distance < bestDistance) {
        best = preset;
        bestDistance = distance;
      }
    });
    return bestDistance <= 0.035 ? best.label : "Custom";
  }

  function syncDenoisePresetButtons() {
    const value = Number(state.data?.i2i?.denoise ?? 0.28);
    root.querySelectorAll('[data-denoise-preset]').forEach((button) => {
      const presetValue = Number(button.dataset.value);
      button.classList.toggle('selected', Math.abs(value - presetValue) <= 0.035);
      button.setAttribute('aria-pressed', Math.abs(value - presetValue) <= 0.035 ? 'true' : 'false');
    });
    const readout = root.querySelector('[data-denoise-readout]');
    if (readout) readout.textContent = `${value.toFixed(2)} ${denoiseLabel(value)}`;
  }

  function resultCompareRegion() {
    const refMode = normalizeReferenceMode(state.data.reference_mode || "single");
    if (refMode.mode === "character_diptych" || refMode.mode === "multi_ref_triptych") {
      const panels = Math.max(1, Number(refMode.panel_count) || 1);
      const target = Math.max(0, Math.min(panels - 1, Number(refMode.target_index) || panels - 1));
      return {
        left: (target / panels) * 100,
        width: 100 / panels,
        right: 100 - ((target + 1) / panels) * 100,
        label: "Target Panel",
      };
    }
    return { left: 0, width: 100, right: 0, label: "Result" };
  }

  function applyResultCompareCss() {
    const split = Math.max(0, Math.min(100, Number(state.resultOverlaySplit) || 50));
    const opacity = Math.max(0, Math.min(1, Number(state.resultOverlayOpacity) || 0));
    const region = resultCompareRegion();
    const bg = artboard.querySelector(".iamccs-isf-result-bg");
    const divider = artboard.querySelector(".iamccs-isf-result-divider");
    const compare = artboard.querySelector(".iamccs-isf-canvas-compare");
    if (bg) {
      bg.style.left = `${region.left}%`;
      bg.style.width = `${region.width}%`;
      bg.style.setProperty("--iamccs-result-opacity", String(opacity));
      bg.style.setProperty("--iamccs-result-split", `${split}%`);
    }
    if (divider) {
      divider.style.left = `${region.left + (region.width * split / 100)}%`;
    }
    if (compare) {
      compare.style.left = `calc(${region.left}% + 18px)`;
      compare.style.right = `calc(${region.right}% + 18px)`;
      const label = compare.querySelector("[data-compare-label]");
      if (label) label.textContent = region.label;
    }
  }

  function referenceWorkflowUsesStructuredJsonOnly() {
    const refMode = normalizeReferenceMode(state.data.reference_mode || "single");
    return refMode.mode === "character_diptych" || refMode.mode === "multi_ref_triptych";
  }

  function syncDirectPromptFields() {
    // Single/direct prompt has been intentionally removed from the UI and runtime.
    // Ideogram conditioning now always uses the structured JSON produced by the canvas boxes.
    state.data.direct_prompt = { enabled: false, text: "" };
    if (directPromptPanel) {
      directPromptPanel.style.display = "none";
      directPromptPanel.setAttribute("aria-hidden", "true");
    }
    if (directPromptFields.enabled) {
      directPromptFields.enabled.checked = false;
      directPromptFields.enabled.disabled = true;
    }
    if (directPromptFields.text) {
      directPromptFields.text.value = "";
      directPromptFields.text.disabled = true;
    }
  }

  function setDirectPromptFromScene() {
    state.data.direct_prompt = { enabled: false, text: "" };
    persist();
    footStatus.textContent = "Structured JSON is used for Ideogram";
  }

  function setDenoise(value, status = "Denoise updated") {
    state.data.i2i = normalizeI2I(state.data.i2i);
    state.data.i2i.enabled = true;
    state.data.i2i.denoise = Math.max(0, Math.min(1, Number(value) || 0));
    if (i2iFields.enabled) i2iFields.enabled.checked = true;
    if (i2iFields.denoise) i2iFields.denoise.value = state.data.i2i.denoise;
    setWidget(node, "i2i_enabled", true);
    setWidget(node, "i2i_denoise", Number(state.data.i2i.denoise));
    setWidget(node, "low_sigma_start_step", Number(state.data.i2i.low_sigma_start_step));
    syncDenoisePresetButtons();
    persist();
    footStatus.textContent = `${status}: denoise ${state.data.i2i.denoise.toFixed(2)} (${denoiseLabel(state.data.i2i.denoise)})`;
  }

  function applyRefinePreset() {
    state.data.i2i = normalizeI2I(state.data.i2i);
    state.data.i2i.enabled = true;
    state.data.i2i.denoise = 0.28;
    state.data.i2i.low_sigma_start_step = 12;
    state.data.i2i.source_mode = "refine_source_image";
    state.data.canvas.width = 1280;
    state.data.canvas.height = 720;
    state.data.canvas.aspect_label = "16:9 Refine 720p";
    if (i2iFields.enabled) i2iFields.enabled.checked = true;
    if (i2iFields.denoise) i2iFields.denoise.value = state.data.i2i.denoise;
    if (i2iFields.step) i2iFields.step.value = state.data.i2i.low_sigma_start_step;
    setWidget(node, "i2i_enabled", true);
    setWidget(node, "i2i_denoise", 0.28);
    setWidget(node, "low_sigma_start_step", 12);
    persist();
    render();
    footStatus.textContent = "REFINE ready: i2i on, denoise 0.28, target 1280x720";
  }

  function openFieldEditor(title, input) {
    if (!input) return;
    const modal = document.createElement('div');
    modal.className = 'iamccs-isf-zoom-modal';
    const value = input.value || '';
    modal.innerHTML = `
      <div class="iamccs-isf-zoom-card" role="dialog" aria-modal="true">
        <div class="iamccs-isf-zoom-head">
          <strong></strong>
          <button class="iamccs-isf-btn" data-zoom="close">Close</button>
        </div>
        <div class="iamccs-isf-zoom-body"><textarea spellcheck="false"></textarea></div>
        <div class="iamccs-isf-zoom-foot">
          <button class="iamccs-isf-btn" data-zoom="cancel">Cancel</button>
          <button class="iamccs-isf-btn primary" data-zoom="save">Save</button>
        </div>
      </div>`;
    modal.querySelector('strong').textContent = title || 'Editor';
    const area = modal.querySelector('textarea');
    area.value = value;
    const close = () => modal.remove();
    const save = () => {
      input.value = area.value;
      input.dispatchEvent(new Event('input', { bubbles: true }));
      input.dispatchEvent(new Event('change', { bubbles: true }));
      close();
    };
    modal.addEventListener('click', (event) => {
      const action = event.target?.dataset?.zoom;
      if (event.target === modal || action === 'close' || action === 'cancel') close();
      if (action === 'save') save();
    });
    modal.addEventListener('keydown', (event) => {
      if (event.key === 'Escape') close();
      if ((event.ctrlKey || event.metaKey) && event.key === 'Enter') save();
    });
    document.body.appendChild(modal);
    setTimeout(() => area.focus(), 0);
  }

  function makeField(container, label, key, opts = {}) {
    const wrap = document.createElement('div');
    wrap.className = `iamccs-isf-field ${opts.small ? 'small' : ''}`;
    const head = document.createElement('div');
    head.className = 'iamccs-isf-field-head';
    const lbl = document.createElement('label');
    lbl.textContent = label;
    const input = opts.multiline ? document.createElement('textarea') : document.createElement('input');
    if (!opts.multiline) {
      input.type = opts.type || 'text';
    }
    head.appendChild(lbl);
    const actions = document.createElement('div');
    actions.className = 'iamccs-isf-field-actions';
    input._iamccsFieldWrap = wrap;
    input._iamccsFieldLabel = lbl;
    if (opts.type === 'number') {
      lbl.dataset.dragNumber = 'true';
      lbl.title = 'Drag left/right to change value; press Enter or leave the field to apply typed values.';
    }
    if (opts.gemmaField) {
      const enhance = document.createElement('button');
      enhance.type = 'button';
      enhance.className = 'iamccs-isf-zoom-btn iamccs-isf-gemma-mini';
      enhance.textContent = 'G';
      enhance.title = `Gemma enhance: ${label}`;
      enhance.setAttribute('aria-label', `Gemma enhance ${label}`);
      enhance.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        runOrPrepareGemma({
          mode: "field_enhance",
          targetField: opts.gemmaField,
          currentText: input.value,
          selectedId: opts.itemScoped ? state.selectedId : "",
          brief: input.value,
          anchor: event.currentTarget,
        });
      });
      const critic = document.createElement('button');
      critic.type = 'button';
      critic.className = 'iamccs-isf-zoom-btn iamccs-isf-gemma-mini critic';
      critic.textContent = 'C';
      critic.title = `Gemma critic: ${label}`;
      critic.setAttribute('aria-label', `Gemma critic ${label}`);
      critic.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        runOrPrepareGemma({
          mode: "prompt_critic",
          targetField: opts.gemmaField,
          currentText: input.value,
          selectedId: opts.itemScoped ? state.selectedId : "",
          brief: input.value,
          anchor: event.currentTarget,
        });
      });
      actions.append(enhance, critic);
    }
    if (opts.multiline || ['high','background','aesthetics','lighting','photo','desc','text'].includes(String(key))) {
      const zoom = document.createElement('button');
      zoom.type = 'button';
      zoom.className = 'iamccs-isf-zoom-btn';
      zoom.textContent = '\u2315';
      zoom.setAttribute('aria-label', `Open large editor for ${label}`);
      zoom.title = `Open large editor for ${label}`;
      zoom.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        openFieldEditor(label, input);
      });
      actions.appendChild(zoom);
    }
    if (actions.childElementCount) head.appendChild(actions);
    wrap.append(head, input);
    container.appendChild(wrap);
    return input;
  }

  function enableNumberDrag(input, commit, opts = {}) {
    const label = input?._iamccsFieldLabel;
    if (!input || !label) return;
    const step = Number(opts.step || input.step || 1) || 1;
    const min = Number.isFinite(Number(opts.min)) ? Number(opts.min) : -Infinity;
    const max = Number.isFinite(Number(opts.max)) ? Number(opts.max) : Infinity;
    let drag = null;
    const clampValue = (value) => Math.max(min, Math.min(max, Number(value) || 0));
    const start = (event) => {
      event.preventDefault();
      drag = { x: event.clientX, start: Number(input.value || 0), pointerId: event.pointerId };
      label.setPointerCapture?.(event.pointerId);
    };
    const move = (event) => {
      if (!drag) return;
      const delta = event.clientX - drag.x;
      const value = clampValue(drag.start + Math.round(delta / 6) * step);
      input.value = String(Math.round(value));
      commit(value, true);
    };
    const end = (event) => {
      if (!drag) return;
      label.releasePointerCapture?.(drag.pointerId);
      drag = null;
      commit(input.value, true);
    };
    label.addEventListener('pointerdown', start);
    label.addEventListener('pointermove', move);
    label.addEventListener('pointerup', end);
    label.addEventListener('pointercancel', end);
  }


  function selectedGalleryKey() {
    const entries = presetEntries();
    if (!hasPresetKey(state.gallerySelectedKey)) state.gallerySelectedKey = Object.keys(entries)[0] || "storyboard";
    return state.gallerySelectedKey;
  }

  function currentWorkflowMode() {
    const modes = workflowModeEntries();
    const refMode = normalizeReferenceMode(state.data.reference_mode || "single");
    if (refMode.mode === "character_diptych") state.workflowModeKey = "character_diptych";
    else if (refMode.mode === "multi_ref_triptych") state.workflowModeKey = "multi_ref_triptych";
    else if (!modes[state.workflowModeKey]) state.workflowModeKey = inferWorkflowModeKey();
    state.workflowModeKey = modes[state.workflowModeKey] ? state.workflowModeKey : "single_image";
    state.data.workflow_mode = state.workflowModeKey;
    return { key: state.workflowModeKey, ...(modes[state.workflowModeKey] || modes.single_image) };
  }

  function inferWorkflowModeKey() {
    const refMode = normalizeReferenceMode(state.data.reference_mode || "single");
    if (refMode.mode === "character_diptych") return "character_diptych";
    if (refMode.mode === "multi_ref_triptych") return "multi_ref_triptych";
    if (state.data.i2i?.enabled && state.data.i2i?.source_mode === "refine_source_image") return "image_refine";
    const grid = gridEntries()[state.gridSelectedKey] || null;
    if (grid?.boxes && state.gridSelectedKey !== "single_frame_1x1" && state.gridSelectedKey !== "free_canvas") return "storyboard_grid";
    return "single_image";
  }

  function currentCanvasMultiplier() {
    const mode = currentWorkflowMode();
    if (mode.grid) {
      const grid = gridEntries()[state.gridSelectedKey] || gridEntries().story_2x3;
      if (grid?.boxes) return { columns: Math.max(1, Number(grid.boxes[0]) || 1), rows: Math.max(1, Number(grid.boxes[1]) || 1), label: grid.label || "Grid" };
    }
    return { columns: Math.max(1, Number(mode.columns) || 1), rows: Math.max(1, Number(mode.rows) || 1), label: mode.label || "Single" };
  }

  function refitFullCanvasImageLayers() {
    (state.data.items || []).forEach((item) => {
      if (item.kind !== "image") return;
      const nearFull = Number(item.w || 0) >= 820 || Number(item.h || 0) >= 820;
      if (!nearFull) return;
      item.x = 0;
      item.y = 0;
      item.w = 1000;
      item.h = 1000;
    });
  }

  function mergeLayoutItems(nextItems, previousItems = []) {
    const previous = Array.isArray(previousItems) ? previousItems : [];
    return nextItems.map((next, index) => {
      const old = previous.find((item) => item?.id === next.id)
        || previous.find((item) => item?.label === next.label)
        || previous[index];
      if (!old) return next;
      const merged = { ...next };
      if (cleanText(old.desc)) merged.desc = old.desc;
      if (cleanText(old.text)) merged.text = old.text;
      if (Array.isArray(old.color_palette) && old.color_palette.length) merged.color_palette = old.color_palette;
      if (next.kind === "image" || old.kind === "image") {
        merged.image_path = cleanText(old.image_path) || cleanText(next.image_path);
        merged.fit = ["cover", "contain", "stretch"].includes(String(old.fit || "").toLowerCase()) ? String(old.fit).toLowerCase() : (next.fit || "contain");
        merged.opacity = Math.max(0, Math.min(1, Number(old.opacity ?? next.opacity ?? 1) || 1));
      }
      return normalizeItem(merged, index);
    });
  }

  function rebuildLayoutForWorkflow(options = {}) {
    const mode = currentWorkflowMode();
    const refMode = normalizeReferenceMode(state.data.reference_mode || "single");
    const previousItems = state.data.items || [];
    if (refMode.mode === "character_diptych" || refMode.mode === "multi_ref_triptych") {
      if (options.rebuildLayout !== false) {
        state.data.items = mergeLayoutItems(makeReferenceItems(refMode.mode, state.data.scene.color_palette || []), previousItems);
        state.selectedId = state.data.items[0]?.id || null;
      }
      return;
    }
    if (mode.grid) {
      const grid = gridEntries()[state.gridSelectedKey] || gridEntries().story_2x3;
      if (grid?.boxes && options.rebuildGrid !== false) {
        state.data.items = mergeLayoutItems(makeGridItems(grid.boxes[0], grid.boxes[1], state.data.scene.color_palette || []), previousItems);
        state.selectedId = state.data.items[0]?.id || null;
      }
      return;
    }
    refitFullCanvasImageLayers();
  }

  function applyTargetResolutionPreset(key, options = {}) {
    const entries = targetResolutionEntries();
    const cleanKey = entries[key] ? key : "hd_720";
    const preset = entries[cleanKey] || entries.hd_720;
    state.targetResolutionKey = cleanKey;
    state.data.canvas.target_resolution_key = cleanKey;
    if (sceneFields.targetResolution && sceneFields.targetResolution.value !== cleanKey) {
      sceneFields.targetResolution.value = cleanKey;
    }
    if (!preset || cleanKey === "custom" || Number(preset.width) <= 0 || Number(preset.height) <= 0) {
      state.data.canvas.target_width = Number(state.data.canvas.width || 1536);
      state.data.canvas.target_height = Number(state.data.canvas.height || 864);
      state.data.canvas.aspect_label = state.data.canvas.aspect_label || `Custom canvas ${state.data.canvas.width}x${state.data.canvas.height}`;
      persist();
      render();
      footStatus.textContent = "Custom target resolution: edit Width and Height manually";
      return;
    }
    const mult = currentCanvasMultiplier();
    const targetW = Math.max(1, Math.round(Number(preset.width)));
    const targetH = Math.max(1, Math.round(Number(preset.height)));
    const nextW = Math.max(256, Math.round(targetW * mult.columns));
    const nextH = Math.max(256, Math.round(targetH * mult.rows));
    state.data.canvas.width = nextW;
    state.data.canvas.height = nextH;
    state.data.canvas.target_width = targetW;
    state.data.canvas.target_height = targetH;
    state.data.canvas.aspect_label = `${preset.label} target / ${mult.label} canvas ${nextW}x${nextH}`;
    rebuildLayoutForWorkflow(options);
    persist();
    render();
    const message = `Target ${preset.label}: canvas ${nextW}x${nextH}`;
    footStatus.textContent = message;
    showToast(message, { tone: "success", ms: 1600 });
  }

  function clearReferenceModeForStoryboard(rebuildItems = true) {
    const refMode = normalizeReferenceMode(state.data.reference_mode || "single");
    state.data.reference_mode = normalizeReferenceMode({ mode: "single" });
    state.data.i2i = normalizeI2I({
      ...state.data.i2i,
      enabled: false,
      source_mode: "canvas_composite",
    });
    state.resultBgUrl = "";
    state.resultBgFinal = false;
    state.resultOverlayVisible = true;
    setWidget(node, "i2i_enabled", false);
    setWidget(node, "i2i_denoise", Number(state.data.i2i.denoise));
    setWidget(node, "low_sigma_start_step", Number(state.data.i2i.low_sigma_start_step));
    if (!rebuildItems) return refMode.mode !== "single";
    const grid = gridEntries()[state.gridSelectedKey] || gridEntries().story_2x3;
    if (grid) {
      state.data.canvas.width = Number(grid.width || state.data.canvas.width || 1536);
      state.data.canvas.height = Number(grid.height || state.data.canvas.height || 864);
      state.data.canvas.aspect_label = grid.aspect_label || grid.label || "Single / Storyboard";
    }
    if (grid?.boxes) {
      state.data.items = makeGridItems(grid.boxes[0], grid.boxes[1], state.data.scene.color_palette || []);
      state.selectedId = state.data.items[0]?.id || null;
    } else if (!Array.isArray(state.data.items) || !state.data.items.length || refMode.mode !== "single") {
      state.data.items = cloneValue(defaultData().items, []).map(normalizeItem);
      state.selectedId = state.data.items[0]?.id || null;
    }
    return refMode.mode !== "single";
  }

  function applyPresetRuntimeMetadata(key, applied) {
    const raw = presetByKey(key) || {};
    const modes = workflowModeEntries();
    const grids = gridEntries();
    const targets = targetResolutionEntries();
    const requestedMode = cleanText(raw.workflow_mode || applied.workflow_mode || inferWorkflowModeFromDesign(applied, state.workflowModeKey));
    const modeKey = modes[requestedMode] ? requestedMode : inferWorkflowModeFromDesign(applied, "storyboard_grid");
    state.workflowModeKey = modes[modeKey] ? modeKey : "storyboard_grid";
    state.data.workflow_mode = state.workflowModeKey;
    const requestedGrid = cleanText(raw.grid_key || applied.grid_key || raw.grid_preset || applied.grid_preset);
    if (state.workflowModeKey === "storyboard_grid") {
      state.gridSelectedKey = grids[requestedGrid] ? requestedGrid : "story_2x3";
    } else {
      state.gridSelectedKey = grids[requestedGrid] ? requestedGrid : "free_canvas";
    }
    const requestedTarget = cleanText(raw.target_resolution_key || applied.target_resolution_key || raw.canvas?.target_resolution_key || applied.canvas?.target_resolution_key);
    state.targetResolutionKey = targets[requestedTarget] ? requestedTarget : (state.workflowModeKey === "storyboard_grid" ? "hd_720" : "custom");
    state.data.grid_key = state.gridSelectedKey;
    state.data.target_resolution_key = state.targetResolutionKey;
    state.data.canvas.target_resolution_key = state.targetResolutionKey;
    const target = targets[state.targetResolutionKey];
    const grid = grids[state.gridSelectedKey];
    if (target && state.targetResolutionKey !== "custom" && Number(target.width) > 0 && Number(target.height) > 0) {
      const cols = state.workflowModeKey === "storyboard_grid" && grid?.boxes ? Math.max(1, Number(grid.boxes[0]) || 1) : 1;
      const rows = state.workflowModeKey === "storyboard_grid" && grid?.boxes ? Math.max(1, Number(grid.boxes[1]) || 1) : 1;
      state.data.canvas.target_width = Number(target.width);
      state.data.canvas.target_height = Number(target.height);
      state.data.canvas.width = Number(target.width) * cols;
      state.data.canvas.height = Number(target.height) * rows;
      state.data.canvas.aspect_label = state.workflowModeKey === "storyboard_grid"
        ? `${grid?.label || "Grid"} / ${target.label || `${target.width}x${target.height}`} panels`
        : `${target.label || `${target.width}x${target.height}`}`;
    } else {
      state.targetResolutionKey = "custom";
      state.data.target_resolution_key = "custom";
      state.data.canvas.target_resolution_key = "custom";
      state.data.canvas.target_width = Number(state.data.canvas.width || raw.canvas?.width || 1536);
      state.data.canvas.target_height = Number(state.data.canvas.height || raw.canvas?.height || 864);
    }
  }

  function hardResetDesignForPreset(applied, key) {
    const cleanBase = defaultData();
    cleanBase.items = [];
    cleanBase.mask_paint = normalizeMaskPaint({ brush_size: 48, strokes: [] });
    cleanBase.json_override = normalizeJsonOverride({ enabled: false, text: "" });
    cleanBase.direct_prompt = normalizeDirectPrompt({ enabled: false, text: "" }, cleanBase.scene, []);
    cleanBase.i2i = normalizeI2I(cleanBase.i2i);
    const normalized = normalizeDesignObject({
      ...cloneValue(applied, {}),
      preset_key: hasPresetKey(key) ? key : cleanText(applied?.preset_key || "storyboard"),
      items: Array.isArray(applied?.items) ? applied.items : [],
      mask_paint: { brush_size: 48, strokes: [] },
      json_override: { enabled: false, text: "" },
      direct_prompt: { enabled: false, text: "" },
    }, cleanBase);
    normalized.items = Array.isArray(applied?.items) ? applied.items.map((item, index) => normalizeItem({ ...item, id: "" }, index)) : [];
    normalized.mask_paint = normalizeMaskPaint({ brush_size: 48, strokes: [] });
    normalized.json_override = normalizeJsonOverride({ enabled: false, text: "" });
    normalized.direct_prompt = normalizeDirectPrompt(normalized.direct_prompt, normalized.scene, normalized.items);
    return normalized;
  }

  function compactPresetGridItems(items, key) {
    const preset = presetByKey(key) || {};
    const grid = gridEntries()[preset.grid_key || state.gridSelectedKey || ""] || null;
    const expected = grid?.boxes ? Math.max(1, Number(grid.boxes[0]) || 1) * Math.max(1, Number(grid.boxes[1]) || 1) : 0;
    const list = Array.isArray(items) ? items.map(normalizeItem) : [];
    if (!expected || list.length <= expected) return list;
    const panelLike = (item) => /^panel[\s_-]*\d+/i.test(cleanText(item?.label || item?.id || item?.desc));
    const panelCount = list.filter(panelLike).length;
    if (panelCount < expected * 2) return list;
    const compacted = list.slice(-expected).map((item, index) => normalizeItem({ ...item, id: "" }, index));
    console.info("[IAMCCS FrameDesigner] compacted duplicate preset grid items", {
      build: JS_BUILD,
      preset: key,
      before: list.length,
      after: compacted.length,
      expected,
    });
    return compacted;
  }

  function applyPresetToBoard(key, mode = "replace") {
    if (mode === "append") {
      console.info("[IAMCCS FrameDesigner] append preset request converted to atomic replace", { build: JS_BUILD, preset: key });
      mode = "replace";
    }
    const rawPreset = presetByKey(key);
    const applied = rawPreset ? normalizeDesignObject({
      ...cloneValue(rawPreset, {}),
      preset_key: hasPresetKey(key) ? key : "storyboard",
    }, { ...defaultData(), items: [] }) : presetData(key);
    state.selectedId = null;
    state.data = defaultData();
    state.data.items = [];
    writeData(node, state.data);
    state.data = hardResetDesignForPreset(applied, key);
    state.data.__iamccs_user_applied_preset = true;
    applyPresetRuntimeMetadata(key, state.data);
    state.data.items = compactPresetGridItems(state.data.items, key);
    state.selectedId = state.data.items?.[0]?.id || null;
    console.info("[IAMCCS FrameDesigner] atomic preset replace", {
      build: JS_BUILD,
      preset: key,
      items: state.data.items?.length || 0,
      first: state.data.items?.[0] ? { x: state.data.items[0].x, y: state.data.items[0].y, w: state.data.items[0].w, h: state.data.items[0].h, label: state.data.items[0].label } : null,
    });
    persist();
    render();
    footStatus.textContent = `Applied preset gallery cleanly: ${presetByKey(key).label}`;
  }

  function applyWorkflowMode(modeKey, options = {}) {
    const modes = workflowModeEntries();
    const mode = modes[modeKey] || modes.storyboard_grid;
    state.workflowModeKey = modes[modeKey] ? modeKey : "storyboard_grid";
    state.data.workflow_mode = state.workflowModeKey;
    state.styleSelectedKey = "default_photoreal_cinema";
    if (state.workflowModeKey === "storyboard_grid") state.gridSelectedKey = state.gridSelectedKey && gridEntries()[state.gridSelectedKey] && state.gridSelectedKey !== "free_canvas" ? state.gridSelectedKey : "story_2x3";
    else state.gridSelectedKey = "free_canvas";
    state.data.preset_key = "storyboard";
    state.data.scene = workflowDefaultScene(state.workflowModeKey);
    state.data.reference_mode = normalizeReferenceMode({ mode: mode.ref_mode || "single" });
    state.data.direct_prompt = { enabled: false, text: "" };
    state.data.i2i = normalizeI2I({
      ...state.data.i2i,
      enabled: Boolean(mode.i2i),
      denoise: state.workflowModeKey === "image_refine" ? 0.38 : (state.workflowModeKey.includes("triptych") || state.workflowModeKey.includes("diptych") ? 0.85 : 0.28),
      source_mode: mode.source_mode || "canvas_composite",
    });
    if (state.workflowModeKey === "single_image") {
      state.data.items = [];
      state.selectedId = null;
    } else if (state.workflowModeKey === "image_refine") {
      state.data.items = [normalizeItem({
        id: "source_image_full_canvas",
        kind: "image",
        label: "SOURCE IMAGE - full canvas",
        x: 0,
        y: 0,
        w: 1000,
        h: 1000,
        desc: "Full source image used as the img2img guide for composition, pose, camera angle, and scene layout.",
        image_path: "",
        fit: "stretch",
        opacity: 1,
        color_palette: state.data.scene.color_palette,
      }, 0)];
      state.selectedId = state.data.items[0]?.id || null;
    }
    setWidget(node, "i2i_enabled", Boolean(state.data.i2i?.enabled));
    setWidget(node, "i2i_denoise", Number(state.data.i2i?.denoise ?? 0.28));
    setWidget(node, "low_sigma_start_step", Number(state.data.i2i?.low_sigma_start_step ?? 12));
    applyTargetResolutionPreset(state.targetResolutionKey || "hd_720", {
      rebuildGrid: options.rebuildGrid !== false,
      rebuildLayout: options.rebuildLayout !== false,
    });
    footStatus.textContent = `Workflow mode set to ${mode.label}: defaults reset, canvas ${state.data.canvas.width}x${state.data.canvas.height}`;
  }

  function applyReferenceMode(mode) {
    if (mode === "character_diptych") return applyWorkflowMode("character_diptych");
    if (mode === "multi_ref_triptych") return applyWorkflowMode("multi_ref_triptych");
    return applyWorkflowMode("storyboard_grid");
  }

  function renderPresetGallery(container) {
    container.innerHTML = "";
    Object.entries(presetEntries()).forEach(([key, preset]) => {
      const design = presetData(key);
      const imported = Boolean(IMPORTED_PRESETS[key]);
      const selected = selectedGalleryKey() === key;
      const card = document.createElement("button");
      card.type = "button";
      card.className = `iamccs-isf-gallery-card ${selected ? "selected" : ""} ${imported ? "imported" : "builtin"}`;
      card.title = `${preset.label || key}\nClick to select. Double-click to replace the board.`;
      card.innerHTML = `
        <div class="iamccs-isf-gallery-thumb"><img alt="" src="${boardPreviewSvgDataUri(design)}"></div>
        <div class="iamccs-isf-gallery-badges"><span class="iamccs-isf-gallery-badge ${imported ? "user" : ""}">${imported ? "Imported" : "Built-in"}</span><span class="iamccs-isf-gallery-badge">${design.workflow_mode || preset.workflow_mode || "storyboard_grid"}</span></div>
        <div class="iamccs-isf-gallery-name"></div>
        <div class="iamccs-isf-gallery-summary"></div>`;
      card.querySelector(".iamccs-isf-gallery-name").textContent = preset.label || key;
      card.querySelector(".iamccs-isf-gallery-summary").textContent = preset.summary || design.scene?.high_level_description || "";
      card.addEventListener("click", () => {
        state.gallerySelectedKey = key;
        container.querySelectorAll(".iamccs-isf-gallery-card").forEach((entry) => {
          entry.classList.toggle("selected", entry === card);
        });
        footStatus.textContent = `Selected preset: ${preset.label || key}. Double-click or press Replace Board to apply.`;
      });
      card.addEventListener("dblclick", (event) => {
        event.preventDefault();
        state.gallerySelectedKey = key;
        applyPresetToBoard(key, "replace");
        showToast(`Preset applied: ${preset.label || key}`, { anchor: card, tone: "success" });
      });
      container.appendChild(card);
    });
  }




  function galleryPresetTags(key, preset, design) {
    const text = [key, preset?.label, preset?.summary, preset?.grid_key, preset?.workflow_mode, design?.board_name, design?.scene?.high_level_description].join(" ").toLowerCase();
    const tags = [];
    if (IMPORTED_PRESETS[key]) tags.push("User");
    else tags.push("Built-in");
    if (text.includes("storyboard") || preset?.workflow_mode === "storyboard_grid" || preset?.grid_key) tags.push("Storyboard");
    if (text.includes("character") || text.includes("infogram") || text.includes("profile")) tags.push("Character");
    if (preset?.workflow_mode === "single_image" || text.includes("single")) tags.push("Single");
    if (preset?.workflow_mode === "image_refine" || design?.i2i?.enabled) tags.push("i2i");
    const grid = cleanText(preset?.grid_key || design?.canvas?.grid_key);
    if (grid) tags.push(grid.replace(/_/g, " "));
    const boxes = currentCanvasMultiplierFromDesign(design);
    if (boxes) tags.push(boxes);
    return [...new Set(tags.filter(Boolean))];
  }

  function currentCanvasMultiplierFromDesign(design) {
    const items = Array.isArray(design?.items) ? design.items.filter((item) => normalizeItem(item).kind !== "image") : [];
    if (!items.length) return "";
    const xs = [...new Set(items.map((item) => Number(normalizeItem(item).x || 0).toFixed(3)))];
    const ys = [...new Set(items.map((item) => Number(normalizeItem(item).y || 0).toFixed(3)))];
    if (xs.length > 1 || ys.length > 1) return String(xs.length) + "x" + String(ys.length);
    return "1x1";
  }

  function galleryPresetRows() {
    const rank = { User: 0, Storyboard: 1, Character: 2, Single: 3, i2i: 4, "Built-in": 9 };
    return Object.entries(presetEntries()).map(([key, preset]) => {
      const design = presetData(key);
      const tags = galleryPresetTags(key, preset, design);
      return {
        key,
        preset,
        design,
        tags,
        imported: Boolean(IMPORTED_PRESETS[key]),
        primaryRank: Math.min(...tags.map((tag) => rank[tag] ?? 6)),
      };
    }).sort((a, b) => a.primaryRank - b.primaryRank || String(a.preset.label || a.key).localeCompare(String(b.preset.label || b.key)));
  }

  function deleteUserGalleryPreset(key, anchor = null, options = {}) {
    const cleanKey = cleanText(key);
    if (!cleanKey) return false;
    if (!IMPORTED_PRESETS[cleanKey]) {
      showToast("Only imported user gallery presets can be deleted here.", { anchor, tone: "warn", ms: 3000 });
      return false;
    }
    delete IMPORTED_PRESETS[cleanKey];
    const persisted = readPersistedObject(PERSISTED_PRESET_KEYS.gallery);
    delete persisted[cleanKey];
    const hidden = hiddenGalleryPresetKeys();
    delete hidden[cleanKey];
    const okA = writePersistedObject(PERSISTED_PRESET_KEYS.gallery, persisted);
    const okB = writeHiddenGalleryPresetKeys(hidden);
    const visible = Object.keys(presetEntries());
    state.gallerySelectedKey = visible.includes(state.gallerySelectedKey) ? state.gallerySelectedKey : (visible[0] || "storyboard");
    if (options.renderNode !== false) render();
    else {
      const gallery = root.querySelector('[data-preset-gallery]');
      if (gallery) renderPresetGallery(gallery);
    }
    showToast("Deleted user preset: " + cleanKey + (okA && okB ? "" : " (session only: browser cache blocked)"), { anchor, tone: okA && okB ? "success" : "warn" });
    return true;
  }

  let activePresetGalleryManager = null;

  function refreshOpenPresetGalleryManager() {
    try { activePresetGalleryManager?.render?.(); } catch (error) { console.warn("[IAMCCS FrameDesigner] gallery manager refresh failed", error); }
  }

  function refreshPresetGalleryViews(selectedKey = "", options = {}) {
    const key = cleanText(selectedKey);
    if (key && hasPresetKey(key)) state.gallerySelectedKey = key;
    const gallery = root.querySelector('[data-preset-gallery]');
    if (gallery) renderPresetGallery(gallery);
    refreshPresetDropdowns();
    try {
      if (options.revealUser && activePresetGalleryManager?.showUser) activePresetGalleryManager.showUser();
      else refreshOpenPresetGalleryManager();
    } catch (error) {
      console.warn("[IAMCCS FrameDesigner] gallery live refresh failed", error);
    }
  }

  function openPresetGalleryManager(anchor = null) {
    const modal = document.createElement("div");
    modal.className = "iamccs-isf-zoom-modal";
    let activeTag = "All";

    const allTags = () => {
      const tags = new Set(["All"]);
      galleryPresetRows().forEach((row) => row.tags.forEach((tag) => tags.add(tag)));
      const order = ["All", "User", "Storyboard", "Character", "Single", "i2i", "Built-in"];
      return [...tags].sort((a, b) => {
        const ai = order.indexOf(a);
        const bi = order.indexOf(b);
        return (ai < 0 ? 99 : ai) - (bi < 0 ? 99 : bi) || a.localeCompare(b);
      });
    };

    const renderManager = () => {
      const rows = galleryPresetRows();
      const filtered = activeTag === "All" ? rows : rows.filter((row) => row.tags.includes(activeTag));
      modal.innerHTML = [
        '<div class="iamccs-isf-gallery-manager-card" role="dialog" aria-modal="true">',
        '<div class="iamccs-isf-gallery-manager-head">',
        '<div><strong>Preset Gallery Manager</strong><span>Browse imported and built-in presets by tag. Replace always clears the current board before loading the selected preset.</span></div>',
        '<div style="display:flex; gap:8px; align-items:center;"><button class="iamccs-isf-btn primary" type="button" data-manager-action="import">Import Preset</button><button class="iamccs-isf-btn" type="button" data-manager-action="close">Close</button></div>',
        '</div>',
        '<div class="iamccs-isf-gallery-manager-tags" data-manager-tags></div>',
        '<div class="iamccs-isf-gallery-manager-grid" data-manager-grid></div>',
        '<div class="iamccs-isf-gallery-manager-foot">' + filtered.length + ' preset(s) shown. Double-click any card to replace the current board.</div>',
        '</div>'
      ].join("");

      const tagBar = modal.querySelector("[data-manager-tags]");
      allTags().forEach((tag) => {
        const button = document.createElement("button");
        button.type = "button";
        button.className = "iamccs-isf-gallery-manager-tag " + (activeTag === tag ? "selected" : "");
        button.textContent = tag;
        button.addEventListener("click", () => {
          activeTag = tag;
          renderManager();
        });
        tagBar.appendChild(button);
      });

      const grid = modal.querySelector("[data-manager-grid]");
      if (!filtered.length) {
        const empty = document.createElement("div");
        empty.className = "iamccs-isf-gallery-manager-empty";
        empty.textContent = "No presets for this tag.";
        grid.appendChild(empty);
      }
      filtered.forEach((row) => {
        const card = document.createElement("div");
        card.className = "iamccs-isf-gallery-manager-item " + (row.imported ? "user" : "builtin") + (state.gallerySelectedKey === row.key ? " selected" : "");
        card.innerHTML = [
          '<div class="iamccs-isf-gallery-manager-thumb"><img alt=""></div>',
          '<div class="iamccs-isf-gallery-manager-title"></div>',
          '<div class="iamccs-isf-gallery-manager-meta">',
          '<div data-role="summary"></div>',
          '<div><strong>Key:</strong> <span data-role="key"></span></div>',
          '<div><strong>Mode:</strong> <span data-role="mode"></span></div>',
          '<div class="iamccs-isf-gallery-manager-chipline" data-role="tags"></div>',
          '</div>',
          '<div class="iamccs-isf-gallery-manager-actions two">',
          '<button class="iamccs-isf-btn primary" type="button" data-card-action="replace">Replace</button>',
          '<button class="iamccs-isf-btn danger-lite" type="button" data-card-action="delete">Delete</button>',
          '</div>'
        ].join("");
        card.querySelector("img").src = boardPreviewSvgDataUri(row.design);
        card.querySelector(".iamccs-isf-gallery-manager-title").textContent = row.preset.label || row.key;
        card.querySelector('[data-role="summary"]').textContent = row.preset.summary || row.design.scene?.high_level_description || "";
        card.querySelector('[data-role="key"]').textContent = row.key;
        card.querySelector('[data-role="mode"]').textContent = row.preset.workflow_mode || row.design.workflow_mode || "storyboard_grid";
        const chipline = card.querySelector('[data-role="tags"]');
        row.tags.forEach((tag) => {
          const chip = document.createElement("span");
          chip.className = "iamccs-isf-gallery-manager-chip " + (tag === "User" ? "user" : "");
          chip.textContent = tag;
          chipline.appendChild(chip);
        });
        card.addEventListener("dblclick", () => {
          state.gallerySelectedKey = row.key;
          applyPresetToBoard(row.key, "replace");
          modal.remove();
        });
        card.querySelector('[data-card-action="replace"]')?.addEventListener("click", () => {
          state.gallerySelectedKey = row.key;
          applyPresetToBoard(row.key, "replace");
          modal.remove();
        });
        const deleteButton = card.querySelector('[data-card-action="delete"]');
        if (!row.imported) {
          deleteButton.disabled = true;
          deleteButton.textContent = "Built-in";
          deleteButton.title = "Built-in presets are locked. Import a user preset pack to manage deletions here.";
        } else {
          deleteButton.addEventListener("click", (event) => {
            event.preventDefault();
            event.stopPropagation();
            if (deleteUserGalleryPreset(row.key, deleteButton, { renderNode: false })) renderManager();
          });
        }
        grid.appendChild(card);
      });

      modal.querySelector('[data-manager-action="import"]')?.addEventListener("click", () => presetGalleryInput?.click());
      modal.querySelector('[data-manager-action="close"]')?.addEventListener("click", () => modal.dispatchEvent(new Event("iamccs-close-manager")));
    };

    activePresetGalleryManager = {
      render: renderManager,
      modal,
      showUser: () => {
        activeTag = "User";
        renderManager();
      },
    };
    renderManager();
    const closeManager = () => {
      if (activePresetGalleryManager?.modal === modal) activePresetGalleryManager = null;
      modal.remove();
    };
    modal.addEventListener("click", (event) => {
      if (event.target === modal) closeManager();
    });
    modal.addEventListener("keydown", (event) => {
      if (event.key === "Escape") closeManager();
    });
    modal.addEventListener("iamccs-close-manager", closeManager);
    document.body.appendChild(modal);
    showToast("Preset Gallery Manager opened", { anchor, tone: "success", ms: 1400 });
  }


  function presetKeyFromLabel(label, prefix) {
    const base = cleanText(label)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "")
      .slice(0, 42);
    return `${prefix}_${base || Date.now()}_${Date.now().toString(36)}`;
  }

  function saveCurrentStylePreset(anchor = null) {
    const scene = state.data.scene || {};
    const labelBase = cleanText(scene.high_level_description).split(/[.,;:]/)[0] || "Custom Style";
    const key = presetKeyFromLabel(labelBase, "style");
    const preset = normalizeStylePreset({
      key,
      label: labelBase,
      summary: "Saved from current FrameDesigner scene/style boxes.",
      high: scene.high_level_description,
      background: scene.background,
      aesthetics: scene.aesthetics,
      lighting: scene.lighting,
      photo: scene.photo,
      medium: scene.medium,
      palette: scene.color_palette,
    }, key);
    IMPORTED_STYLE_PRESETS[key] = preset;
    const persisted = readPersistedObject(PERSISTED_PRESET_KEYS.styles);
    persisted[key] = preset;
    const persistedOk = writePersistedObject(PERSISTED_PRESET_KEYS.styles, persisted);
    state.styleSelectedKey = key;
    downloadJsonFile(`${key}.style-preset.json`, {
      schema: "iamccs.frame_v2.style_presets",
      presets: { [key]: preset },
    });
    render();
    showToast(`Saved style preset: ${preset.label}${persistedOk ? "" : " (downloaded; browser cache blocked)"}`, { anchor, tone: persistedOk ? "success" : "warn" });
  }

  function inferGridBoxesFromItems() {
    const items = (state.data.items || []).filter((item) => item.kind !== "image" || String(item.id || "").startsWith("panel_"));
    const xs = new Set();
    const ys = new Set();
    items.forEach((item) => {
      xs.add(Math.round(Number(item.x || 0)));
      ys.add(Math.round(Number(item.y || 0)));
    });
    const cols = Math.max(1, xs.size || currentCanvasMultiplier().columns || 1);
    const rows = Math.max(1, ys.size || currentCanvasMultiplier().rows || 1);
    return [cols, rows];
  }

  function saveCurrentGridPreset(anchor = null) {
    const boxes = inferGridBoxesFromItems();
    const label = `${boxes[0]}x${boxes[1]} ${state.data.canvas?.aspect_label || "Custom Grid"}`.replace(/\s+/g, " ").trim();
    const key = presetKeyFromLabel(label, "grid");
    const items = (state.data.items || []).map((item, index) => normalizeItem(item, index));
    const preset = normalizeGridPreset({
      key,
      label,
      summary: "Saved from current FrameDesigner grid, including panel text boxes.",
      width: state.data.canvas?.width || 1536,
      height: state.data.canvas?.height || 864,
      aspect_label: state.data.canvas?.aspect_label || label,
      boxes,
      order: "row_major",
      items,
    }, key);
    IMPORTED_GRID_PRESETS[key] = preset;
    state.showSheetSpecs = true;
    const persisted = readPersistedObject(PERSISTED_PRESET_KEYS.grids);
    persisted[key] = preset;
    const persistedOk = writePersistedObject(PERSISTED_PRESET_KEYS.grids, persisted);
    state.gridSelectedKey = key;
    downloadJsonFile(`${key}.grid-preset.json`, {
      schema: "iamccs.frame_v2.grid_presets",
      presets: { [key]: preset },
    });
    render();
    showToast(`Saved grid preset: ${preset.label}${persistedOk ? "" : " (downloaded; browser cache blocked)"}`, { anchor, tone: persistedOk ? "success" : "warn" });
  }

  function renderSelectOptions(select, entries, selectedKey) {
    if (!select) return;
    const current = select.value;
    const normalizedEntries = entries || {};
    select.innerHTML = "";
    Object.entries(normalizedEntries).forEach(([key, preset]) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = preset.label || key;
      select.appendChild(option);
    });
    const wanted = normalizedEntries[selectedKey] ? selectedKey : (normalizedEntries[current] ? current : Object.keys(normalizedEntries)[0] || "");
    select.value = wanted;
  }

  function syncSheetSpecsVisibility() {
    const sheetPanel = root.querySelector('[data-role="sheetbuilder-specs-panel"]');
    const toggle = root.querySelector('[data-role="toggle-sheet-specs"]');
    if (sheetPanel) sheetPanel.classList.toggle('collapsed', !state.showSheetSpecs);
    if (toggle) toggle.checked = Boolean(state.showSheetSpecs);
  }

  function refreshPresetDropdowns(message = "") {
    const styleSelect = root.querySelector('[data-role="style-preset-select"]');
    const gridSelect = root.querySelector('[data-role="grid-preset-select"]');
    const targetSelect = root.querySelector('[data-role="target-resolution"]');
    syncSheetSpecsVisibility();
    if (!styleEntries()[state.styleSelectedKey]) state.styleSelectedKey = "default_photoreal_cinema";
    if (styleSelect) renderSelectOptions(styleSelect, styleEntries(), state.styleSelectedKey);
    if (gridSelect) {
      renderSelectOptions(gridSelect, gridEntries(), state.gridSelectedKey);
      gridSelect.disabled = false;
    }
    if (targetSelect) renderSelectOptions(targetSelect, targetResolutionEntries(), state.targetResolutionKey);
  }

  function deleteImportedPreset(kind, key, anchor = null) {
    const cleanKey = cleanText(key);
    if (!cleanKey) return;
    const config = {
      gallery: { imported: IMPORTED_PRESETS, builtin: PRESETS, store: PERSISTED_PRESET_KEYS.gallery, stateKey: "gallerySelectedKey", fallback: "storyboard", label: "gallery" },
      style: { imported: IMPORTED_STYLE_PRESETS, builtin: STYLE_PRESETS, store: PERSISTED_PRESET_KEYS.styles, stateKey: "styleSelectedKey", fallback: "default_photoreal_cinema", label: "style" },
      grid: { imported: IMPORTED_GRID_PRESETS, builtin: GRID_PRESETS, store: PERSISTED_PRESET_KEYS.grids, stateKey: "gridSelectedKey", fallback: "free_canvas", label: "grid" },
    }[kind];
    if (!config) return;

    if (kind === "gallery") {
      if (config.imported[cleanKey]) {
        deleteUserGalleryPreset(cleanKey, anchor);
        return;
      }
      if (config.builtin[cleanKey]) {
        const hidden = hiddenGalleryPresetKeys();
        hidden[cleanKey] = true;
        const hiddenOk = writeHiddenGalleryPresetKeys(hidden);
        state[config.stateKey] = Object.keys(presetEntries())[0] || config.fallback;
        render();
        showToast(`Hidden built-in gallery preset: ${cleanKey}${hiddenOk ? "" : " (session only: browser cache blocked)"}`, { anchor, tone: hiddenOk ? "success" : "warn" });
        return;
      }
      showToast(`No gallery preset selected: ${cleanKey}`, { anchor, tone: "warn" });
      return;
    }

    if (config.builtin[cleanKey] && !config.imported[cleanKey]) {
      showToast(`Cannot delete built-in ${config.label} preset: ${cleanKey}`, { anchor, tone: "warn" });
      return;
    }
    if (!config.imported[cleanKey]) {
      showToast(`No imported ${config.label} preset selected`, { anchor, tone: "warn" });
      return;
    }
    delete config.imported[cleanKey];
    const persisted = readPersistedObject(config.store);
    delete persisted[cleanKey];
    writePersistedObject(config.store, persisted);
    state[config.stateKey] = Object.keys(config.imported)[0] || config.fallback;
    state.showSheetSpecs = true;
    render();
    showToast(`Deleted ${config.label} preset: ${cleanKey}`, { anchor, tone: "success" });
  }

  function applyStylePreset(key) {
    const entries = styleEntries();
    const cleanKey = entries[key] ? key : "default_photoreal_cinema";
    const preset = entries[cleanKey] || entries.default_photoreal_cinema;
    if (!preset) return;
    state.styleSelectedKey = cleanKey;
    state.data.scene.high_level_description = preset.high || state.data.scene.high_level_description;
    state.data.scene.background = preset.background || state.data.scene.background;
    state.data.scene.aesthetics = preset.aesthetics || state.data.scene.aesthetics;
    state.data.scene.lighting = preset.lighting || state.data.scene.lighting;
    state.data.scene.photo = preset.photo || state.data.scene.photo;
    state.data.scene.medium = preset.medium || state.data.scene.medium;
    state.data.scene.color_palette = paletteList(preset.palette, state.data.scene.color_palette);
    rebuildLayoutForWorkflow({ rebuildGrid: false, rebuildLayout: false });
    persist();
    render();
    footStatus.textContent = `Applied style preset: ${preset.label || cleanKey}`;
  }

  function applyGridPreset(key) {
    const preset = gridEntries()[key] || gridEntries().free_canvas || gridEntries().story_2x3;
    if (!preset) return;
    state.gridSelectedKey = key;
    if (key === "free_canvas" || !preset.boxes) {
      state.workflowModeKey = "single_image";
      state.data.workflow_mode = state.workflowModeKey;
      state.data.grid_key = "free_canvas";
      state.data.canvas.width = Number(preset.width || state.data.canvas.width || 1280);
      state.data.canvas.height = Number(preset.height || state.data.canvas.height || 720);
      state.data.canvas.aspect_label = preset.aspect_label || preset.label || "Free Canvas";
      state.data.items = [];
      state.selectedId = null;
      persist();
      render();
      footStatus.textContent = `Applied grid preset: ${preset.label || key}`;
      return;
    }
    state.workflowModeKey = "storyboard_grid";
    state.data.workflow_mode = state.workflowModeKey;
    state.data.grid_key = key;
    clearReferenceModeForStoryboard(false);
    state.data.preset_key = "storyboard";
    state.data.canvas.width = Number(preset.width || state.data.canvas.width || 1536);
    state.data.canvas.height = Number(preset.height || state.data.canvas.height || 864);
    state.data.canvas.aspect_label = preset.aspect_label || preset.label || state.data.canvas.aspect_label;
    if (preset.items?.length) {
      state.data.items = preset.items.map((item, index) => normalizeItem(item, index));
      state.selectedId = state.data.items[0]?.id || null;
    } else if (preset.boxes) {
      state.data.items = mergeLayoutItems(makeGridItems(preset.boxes[0], preset.boxes[1], state.data.scene.color_palette || []), state.data.items || []);
      state.selectedId = state.data.items[0]?.id || null;
    }
    const targetPreset = targetResolutionEntries()[state.targetResolutionKey];
    if (targetPreset && state.targetResolutionKey !== "custom" && Number(targetPreset.width) > 0 && Number(targetPreset.height) > 0 && preset.boxes) {
      state.data.canvas.width = Number(targetPreset.width) * Math.max(1, Number(preset.boxes[0]) || 1);
      state.data.canvas.height = Number(targetPreset.height) * Math.max(1, Number(preset.boxes[1]) || 1);
      state.data.canvas.aspect_label = `${targetPreset.label} target / ${preset.label || key} canvas ${state.data.canvas.width}x${state.data.canvas.height}`;
    }
    persist();
    render();
    footStatus.textContent = `Applied grid preset: ${preset.label || key}`;
  }

  function importStylePresetFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = parseJsonLoose(String(reader.result || ""));
        const gallery = { ...normalizeStyleGallery(parsed), ...directStylePresetsFromJson(parsed) };
        const entries = Object.entries(gallery);
        if (!entries.length) throw new Error("empty style preset file");
        const persisted = readPersistedObject(PERSISTED_PRESET_KEYS.styles);
        entries.forEach(([key, preset], index) => {
          IMPORTED_STYLE_PRESETS[key] = preset;
          persisted[key] = preset;
          if (index === 0) state.styleSelectedKey = key;
        });
        const persistedOk = writePersistedObject(PERSISTED_PRESET_KEYS.styles, persisted);
        console.info("[IAMCCS FrameDesigner] import style", { file: file.name, entries });
        render();
        const message = `Imported ${entries.length} style preset${entries.length === 1 ? "" : "s"} from ${file.name}. Press Apply to update the scene boxes.${persistedOk ? "" : " (session only: browser cache blocked)"}`;
        footStatus.textContent = message;
        showToast(message, { tone: persistedOk ? "success" : "warn" });
      } catch (error) {
        const message = `Style preset import failed: ${error?.message || file.name}`;
        footStatus.textContent = message;
        showToast(message, { tone: "warn" });
      }
    };
    reader.readAsText(file);
  }

  function importGridPresetFile(file) {
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = parseJsonLoose(String(reader.result || ""));
        const gallery = { ...normalizeGridGallery(parsed), ...directGridPresetsFromJson(parsed) };
        const entries = Object.entries(gallery);
        if (!entries.length) throw new Error("empty grid preset file");
        const persisted = readPersistedObject(PERSISTED_PRESET_KEYS.grids);
        entries.forEach(([key, preset], index) => {
          IMPORTED_GRID_PRESETS[key] = preset;
          persisted[key] = preset;
          state.showSheetSpecs = true;
          if (index === 0) state.gridSelectedKey = key;
        });
        const persistedOk = writePersistedObject(PERSISTED_PRESET_KEYS.grids, persisted);
        console.info("[IAMCCS FrameDesigner] import grid", { file: file.name, entries });
        render();
        const message = `Imported ${entries.length} grid preset${entries.length === 1 ? "" : "s"} from ${file.name}. Press Apply to update the canvas/grid.${persistedOk ? "" : " (session only: browser cache blocked)"}`;
        footStatus.textContent = message;
        showToast(message, { tone: persistedOk ? "success" : "warn" });
      } catch (error) {
        const message = `Grid preset import failed: ${error?.message || file.name}`;
        footStatus.textContent = message;
        showToast(message, { tone: "warn" });
      }
    };
    reader.readAsText(file);
  }

  function moveLayer(id, mode) {
    const items = state.data.items || [];
    const index = items.findIndex((entry) => entry.id === id);
    if (index < 0) return;
    const [item] = items.splice(index, 1);
    let nextIndex = index;
    if (mode === "front") nextIndex = items.length;
    else if (mode === "back") nextIndex = 0;
    else if (mode === "up") nextIndex = Math.min(items.length, index + 1);
    else if (mode === "down") nextIndex = Math.max(0, index - 1);
    items.splice(nextIndex, 0, item);
    state.selectedId = item.id;
    persist();
    render();
    footStatus.textContent = `Layer ${item.label || item.id} moved ${mode}`;
  }

  function buildScenePane() {
    scenePane.innerHTML = '';
    const presets = document.createElement('div');
    presets.className = 'iamccs-isf-panel';
    presets.innerHTML = '<h4>Preset Gallery</h4><div class="iamccs-isf-gallery-grid" data-preset-gallery></div><div class="iamccs-isf-gallery-actions"><button class="iamccs-isf-btn primary" data-gallery-action="replace">Replace Board</button><button class="iamccs-isf-btn" data-gallery-action="show-manager">Show Gallery</button><button class="iamccs-isf-btn" data-gallery-action="import">Import Preset</button></div>';
    scenePane.appendChild(presets);

    const presetGallery = presets.querySelector('[data-preset-gallery]');
    renderPresetGallery(presetGallery);
    presets.querySelector('[data-gallery-action="replace"]')?.addEventListener('click', () => applyPresetToBoard(selectedGalleryKey(), 'replace'));
    presets.querySelector('[data-gallery-action="show-manager"]')?.addEventListener('click', (event) => openPresetGalleryManager(event.currentTarget));
    presets.querySelector('[data-gallery-action="import"]')?.addEventListener('click', () => presetGalleryInput?.click());

    const modePanel = document.createElement('div');
    modePanel.className = 'iamccs-isf-panel mode-panel';
    const activeMode = currentWorkflowMode();
    modePanel.innerHTML = `
      <h4>Workflow Mode</h4>
      <div class="iamccs-isf-field">
        <label>Mode</label>
        <select data-role="workflow-mode"></select>
      </div>
      <p class="iamccs-isf-direct-note" data-role="workflow-mode-note"></p>
      <p class="iamccs-isf-direct-note">Mode controls defaults, canvas multiplier, i2i source, reference layout, and target crop math. Change Panel Size below; the total canvas follows automatically.</p>
    `;
    scenePane.appendChild(modePanel);
    const workflowSelect = modePanel.querySelector('[data-role="workflow-mode"]');
    const workflowNote = modePanel.querySelector('[data-role="workflow-mode-note"]');
    const visibleWorkflowModes = workflowModeDropdownEntries();
    renderSelectOptions(workflowSelect, visibleWorkflowModes, visibleWorkflowModes[activeMode.key] ? activeMode.key : "storyboard_grid");
    if (workflowNote) workflowNote.textContent = activeMode.summary || "";
    workflowSelect?.addEventListener('change', () => applyWorkflowMode(workflowSelect.value));

    const sheetTools = document.createElement('div');
    sheetTools.dataset.role = "sheetbuilder-specs-panel";
    sheetTools.className = `iamccs-isf-panel ${state.showSheetSpecs ? '' : 'collapsed'}`;
    sheetTools.innerHTML = `
      <div class="iamccs-isf-spec-head">
        <h4>Grid Preset</h4>
        <label class="iamccs-isf-toggle"><input type="checkbox" data-role="toggle-sheet-specs" ${state.showSheetSpecs ? 'checked' : ''}> Show</label>
      </div>
      <div class="iamccs-isf-sheet-controls">
        <label>Storyboard / Layout Grid<select data-role="grid-preset-select"></select></label>
        <div class="iamccs-isf-mini-actions">
          <button class="iamccs-isf-btn primary" type="button" data-action-local="apply-grid">Apply</button>
          <button class="iamccs-isf-btn" type="button" data-action-local="import-grid">Import</button>
          <button class="iamccs-isf-btn" type="button" data-action-local="save-grid">Save</button>
          <button class="iamccs-isf-btn danger-lite" type="button" data-action-local="delete-grid">Delete</button>
        </div>
      </div>`;
    scenePane.appendChild(sheetTools);
    const specsToggle = sheetTools.querySelector('[data-role="toggle-sheet-specs"]');
    specsToggle?.addEventListener('change', () => {
      state.showSheetSpecs = Boolean(specsToggle.checked);
      sheetTools.classList.toggle('collapsed', !state.showSheetSpecs);
      if (!state.showSheetSpecs) {
        state.data.canvas.width = Number(state.data.canvas.width || 1536) || 1536;
        state.data.canvas.height = Number(state.data.canvas.height || 864) || 864;
        if (!state.data.canvas.aspect_label) state.data.canvas.aspect_label = '16:9 Storyboard';
        persist();
        syncSceneFields();
        updatePromptPreview();
        footStatus.textContent = 'SheetBuilder Specs hidden; canvas controls remain manual.';
      }
    });
    const gridSelect = sheetTools.querySelector('[data-role="grid-preset-select"]');
    renderSelectOptions(gridSelect, gridEntries(), state.gridSelectedKey);
    if (gridSelect) gridSelect.disabled = false;
    sheetTools.querySelector('[data-action-local="apply-grid"]')?.addEventListener('click', () => applyGridPreset(gridSelect.value));
    sheetTools.querySelector('[data-action-local="import-grid"]')?.addEventListener('click', () => gridPresetInput?.click());
    sheetTools.querySelector('[data-action-local="save-grid"]')?.addEventListener('click', (event) => saveCurrentGridPreset(event.currentTarget));
    sheetTools.querySelector('[data-action-local="delete-grid"]')?.addEventListener('click', (event) => deleteImportedPreset('grid', gridSelect.value, event.currentTarget));
    gridSelect?.addEventListener('change', () => { state.gridSelectedKey = gridSelect.value; refreshPresetDropdowns("grid select changed"); });

    const canvas = document.createElement('div');
    canvas.className = 'iamccs-isf-panel';
    canvas.innerHTML = '<h4>Canvas Controls</h4>';
    scenePane.appendChild(canvas);
    const targetWrap = document.createElement('div');
    targetWrap.className = 'iamccs-isf-field';
    targetWrap.innerHTML = `
      <label>Target Resolution / Panel Size</label>
      <select data-role="target-resolution"></select>
      <p class="iamccs-isf-direct-note" data-role="target-resolution-note"></p>
    `;
    canvas.appendChild(targetWrap);
    sceneFields.targetResolution = targetWrap.querySelector('[data-role="target-resolution"]');
    sceneFields.targetResolutionNote = targetWrap.querySelector('[data-role="target-resolution-note"]');
    const canvasGrid = document.createElement('div');
    canvasGrid.className = 'iamccs-isf-posgrid';
    canvas.appendChild(canvasGrid);
    sceneFields.width = makeField(canvasGrid, 'Width', 'width', { type: 'number' });
    sceneFields.height = makeField(canvasGrid, 'Height', 'height', { type: 'number' });
    sceneFields.aspect = makeField(canvas, 'Aspect Label', 'aspect');

    state.data.direct_prompt = { enabled: false, text: "" };

    const stylePresetPanel = document.createElement('div');
    stylePresetPanel.className = 'iamccs-isf-panel iamccs-isf-style-preset-panel';
    stylePresetPanel.innerHTML = `
      <h4>Style Preset</h4>
      <div class="iamccs-isf-sheet-controls">
        <label>Cinema / Look<select data-role="style-preset-select"></select></label>
        <div class="iamccs-isf-mini-actions">
          <button class="iamccs-isf-btn primary" type="button" data-action-local="apply-style">Apply</button>
          <button class="iamccs-isf-btn" type="button" data-action-local="import-style">Import</button>
          <button class="iamccs-isf-btn" type="button" data-action-local="save-style">Save</button>
          <button class="iamccs-isf-btn danger-lite" type="button" data-action-local="delete-style">Delete</button>
        </div>
      </div>`;
    scenePane.appendChild(stylePresetPanel);
    const styleSelect = stylePresetPanel.querySelector('[data-role="style-preset-select"]');
    renderSelectOptions(styleSelect, styleEntries(), state.styleSelectedKey);
    stylePresetPanel.querySelector('[data-action-local="apply-style"]')?.addEventListener('click', () => applyStylePreset(styleSelect.value));
    stylePresetPanel.querySelector('[data-action-local="import-style"]')?.addEventListener('click', () => stylePresetInput?.click());
    stylePresetPanel.querySelector('[data-action-local="save-style"]')?.addEventListener('click', (event) => saveCurrentStylePreset(event.currentTarget));
    stylePresetPanel.querySelector('[data-action-local="delete-style"]')?.addEventListener('click', (event) => deleteImportedPreset('style', styleSelect.value, event.currentTarget));
    styleSelect?.addEventListener('change', () => { state.styleSelectedKey = styleSelect.value; refreshPresetDropdowns("style select changed"); });

    const summary = document.createElement('div');
    summary.className = 'iamccs-isf-panel iamccs-isf-scene-panel';
    summary.innerHTML = '<h4>Scene Direction</h4>';
    scenePane.appendChild(summary);
    sceneFields.high = makeField(summary, 'High-Level Description', 'high', { multiline: true, gemmaField: 'scene.high_level_description' });
    sceneFields.background = makeField(summary, 'Background', 'background', { multiline: true, small: true, gemmaField: 'scene.background' });

    const style = document.createElement('div');
    style.className = 'iamccs-isf-panel iamccs-isf-art-panel';
    style.innerHTML = '<h4>Art Direction</h4>';
    scenePane.appendChild(style);
    sceneFields.aesthetics = makeField(style, 'Aesthetics', 'aesthetics', { multiline: true, small: true, gemmaField: 'scene.aesthetics' });
    sceneFields.lighting = makeField(style, 'Lighting', 'lighting', { multiline: true, small: true, gemmaField: 'scene.lighting' });
    sceneFields.medium = makeField(style, 'Medium', 'medium', { gemmaField: 'scene.medium' });
    sceneFields.photo = makeField(style, 'Photo / Lens Notes', 'photo', { multiline: true, small: true, gemmaField: 'scene.photo' });
    sceneFields.palette = makeField(style, 'Global Palette', 'palette', { gemmaField: 'scene.color_palette' });
    const scenePaletteTools = document.createElement('div');
    scenePaletteTools.className = 'iamccs-isf-field iamccs-isf-palette-tools';
    scenePaletteTools.innerHTML = `
      <label>Palette Preset</label>
      <select data-role="scene-palette-preset"></select>
      <p class="iamccs-isf-direct-note" data-role="scene-palette-note"></p>
    `;
    style.appendChild(scenePaletteTools);
    sceneFields.palettePreset = scenePaletteTools.querySelector('[data-role="scene-palette-preset"]');
    sceneFields.paletteNote = scenePaletteTools.querySelector('[data-role="scene-palette-note"]');
    renderSelectOptions(sceneFields.palettePreset, paletteEntries(), paletteKeyForColors(state.data.scene.color_palette));
    sceneFields.palettePreset?.addEventListener('change', (event) => {
      const key = event.currentTarget.value || "custom";
      if (key === "custom") return;
      state.data.scene.color_palette = applyPalettePresetToList(key, state.data.scene.color_palette);
      persist();
      render();
      showToast(`Global palette: ${paletteEntries()[key]?.label || key}`, { anchor: event.currentTarget, tone: "success", ms: 1800 });
    });

    const gemmaPanel = document.createElement('div');
    gemmaPanel.className = 'iamccs-isf-panel iamccs-isf-gemma-panel';
    gemmaPanel.innerHTML = `
      <div class="iamccs-isf-panel-headline">
        <h4>Gemma Assistant</h4>
        <label class="iamccs-isf-toggle"><input type="checkbox" data-role="gemma-enabled"> Enable</label>
      </div>
      <div class="iamccs-isf-gemma-body" data-role="gemma-body">
        <p class="iamccs-isf-direct-note">Local Gemma 4 assistant for box enhancement, prompt critique, and full ideoboard JSON preparation. It updates text/JSON only; it never queues Ideogram image generation.</p>
        <div class="iamccs-isf-field">
          <label>Gemma Text Encoder</label>
          <select data-role="gemma-model"></select>
        </div>
        <div class="iamccs-isf-field">
          <label>Mode</label>
          <select data-role="gemma-mode">
            <option value="full_json_enhance">Enhance Full Board</option>
            <option value="brief_to_ideoboard">Brief To Ideoboard</option>
            <option value="prompt_critic">Prompt Critic Only</option>
          </select>
        </div>
        <div class="iamccs-isf-field">
          <label>Speed</label>
          <select data-role="gemma-speed">
            <option value="fast">Fast / shorter</option>
            <option value="detailed">Detailed / slower</option>
          </select>
        </div>
        <div class="iamccs-isf-field">
          <label>Natural Language Direction</label>
          <textarea data-role="gemma-brief" rows="5" placeholder="Example: make this a 2x3 photoreal storyboard with close-ups, one physical action beat, muted color, and one exhausted astronaut on an ice planet."></textarea>
        </div>
        <div class="iamccs-isf-mini-actions">
          <button class="iamccs-isf-btn primary" type="button" data-role="gemma-run">Run Gemma</button>
          <button class="iamccs-isf-btn danger-lite" type="button" data-role="gemma-stop" disabled>Stop Gemma</button>
          <button class="iamccs-isf-btn" type="button" data-role="gemma-copy-request">Copy Request</button>
        </div>
        <div class="iamccs-isf-gemma-result" data-role="gemma-result" hidden>
          <div class="iamccs-isf-gemma-result-head">
            <strong>Gemma Proposal</strong>
            <span data-role="gemma-result-label"></span>
          </div>
          <textarea data-role="gemma-result-text" readonly spellcheck="false"></textarea>
          <div class="iamccs-isf-mini-actions">
            <button class="iamccs-isf-btn primary" type="button" data-role="gemma-apply" disabled>Apply</button>
            <button class="iamccs-isf-btn danger-lite" type="button" data-role="gemma-reject" disabled>Reject</button>
            <button class="iamccs-isf-btn" type="button" data-role="gemma-copy-result" disabled>Copy</button>
          </div>
        </div>
      </div>
      <p class="iamccs-isf-direct-note" data-role="gemma-status">Ready. This does not queue image generation.</p>
    `;
    scenePane.insertBefore(gemmaPanel, stylePresetPanel);
    const gemmaEnabled = gemmaPanel.querySelector('[data-role="gemma-enabled"]');
    const gemmaBody = gemmaPanel.querySelector('[data-role="gemma-body"]');
    const gemmaMode = gemmaPanel.querySelector('[data-role="gemma-mode"]');
    const gemmaModel = gemmaPanel.querySelector('[data-role="gemma-model"]');
    const gemmaSpeed = gemmaPanel.querySelector('[data-role="gemma-speed"]');
    const gemmaBrief = gemmaPanel.querySelector('[data-role="gemma-brief"]');
    const gemmaStatus = gemmaPanel.querySelector('[data-role="gemma-status"]');
    if (gemmaSpeed) gemmaSpeed.value = state.data.gemma_assistant?.speed === "detailed" ? "detailed" : "fast";
    renderGemmaPendingResult();
    const syncGemmaEnabledUi = () => {
      const enabled = Boolean(state.data.gemma_assistant?.enabled);
      if (gemmaEnabled) gemmaEnabled.checked = enabled;
      if (gemmaBody) gemmaBody.hidden = !enabled;
      root.classList.toggle('gemma-enabled', enabled);
      if (gemmaStatus) gemmaStatus.textContent = enabled
        ? 'Ready. Uses local Gemma 4 TextGenerate backend only.'
        : 'Disabled. Enable Gemma Assistant to show and use field buttons.';
    };
    syncGemmaEnabledUi();
    loadGemmaModelOptions(gemmaModel, gemmaStatus);
    gemmaEnabled?.addEventListener('change', () => {
      state.data.gemma_assistant = {
        ...(state.data.gemma_assistant || {}),
        enabled: Boolean(gemmaEnabled.checked),
        provider: "local_gemma",
        model: gemmaModel?.value || "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors",
      };
      persist();
      syncGemmaEnabledUi();
    });
    gemmaSpeed?.addEventListener('change', () => {
      state.data.gemma_assistant = {
        ...(state.data.gemma_assistant || {}),
        speed: gemmaSpeed.value === "detailed" ? "detailed" : "fast",
      };
      persist();
      if (gemmaStatus) gemmaStatus.textContent = `Gemma speed set to ${state.data.gemma_assistant.speed}.`;
    });
    gemmaModel?.addEventListener('change', () => {
      state.data.gemma_assistant = {
        ...(state.data.gemma_assistant || {}),
        provider: "local_gemma",
        model: gemmaModel.value || "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors",
      };
      persist();
    });
    gemmaPanel.querySelector('[data-role="gemma-run"]')?.addEventListener('click', (event) => {
      runOrPrepareGemma({
        mode: gemmaMode?.value || "full_json_enhance",
        brief: gemmaBrief?.value || "",
        speed: gemmaSpeed?.value || "fast",
        statusHost: gemmaStatus,
        anchor: event.currentTarget,
      });
    });
    gemmaPanel.querySelector('[data-role="gemma-stop"]')?.addEventListener('click', (event) => {
      stopGemmaAssistant(event.currentTarget, gemmaStatus);
    });
    gemmaPanel.querySelector('[data-role="gemma-apply"]')?.addEventListener('click', (event) => {
      applyGemmaPending(event.currentTarget, gemmaStatus);
    });
    gemmaPanel.querySelector('[data-role="gemma-reject"]')?.addEventListener('click', (event) => {
      clearGemmaPending(gemmaStatus, "Gemma proposal rejected.");
      showToast("Gemma proposal rejected.", { anchor: event.currentTarget, tone: "warn", ms: 2400 });
    });
    gemmaPanel.querySelector('[data-role="gemma-copy-result"]')?.addEventListener('click', (event) => {
      copyGemmaPending(event.currentTarget);
    });
    gemmaPanel.querySelector('[data-role="gemma-copy-request"]')?.addEventListener('click', (event) => {
      const request = buildGemmaFrontendRequest(gemmaMode?.value || "full_json_enhance", gemmaBrief?.value || "", { speed: gemmaSpeed?.value || "fast" });
      request.model = gemmaModel?.value || "text_encoders\\gemma4_e4b_it_fp8_scaled.safetensors";
      navigator.clipboard?.writeText(JSON.stringify(request, null, 2));
      showToast("Gemma request copied", { anchor: event.currentTarget, tone: "success" });
    });

    if (isV2) {
      const i2iPanel = document.createElement('div');
      i2iPanel.className = 'iamccs-isf-panel iamccs-isf-refine-panel';
      i2iPanel.innerHTML = `
        <div class="iamccs-isf-panel-headline">
          <h4>Refine / Image Guide <span class="iamccs-isf-beta">BETA</span></h4>
          <label class="iamccs-isf-toggle"><input type="checkbox" data-role="toggle-refine-guide" ${state.showRefineGuide ? 'checked' : ''}> Show</label>
        </div>
        <div class="iamccs-isf-refine-body" data-role="refine-guide-body" ${state.showRefineGuide ? '' : 'hidden'}>
          <button class="iamccs-isf-refine-hero" type="button" data-action="refine">
            <strong>REFINE SELECTED GUIDE</strong>
            <span>Preserve composition and identity: i2i on, denoise 0.28, 1280x720.</span>
          </button>
        </div>
      `;
      scenePane.appendChild(i2iPanel);
      const refineBody = i2iPanel.querySelector('[data-role="refine-guide-body"]');
      const refineToggle = i2iPanel.querySelector('[data-role="toggle-refine-guide"]');
      refineToggle?.addEventListener('change', () => {
        state.showRefineGuide = Boolean(refineToggle.checked);
        if (refineBody) refineBody.hidden = !state.showRefineGuide;
        persistUiState();
      });
      i2iPanel.querySelector('[data-action="refine"]')?.addEventListener('click', applyRefinePreset);
      i2iFields.enabled = makeField(refineBody || i2iPanel, 'Use Canvas/Image as i2i Guide', 'i2i_enabled', { type: 'checkbox' });
      i2iFields.enabled.type = 'checkbox';
      const i2iGuideNote = document.createElement('p');
      i2iGuideNote.className = 'iamccs-isf-direct-note';
      i2iGuideNote.textContent = 'ON: the current canvas or imported image keeps composition and subject placement. OFF: Ideogram follows the prompt more freely.';
      (refineBody || i2iPanel).appendChild(i2iGuideNote);
      const denoiseGuide = document.createElement('div');
      denoiseGuide.className = 'iamccs-isf-denoise-guide';
      denoiseGuide.innerHTML = `
        <div class="iamccs-isf-denoise-title">
          <span>Modification Degree</span>
          <span class="iamccs-isf-denoise-value" data-denoise-readout></span>
        </div>
        <div class="iamccs-isf-denoise-buttons">
          ${DENOISE_PRESETS.map((preset) => `
            <button class="iamccs-isf-denoise-preset" type="button" data-denoise-preset="${preset.key}" data-value="${preset.value}" title="${preset.hint}">
              <strong><span>${preset.label}</span><span>${preset.value.toFixed(2)}</span></strong>
              <span>${preset.hint}</span>
            </button>
          `).join('')}
        </div>
        <p class="iamccs-isf-denoise-help">This writes to the node output <strong>i2i_denoise</strong>. Lower values keep the canvas image; higher values let Ideogram rebuild more of the shot.</p>
      `;
      (refineBody || i2iPanel).appendChild(denoiseGuide);

      i2iFields.denoise = makeField(refineBody || i2iPanel, 'Manual Denoise Value', 'i2i_denoise', { type: 'number' });
      i2iFields.denoise.min = '0';
      i2iFields.denoise.max = '1';
      i2iFields.denoise.step = '0.01';
      i2iFields.step = makeField(refineBody || i2iPanel, 'Low Sigma Start Step', 'low_sigma_start_step', { type: 'number' });

      i2iPanel.querySelectorAll('[data-denoise-preset]').forEach((button) => {
        button.addEventListener('click', () => {
          setDenoise(Number(button.dataset.value), `Applied ${button.textContent.trim().split(/\s+/)[0]} preset`);
          render();
        });
      });

      Object.entries(i2iFields).forEach(([key, input]) => {
        input.addEventListener('input', () => {
          state.data.i2i = normalizeI2I(state.data.i2i);
          if (key === 'enabled') state.data.i2i.enabled = Boolean(input.checked);
          else if (key === 'denoise') state.data.i2i.denoise = Math.max(0, Math.min(1, Number(input.value) || 0));
          else if (key === 'step') state.data.i2i.low_sigma_start_step = clampInt(input.value, 0, 1000, 12);
          setWidget(node, "i2i_enabled", Boolean(state.data.i2i.enabled));
          setWidget(node, "i2i_denoise", Number(state.data.i2i.denoise));
          setWidget(node, "low_sigma_start_step", Number(state.data.i2i.low_sigma_start_step));
          syncDenoisePresetButtons();
          persist();
          render();
        });
        input.addEventListener('change', () => input.dispatchEvent(new Event('input')));
      });
      syncDenoisePresetButtons();
    }

    renderSelectOptions(sceneFields.targetResolution, targetResolutionEntries(), state.targetResolutionKey);
    const handleTargetResolutionChange = (event) => {
      const value = event.currentTarget?.value || event.target?.value || "";
      if (!value) return;
      applyTargetResolutionPreset(value);
    };
    sceneFields.targetResolution?.addEventListener('change', handleTargetResolutionChange);
    sceneFields.targetResolution?.addEventListener('input', handleTargetResolutionChange);

    const commitCanvasDimension = (key, value, fromDrag = false) => {
      const fallback = key === 'width' ? (state.data.canvas.width || 1536) : (state.data.canvas.height || 864);
      const next = clampInt(value, 256, 16384, fallback);
      state.data.canvas[key] = next;
      state.targetResolutionKey = "custom";
      state.data.canvas.target_resolution_key = "custom";
      state.data.canvas.target_width = Number(state.data.canvas.width || 1536);
      state.data.canvas.target_height = Number(state.data.canvas.height || 864);
      state.data.canvas.aspect_label = `Custom canvas ${state.data.canvas.width || 1536}x${state.data.canvas.height || 864}`;
      const field = key === 'width' ? sceneFields.width : sceneFields.height;
      if (field) field.value = String(next);
      persist();
      render();
      footStatus.textContent = `${key === 'width' ? 'Width' : 'Height'} set to ${next}${fromDrag ? ' by drag' : ''}`;
    };
    ['width', 'height'].forEach((dimension) => {
      const input = sceneFields[dimension];
      input.addEventListener('keydown', (event) => {
        if (event.key === 'Enter') {
          event.preventDefault();
          commitCanvasDimension(dimension, input.value);
          input.blur();
        } else if (event.key === 'Escape') {
          input.value = String(state.data.canvas[dimension] || (dimension === 'width' ? 1536 : 864));
          input.blur();
        }
      });
      input.addEventListener('blur', () => commitCanvasDimension(dimension, input.value));
      input.addEventListener('change', () => commitCanvasDimension(dimension, input.value));
      enableNumberDrag(input, (value, fromDrag) => commitCanvasDimension(dimension, value, fromDrag), { step: 16, min: 256, max: 16384 });
    });
    Object.entries(sceneFields).forEach(([key, input]) => {
      if (['width', 'height', 'targetResolution', 'targetResolutionNote', 'palettePreset', 'paletteNote'].includes(key)) return;
      input.addEventListener('input', () => {
        if (key === 'aspect') state.data.canvas.aspect_label = input.value;
        else if (key === 'high') state.data.scene.high_level_description = input.value;
        else if (key === 'background') state.data.scene.background = input.value;
        else if (key === 'aesthetics') state.data.scene.aesthetics = input.value;
        else if (key === 'lighting') state.data.scene.lighting = input.value;
        else if (key === 'medium') state.data.scene.medium = input.value;
        else if (key === 'photo') state.data.scene.photo = input.value;
        else if (key === 'palette') state.data.scene.color_palette = paletteList(input.value, state.data.scene.color_palette);
        persist();
        render();
      });
    });
  }

  function buildInspectorPane() {
    inspectorPane.innerHTML = '';
    if (isV2) {
      const maskPanel = document.createElement('div');
      maskPanel.className = 'iamccs-isf-panel iamccs-isf-mask-panel';
      maskPanel.innerHTML = `
        <h4>Image Mask Paint</h4>
        <div class="iamccs-isf-mask-tools">
          <button class="iamccs-isf-mask-tool" type="button" data-mask-tool="select">Select</button>
          <button class="iamccs-isf-mask-tool" type="button" data-mask-tool="shape">Shape</button>
          <button class="iamccs-isf-mask-tool" type="button" data-mask-tool="brush">Brush</button>
          <button class="iamccs-isf-mask-tool" type="button" data-mask-tool="erase">Erase</button>
        </div>
        <div class="iamccs-isf-mask-row">
          <div class="iamccs-isf-field">
            <label>Brush Size</label>
            <input type="number" min="1" max="240" step="1" data-role="mask-brush-size" />
          </div>
          <button class="iamccs-isf-btn danger-lite" type="button" data-mask-clear>Clear</button>
        </div>
        <p class="iamccs-isf-mask-status" data-mask-status></p>`;
      inspectorPane.appendChild(maskPanel);
      maskBrushSizeInput = maskPanel.querySelector('[data-role="mask-brush-size"]');
      maskStatusHost = maskPanel.querySelector('[data-mask-status]');
      maskDebugHost = null;
      maskPreviewCanvas = null;
      maskPanel.querySelectorAll('[data-mask-tool]').forEach((button) => {
        button.addEventListener('click', () => {
          state.paintTool = button.dataset.maskTool || 'select';
          syncMaskUi();
          renderArtboard();
        });
      });
      maskBrushSizeInput?.addEventListener('input', () => {
        const paint = maskPaintData();
        paint.brush_size = clampInt(maskBrushSizeInput.value, 1, 240, paint.brush_size || 48);
        state.data.mask_paint = paint;
        writeData(node, state.data);
        syncMaskUi();
      });
      maskPanel.querySelector('[data-mask-clear]')?.addEventListener('click', (event) => clearCurrentImageMask(event.currentTarget));
    }
    const showImgPanel = document.createElement('div');
    showImgPanel.className = 'iamccs-isf-panel iamccs-isf-show-img-panel';
    showImgPanel.innerHTML = `
      <h4>Show Img Under Boxes</h4>
      <div class="iamccs-isf-show-img-row">
        <label class="iamccs-isf-toggle"><input type="checkbox" data-role="show-img-toggle"> Show Img</label>
        <button class="iamccs-isf-btn" type="button" data-role="show-img-grab">Grab Last</button>
      </div>
      <div class="iamccs-isf-range-row">
        <input type="range" min="0" max="100" step="1" data-role="show-img-opacity" />
        <strong data-role="show-img-opacity-readout">82%</strong>
      </div>
      <div class="iamccs-isf-show-img-row" style="margin-top:8px;">
        <span class="iamccs-isf-show-img-note">Visual only: this does not enter the JSON, IDEO_LINX, crops, or final render.</span>
        <button class="iamccs-isf-btn danger-lite" type="button" data-role="show-img-clear">Clear</button>
      </div>`;
    inspectorPane.appendChild(showImgPanel);
    showImgPanel.querySelector('[data-role="show-img-toggle"]')?.addEventListener('change', (event) => {
      state.showResultImage = Boolean(event.currentTarget.checked);
      if (state.showResultImage && state.resultBgUrl) state.resultOverlayVisible = true;
      persistUiState();
      syncShowImgControls();
      renderArtboard();
    });
    showImgPanel.querySelector('[data-role="show-img-opacity"]')?.addEventListener('input', (event) => {
      state.resultOverlayOpacity = Math.max(0, Math.min(1, Number(event.currentTarget.value || 0) / 100));
      persistUiState();
      syncShowImgControls();
    });
    showImgPanel.querySelector('[data-role="show-img-grab"]')?.addEventListener('click', (event) => grabLastResultBackground(event.currentTarget));
    showImgPanel.querySelector('[data-role="show-img-clear"]')?.addEventListener('click', () => clearResultBackground());

    const layers = document.createElement('div');
    layers.className = 'iamccs-isf-panel layers-panel';
    layers.innerHTML = '<h4>Frame Layers</h4><div class="iamccs-isf-list" data-layer-list></div>';
    inspectorPane.appendChild(layers);
    layerListHost = layers.querySelector('[data-layer-list]');

    const layerActions = document.createElement('div');
    layerActions.className = 'iamccs-isf-layer-actions';
    layerActions.innerHTML = `
      <button class="iamccs-isf-btn" type="button" data-layer-action="back">Send Back</button>
      <button class="iamccs-isf-btn" type="button" data-layer-action="down">Move Down</button>
      <button class="iamccs-isf-btn" type="button" data-layer-action="up">Move Up</button>
      <button class="iamccs-isf-btn" type="button" data-layer-action="front">Bring Front</button>`;
    layers.appendChild(layerActions);
    layerActions.querySelectorAll('[data-layer-action]').forEach((button) => {
      button.addEventListener('click', () => moveLayer(state.selectedId, button.dataset.layerAction));
    });

    const itemPanel = document.createElement('div');
    itemPanel.className = 'iamccs-isf-panel editor-panel';
    itemPanel.innerHTML = '<h4>Selected Layer</h4>';
    inspectorPane.appendChild(itemPanel);
    itemFields.label = makeField(itemPanel, 'Layer Label', 'label', { gemmaField: 'item.label', itemScoped: true });
    itemFields.text = makeField(itemPanel, 'Rendered Text', 'text', { multiline: true, small: true, gemmaField: 'item.text', itemScoped: true });
    itemFields.desc = makeField(itemPanel, 'Visual Description', 'desc', { multiline: true, gemmaField: 'item.desc', itemScoped: true });
    itemFields.palette = makeField(itemPanel, 'Layer Palette', 'palette', { gemmaField: 'item.palette', itemScoped: true });
    const layerPaletteTools = document.createElement('div');
    layerPaletteTools.className = 'iamccs-isf-field iamccs-isf-palette-tools';
    layerPaletteTools.innerHTML = `
      <label>Layer Palette Preset</label>
      <select data-role="layer-palette-preset"></select>
      <p class="iamccs-isf-direct-note" data-role="layer-palette-note"></p>
    `;
    itemPanel.appendChild(layerPaletteTools);
    itemFields.palettePreset = layerPaletteTools.querySelector('[data-role="layer-palette-preset"]');
    itemFields.paletteNote = layerPaletteTools.querySelector('[data-role="layer-palette-note"]');
    renderSelectOptions(itemFields.palettePreset, paletteEntries(), "custom");
    itemFields.palettePreset?.addEventListener('change', (event) => {
      const item = currentItem();
      const key = event.currentTarget.value || "custom";
      if (!item || key === "custom") return;
      item.color_palette = applyPalettePresetToList(key, item.color_palette);
      persist();
      render();
      showToast(`Layer palette: ${paletteEntries()[key]?.label || key}`, { anchor: event.currentTarget, tone: "success", ms: 1800 });
    });
    itemFields.image_path = makeField(itemPanel, 'Image Path', 'image_path');
    itemFields.fit = makeField(itemPanel, 'Image Fit', 'fit');
    itemFields.opacity = makeField(itemPanel, 'Image Opacity', 'opacity', { type: 'number' });
    const pos = document.createElement('div');
    pos.className = 'iamccs-isf-posgrid';
    itemPanel.appendChild(pos);
    itemFields.x = makeField(pos, 'X', 'x', { type: 'number' });
    itemFields.y = makeField(pos, 'Y', 'y', { type: 'number' });
    itemFields.w = makeField(pos, 'Width', 'w', { type: 'number' });
    itemFields.h = makeField(pos, 'Height', 'h', { type: 'number' });

    const previewPanel = document.createElement('div');
    previewPanel.className = 'iamccs-isf-panel export-panel';
    previewPanel.innerHTML = `
      <h4>Prompt JSON Export</h4>
      <div class="iamccs-isf-json-tools">
        <label class="iamccs-isf-toggle"><input type="checkbox" data-role="json-override-toggle"> Use JSON</label>
        <button class="iamccs-isf-btn primary" type="button" data-role="json-override-apply">Apply JSON</button>
        <button class="iamccs-isf-btn danger-lite" type="button" data-role="json-override-clear">Clear</button>
      </div>`;
    previewField = document.createElement('textarea');
    previewField.className = 'iamccs-isf-preview';
    previewField.readOnly = false;
    previewPanel.appendChild(previewField);
    inspectorPane.appendChild(previewPanel);
    previewField.addEventListener('input', () => {
      const enabled = Boolean(root.querySelector('[data-role="json-override-toggle"]')?.checked);
      state.data.json_override = { enabled, text: previewField.value };
      writeData(node, state.data);
    });
    previewPanel.querySelector('[data-role="json-override-toggle"]')?.addEventListener('change', (event) => {
      if (event.currentTarget.checked) {
        state.data.json_override = { enabled: true, text: previewField.value };
        writeData(node, state.data);
        updatePromptPreview();
        showToast("JSON override enabled. Press Apply JSON to populate boxes and canvas.", { anchor: event.currentTarget, tone: "success", ms: 3200 });
      } else {
        state.data.json_override = { enabled: false, text: "" };
        persist();
        render();
        showToast("JSON override disabled; canvas fields are source of truth", { anchor: event.currentTarget, tone: "success" });
      }
    });
    previewPanel.querySelector('[data-role="json-override-apply"]')?.addEventListener('click', (event) => {
      root.querySelector('[data-role="json-override-toggle"]').checked = true;
      applyJsonOverrideFromPreview(event.currentTarget);
    });
    previewPanel.querySelector('[data-role="json-override-clear"]')?.addEventListener('click', (event) => {
      state.data.json_override = { enabled: false, text: "" };
      persist();
      render();
      showToast("JSON override cleared", { anchor: event.currentTarget, tone: "success" });
    });

    Object.entries(itemFields).forEach(([key, input]) => {
      if (['palettePreset', 'paletteNote'].includes(key)) return;
      input.addEventListener('input', () => {
        const item = currentItem();
        if (!item) return;
        if (['x', 'y', 'w', 'h'].includes(key)) {
          item[key] = clampInt(input.value, key === 'w' || key === 'h' ? 20 : 0, 1000, item[key]);
          if (key === 'x') item.w = Math.min(item.w, 1000 - item.x);
          if (key === 'y') item.h = Math.min(item.h, 1000 - item.y);
          if (key === 'w') item.w = Math.min(item.w, 1000 - item.x);
          if (key === 'h') item.h = Math.min(item.h, 1000 - item.y);
        } else if (key === 'palette') {
          item.color_palette = paletteList(input.value, item.color_palette);
        } else if (key === 'fit') {
          item.fit = ["cover", "contain", "stretch"].includes(String(input.value).toLowerCase()) ? String(input.value).toLowerCase() : "cover";
        } else if (key === 'opacity') {
          item.opacity = Math.max(0, Math.min(1, Number(input.value) || 1));
        } else {
          item[key] = input.value;
        }
        persist();
        render();
      });
    });
  }

  function defaultImageRect(offset = 0) {
    return {
      x: 0,
      y: 0,
      w: 1000,
      h: 1000,
    };
  }

  function imageRectForDimensions(width, height, offset = 0) {
    const canvasW = Math.max(1, Number(state.data?.canvas?.width || 1000));
    const canvasH = Math.max(1, Number(state.data?.canvas?.height || 1000));
    const imageW = Math.max(0, Number(width || 0));
    const imageH = Math.max(0, Number(height || 0));
    if (!imageW || !imageH) return defaultImageRect(offset);

    const canvasAspect = canvasW / canvasH;
    const imageAspect = imageW / imageH;
    const exactCanvasSize = Math.abs(imageW - canvasW) <= 2 && Math.abs(imageH - canvasH) <= 2;
    const sameAspect = Math.abs(imageAspect - canvasAspect) <= 0.015;
    if (exactCanvasSize || sameAspect) return defaultImageRect(offset);

    if (imageAspect > canvasAspect) {
      const h = clampInt((canvasAspect / imageAspect) * 1000, 20, 1000, 1000);
      return { x: 0, y: Math.round((1000 - h) / 2), w: 1000, h };
    }
    const w = clampInt((imageAspect / canvasAspect) * 1000, 20, 1000, 1000);
    return { x: Math.round((1000 - w) / 2), y: 0, w, h: 1000 };
  }

  function imageFileDimensions(file) {
    if (!file) return Promise.resolve({ width: 0, height: 0 });
    if (typeof createImageBitmap === 'function') {
      return createImageBitmap(file)
        .then((bitmap) => {
          const result = { width: bitmap.width || 0, height: bitmap.height || 0 };
          bitmap.close?.();
          return result;
        })
        .catch(() => ({ width: 0, height: 0 }));
    }
    return new Promise((resolve) => {
      const url = URL.createObjectURL(file);
      const img = new Image();
      img.onload = () => {
        const result = { width: img.naturalWidth || 0, height: img.naturalHeight || 0 };
        URL.revokeObjectURL(url);
        resolve(result);
      };
      img.onerror = () => {
        URL.revokeObjectURL(url);
        resolve({ width: 0, height: 0 });
      };
      img.src = url;
    });
  }

  function createItem(kind, extras = {}) {
    const next = (state.data.items?.length || 0) + 1;
    const imageRect = kind === 'image' ? defaultImageRect(next - 1) : null;
    const item = {
      id: `item_${String(next).padStart(3, '0')}`,
      kind,
      label: kind === 'text' ? `Text ${next}` : (kind === 'image' ? `Image ${next}` : (kind === 'mask' ? `Mask ${next}` : `Object ${next}`)),
      text: kind === 'text' ? 'TEXT' : '',
      x: kind === 'image' ? imageRect.x : 170 + (next * 12),
      y: kind === 'image' ? imageRect.y : 150 + (next * 10),
      w: kind === 'text' ? 240 : (kind === 'image' ? imageRect.w : (kind === 'mask' ? 180 : 280)),
      h: kind === 'text' ? 110 : (kind === 'image' ? imageRect.h : (kind === 'mask' ? 160 : 240)),
      desc: kind === 'text' ? 'Readable in-frame text element with deliberate styling and placement.' : (kind === 'image' ? 'Source image layer used for i2i composition and visual reference.' : (kind === 'mask' ? 'White inpaint mask zone. Resize and place this over the area to modify.' : 'Visual subject block placed for clear silhouette and cinematic balance.')),
      color_palette: kind === 'text' ? ['#FFB65D', '#FFE4B5'] : (kind === 'mask' ? ['#FFFFFF', '#FF4D6D'] : ['#8B4513', '#1A1A2E', '#FFE4B5']),
    };
    if (kind === 'image') {
      item.image_path = '';
      item.fit = 'contain';
      item.opacity = 1;
    } else if (kind === 'mask') {
      item.shape = 'rect';
    }
    return normalizeItem({ ...item, ...extras }, next - 1);
  }

  function syncSceneFields() {
    if (!targetResolutionEntries()[state.targetResolutionKey]) {
      state.targetResolutionKey = targetResolutionEntries()[state.data.canvas?.target_resolution_key]
        ? state.data.canvas.target_resolution_key
        : "hd_720";
    }
    state.data.canvas.target_resolution_key = state.targetResolutionKey;
    sceneFields.width.value = state.data.canvas.width || 1024;
    sceneFields.height.value = state.data.canvas.height || 1024;
    sceneFields.aspect.value = state.data.canvas.aspect_label || '';
    if (sceneFields.targetResolution) {
      renderSelectOptions(sceneFields.targetResolution, targetResolutionEntries(), state.targetResolutionKey);
      sceneFields.targetResolution.value = targetResolutionEntries()[state.targetResolutionKey] ? state.targetResolutionKey : "custom";
    }
    if (sceneFields.targetResolutionNote) {
      const preset = targetResolutionEntries()[state.targetResolutionKey] || targetResolutionEntries().custom;
      const mult = currentCanvasMultiplier();
      sceneFields.targetResolutionNote.textContent = state.targetResolutionKey === "custom"
        ? "Custom: Width and Height define the total canvas. Use custom only for advanced layouts."
        : `${preset.note || ""} Workflow: ${currentWorkflowMode().label}. Canvas multiplier: ${mult.columns}x${mult.rows}.`;
    }
    sceneFields.high.value = state.data.scene.high_level_description || '';
    sceneFields.background.value = state.data.scene.background || '';
    sceneFields.aesthetics.value = state.data.scene.aesthetics || '';
    sceneFields.lighting.value = state.data.scene.lighting || '';
    sceneFields.medium.value = state.data.scene.medium || '';
    sceneFields.photo.value = state.data.scene.photo || '';
    sceneFields.palette.value = (state.data.scene.color_palette || []).join(', ');
    if (sceneFields.palettePreset) {
      renderSelectOptions(sceneFields.palettePreset, paletteEntries(), paletteKeyForColors(state.data.scene.color_palette));
      sceneFields.palettePreset.value = paletteKeyForColors(state.data.scene.color_palette);
    }
    if (sceneFields.paletteNote) {
      const key = paletteKeyForColors(state.data.scene.color_palette);
      sceneFields.paletteNote.textContent = paletteEntries()[key]?.summary || "Typed hex palette is exported to Ideogram JSON.";
    }
    state.data.reference_mode = normalizeReferenceMode(state.data.reference_mode);
    if (workflowModeEntries()[state.data.workflow_mode]) state.workflowModeKey = state.data.workflow_mode;
    state.workflowModeKey = currentWorkflowMode().key;
    const workflowSelect = root.querySelector('[data-role="workflow-mode"]');
    if (workflowSelect) workflowSelect.value = state.workflowModeKey;
    const workflowNote = root.querySelector('[data-role="workflow-mode-note"]');
    if (workflowNote) workflowNote.textContent = currentWorkflowMode().summary || "";
    state.data.i2i = normalizeI2I(state.data.i2i);
    if (i2iFields.enabled) i2iFields.enabled.checked = Boolean(state.data.i2i.enabled);
    if (i2iFields.denoise) i2iFields.denoise.value = state.data.i2i.denoise;
    if (i2iFields.step) i2iFields.step.value = state.data.i2i.low_sigma_start_step;
    setWidget(node, "i2i_enabled", Boolean(state.data.i2i.enabled));
    setWidget(node, "i2i_denoise", Number(state.data.i2i.denoise));
    setWidget(node, "low_sigma_start_step", Number(state.data.i2i.low_sigma_start_step));
    syncDenoisePresetButtons();
    syncDirectPromptFields();
    stageSize.textContent = `${state.data.canvas.width} x ${state.data.canvas.height} - ${state.data.canvas.aspect_label || 'Canvas'}`;
  }

  function deleteLayer(id) {
    const targetId = id || state.selectedId;
    if (!targetId) return;
    const index = state.data.items.findIndex((entry) => entry.id === targetId);
    if (index < 0) return;
    state.data.items.splice(index, 1);
    state.selectedId = state.data.items[Math.min(index, state.data.items.length - 1)]?.id || state.data.items[0]?.id || null;
    persist();
    render();
    footStatus.textContent = targetId ? `Deleted layer ${targetId}` : 'Layer deleted';
  }

  function maskPaintData() {
    state.data.mask_paint = normalizeMaskPaint(state.data.mask_paint || defaultData().mask_paint, defaultData().mask_paint);
    return state.data.mask_paint;
  }

  function clearMaskPaintForNewImage() {
    const current = maskPaintData();
    state.data.mask_paint = normalizeMaskPaint({
      brush_size: current.brush_size || defaultData().mask_paint.brush_size,
      strokes: [],
    }, defaultData().mask_paint);
  }

  function selectedImageItem() {
    const item = currentItem();
    return item && item.kind === "image" ? item : null;
  }

  function maskStrokesForItem(itemId) {
    const paint = maskPaintData();
    return (paint.strokes || []).filter((stroke) => {
      const target = stroke?.target || {};
      if (target.kind === "canvas_full") return true;
      return target.kind === "image_content" && target.item_id === itemId;
    });
  }

  function maskSvgMarkupForItem(item) {
    const strokes = maskStrokesForItem(item.id);
    if (!strokes.length) return "";
    return strokes.map((stroke) => {
      const points = Array.isArray(stroke.points) ? stroke.points : [];
      if (!points.length) return "";
      const shape = String(stroke.shape || "stroke").toLowerCase();
      const mode = String(stroke.mode || "paint").toLowerCase();
      const klass = mode === "erase" ? "mask-fill mask-erase" : "mask-fill";
      const xy = (point) => `${clampInt(point[0], 0, 1000, 0)},${clampInt(point[1], 0, 1000, 0)}`;
      if (shape === "rect" && points.length >= 2) {
        const a = points[0];
        const b = points[points.length - 1];
        const x = Math.min(clampInt(a[0], 0, 1000, 0), clampInt(b[0], 0, 1000, 0));
        const y = Math.min(clampInt(a[1], 0, 1000, 0), clampInt(b[1], 0, 1000, 0));
        const w = Math.abs(clampInt(b[0], 0, 1000, 0) - clampInt(a[0], 0, 1000, 0));
        const h = Math.abs(clampInt(b[1], 0, 1000, 0) - clampInt(a[1], 0, 1000, 0));
        return `<rect class="${klass}" x="${x}" y="${y}" width="${w}" height="${h}" />`;
      }
      if (shape === "lasso" && points.length >= 3) {
        return `<polygon class="${klass}" points="${points.map(xy).join(" ")}" />`;
      }
      if (points.length === 1) {
        const radius = Math.max(8, clampInt(stroke.size, 1, 240, maskPaintData().brush_size || 48) / 2);
        return `<circle class="${mode === "erase" ? "mask-erase" : "mask-fill"}" cx="${clampInt(points[0][0], 0, 1000, 0)}" cy="${clampInt(points[0][1], 0, 1000, 0)}" r="${radius}" />`;
      }
      const d = points.map((point, index) => `${index ? "L" : "M"}${xy(point)}`).join(" ");
      return `<path class="mask-stroke ${mode === "erase" ? "mask-erase" : ""}" d="${d}" />`;
    }).join("");
  }

  function imageContentRectForElement(itemEl, item, imgEl = null) {
    const rect = itemEl.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const fit = String(item?.fit || "contain").toLowerCase();
    if (fit !== "contain") return { x: 0, y: 0, w: width, h: height };
    const naturalW = Number(imgEl?.naturalWidth || 0);
    const naturalH = Number(imgEl?.naturalHeight || 0);
    if (!naturalW || !naturalH) return { x: 0, y: 0, w: width, h: height };
    const imageAspect = naturalW / naturalH;
    const boxAspect = width / height;
    if (imageAspect > boxAspect) {
      const contentH = width / imageAspect;
      return { x: 0, y: (height - contentH) / 2, w: width, h: contentH };
    }
    const contentW = height * imageAspect;
    return { x: (width - contentW) / 2, y: 0, w: contentW, h: height };
  }

  function imageContentTargetForItem(item, itemEl, imgEl = null) {
    let x = clampInt(item.x, 0, 999, 0);
    let y = clampInt(item.y, 0, 999, 0);
    let w = clampInt(item.w, 1, Math.max(1, 1000 - x), Math.max(1, 1000 - x));
    let h = clampInt(item.h, 1, Math.max(1, 1000 - y), Math.max(1, 1000 - y));
    const fit = String(item.fit || "contain").toLowerCase();
    if (fit === "contain") {
      const naturalW = Number(imgEl?.naturalWidth || 0);
      const naturalH = Number(imgEl?.naturalHeight || 0);
      if (naturalW > 0 && naturalH > 0 && w > 0 && h > 0) {
        const imageAspect = naturalW / naturalH;
        const boxAspect = w / h;
        if (imageAspect > boxAspect) {
          const contentH = w / imageAspect;
          y += Math.round((h - contentH) / 2);
          h = Math.round(contentH);
        } else {
          const contentW = h * imageAspect;
          x += Math.round((w - contentW) / 2);
          w = Math.round(contentW);
        }
      }
    }
    x = clampInt(x, 0, 999, 0);
    y = clampInt(y, 0, 999, 0);
    w = clampInt(w, 1, Math.max(1, 1000 - x), Math.max(1, 1000 - x));
    h = clampInt(h, 1, Math.max(1, 1000 - y), Math.max(1, 1000 - y));
    return { kind: "image_content", item_id: item.id, x, y, w, h, fit };
  }

  function maskPointFromPointer(event, itemEl, item, imgEl = null) {
    const rect = itemEl.getBoundingClientRect();
    const content = imageContentRectForElement(itemEl, item, imgEl);
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    if (localX < content.x || localY < content.y || localX > content.x + content.w || localY > content.y + content.h) return null;
    return [
      clampInt(((localX - content.x) / Math.max(1, content.w)) * 1000, 0, 1000, 0),
      clampInt(((localY - content.y) / Math.max(1, content.h)) * 1000, 0, 1000, 0),
    ];
  }

  function maskPointFromCanvasPointer(event, canvas) {
    const rect = canvas?.getBoundingClientRect?.();
    if (!rect || rect.width <= 0 || rect.height <= 0) return null;
    const localX = event.clientX - rect.left;
    const localY = event.clientY - rect.top;
    if (localX < 0 || localY < 0 || localX > rect.width || localY > rect.height) return null;
    return [
      clampInt((localX / Math.max(1, rect.width)) * 1000, 0, 1000, 0),
      clampInt((localY / Math.max(1, rect.height)) * 1000, 0, 1000, 0),
    ];
  }

  function drawStrokePath(ctx, points, content, size, mode, shape = "stroke") {
    if (!points?.length) return;
    const px = (point) => [content.x + (point[0] / 1000) * content.w, content.y + (point[1] / 1000) * content.h];
    const radius = Math.max(1, size / 2);
    const lineWidth = Math.max(1, size);
    const shapeMode = String(shape || "stroke").toLowerCase();
    const drawCap = (point) => {
      const [x, y] = px(point);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fill();
    };
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = lineWidth;
    ctx.globalCompositeOperation = mode === "erase" ? "destination-out" : "source-over";
    ctx.strokeStyle = "rgba(255, 28, 62, 0.96)";
    ctx.fillStyle = "rgba(255, 28, 62, 0.62)";
    if (shapeMode === "rect" && points.length >= 2) {
      const a = px(points[0]);
      const b = px(points[points.length - 1]);
      const x = Math.min(a[0], b[0]);
      const y = Math.min(a[1], b[1]);
      const w = Math.abs(b[0] - a[0]);
      const h = Math.abs(b[1] - a[1]);
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);
      ctx.restore();
      return;
    }
    if (shapeMode === "lasso" && points.length >= 3) {
      ctx.beginPath();
      const first = px(points[0]);
      ctx.moveTo(first[0], first[1]);
      for (let i = 1; i < points.length; i++) {
        const p = px(points[i]);
        ctx.lineTo(p[0], p[1]);
      }
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
      return;
    }
    ctx.fillStyle = "rgba(255, 28, 62, 0.82)";
    if (points.length === 1) {
      drawCap(points[0]);
    } else {
      for (let i = 1; i < points.length; i++) {
        const a = px(points[i - 1]);
        const b = px(points[i]);
        ctx.beginPath();
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
        ctx.stroke();
        if (i % 8 === 0) drawCap(points[i]);
      }
      drawCap(points[0]);
      drawCap(points[points.length - 1]);
    }
    ctx.restore();
  }

  function maskDebug(eventName, payload = {}) {
    const safe = {};
    Object.entries(payload || {}).forEach(([key, value]) => {
      if (value == null) safe[key] = value;
      else if (typeof value === "number" || typeof value === "string" || typeof value === "boolean") safe[key] = value;
      else if (Array.isArray(value)) safe[key] = value.map((entry) => typeof entry === "number" ? Number(entry.toFixed?.(2) ?? entry) : entry).slice(0, 8);
      else if (value instanceof DOMRect) safe[key] = { x: Math.round(value.x), y: Math.round(value.y), w: Math.round(value.width), h: Math.round(value.height) };
      else safe[key] = String(value);
    });
    const line = `${new Date().toLocaleTimeString()} ${eventName} ${JSON.stringify(safe)}`;
    state.maskDebugLog = [line, ...(state.maskDebugLog || [])].slice(0, 18);
    if (maskDebugHost) maskDebugHost.textContent = state.maskDebugLog.join("\n");
  }

  function renderMaskCanvas(canvas, item, itemEl) {
    if (!canvas || !itemEl || !item) return;
    const itemRect = itemEl.getBoundingClientRect();
    const imgEl = itemEl.querySelector("img.iamccs-isf-item-image");
    const content = imageContentRectForElement(itemEl, item, imgEl);
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(content.w));
    const height = Math.max(1, Math.round(content.h));
    canvas.style.left = `${content.x}px`;
    canvas.style.top = `${content.y}px`;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.style.right = "auto";
    canvas.style.bottom = "auto";
    if (canvas.width !== Math.round(width * dpr) || canvas.height !== Math.round(height * dpr)) {
      canvas.width = Math.round(width * dpr);
      canvas.height = Math.round(height * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    const canvasContent = { x: 0, y: 0, w: width, h: height };
    maskStrokesForItem(item.id).forEach((stroke) => {
      const scale = (width + height) / 2000;
      const size = Math.max(1, Number(stroke.size || maskPaintData().brush_size || 48) * Math.max(0.1, scale));
      drawStrokePath(ctx, stroke.points || [], canvasContent, size, stroke.mode, stroke.shape);
    });
    canvas.dataset.maskBitmapRect = `${Math.round(content.x)},${Math.round(content.y)},${width},${height}`;
    canvas.dataset.maskItemRect = `${Math.round(itemRect.width)}x${Math.round(itemRect.height)}`;
    const svg = itemEl.querySelector(`[data-mask-svg="${CSS.escape(item.id)}"]`);
    if (svg) {
      svg.style.left = `${content.x}px`;
      svg.style.top = `${content.y}px`;
      svg.style.width = `${width}px`;
      svg.style.height = `${height}px`;
      svg.innerHTML = maskSvgMarkupForItem(item);
    }
  }

  function renderMaskTechnicalPreview(item = selectedImageItem()) {
    if (!maskPreviewCanvas) {
      if (!item) return null;
      const strokes = maskStrokesForItem(item.id);
      if (!strokes.length) return "0.00";
      let estimated = 0;
      strokes.forEach((stroke) => {
        const points = Array.isArray(stroke.points) ? stroke.points.length : 0;
        const radius = Math.max(1, Number(stroke.size || maskPaintData().brush_size || 48) / 2);
        estimated += points * Math.PI * radius * radius;
      });
      return ((estimated / 1000000) * 100).toFixed(2);
    }
    const canvas = maskPreviewCanvas;
    const cssW = Math.max(1, Math.round(canvas.clientWidth || 240));
    const canvasInfo = state.data?.canvas || {};
    const aspect = Math.max(0.1, Math.min(10, Number(canvasInfo.width || 1) / Math.max(1, Number(canvasInfo.height || 1))));
    const cssH = Math.max(48, Math.round(cssW / aspect));
    canvas.style.height = `${cssH}px`;
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    ctx.fillStyle = "#05080a";
    ctx.fillRect(0, 0, cssW, cssH);
    if (!item) {
      ctx.fillStyle = "#ffd5dd";
      ctx.font = "11px Consolas, monospace";
      ctx.fillText("No selected image mask target", 8, 22);
      return null;
    }
    const content = { x: 0, y: 0, w: cssW, h: cssH };
    maskStrokesForItem(item.id).forEach((stroke) => {
      const size = Math.max(1, Number(stroke.size || maskPaintData().brush_size || 48) * 0.16);
      drawStrokePath(ctx, stroke.points || [], content, size, stroke.mode, stroke.shape);
    });
    try {
      const data = ctx.getImageData(0, 0, cssW, cssH).data;
      let painted = 0;
      for (let i = 0; i < data.length; i += 4) {
        if (data[i] > 150 && data[i + 1] < 130 && data[i + 2] < 160) painted += 1;
      }
      const pct = ((painted / Math.max(1, cssW * cssH)) * 100).toFixed(2);
      return pct;
    } catch {
      return null;
    }
  }

  function syncMaskUi() {
    const paint = maskPaintData();
    if (maskBrushSizeInput && document.activeElement !== maskBrushSizeInput) maskBrushSizeInput.value = String(paint.brush_size || 48);
    root.querySelectorAll("[data-mask-tool]").forEach((button) => {
      button.classList.toggle("active", button.dataset.maskTool === state.paintTool);
      button.setAttribute("aria-pressed", button.dataset.maskTool === state.paintTool ? "true" : "false");
    });
    const item = selectedImageItem();
    const count = item ? maskStrokesForItem(item.id).length : (paint.strokes || []).length;
    if (maskStatusHost) {
      maskStatusHost.textContent = item
        ? `Mask target: ${item.label || item.id}. Strokes: ${count}.`
        : `Select an image layer, choose Shape, then drag a freeform filled mask on the image. Shift-drag makes a rectangle. Total strokes: ${count}.`;
    }
    const coverage = renderMaskTechnicalPreview(item);
    if (coverage != null && maskStatusHost && item) maskStatusHost.textContent += ` Coverage: ${coverage}%.`;
    if (maskDebugHost) maskDebugHost.textContent = (state.maskDebugLog || []).join("\n") || "Mask debug ready.";
  }

  function clearCurrentImageMask(anchor = null) {
    const paint = maskPaintData();
    const item = selectedImageItem();
    const before = (paint.strokes || []).length;
    if (item) {
      paint.strokes = (paint.strokes || []).filter((stroke) => {
        const target = stroke?.target || {};
        if (target.kind === "canvas_full") return false;
        return target.item_id !== item.id;
      });
    } else {
      paint.strokes = [];
    }
    state.data.mask_paint = paint;
    persist();
    render();
    const removed = before - (paint.strokes || []).length;
    showToast(item ? `Cleared ${removed} mask stroke(s) for selected image` : "Cleared all painted mask strokes", { anchor, tone: "success" });
  }

  function bindMaskCanvas(canvas, item, itemEl) {
    if (!canvas || !item || item.kind !== "image") return;
    const imgEl = itemEl.querySelector("img.iamccs-isf-item-image");
    let renderQueued = false;
    const queueMaskRender = () => {
      if (renderQueued) return;
      renderQueued = true;
      requestAnimationFrame(() => {
        renderQueued = false;
        renderMaskCanvas(canvas, item, itemEl);
        renderMaskTechnicalPreview(item);
      });
    };
    renderMaskCanvas(canvas, item, itemEl);
    imgEl?.addEventListener("load", () => { renderMaskCanvas(canvas, item, itemEl); renderMaskTechnicalPreview(item); }, { once: true });

    const stopEvent = (event) => {
      event?.preventDefault?.();
      event?.stopPropagation?.();
      event?.stopImmediatePropagation?.();
    };
    const appendPointFromEvent = (event, drawing, source = "move") => {
      let events = [event];
      try {
        if (typeof event.getCoalescedEvents === "function") {
          const coalesced = event.getCoalescedEvents();
          if (Array.isArray(coalesced) && coalesced.length) events = coalesced;
        }
      } catch (error) {
        maskDebug("coalesced events failed", { item: item.id, error: error?.message || String(error) });
      }
      let appended = 0;
      for (const single of events) {
        const point = maskPointFromCanvasPointer(single, canvas);
        if (!point) {
          if (source !== "coalesced") maskDebug(`${source} outside bitmap canvas`, { item: item.id, x: Math.round(single.clientX || 0), y: Math.round(single.clientY || 0), bitmap: canvas.dataset.maskBitmapRect || "none" });
          continue;
        }
        const points = drawing.stroke.points || [];
        const last = points[points.length - 1];
        if (drawing.stroke.shape === "rect") {
          drawing.stroke.points = [drawing.startPoint || points[0] || point, point];
          appended += 1;
          continue;
        }
        if (last && Math.hypot(point[0] - last[0], point[1] - last[1]) < (drawing.stroke.shape === "lasso" ? 4 : 1.25)) continue;
        points.push(point);
        drawing.stroke.points = points;
        appended += 1;
      }
      return appended;
    };
    const detachGlobal = () => {
      window.removeEventListener("pointermove", globalMove, true);
      window.removeEventListener("pointerup", globalEnd, true);
      window.removeEventListener("pointercancel", globalEnd, true);
      window.removeEventListener("blur", globalBlur, true);
    };
    const finish = (event, reason = "stroke end") => {
      const drawing = state.paintDrawing;
      if (!drawing || drawing.itemId !== item.id) return;
      if (event && drawing.pointerId != null && event.pointerId != null && event.pointerId !== drawing.pointerId) return;
      stopEvent(event);
      detachGlobal();
      const pointCount = drawing.stroke?.points?.length || 0;
      state.paintDrawing = null;
      try { canvas.releasePointerCapture?.(drawing.pointerId); } catch (error) { maskDebug("release failed", { error: error?.message || String(error) }); }
      renderMaskCanvas(canvas, item, itemEl);
      const coverage = renderMaskTechnicalPreview(item);
      writeData(node, state.data);
      persist();
      const itemStrokes = maskStrokesForItem(item.id);
      const totalPoints = itemStrokes.reduce((sum, stroke) => sum + ((stroke.points || []).length), 0);
      const jsonChars = JSON.stringify(state.data || {}).length;
      maskDebug(reason, { item: item.id, points: pointCount, strokes: itemStrokes.length, total_points: totalPoints, json_chars: jsonChars, coverage });
      syncMaskUi();
      footStatus.textContent = `Paint mask saved for ${item.label || item.id}: ${maskStrokesForItem(item.id).length} stroke(s)`;
    };
    const globalMove = (event) => {
      const drawing = state.paintDrawing;
      if (!drawing || drawing.itemId !== item.id) return;
      if (drawing.pointerId != null && event.pointerId != null && event.pointerId !== drawing.pointerId) { maskDebug("move ignored pointer", { expected: drawing.pointerId, got: event.pointerId }); return; }
      if (event.buttons != null && event.buttons === 0) { finish(event, "stroke end buttons=0"); return; }
      stopEvent(event);
      const appended = appendPointFromEvent(event, drawing);
      if (appended > 0) {
        const points = drawing.stroke.points || [];
        queueMaskRender();
        const svg = itemEl.querySelector(`[data-mask-svg="${CSS.escape(item.id)}"]`);
        if (svg) svg.innerHTML = maskSvgMarkupForItem(item);
        if (points.length === 2 || points.length % 20 === 0) maskDebug("stroke move", { item: item.id, points: points.length, appended, shape: drawing.stroke.shape || "stroke", last: points[points.length - 1] });
      }
    };
    const globalEnd = (event) => finish(event, event?.type === "pointercancel" ? "stroke cancel" : "stroke end");
    const globalBlur = () => finish(null, "window blur end");

    const start = (event) => {
      if (!["brush", "erase", "shape"].includes(state.paintTool)) { maskDebug("down ignored", { tool: state.paintTool }); return; }
      if (event.button != null && event.button !== 0) { maskDebug("down ignored", { button: event.button }); return; }
      if (!guardImageMaskPaint(canvas)) {
        stopEvent(event);
        return;
      }
      renderMaskCanvas(canvas, item, itemEl);
      const point = maskPointFromCanvasPointer(event, canvas);
      if (!point) { maskDebug("down outside bitmap canvas", { x: Math.round(event.clientX || 0), y: Math.round(event.clientY || 0), bitmap: canvas.dataset.maskBitmapRect || "none" }); return; }
      stopEvent(event);
      detachGlobal();
      state.drag = null;
      window.removeEventListener('pointermove', onPointerMove);
      state.selectedId = item.id;
      const paint = maskPaintData();
      const stroke = {
        mode: state.paintTool === "erase" ? "erase" : "paint",
        shape: state.paintTool === "shape" ? (event.shiftKey ? "rect" : "lasso") : "stroke",
        size: clampInt(paint.brush_size, 1, 240, 48),
        target: imageContentTargetForItem(item, itemEl, imgEl),
        points: [point],
      };
      paint.strokes.push(stroke);
      state.data.mask_paint = paint;
      state.paintDrawing = { pointerId: event.pointerId, itemId: item.id, stroke, startPoint: point, startedAt: performance.now() };
      try { canvas.setPointerCapture?.(event.pointerId); } catch (error) { maskDebug("capture failed", { error: error?.message || String(error) }); }
      window.addEventListener("pointermove", globalMove, true);
      window.addEventListener("pointerup", globalEnd, true);
      window.addEventListener("pointercancel", globalEnd, true);
      window.addEventListener("blur", globalBlur, true);
      renderMaskCanvas(canvas, item, itemEl);
      const coverage = renderMaskTechnicalPreview(item);
      maskDebug("stroke start", { item: item.id, pointerId: event.pointerId, point, brush: stroke.size, target: `${stroke.target.x},${stroke.target.y},${stroke.target.w},${stroke.target.h}`, bitmap: canvas.dataset.maskBitmapRect || "none", coverage });
      syncMaskUi();
    };
    canvas.addEventListener("pointerdown", start, true);
    canvas.addEventListener("pointermove", globalMove, true);
    canvas.addEventListener("pointerup", globalEnd, true);
    canvas.addEventListener("pointercancel", globalEnd, true);
    canvas.addEventListener("lostpointercapture", (event) => {
      if (state.paintDrawing?.itemId === item.id) {
        maskDebug("lost capture ignored", { item: item.id, pointerId: event?.pointerId ?? null });
      }
    }, true);
  }

  function syncItemFields() {
    const item = currentItem();
    const disabled = !item;
    Object.values(itemFields).forEach((input) => { input.disabled = disabled; });
    if (!item) {
      Object.values(itemFields).forEach((input) => { input.value = ''; });
      return;
    }
    itemFields.label.value = item.label || '';
    itemFields.text.value = item.text || '';
    itemFields.text.disabled = item.kind !== 'text';
    itemFields.desc.value = item.desc || '';
    itemFields.palette.value = (item.color_palette || []).join(', ');
    if (itemFields.palettePreset) {
      renderSelectOptions(itemFields.palettePreset, paletteEntries(), paletteKeyForColors(item.color_palette));
      itemFields.palettePreset.value = paletteKeyForColors(item.color_palette);
      itemFields.palettePreset.disabled = disabled;
    }
    if (itemFields.paletteNote) {
      const key = paletteKeyForColors(item.color_palette);
      itemFields.paletteNote.textContent = paletteEntries()[key]?.summary || "Typed hex palette is exported on this layer.";
    }
    itemFields.image_path.value = item.image_path || '';
    itemFields.fit.value = item.fit || 'cover';
    itemFields.opacity.value = item.opacity ?? 1;
    [itemFields.image_path, itemFields.fit, itemFields.opacity].forEach((input) => { input.disabled = item.kind !== 'image'; });
    itemFields.x.value = item.x;
    itemFields.y.value = item.y;
    itemFields.w.value = item.w;
    itemFields.h.value = item.h;
    syncMaskUi();
  }

  function renderLayerList() {
    const host = layerListHost;
    if (!host) return;
    host.innerHTML = '';
    state.data.items.forEach((item) => {
      const card = document.createElement('div');
      card.dataset.itemId = item.id;
      card.className = `iamccs-isf-card ${item.id === state.selectedId ? 'selected' : ''}`;
      card.innerHTML = `
        <div class="iamccs-isf-card-title">
          <span>${item.label || item.id}</span>
          <span class="iamccs-isf-chip ${item.kind === 'text' ? 'text' : (item.kind === 'image' ? 'image' : '')}">${item.kind}</span>
          <button class="iamccs-isf-delete" type="button" title="Delete layer" aria-label="Delete layer" data-delete-layer="${item.id}">X</button>
        </div>
        <p>${item.desc || ''}</p>
        <div class="iamccs-isf-palette">${(item.color_palette || []).map((color) => `<span class="iamccs-isf-swatch" style="background:${color}"></span>`).join('')}</div>
      `;
      card.querySelector('[data-delete-layer]')?.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        deleteLayer(item.id);
      });
      card.addEventListener('click', () => {
        state.selectedId = item.id;
        render();
        syncSelectionState();
      });
      host.appendChild(card);
    });
  }

  function itemStackAtPoint(event) {
    return Array.from(document.elementsFromPoint(event.clientX, event.clientY) || [])
      .filter((el) => el?.classList?.contains('iamccs-isf-item') && el.dataset.itemId)
      .map((el) => el.dataset.itemId)
      .filter((id, index, ids) => ids.indexOf(id) === index);
  }

  function cycleSelectionAtPoint(event) {
    const ids = itemStackAtPoint(event);
    if (!ids.length) return false;
    const currentIndex = ids.indexOf(state.selectedId);
    const nextId = ids[(currentIndex + 1 + ids.length) % ids.length];
    state.selectedId = nextId;
    render();
    syncSelectionState();
    footStatus.textContent = ids.length > 1
      ? `Selected ${nextId}; Alt/Ctrl-click cycles ${ids.length} overlapped layers`
      : `Selected ${nextId}`;
    return true;
  }

  function startDrag(event, item, mode) {
    const rect = artboard.getBoundingClientRect();
    state.drag = {
      mode,
      id: item.id,
      startX: event.clientX,
      startY: event.clientY,
      origin: { x: item.x, y: item.y, w: item.w, h: item.h },
      rect,
    };
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', stopDrag, { once: true });
  }

  function onPointerMove(event) {
    if (!state.drag) return;
    const item = state.data.items.find((entry) => entry.id === state.drag.id);
    if (!item) return;
    const dx = ((event.clientX - state.drag.startX) / state.drag.rect.width) * 1000;
    const dy = ((event.clientY - state.drag.startY) / state.drag.rect.height) * 1000;
    if (state.drag.mode === 'move') {
      item.x = clampInt(state.drag.origin.x + dx, 0, 1000 - item.w, item.x);
      item.y = clampInt(state.drag.origin.y + dy, 0, 1000 - item.h, item.y);
    } else {
      item.w = clampInt(state.drag.origin.w + dx, 20, 1000 - item.x, item.w);
      item.h = clampInt(state.drag.origin.h + dy, 20, 1000 - item.y, item.h);
    }
    persist();
    render();
  }

  function stopDrag() {
    state.drag = null;
    window.removeEventListener('pointermove', onPointerMove);
  }



  function renderImageEditorMaskCanvas(canvas, item) {
    if (!canvas || !item) return;
    canvas.style.left = "0px";
    canvas.style.top = "0px";
    canvas.style.width = "100%";
    canvas.style.height = "100%";
    const rect = canvas.getBoundingClientRect();
    const cssW = Math.max(1, Math.round(rect.width || canvas.clientWidth || 1));
    const cssH = Math.max(1, Math.round(rect.height || canvas.clientHeight || 1));
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    if (canvas.width !== Math.round(cssW * dpr) || canvas.height !== Math.round(cssH * dpr)) {
      canvas.width = Math.round(cssW * dpr);
      canvas.height = Math.round(cssH * dpr);
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);
    const content = { x: 0, y: 0, w: cssW, h: cssH };
    maskStrokesForItem(item.id).forEach((stroke) => {
      const size = Math.max(1, Number(stroke.size || maskPaintData().brush_size || 48) * ((cssW + cssH) / 2000));
      drawStrokePath(ctx, stroke.points || [], content, size, stroke.mode, stroke.shape);
    });
    if (state.imageEditorDrawing?.itemId === item.id && state.imageEditorDrawing.stroke) {
      const stroke = state.imageEditorDrawing.stroke;
      const size = Math.max(1, Number(stroke.size || maskPaintData().brush_size || 48) * ((cssW + cssH) / 2000));
      drawStrokePath(ctx, stroke.points || [], content, size, stroke.mode, stroke.shape);
    }
    const svg = canvas.parentElement?.querySelector?.(`[data-mask-svg="${CSS.escape(item.id)}"]`);
    if (svg) {
      svg.style.left = "0px";
      svg.style.top = "0px";
      svg.style.width = "100%";
      svg.style.height = "100%";
      svg.innerHTML = maskSvgMarkupForItem(item);
    }
  }

  function bindImageEditorMaskCanvas(canvas, item) {
    if (!canvas || !item || canvas.dataset.imageEditorBound === "1") return;
    canvas.dataset.imageEditorBound = "1";
    const stopEvent = (event) => {
      event?.preventDefault?.();
      event?.stopPropagation?.();
      event?.stopImmediatePropagation?.();
    };
    const pointFromEvent = (event) => {
      const rect = canvas.getBoundingClientRect();
      if (!rect || rect.width <= 0 || rect.height <= 0) return null;
      const x = ((event.clientX - rect.left) / rect.width) * 1000;
      const y = ((event.clientY - rect.top) / rect.height) * 1000;
      if (x < 0 || y < 0 || x > 1000 || y > 1000) return null;
      return [clampInt(x, 0, 1000, 0), clampInt(y, 0, 1000, 0)];
    };
    const appendPoint = (event) => {
      const drawing = state.imageEditorDrawing;
      if (!drawing || drawing.itemId !== item.id) return 0;
      const point = pointFromEvent(event);
      if (!point) return 0;
      const stroke = drawing.stroke;
      if (stroke.shape === "rect") {
        stroke.points = [drawing.startPoint, point];
        return 1;
      }
      const points = stroke.points || [];
      const last = points[points.length - 1];
      const minDist = stroke.shape === "lasso" ? 3 : 1.25;
      if (last && Math.hypot(point[0] - last[0], point[1] - last[1]) < minDist) return 0;
      points.push(point);
      stroke.points = points;
      return 1;
    };
    const detach = () => {
      window.removeEventListener("pointermove", onMove, true);
      window.removeEventListener("pointerup", onEnd, true);
      window.removeEventListener("pointercancel", onEnd, true);
      window.removeEventListener("blur", onBlur, true);
    };
    const onMove = (event) => {
      const drawing = state.imageEditorDrawing;
      if (!drawing || drawing.itemId !== item.id) return;
      if (event.pointerId != null && drawing.pointerId != null && event.pointerId !== drawing.pointerId) return;
      stopEvent(event);
      const added = appendPoint(event);
      if (added) renderImageEditorMaskCanvas(canvas, item);
    };
    const onEnd = (event) => {
      const drawing = state.imageEditorDrawing;
      if (!drawing || drawing.itemId !== item.id) return;
      if (event && event.pointerId != null && drawing.pointerId != null && event.pointerId !== drawing.pointerId) return;
      stopEvent(event);
      detach();
      const stroke = drawing.stroke;
      state.imageEditorDrawing = null;
      const points = stroke.points || [];
      if (points.length && (stroke.shape !== "lasso" || points.length >= 3) && (stroke.shape !== "rect" || points.length >= 2)) {
        const paint = maskPaintData();
        paint.strokes = Array.isArray(paint.strokes) ? paint.strokes : [];
        paint.strokes.push(stroke);
        state.data.mask_paint = paint;
        writeData(node, state.data);
        persist();
        maskDebug("image editor mask commit", { item: item.id, shape: stroke.shape, points: points.length, strokes: paint.strokes.length });
      } else {
        maskDebug("image editor mask discarded", { item: item.id, shape: stroke.shape, points: points.length });
      }
      renderImageEditorMaskCanvas(canvas, item);
      renderMaskTechnicalPreview(item);
      syncMaskUi();
    };
    const onBlur = () => onEnd(null);
    const onDown = (event) => {
      if (!["brush", "erase", "shape"].includes(state.paintTool)) return;
      if (event.button != null && event.button !== 0) return;
      if (!guardImageMaskPaint(canvas)) {
        stopEvent(event);
        return;
      }
      const point = pointFromEvent(event);
      if (!point) return;
      stopEvent(event);
      state.drag = null;
      window.removeEventListener('pointermove', onPointerMove);
      state.selectedId = item.id;
      const paint = maskPaintData();
      const stroke = {
        mode: state.paintTool === "erase" ? "erase" : "paint",
        shape: state.paintTool === "shape" ? (event.shiftKey ? "rect" : "lasso") : "stroke",
        size: clampInt(paint.brush_size, 1, 240, 48),
        target: { kind: "canvas_full", x: 0, y: 0, w: 1000, h: 1000 },
        points: [point],
      };
      state.imageEditorDrawing = { pointerId: event.pointerId, itemId: item.id, startPoint: point, stroke };
      window.addEventListener("pointermove", onMove, true);
      window.addEventListener("pointerup", onEnd, true);
      window.addEventListener("pointercancel", onEnd, true);
      window.addEventListener("blur", onBlur, true);
      renderImageEditorMaskCanvas(canvas, item);
      maskDebug("image editor mask start", { item: item.id, tool: state.paintTool, shape: stroke.shape, point });
      footStatus.textContent = `Image Editor mask drawing: ${stroke.shape}`;
    };
    canvas.addEventListener("pointerdown", onDown, true);
    canvas.addEventListener("contextmenu", stopEvent, true);
    canvas.addEventListener("wheel", stopEvent, true);
    renderImageEditorMaskCanvas(canvas, item);
  }

  function renderImageEditor() {
    if (!boardPreview) return;
    const rawItems = Array.isArray(state.data.items) ? state.data.items : [];
    let items = rawItems.map((item, index) => normalizeItem(item, index));
    let imageItem = items.find((item) => item.kind === "image");
    if (!imageItem) {
      imageItem = normalizeItem({
        id: "source_image_full_canvas",
        kind: "image",
        label: "SOURCE IMAGE - full canvas",
        x: 0,
        y: 0,
        w: 1000,
        h: 1000,
        desc: "Source image used as the i2i/inpaint guide.",
        image_path: "",
        fit: "stretch",
        opacity: 1,
        color_palette: state.data.scene?.color_palette || ["#123A5A", "#2F6F9E", "#C4A15D", "#7C6B55"],
      }, 0);
      items = [imageItem];
      state.data.items = items;
      state.selectedId = imageItem.id;
      persist();
    } else if (state.selectedId !== imageItem.id) {
      state.selectedId = imageItem.id;
    }
    if (imageItem) {
      imageItem.x = 0;
      imageItem.y = 0;
      imageItem.w = 1000;
      imageItem.h = 1000;
      imageItem.fit = "stretch";
      state.data.items = [imageItem];
      const paint = maskPaintData();
      paint.strokes = (paint.strokes || []).map((stroke) => {
        if (!stroke || stroke?.target?.item_id !== imageItem.id) return stroke;
        return {
          ...stroke,
          target: { kind: "canvas_full", x: 0, y: 0, w: 1000, h: 1000 },
        };
      });
      state.data.mask_paint = paint;
      writeData(node, state.data);
    }
    const canvas = state.data.canvas || {};
    const canvasW = Math.max(1, Number(canvas.width || 1024));
    const canvasH = Math.max(1, Number(canvas.height || 1024));
    const aspect = canvasW / canvasH;
    const hasImage = Boolean(cleanText(imageItem.image_path));
    const src = hasImage ? imageViewUrl(imageItem.image_path) : "";
    boardPreview.innerHTML = `
      <div class="iamccs-isf-image-editor" data-artboard>
        <div class="iamccs-isf-image-editor-meta">
          <span>Image Editor / i2i Inpaint - ${escapeXml(canvasW)} x ${escapeXml(canvasH)}</span>
          <span>${hasImage ? escapeXml(imageItem.label || imageItem.image_path) : "Load one source image"}</span>
        </div>
        ${hasImage ? `
          <div class="iamccs-isf-image-editor-frame mask-paint-active" data-item-id="${escapeXml(imageItem.id)}" style="--iamccs-image-editor-aspect:${aspect};">
            <img class="iamccs-isf-item-image" src="${src}" draggable="false" style="object-fit:${escapeXml((imageItem.fit || "contain") === "stretch" ? "fill" : (imageItem.fit || "contain"))};opacity:${Number(imageItem.opacity ?? 1)}" />
            <svg class="iamccs-isf-mask-svg" data-mask-svg="${escapeXml(imageItem.id)}" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">${maskSvgMarkupForItem(imageItem)}</svg>
            <canvas class="iamccs-isf-mask-canvas" data-mask-canvas="${escapeXml(imageItem.id)}"></canvas>
          </div>` : `
          <div class="iamccs-isf-image-editor-empty">
            <div>
              <div>Image Editor is active.</div>
              <div>Use Add Image to load the source frame, then choose Shape or Brush in Image Mask Paint.</div>
              <button class="iamccs-isf-btn primary" type="button" data-editor-load-image>Load Image</button>
            </div>
          </div>`}
      </div>`;
    artboard = boardPreview.querySelector('[data-artboard]') || boardPreview;
    if (hasImage) {
      const frame = boardPreview.querySelector('[data-item-id]');
      const maskCanvas = frame?.querySelector('[data-mask-canvas]');
      if (maskCanvas) bindImageEditorMaskCanvas(maskCanvas, imageItem);
      frame?.addEventListener('pointerdown', (event) => {
        if (event.target?.closest?.('[data-mask-canvas]')) return;
        event.preventDefault();
        event.stopPropagation();
        state.selectedId = imageItem.id;
        syncItemFields();
        renderLayerList();
      });
    } else {
      boardPreview.querySelector('[data-editor-load-image]')?.addEventListener('click', (event) => {
        event.preventDefault();
        event.stopPropagation();
        imageInput?.click();
      });
    }
    footStatus.textContent = hasImage
      ? `Image Editor active: mask target is ${imageItem.label || imageItem.id}`
      : 'Image Editor active: load a source image to paint a mask';
    syncMaskUi();
  }

  function renderArtboard() {
    if (!boardPreview) return;
    if (state.workflowModeKey === "image_refine" || state.data?.workflow_mode === "image_refine") {
      renderImageEditor();
      return;
    }
    const rawItems = Array.isArray(state.data.items) ? state.data.items : [];
    const items = rawItems.map((item, index) => normalizeItem(item, index));
    state.data.items = items;
    const canvas = state.data.canvas || {};
    const canvasW = Math.max(1, Number(canvas.width || 1024));
    const canvasH = Math.max(1, Number(canvas.height || 1024));
    const aspect = canvasW / canvasH;
    const visibleItems = items.filter((item) => item?.kind !== "mask");
    const meta = `${canvasW} x ${canvasH} - ${canvas.aspect_label || "Canvas"} - ${visibleItems.length} layer${visibleItems.length === 1 ? "" : "s"}`;
    const showResultBg = Boolean(state.showResultImage && state.resultOverlayVisible && cleanText(state.resultBgUrl));
    const resultOpacity = Math.max(0, Math.min(1, Number(state.resultOverlayOpacity) || 0));
    const boxFillOpacity = showResultBg ? Math.max(0.08, 0.82 - (resultOpacity * 0.74)) : 0.82;
    const boxHeadOpacity = showResultBg ? Math.max(0.26, 0.86 - (resultOpacity * 0.58)) : 0.86;
    const boxBodyOpacity = showResultBg ? Math.max(0.04, 0.22 - (resultOpacity * 0.18)) : 0.22;
    const resultBgHtml = showResultBg
      ? `<div class="iamccs-isf-result-bg" aria-hidden="true" style="background-image:url('${escapeXml(state.resultBgUrl)}');--iamccs-result-opacity:${resultOpacity};"></div>`
      : "";
    const itemHtml = visibleItems.map((item, index) => {
      const kind = cleanText(item.kind || "obj") || "obj";
      const label = cleanText(item.label || item.id || `Layer ${index + 1}`);
      const body = cleanText(kind === "text" ? (item.text || item.desc) : item.desc) || (kind === "image" ? "Image reference layer" : "Write the prompt for this visual box.");
      const left = clampInt(Number(item.x), 0, 980, 0) / 10;
      const top = clampInt(Number(item.y), 0, 980, 0) / 10;
      const width = clampInt(Number(item.w), 40, 1000, 220) / 10;
      const height = clampInt(Number(item.h), 40, 1000, 160) / 10;
      const z = item.id === state.selectedId ? 80 : (10 + index);
      const isImage = kind === "image";
      const isPaintActive = isImage && item.id === state.selectedId && (state.paintTool === "brush" || state.paintTool === "erase" || state.paintTool === "shape");
      const img = isImage && cleanText(item.image_path)
        ? `<img class="iamccs-isf-item-image" src="${imageViewUrl(item.image_path)}" draggable="false" style="object-fit:${escapeXml(item.fit || "contain")};opacity:${Number(item.opacity ?? 1)}" />`
        : "";
      const maskCanvas = isImage ? `<svg class="iamccs-isf-mask-svg" data-mask-svg="${escapeXml(item.id)}" viewBox="0 0 1000 1000" preserveAspectRatio="none" aria-hidden="true">${maskSvgMarkupForItem(item)}</svg><canvas class="iamccs-isf-mask-canvas" data-mask-canvas="${escapeXml(item.id)}"></canvas>` : "";
      const bodyAttr = isImage ? "" : " contenteditable=\"true\" spellcheck=\"false\"";
      const chip = kind === "text" ? "text" : (kind === "image" ? "image" : "obj");
      return `
        <div class="iamccs-isf-item ${escapeXml(kind)} ${item.id === state.selectedId ? "selected" : ""} ${isPaintActive ? "mask-paint-active" : ""}" data-item-id="${escapeXml(item.id)}" data-render-debug="${escapeXml(`${kind}:${left},${top},${width},${height}`)}" style="left:${left}%;top:${top}%;width:${width}%;height:${height}%;z-index:${z};">
          ${img}
          ${maskCanvas}
          <div class="iamccs-isf-item-head" data-drag-handle="move"><span>${index + 1}. ${escapeXml(label)}</span><span class="iamccs-isf-item-kind ${escapeXml(chip)}">${escapeXml(kind)}</span></div>
          <div class="iamccs-isf-item-body" data-edit-body="1"${bodyAttr}>${escapeXml(body)}</div>
          <div class="iamccs-isf-handle" data-drag-handle="resize" title="Resize"></div>
        </div>`;
    }).join("");
    boardPreview.innerHTML = `
      <div class="iamccs-isf-board-artboard-meta">${escapeXml(meta)}</div>
      <div class="iamccs-isf-board-artboard ${showResultBg ? "has-result-bg" : ""}" data-artboard style="--iamccs-artboard-aspect:${aspect};--iamccs-box-fill-opacity:${boxFillOpacity};--iamccs-box-head-opacity:${boxHeadOpacity};--iamccs-box-body-opacity:${boxBodyOpacity};">
        ${resultBgHtml}
        ${itemHtml || (showResultBg ? "" : `<div class="iamccs-isf-board-artboard-empty">Add Object, Add Text, or Add Image. Drag boxes directly on this board; resize with the white handle.</div>`)}
      </div>`;
    artboard = boardPreview.querySelector('[data-artboard]') || boardPreview;
    applyResultCompareCss();
    footStatus.textContent = `Canvas layers visible: ${visibleItems.length}`;
    artboard.querySelectorAll('[data-item-id]').forEach((el) => {
      const id = el.getAttribute('data-item-id');
      const item = items.find((entry) => entry.id === id);
      if (!item) return;
      const maskCanvas = el.querySelector('[data-mask-canvas]');
      if (maskCanvas) bindMaskCanvas(maskCanvas, item, el);
      el.addEventListener('pointerdown', (event) => {
        if (event.button != null && event.button !== 0) return;
        if (event.target?.closest?.('[data-mask-canvas]')) return;
        if (event.target?.closest?.('[contenteditable="true"]')) return;
        event.preventDefault();
        event.stopPropagation();
        if (event.altKey || event.ctrlKey || event.metaKey) {
          if (cycleSelectionAtPoint(event)) return;
        }
        state.selectedId = item.id;
        syncItemFields();
        renderLayerList();
        const mode = event.target?.closest?.('[data-drag-handle="resize"]') ? 'resize' : 'move';
        startDrag(event, item, mode);
      });
      const body = el.querySelector('[data-edit-body]');
      const commitBodyEdit = () => {
        const stored = (state.data.items || []).find((entry) => entry.id === item.id) || item;
        const value = cleanText(body?.textContent || "");
        if (stored.kind === 'text') {
          stored.text = value;
          item.text = value;
        } else if (stored.kind !== 'image') {
          stored.desc = value;
          item.desc = value;
        }
        if (state.selectedId === stored.id) {
          if (stored.kind === 'text' && itemFields.text && document.activeElement !== itemFields.text) itemFields.text.value = stored.text || "";
          if (stored.kind !== 'image' && itemFields.desc && document.activeElement !== itemFields.desc) itemFields.desc.value = stored.desc || "";
        }
        writeData(node, state.data);
        updatePromptPreview();
      };
      body?.addEventListener('pointerdown', (event) => {
        if (body.isContentEditable) {
          state.selectedId = item.id;
          syncItemFields();
          renderLayerList();
          event.stopPropagation();
        }
      });
      body?.addEventListener('input', () => {
        commitBodyEdit();
      });
      body?.addEventListener('paste', () => {
        window.setTimeout(commitBodyEdit, 0);
      });
      body?.addEventListener('keydown', (event) => {
        if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === 's') {
          event.preventDefault();
          commitBodyEdit();
          showToast("Box text saved to Visual Description", { anchor: body, tone: "success", ms: 1400 });
        }
      });
      body?.addEventListener('blur', () => {
        commitBodyEdit();
        syncItemFields();
        renderLayerList();
        updatePromptPreview();
      });
    });
    syncMaskUi();
  }
  function removeDefaultScaffoldItems(reason = "") {
    const items = Array.isArray(state.data.items) ? state.data.items : [];
    if (!items.length) return false;
    const hasUserImage = items.some((item) => item?.kind === "image" && cleanText(item.image_path));
    if (!hasUserImage) return false;
    const before = items.length;
    state.data.items = items.filter((item) => {
      if (!item || item.kind === "image") return true;
      const id = cleanText(item.id).toLowerCase();
      const label = cleanText(item.label).toLowerCase();
      const text = cleanText(item.text || item.desc).toLowerCase();
      const looksLikeDefaultScaffold =
        id.includes("source_full_canvas") ||
        id.includes("default_scaffold") ||
        label.includes("source image") ||
        label.includes("visible test") ||
        text.includes("source image") ||
        text.includes("visible test target") ||
        text.includes("full source image used as img2img guide");
      return !looksLikeDefaultScaffold;
    });
    const changed = state.data.items.length !== before;
    if (changed) {
      const selectedStillExists = state.data.items.some((item) => item.id === state.selectedId);
      if (!selectedStillExists) state.selectedId = state.data.items.find((item) => item.kind === "image")?.id || state.data.items[0]?.id || null;
      console.info("[IAMCCS FrameDesigner] removed default scaffold items", { reason, removed: before - state.data.items.length });
    }
    return changed;
  }

  function render() {
    if ((Array.isArray(state.data.items) ? state.data.items : []).some((item) => item?.kind === "image")) {
      if (removeDefaultScaffoldItems("render with image layer")) writeData(node, state.data);
    }
    syncSceneFields();
    refreshPresetDropdowns();
    syncItemFields();
    renderLayerList();
    renderArtboard();
    updatePromptPreview();
    syncShowImgControls();
  }

  async function uploadImageFiles(files) {
    const uploaded = [];
    for (const file of Array.from(files || [])) {
      if (!String(file?.type || '').startsWith('image/')) continue;
      const dimensions = await imageFileDimensions(file);
      const body = new FormData();
      body.append('image', file);
      try {
        const resp = await api.fetchApi('/upload/image', { method: 'POST', body });
        if (resp?.status === 200) {
          const data = await resp.json();
          let name = data.name || file.name;
          if (data.subfolder) name = `${data.subfolder}/${name}`;
          uploaded.push({ path: name, width: dimensions.width, height: dimensions.height });
        }
      } catch (error) {
        console.error('[IAMCCS StoryboardFrame V2] image upload failed', error);
      }
    }
    if (!isV2 || !uploaded.length) return;
    if (state.workflowModeKey === "image_refine" || state.data?.workflow_mode === "image_refine") {
      const entry = uploaded[0];
      clearMaskPaintForNewImage();
      state.data.items = [normalizeItem({
        id: "source_image_full_canvas",
        kind: "image",
        label: entry.path.split(/[\/]/).pop() || "Source image",
        x: 0,
        y: 0,
        w: 1000,
        h: 1000,
        desc: "Source image used as the i2i/inpaint guide. Write the edit instruction in the prompt boxes; paint the target area with Image Mask Paint.",
        image_path: entry.path,
        fit: "stretch",
        opacity: 1,
        color_palette: state.data.scene?.color_palette || ["#123A5A", "#2F6F9E", "#C4A15D", "#7C6B55"],
      }, 0)];
      state.selectedId = state.data.items[0]?.id || null;
      state.data.i2i = normalizeI2I({ ...(state.data.i2i || {}), enabled: true, source_mode: "refine_source_image" });
      if (entry.width && entry.height) {
        state.data.canvas.width = clampInt(entry.width, 1, 16384, state.data.canvas.width || 1024);
        state.data.canvas.height = clampInt(entry.height, 1, 16384, state.data.canvas.height || 1024);
        state.data.canvas.target_width = state.data.canvas.width;
        state.data.canvas.target_height = state.data.canvas.height;
        state.data.canvas.aspect_label = `Image Editor ${state.data.canvas.width}x${state.data.canvas.height}`;
        state.targetResolutionKey = "custom";
      }
      setWidget(node, "i2i_enabled", true);
      persist();
      render();
      footStatus.textContent = `Loaded source image for Image Editor: ${state.data.canvas.width}x${state.data.canvas.height}`;
      return;
    }
    clearMaskPaintForNewImage();
    removeDefaultScaffoldItems("image import");
    uploaded.forEach((entry, index) => {
      const rect = imageRectForDimensions(entry.width, entry.height, index);
      const item = createItem('image', {
        ...rect,
        image_path: entry.path,
        fit: 'contain',
        label: entry.path.split(/[\\/]/).pop() || `Image ${index + 1}`,
      });
      state.data.items.push(item);
      state.selectedId = item.id;
    });
    state.data.i2i = normalizeI2I({ ...(state.data.i2i || {}), enabled: true });
    persist();
    render();
    footStatus.textContent = `Added ${uploaded.length} image layer${uploaded.length === 1 ? '' : 's'} to i2i canvas`;
  }

  importInput?.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    importIdeoboardFile(file);
    event.target.value = '';
  });
  presetGalleryInput?.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    importPresetGalleryFile(file);
    event.target.value = '';
  });
  stylePresetInput?.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    importStylePresetFile(file);
    event.target.value = '';
  });
  gridPresetInput?.addEventListener('change', (event) => {
    const file = event.target.files?.[0];
    importGridPresetFile(file);
    event.target.value = '';
  });
  imageInput?.addEventListener('change', (event) => {
    const count = event.target.files?.length || 0;
    footStatus.textContent = count ? `Selected ${count} image file(s)` : 'Image picker closed without selection';
    uploadImageFiles(event.target.files);
    event.target.value = '';
  });
  boardPreview?.addEventListener('dragover', (event) => {
    event.preventDefault();
    event.stopPropagation();
  });
  boardPreview?.addEventListener('drop', (event) => {
    event.preventDefault();
    event.stopPropagation();
    uploadImageFiles(event.dataTransfer?.files);
  });
  const pasteHandler = (event) => {
    if (!(app.canvas?.selected_nodes && app.canvas.selected_nodes[node.id])) return;
    const files = [];
    Array.from(event.clipboardData?.items || []).forEach((item) => {
      if (item.kind === 'file' && String(item.type || '').startsWith('image/')) files.push(item.getAsFile());
    });
    if (!files.length) return;
    event.preventDefault();
    event.stopImmediatePropagation();
    uploadImageFiles(files);
  };
  document.addEventListener('paste', pasteHandler, { capture: true });
  const previousOnRemoved = node.onRemoved;
  node.onRemoved = function () {
    document.removeEventListener('paste', pasteHandler, { capture: true });
    iamccsLiveFrameNodes.delete(node);
    iamccsFrameDesignerNodes.delete(node);
    previousOnRemoved?.apply(this, arguments);
  };

  iamccsFrameDesignerNodes.add(node);

  function runHeaderAction(button, event = null) {
    const action = button?.dataset?.action;
    if (!action) return;
    event?.preventDefault?.();
    event?.stopPropagation?.();
    event?.stopImmediatePropagation?.();
    footStatus.textContent = `Action: ${action}`;
    if (action === 'save-ideoboard') {
      downloadIdeoboard(button);
      return;
    } else if (action === 'import-ideoboard') {
      importInput?.click();
      return;
    } else if (action === 'import-preset-gallery') {
      presetGalleryInput?.click();
      return;
    } else if (action === 'add-object') {
      const item = createItem('obj');
      state.data.items.push(item);
      state.selectedId = item.id;
      footStatus.textContent = `Added object layer ${item.id}`;
    } else if (action === 'add-text') {
      const item = createItem('text');
      state.data.items.push(item);
      state.selectedId = item.id;
      footStatus.textContent = `Added text layer ${item.id}`;
    } else if (action === 'add-image') {
      if (!isV2) return;
      if (!imageInput) {
        showToast('Image input is not available in this node instance.', { anchor: button, tone: 'error' });
        return;
      }
      footStatus.textContent = 'Opening image picker...';
      imageInput.click();
      return;
    } else if (action === 'refine') {
      if (!isV2) return;
      applyRefinePreset();
      return;
    } else if (action === 'duplicate') {
      const item = currentItem();
      if (!item) return;
      const clone = normalizeItem({ ...item, id: '', x: item.x + 24, y: item.y + 24, label: `${item.label} copy` }, state.data.items.length);
      state.data.items.push(clone);
      state.selectedId = clone.id;
      footStatus.textContent = `Duplicated layer ${clone.id}`;
    } else if (action === 'delete') {
      deleteLayer(state.selectedId);
      return;
    } else if (action === 'clear-boxes') {
      state.data.scene = {
        ...(state.data.scene || {}),
        high_level_description: '',
        aesthetics: '',
        lighting: '',
        photo: '',
        medium: state.data.scene?.medium || 'photography',
        art_style: '',
        background: '',
      };
      state.data.items = (state.data.items || []).filter((item) => item?.kind === 'image').map(normalizeItem);
      state.selectedId = state.data.items[0]?.id || null;
      footStatus.textContent = 'Prompt boxes and non-image layers cleared; image layers and painted masks are kept.';
    } else if (action === 'toggle-paper') {
      state.paperMode = !state.paperMode;
      state.data.ui = { ...(state.data.ui || {}), paper_mode: state.paperMode };
      syncPaperModeButton();
      persistUiState();
      persist();
      return;
    } else if (action === 'toggle-show-img') {
      state.showResultImage = !state.showResultImage;
      if (state.showResultImage && state.resultBgUrl) state.resultOverlayVisible = true;
      persistUiState();
      syncShowImgControls();
      renderArtboard();
      showToast(state.showResultImage ? "Generated image will show under boxes" : "Generated image hidden under boxes", { anchor: button, tone: "success", ms: 2200 });
      return;
    } else if (action === 'copy-json') {
      updatePromptPreview();
      const payload = previewField?.value || JSON.stringify(toPrompt(state.data), null, 2);
      navigator.clipboard?.writeText(payload).then(() => {
        footStatus.textContent = 'Prompt JSON copied to clipboard';
      }).catch(() => {
        previewField?.focus();
        previewField?.select();
        footStatus.textContent = 'Prompt JSON selected for manual copy';
      });
      return;
    } else if (action === 'toggle-fullscreen') {
      toggleFullscreen();
      return;
    } else if (action === 'reset') {
      state.data = defaultData();
      state.selectedId = state.data.items[0]?.id || null;
      writeInputSignature(node, "");
      footStatus.textContent = 'Layout reset';
    }
    persist();
    render();
  }

  root.querySelectorAll('[data-action]').forEach((button) => {
    button.addEventListener('pointerdown', (event) => {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
    }, true);
    button.addEventListener('pointerup', (event) => runHeaderAction(button, event), true);
    button.addEventListener('click', (event) => {
      event.preventDefault();
      event.stopPropagation();
      event.stopImmediatePropagation?.();
    }, true);
  });

  buildScenePane();
  buildInspectorPane();
  mountHost.appendChild(root);

  const domWidget = node.addDOMWidget('IAMCCS StoryboardFrame Director', 'iamccs_storyboard_frame_designer', mountHost, { serialize: false });
  domWidget.computeSize = () => FRAME_WIDGET_SIZE.slice();
  const originalOnResize = node.onResize;
  node.onResize = function () {
    mountHost.style.width = `${FRAME_WIDGET_SIZE[0]}px`;
    mountHost.style.height = `${FRAME_WIDGET_SIZE[1]}px`;
    mountHost.style.maxHeight = `${FRAME_WIDGET_SIZE[1]}px`;
    return originalOnResize?.apply(this, arguments);
  };
  const originalOnExecuted = node.onExecuted;
  node.onExecuted = function (message) {
    const result = originalOnExecuted?.apply(this, arguments);
    const pick = (value) => Array.isArray(value) ? value[0] : value;
    const payload = pick(message?.design_data);
    if (payload) {
      applyRuntimeIdeoboard(
        payload,
        pick(message?.ideoboard_input_signature),
        Boolean(pick(message?.used_ideoboard_input))
      );
    }
    return result;
  };
  applyNodeSize();
  window.addEventListener('keydown', (event) => {
    if (event.key === 'Escape' && root.classList.contains('iamccs-isf-fullscreen')) {
      toggleFullscreen(false);
    }
  });
  render();
}

function installJsonPreviewPass(node) {
  if (!node || node._iamccsJsonPreviewPassReady || typeof node.addDOMWidget !== "function") return;
  node._iamccsJsonPreviewPassReady = true;
  ensureStyles();
  const host = document.createElement("div");
  host.className = "iamccs-jsonpass";
  host.innerHTML = '<div><strong>IAMCCS JSON Preview / Pass</strong><div data-report>Queue once to preview the exact JSON passed downstream.</div></div><textarea spellcheck="false" readonly></textarea>';
  const report = host.querySelector("[data-report]");
  const area = host.querySelector("textarea");
  const domWidget = node.addDOMWidget("IAMCCS JSON Preview / Pass", "iamccs_json_preview_pass", host, { serialize: false });
  domWidget.computeSize = () => [420, 260];
  const originalOnExecuted = node.onExecuted;
  node.onExecuted = function (message) {
    const result = originalOnExecuted?.apply(this, arguments);
    const text = Array.isArray(message?.text) ? message.text : [];
    if (report) report.textContent = text[0] || "JSON passed through.";
    if (area) area.value = text[1] || "";
    return result;
  };
}

app.registerExtension({
  name: 'IAMCCS.StoryboardFrameDesigner',
  nodeCreated(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || '';
    if (type === TYPE || type === TYPE_V2 || node?.type === TYPE || node?.type === TYPE_V2) [0, 180, 600].forEach((delay) => setTimeout(() => install(node), delay));
    if (type === TYPE_JSON_PASS || node?.type === TYPE_JSON_PASS) [0, 180, 600].forEach((delay) => setTimeout(() => installJsonPreviewPass(node), delay));
  },
  loadedGraphNode(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || '';
    if (type === TYPE || type === TYPE_V2 || node?.type === TYPE || node?.type === TYPE_V2) [0, 180, 600].forEach((delay) => setTimeout(() => install(node), delay));
    if (type === TYPE_JSON_PASS || node?.type === TYPE_JSON_PASS) [0, 180, 600].forEach((delay) => setTimeout(() => installJsonPreviewPass(node), delay));
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TYPE && nodeData?.name !== TYPE_V2 && nodeData?.name !== TYPE_JSON_PASS) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      if (nodeData?.name === TYPE_JSON_PASS) [0, 180, 600].forEach((delay) => setTimeout(() => installJsonPreviewPass(this), delay));
      else [0, 180, 600].forEach((delay) => setTimeout(() => install(this), delay));
    };
  },
});


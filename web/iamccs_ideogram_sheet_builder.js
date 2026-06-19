import { app } from "../../scripts/app.js";

const TYPE = "IAMCCS_IdeogramSheetBuilder";
const STYLE_ID = "iamccs-sheet-builder-style";

const STYLE_PRESETS = {
  cinematic_scifi: {
    label: "Default Cinematic Sci-Fi",
    style: "photo",
    high: "A cinematic storyboard contact sheet for a grounded photoreal science fiction film, practical sets, real actors, restrained production design, atmospheric lighting, each panel framed as a different shot from the same sequence.",
    background: "A clean storyboard contact sheet divided into equal panels, all sharing the same believable near-future world with practical architecture, tactile props, controlled haze, realistic costumes, and cinematic continuity. photographic continuity, clean grid structure, readable panel separation, documented structured JSON fields only.",
    aesthetics: "photoreal live-action science fiction cinema, practical sets, real actors, grounded production design, tactile materials, subtle futuristic technology, cinematic realism",
    lighting: "soft volumetric key light, cool practical LEDs, controlled contrast, atmospheric haze, realistic reflections",
    photo: "anamorphic 35mm cinematic stills, ARRI Alexa style color science, shallow depth of field when needed, film grain, practical set photography",
    medium: "photograph",
    palette: ["#0A0D0F", "#263238", "#53605C", "#D6D0C2", "#8B1E2D"]
  },
  vintage_fantasy: {
    label: "Ultra Vintage 1974",
    style: "photo",
    high: "A WIDE LANDSCAPE live-action vintage fantasy film contact sheet, exactly 2 columns by 3 rows, exactly six large 16:9 panels only. It looks like frames scanned from a lost 1974 European surreal fairy-tale film, hyperreal practical costumes and physical sets, photographic live-action realism. The story follows one thin red-haired Alice-like traveler through a diseased medieval dream kingdom; she appears as one clearly staged character when present, often small or from behind.",
    background: "One single WIDE LANDSCAPE contact sheet, wide landscape orientation, divided by thin black grid lines into exactly six equal landscape 16:9 panels. Every panel is a live-action film still with real actors in handmade costumes, practical monster suits, prosthetic masks, miniature castles, matte paintings, theatrical fog, and faded Eastmancolor film texture. Background figures may be masked jesters, puppet creatures, villagers, guards, or monsters, each role has a distinct costume, mask, posture, and distance from camera. The image is authored as one structured 2x3 contact sheet with six large landscape panels and clear row-major grid geometry.",
    aesthetics: "hyperreal live-action photography, lost 1974 European surreal fantasy film, practical costumes, rubber masks, prosthetic faces, real actors, handmade theatrical props, miniature castles, matte painting landscapes, forced perspective, painted backdrops, dark surreal fairy tale, diseased medieval dream kingdom, unsettling cinematic realism",
    lighting: "faded Eastmancolor print, Technicolor blue-cyan palette, blue-cyan moonlight, tungsten theatrical key light, optical diffusion, atmospheric fog, soft focus bloom, analog film grain, mild gate weave, practical stage lighting",
    photo: "vintage anamorphic lenses, 35mm/70mm scanned film frame, optical diffusion, analog grain, mild gate weave, soft lens bloom, practical creature effects photographed on real sets, matte painting backgrounds, forced perspective miniatures",
    medium: "photograph",
    palette: ["#123A5A", "#2F6F9E", "#C4A15D", "#7C6B55", "#D8C7A0", "#1B1E22"],
    panelTips: [
      "Tip: stage one clear subject or clearly different masked roles with distinct silhouettes.",
      "Tip: if the red-haired traveler appears, stage her small in the scene or from behind.",
      "Tip: favor establishing composition, matte painting scale, and miniature realism.",
      "Tip: background figures should read as guards, jesters, villagers, puppets, or monsters with distinct costumes.",
      "Tip: if two figures appear, give them different masks, costume shapes, and physical roles.",
      "Tip: final panel may focus on monsters or environment while preserving practical suit/puppet realism."
    ]
  },
  samurai_cinema: {
    label: "Samurai Cinema",
    style: "photo",
    aesthetics: "hyperreal period samurai cinema, practical costumes, weathered armor, mud, smoke, disciplined cinematic blocking, Kurosawa-inspired battlefield realism",
    lighting: "overcast natural light, smoky atmosphere, hard rim light through dust, restrained contrast",
    photo: "vintage anamorphic lenses, 35mm film scan, practical set photography, long lens compression",
    medium: "photograph",
    palette: ["#20201B", "#6E2C24", "#BFA36A", "#D8D0B8", "#5C6770"]
  },
  tarkovsky_scifi: {
    label: "Tarkovsky Sci-Fi",
    style: "photo",
    aesthetics: "philosophical Eastern European science fiction, decayed industrial sacred architecture, wet concrete, rust, memory, silence, photoreal practical sets",
    lighting: "cold window light, milky haze, dim emergency practicals, reflective water, deep corridors",
    photo: "slow observational 35mm cinema, restrained anamorphic framing, film grain, soft halation",
    medium: "photograph",
    palette: ["#0A0D0F", "#263238", "#53605C", "#8B1E2D", "#D6D0C2"]
  },
  horror_vhs: {
    label: "Horror VHS",
    style: "photo",
    aesthetics: "late 1980s horror film still, practical gore, grimy interiors, tape degradation, low budget physical effects",
    lighting: "sick fluorescent spill, hard flashlight beam, underexposed corners, red practical light",
    photo: "VHS transfer from 16mm, analog noise, chroma bleed, soft detail, handheld framing",
    medium: "photograph",
    palette: ["#111111", "#5D1A1A", "#6E7B70", "#D1C3A0", "#2F3A4A"]
  },
  silent_bw: {
    label: "Silent B/W 1920s",
    style: "photo",
    aesthetics: "silent era black and white cinema, expressionistic blocking, theatrical faces, practical sets, orthochromatic film response, hand-cranked camera feel",
    lighting: "hard carbon arc studio lighting, deep shadows, silver highlights, visible set haze, iris vignette",
    photo: "1920s black and white nitrate film scan, 1.33 academy framing influence adapted to storyboard panels, high contrast grain, slight flicker, intertitle-era composition",
    medium: "photograph",
    palette: ["#050505", "#2D2D2D", "#777777", "#C8C8C8", "#F1F1E8"]
  },
  cinemascope_epic: {
    label: "CinemaScope Epic",
    style: "photo",
    aesthetics: "large format 1950s CinemaScope epic, monumental blocking, wide horizontal staging, lavish costumes, matte paintings, photochemical realism",
    lighting: "Technicolor daylight, controlled studio fill, golden rim light, broad scenic visibility",
    photo: "CinemaScope anamorphic lenses, 35mm dye-transfer print, wide tableau composition, rich photochemical color, practical sets and matte paintings",
    medium: "photograph",
    palette: ["#1E3D59", "#C99E45", "#8A3B2E", "#D8C7A0", "#0D1117"]
  },
  noir_1940s: {
    label: "Noir 1940s",
    style: "photo",
    aesthetics: "1940s film noir, wet streets, venetian blind shadows, fatalistic mood, sharp suits, smoke, crime drama realism",
    lighting: "low key tungsten lighting, hard slashes of shadow, rim-lit smoke, glossy black reflections",
    photo: "black and white 35mm film still, deep focus noir cinematography, high contrast grain, studio backlot realism",
    medium: "photograph",
    palette: ["#030303", "#191919", "#4D4D4D", "#A8A8A8", "#E8E1D0"]
  },
  italian_neorealism: {
    label: "Italian Neorealism",
    style: "photo",
    aesthetics: "postwar Italian neorealist cinema, non-glamorous faces, real streets, social realism, weathered walls, humanist blocking",
    lighting: "available daylight, soft overcast street light, practical interiors, low contrast grey skies",
    photo: "1940s-1950s black and white 35mm location photography, naturalistic handheld feeling, unpolished documentary texture",
    medium: "photograph",
    palette: ["#101010", "#3A3A35", "#777468", "#B8B1A0", "#E5DDC8"]
  },
  giallo_1970s: {
    label: "Italian Giallo",
    style: "photo",
    aesthetics: "1970s Italian giallo thriller, stylized murder mystery, leather gloves, baroque interiors, saturated color, psychological unease",
    lighting: "red blue green gel lighting, hard practical lamps, glossy shadows, night interiors",
    photo: "1970s 35mm anamorphic thriller still, saturated Eastmancolor, zoom lens drama, optical softness, film grain",
    medium: "photograph",
    palette: ["#08080A", "#B5122B", "#0A5F76", "#D6B45B", "#194D2D"]
  },
  hammer_gothic: {
    label: "Hammer Gothic",
    style: "photo",
    aesthetics: "1960s British gothic horror, castles, foggy graveyards, theatrical blood, velvet costumes, practical sets",
    lighting: "colored studio moonlight, warm candle practicals, theatrical fog, high contrast gothic shadows",
    photo: "1960s gothic horror film still, saturated color print, studio set photography, soft diffusion, practical makeup effects",
    medium: "photograph",
    palette: ["#0D0C10", "#5E0E18", "#254159", "#C8A05A", "#D8D0C2"]
  },
  french_new_wave: {
    label: "French New Wave",
    style: "photo",
    aesthetics: "1960s French New Wave cinema, street realism, spontaneous blocking, existential mood, cafes, apartments, urban texture",
    lighting: "natural window light, available street light, high contrast sun and shade, imperfect exposure",
    photo: "black and white 35mm handheld film still, fast lens, documentary immediacy, visible grain",
    medium: "photograph",
    palette: ["#080808", "#2E2E2E", "#737373", "#C7C2B5", "#F0E7D8"]
  },
  spaghetti_western: {
    label: "Spaghetti Western",
    style: "photo",
    aesthetics: "1960s spaghetti western, dusty landscapes, weathered faces, ponchos, standoffs, violent operatic framing",
    lighting: "hard desert sun, smoky saloon practicals, warm dust haze, high contrast silhouettes",
    photo: "Techniscope 2-perf 35mm western still, long lens closeups, wide desert anamorphic framing, film grain",
    medium: "photograph",
    palette: ["#1B1510", "#7C4A25", "#C49A52", "#D7C4A0", "#40535B"]
  },
  soviet_mosfilm: {
    label: "Soviet Mosfilm",
    style: "photo",
    aesthetics: "1960s Soviet historical and philosophical cinema, monumental faces, austere landscapes, painterly realism, disciplined mise en scene",
    lighting: "cold diffuse sky, smoky interiors, sculptural side light, restrained contrast",
    photo: "Mosfilm 35mm film still, slow cinema composition, practical sets, textured grain, muted photochemical color",
    medium: "photograph",
    palette: ["#111717", "#394A45", "#6A6D5D", "#A89A72", "#D4C7AE"]
  },
  wuxia_shaw: {
    label: "Shaw Wuxia",
    style: "photo",
    aesthetics: "1970s Hong Kong wuxia cinema, elegant fighters, studio forests, painted skies, wirework, ornate costumes",
    lighting: "bright studio moonlight, saturated color gels, theatrical backlight, clean action visibility",
    photo: "Shaw Brothers 35mm color film still, studio set fantasy realism, anamorphic martial arts framing, film grain",
    medium: "photograph",
    palette: ["#0D1D2A", "#1D6F5E", "#B8282F", "#D9B45C", "#E6D2B1"]
  },
  technicolor_musical: {
    label: "Technicolor Musical",
    style: "photo",
    aesthetics: "classic Hollywood Technicolor musical, choreographed staging, theatrical sets, saturated costumes, elegant production design",
    lighting: "bright studio key light, glossy highlights, colorful practicals, clean beauty lighting",
    photo: "three-strip Technicolor 35mm film still, full body choreography framing, saturated dye-transfer color, studio realism",
    medium: "photograph",
    palette: ["#103E75", "#D9263E", "#F0C85A", "#2D8B68", "#F5E7CE"]
  }
};

const GRID_PRESETS = {
  free: { label: "Free", width: 1024, height: 1024, boxes: null },
  story_2x3: { label: "Storyboard 2x3", width: 2048, height: 1728, boxes: [2, 3], order: "row_major" },
  story_3x3_16x9: { label: "Storyboard 3x3 16:9", width: 2304, height: 1296, boxes: [3, 3], order: "row_major" },
  storyboard_2x2_scope: { label: "Storyboard 2x2 2.39:1", width: 2048, height: 856, boxes: [2, 2], order: "row_major" },
  storyboard_3x2_scope: { label: "Storyboard 3x2 2.39:1", width: 3072, height: 856, boxes: [3, 2], order: "row_major" },
  frame_16x9: { label: "16:9 Frame", width: 1536, height: 864, boxes: [1, 1], order: "row_major" },
  frame_scope: { label: "2.39:1 Frame", width: 2048, height: 856, boxes: [1, 1], order: "row_major" }
};

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

function clean(value) {
  return String(value || "").trim();
}

function clamp(value, min, max, fallback = min) {
  const n = Number(value);
  if (!Number.isFinite(n)) return fallback;
  return Math.max(min, Math.min(max, n));
}

function parseList(raw, fallback = []) {
  try {
    const value = JSON.parse(String(raw || ""));
    return Array.isArray(value) ? value : fallback;
  } catch {
    return fallback;
  }
}

function normalizeColor(value) {
  let text = clean(value).toUpperCase();
  if (!text) return "";
  if (!text.startsWith("#")) text = `#${text}`;
  return /^#[0-9A-F]{6}$/.test(text) ? text : "";
}

function paletteFromText(raw) {
  return String(raw || "").split(",").map(normalizeColor).filter(Boolean);
}

function titleCaseFromDesc(desc, fallback) {
  const text = clean(desc).replace(/^Panel\s+\d+\s*,?\s*/i, "");
  const beforeColon = text.split(":")[0]?.trim();
  if (beforeColon && beforeColon.length <= 48 && !beforeColon.includes(".")) return beforeColon.toUpperCase();
  return fallback;
}

function ideogramPanelPrompt(box, index) {
  const rawTitle = clean(box.title) || titleCaseFromDesc(box.desc, `PANEL ${index + 1}`);
  const title = rawTitle.toUpperCase();
  let desc = clean(box.desc)
    .replace(/^Panel\s+\d+\s*,?\s*/i, "")
    .replace(new RegExp(`^${title}\\s*:\\s*`, "i"), "")
    .trim();
  if (!desc) desc = "wide cinematic shot, subject performs a clear action inside the scene, practical set environment, vintage anamorphic 35mm film language";
  const hasShot = /(wide|medium|close|macro|overhead|low-angle|high-angle|tracking|establishing|portrait|detail)/i.test(desc);
  const hasCamera = /(lens|35mm|anamorphic|film|photograph|camera|diffusion|grain|Eastmancolor|Technicolor|practical set)/i.test(desc);
  const shot = hasShot ? "" : "clear cinematic shot scale, ";
  const camera = hasCamera ? "" : ", vintage anamorphic 35mm scanned film frame, practical set photography";
  return {
    ...box,
    title,
    desc: `Panel ${index + 1}, ${title}: ${shot}${desc}${camera}.`.replace(/\s+/g, " ").trim(),
    useTips: false
  };
}

function currentStyleKind(node) {
  const value = widget(node, "style")?.value;
  if (typeof value === "string") return value;
  if (value && typeof value === "object" && value.style) return value.style;
  return "photo";
}

function setStyleKind(node, style) {
  setWidget(node, "style", style);
}

function defaultBoxes() {
  return [
  {
    "type": "obj",
    "text": "",
    "desc": "Panel 1, landscape 16:9 cell: one masked blue court jester in a handmade cloth costume stands in a ruined palace corridor, practical face paint and gold trim, real actor photographed on set; the red-haired traveler is absent. Background contains different masked attendants only, distinct supporting roles. This is one of exactly six large panels; keep this as one complete panel cell. The image is authored as one structured 2x3 contact sheet with six large landscape panels and clear row-major grid geometry.",
    "palette": [
      "#123A5A",
      "#2F6F9E",
      "#C4A15D",
      "#D8C7A0"
    ],
    "x": 0.0,
    "y": 0.0,
    "w": 0.5,
    "h": 0.333
  },
  {
    "type": "obj",
    "text": "",
    "desc": "Panel 2, landscape 16:9 cell: one bone-masked scarecrow pilgrim walks through a dead medieval village under a giant blue moon, real tattered costume, rubber mask, dusty practical set, live-action vintage film still; the red-haired traveler reads as one consistent role. This is one of exactly six large panels; keep this as one complete panel cell. The image is authored as one structured 2x3 contact sheet with six large landscape panels and clear row-major grid geometry.",
    "palette": [
      "#123A5A",
      "#2F6F9E",
      "#C4A15D",
      "#D8C7A0"
    ],
    "x": 0.0,
    "y": 0.333,
    "w": 0.5,
    "h": 0.334
  },
  {
    "type": "obj",
    "text": "",
    "desc": "Panel 3, landscape 16:9 cell: distant establishing shot of a floating stone castle on a rocky cliff above a blue sea, sailboat foreground, matte painting landscape, miniature castle realism, environment-led composition with the protagonist outside the foreground. This is one of exactly six large panels; keep this as one complete panel cell. The image is authored as one structured 2x3 contact sheet with six large landscape panels and clear row-major grid geometry.",
    "palette": [
      "#123A5A",
      "#2F6F9E",
      "#C4A15D",
      "#D8C7A0"
    ],
    "x": 0.0,
    "y": 0.667,
    "w": 0.5,
    "h": 0.333
  },
  {
    "type": "obj",
    "text": "",
    "desc": "Panel 4, landscape 16:9 cell: the single thin red-haired traveler is seen from behind, small in frame, walking toward an impossible ruined town and clock tower suspended on stone arches, physical set plus matte painting, one readable traveler silhouette. This is one of exactly six large panels; keep this as one complete panel cell. The image is authored as one structured 2x3 contact sheet with six large landscape panels and clear row-major grid geometry.",
    "palette": [
      "#123A5A",
      "#2F6F9E",
      "#C4A15D",
      "#D8C7A0"
    ],
    "x": 0.5,
    "y": 0.0,
    "w": 0.5,
    "h": 0.333
  },
  {
    "type": "obj",
    "text": "",
    "desc": "Panel 5, landscape 16:9 cell: close live-action shot of two masked carnival guards in practical theatrical costumes, white porcelain masks, gold collars, dark palace interior; these are different masked roles, distinct supporting roles. This is one of exactly six large panels; keep this as one complete panel cell. The image is authored as one structured 2x3 contact sheet with six large landscape panels and clear row-major grid geometry.",
    "palette": [
      "#123A5A",
      "#2F6F9E",
      "#C4A15D",
      "#D8C7A0"
    ],
    "x": 0.5,
    "y": 0.333,
    "w": 0.5,
    "h": 0.334
  },
  {
    "type": "obj",
    "text": "",
    "desc": "Panel 6, landscape 16:9 cell: a practical rubber monster and one smaller masked creature cross a foggy swamp path, photographed as old monster suits and puppets under blue stage light; the traveler is absent, distinct creature roles. This is one of exactly six large panels; keep this as one complete panel cell. The image is authored as one structured 2x3 contact sheet with six large landscape panels and clear row-major grid geometry.",
    "palette": [
      "#123A5A",
      "#2F6F9E",
      "#C4A15D",
      "#D8C7A0"
    ],
    "x": 0.5,
    "y": 0.667,
    "w": 0.5,
    "h": 0.333
  }
];
}

function makeGridBoxes(cols, rows, order = "row_major") {
  const boxes = [];
  const coords = [];
  if (order === "column_major") {
    for (let col = 0; col < cols; col++) for (let row = 0; row < rows; row++) coords.push([col, row]);
  } else {
    for (let row = 0; row < rows; row++) for (let col = 0; col < cols; col++) coords.push([col, row]);
  }
  for (const [col, row] of coords) {
      const index = boxes.length + 1;
      boxes.push({
        x: col / cols,
        y: row / rows,
        w: 1 / cols,
        h: 1 / rows,
        type: "obj",
        text: "",
        desc: `Panel ${index}, landscape 16:9 cell: clear cinematic action, one physical location, one readable subject or group, practical set photography, vintage film language.`,
        tips: "Tip: write what must appear in this panel using positive visual language: shot scale, action, environment, lens, continuity cue.",
        useTips: false,
        palette: ["#123A5A", "#2F6F9E", "#C4A15D", "#D8C7A0"]
      });
  }
  return boxes;
}

function captionToBoxes(caption) {
  const elements = caption?.compositional_deconstruction?.elements;
  if (!Array.isArray(elements)) return [];
  return elements.map((el, index) => {
    const bb = Array.isArray(el.bbox) && el.bbox.length === 4 ? el.bbox : null;
    const box = {
      type: el.type === "text" ? "text" : "obj",
      text: el.text || "",
      desc: el.desc || "",
      tips: "",
      useTips: false,
      palette: Array.isArray(el.color_palette) ? el.color_palette : ["#C4A15D"],
      x: 0.03 + index * 0.03,
      y: 0.03 + index * 0.03,
      w: 0.28,
      h: 0.18
    };
    if (bb) {
      const [ymin, xmin, ymax, xmax] = bb;
      box.x = xmin / 1000;
      box.y = ymin / 1000;
      box.w = (xmax - xmin) / 1000;
      box.h = (ymax - ymin) / 1000;
    } else {
      box.nobbox = true;
    }
    return box;
  });
}

function buildCaption(node, boxes) {
  const style = currentStyleKind(node);
  const caption = {};
  const high = clean(widget(node, "high_level_description")?.value);
  if (high) caption.high_level_description = high;
  if (style !== "none") {
    const sd = {
      aesthetics: clean(widget(node, "aesthetics")?.value),
      lighting: clean(widget(node, "lighting")?.value)
    };
    if (style === "photo") {
      sd.photo = clean(widget(node, "photo")?.value);
      sd.medium = clean(widget(node, "medium")?.value);
    } else {
      sd.medium = clean(widget(node, "medium")?.value);
      sd.art_style = clean(widget(node, "art_style")?.value);
    }
    const palette = parseList(widget(node, "style_palette_data")?.value, []);
    if (palette.length) sd.color_palette = palette;
    caption.style_description = sd;
  }
  caption.compositional_deconstruction = {
    background: clean(widget(node, "background")?.value),
    elements: boxes.map((box) => {
      const elem = { type: box.type === "text" ? "text" : "obj" };
      if (!box.nobbox) {
        elem.bbox = [
          Math.round(clamp(box.y, 0, 1, 0) * 1000),
          Math.round(clamp(box.x, 0, 1, 0) * 1000),
          Math.round(clamp(box.y + box.h, 0, 1, 0) * 1000),
          Math.round(clamp(box.x + box.w, 0, 1, 0) * 1000)
        ];
      }
      if (elem.type === "text") elem.text = box.text || "";
      let desc = box.desc || "";
      if (box.title && !/^Panel\s+\d+/i.test(desc)) desc = `Panel ${boxes.indexOf(box) + 1}, ${box.title}: ${desc}`;
      if (box.useTips && clean(box.tips)) desc = `${desc.trim()} ${clean(box.tips)}`.trim();
      elem.desc = desc;
      if (Array.isArray(box.palette) && box.palette.length) elem.color_palette = box.palette.slice(0, 5);
      return elem;
    })
  };
  return caption;
}

function installStyle() {
  if (document.getElementById(STYLE_ID)) return;
  const style = document.createElement("style");
  style.id = STYLE_ID;
  style.textContent = `
.iamccs-sheet{box-sizing:border-box;width:100%;min-width:1260px;height:940px;background:#0d0f12;color:#ece7dc;border:1px solid #282c32;border-radius:8px;overflow:hidden;font-family:Inter,Segoe UI,Arial,sans-serif;display:grid;grid-template-rows:auto 1fr auto}
.iamccs-sheet *{box-sizing:border-box;letter-spacing:0}
.iamccs-sheet-top{height:46px;display:flex;align-items:center;gap:8px;padding:8px 12px;background:#14171b;border-bottom:1px solid #282c32}
.iamccs-sheet-brand{font-size:13px;font-weight:800;color:#f1dfb8;margin-right:8px}
.iamccs-sheet-status{margin-left:auto;color:#aeb5bf;font-size:12px}
.iamccs-sheet-main{min-height:0;display:grid;grid-template-columns:320px 1fr 390px}
.iamccs-sheet-left,.iamccs-sheet-mid,.iamccs-sheet-right{min-height:0;overflow:auto;padding:12px}
.iamccs-sheet-left{border-right:1px solid #282c32;background:#101216}
.iamccs-sheet-right{border-left:1px solid #282c32;background:#101216}
.iamccs-sheet-section{background:#16191e;border:1px solid #30353d;border-radius:8px;padding:10px;margin-bottom:10px}
.iamccs-sheet-left .iamccs-sheet-section:nth-child(1){background:#14201c;border-color:#315749}
.iamccs-sheet-left .iamccs-sheet-section:nth-child(2){background:#201a14;border-color:#6b5131}
.iamccs-sheet-left .iamccs-sheet-section:nth-child(3){background:#151b25;border-color:#33496b}
.iamccs-sheet-right .iamccs-sheet-section:nth-child(1){background:#1c1724;border-color:#57416f}
.iamccs-sheet-right .iamccs-sheet-section:nth-child(2){background:#141d24;border-color:#31576b}
.iamccs-sheet-section h4{margin:0 0 8px;color:#cdbf9e;font-size:11px;text-transform:uppercase;font-weight:800}
.iamccs-sheet-row{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.iamccs-sheet-field{display:flex;flex-direction:column;gap:4px;margin-bottom:8px}
.iamccs-sheet-field label{font-size:11px;color:#aeb5bf}
.iamccs-sheet input,.iamccs-sheet textarea,.iamccs-sheet select{width:100%;background:#07080a;border:1px solid #333943;border-radius:6px;color:#f5f0e8;padding:7px 8px;font-size:12px;outline:none}
.iamccs-sheet textarea{min-height:70px;resize:vertical;line-height:1.35}
.iamccs-sheet input:focus,.iamccs-sheet textarea:focus,.iamccs-sheet select:focus{border-color:#b9a66d;box-shadow:0 0 0 1px #b9a66d55}
.iamccs-sheet-btn{height:30px;border:1px solid #383e47;background:#1a1e24;color:#eee7da;border-radius:6px;font-size:12px;font-weight:700;cursor:pointer;padding:0 9px}
.iamccs-sheet-btn:hover{border-color:#b9a66d;color:#fff6dc}
.iamccs-sheet-btn.primary{background:#e6dcc4;border-color:#e6dcc4;color:#090a0b}
.iamccs-sheet-board-wrap{height:520px;background:#0a0b0d;border:1px solid #30353d;border-radius:8px;padding:10px;margin-bottom:10px}
.iamccs-sheet-board{position:relative;width:100%;height:100%;background:#1a1b1d;border:1px solid #5b616b;overflow:hidden}
.iamccs-sheet-box{position:absolute;background:#fff;color:#050505;border:2px solid #c4a15d;border-radius:4px;padding:6px;overflow:hidden;cursor:pointer;display:flex;flex-direction:column;gap:4px}
.iamccs-sheet-box.selected{outline:3px solid #4da3ff;z-index:5}
.iamccs-sheet-box-title{font-size:12px;font-weight:900;line-height:1.15;overflow:hidden;white-space:nowrap;text-overflow:ellipsis}
.iamccs-sheet-box-body{font-size:10px;line-height:1.16;overflow:hidden;min-height:0;flex:1}
.iamccs-sheet-box-tip{font-size:9px;line-height:1.12;color:#49515b;border-top:1px solid #d4d4d4;padding-top:3px;overflow:hidden;max-height:28%}
.iamccs-sheet-list{max-height:270px;overflow:auto;padding-right:4px}
.iamccs-sheet-list-item{border:1px solid #333943;background:#14171c;border-radius:7px;padding:8px;margin-bottom:6px;cursor:pointer}
.iamccs-sheet-list-item.selected{border-color:#b9a66d;background:#201e18}
.iamccs-sheet-list-item strong{display:block;color:#f1dfb8;font-size:12px;margin-bottom:3px}
.iamccs-sheet-list-item span{display:block;color:#aeb5bf;font-size:11px;line-height:1.25}
.iamccs-sheet-palette{display:flex;gap:5px;flex-wrap:wrap}
.iamccs-sheet-swatch{width:20px;height:20px;border-radius:5px;border:1px solid #000}
.iamccs-sheet-code{height:150px;font-family:Consolas,monospace;font-size:11px}
.iamccs-sheet-actions{display:flex;gap:6px;flex-wrap:wrap}
.iamccs-sheet-hint{font-size:11px;line-height:1.35;color:#b9c0ca;margin:-2px 0 8px}
.iamccs-sheet-fullscreen-host{position:fixed;inset:0;z-index:99999;background:#07080a;padding:10px;box-sizing:border-box}
.iamccs-sheet.is-fullscreen{width:100%;height:100%;min-width:0;border-radius:0;box-shadow:none}
.iamccs-sheet.is-fullscreen .iamccs-sheet-main{height:calc(100% - 78px)}
.iamccs-sheet.is-fullscreen .iamccs-sheet-board-wrap{height:calc(100% - 322px);min-height:420px}
.iamccs-sheet-editor-backdrop{position:fixed;inset:0;background:#000b;z-index:100000;display:flex;align-items:center;justify-content:center;padding:24px;box-sizing:border-box}
.iamccs-sheet-editor{width:min(1060px,96vw);max-height:92vh;background:#f7f2e8;color:#111;border:1px solid #d0c2a8;border-radius:10px;box-shadow:0 24px 80px #000;display:grid;grid-template-rows:auto 1fr auto;overflow:hidden}
.iamccs-sheet-editor-head{display:flex;align-items:center;justify-content:space-between;padding:14px 16px;border-bottom:1px solid #d8c9ae;background:#fffaf0}
.iamccs-sheet-editor-head strong{font-size:16px;color:#111}
.iamccs-sheet-editor-body{padding:16px;overflow:auto;display:grid;grid-template-columns:1fr 1fr;gap:12px}
.iamccs-sheet-editor-field{display:flex;flex-direction:column;gap:5px}
.iamccs-sheet-editor-field.full{grid-column:1 / -1}
.iamccs-sheet-editor-field label{font-size:12px;font-weight:800;color:#463d31;text-transform:uppercase}
.iamccs-sheet-editor input,.iamccs-sheet-editor textarea,.iamccs-sheet-editor select{background:#fff;border:1px solid #b9aa90;color:#111;border-radius:7px;padding:8px 9px;font-size:14px;line-height:1.35;outline:none}
.iamccs-sheet-editor textarea{min-height:260px;resize:vertical}
.iamccs-sheet-editor-foot{display:flex;justify-content:flex-end;gap:8px;padding:12px 16px;border-top:1px solid #d8c9ae;background:#fffaf0}
.iamccs-sheet-bottom{height:34px;display:flex;align-items:center;gap:10px;padding:8px 12px;border-top:1px solid #282c32;background:#14171b;color:#aeb5bf;font-size:12px}
`;
  document.head.appendChild(style);
}

function install(node) {
  if (!node || node._iamccsKjpbReady || typeof node.addDOMWidget !== "function") return;
  node._iamccsKjpbReady = true;
  installStyle();

  [
    "high_level_description", "background", "photo", "art_style", "aesthetics",
    "lighting", "medium", "style_palette_data", "elements_data", "grid_columns", "grid_rows", "bg_brightness"
  ].forEach((name) => hideWidget(widget(node, name)));

  const state = {
    selected: 0,
    boxes: parseList(widget(node, "elements_data")?.value, defaultBoxes())
  };
  if (!state.boxes.length) state.boxes = defaultBoxes();

  const host = document.createElement("div");
  const root = document.createElement("div");
  root.className = "iamccs-sheet";
  root.innerHTML = `
    <div class="iamccs-sheet-top">
      <div class="iamccs-sheet-brand">IAMCCS Ideogram Sheet Builder V1</div>
      <button class="iamccs-sheet-btn primary" data-action="toggleFullscreen">Open Editor</button>
      <button class="iamccs-sheet-btn primary" data-action="apply">Apply</button>
      <button class="iamccs-sheet-btn" data-action="add">Add Box</button>
      <button class="iamccs-sheet-btn" data-action="duplicate">Duplicate</button>
      <button class="iamccs-sheet-btn" data-action="delete">Delete</button>
      <button class="iamccs-sheet-btn" data-action="copy">Copy JSON</button>
      <div class="iamccs-sheet-status" data-status>Ready</div>
    </div>
    <div class="iamccs-sheet-main">
      <div class="iamccs-sheet-left">
        <div class="iamccs-sheet-section">
          <h4>1. Presets and Canvas</h4>
          <div class="iamccs-sheet-field"><label>Style preset</label><select data-ui="stylePreset"></select></div>
          <div class="iamccs-sheet-actions">
            <button class="iamccs-sheet-btn" data-action="importStylePreset">Import Style Preset</button>
            <input data-file="stylePreset" type="file" accept=".json,application/json" style="display:none">
          </div>
          <div class="iamccs-sheet-field"><label>Grid preset</label><select data-ui="gridPreset"></select></div>
          <div class="iamccs-sheet-actions">
            <button class="iamccs-sheet-btn" data-action="importGridPreset">Import Grid Preset</button>
            <input data-file="gridPreset" type="file" accept=".json,application/json" style="display:none">
          </div>
          <div class="iamccs-sheet-row">
            <div class="iamccs-sheet-field"><label>Width</label><input data-ui="width" type="number" min="64" max="16384" step="16"></div>
            <div class="iamccs-sheet-field"><label>Height</label><input data-ui="height" type="number" min="64" max="16384" step="16"></div>
          </div>
          <div class="iamccs-sheet-field"><label>Style mode</label><select data-ui="style"><option>none</option><option selected>photo</option><option>art_style</option></select></div>
          <div class="iamccs-sheet-field"><label>Palette</label><input data-ui="palette"></div>
          <div class="iamccs-sheet-palette" data-swatches></div>
        </div>
        <div class="iamccs-sheet-section">
          <h4>2. Global Ideogram JSON</h4>
          <div class="iamccs-sheet-hint">Write positive, concrete visual language. Use the panel editor for each storyboard cell.</div>
          <div class="iamccs-sheet-field"><label>High level</label><textarea data-ui="high"></textarea></div>
          <div class="iamccs-sheet-field"><label>Background</label><textarea data-ui="background"></textarea></div>
          <div class="iamccs-sheet-field"><label>Aesthetics</label><textarea data-ui="aesthetics"></textarea></div>
          <div class="iamccs-sheet-field"><label>Lighting</label><textarea data-ui="lighting"></textarea></div>
          <div class="iamccs-sheet-field"><label>Photo / Art Style</label><textarea data-ui="photo"></textarea></div>
          <div class="iamccs-sheet-field"><label>Medium</label><input data-ui="medium"></div>
        </div>
      </div>
      <div class="iamccs-sheet-mid">
        <div class="iamccs-sheet-board-wrap"><div class="iamccs-sheet-board" data-board></div></div>
        <div class="iamccs-sheet-section">
          <h4>3. Panel List</h4>
          <div class="iamccs-sheet-list" data-list></div>
        </div>
      </div>
      <div class="iamccs-sheet-right">
        <div class="iamccs-sheet-section">
          <h4>4. Selected Panel</h4>
          <div class="iamccs-sheet-row">
            <div class="iamccs-sheet-field"><label>Type</label><select data-box="type"><option>obj</option><option>text</option></select></div>
            <div class="iamccs-sheet-field"><label>Use tips</label><select data-box="useTips"><option value="false">No</option><option value="true">Yes</option></select></div>
            <div class="iamccs-sheet-field"><label>X</label><input data-box="x" type="number" step="0.001"></div>
            <div class="iamccs-sheet-field"><label>Y</label><input data-box="y" type="number" step="0.001"></div>
            <div class="iamccs-sheet-field"><label>W</label><input data-box="w" type="number" step="0.001"></div>
            <div class="iamccs-sheet-field"><label>H</label><input data-box="h" type="number" step="0.001"></div>
          </div>
          <div class="iamccs-sheet-field"><label>Panel title</label><input data-box="title" placeholder="Example: THE BLUE JESTER"></div>
          <div class="iamccs-sheet-field"><label>Text object content</label><input data-box="text"></div>
          <div class="iamccs-sheet-field"><label>Your panel prompt</label><textarea data-box="desc" placeholder="Shot scale + action + environment + camera/lens/film + guard"></textarea></div>
          <div class="iamccs-sheet-actions">
            <button class="iamccs-sheet-btn primary" data-action="openPanelEditor">Open Panel Editor</button>
            <button class="iamccs-sheet-btn" data-action="improveBoxLocal">Structure Panel</button>
          </div>
          <div class="iamccs-sheet-field"><label>Box palette</label><input data-box="palette"></div>
        </div>
        <div class="iamccs-sheet-section">
          <h4>5. Final JSON Preview</h4>
          <textarea class="iamccs-sheet-code" data-json readonly></textarea>
        </div>
      </div>
    </div>
    <div class="iamccs-sheet-bottom">
      <span data-size></span>
      <span>Outputs: prompt, preview, bboxes, width, height, current_sheet_json, grid, ideoboard_for_frame_v2</span>
    </div>
  `;
  host.appendChild(root);

  const q = (sel) => root.querySelector(sel);
  const qa = (sel) => Array.from(root.querySelectorAll(sel));
  const board = q("[data-board]");
  const list = q("[data-list]");
  const status = q("[data-status]");
  const jsonPreview = q("[data-json]");
  const sizeLabel = q("[data-size]");

  function selectedBox() {
    if (!state.boxes.length) state.boxes = defaultBoxes();
    state.selected = Math.max(0, Math.min(state.boxes.length - 1, state.selected));
    return state.boxes[state.selected];
  }

  function persist() {
    setWidget(node, "width", Number(q('[data-ui="width"]').value || widget(node, "width")?.value || 1024));
    setWidget(node, "height", Number(q('[data-ui="height"]').value || widget(node, "height")?.value || 1024));
    setStyleKind(node, q('[data-ui="style"]').value);
    setWidget(node, "high_level_description", q('[data-ui="high"]').value);
    setWidget(node, "background", q('[data-ui="background"]').value);
    setWidget(node, "aesthetics", q('[data-ui="aesthetics"]').value);
    setWidget(node, "lighting", q('[data-ui="lighting"]').value);
    const photoValue = q('[data-ui="photo"]').value;
    setWidget(node, "photo", photoValue);
    setWidget(node, "art_style", photoValue);
    setWidget(node, "medium", q('[data-ui="medium"]').value || "photograph");
    setWidget(node, "style_palette_data", JSON.stringify(paletteFromText(q('[data-ui="palette"]').value)));
    setWidget(node, "elements_data", JSON.stringify(state.boxes));
    const preset = GRID_PRESETS[q('[data-ui="gridPreset"]')?.value] || {};
    const inferredCols = Math.max(1, ...state.boxes.map((box) => Math.round(1 / Math.max(0.001, box.w))).filter(Boolean));
    const inferredRows = Math.max(1, ...state.boxes.map((box) => Math.round(1 / Math.max(0.001, box.h))).filter(Boolean));
    setWidget(node, "grid_columns", Number(preset.boxes?.[0] || inferredCols || 1));
    setWidget(node, "grid_rows", Number(preset.boxes?.[1] || inferredRows || 1));
  }

  function loadFromWidgets() {
    q('[data-ui="width"]').value = widget(node, "width")?.value ?? 1024;
    q('[data-ui="height"]').value = widget(node, "height")?.value ?? 1024;
    q('[data-ui="style"]').value = currentStyleKind(node);
    q('[data-ui="high"]').value = widget(node, "high_level_description")?.value || "";
    q('[data-ui="background"]').value = widget(node, "background")?.value || "";
    q('[data-ui="aesthetics"]').value = widget(node, "aesthetics")?.value || "";
    q('[data-ui="lighting"]').value = widget(node, "lighting")?.value || "";
    q('[data-ui="photo"]').value = widget(node, "photo")?.value || widget(node, "art_style")?.value || "";
    q('[data-ui="medium"]').value = widget(node, "medium")?.value || "photograph";
    q('[data-ui="palette"]').value = parseList(widget(node, "style_palette_data")?.value, STYLE_PRESETS.vintage_fantasy.palette).join(", ");
  }

  function renderStylePresetOptions() {
    const select = q('[data-ui="stylePreset"]');
    select.innerHTML = "";
    Object.entries(STYLE_PRESETS).forEach(([key, preset]) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = preset.label;
      select.appendChild(option);
    });
    select.value = "cinematic_scifi";
  }

  function renderGridPresetOptions() {
    const select = q('[data-ui="gridPreset"]');
    select.innerHTML = "";
    Object.entries(GRID_PRESETS).forEach(([key, preset]) => {
      const option = document.createElement("option");
      option.value = key;
      option.textContent = preset.label;
      select.appendChild(option);
    });
    select.value = "story_2x3";
  }

  function renderBoard() {
    const width = Number(q('[data-ui="width"]').value || 1024);
    const height = Number(q('[data-ui="height"]').value || 1024);
    sizeLabel.textContent = `${width} x ${height}`;
    board.innerHTML = "";
    state.boxes.forEach((box, index) => {
      const div = document.createElement("div");
      div.className = `iamccs-sheet-box ${index === state.selected ? "selected" : ""}`;
      div.style.left = `${clamp(box.x, 0, 1, 0) * 100}%`;
      div.style.top = `${clamp(box.y, 0, 1, 0) * 100}%`;
      div.style.width = `${clamp(box.w, 0.01, 1, 0.2) * 100}%`;
      div.style.height = `${clamp(box.h, 0.01, 1, 0.2) * 100}%`;
      div.style.borderColor = (box.palette && box.palette[0]) || "#C4A15D";
      div.innerHTML = `<div class="iamccs-sheet-box-title"></div><div class="iamccs-sheet-box-body"></div><div class="iamccs-sheet-box-tip"></div>`;
      div.querySelector(".iamccs-sheet-box-title").textContent = `#${index + 1} ${box.title || box.type || "obj"}`;
      div.querySelector(".iamccs-sheet-box-body").textContent = box.desc || "";
      div.querySelector(".iamccs-sheet-box-tip").textContent = box.tips || "";
      div.addEventListener("click", () => {
        state.selected = index;
        render();
      });
      div.addEventListener("dblclick", (event) => {
        event.preventDefault();
        event.stopPropagation();
        openPanelEditor(index);
      });
      board.appendChild(div);
    });
  }

  function renderList() {
    list.innerHTML = "";
    state.boxes.forEach((box, index) => {
      const item = document.createElement("div");
      item.className = `iamccs-sheet-list-item ${index === state.selected ? "selected" : ""}`;
      item.innerHTML = `<strong>${index + 1}. ${box.title || "Untitled panel"}</strong><span></span>`;
      item.querySelector("span").textContent = box.desc || "";
      item.addEventListener("click", () => {
        state.selected = index;
        render();
      });
      list.appendChild(item);
    });
  }

  function renderSelected() {
    const box = selectedBox();
    qa("[data-box]").forEach((input) => {
      const key = input.dataset.box;
      if (key === "palette") {
        input.value = (box.palette || []).join(", ");
      } else if (key === "useTips") {
        input.value = box.useTips ? "true" : "false";
      } else {
        input.value = box[key] ?? "";
      }
    });
  }

  function renderSwatches() {
    const host = q("[data-swatches]");
    host.innerHTML = "";
    paletteFromText(q('[data-ui="palette"]').value).forEach((color) => {
      const sw = document.createElement("span");
      sw.className = "iamccs-sheet-swatch";
      sw.style.background = color;
      host.appendChild(sw);
    });
  }


  function openPanelEditor(index = state.selected) {
    state.selected = Math.max(0, Math.min(state.boxes.length - 1, index));
    const box = selectedBox();
    const backdrop = document.createElement("div");
    backdrop.className = "iamccs-sheet-editor-backdrop";
    const escape = (value) => String(value ?? "").replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
    backdrop.innerHTML = `
      <div class="iamccs-sheet-editor" role="dialog" aria-modal="true">
        <div class="iamccs-sheet-editor-head">
          <strong>Panel ${state.selected + 1} Editor</strong>
          <button class="iamccs-sheet-btn" data-editor="close">Close</button>
        </div>
        <div class="iamccs-sheet-editor-body">
          <div class="iamccs-sheet-editor-field"><label>Panel title</label><input data-editor="title" value="${escape(box.title || "")}"></div>
          <div class="iamccs-sheet-editor-field"><label>Type</label><select data-editor="type"><option value="obj">Object / Scene</option><option value="text">Text Layer</option></select></div>
          <div class="iamccs-sheet-editor-field full"><label>Panel prompt</label><textarea data-editor="desc">${escape(box.desc || "")}</textarea></div>
          <div class="iamccs-sheet-editor-field full"><label>Text layer content</label><input data-editor="text" value="${escape(box.text || "")}"></div>
          <div class="iamccs-sheet-editor-field full"><label>Palette</label><input data-editor="palette" value="${escape((box.palette || []).join(", "))}"></div>
        </div>
        <div class="iamccs-sheet-editor-foot">
          <button class="iamccs-sheet-btn" data-editor="structure">Structure Panel</button>
          <button class="iamccs-sheet-btn" data-editor="delete">Delete</button>
          <button class="iamccs-sheet-btn primary" data-editor="save">Save Panel</button>
        </div>
      </div>`;
    document.body.appendChild(backdrop);
    backdrop.querySelector('[data-editor="type"]').value = box.type === "text" ? "text" : "obj";
    const title = backdrop.querySelector('[data-editor="title"]');
    const desc = backdrop.querySelector('[data-editor="desc"]');
    const textInput = backdrop.querySelector('[data-editor="text"]');
    const type = backdrop.querySelector('[data-editor="type"]');
    const palette = backdrop.querySelector('[data-editor="palette"]');
    const close = () => backdrop.remove();
    const save = () => {
      box.title = title.value;
      box.desc = desc.value;
      box.text = textInput.value;
      box.type = type.value;
      box.palette = paletteFromText(palette.value);
      persist();
      render();
      close();
    };
    backdrop.addEventListener("click", (event) => {
      const action = event.target?.dataset?.editor;
      if (!action && event.target === backdrop) close();
      if (action === "close") close();
      if (action === "save") save();
      if (action === "delete") {
        state.boxes.splice(state.selected, 1);
        if (!state.boxes.length) state.boxes = defaultBoxes();
        state.selected = Math.max(0, state.selected - 1);
        persist();
        render();
        close();
      }
      if (action === "structure") {
        const structured = ideogramPanelPrompt({ ...box, title: title.value, desc: desc.value, text: textInput.value, type: type.value, palette: paletteFromText(palette.value) }, state.selected);
        title.value = structured.title || title.value;
        desc.value = structured.desc || desc.value;
        palette.value = (structured.palette || []).join(", ");
      }
    });
    backdrop.addEventListener("keydown", (event) => {
      if (event.key === "Escape") close();
      if ((event.ctrlKey || event.metaKey) && event.key === "Enter") save();
    });
    setTimeout(() => desc.focus(), 0);
  }

  function toggleFullscreen() {
    if (root.classList.contains("is-fullscreen")) {
      root.classList.remove("is-fullscreen");
      const wrapper = root.parentElement;
      host.appendChild(root);
      wrapper?.remove();
      status.textContent = "Editor docked in node";
    } else {
      const wrapper = document.createElement("div");
      wrapper.className = "iamccs-sheet-fullscreen-host";
      document.body.appendChild(wrapper);
      wrapper.appendChild(root);
      root.classList.add("is-fullscreen");
      status.textContent = "Fullscreen editor open";
    }
    app.graph?.setDirtyCanvas?.(true, true);
  }

  function renderJson() {
    persist();
    jsonPreview.value = JSON.stringify(buildCaption(node, state.boxes), null, 2);
  }

  function render() {
    renderBoard();
    renderList();
    renderSelected();
    renderSwatches();
    renderJson();
  }

  renderStylePresetOptions();
  renderGridPresetOptions();
  loadFromWidgets();
  if (!parseList(widget(node, "style_palette_data")?.value, []).length) {
    const p = STYLE_PRESETS.vintage_fantasy;
    q('[data-ui="palette"]').value = p.palette.join(", ");
    q('[data-ui="style"]').value = p.style;
    if (p.high) q('[data-ui="high"]').value = p.high;
    if (p.background) q('[data-ui="background"]').value = p.background;
    q('[data-ui="aesthetics"]').value = p.aesthetics;
    q('[data-ui="lighting"]').value = p.lighting;
    q('[data-ui="photo"]').value = p.photo;
    q('[data-ui="medium"]').value = p.medium;
    if (Array.isArray(p.panelTips)) {
      state.boxes = state.boxes.map((box, index) => ({ ...box, tips: p.panelTips[index % p.panelTips.length] || box.tips || "" }));
    }
  }
  persist();

  q('[data-ui="stylePreset"]').addEventListener("change", (event) => {
    const preset = STYLE_PRESETS[event.target.value] || STYLE_PRESETS.none;
    q('[data-ui="style"]').value = preset.style;
    if (preset.high) q('[data-ui="high"]').value = preset.high;
    if (preset.background) q('[data-ui="background"]').value = preset.background;
    q('[data-ui="aesthetics"]').value = preset.aesthetics;
    q('[data-ui="lighting"]').value = preset.lighting;
    q('[data-ui="photo"]').value = preset.photo;
    q('[data-ui="medium"]').value = preset.medium;
    q('[data-ui="palette"]').value = preset.palette.join(", ");
    if (Array.isArray(preset.panelTips)) {
      state.boxes = state.boxes.map((box, index) => ({ ...box, tips: preset.panelTips[index % preset.panelTips.length] || box.tips || "" }));
    }
    status.textContent = `${preset.label} applied`;
    render();
  });

  q('[data-ui="gridPreset"]').addEventListener("change", (event) => {
    const preset = GRID_PRESETS[event.target.value] || GRID_PRESETS.free;
    q('[data-ui="width"]').value = preset.width;
    q('[data-ui="height"]').value = preset.height;
    if (preset.boxes) state.boxes = makeGridBoxes(preset.boxes[0], preset.boxes[1], preset.order || "row_major");
    state.selected = 0;
    status.textContent = `${preset.label} layout applied`;
    render();
  });

  async function importPresetFile(kind, file) {
    const raw = await file.text();
    const data = JSON.parse(raw);
    const key = data.key || file.name.replace(/\.json$/i, "").replace(/[^a-z0-9_]+/gi, "_").toLowerCase();
    if (kind === "style") {
      STYLE_PRESETS[key] = data;
      renderStylePresetOptions();
      q('[data-ui="stylePreset"]').value = key;
      q('[data-ui="stylePreset"]').dispatchEvent(new Event("change"));
    } else {
      GRID_PRESETS[key] = data;
      renderGridPresetOptions();
      q('[data-ui="gridPreset"]').value = key;
      q('[data-ui="gridPreset"]').dispatchEvent(new Event("change"));
    }
    status.textContent = `${data.label || key} imported`;
  }

  q('[data-file="stylePreset"]').addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) importPresetFile("style", file).catch((err) => { status.textContent = `Style import error: ${err.message || err}`; });
    event.target.value = "";
  });

  q('[data-file="gridPreset"]').addEventListener("change", (event) => {
    const file = event.target.files?.[0];
    if (file) importPresetFile("grid", file).catch((err) => { status.textContent = `Grid import error: ${err.message || err}`; });
    event.target.value = "";
  });

  qa("[data-ui]").forEach((input) => {
    if (["stylePreset", "gridPreset"].includes(input.dataset.ui)) return;
    input.addEventListener("input", render);
    input.addEventListener("change", render);
  });

  qa("[data-box]").forEach((input) => {
    input.addEventListener("input", () => {
      const box = selectedBox();
      const key = input.dataset.box;
      if (["x", "y", "w", "h"].includes(key)) {
        box[key] = Number(input.value);
      } else if (key === "palette") {
        box.palette = paletteFromText(input.value);
      } else if (key === "useTips") {
        box.useTips = input.value === "true";
      } else {
        box[key] = input.value;
      }
      render();
    });
    input.addEventListener("change", () => input.dispatchEvent(new Event("input")));
  });

  root.querySelectorAll("[data-action]").forEach((button) => {
    button.addEventListener("click", () => {
      const action = button.dataset.action;
      if (action === "importStylePreset") {
        q('[data-file="stylePreset"]').click();
      } else if (action === "importGridPreset") {
        q('[data-file="gridPreset"]').click();
      } else if (action === "add") {
        state.boxes.push({ x: 0.05, y: 0.05, w: 0.25, h: 0.2, type: "obj", text: "", desc: "New object region.", tips: "", useTips: false, palette: paletteFromText(q('[data-ui="palette"]').value).slice(0, 4) });
        state.selected = state.boxes.length - 1;
      } else if (action === "duplicate") {
        const copy = JSON.parse(JSON.stringify(selectedBox()));
        copy.x = clamp(copy.x + 0.03, 0, 0.95, copy.x);
        copy.y = clamp(copy.y + 0.03, 0, 0.95, copy.y);
        state.boxes.splice(state.selected + 1, 0, copy);
        state.selected += 1;
      } else if (action === "delete") {
        state.boxes.splice(state.selected, 1);
        if (!state.boxes.length) state.boxes = defaultBoxes();
        state.selected = Math.max(0, state.selected - 1);
      } else if (action === "copy") {
        navigator.clipboard?.writeText(JSON.stringify(buildCaption(node, state.boxes), null, 2));
        status.textContent = "JSON copied";
      } else if (action === "improveAllLocal") {
        state.boxes = state.boxes.map((box, index) => ideogramPanelPrompt(box, index));
        status.textContent = "All panel prompts improved locally with Ideogram 4 structure";
      } else if (action === "improveBoxLocal") {
        state.boxes[state.selected] = ideogramPanelPrompt(selectedBox(), state.selected);
        status.textContent = `Box ${state.selected + 1} improved locally with Ideogram 4 structure`;
      } else if (action === "openPanelEditor") {
        openPanelEditor(state.selected);
      } else if (action === "toggleFullscreen") {
        toggleFullscreen();
      } else if (action === "apply") {
        persist();
        status.textContent = "Applied to Ideogram-compatible widgets";
      }
      render();
    });
  });

  const originalOnExecuted = node.onExecuted;
  node.onExecuted = function (message) {
    const result = originalOnExecuted?.apply(this, arguments);
    const caption = Array.isArray(message?.caption) ? message.caption[0] : message?.caption;
    const boxes = Array.isArray(message?.boxes) ? message.boxes[0] : message?.boxes;
    if (caption) {
      try {
        state.boxes = captionToBoxes(JSON.parse(caption));
        state.selected = 0;
      } catch {}
    } else if (boxes) {
      try {
        const parsed = JSON.parse(boxes);
        if (Array.isArray(parsed)) state.boxes = parsed;
      } catch {}
    }
    render();
    return result;
  };

  render();
  const domWidget = node.addDOMWidget("IAMCCS Ideogram Sheet Builder", "iamccs_ideogram_sheet_builder", host, { serialize: false });
  const desiredSize = [1320, 1010];
  domWidget.computeSize = () => desiredSize;

  function fitNodeToSheet() {
    const current = node.size || [0, 0];
    if (current[0] < desiredSize[0] || current[1] < desiredSize[1]) {
      if (typeof node.setSize === "function") {
        node.setSize([Math.max(current[0], desiredSize[0]), Math.max(current[1], desiredSize[1])]);
      } else {
        node.size = [Math.max(current[0], desiredSize[0]), Math.max(current[1], desiredSize[1])];
      }
      node.setDirtyCanvas?.(true, true);
      app.graph?.setDirtyCanvas?.(true, true);
    }
  }

  fitNodeToSheet();
  setTimeout(fitNodeToSheet, 50);
  setTimeout(fitNodeToSheet, 250);
}

app.registerExtension({
  name: "IAMCCS.IdeogramSheetBuilder",
  nodeCreated(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || "";
    if (type === TYPE || node?.type === TYPE) [0, 150, 500].forEach((delay) => setTimeout(() => install(node), delay));
  },
  loadedGraphNode(node) {
    const type = node?.comfyClass || node?.type || node?.constructor?.type || "";
    if (type === TYPE || node?.type === TYPE) [0, 150, 500].forEach((delay) => setTimeout(() => install(node), delay));
  },
  async beforeRegisterNodeDef(nodeType, nodeData) {
    if (nodeData?.name !== TYPE) return;
    const original = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
      original?.apply(this, arguments);
      [0, 150, 500].forEach((delay) => setTimeout(() => install(this), delay));
    };
  }
});

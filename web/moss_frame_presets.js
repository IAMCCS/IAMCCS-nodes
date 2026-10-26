// Plain scene text, adapted from goyAIcanvas Advanced G71 and photographic practice.
// Applying a preset replaces only its named field; no prompt wrappers or dependencies.
export const LIGHT_PRESETS = [
  ["Soft studio light", "Large diffused softbox above and to camera left, gentle fill from a white reflector, soft-edged shadows and restrained specular highlights."],
  ["Window light", "Soft daylight from a large window on camera left, gradual shadow falloff across the face, a softly lit room behind the subject."],
  ["Backlight", "Low sun behind the subject creates a bright edge around hair and shoulders; gentle reflected fill keeps the face readable."],
  ["Studio rim light", "A narrow strip light behind the subject outlines the shoulders and hair; a soft frontal key preserves facial detail against a darker background."],
  ["Rembrandt portrait", "A soft key light high on camera left forms a small triangle of light on the shadow-side cheek; subdued fill and a dark background."],
  ["Butterfly beauty", "A large diffused key light just above the lens produces a small butterfly shadow beneath the nose; a white reflector softly fills the chin."],
  ["Overcast daylight", "Soft overcast daylight, consistent illumination across the scene, low contrast and subtle contact shadows."],
  ["Hard noon sunlight", "Direct overhead midday sunlight, crisp short shadows, bright highlights and strong separation between sunlit and shaded surfaces."],
  ["Low-key studio", "A small directional side light illuminates one side of the subject; deep shadows, minimal fill and controlled highlights."],
  ["Classic noir", "Hard side light through venetian blinds casts distinct striped shadows, deep blacks and sharply separated pools of light."],
  ["Practical night light", "Warm light from visible tungsten lamps illuminates the subject, with cooler ambient night light in the background and natural exposure falloff."],
  ["Neon noir", "Magenta and cyan neon signs light the scene from opposite sides, with colored reflections on wet surfaces and dark unlit areas."],
];
export const CAMERA_PRESETS = [
  ["24mm environmental", "Eye-level environmental photograph with a 24mm lens, broad spatial context, visible foreground and background, moderate depth of field."],
  ["35mm documentary", "Observational 35mm photograph, candid framing, natural perspective, readable surroundings and restrained fine grain."],
  ["50mm natural", "Eye-level photograph with a 50mm lens, balanced geometry, focus on the subject and natural background separation."],
  ["85mm portrait", "Portrait photographed with an 85mm lens, gentle perspective compression, sharp eyes, natural skin texture and a softly defocused background."],
  ["100mm macro", "Close detail photographed with a 100mm macro lens, a narrow plane of focus, visible surface texture and gradual optical falloff."],
  ["135mm telephoto", "Photograph with a 135mm lens, compressed background layers, subject isolation and shallow depth of field."],
  ["Cinema scope", "Wide anamorphic cinema composition, deliberate foreground and background staging, subtle horizontal flare and gentle oval background bokeh."],
  ["Deep-focus tableau", "Static wide cinematic tableau, stopped-down lens, foreground and background both readable, precise geometric staging."],
  ["Handheld 35mm film", "Handheld 35mm motion-picture frame, slightly imperfect framing, fine organic grain, restrained halation and soft highlight rolloff."],
  ["Smartphone candid", "Candid smartphone photograph, deep depth of field, natural colors, ordinary framing and available-light exposure."],
];
export const STYLE_PRESETS = [
  ["Photographic natural", "A natural unretouched photograph with visible skin pores, small surface imperfections and physically plausible materials."],
  ["Editorial neutral", "Neutral editorial photography, natural skin reproduction, balanced color and restrained microcontrast."],
  ["Documentary realism", "Observational documentary photograph, ordinary real-world surfaces, lived-in details and candid human presence."],
  ["Fine film grain", "Photographic 35mm film texture, fine organic grain, soft highlight rolloff and subtle analog color variation."],
  ["European New Wave", "Black-and-white European New Wave cinema, naturalistic staging, visible film grain and restrained tonal contrast."],
  ["Grunge realism", "Gritty photographic realism, worn materials, small imperfections, rough surface texture and subdued color."],
  ["Surreal practical cinema", "A live-action surreal film scene built with tangible physical sets, practical props and real costumes, photographed as material objects."],
  ["Dark surrealism", "Dark surreal cinema, symbolic staging, dense atmosphere, muted color and dreamlike ambiguity."],
  ["2D anime / cel shading", "Hand-drawn 2D anime illustration, clean ink outlines, flat cel-shaded color areas and simplified drawn textures."],
  ["Manga ink", "Black-and-white manga illustration, crisp ink contours, hatching and flat graphic shadows."],
  ["Watercolor", "Watercolor illustration on textured paper, translucent pigment, irregular washes and softly bleeding edges."],
  ["Oil painting", "Oil painting with visible layered brushstrokes, textured canvas and richly mixed pigments."],
];

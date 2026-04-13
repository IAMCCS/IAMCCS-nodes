import { app } from "../../scripts/app.js";

const TARGET_NODE = "IAMCCS_QE_PromptEnhancer";
const PREMIUM_STYLE_ID = "iamccs-qe-premium-camera-style";

const CAMERA_ANGLE_META = {
    "Back View": { family: "reverse", tone: "wine", icon: "back" },
    "Front View": { family: "framing", tone: "ivory", icon: "front" },
    "Side View": { family: "framing", tone: "gold", icon: "side" },
    "Left 45°": { family: "rotation", tone: "copper", icon: "left45" },
    "Right 45°": { family: "rotation", tone: "copper", icon: "right45" },
    "90° Left": { family: "rotation", tone: "gold", icon: "left90" },
    "90° Right": { family: "rotation", tone: "gold", icon: "right90" },
    "Low Angle": { family: "vertical", tone: "copper", icon: "low" },
    "High Angle": { family: "vertical", tone: "ivory", icon: "high" },
    "Worm's Eye": { family: "vertical", tone: "teal", icon: "worm" },
    "Bird's Eye": { family: "vertical", tone: "ivory", icon: "bird" },
    "Zoom Out": { family: "lens", tone: "gold", icon: "zoomout" },
    "Zoom In": { family: "lens", tone: "copper", icon: "zoomin" },
    "Flip Scene": { family: "reverse", tone: "wine", icon: "flip" },
    "Other Side": { family: "reverse", tone: "teal", icon: "otherside" },
    "Behind Head": { family: "follow", tone: "ivory", icon: "headback" },
    "Behind Back": { family: "follow", tone: "wine", icon: "backfollow" },
    "Reverse View": { family: "reverse", tone: "wine", icon: "reverse" },
    "3/4 View": { family: "rotation", tone: "ivory", icon: "threequarter" },
    "Close Up": { family: "lens", tone: "copper", icon: "close" },
    "Wide Angle": { family: "lens", tone: "gold", icon: "wide" },
    "Ultra Wide Angle": { family: "lens", tone: "teal", icon: "ultrawide" },
    "Ultra Close Up": { family: "lens", tone: "copper", icon: "ultraclose" },
    "Fish-Eye Lens": { family: "special", tone: "teal", icon: "fisheye" },
};

const CAMERA_ANGLE_COPY = {
    "Back View": "rear-axis frame and silhouette reveal",
    "Front View": "face-on frame and centered lens",
    "Side View": "clean lateral profile and edge line",
    "Left 45°": "soft left three-quarter orbit",
    "Right 45°": "soft right three-quarter orbit",
    "90° Left": "strict left profile rotation",
    "90° Right": "strict right profile rotation",
    "Low Angle": "grounded rise with upward emphasis",
    "High Angle": "top-down read with lifted horizon",
    "Worm's Eye": "heroic underside and vertical reach",
    "Bird's Eye": "overhead read and plan view control",
    "Zoom Out": "expanded context and wider scene read",
    "Zoom In": "tight focus and subject isolation",
    "Flip Scene": "reverse axis and mirrored staging",
    "Other Side": "cross-axis swap to opposing view",
    "Behind Head": "follow frame from the nape line",
    "Behind Back": "rear follow with body-led framing",
    "Reverse View": "full turnaround with backward read",
    "3/4 View": "balanced diagonal portrait framing",
    "Close Up": "compressed portrait crop and detail",
    "Wide Angle": "broad field with spatial expansion",
    "Ultra Wide Angle": "extreme spread and edge stretch",
    "Ultra Close Up": "macro-intimate focal pressure",
    "Fish-Eye Lens": "spherical distortion and bowed space",
};

const STYLE_EFFECT_META = {
    "Photorealistic": { family: "style", tone: "ivory", code: "PHO", motif: "style" },
    "Pencil Sketch": { family: "style", tone: "gold", code: "SK", motif: "style" },
    "Line Art": { family: "style", tone: "ivory", code: "LIN", motif: "style" },
    "Sepia Tone": { family: "style", tone: "copper", code: "SEP", motif: "style" },
    "Black & White": { family: "style", tone: "ivory", code: "B&W", motif: "style" },
    "Add Colors": { family: "style", tone: "teal", code: "COL", motif: "style" },
    "Enhance": { family: "style", tone: "gold", code: "ENH", motif: "style" },
    "White": { family: "style", tone: "ivory", code: "WHT", motif: "style" },
    "Brighten": { family: "style", tone: "gold", code: "BRI", motif: "style" },
    "Contrast": { family: "style", tone: "wine", code: "CON", motif: "style" },
    "Color Pop": { family: "style", tone: "teal", code: "POP", motif: "style" },
    "Warm Tone": { family: "style", tone: "copper", code: "WRM", motif: "style" },
    "Cool Tone": { family: "style", tone: "teal", code: "CLD", motif: "style" },
    "Denoise": { family: "style", tone: "ivory", code: "DNO", motif: "style" },
    "Add Detail": { family: "style", tone: "gold", code: "DET", motif: "style" },
    "Smooth": { family: "style", tone: "ivory", code: "SMO", motif: "style" },
    "Stylize": { family: "style", tone: "wine", code: "ART", motif: "style" },
};

const SCENE_CHANGE_META = {
    "Daytime": { family: "scene", tone: "gold", code: "DAY", motif: "scene" },
    "Night": { family: "scene", tone: "teal", code: "NGT", motif: "scene" },
    "Sunrise": { family: "scene", tone: "gold", code: "SUN", motif: "scene" },
    "Sunset": { family: "scene", tone: "copper", code: "DUS", motif: "scene" },
    "Heavy Rain": { family: "scene", tone: "teal", code: "RAN", motif: "scene" },
    "Snow": { family: "scene", tone: "ivory", code: "SNW", motif: "scene" },
    "Fog": { family: "scene", tone: "ivory", code: "FOG", motif: "scene" },
    "Clear Sky": { family: "scene", tone: "ivory", code: "SKY", motif: "scene" },
    "Remove BG": { family: "cleanup", tone: "wine", code: "CUT", motif: "view" },
    "Blur BG": { family: "cleanup", tone: "teal", code: "BLR", motif: "view" },
    "Clean": { family: "cleanup", tone: "ivory", code: "CLN", motif: "effect" },
    "Restore": { family: "cleanup", tone: "gold", code: "RST", motif: "effect" },
    "Upscale": { family: "cleanup", tone: "gold", code: "UP", motif: "view" },
    "Fix Face": { family: "cleanup", tone: "copper", code: "FACE", motif: "portrait" },
    "Fix Hands": { family: "cleanup", tone: "copper", code: "HAND", motif: "portrait" },
    "Lighting": { family: "scene", tone: "gold", code: "LIT", motif: "scene" },
};

const MULTI_IMAGE_META = {
    "Character Sit": { family: "merge", tone: "ivory", code: "SIT", motif: "merge" },
    "Change Dress": { family: "merge", tone: "copper", code: "DRS", motif: "merge" },
    "Swap Outfit": { family: "merge", tone: "gold", code: "OUT", motif: "merge" },
    "Characters Together": { family: "merge", tone: "wine", code: "TEAM", motif: "merge" },
    "Merge Background": { family: "merge", tone: "teal", code: "BG", motif: "merge" },
    "Copy Hair Style": { family: "merge", tone: "copper", code: "HAIR", motif: "portrait" },
    "Transfer Face": { family: "merge", tone: "ivory", code: "XFR", motif: "portrait" },
    "Blend Objects": { family: "merge", tone: "gold", code: "BLD", motif: "merge" },
    "Copy Pose": { family: "merge", tone: "teal", code: "POSE", motif: "merge" },
    "Scene Merge": { family: "merge", tone: "wine", code: "SCN", motif: "merge" },
    "Object Replace": { family: "merge", tone: "copper", code: "SWP", motif: "merge" },
    "Add Character": { family: "merge", tone: "ivory", code: "ADD", motif: "merge" },
    "Swap Accessories": { family: "merge", tone: "gold", code: "ACC", motif: "merge" },
    "Copy Lighting": { family: "merge", tone: "gold", code: "LUX", motif: "scene" },
    "Blend Styles": { family: "merge", tone: "wine", code: "MIX", motif: "style" },
    "Group Portrait": { family: "merge", tone: "ivory", code: "GRP", motif: "portrait" },
};

const ADDITIONAL_EFFECT_META = {
    "Add Motion Blur": { family: "fx", tone: "teal", code: "MB", motif: "effect" },
    "Add Bokeh": { family: "fx", tone: "gold", code: "BOK", motif: "effect" },
    "Add Glow": { family: "fx", tone: "gold", code: "GLW", motif: "effect" },
    "Add Reflection": { family: "fx", tone: "ivory", code: "RFL", motif: "effect" },
    "Add Shadows": { family: "fx", tone: "wine", code: "SHD", motif: "effect" },
    "Film Grain": { family: "fx", tone: "copper", code: "GRN", motif: "cinema" },
    "Vintage Look": { family: "fx", tone: "copper", code: "VNT", motif: "cinema" },
    "HDR Effect": { family: "fx", tone: "gold", code: "HDR", motif: "effect" },
    "Tilt Shift": { family: "fx", tone: "teal", code: "TS", motif: "effect" },
    "Lens Flare": { family: "fx", tone: "gold", code: "LFR", motif: "effect" },
    "Vignette": { family: "fx", tone: "wine", code: "VIG", motif: "cinema" },
    "Chromatic": { family: "fx", tone: "teal", code: "RGB", motif: "effect" },
    "Oil Painting": { family: "fx", tone: "copper", code: "OIL", motif: "style" },
    "Watercolor": { family: "fx", tone: "teal", code: "WAT", motif: "style" },
    "Comic Style": { family: "fx", tone: "wine", code: "CMC", motif: "style" },
    "Anime Style": { family: "fx", tone: "ivory", code: "ANI", motif: "style" },
    "3D Render": { family: "fx", tone: "teal", code: "3D", motif: "effect" },
    "Clay Art": { family: "fx", tone: "copper", code: "CLY", motif: "style" },
    "Neon Glow": { family: "fx", tone: "teal", code: "NEO", motif: "effect" },
    "Double Exposure": { family: "fx", tone: "wine", code: "DXP", motif: "effect" },
};

const OTHER_PROMPT_META = {
    "High Angle": { family: "view", tone: "ivory", code: "HIG", motif: "view" },
    "Low Angle": { family: "view", tone: "copper", code: "LOW", motif: "view" },
    "Side Angle": { family: "view", tone: "gold", code: "SID", motif: "view" },
    "Ultra Low": { family: "view", tone: "wine", code: "ULO", motif: "view" },
    "Bird House": { family: "view", tone: "ivory", code: "ROOF", motif: "scene" },
    "Close Portrait": { family: "view", tone: "copper", code: "CLS", motif: "portrait" },
    "Over Shoulder": { family: "view", tone: "ivory", code: "OTS", motif: "portrait" },
    "Dutch Angle": { family: "view", tone: "wine", code: "DUT", motif: "view" },
    "Soft Focus": { family: "view", tone: "ivory", code: "SOF", motif: "effect" },
    "Symmetry View": { family: "view", tone: "gold", code: "SYM", motif: "view" },
    "Panoramic View": { family: "view", tone: "teal", code: "PAN", motif: "view" },
    "Macro Detail": { family: "view", tone: "copper", code: "MAC", motif: "portrait" },
    "Reflected View": { family: "view", tone: "ivory", code: "REF", motif: "effect" },
    "Motion Shot": { family: "view", tone: "teal", code: "MOT", motif: "effect" },
    "Top Framing": { family: "view", tone: "gold", code: "TOP", motif: "view" },
    "Cinematic Crop": { family: "view", tone: "wine", code: "CRP", motif: "cinema" },
};

const TRAVEL_META = {
    "Airplane": { family: "travel", tone: "ivory", code: "AIR", motif: "travel" },
    "Train": { family: "travel", tone: "gold", code: "TRN", motif: "travel" },
    "Tokyo Street": { family: "travel", tone: "teal", code: "TKY", motif: "travel" },
    "New York": { family: "travel", tone: "wine", code: "NYC", motif: "travel" },
    "Paris Café": { family: "travel", tone: "copper", code: "PAR", motif: "travel" },
    "Desert Sunset": { family: "travel", tone: "gold", code: "DST", motif: "travel" },
    "Beach": { family: "travel", tone: "teal", code: "BCH", motif: "travel" },
    "Forest": { family: "travel", tone: "ivory", code: "FOR", motif: "travel" },
    "Waterfall": { family: "travel", tone: "teal", code: "WTR", motif: "travel" },
    "Snowy Village": { family: "travel", tone: "ivory", code: "VIL", motif: "travel" },
    "Rome": { family: "travel", tone: "copper", code: "ROM", motif: "travel" },
    "Countryside Bus": { family: "travel", tone: "gold", code: "BUS", motif: "travel" },
    "Airport": { family: "travel", tone: "ivory", code: "APT", motif: "travel" },
    "Mountain Peak": { family: "travel", tone: "gold", code: "MTN", motif: "travel" },
    "Moroccan Bazaar": { family: "travel", tone: "copper", code: "BZR", motif: "travel" },
    "Venice Gondola": { family: "travel", tone: "teal", code: "VEN", motif: "travel" },
};

const CINEMATIC_LOOK_META = {
    "Realistic Cinematic Shot": { family: "cinema", tone: "gold", code: "CSH", motif: "cinema" },
    "Kodak 5219 Look": { family: "cinema", tone: "copper", code: "K21", motif: "cinema" },
    "Bleach Bypass": { family: "cinema", tone: "ivory", code: "BLC", motif: "cinema" },
    "Anamorphic Lens": { family: "cinema", tone: "teal", code: "ANA", motif: "cinema" },
    "Fujifilm Stock": { family: "cinema", tone: "ivory", code: "FUJ", motif: "cinema" },
    "Lens Flare": { family: "cinema", tone: "gold", code: "LFL", motif: "cinema" },
    "Vintage 35mm": { family: "cinema", tone: "copper", code: "35M", motif: "cinema" },
    "Tungsten Light": { family: "cinema", tone: "copper", code: "TNG", motif: "cinema" },
    "Cold Daylight": { family: "cinema", tone: "teal", code: "CDL", motif: "cinema" },
    "High Key Film": { family: "cinema", tone: "ivory", code: "HKF", motif: "cinema" },
    "Teal-Orange": { family: "cinema", tone: "teal", code: "TEO", motif: "cinema" },
    "Tilt-Shift": { family: "cinema", tone: "teal", code: "TSF", motif: "cinema" },
    "Push Processed": { family: "cinema", tone: "wine", code: "PSH", motif: "cinema" },
    "Cinematic LUT": { family: "cinema", tone: "gold", code: "LUT", motif: "cinema" },
    "Handheld 16mm": { family: "cinema", tone: "copper", code: "16M", motif: "cinema" },
    "Underexposed Film": { family: "cinema", tone: "wine", code: "UND", motif: "cinema" },
    "Soft Vignette": { family: "cinema", tone: "ivory", code: "SVG", motif: "cinema" },
    "High ISO": { family: "cinema", tone: "ivory", code: "ISO", motif: "cinema" },
    "Polaroid Look": { family: "cinema", tone: "copper", code: "POL", motif: "cinema" },
    "Cinematic Depth": { family: "cinema", tone: "gold", code: "DOF", motif: "cinema" },
};

const DATASET_GENERATOR_1_META = {
    "Dark Background": { family: "dataset", tone: "wine", code: "DBG", motif: "portrait" },
    "Photo Angles": { family: "dataset", tone: "ivory", code: "ANG", motif: "view" },
    "Luxury Studio": { family: "dataset", tone: "gold", code: "LUX", motif: "portrait" },
    "Neon Cinematic": { family: "dataset", tone: "teal", code: "NEO", motif: "effect" },
    "Nightclub": { family: "dataset", tone: "wine", code: "NIT", motif: "effect" },
    "Backlit Portrait": { family: "dataset", tone: "gold", code: "HAL", motif: "scene" },
    "Underlight": { family: "dataset", tone: "copper", code: "ULT", motif: "scene" },
    "Golden Hour": { family: "dataset", tone: "gold", code: "GHR", motif: "scene" },
    "High Fashion": { family: "dataset", tone: "copper", code: "HFS", motif: "portrait" },
    "Urban Street": { family: "dataset", tone: "teal", code: "URB", motif: "travel" },
    "Soft Natural": { family: "dataset", tone: "ivory", code: "NAT", motif: "portrait" },
    "Fantasy Light": { family: "dataset", tone: "teal", code: "FNT", motif: "effect" },
    "Product Clean": { family: "dataset", tone: "ivory", code: "PRD", motif: "view" },
    "Dramatic Side": { family: "dataset", tone: "wine", code: "DRM", motif: "portrait" },
    "Foggy Mood": { family: "dataset", tone: "ivory", code: "FOG", motif: "scene" },
    "Vintage Analog": { family: "dataset", tone: "copper", code: "VGA", motif: "cinema" },
};

const DATASET_GENERATOR_2_META = {
    "Full Front Neutral": { family: "pose", tone: "ivory", code: "FRT", motif: "portrait" },
    "Full Back Neutral": { family: "pose", tone: "wine", code: "BCK", motif: "portrait" },
    "Left Profile": { family: "pose", tone: "gold", code: "LPR", motif: "view" },
    "Right Profile": { family: "pose", tone: "gold", code: "RPR", motif: "view" },
    "T-Pose Front": { family: "pose", tone: "ivory", code: "TPF", motif: "portrait" },
    "T-Pose Back": { family: "pose", tone: "wine", code: "TPB", motif: "portrait" },
    "A-Pose Relaxed": { family: "pose", tone: "copper", code: "APS", motif: "portrait" },
    "Walk Cycle Step": { family: "pose", tone: "teal", code: "WLK", motif: "portrait" },
    "Seated Neutral": { family: "pose", tone: "ivory", code: "SIT", motif: "portrait" },
    "Arms Raised": { family: "pose", tone: "gold", code: "ARM", motif: "portrait" },
    "Hands Visible": { family: "pose", tone: "copper", code: "HND", motif: "portrait" },
    "3/4 Turnaround": { family: "pose", tone: "teal", code: "TRN", motif: "view" },
    "Headshot Neutral": { family: "pose", tone: "ivory", code: "HSN", motif: "portrait" },
    "Expression Sheet": { family: "pose", tone: "wine", code: "EXP", motif: "portrait" },
    "Clean White Studio": { family: "pose", tone: "ivory", code: "WHT", motif: "view" },
    "Neutral Gray Studio": { family: "pose", tone: "ivory", code: "GRY", motif: "view" },
};

const DATASET_GENERATOR_3_META = {
    "Soft Smile Close-Up": { family: "study", tone: "gold", code: "SSM", motif: "portrait" },
    "Serious Close-Up": { family: "study", tone: "wine", code: "SER", motif: "portrait" },
    "Surprised Face": { family: "study", tone: "copper", code: "SPR", motif: "portrait" },
    "Eyes Closed Calm": { family: "study", tone: "ivory", code: "CAL", motif: "portrait" },
    "Extreme Face Macro": { family: "study", tone: "teal", code: "XFM", motif: "portrait" },
    "Eyes Only Macro": { family: "study", tone: "teal", code: "EYE", motif: "portrait" },
    "Half Body Crop": { family: "study", tone: "ivory", code: "HBC", motif: "portrait" },
    "Torso Detail": { family: "study", tone: "copper", code: "TOR", motif: "portrait" },
    "Hand Detail": { family: "study", tone: "gold", code: "HDT", motif: "portrait" },
    "Legs Detail": { family: "study", tone: "gold", code: "LEG", motif: "portrait" },
    "Indoor Room": { family: "study", tone: "ivory", code: "IND", motif: "scene" },
    "Urban Exterior": { family: "study", tone: "teal", code: "EXT", motif: "travel" },
    "Nature Location": { family: "study", tone: "ivory", code: "NTR", motif: "travel" },
    "Seated Desk Close-Up": { family: "study", tone: "gold", code: "DSK", motif: "portrait" },
    "Crouched Action Pose": { family: "study", tone: "wine", code: "ACT", motif: "portrait" },
    "Overhead Layout": { family: "study", tone: "teal", code: "OVR", motif: "view" },
};

const DATASET_GENERATOR_4_META = {
    "Extreme Backlight Halo": { family: "lighting", tone: "gold", code: "HAL", motif: "scene" },
    "Hard Split Light": { family: "lighting", tone: "wine", code: "SPL", motif: "scene" },
    "Neon Shock": { family: "lighting", tone: "teal", code: "NSK", motif: "effect" },
    "Underlight Horror": { family: "lighting", tone: "copper", code: "UHL", motif: "scene" },
    "Overexposed Window": { family: "lighting", tone: "ivory", code: "OVR", motif: "scene" },
    "Silhouette Smoke": { family: "lighting", tone: "ivory", code: "SMK", motif: "scene" },
    "Nightclub Flash": { family: "lighting", tone: "wine", code: "NCF", motif: "effect" },
    "Prism Fantasy": { family: "lighting", tone: "teal", code: "PRI", motif: "effect" },
    "Fashion Strobe Burst": { family: "lighting", tone: "gold", code: "STR", motif: "portrait" },
    "Wet Street Neon": { family: "lighting", tone: "teal", code: "WET", motif: "travel" },
    "Shadow Veil": { family: "lighting", tone: "wine", code: "SHV", motif: "portrait" },
    "White Cyclorama Flash": { family: "lighting", tone: "ivory", code: "CYC", motif: "view" },
    "Top Beam Halo": { family: "lighting", tone: "gold", code: "TOP", motif: "scene" },
    "Sunset Rim Fire": { family: "lighting", tone: "copper", code: "SUN", motif: "scene" },
    "Noir Analog Shadow": { family: "lighting", tone: "wine", code: "NOI", motif: "cinema" },
    "Color Gel Crossfire": { family: "lighting", tone: "teal", code: "GEL", motif: "effect" },
};

const DATASET_GENERATOR_5_META = {
    "Hands Framing Face": { family: "gesture", tone: "ivory", code: "FRM", motif: "portrait" },
    "Open Palms Forward": { family: "gesture", tone: "gold", code: "PLM", motif: "portrait" },
    "Fingers In Hair": { family: "gesture", tone: "copper", code: "HAI", motif: "portrait" },
    "Laughing Teeth Visible": { family: "gesture", tone: "gold", code: "LAF", motif: "portrait" },
    "Mouth Open Surprise": { family: "gesture", tone: "wine", code: "MTH", motif: "portrait" },
    "Eyes Wide Open": { family: "gesture", tone: "teal", code: "WID", motif: "portrait" },
    "Open Mouth Profile": { family: "gesture", tone: "copper", code: "OMP", motif: "portrait" },
    "Teeth Detail Smile": { family: "gesture", tone: "ivory", code: "TTH", motif: "portrait" },
    "Hands On Cheeks": { family: "gesture", tone: "copper", code: "CHK", motif: "portrait" },
    "Torso Twist Detail": { family: "gesture", tone: "copper", code: "TWS", motif: "portrait" },
    "Shoulder Collarbone Detail": { family: "gesture", tone: "ivory", code: "COL", motif: "portrait" },
    "Back Over Shoulder": { family: "gesture", tone: "wine", code: "BOS", motif: "view" },
    "Waist And Hands Crop": { family: "gesture", tone: "gold", code: "WST", motif: "portrait" },
    "Legs And Stance Detail": { family: "gesture", tone: "gold", code: "STN", motif: "portrait" },
    "Seated Hands Visible": { family: "gesture", tone: "ivory", code: "SEA", motif: "portrait" },
    "Full Body Expressive Gesture": { family: "gesture", tone: "teal", code: "EXP", motif: "portrait" },
};

const DATASET_GENERATOR_6_META = {
    "Front Close-Up": { family: "turnaround", tone: "ivory", code: "FCL", motif: "view" },
    "FR Quarter Close-Up": { family: "turnaround", tone: "gold", code: "FRQ", motif: "view" },
    "Right Side Close-Up": { family: "turnaround", tone: "gold", code: "RCL", motif: "view" },
    "BR Quarter Close-Up": { family: "turnaround", tone: "wine", code: "BRQ", motif: "view" },
    "Back Close-Up": { family: "turnaround", tone: "wine", code: "BCL", motif: "view" },
    "BL Quarter Close-Up": { family: "turnaround", tone: "wine", code: "BLQ", motif: "view" },
    "Left Side Close-Up": { family: "turnaround", tone: "gold", code: "LCL", motif: "view" },
    "FL Quarter Close-Up": { family: "turnaround", tone: "ivory", code: "FLQ", motif: "view" },
    "Front Medium Shot": { family: "turnaround", tone: "ivory", code: "FMD", motif: "view" },
    "FR Quarter Medium": { family: "turnaround", tone: "gold", code: "FRM", motif: "view" },
    "Right Side Medium": { family: "turnaround", tone: "gold", code: "RMD", motif: "view" },
    "BR Quarter Medium": { family: "turnaround", tone: "wine", code: "BRM", motif: "view" },
    "Back Medium Shot": { family: "turnaround", tone: "wine", code: "BMD", motif: "view" },
    "BL Quarter Medium": { family: "turnaround", tone: "wine", code: "BLM", motif: "view" },
    "Left Side Medium": { family: "turnaround", tone: "gold", code: "LMD", motif: "view" },
    "FL Quarter Medium": { family: "turnaround", tone: "ivory", code: "FLM", motif: "view" },
};

const DATASET_GENERATOR_7_META = {
    "Low-Angle Front": { family: "elevation", tone: "copper", code: "LAF", motif: "view" },
    "Low-Angle Right": { family: "elevation", tone: "copper", code: "LAR", motif: "view" },
    "Low-Angle Back": { family: "elevation", tone: "wine", code: "LAB", motif: "view" },
    "Low-Angle Left": { family: "elevation", tone: "copper", code: "LAL", motif: "view" },
    "Eye-Level Front": { family: "elevation", tone: "ivory", code: "ELF", motif: "view" },
    "Eye-Level Right": { family: "elevation", tone: "ivory", code: "ELR", motif: "view" },
    "Eye-Level Back": { family: "elevation", tone: "wine", code: "ELB", motif: "view" },
    "Eye-Level Left": { family: "elevation", tone: "ivory", code: "ELL", motif: "view" },
    "Elevated Front": { family: "elevation", tone: "gold", code: "EVF", motif: "view" },
    "Elevated Right": { family: "elevation", tone: "gold", code: "EVR", motif: "view" },
    "Elevated Back": { family: "elevation", tone: "gold", code: "EVB", motif: "view" },
    "Elevated Left": { family: "elevation", tone: "gold", code: "EVL", motif: "view" },
    "High-Angle Front": { family: "elevation", tone: "teal", code: "HAF", motif: "view" },
    "High-Angle Right": { family: "elevation", tone: "teal", code: "HAR", motif: "view" },
    "High-Angle Back": { family: "elevation", tone: "teal", code: "HAB", motif: "view" },
    "High-Angle Left": { family: "elevation", tone: "teal", code: "HAL", motif: "view" },
};

const DATASET_GENERATOR_8_META = {
    "Worm-Eye Front Close": { family: "cinematic", tone: "copper", code: "WEF", motif: "cinema" },
    "Bird-Eye Front Close": { family: "cinematic", tone: "teal", code: "BEF", motif: "cinema" },
    "Low Front Wide Shot": { family: "cinematic", tone: "copper", code: "LFW", motif: "cinema" },
    "High Front Wide Shot": { family: "cinematic", tone: "teal", code: "HFW", motif: "cinema" },
    "Low FR Quarter Close": { family: "cinematic", tone: "gold", code: "LFR", motif: "cinema" },
    "High FR Quarter Wide": { family: "cinematic", tone: "teal", code: "HFR", motif: "cinema" },
    "Low Right Side Wide": { family: "cinematic", tone: "copper", code: "LRS", motif: "cinema" },
    "High Right Side Close": { family: "cinematic", tone: "teal", code: "HRC", motif: "cinema" },
    "Low Back Close": { family: "cinematic", tone: "wine", code: "LBC", motif: "cinema" },
    "High Back Wide Shot": { family: "cinematic", tone: "teal", code: "HBW", motif: "cinema" },
    "Low FL Quarter Wide": { family: "cinematic", tone: "gold", code: "LFL", motif: "cinema" },
    "High FL Quarter Medium": { family: "cinematic", tone: "teal", code: "HFL", motif: "cinema" },
    "Eye BL Quarter Close": { family: "cinematic", tone: "ivory", code: "EBC", motif: "cinema" },
    "Eye BR Quarter Wide": { family: "cinematic", tone: "ivory", code: "EBW", motif: "cinema" },
    "High Left Side Close": { family: "cinematic", tone: "teal", code: "HLC", motif: "cinema" },
    "Low Left Side Medium": { family: "cinematic", tone: "copper", code: "LLM", motif: "cinema" },
};

const DATASET_GENERATOR_9_META = {
    "360° Eye-Level Medium": { family: "orbit", tone: "ivory", code: "O1", motif: "view" },
    "360° Eye-Level Close-Up": { family: "orbit", tone: "gold", code: "O2", motif: "view" },
    "360° Low-Angle Medium": { family: "orbit", tone: "copper", code: "O3", motif: "view" },
    "360° Elevated Medium": { family: "orbit", tone: "gold", code: "O4", motif: "view" },
    "Vertical Arc Front": { family: "orbit", tone: "ivory", code: "ARC", motif: "view" },
    "Vertical Arc Right": { family: "orbit", tone: "gold", code: "VAR", motif: "view" },
    "Ascending Spiral": { family: "orbit", tone: "teal", code: "SPI", motif: "view" },
    "360° High-Angle Wide": { family: "orbit", tone: "teal", code: "O8", motif: "view" },
};

const DATASET_GENERATOR_10_META = {
    "Macro Face 360°": { family: "macro", tone: "ivory", code: "MF1", motif: "portrait" },
    "Macro Face Elevation": { family: "macro", tone: "gold", code: "MF2", motif: "portrait" },
    "Macro Upper Body 360°": { family: "macro", tone: "ivory", code: "MUB", motif: "portrait" },
    "Macro Right Profile Arc": { family: "macro", tone: "copper", code: "MRP", motif: "portrait" },
    "Macro Low Face": { family: "macro", tone: "copper", code: "MLF", motif: "portrait" },
    "Macro High Face": { family: "macro", tone: "teal", code: "MHF", motif: "portrait" },
    "Macro Hands & Arms": { family: "macro", tone: "gold", code: "MHA", motif: "portrait" },
    "Macro Full Figure 360°": { family: "macro", tone: "teal", code: "MFF", motif: "portrait" },
};

const DATASET_GENERATOR_11_META = {
    "Inspect Front Scan": { family: "inspect", tone: "ivory", code: "IFS", motif: "effect" },
    "Inspect Right Orbit": { family: "inspect", tone: "gold", code: "IRO", motif: "view" },
    "Inspect Left Orbit": { family: "inspect", tone: "gold", code: "ILO", motif: "view" },
    "Inspect Top-Down": { family: "inspect", tone: "teal", code: "ITD", motif: "view" },
    "Inspect Low Probe": { family: "inspect", tone: "copper", code: "ILP", motif: "view" },
    "Inspect Surface Texture": { family: "inspect", tone: "ivory", code: "TEX", motif: "effect" },
    "Inspect Edge & Rim": { family: "inspect", tone: "wine", code: "EDG", motif: "effect" },
    "Inspect Feature Isolate": { family: "inspect", tone: "teal", code: "ISO", motif: "effect" },
};

const DATASET_GENERATOR_12_META = {
    "HRP Face 360°": { family: "hrp", tone: "ivory", code: "HR1", motif: "portrait" },
    "HRP Pore Scan": { family: "hrp", tone: "gold", code: "PORE", motif: "effect" },
    "HRP Moisture Arc": { family: "hrp", tone: "teal", code: "MST", motif: "effect" },
    "HRP Eye Focus": { family: "hrp", tone: "ivory", code: "EYE", motif: "portrait" },
    "HRP Lip Detail": { family: "hrp", tone: "copper", code: "LIP", motif: "portrait" },
    "HRP Wet Hair Contour": { family: "hrp", tone: "teal", code: "WET", motif: "portrait" },
    "HRP Cool Softbox Arc": { family: "hrp", tone: "ivory", code: "BOX", motif: "portrait" },
    "HRP Beauty Editorial Arc": { family: "hrp", tone: "gold", code: "BEA", motif: "portrait" },
};

const DATASET_PRESET_META = [
    ["Dataset generator_1", DATASET_GENERATOR_1_META],
    ["Dataset generator_2", DATASET_GENERATOR_2_META],
    ["Dataset generator_3", DATASET_GENERATOR_3_META],
    ["Dataset generator_4", DATASET_GENERATOR_4_META],
    ["Dataset generator_5", DATASET_GENERATOR_5_META],
    ["Dataset generator_6", DATASET_GENERATOR_6_META],
    ["Dataset generator_7", DATASET_GENERATOR_7_META],
    ["Dataset generator_8", DATASET_GENERATOR_8_META],
    ["Dataset generator_9", DATASET_GENERATOR_9_META],
    ["Dataset generator_10", DATASET_GENERATOR_10_META],
    ["Dataset generator_11", DATASET_GENERATOR_11_META],
    ["Dataset generator_12", DATASET_GENERATOR_12_META],
];

function ensurePremiumStyle() {
    if (document.getElementById(PREMIUM_STYLE_ID)) return;

    const style = document.createElement("style");
    style.id = PREMIUM_STYLE_ID;
    style.textContent = `
            .iamccs-qe-prompt-styles {
                background: #050505;
                box-sizing: border-box;
                padding-right: 2px;
            }

            .iamccs-qe-prompt-styles .tools {
                border-bottom-color: rgba(255, 255, 255, 0.1);
            }

            .iamccs-qe-prompt-styles .tools button.delete,
            .iamccs-qe-prompt-styles .tools textarea.search,
            .iamccs-qe-prompt-styles .tools input.search,
            .iamccs-qe-prompt-styles .tools .character-profile-input {
                background: rgba(255, 255, 255, 0.05);
                border-color: rgba(255, 255, 255, 0.12);
                color: rgba(248, 246, 240, 0.94);
            }

            .iamccs-qe-prompt-styles .tools textarea.search::placeholder,
            .iamccs-qe-prompt-styles .tools input.search::placeholder,
            .iamccs-qe-prompt-styles .tools .character-profile-input::placeholder {
                color: rgba(248, 246, 240, 0.42);
            }

            .iamccs-qe-prompt-styles-list {
                box-sizing: border-box;
                padding: 0 2px 20px 0 !important;
                scrollbar-gutter: stable;
            }

      .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid {
                box-sizing: border-box;
        gap: 10px;
                padding: 0 2px 20px 0 !important;
                scrollbar-gutter: stable;
      }

      .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid .iamccs-qe-prompt-style-card {
        position: relative;
                display: block;
                border-radius: 16px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 12px 26px rgba(0, 0, 0, 0.28);
        overflow: hidden;
                aspect-ratio: 1 / 1.18;
      }

      .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid .iamccs-qe-prompt-style-card::before {
        content: "";
        position: absolute;
        inset: 1px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.04);
        pointer-events: none;
      }

      .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid .iamccs-qe-prompt-style-card:hover {
        transform: translateY(-1px);
        filter: none;
        border-color: rgba(224, 190, 121, 0.28);
      }

      .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid .iamccs-qe-prompt-style-card.selected {
        border-color: rgba(224, 190, 121, 0.52);
        box-shadow: 0 0 0 1px rgba(224, 190, 121, 0.18), 0 14px 28px rgba(0, 0, 0, 0.34);
      }

      .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid .iamccs-qe-prompt-style-image {
                position: absolute;
                inset: 0;
        border: none;
        border-radius: 0;
        background: transparent;
                padding: 0;
                display: block;
      }

      .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid .iamccs-qe-prompt-style-image img {
        display: none;
      }

      .iamccs-qe-premium-art {
                position: absolute;
                inset: 0;
        width: 100%;
        height: 100%;
                min-height: 0;
                border-radius: 14px;
        overflow: hidden;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
      }

      .iamccs-qe-premium-art svg {
        width: 100%;
        height: 100%;
        display: block;
        filter: drop-shadow(0 8px 14px rgba(0,0,0,0.28));
                transform: scale(0.79) translateY(-14px);
                transform-origin: center center;
      }

      .iamccs-qe-premium-chip,
      .iamccs-qe-premium-slot {
        position: absolute;
                top: 10px;
                min-height: 18px;
        display: inline-flex;
        align-items: center;
                padding: 0 9px;
        border-radius: 999px;
                background: rgba(10, 10, 12, 0.62);
                border: 1px solid rgba(255, 255, 255, 0.18);
                font-size: 8px;
                letter-spacing: 0.12em;
        text-transform: uppercase;
        color: rgba(255, 248, 235, 0.92);
        pointer-events: none;
      }

      .iamccs-qe-premium-slot {
                left: 10px;
      }

      .iamccs-qe-premium-chip {
                right: 10px;
      }

      .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid .iamccs-qe-prompt-style-label {
                position: absolute;
                left: 0;
                right: 0;
                bottom: 0;
                z-index: 2;
                background: linear-gradient(180deg, rgba(8, 8, 10, 0), rgba(8, 8, 10, 0.12) 18%, rgba(8, 8, 10, 0.48) 58%, rgba(8, 8, 10, 0.74) 100%);
                padding: 42px 12px 12px;
        text-align: left;
        white-space: normal;
                min-height: 86px;
                box-sizing: border-box;
      }

      .iamccs-qe-premium-title {
        display: block;
        font-family: "Bahnschrift SemiCondensed", "Aptos Display", sans-serif;
                font-size: 14px;
                line-height: 1.08;
        color: #f3ecdf;
        letter-spacing: 0.01em;
      }

      .iamccs-qe-premium-sub {
        display: block;
                margin-top: 6px;
                font-size: 10px;
                line-height: 1.28;
                letter-spacing: 0.01em;
                text-transform: none;
                color: rgba(243, 236, 223, 0.68);
      }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid .iamccs-qe-prompt-style-card:hover .iamccs-qe-prompt-style-image > * {
                transform: none;
            }

      .iamccs-qe-premium-tone-gold .iamccs-qe-premium-art {
        background: radial-gradient(circle at 18% 10%, rgba(224,190,121,0.22), transparent 24%), linear-gradient(160deg, #20170f, #120d09 58%, #090605);
        color: #e0be79;
      }

            .iamccs-qe-premium-tone-gold {
                background: radial-gradient(circle at 18% 10%, rgba(224,190,121,0.22), transparent 24%), linear-gradient(160deg, #20170f, #120d09 58%, #090605);
            }

      .iamccs-qe-premium-tone-copper .iamccs-qe-premium-art {
        background: radial-gradient(circle at 18% 10%, rgba(207,143,87,0.22), transparent 24%), linear-gradient(160deg, #21140e, #170f0b 58%, #090504);
        color: #cf8f57;
      }

            .iamccs-qe-premium-tone-copper {
                background: radial-gradient(circle at 18% 10%, rgba(207,143,87,0.22), transparent 24%), linear-gradient(160deg, #21140e, #170f0b 58%, #090504);
            }

      .iamccs-qe-premium-tone-wine .iamccs-qe-premium-art {
        background: radial-gradient(circle at 18% 10%, rgba(185,103,115,0.2), transparent 24%), linear-gradient(160deg, #1e1013, #140b0d 58%, #080405);
        color: #cf8594;
      }

            .iamccs-qe-premium-tone-wine {
                background: radial-gradient(circle at 18% 10%, rgba(185,103,115,0.2), transparent 24%), linear-gradient(160deg, #1e1013, #140b0d 58%, #080405);
            }

      .iamccs-qe-premium-tone-ivory .iamccs-qe-premium-art {
        background: radial-gradient(circle at 18% 10%, rgba(233,220,194,0.18), transparent 24%), linear-gradient(160deg, #1b1511, #120f0c 58%, #070605);
        color: #ead9b4;
      }

            .iamccs-qe-premium-tone-ivory {
                background: radial-gradient(circle at 18% 10%, rgba(233,220,194,0.18), transparent 24%), linear-gradient(160deg, #1b1511, #120f0c 58%, #070605);
            }

      .iamccs-qe-premium-tone-teal .iamccs-qe-premium-art {
        background: radial-gradient(circle at 18% 10%, rgba(114,181,178,0.2), transparent 24%), linear-gradient(160deg, #101918, #0b1212 58%, #050707);
        color: #7bcfca;
      }

            .iamccs-qe-premium-tone-teal {
                background: radial-gradient(circle at 18% 10%, rgba(114,181,178,0.2), transparent 24%), linear-gradient(160deg, #101918, #0b1212 58%, #050707);
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-prompt-style-card {
                background: linear-gradient(160deg, #08090b, #030304 62%, #000000);
                border-color: rgba(255, 255, 255, 0.14);
                box-shadow: 0 14px 28px rgba(0, 0, 0, 0.52);
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-prompt-style-card::before {
                border-color: rgba(255, 255, 255, 0.08);
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-art,
            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-tone-gold .iamccs-qe-premium-art,
            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-tone-copper .iamccs-qe-premium-art,
            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-tone-wine .iamccs-qe-premium-art,
            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-tone-ivory .iamccs-qe-premium-art,
            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-tone-teal .iamccs-qe-premium-art {
                background: radial-gradient(circle at 18% 10%, rgba(255,255,255,0.1), transparent 24%), linear-gradient(160deg, #0d0e11, #060709 62%, #010102);
                color: #f2f4f6;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-prompt-style-label {
                background: linear-gradient(180deg, rgba(4, 4, 6, 0), rgba(4, 4, 6, 0.18) 16%, rgba(4, 4, 6, 0.58) 58%, rgba(4, 4, 6, 0.86) 100%);
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-title {
                color: #f5f7fa;
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-sub {
                color: rgba(245, 247, 250, 0.72);
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-slot,
            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-chip {
                background: rgba(0, 0, 0, 0.54);
                border-color: rgba(255, 255, 255, 0.22);
                color: rgba(245, 247, 250, 0.92);
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-pattern {
                opacity: 0.16;
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-wire {
                opacity: 0.58;
            }

            .iamccs-qe-prompt-styles-list.iamccs-qe-premium-grid.iamccs-qe-dark-previews .iamccs-qe-premium-core {
                opacity: 1;
            }

      .iamccs-qe-premium-pattern { opacity: 0.28; }
      .iamccs-qe-premium-wire { opacity: 0.7; }
      .iamccs-qe-premium-core { opacity: 0.96; }
    `;
    document.head.appendChild(style);
}

function normalizeLabel(label) {
    const text = String(label || "").trim();
    return text
        .replace(/\s+[\p{Extended_Pictographic}\p{So}\p{Sk}\p{Sm}\uFE0F\u200D]+$/gu, "")
        .replace(/\s{2,}/g, " ");
}

function svgShell(content) {
    return `<svg viewBox="0 0 240 180" preserveAspectRatio="xMidYMid slice" xmlns="http://www.w3.org/2000/svg" fill="none">${content}</svg>`;
}

function patternGrid() {
    return `<g class="iamccs-qe-premium-pattern" stroke="currentColor" stroke-width="1"><path d="M30 42H210"/><path d="M30 78H210"/><path d="M30 114H210"/><path d="M30 150H210"/><path d="M54 24V160"/><path d="M98 24V160"/><path d="M142 24V160"/><path d="M186 24V160"/></g>`;
}

function patternRays() {
    return `<g class="iamccs-qe-premium-pattern" stroke="currentColor" stroke-width="1.2"><path d="M28 148L82 94"/><path d="M48 160L104 104"/><path d="M212 148L158 94"/><path d="M192 160L136 104"/></g>`;
}

function ring(cx, cy, r) {
    return `<circle class="iamccs-qe-premium-wire" cx="${cx}" cy="${cy}" r="${r}" stroke="currentColor" stroke-width="1.5"/>`;
}

function buildIcon(type) {
    switch (type) {
        case "front":
            return svgShell(`${patternGrid()}<rect class="iamccs-qe-premium-core" x="64" y="34" width="112" height="112" rx="26" stroke="currentColor" stroke-width="4"/><circle class="iamccs-qe-premium-core" cx="120" cy="90" r="18" stroke="currentColor" stroke-width="3"/><path class="iamccs-qe-premium-core" d="M120 52V128" stroke="currentColor" stroke-width="3"/>`);
        case "back":
            return svgShell(`${patternRays()}${ring(120,88,54)}<path class="iamccs-qe-premium-core" d="M120 42V126M92 60C92 80 80 96 62 108M148 60C148 80 160 96 178 108" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "side":
            return svgShell(`${patternRays()}<path class="iamccs-qe-premium-core" d="M74 42H156M96 54C140 54 166 78 166 100C166 122 142 136 114 136M74 42V138" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "left45":
            return svgShell(`${ring(120,92,58)}<path class="iamccs-qe-premium-core" d="M150 52C125 48 92 62 74 86C64 100 60 112 60 124" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><path class="iamccs-qe-premium-core" d="M60 124L78 116M60 124L70 106" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "right45":
            return svgShell(`${ring(120,92,58)}<path class="iamccs-qe-premium-core" d="M90 52C115 48 148 62 166 86C176 100 180 112 180 124" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><path class="iamccs-qe-premium-core" d="M180 124L162 116M180 124L170 106" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "left90":
            return svgShell(`${patternGrid()}<path class="iamccs-qe-premium-core" d="M166 54H88V124" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><path class="iamccs-qe-premium-core" d="M88 124L62 98M88 124L114 98" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "right90":
            return svgShell(`${patternGrid()}<path class="iamccs-qe-premium-core" d="M74 54H152V124" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><path class="iamccs-qe-premium-core" d="M152 124L126 98M152 124L178 98" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "low":
            return svgShell(`${patternRays()}<path class="iamccs-qe-premium-core" d="M58 138L120 48L182 138M120 48V136" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "high":
            return svgShell(`${patternRays()}<path class="iamccs-qe-premium-core" d="M58 48L120 138L182 48M120 48V136" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "worm":
            return svgShell(`${ring(120,122,42)}<path class="iamccs-qe-premium-core" d="M120 144V32M84 68L120 32L156 68" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "bird":
            return svgShell(`${patternGrid()}<circle class="iamccs-qe-premium-core" cx="120" cy="92" r="48" stroke="currentColor" stroke-width="3.4"/><circle class="iamccs-qe-premium-core" cx="120" cy="92" r="20" stroke="currentColor" stroke-width="3.4"/>`);
        case "zoomout":
            return svgShell(`${patternRays()}<circle class="iamccs-qe-premium-core" cx="120" cy="92" r="24" stroke="currentColor" stroke-width="4"/><path class="iamccs-qe-premium-core" d="M84 102L54 132M156 102L186 132M104 82L74 52M136 82L166 52" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "zoomin":
            return svgShell(`${patternRays()}<circle class="iamccs-qe-premium-core" cx="120" cy="92" r="24" stroke="currentColor" stroke-width="4"/><path class="iamccs-qe-premium-core" d="M74 46L104 76M166 46L136 76M54 138L84 108M186 138L156 108" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "flip":
            return svgShell(`${ring(120,92,52)}<path class="iamccs-qe-premium-core" d="M68 90C78 58 108 48 132 52M132 52L118 42M172 94C162 126 132 136 108 132M108 132L122 142" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "otherside":
            return svgShell(`${patternGrid()}<path class="iamccs-qe-premium-core" d="M42 92H198M72 64L42 92L72 120M168 64L198 92L168 120" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "headback":
            return svgShell(`${ring(120,88,56)}<circle class="iamccs-qe-premium-core" cx="120" cy="74" r="22" stroke="currentColor" stroke-width="4"/><path class="iamccs-qe-premium-core" d="M92 130C98 108 110 98 120 98C130 98 142 108 148 130" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "backfollow":
            return svgShell(`${patternRays()}<path class="iamccs-qe-premium-core" d="M120 46V120M92 70C92 88 78 108 62 120M148 70C148 88 162 108 178 120M72 138C92 124 104 118 120 118C136 118 148 124 168 138" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "reverse":
            return svgShell(`${ring(120,92,56)}<path class="iamccs-qe-premium-core" d="M48 92H192M82 64L48 92L82 120M158 64L192 92L158 120" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "threequarter":
            return svgShell(`${patternGrid()}<path class="iamccs-qe-premium-core" d="M78 130V62C78 52 86 44 96 44H144" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><path class="iamccs-qe-premium-core" d="M144 44C160 44 172 56 172 72V118" stroke="currentColor" stroke-width="4" stroke-linecap="round" opacity="0.62"/><path class="iamccs-qe-premium-core" d="M112 58L154 82" stroke="currentColor" stroke-width="4" stroke-linecap="round"/><path class="iamccs-qe-premium-core" d="M92 126H154" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "close":
            return svgShell(`${patternGrid()}<rect class="iamccs-qe-premium-core" x="56" y="28" width="128" height="128" rx="18" stroke="currentColor" stroke-width="4"/><rect class="iamccs-qe-premium-core" x="82" y="54" width="76" height="76" rx="12" stroke="currentColor" stroke-width="3"/>`);
        case "wide":
            return svgShell(`${patternRays()}<path class="iamccs-qe-premium-core" d="M38 126L88 54M202 126L152 54M38 126H202" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "ultrawide":
            return svgShell(`${patternRays()}<path class="iamccs-qe-premium-core" d="M24 136L92 44M216 136L148 44M24 136H216M70 110H170" stroke="currentColor" stroke-width="4" stroke-linecap="round"/>`);
        case "ultraclose":
            return svgShell(`${ring(120,92,62)}<rect class="iamccs-qe-premium-core" x="70" y="42" width="100" height="100" rx="18" stroke="currentColor" stroke-width="4"/><circle class="iamccs-qe-premium-core" cx="120" cy="92" r="20" stroke="currentColor" stroke-width="4"/>`);
        case "fisheye":
            return svgShell(`${patternGrid()}<circle class="iamccs-qe-premium-core" cx="120" cy="92" r="64" stroke="currentColor" stroke-width="3.5"/><path class="iamccs-qe-premium-core" d="M56 92C72 70 92 58 120 58C148 58 168 70 184 92M56 92C72 114 92 126 120 126C148 126 168 114 184 92" stroke="currentColor" stroke-width="3.5"/>`);
        default:
            return svgShell(`${patternGrid()}<circle class="iamccs-qe-premium-core" cx="120" cy="92" r="46" stroke="currentColor" stroke-width="4"/>`);
    }
}

function badgePattern(motif) {
    switch (motif) {
        case "style":
            return `<path class="iamccs-qe-premium-pattern" d="M42 136C68 88 92 64 132 54C158 48 182 54 198 70" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle class="iamccs-qe-premium-wire" cx="174" cy="58" r="16" stroke="currentColor" stroke-width="1.6"/>`;
        case "scene":
            return `<path class="iamccs-qe-premium-pattern" d="M34 126H206" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><path class="iamccs-qe-premium-wire" d="M66 126C82 102 100 88 120 88C140 88 158 102 174 126" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/><circle class="iamccs-qe-premium-wire" cx="168" cy="54" r="18" stroke="currentColor" stroke-width="1.5"/>`;
        case "merge":
            return `<circle class="iamccs-qe-premium-wire" cx="92" cy="92" r="34" stroke="currentColor" stroke-width="1.8"/><circle class="iamccs-qe-premium-wire" cx="148" cy="92" r="34" stroke="currentColor" stroke-width="1.8"/><path class="iamccs-qe-premium-pattern" d="M64 126H176" stroke="currentColor" stroke-width="1.6" stroke-linecap="round"/>`;
        case "effect":
            return `<circle class="iamccs-qe-premium-wire" cx="120" cy="92" r="44" stroke="currentColor" stroke-width="1.6"/><path class="iamccs-qe-premium-pattern" d="M120 32V54M120 130V152M60 92H82M158 92H180M78 50L92 64M148 120L162 134M162 50L148 64M92 120L78 134" stroke="currentColor" stroke-width="1.8" stroke-linecap="round"/>`;
        case "portrait":
            return `<circle class="iamccs-qe-premium-wire" cx="120" cy="72" r="22" stroke="currentColor" stroke-width="1.7"/><path class="iamccs-qe-premium-pattern" d="M84 136C92 110 106 96 120 96C134 96 148 110 156 136" stroke="currentColor" stroke-width="1.9" stroke-linecap="round"/><path class="iamccs-qe-premium-wire" d="M60 38H180" stroke="currentColor" stroke-width="1.2"/>`;
        case "travel":
            return `<path class="iamccs-qe-premium-pattern" d="M54 132C82 120 100 102 118 84C134 68 152 56 186 48" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle class="iamccs-qe-premium-wire" cx="58" cy="130" r="8" stroke="currentColor" stroke-width="1.6"/><path class="iamccs-qe-premium-wire" d="M182 42L198 48L190 64" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>`;
        case "cinema":
            return `<rect class="iamccs-qe-premium-wire" x="52" y="46" width="136" height="92" rx="14" stroke="currentColor" stroke-width="1.6"/><path class="iamccs-qe-premium-pattern" d="M66 56V128M82 56V128M158 56V128M174 56V128" stroke="currentColor" stroke-width="1.2"/><path class="iamccs-qe-premium-pattern" d="M52 74H188M52 110H188" stroke="currentColor" stroke-width="1.2"/>`;
        case "view":
        default:
            return `<path class="iamccs-qe-premium-pattern" d="M44 62H76M44 62V94M196 62H164M196 62V94M44 122H76M44 122V90M196 122H164M196 122V90" stroke="currentColor" stroke-width="2" stroke-linecap="round"/><circle class="iamccs-qe-premium-wire" cx="120" cy="92" r="36" stroke="currentColor" stroke-width="1.6"/>`;
    }
}

function badgeFontSize(code) {
    if (code.length <= 2) return 42;
    if (code.length === 3) return 34;
    return 28;
}

function buildBadgeIcon(code, motif) {
    const size = badgeFontSize(code);
    return svgShell(`${badgePattern(motif)}<text x="120" y="108" text-anchor="middle" fill="currentColor" font-family="Bahnschrift SemiCondensed, Aptos Display, sans-serif" font-size="${size}" letter-spacing="2">${code}</text>`);
}

function getCurrentPresetName(node) {
    const presetWidget = node?.widgets?.find?.((widget) => widget?.name === "preset");
    return String(presetWidget?.value || "").trim();
}

function resolveDatasetMeta(presetName, cleanLabel) {
    const orderedMaps = [...DATASET_PRESET_META].sort((left, right) => right[0].length - left[0].length);
    for (const [prefix, metaMap] of orderedMaps) {
        if (presetName.startsWith(prefix)) return metaMap[cleanLabel] || null;
    }
    return null;
}

function resolvePremiumMeta(presetName, cleanLabel) {
    if (presetName.startsWith("Camera Angles")) return CAMERA_ANGLE_META[cleanLabel] || null;
    if (presetName.startsWith("Style Effects")) return STYLE_EFFECT_META[cleanLabel] || null;
    if (presetName.startsWith("Scene Changes")) return SCENE_CHANGE_META[cleanLabel] || null;
    if (presetName.startsWith("Multi-Image Edits")) return MULTI_IMAGE_META[cleanLabel] || null;
    if (presetName.startsWith("Additional Effects")) return ADDITIONAL_EFFECT_META[cleanLabel] || null;
    if (presetName.startsWith("Other amazing prompts")) return OTHER_PROMPT_META[cleanLabel] || null;
    if (presetName.startsWith("Travel")) return TRAVEL_META[cleanLabel] || null;
    if (presetName.startsWith("Cinematic Looks")) return CINEMATIC_LOOK_META[cleanLabel] || null;
    if (presetName.startsWith("Dataset generator_")) return resolveDatasetMeta(presetName, cleanLabel);
    return null;
}

function clearPremiumToneClasses(card) {
    card?.classList.remove(
        "iamccs-qe-premium-card",
        "iamccs-qe-premium-tone-gold",
        "iamccs-qe-premium-tone-copper",
        "iamccs-qe-premium-tone-wine",
        "iamccs-qe-premium-tone-ivory",
        "iamccs-qe-premium-tone-teal",
    );
}

function premiumLabelMarkup(cleanLabel, family) {
    return `<span class="iamccs-qe-premium-title">${cleanLabel}</span><span class="iamccs-qe-premium-sub">${family}</span>`;
}

function premiumSubline(presetName, cleanLabel, meta) {
    if (presetName.startsWith("Camera Angles")) {
        return CAMERA_ANGLE_COPY[cleanLabel] || meta.family;
    }
    return meta.family;
}

function premiumRenderKey(presetName, meta, cleanLabel, slot) {
    return [presetName, cleanLabel, meta.family, meta.tone, meta.icon || meta.code || "", meta.motif || "", slot || ""].join("|");
}

function applyPremiumToCard(card, presetName) {
    const rawLabel = card?.dataset?.label || "";
    const cleanLabel = normalizeLabel(rawLabel);
    const meta = resolvePremiumMeta(presetName, cleanLabel);
    const image = card?.querySelector(".iamccs-qe-prompt-style-image");
    const label = card?.querySelector(".iamccs-qe-prompt-style-label");

    if (!meta || !image || !label) {
        clearPremiumToneClasses(card);
        return false;
    }

    clearPremiumToneClasses(card);
    card.classList.add("iamccs-qe-premium-card", `iamccs-qe-premium-tone-${meta.tone}`);
    const renderKey = premiumRenderKey(presetName, meta, cleanLabel, card.dataset.slot);
    const artMarkup = meta.icon ? buildIcon(meta.icon) : buildBadgeIcon(meta.code, meta.motif);
    if (card.dataset.iamccsPremiumKey !== renderKey) {
        image.innerHTML = `<div class="iamccs-qe-premium-art">${artMarkup}<span class="iamccs-qe-premium-slot">${String(card.dataset.slot).padStart(2, "0")}</span></div>`;
        label.innerHTML = premiumLabelMarkup(cleanLabel, premiumSubline(presetName, cleanLabel, meta));
        card.dataset.iamccsPremiumKey = renderKey;
    }
    return true;
}

function renderPremiumGrid(node) {
    const grid = node?._qeGrid;
    if (!grid || node?._iamccsQePremiumRendering) return;
    node._iamccsQePremiumRendering = true;
    const presetName = getCurrentPresetName(node);

    try {
        const cards = Array.from(grid.querySelectorAll(".iamccs-qe-prompt-style-card"));
        let premiumCount = 0;

        for (const card of cards) {
            if (applyPremiumToCard(card, presetName)) premiumCount += 1;
        }

        if (premiumCount > 0) {
            grid.classList.add("iamccs-qe-premium-grid");
        } else {
            grid.classList.remove("iamccs-qe-premium-grid");
        }

        try {
            app.canvas?.setDirty(true, true);
        } catch {
            // ignore redraw failures
        }
    } finally {
        node._iamccsQePremiumRendering = false;
    }
}

function installPremiumCompat(node, attemptsLeft = 20) {
    if (!node || attemptsLeft <= 0) return;
    if (!node._qeGrid) {
        setTimeout(() => installPremiumCompat(node, attemptsLeft - 1), 100);
        return;
    }

    ensurePremiumStyle();
    renderPremiumGrid(node);

    if (node._iamccsQePremiumObserverInstalled) return;

    const observer = new MutationObserver((mutations) => {
        if (node._iamccsQePremiumRendering) return;
        const hasExternalMutation = mutations.some((mutation) => {
            return Array.from(mutation.addedNodes || []).some((addedNode) => {
                return !(addedNode instanceof HTMLElement) || !addedNode.classList.contains("iamccs-qe-premium-art");
            });
        });
        if (!hasExternalMutation) return;
        renderPremiumGrid(node);
    });

    observer.observe(node._qeGrid, {
        childList: true,
        subtree: true,
    });

    node._iamccsQePremiumObserverInstalled = true;
    node._iamccsQePremiumObserver = observer;

    const originalOnRemoved = node.onRemoved;
    node.onRemoved = function () {
        try {
            observer.disconnect();
        } catch {
            // ignore cleanup failures
        }
        return originalOnRemoved?.apply(this, arguments);
    };
}

app.registerExtension({
    name: "iamccs.qe_prompt_enhancer.compat",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== TARGET_NODE) return;

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalOnNodeCreated?.apply(this, arguments);
            installPremiumCompat(this);
            return result;
        };
    },
});

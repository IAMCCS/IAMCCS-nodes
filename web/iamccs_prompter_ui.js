// SPDX-License-Identifier: GPL-3.0-or-later

import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_TYPE = "IAMCCS_Prompter";

const MODE_META = {
    t2va: {
        label: "T2VA",
        subtitle: "Text → video + native audio",
        sections: [
            ["scene", "Scene", "Where, when, atmosphere, subjects and the dramatic situation."],
            ["shot_list", "Timed shot list", "Write a chronological beat list. One primary camera idea is usually stronger than many moves."],
            ["acting", "Acting & motion", "Order visible reactions and body mechanics; describe restrained, filmable behavior."],
            ["dialogue", "Dialogue", "Use <Subject N> (S1) and <d>[English] exact spoken line</d> when speech is required."],
            ["light_and_image", "Light & image", "Lighting direction, exposure, palette, texture, depth and realism."],
            ["camera", "Camera", "Framing, camera height, lens feeling and one coherent physical movement."],
            ["production_sound", "Production sound", "Chronological ambience, dialogue, contact sounds, effects and perspective."],
            ["non_diegetic_music", "Non-diegetic music", "Audience-only score. Keep it distinct from sound that exists inside the scene."],
            ["negatives", "Continuity safeguards", "Concrete failures to avoid: identity drift, duplicate limbs, cuts, text, logos or unwanted redesign."],
        ],
    },
    i2va: {
        label: "I2VA",
        subtitle: "Opening image → video + native audio",
        sections: [
            ["reference_use", "Reference use", "State that <Picture 1> is the complete opening-frame authority, not a loose inspiration."],
            ["identity_continuity_locks", "Identity / continuity locks", "List the visual facts that cannot change: face, clothes, props, geography, light and screen side."],
            ["scene", "Scene", "Describe what becomes active beyond the still opening image."],
            ["shot_list", "Timed shot list", "Chronological beats from the supplied frame to the final moment."],
            ["acting", "Acting & motion", "Natural sequencing of gaze, breath, hands, weight shift and interaction."],
            ["dialogue", "Dialogue", "Exact words and speaker labels only when needed."],
            ["light_and_image", "Light & image", "Preserve the source image unless a motivated lighting change is part of the action."],
            ["camera", "Camera", "Animate from the source perspective with one physically plausible move."],
            ["production_sound", "Production sound", "Sound events tied to visible actions, space and camera distance."],
            ["non_diegetic_music", "Non-diegetic music", "Optional audience-only score and its entrance or exit."],
            ["negatives", "Continuity safeguards", "Prevent source-frame redesign, face drift, unwanted cuts and synthetic artifacts."],
        ],
    },
    fl2va: {
        label: "FL2VA",
        subtitle: "First + last frame → video + native audio",
        sections: [
            ["boundary_frames", "Boundary frames", "Define <Picture 1> as the opening and <Picture 2> as the ending composition. The Shotboard supplies the exact final timestamp."],
            ["reference_use", "Reference use", "Explain which visual facts come from each boundary image."],
            ["identity_continuity_locks", "Identity / continuity locks", "Lock subject identity, wardrobe, props, geography, lighting logic and screen direction across the bridge."],
            ["action", "Connecting action", "Describe the physical transformation that plausibly connects the two frames."],
            ["shot_list", "Timed shot list", "Order the intermediate action; reach the final framing only near the end."],
            ["acting", "Acting & motion", "Performance beats and body mechanics through the transition."],
            ["dialogue", "Dialogue", "Exact spoken lines with subject labels; keep timing compatible with the chunk."],
            ["light_and_image", "Light & image", "Preserve or motivate changes between both boundary frames."],
            ["camera", "Camera", "One continuous camera path connecting the start and final composition."],
            ["production_sound", "Production sound", "Chronological ambience, movement, effects and speech over the whole bridge."],
            ["non_diegetic_music", "Non-diegetic music", "Optional score, separated from the diegetic soundscape."],
            ["negatives", "Continuity safeguards", "Prevent dissolves, morphs, teleports, identity drift, early arrival and unwanted edits."],
        ],
    },
    ref2va: {
        label: "REF2VA",
        subtitle: "Multi-reference image / video / audio",
        sections: [
            ["subject_definitions", "subject_definitions", "Define every persistent <Subject N> and the references that establish identity, objects or environment."],
            ["summary", "summary", "A concise intent statement: who, where, what changes and what the finished moment should feel like."],
            ["retention_analysis", "retention_analysis", "Assign what must be retained from each <Picture N>, <Video N> and <Audio N>; separate identity, motion, composition, style and sound roles."],
            ["detailed_description", "detailed_description", "Write the chronological audiovisual event with subject labels, actions, framing, continuity and exact dialogue."],
            ["overall_soundscape", "overall_soundscape", "All audible in-world layers, their timing, perspective, space, dialogue and effects."],
            ["non_diegetic_music", "non_diegetic_music", "Only audience-facing music; say none when no score is wanted."],
        ],
    },
};

const EXAMPLES = {
    t2va: {
        scene: "Night inside a nearly empty glasshouse café. <Subject 1>, a botanist in a dark green apron, notices one fern moving although every window is closed.",
        shot_list: "0.00-2.00s: medium-wide stillness around the counter. 2.00-4.50s: the fern bends toward <Subject 1> and droplets fall. 4.50s-end: <Subject 1> approaches by one step and raises a hand without touching it.",
        acting: "The reaction begins in the eyes, then the breath stops, then one cautious step. Keep hands anatomically stable and movement understated.",
        dialogue: "<Subject 1> (S1): <d>[English] You heard it too.</d>",
        light_and_image: "Moon-blue glass reflections, one warm counter practical, damp leaves with realistic specular detail, restrained contrast and shallow atmospheric depth.",
        camera: "One slow slider move from right to left at counter height, using foreground leaves for natural parallax; no cut.",
        production_sound: "Low refrigerator hum, soft rain on glass, one ceramic cup settling, leaf droplets, close quiet dialogue with greenhouse reflections.",
        non_diegetic_music: "One distant bowed-glass tone begins after the fern moves, very low beneath the location sound.",
        negatives: "No extra people, no face drift, no plant morphing, no sudden zoom, no jump cut, no text, no logo, no exaggerated horror expression.",
    },
    i2va: {
        reference_use: "Use <Picture 1> as the exact opening-frame authority for <Subject 1>, wardrobe, bicycle, street geometry, lens perspective and morning light. Continue from it without redesign.",
        identity_continuity_locks: "Keep <Subject 1>'s face, yellow rain jacket, black helmet and red bicycle unchanged. Preserve left-to-right travel and the wet market street layout.",
        scene: "The market wakes after rain as delivery shutters rise and <Subject 1> prepares to ride through the narrow lane.",
        shot_list: "0.00-1.50s: preserve the supplied composition. 1.50-4.00s: <Subject 1> pushes off and passes the first stall. 4.00s-end: the camera follows as a flock of pigeons lifts ahead.",
        acting: "One foot pushes, hips settle onto the saddle, hands remain fixed on the bars, gaze tracks the opening in the street.",
        dialogue: "",
        light_and_image: "Retain the source overcast light and wet color palette; moving reflections remain physically tied to the bicycle and market awnings.",
        camera: "Begin from the source perspective, then perform one smooth parallel tracking move with mild background parallax.",
        production_sound: "Bicycle chain, wet tire hiss, shutters lifting, vendors in the distance, pigeon wings crossing camera perspective.",
        non_diegetic_music: "No score.",
        negatives: "No wardrobe change, no bicycle deformation, no altered storefronts, no duplicate rider, no speed ramp, no cut, no captions.",
    },
    fl2va: {
        boundary_frames: "Open exactly on <Picture 1> and arrive naturally at <Picture 2> as the final composition. Treat both as complete boundary frames; do not dissolve or morph between them.",
        reference_use: "<Picture 1> fixes the wide workshop layout and opening pose. <Picture 2> fixes the close final framing, finished clay vessel and hand placement.",
        identity_continuity_locks: "Preserve <Subject 1>'s face, linen shirt, apron, wheel, clay color, window direction and left-hand screen position throughout.",
        action: "<Subject 1> leans into the wheel, narrows the vessel neck with both hands and lifts the gaze as the camera approaches the final close framing.",
        shot_list: "Opening third: hold the wide spatial relationship. Middle: travel forward while the hands shape the spinning neck. Final third: slow the move, complete the vessel and settle exactly into <Picture 2> near the end.",
        acting: "Focused breathing, small finger pressure changes, stable wrists and one calm glance upward after the shape is complete.",
        dialogue: "",
        light_and_image: "Warm side window, fine clay dust, realistic wet clay highlights; preserve exposure and color continuity between both references.",
        camera: "One continuous gentle dolly-in with a slight lowering of camera height; arrive at the final lens perspective only during the last beat.",
        production_sound: "Steady wheel motor, wet clay friction, apron movement, a distant workshop door, breathing close to the final camera position.",
        non_diegetic_music: "No score; let the wheel rhythm carry the scene.",
        negatives: "No dissolve, no object morph, no hand duplication, no face drift, no sudden lens change, no early arrival at the final frame, no text.",
    },
    ref2va: {
        subject_definitions: "<Subject 1>: the street drummer established by <Picture 1>; retain face, shaved hair, indigo jacket and silver wrist tape.\n<Subject 2>: the compact red drum kit established by <Picture 2>; retain shell color, hardware layout and scale.",
        summary: "At dusk beneath an overpass, <Subject 1> builds a restrained rhythm on <Subject 2> while the camera makes one low circular move and nearby pedestrians gradually notice.",
        retention_analysis: "Use <Picture 1> for <Subject 1> identity and wardrobe. Use <Picture 2> for <Subject 2> construction and color. If <Video 1> is connected, retain only its hand rhythm and camera cadence, not its performer identity or background. If <Audio 1> is connected, retain drum timbre and tempo, not unrelated ambience.",
        detailed_description: "Begin low and close to <Subject 2> as <Subject 1> taps a sparse pattern with brushes. Continue in one clockwise move, revealing the performer and concrete columns. The rhythm grows through one controlled fill, then stops on a precise final hit as <Subject 1> looks toward an off-camera listener. Preserve subject labels and the spatial side of every drum throughout.",
        overall_soundscape: "Dry brush hits expand into a compact stereo kit, with traffic wash above, shoe movement on concrete and short overpass reflections. Keep the drum transients synchronized to visible contacts and reduce traffic as the camera closes in.",
        non_diegetic_music: "None. Every musical element is performed visibly by <Subject 1> on <Subject 2>.",
    },
};

const T2V_PROJECTS = [
    {
        id: "glasshouse_signal",
        name: "Glasshouse Signal",
        sections: EXAMPLES.t2va,
    },
    {
        id: "railway_blue_hour",
        name: "Railway Blue Hour",
        sections: {
            scene: "Before sunrise on a rain-polished railway platform, <Subject 1>, a tired courier in a charcoal coat, waits beside a silver case as an empty train approaches through blue mist.",
            shot_list: "0.00-1.50s: hold a medium-wide profile and establish the empty platform. 1.50-3.70s: the train enters and moving reflections cross <Subject 1>. 3.70s-end: <Subject 1> turns toward camera and grips the case.",
            acting: "Tension begins in the shoulders, then the eyes react, then one deliberate turn. Preserve natural blinking, breathing and restrained hand movement.",
            dialogue: "<Subject 1> (S1): <d>[English] Not this train.</d>",
            light_and_image: "Cool predawn ambience, practical sodium lamps, wet reflections, realistic skin texture, restrained contrast and cinematic depth without artificial glow.",
            camera: "One slow lateral tracking move at chest height with mild foreground parallax; maintain one lens language and do not cut.",
            production_sound: "Distant rail vibration, light rain on the metal roof, one approaching brake squeal, coat movement and close dialogue with platform reflections.",
            non_diegetic_music: "A sparse low cello pulse enters only after the train becomes visible, beneath the physical scene sound.",
            negatives: "No identity drift, duplicate people, wardrobe change, warped hands, sudden zoom, jump cut, subtitles, text or logo.",
        },
    },
    {
        id: "desert_convoy",
        name: "Desert Convoy",
        sections: {
            scene: "Late afternoon on a vast salt desert. <Subject 1>, a mechanic in a faded red scarf, stands beside a stalled solar rover while a distant dust column approaches.",
            shot_list: "0.00-1.20s: wide stillness around the rover. 1.20-3.40s: <Subject 1> notices the dust and closes the engine panel. 3.40s-end: the rover powers on as the approaching convoy resolves in the heat haze.",
            acting: "A small glance triggers a practical sequence: close the panel, secure the latch, rise and shield the eyes. Keep weight and hand contacts physically grounded.",
            dialogue: "<Subject 1> (S1): <d>[English] Right on time.</d>",
            light_and_image: "Hard amber side light, pale salt reflections, fine airborne dust, sun-worn fabric and realistic metallic heat shimmer.",
            camera: "Begin low beside the rover wheel and perform one measured crane rise into a wider reveal; no cut.",
            production_sound: "Dry wind, cooling metal ticks, latch contact, electric motor startup and a distant layered engine rumble.",
            non_diegetic_music: "One restrained analog bass note appears with the convoy silhouette, then holds.",
            negatives: "No extra vehicles appearing suddenly, no rover deformation, no floating dust indoors, no face drift, no speed ramp, no text or logo.",
        },
    },
    {
        id: "noir_diner",
        name: "Noir Diner Dialogue",
        sections: {
            scene: "Midnight inside a nearly empty roadside diner. <Subject 1>, a private investigator with a damp wool coat, sits opposite <Subject 2>, an exhausted night waitress, beneath a flickering sign.",
            shot_list: "0.00-1.40s: hold both subjects across the booth. 1.40-3.30s: <Subject 2> slides a sealed envelope across the table. 3.30s-end: <Subject 1> stops it with two fingers and looks up.",
            acting: "Use quiet micro-performance: guarded eye contact, one controlled breath and precise hand contact with the envelope. Both subjects remain seated.",
            dialogue: "<Subject 2> (S2): <d>[English] You were never here.</d> <Subject 1> answers only with a small nod.",
            light_and_image: "Green-blue window spill, warm tungsten practicals, rain traces on glass, natural skin exposure and subtle film grain.",
            camera: "One slow push across table height, maintaining the two-shot until the final emphasis on <Subject 1>; no reverse angle and no cut.",
            production_sound: "Rain against glass, refrigerator hum, distant tire wash, ceramic cup resonance, paper sliding and intimate booth dialogue.",
            non_diegetic_music: "No score until the envelope stops; then one almost inaudible brushed-cymbal swell.",
            negatives: "No lip-sync drift, no duplicated hands, no changing envelope, no added customers, no neon color shift, no jump cut, no captions.",
        },
    },
    {
        id: "orbital_rescue",
        name: "Orbital Rescue",
        sections: {
            scene: "In low orbit above a blue planet, <Subject 1>, an astronaut in a practical white EVA suit, is tethered outside a damaged research station while a loose equipment case drifts away.",
            shot_list: "0.00-1.30s: establish the astronaut, station hull and drifting case. 1.30-3.40s: <Subject 1> fires one short maneuvering burst and reaches along the tether. 3.40s-end: the glove catches the case handle and rotation settles.",
            acting: "Movement is slow and inertial. The torso reacts after each thruster burst, the tether stays under tension and the catch transfers momentum through the arm.",
            dialogue: "<Subject 1> (S1): <d>[English] Payload secured.</d>",
            light_and_image: "Unfiltered orbital sunlight, deep black space, controlled visor reflections, detailed suit fabric and physically plausible Earth bounce light.",
            camera: "One stabilized exterior tracking move parallel to the hull with subtle orbital drift; preserve orientation and never cut.",
            production_sound: "Inside-suit breathing, radio compression, short thruster impulses, tether vibration and muted glove contact; exterior space remains silent.",
            non_diegetic_music: "A restrained high string harmonic enters during the reach and resolves on the catch.",
            negatives: "No gravity-like falling, no flapping cloth, no changing station geometry, no extra limbs, no visor face drift, no lens flare overload, no text.",
        },
    },
];

function widget(node, name) {
    return node.widgets?.find((item) => item.name === name);
}

function setWidget(node, name, value) {
    const target = widget(node, name);
    if (!target) return;
    target.value = value;
    try { target.callback?.(value); } catch {}
}

function hideWidget(target) {
    if (!target) return;
    if (target._iamccsPrompterHidden) {
        target.type = "hidden";
        target.hidden = true;
        target.computeSize = () => [0, 0];
        target.draw = () => {};
        return;
    }
    target.origType = target.origType || target.type;
    target._iamccsPrompterOrigCompute = target.computeSize;
    target._iamccsPrompterOrigDraw = target.draw;
    target.type = "hidden";
    target.hidden = true;
    target.computeSize = () => [0, 0];
    target.draw = () => {};
    target.serializeValue = target.serializeValue || (() => target.value);
    target._iamccsPrompterHidden = true;
}

function safeProject(raw) {
    let parsed = {};
    try { parsed = JSON.parse(String(raw || "{}")); } catch {}
    return {
        schema: "iamccs.minimax_h3.prompter_project",
        schema_version: 2,
        project_name: String(parsed.project_name || "Untitled H3 Prompt"),
        task_mode: MODE_META[parsed.task_mode] ? parsed.task_mode : "t2va",
        injection_target: ["global", "local_auto", "local_1", "local_2", "local_3"].includes(parsed.injection_target) ? parsed.injection_target : "global",
        writing_mode: ["manual", "guided", "assistant_fill"].includes(parsed.writing_mode) ? parsed.writing_mode : "guided",
        merge_policy: ["replace", "append"].includes(parsed.merge_policy) ? parsed.merge_policy : "replace",
        ai_direction: String(parsed.ai_direction || ""),
        ai_scope: String(parsed.ai_scope || "active_field"),
        ai_visual_roles: parsed.ai_visual_roles && typeof parsed.ai_visual_roles === "object" ? { ...parsed.ai_visual_roles } : {},
        sections: parsed.sections && typeof parsed.sections === "object" ? { ...parsed.sections } : {},
    };
}

function composePrompt(project) {
    const mode = MODE_META[project.task_mode] ? project.task_mode : "t2va";
    return MODE_META[mode].sections
        .map(([key, label]) => {
            const body = String(project.sections?.[key] || "").trim();
            if (!body) return "";
            return mode === "ref2va" ? `${label}:\n${body}` : `[${label.toUpperCase()}]\n${body}`;
        })
        .filter(Boolean)
        .join("\n\n");
}

function el(tag, className = "", text = "") {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text) node.textContent = text;
    return node;
}

function button(label, className = "") {
    const result = el("button", `iamccs-pr-btn ${className}`, label);
    result.type = "button";
    return result;
}

function downloadProject(project) {
    const clean = String(project.project_name || "iamccs_h3_prompt").replace(/[^a-z0-9._-]+/gi, "_").replace(/^_+|_+$/g, "") || "iamccs_h3_prompt";
    const blob = new Blob([JSON.stringify(project, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${clean}.iamccs-h3-prompt.json`;
    document.body.appendChild(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function nodeType(node) {
    return String(node?.comfyClass || node?.type || node?.constructor?.comfyClass || "");
}

function shotboardsForPrompter(node) {
    const graph = node?.graph || app?.graph;
    const connected = [];
    for (const output of node?.outputs || []) {
        for (const linkId of Array.isArray(output?.links) ? output.links : []) {
            const link = graph?.links?.[linkId];
            const targetId = link?.target_id ?? link?.[3];
            const target = targetId != null ? graph?.getNodeById?.(targetId) : null;
            if (nodeType(target) === "IAMCCS_MiniMaxH3ShotPlanner") connected.push(target);
        }
    }
    if (connected.length) return [...new Set(connected)];
    const all = Array.isArray(graph?._nodes) ? graph._nodes.filter((item) => nodeType(item) === "IAMCCS_MiniMaxH3ShotPlanner") : [];
    if (all.length <= 1) return all;
    return all.slice().sort((a, b) => {
        const distance = (item) => Math.hypot(Number(item?.pos?.[0] || 0) - Number(node?.pos?.[0] || 0), Number(item?.pos?.[1] || 0) - Number(node?.pos?.[1] || 0));
        return distance(a) - distance(b);
    }).slice(0, 1);
}

function mountPrompter(node) {
    if (node._iamccsPrompterMounted) return;
    node._iamccsPrompterMounted = true;

    const rawNames = ["project_data", "task_mode", "injection_target", "writing_mode", "merge_policy", "character_budget"];
    rawNames.forEach((name) => hideWidget(widget(node, name)));

    let project = safeProject(widget(node, "project_data")?.value);
    project.task_mode = String(widget(node, "task_mode")?.value || project.task_mode);
    project.injection_target = String(widget(node, "injection_target")?.value || project.injection_target);
    project.writing_mode = String(widget(node, "writing_mode")?.value || project.writing_mode);
    project.merge_policy = String(widget(node, "merge_policy")?.value || project.merge_policy);

    const root = el("div", "iamccs-pr-root");
    root.innerHTML = `
        <style>
            .iamccs-pr-root{--ink:#17191d;--paper:#f3efe5;--paper2:#e7e0d0;--gold:#d9ad58;--blue:#79a8d8;--muted:#9aa3ad;width:960px;height:720px;background:linear-gradient(140deg,#151820,#0c0e13 72%);color:#e9edf2;border:1px solid #363c48;border-radius:12px;overflow:hidden;font:12px Inter,Segoe UI,sans-serif;box-shadow:0 18px 50px #0008;display:flex;flex-direction:column}
            .iamccs-pr-root *{box-sizing:border-box}.iamccs-pr-top{height:58px;display:flex;align-items:center;gap:12px;padding:9px 14px;border-bottom:1px solid #303641;background:#10131a}.iamccs-pr-mark{width:34px;height:34px;border-radius:9px;display:grid;place-items:center;background:linear-gradient(135deg,#e0b660,#9b6a25);color:#17130a;font:800 15px Georgia}.iamccs-pr-brand{min-width:180px}.iamccs-pr-title{font:700 15px Georgia,serif;letter-spacing:.4px}.iamccs-pr-sub{font-size:10px;color:#9fa8b5;margin-top:2px}.iamccs-pr-name{height:34px;flex:1;min-width:160px;border:1px solid #38404d!important;border-radius:7px!important;background:#171b23!important;color:#f4f6f8!important;padding:0 10px!important}.iamccs-pr-actions{display:flex;gap:6px}.iamccs-pr-btn{height:30px;border:1px solid #3b4350;border-radius:6px;background:#202630;color:#e6ebf0;padding:0 10px;cursor:pointer;font:600 11px Inter,Segoe UI,sans-serif}.iamccs-pr-btn:hover{border-color:#d9ad58;color:#fff}.iamccs-pr-btn.primary{background:#b78537;border-color:#e1ba70;color:#15110a}.iamccs-pr-btn.danger{color:#e9a29c}.iamccs-pr-modes{height:46px;padding:7px 14px;display:flex;align-items:center;gap:7px;border-bottom:1px solid #303641;background:#141820}.iamccs-pr-mode{height:30px;min-width:72px}.iamccs-pr-mode.active{background:#30455d;border-color:#79a8d8;color:#fff}.iamccs-pr-mode-note{margin-left:auto;color:#9ea7b2;font-size:10px}.iamccs-pr-layout{display:grid;grid-template-columns:178px minmax(0,1fr) 292px;min-height:0;flex:1}.iamccs-pr-left{border-right:1px solid #303641;padding:11px;background:#11151c;overflow:auto}.iamccs-pr-kicker{font-size:9px;text-transform:uppercase;letter-spacing:1.4px;color:#d9ad58;margin:2px 0 7px}.iamccs-pr-targets,.iamccs-pr-writing{display:grid;gap:5px;margin-bottom:13px}.iamccs-pr-target,.iamccs-pr-write{height:29px;text-align:left}.iamccs-pr-target.active,.iamccs-pr-write.active{border-color:#d9ad58;background:#382f20;color:#ffe6ad}.iamccs-pr-hint{font-size:10px;line-height:1.45;color:#929ba7;padding:8px;border-radius:7px;background:#181d25;border:1px solid #2c333e;margin-bottom:12px}.iamccs-pr-policy{width:100%;height:30px;background:#1b2029;color:#e9edf2;border:1px solid #343c48;border-radius:6px;padding:0 7px}.iamccs-pr-center{overflow:auto;padding:12px 14px;background:radial-gradient(circle at 50% -10%,#262d3a 0,#181c24 45%,#141820 100%)}.iamccs-pr-section{background:#f4f0e6;color:#17191d;border-radius:6px;margin-bottom:10px;box-shadow:0 4px 12px #0005;overflow:hidden;border:1px solid #cfc5b1}.iamccs-pr-section-head{height:35px;display:flex;align-items:center;gap:8px;padding:0 10px;background:#e7e0d2;border-bottom:1px solid #cdc3b1}.iamccs-pr-num{width:20px;height:20px;border-radius:50%;display:grid;place-items:center;background:#1e2938;color:#f4d596;font:700 10px Georgia}.iamccs-pr-section-title{font:700 12px Georgia,serif;letter-spacing:.3px}.iamccs-pr-state{margin-left:auto;color:#75808c;font-size:9px;text-transform:uppercase}.iamccs-pr-text{display:block;width:100%;min-height:82px;resize:vertical;border:0!important;outline:0!important;background:#f8f5ed!important;color:#181a1d!important;padding:10px 12px!important;font:12px/1.5 'Courier New',monospace!important}.iamccs-pr-tip{padding:7px 11px;background:#eee8dc;color:#66645e;font-size:10px;line-height:1.35;border-top:1px dashed #d3c8b5}.iamccs-pr-right{border-left:1px solid #303641;background:#10141a;padding:11px;display:flex;min-height:0;flex-direction:column}.iamccs-pr-status{display:flex;gap:6px;margin-bottom:8px}.iamccs-pr-pill{border-radius:10px;padding:3px 7px;background:#222a35;color:#aeb7c2;font-size:9px}.iamccs-pr-pill.ok{background:#1d3a2b;color:#99ddb2}.iamccs-pr-pill.warn{background:#493322;color:#f3c184}.iamccs-pr-preview{flex:1;min-height:0;overflow:auto;border:1px solid #3a414c;border-radius:6px;background:#f4f0e7;color:#1b1b1b;padding:12px;white-space:pre-wrap;font:11px/1.5 'Courier New',monospace}.iamccs-pr-preview:empty:before{content:'The composed H3 prompt will appear here.';color:#8c8981}.iamccs-pr-footer{margin-top:8px;display:flex;gap:6px}.iamccs-pr-footer .iamccs-pr-btn{flex:1}.iamccs-pr-assist{display:none;margin-bottom:8px;padding:8px;border:1px solid #44637d;background:#172635;color:#bed8ec;border-radius:6px;font-size:10px;line-height:1.4}.iamccs-pr-assist.show{display:block}.iamccs-pr-load{display:none}.iamccs-pr-empty .iamccs-pr-section-head{background:#f0e0d7}.iamccs-pr-empty .iamccs-pr-state{color:#b26751}.iamccs-pr-root.mode-manual .iamccs-pr-tip{display:none}
        </style>`;
    const aiStyle = document.createElement("style");
    aiStyle.textContent = `
        .iamccs-pr-ai{display:none;margin-top:10px;padding:9px;border:1px solid #425d78;border-radius:8px;background:linear-gradient(145deg,#172330,#111922);gap:7px}
        .iamccs-pr-ai.show{display:grid}.iamccs-pr-ai-title{color:#9fc9ef;font-size:10px;font-weight:800;letter-spacing:.08em;text-transform:uppercase}
        .iamccs-pr-ai-row{display:grid;grid-template-columns:1fr 1fr;gap:6px}.iamccs-pr-ai label{display:grid;gap:3px;color:#8999aa;font-size:9px;font-weight:700}
        .iamccs-pr-ai input,.iamccs-pr-ai select{width:100%;height:29px;border:1px solid #35485b;border-radius:5px;background:#0e151d;color:#e7eef5;padding:0 6px;font-size:10px}
        .iamccs-pr-ai textarea{width:100%;min-height:66px;resize:vertical;border:1px solid #35485b;border-radius:5px;background:#0e151d;color:#e7eef5;padding:7px;font:10px/1.4 Inter,Segoe UI,sans-serif}
        .iamccs-pr-ai-status{min-height:28px;color:#91a4b5;font-size:9px;line-height:1.35}.iamccs-pr-ai-status.ok{color:#8fd1aa}.iamccs-pr-ai-status.error{color:#ed9c92}
        .iamccs-pr-ai .iamccs-pr-btn{width:100%;border-color:#6094c0;background:#274866;color:#eef7ff}
        .iamccs-pr-ai-modelrow{display:grid;grid-template-columns:minmax(0,1fr) 30px;gap:5px}.iamccs-pr-ai-modelrow .iamccs-pr-btn{height:29px;padding:0!important}
        .iamccs-pr-ai-images{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:5px}.iamccs-pr-ai-image{display:grid;grid-template-columns:44px minmax(0,1fr);gap:5px;padding:4px;border:1px solid #304255;border-radius:6px;background:#0c141c;min-width:0}.iamccs-pr-ai-thumb{width:44px;height:44px;object-fit:cover;border-radius:4px;background:#202832}.iamccs-pr-ai-image-meta{display:grid;gap:3px;min-width:0}.iamccs-pr-ai-image-name{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#aebdca;font-size:8px}.iamccs-pr-ai-image select{height:24px!important;font-size:8px!important}.iamccs-pr-ai-file{display:none}
        .iamccs-pr-example-select{height:30px;max-width:146px;border:1px solid #3b4350;border-radius:6px;background:#171b23;color:#e9edf2;padding:0 6px;font:600 10px Inter,Segoe UI,sans-serif}
        .iamccs-pr-inject{width:100%;height:38px!important;margin:0 0 7px;background:linear-gradient(135deg,#d3a447,#8d5c20)!important;border:1px solid #f0ca7d!important;color:#171109!important;font-size:12px!important;font-weight:900!important;letter-spacing:.06em;box-shadow:0 5px 14px #0007}
        .iamccs-pr-inject-status{min-height:30px;margin-bottom:12px;padding:7px;border:1px solid #303944;border-radius:6px;background:#151b22;color:#91a0ae;font-size:9px;line-height:1.35}.iamccs-pr-inject-status.ok{border-color:#3f7957;color:#9fe0b7}.iamccs-pr-inject-status.error{border-color:#824b45;color:#efaaa1}
        .iamccs-pr-tagdeck{position:sticky;top:-12px;z-index:4;margin:-2px 0 12px;padding:9px 10px;border:1px solid #45505f;border-radius:8px;background:linear-gradient(145deg,#111720f5,#1c2430f5);box-shadow:0 5px 16px #0008;backdrop-filter:blur(6px)}
        .iamccs-pr-taghead{display:flex;align-items:center;gap:8px;margin-bottom:7px}.iamccs-pr-tagtitle{color:#f1d492;font:800 10px Georgia,serif;letter-spacing:.09em}.iamccs-pr-taghint{margin-left:auto;color:#93a1b1;font-size:9px}.iamccs-pr-tagrows{display:grid;gap:5px}.iamccs-pr-tagrow{display:flex;align-items:center;gap:4px;flex-wrap:wrap}.iamccs-pr-taglabel{width:51px;color:#718297;font-size:8px;font-weight:900;letter-spacing:.08em;text-transform:uppercase}.iamccs-pr-tag{height:24px!important;padding:0 7px!important;border-color:#3d4a5b!important;background:#1c2632!important;color:#dce7f2!important;font:700 9px 'Courier New',monospace!important}.iamccs-pr-tag:hover{border-color:#d9ad58!important;color:#ffe5ab!important}.iamccs-pr-tag.syntax{background:#342a1c!important;border-color:#685536!important;color:#f5d38e!important}
    `;
    root.appendChild(aiStyle);

    const top = el("div", "iamccs-pr-top");
    top.innerHTML = `<div class="iamccs-pr-mark">H3</div><div class="iamccs-pr-brand"><div class="iamccs-pr-title">IAMCCS Prompter</div><div class="iamccs-pr-sub">Structured screenplay desk · MiniMax H3</div></div>`;
    const nameInput = el("input", "iamccs-pr-name");
    nameInput.type = "text";
    nameInput.placeholder = "Project title";
    top.appendChild(nameInput);
    const actions = el("div", "iamccs-pr-actions");
    const exampleSelect = el("select", "iamccs-pr-example-select");
    T2V_PROJECTS.forEach((preset) => {
        const option = document.createElement("option");
        option.value = preset.id;
        option.textContent = preset.name;
        exampleSelect.appendChild(option);
    });
    const exampleBtn = button("Load Example");
    const loadBtn = button("Load Project");
    const saveBtn = button("Save Project", "primary");
    const fileInput = el("input", "iamccs-pr-load");
    fileInput.type = "file";
    fileInput.accept = ".json,.iamccs-h3-prompt.json,application/json";
    actions.append(exampleSelect, exampleBtn, loadBtn, saveBtn, fileInput);
    top.appendChild(actions);

    const modeBar = el("div", "iamccs-pr-modes");
    const modeButtons = new Map();
    Object.entries(MODE_META).forEach(([key, meta]) => {
        const item = button(meta.label, "iamccs-pr-mode");
        item.dataset.mode = key;
        modeButtons.set(key, item);
        modeBar.appendChild(item);
    });
    const modeNote = el("div", "iamccs-pr-mode-note");
    modeBar.appendChild(modeNote);

    const layout = el("div", "iamccs-pr-layout");
    const left = el("aside", "iamccs-pr-left");
    left.appendChild(el("div", "iamccs-pr-kicker", "Inject into Shotboard"));
    const targets = el("div", "iamccs-pr-targets");
    const targetLabels = {
        global: "Global Prompt",
        local_auto: "Local · Auto detect",
        local_1: "Local Prompt 1",
        local_2: "Local Prompt 2",
        local_3: "Local Prompt 3",
    };
    const targetButtons = new Map();
    Object.entries(targetLabels).forEach(([key, label]) => {
        const item = button(label, "iamccs-pr-target");
        item.dataset.target = key;
        targetButtons.set(key, item);
        targets.appendChild(item);
    });
    left.appendChild(targets);
    const targetHint = el("div", "iamccs-pr-hint");
    left.appendChild(targetHint);
    const injectBtn = button("INJECT → SHOTBOARD", "iamccs-pr-inject");
    injectBtn.title = "Write the composed prompt immediately into the connected MiniMax Shotboard and keep the CineLinX queue-time injection contract synchronized.";
    const injectStatus = el("div", "iamccs-pr-inject-status", "Connect CineLinX to a MiniMax Shotboard, choose a target, then inject.");
    left.append(injectBtn, injectStatus);
    left.appendChild(el("div", "iamccs-pr-kicker", "Writing mode"));
    const writing = el("div", "iamccs-pr-writing");
    const writingLabels = { manual: "Manual", guided: "Guided checklist", assistant_fill: "AI rewrite fields" };
    const writingButtons = new Map();
    Object.entries(writingLabels).forEach(([key, label]) => {
        const item = button(label, "iamccs-pr-write");
        item.dataset.writing = key;
        writingButtons.set(key, item);
        writing.appendChild(item);
    });
    left.appendChild(writing);
    left.appendChild(el("div", "iamccs-pr-kicker", "Existing text"));
    const policy = el("select", "iamccs-pr-policy");
    policy.innerHTML = `<option value="replace">Replace target</option><option value="append">Append to target</option>`;
    left.appendChild(policy);
    const assistantHint = el("div", "iamccs-pr-hint", "AI Rewrite treats every filled box as your rough idea, then rewrites those same boxes into MiniMax H3-ready English in one request. Blank boxes stay blank and your project remains editable before queueing.");
    left.appendChild(assistantHint);
    const aiPanel = el("div", "iamccs-pr-ai");
    aiPanel.appendChild(el("div", "iamccs-pr-ai-title", "Autonomous MiniMax assistant"));
    const aiScope = el("select");
    const aiDirection = el("textarea");
    aiDirection.placeholder = "Your direction for the AI: what to preserve, emphasize, simplify or change. The rough idea remains in the selected prompt field.";
    const aiScopeLabel = el("label", "", "Improve target"); aiScopeLabel.appendChild(aiScope);
    const aiDirectionLabel = el("label", "", "User direction (applies only when AI Rewrite is active)"); aiDirectionLabel.appendChild(aiDirection);
    aiPanel.append(aiScopeLabel, aiDirectionLabel);
    const aiProvider = el("select");
    aiProvider.innerHTML = `<option value="ollama">Ollama / local</option><option value="openai_compatible">OpenAI-compatible</option><option value="gemini">Google Gemini</option><option value="anthropic">Anthropic</option>`;
    const aiBaseUrl = el("input");
    aiBaseUrl.placeholder = "Provider base URL";
    const aiModel = el("input");
    aiModel.placeholder = "Model name";
    const aiModelList = el("datalist");
    aiModelList.id = `iamccs-prompter-models-${node.id || Math.random().toString(16).slice(2)}`;
    aiModel.setAttribute("list", aiModelList.id);
    const refreshModelsBtn = button("↻");
    refreshModelsBtn.title = "Read the models installed in Ollama";
    const aiApiKey = el("input");
    aiApiKey.type = "password";
    aiApiKey.autocomplete = "off";
    aiApiKey.placeholder = "API key or environment variable";
    const aiTemperature = el("input");
    aiTemperature.type = "number";
    aiTemperature.min = "0";
    aiTemperature.max = "1.5";
    aiTemperature.step = "0.05";
    aiTemperature.value = "0.35";
    const aiRow1 = el("div", "iamccs-pr-ai-row");
    const providerLabel = el("label", "", "Provider"); providerLabel.appendChild(aiProvider);
    const modelLabel = el("label", "", "Model");
    const modelRow = el("div", "iamccs-pr-ai-modelrow"); modelRow.append(aiModel, refreshModelsBtn, aiModelList); modelLabel.appendChild(modelRow);
    aiRow1.append(providerLabel, modelLabel);
    const aiRow2 = el("div", "iamccs-pr-ai-row");
    const urlLabel = el("label", "", "Base URL"); urlLabel.appendChild(aiBaseUrl);
    const tempLabel = el("label", "", "Creativity"); tempLabel.appendChild(aiTemperature);
    aiRow2.append(urlLabel, tempLabel);
    const keyLabel = el("label", "", "API key (never saved)"); keyLabel.appendChild(aiApiKey);
    const aiImageInput = el("input", "iamccs-pr-ai-file");
    aiImageInput.type = "file";
    aiImageInput.accept = "image/png,image/jpeg,image/webp";
    aiImageInput.multiple = true;
    const addAIImagesBtn = button("Add up to 4 AI image references");
    const aiImages = el("div", "iamccs-pr-ai-images");
    const rewriteBtn = button("Improve selected prompt with AI");
    const aiStatus = el("div", "iamccs-pr-ai-status", "Ollama is local. Choose the field to improve; cloud keys are never stored in the workflow.");
    aiPanel.append(aiRow1, aiRow2, keyLabel, addAIImagesBtn, aiImageInput, aiImages, rewriteBtn, aiStatus);
    left.appendChild(aiPanel);

    const center = el("main", "iamccs-pr-center");
    let activePromptArea = null;
    let activePromptKey = null;
    const tagDeck = el("section", "iamccs-pr-tagdeck");
    const tagHead = el("div", "iamccs-pr-taghead");
    tagHead.append(el("div", "iamccs-pr-tagtitle", "MINIMAX H3 PROMPT TAGS"));
    const tagHint = el("div", "iamccs-pr-taghint", "Click a field, then insert a tag");
    tagHead.appendChild(tagHint);
    const tagRows = el("div", "iamccs-pr-tagrows");
    tagDeck.append(tagHead, tagRows);

    const insertIntoActiveField = (text, selectionText = "") => {
        const area = activePromptArea || center.querySelector("textarea.iamccs-pr-text");
        if (!area) {
            tagHint.textContent = "No prompt field is available in this mode";
            return;
        }
        const start = Number.isFinite(area.selectionStart) ? area.selectionStart : area.value.length;
        const end = Number.isFinite(area.selectionEnd) ? area.selectionEnd : start;
        const before = area.value.slice(0, start);
        const after = area.value.slice(end);
        const prefix = before && !/[\s\n]$/.test(before) ? " " : "";
        const suffix = after && !/^[\s\n.,;:!?]/.test(after) ? " " : "";
        const insertion = `${prefix}${text}${suffix}`;
        area.setRangeText(insertion, start, end, "end");
        if (selectionText) {
            const selectionOffset = insertion.indexOf(selectionText);
            if (selectionOffset >= 0) {
                area.setSelectionRange(start + selectionOffset, start + selectionOffset + selectionText.length);
            }
        }
        area.focus();
        area.dispatchEvent(new Event("input", { bubbles: true }));
        tagHint.textContent = `${text} inserted in the active field`;
    };
    const addTagRow = (label, definitions) => {
        const row = el("div", "iamccs-pr-tagrow");
        row.appendChild(el("div", "iamccs-pr-taglabel", label));
        definitions.forEach(({ caption, value, select = "", syntax = false, title = "" }) => {
            const item = button(caption, `iamccs-pr-tag${syntax ? " syntax" : ""}`);
            item.type = "button";
            item.title = title || `Insert ${value}`;
            item.addEventListener("pointerdown", (event) => event.preventDefault());
            item.onclick = () => insertIntoActiveField(value, select);
            row.appendChild(item);
        });
        tagRows.appendChild(row);
    };
    addTagRow("Subject", [1, 2, 3, 4].map((index) => ({ caption: `<Subject ${index}>`, value: `<Subject ${index}>` })));
    addTagRow("Picture", [1, 2, 3, 4].map((index) => ({ caption: `<Picture ${index}>`, value: `<Picture ${index}>` })));
    addTagRow("Media", [
        { caption: "<Video 1>", value: "<Video 1>" },
        { caption: "<Video 2>", value: "<Video 2>" },
        { caption: "<Audio 1>", value: "<Audio 1>" },
        { caption: "<Audio 2>", value: "<Audio 2>" },
    ]);
    addTagRow("Speech", [
        { caption: "(S1)", value: "(S1)", syntax: true, title: "Stable speaker identity 1" },
        { caption: "(S2)", value: "(S2)", syntax: true, title: "Stable speaker identity 2" },
        { caption: "<d> dialogue", value: "<d>[English] dialogue</d>", select: "dialogue", syntax: true, title: "MiniMax dialogue or lyrics block" },
        { caption: "<scenetrans>", value: "<scenetrans>", syntax: true, title: "Dialogue continues across a scene transition" },
        { caption: "<cutoff>", value: "<cutoff>", syntax: true, title: "Speech is intentionally cut off by the video ending" },
    ]);
    const right = el("aside", "iamccs-pr-right");
    right.appendChild(el("div", "iamccs-pr-kicker", "Final prompt"));
    const assistantBanner = el("div", "iamccs-pr-assist", "AI Rewrite is active. Write a rough idea in any field, choose an engine, then rewrite. Review and edit the result before queueing MiniMax H3.");
    right.appendChild(assistantBanner);
    const status = el("div", "iamccs-pr-status");
    const charPill = el("div", "iamccs-pr-pill");
    const completePill = el("div", "iamccs-pr-pill");
    status.append(charPill, completePill);
    right.appendChild(status);
    const preview = el("div", "iamccs-pr-preview");
    right.appendChild(preview);
    const footer = el("div", "iamccs-pr-footer");
    const copyBtn = button("Copy Prompt");
    const clearBtn = button("Clear Mode", "danger");
    footer.append(copyBtn, clearBtn);
    right.appendChild(footer);

    layout.append(left, center, right);
    root.append(top, modeBar, layout);

    const commit = () => {
        project.project_name = nameInput.value.trim() || "Untitled H3 Prompt";
        project.merge_policy = policy.value;
        project.ai_direction = aiDirection.value;
        project.ai_scope = aiScope.value || "active_field";
        setWidget(node, "project_data", JSON.stringify(project));
        setWidget(node, "task_mode", project.task_mode);
        setWidget(node, "injection_target", project.injection_target);
        setWidget(node, "writing_mode", project.writing_mode);
        setWidget(node, "merge_policy", project.merge_policy);
        node.setDirtyCanvas?.(true, true);
    };

    const aiDefaults = {
        ollama: { baseUrl: "http://127.0.0.1:11434", model: "" },
        openai_compatible: { baseUrl: "https://api.openai.com/v1", model: "gpt-4.1-mini" },
        gemini: { baseUrl: "https://generativelanguage.googleapis.com/v1beta", model: "gemini-2.5-flash" },
        anthropic: { baseUrl: "https://api.anthropic.com/v1", model: "claude-sonnet-4-5" },
    };
    node.properties = node.properties || {};
    const savedAI = node.properties.iamccs_prompter_ai || {};
    aiProvider.value = String(savedAI.provider || "ollama");
    aiBaseUrl.value = String(savedAI.base_url || aiDefaults[aiProvider.value]?.baseUrl || "");
    aiModel.value = String(savedAI.model || aiDefaults[aiProvider.value]?.model || "");
    aiTemperature.value = String(savedAI.temperature ?? 0.35);
    const persistAI = () => {
        node.properties.iamccs_prompter_ai = {
            provider: aiProvider.value,
            base_url: aiBaseUrl.value.trim(),
            model: aiModel.value.trim(),
            temperature: Number(aiTemperature.value || 0.35),
        };
    };
    const aiVisualFiles = [];
    const visualRolesForTarget = () => {
        project.ai_visual_roles = project.ai_visual_roles && typeof project.ai_visual_roles === "object" ? project.ai_visual_roles : {};
        const key = project.injection_target || "global";
        project.ai_visual_roles[key] = project.ai_visual_roles[key] && typeof project.ai_visual_roles[key] === "object" ? project.ai_visual_roles[key] : {};
        return project.ai_visual_roles[key];
    };
    const readFileDataUrl = (file) => new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(String(reader.result || ""));
        reader.onerror = () => reject(reader.error || new Error("Unable to read image"));
        reader.readAsDataURL(file);
    });
    const renderAIImages = () => {
        aiImages.replaceChildren();
        const roles = visualRolesForTarget();
        aiVisualFiles.forEach((item, index) => {
            const slot = String(index + 1);
            const card = el("div", "iamccs-pr-ai-image");
            const thumb = el("img", "iamccs-pr-ai-thumb");
            thumb.src = item.dataUrl;
            const meta = el("div", "iamccs-pr-ai-image-meta");
            meta.appendChild(el("div", "iamccs-pr-ai-image-name", `Picture ${slot} · ${item.file.name}`));
            const role = el("select");
            ["ignore", "opening", "closing", "identity", "composition", "style", "reference"].forEach((value) => {
                const option = document.createElement("option"); option.value = value; option.textContent = value; role.appendChild(option);
            });
            role.value = String(roles[slot] || (index === 0 ? "opening" : index === 1 ? "closing" : "reference"));
            role.onchange = () => { visualRolesForTarget()[slot] = role.value; commit(); };
            meta.appendChild(role);
            card.append(thumb, meta);
            aiImages.appendChild(card);
        });
    };
    const loadOllamaModels = async ({ quiet = false } = {}) => {
        if (aiProvider.value !== "ollama") return [];
        refreshModelsBtn.disabled = true;
        if (!quiet) aiStatus.textContent = "Reading installed Ollama models…";
        try {
            const response = await api.fetchApi(`/iamccs/prompter/ollama/models?base_url=${encodeURIComponent(aiBaseUrl.value.trim() || "http://127.0.0.1:11434")}`);
            const data = await response.json();
            if (!response.ok || !data?.ok) throw new Error(data?.error || `HTTP ${response.status}`);
            const names = (data.models || []).map((item) => String(item.name || "")).filter(Boolean);
            aiModelList.replaceChildren(...names.map((name) => {
                const option = document.createElement("option"); option.value = name; return option;
            }));
            if ((!aiModel.value.trim() || !names.includes(aiModel.value.trim())) && names.length) aiModel.value = names[0];
            persistAI();
            aiStatus.className = "iamccs-pr-ai-status ok";
            aiStatus.textContent = names.length ? `${names.length} Ollama model(s) available. Selected: ${aiModel.value}.` : "Ollama is reachable but has no installed models.";
            return names;
        } catch (error) {
            aiStatus.className = "iamccs-pr-ai-status error";
            aiStatus.textContent = `Ollama unavailable: ${error?.message || error}`;
            return [];
        } finally {
            refreshModelsBtn.disabled = false;
        }
    };
    aiProvider.onchange = async () => {
        const selected = aiDefaults[aiProvider.value] || {};
        aiBaseUrl.value = selected.baseUrl || "";
        aiModel.value = selected.model || "";
        aiApiKey.value = "";
        persistAI();
        if (aiProvider.value === "ollama") await loadOllamaModels();
    };
    [aiBaseUrl, aiModel, aiTemperature].forEach((control) => control.addEventListener("change", persistAI));
    refreshModelsBtn.onclick = () => loadOllamaModels();
    addAIImagesBtn.onclick = () => aiImageInput.click();
    aiImageInput.onchange = async () => {
        const files = Array.from(aiImageInput.files || []).filter((file) => /^image\//.test(file.type)).slice(0, 4);
        aiVisualFiles.splice(0, aiVisualFiles.length);
        for (const file of files) aiVisualFiles.push({ file, dataUrl: await readFileDataUrl(file) });
        renderAIImages();
        aiImageInput.value = "";
        aiStatus.className = "iamccs-pr-ai-status";
        aiStatus.textContent = `${aiVisualFiles.length} temporary AI image reference(s). Assign roles for ${project.injection_target}; images are not saved inside the workflow.`;
    };

    const renderPreview = () => {
        const prompt = composePrompt(project);
        preview.textContent = prompt;
        const budget = Number(widget(node, "character_budget")?.value || 6800);
        charPill.textContent = `${prompt.length} / ${budget} chars`;
        charPill.className = `iamccs-pr-pill ${prompt.length > 7000 ? "warn" : "ok"}`;
        const fields = MODE_META[project.task_mode].sections;
        const filled = fields.filter(([key]) => String(project.sections?.[key] || "").trim()).length;
        completePill.textContent = `${filled}/${fields.length} sections`;
        completePill.className = `iamccs-pr-pill ${filled === fields.length ? "ok" : "warn"}`;
        assistantBanner.classList.toggle("show", project.writing_mode === "assistant_fill");
    };

    const renderSections = () => {
        center.replaceChildren();
        activePromptArea = null;
        activePromptKey = null;
        center.appendChild(tagDeck);
        const meta = MODE_META[project.task_mode];
        meta.sections.forEach(([key, label, tip], index) => {
            const card = el("section", "iamccs-pr-section");
            const head = el("div", "iamccs-pr-section-head");
            head.append(el("div", "iamccs-pr-num", String(index + 1)), el("div", "iamccs-pr-section-title", label));
            const state = el("div", "iamccs-pr-state");
            head.appendChild(state);
            const area = el("textarea", "iamccs-pr-text");
            area.dataset.sectionKey = key;
            area.addEventListener("focus", () => {
                activePromptArea = area;
                activePromptKey = key;
                tagHint.textContent = `Active field: ${label}`;
            });
            area.value = String(project.sections?.[key] || "");
            area.placeholder = `Write ${label}…`;
            const refreshState = () => {
                const empty = !area.value.trim();
                card.classList.toggle("iamccs-pr-empty", empty);
                state.textContent = empty ? "empty" : `${area.value.trim().length} chars`;
            };
            area.addEventListener("input", () => {
                project.sections[key] = area.value;
                refreshState();
                renderPreview();
                commit();
            });
            card.append(head, area, el("div", "iamccs-pr-tip", tip));
            center.appendChild(card);
            refreshState();
        });
        modeButtons.forEach((item, key) => item.classList.toggle("active", key === project.task_mode));
        modeNote.textContent = meta.subtitle;
        renderPreview();
    };

    const populateAIScope = () => {
        const previous = String(project.ai_scope || aiScope.value || "active_field");
        aiScope.replaceChildren();
        const choices = [
            ["active_field", "Active prompt field"],
            ["all_filled", "All filled fields"],
            ...MODE_META[project.task_mode].sections.map(([key, label]) => [key, `Section · ${label}`]),
        ];
        choices.forEach(([value, label]) => {
            const option = document.createElement("option"); option.value = value; option.textContent = label; aiScope.appendChild(option);
        });
        aiScope.value = choices.some(([value]) => value === previous) ? previous : "active_field";
        project.ai_scope = aiScope.value;
    };

    const renderControls = () => {
        nameInput.value = project.project_name;
        policy.value = project.merge_policy;
        exampleSelect.disabled = project.task_mode !== "t2va";
        exampleSelect.title = project.task_mode === "t2va" ? "Choose a cinematic T2V prompt project" : "T2V cinematic projects are available in T2VA mode";
        targetButtons.forEach((item, key) => item.classList.toggle("active", key === project.injection_target));
        writingButtons.forEach((item, key) => item.classList.toggle("active", key === project.writing_mode));
        populateAIScope();
        aiDirection.value = String(project.ai_direction || "");
        root.classList.toggle("mode-manual", project.writing_mode === "manual");
        aiPanel.classList.toggle("show", project.writing_mode === "assistant_fill");
        renderAIImages();
        targetHint.textContent = project.injection_target === "local_auto"
            ? "The MiniMax Shotboard reads its timeline, selects the first empty local slot among 1–3, and appends to Local 3 only when all three already contain text."
            : project.injection_target === "global"
                ? "The composed prompt becomes the Shotboard global context. Local prompts can still add per-chunk action."
                : `Targets ${targetLabels[project.injection_target]}. If that slot does not exist, the Shotboard detects the available slots and uses a safe fallback.`;
    };

    const loadExample = (mode) => {
        project.task_mode = mode;
        const selectedT2V = mode === "t2va" ? T2V_PROJECTS.find((item) => item.id === exampleSelect.value) : null;
        const exampleSections = selectedT2V?.sections || EXAMPLES[mode] || {};
        project.sections = { ...project.sections, ...exampleSections };
        project.project_name = selectedT2V?.name || `${MODE_META[mode].label} Example Project`;
        renderControls();
        renderSections();
        commit();
    };

    rewriteBtn.onclick = async () => {
        const allSections = Object.fromEntries(
            MODE_META[project.task_mode].sections.map(([key]) => [key, String(project.sections?.[key] || "").trim()])
        );
        let targetKeys = [];
        if (aiScope.value === "all_filled") {
            targetKeys = Object.entries(allSections).filter(([, value]) => value).map(([key]) => key);
        } else if (aiScope.value === "active_field") {
            if (activePromptKey) targetKeys = [activePromptKey];
        } else if (Object.prototype.hasOwnProperty.call(allSections, aiScope.value)) {
            targetKeys = [aiScope.value];
        }
        const direction = aiDirection.value.trim();
        const hasRoughText = targetKeys.some((key) => String(allSections[key] || "").trim());
        if (!targetKeys.length) {
            aiStatus.className = "iamccs-pr-ai-status error";
            aiStatus.textContent = aiScope.value === "active_field" ? "Click the prompt field you want the AI to improve first." : "No filled field is available for this target.";
            return;
        }
        if (!hasRoughText && !direction && !aiVisualFiles.length) {
            aiStatus.className = "iamccs-pr-ai-status error";
            aiStatus.textContent = "Write a rough idea in the selected field or in User direction first.";
            return;
        }
        project.ai_direction = direction;
        project.ai_scope = aiScope.value;
        persistAI();
        commit();
        rewriteBtn.disabled = true;
        rewriteBtn.textContent = "Rewriting MiniMax fieldsâ€¦";
        aiStatus.className = "iamccs-pr-ai-status";
        aiStatus.textContent = `Sending ${targetKeys.join(", ")} to ${aiProvider.options[aiProvider.selectedIndex]?.text || aiProvider.value}.`;
        try {
            const roles = visualRolesForTarget();
            const imagePayload = aiVisualFiles.map((item, index) => ({
                slot: index + 1,
                name: item.file.name,
                role: String(roles[String(index + 1)] || (index === 0 ? "opening" : index === 1 ? "closing" : "reference")),
                mime_type: item.file.type || "image/png",
                data: item.dataUrl,
            })).filter((item) => item.role !== "ignore");
            const response = await api.fetchApi("/iamccs/prompter/rewrite", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    provider: aiProvider.value,
                    base_url: aiBaseUrl.value.trim(),
                    model: aiModel.value.trim(),
                    api_key: aiApiKey.value,
                    task_mode: project.task_mode,
                    sections: allSections,
                    target_keys: targetKeys,
                    user_direction: direction,
                    images: imagePayload,
                    temperature: Number(aiTemperature.value || 0.35),
                    timeout: 180,
                }),
            });
            const data = await response.json();
            if (!response.ok || !data?.ok) throw new Error(data?.error || `HTTP ${response.status}`);
            Object.entries(data.sections || {}).forEach(([key, value]) => {
                if (targetKeys.includes(key)) project.sections[key] = String(value || "");
            });
            renderControls();
            renderSections();
            commit();
            aiStatus.className = "iamccs-pr-ai-status ok";
            const visualCount = Number(data.report?.visual_references?.length || 0);
            aiStatus.textContent = `Improved: ${(data.report?.rewritten_sections || Object.keys(data.sections || {})).join(", ")}${visualCount ? ` with ${visualCount} visual reference(s)` : ""}. Review, then inject.`;
        } catch (error) {
            aiStatus.className = "iamccs-pr-ai-status error";
            aiStatus.textContent = `Rewrite failed: ${error?.message || error}`;
        } finally {
            aiApiKey.value = "";
            rewriteBtn.disabled = false;
            rewriteBtn.textContent = "Improve selected prompt with AI";
        }
    };

    modeButtons.forEach((item, key) => {
        item.onclick = () => {
            project.task_mode = key;
            renderControls();
            renderSections();
            commit();
        };
    });
    targetButtons.forEach((item, key) => {
        item.onclick = () => {
            project.injection_target = key;
            renderControls();
            commit();
        };
    });
    injectBtn.onclick = () => {
        commit();
        const prompt = composePrompt(project);
        if (!prompt.trim()) {
            injectStatus.className = "iamccs-pr-inject-status error";
            injectStatus.textContent = "Nothing injected: fill at least one prompt section.";
            return;
        }
        const targets = shotboardsForPrompter(node);
        if (!targets.length) {
            injectStatus.className = "iamccs-pr-inject-status error";
            injectStatus.textContent = "MiniMax Shotboard not found. Add one to this workflow and connect the Prompter CineLinX output.";
            return;
        }
        const shotboard = targets[0];
        try {
            if (typeof shotboard._iamccsMiniMaxInjectPrompt !== "function") {
                throw new Error("Shotboard UI bridge is not ready; reload ComfyUI once");
            }
            const result = shotboard._iamccsMiniMaxInjectPrompt({
                prompt,
                target: project.injection_target,
                mergePolicy: project.merge_policy,
            });
            injectStatus.className = "iamccs-pr-inject-status ok";
            injectStatus.textContent = `Injected into ${result?.actualTarget || project.injection_target} (${result?.mergePolicy || project.merge_policy}). The visible Shotboard and CineLinX queue request are synchronized.`;
            injectBtn.textContent = "INJECTED ✓";
            setTimeout(() => { injectBtn.textContent = "INJECT → SHOTBOARD"; }, 1200);
        } catch (error) {
            injectStatus.className = "iamccs-pr-inject-status error";
            injectStatus.textContent = `Injection failed: ${error?.message || error}`;
        }
    };
    writingButtons.forEach((item, key) => {
        item.onclick = () => {
            project.writing_mode = key;
            renderControls();
            renderPreview();
            commit();
        };
    });
    aiScope.onchange = () => { project.ai_scope = aiScope.value; commit(); };
    aiDirection.addEventListener("input", () => { project.ai_direction = aiDirection.value; commit(); });
    nameInput.addEventListener("input", commit);
    policy.addEventListener("change", commit);
    exampleBtn.onclick = () => loadExample(project.task_mode);
    saveBtn.onclick = () => { commit(); downloadProject(project); };
    loadBtn.onclick = () => fileInput.click();
    fileInput.onchange = async () => {
        const file = fileInput.files?.[0];
        if (!file) return;
        try {
            project = safeProject(await file.text());
            renderControls();
            renderSections();
            commit();
        } catch (error) {
            alert(`IAMCCS Prompter: invalid project file\n${error?.message || error}`);
        } finally {
            fileInput.value = "";
        }
    };
    copyBtn.onclick = async () => {
        const prompt = composePrompt(project);
        try {
            await navigator.clipboard.writeText(prompt);
            copyBtn.textContent = "Copied";
            setTimeout(() => { copyBtn.textContent = "Copy Prompt"; }, 900);
        } catch {
            const area = document.createElement("textarea");
            area.value = prompt;
            document.body.appendChild(area);
            area.select();
            document.execCommand("copy");
            area.remove();
        }
    };
    clearBtn.onclick = () => {
        if (!confirm(`Clear all ${MODE_META[project.task_mode].label} boxes?`)) return;
        MODE_META[project.task_mode].sections.forEach(([key]) => { project.sections[key] = ""; });
        renderSections();
        commit();
    };

    const domWidget = node.addDOMWidget("IAMCCS Prompter", "iamccs_prompter", root, { serialize: false });
    domWidget.computeSize = () => [980, 740];
    node.size = [980, 790];
    node.resizable = false;

    const originalSerialize = node.onSerialize;
    node.onSerialize = function(serialized) {
        commit();
        return originalSerialize?.call?.(this, serialized);
    };
    const originalConfigure = node.onConfigure;
    node.onConfigure = function(info) {
        const result = originalConfigure?.call?.(this, info);
        setTimeout(() => {
            project = safeProject(widget(node, "project_data")?.value);
            renderControls();
            renderSections();
        }, 0);
        return result;
    };

    renderControls();
    renderSections();
    commit();
    if (aiProvider.value === "ollama") setTimeout(() => loadOllamaModels({ quiet: true }), 0);
}

app.registerExtension({
    name: "IAMCCS.Prompter.MiniMaxH3",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_TYPE) return;
        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function() {
            const result = originalCreated?.apply?.(this, arguments);
            try { mountPrompter(this); } catch (error) { console.error("[IAMCCS Prompter] UI mount failed", error); }
            return result;
        };
    },
});

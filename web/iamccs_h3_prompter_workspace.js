// SPDX-License-Identifier: GPL-3.0-or-later

// Presentation only: move existing controls without replacing their handlers.
export function createPrompterWorkspace({ root, layout, left, center, right, modeBar, audioPanel, aiPanel, references, injectBtn, injectStatus }) {
    const make = (tag, className, text = "") => {
        const element = document.createElement(tag);
        element.className = className;
        element.textContent = text;
        return element;
    };
    root.classList.add("iamccs-pr-workspace");
    const style = document.createElement("style");
    style.textContent = `
    .iamccs-pr-workspace{border-color:#A97C38;border-radius:3px;background:#171D20;color:#F4F0E8}
    .iamccs-pr-workspace .iamccs-pr-top{background:linear-gradient(110deg,#293A3B,#211A11);border-bottom-color:#A97C38;flex-wrap:wrap;flex-shrink:0}
    .iamccs-pr-workspace .iamccs-pr-title{font-family:Arial,sans-serif;color:#F5D18A}.iamccs-pr-workspace .iamccs-pr-sub{color:#BAC8C8}
    .iamccs-pr-workspace .iamccs-pr-layout{grid-template-columns:minmax(260px,1fr) minmax(0,2fr)}
    .iamccs-pr-workspace .iamccs-pr-left{background:linear-gradient(160deg,#192B2B,#11191B);padding:14px;border-color:#53645F}
    .iamccs-pr-workspace .iamccs-pr-workarea{display:flex;flex-direction:column;min-width:0;min-height:0;background:#172124}
    .iamccs-pr-workspace .iamccs-pr-center{flex:1;min-height:0;padding:16px;background:linear-gradient(145deg,#1B292D,#171D20);overflow:auto}
    .iamccs-pr-workspace .iamccs-pr-right{flex:1;min-height:0;border:0;padding:16px;background:#172124}
    .iamccs-pr-workspace .iamccs-pr-worktabs{display:flex;flex-wrap:wrap;gap:5px;padding:12px 16px;border-bottom:1px solid #526469;background:#202B2E;flex-shrink:0}
    .iamccs-pr-workspace .iamccs-pr-worktab{height:32px;border-color:#587177;background:#172124;color:#D8E4E1;padding:0 13px;font-size:11px}
    .iamccs-pr-workspace .iamccs-pr-worktab[data-view=direction].active{background:#45325F;border-color:#B79ADB;color:#F3E8FF}
    .iamccs-pr-workspace .iamccs-pr-worktab[data-view=structure].active{background:#5D3C14;border-color:#D7A74F;color:#FFF0C5}
    .iamccs-pr-workspace .iamccs-pr-worktab[data-view=shots].active{background:#20445A;border-color:#6DAED1;color:#DEF3FF}
    .iamccs-pr-workspace .iamccs-pr-worktab[data-view=voices].active{background:#234D40;border-color:#62C7A9;color:#D6F6E4}
    .iamccs-pr-workspace .iamccs-pr-worktab[data-view=review].active{background:#5D3C14;border-color:#F2C671;color:#FFF0C5}
    .iamccs-pr-workspace .iamccs-pr-hidden{display:none!important}
    .iamccs-pr-workspace .iamccs-pr-group{margin:0 0 13px;padding:10px;border:1px solid #526469;border-radius:3px;background:#172326}
    .iamccs-pr-workspace .iamccs-pr-group>summary{color:#EAC77F;font-size:11px;font-weight:700;cursor:pointer;padding:3px 0}
    .iamccs-pr-workspace .iamccs-pr-groupbody{padding-top:10px;min-width:0}
    .iamccs-pr-workspace .iamccs-pr-group[data-tone=violet]{border-color:#795B9C;background:#241E30}
    .iamccs-pr-workspace .iamccs-pr-group[data-tone=blue]{border-color:#477C9C;background:#152C38}
    .iamccs-pr-workspace .iamccs-pr-group[data-tone=green]{border-color:#4E8C70;background:#17352D}
    .iamccs-pr-workspace .iamccs-pr-modepicker{width:100%;min-height:36px;background:#5D3C14;color:#FFF0C5;border:1px solid #D7A74F;border-radius:3px;padding:5px 8px;font-size:12px;margin-bottom:10px}
    .iamccs-pr-workspace .iamccs-pr-modes{height:auto;display:flex;flex-wrap:wrap;overflow:visible;padding:5px 0;background:transparent;border:0;gap:5px}
    .iamccs-pr-workspace .iamccs-pr-mode{flex:1 1 115px;white-space:normal;height:auto;min-height:30px;font-size:10px;padding:6px}
    .iamccs-pr-workspace .iamccs-pr-mode-note{max-width:none;flex:1 0 100%;white-space:normal;color:#BAC8C8;margin:5px 0}
    .iamccs-pr-workspace .iamccs-pr-ai{box-shadow:none;margin:0;border:0;padding:0;background:transparent}
    .iamccs-pr-workspace .iamccs-pr-references .iamccs-pr-ai-images{margin:10px 0}.iamccs-pr-workspace .iamccs-pr-references>.iamccs-pr-btn{width:100%;height:auto;min-height:32px;padding:7px;white-space:normal}
    .iamccs-pr-workspace .iamccs-pr-references select{width:100%;min-width:0;max-width:100%;background:#172124;color:#D8E4E1;border:1px solid #587177;padding:6px;font-size:10px}
    .iamccs-pr-workspace .iamccs-pr-section{box-shadow:none;border-radius:3px}.iamccs-pr-workspace .iamccs-pr-section-head{height:auto;min-height:38px;flex-wrap:wrap;padding:7px 10px}
    .iamccs-pr-workspace .iamccs-pr-text{font-size:13px!important;line-height:1.6!important;min-height:110px}
    .iamccs-pr-workspace .iamccs-pr-section[data-workspace=direction] .iamccs-pr-text{min-height:180px}
    .iamccs-pr-workspace .iamccs-pr-tagdeck{position:static;box-shadow:none;background:#203136;border-color:#587177}
    .iamccs-pr-workspace .iamccs-pr-workfooter{flex-shrink:0;padding:12px 16px;border-top:1px solid #66746B;background:#202B2E;display:grid;grid-template-columns:minmax(0,1fr) auto;gap:10px;align-items:center}
    .iamccs-pr-workspace .iamccs-pr-workfooter .iamccs-pr-inject{grid-column:2;grid-row:1;width:auto;max-width:240px;margin:0;padding:0 16px;box-shadow:none;background:linear-gradient(135deg,#D7A74F,#825A24)!important}
    .iamccs-pr-workspace .iamccs-pr-workfooter .iamccs-pr-inject-status{grid-column:1;grid-row:1;margin:0;min-height:0;background:transparent;border:0;font-size:10px}
    .iamccs-pr-workspace .iamccs-pr-voice-tools{overflow:auto;min-height:0;padding:0 16px 16px;flex:1}.iamccs-pr-workspace .iamccs-pr-voice-tools .iamccs-pr-ai{display:grid;padding:12px;border:1px solid #4E8C70;background:#17352D}
    .iamccs-pr-workspace .iamccs-pr-preview{font-size:13px;line-height:1.6}.iamccs-pr-workspace .iamccs-pr-kicker{font-size:10px;color:#EAC77F}
    `;
    root.appendChild(style);
    const group = (title, tone, open = false) => {
        const details = make("details", "iamccs-pr-group");
        details.dataset.tone = tone;
        details.open = open;
        const body = make("div", "iamccs-pr-groupbody");
        details.append(make("summary", "", title), body);
        return { details, body };
    };
    const modes = group("All mode presets", "gold");
    const modePicker = make("select", "iamccs-pr-modepicker");
    modePicker.setAttribute("aria-label", "Prompter mode");
    const modeButtons = [...modeBar.querySelectorAll("button")];
    modeButtons.forEach((button, index) => modePicker.add(new Option(button.textContent, String(index))));
    modePicker.onchange = () => modeButtons[Number(modePicker.value)].click();
    modes.body.appendChild(modeBar);
    const referenceGroup = group("Reference images", "blue", true);
    referenceGroup.body.classList.add("iamccs-pr-references");
    references.forEach(element => referenceGroup.body.appendChild(element));
    const oldChildren = [...left.children];
    left.replaceChildren(modePicker, modes.details, referenceGroup.details);
    let current = null;
    oldChildren.forEach(element => {
        if ([audioPanel, aiPanel, injectBtn, injectStatus].includes(element)) return;
        if (element.classList.contains("iamccs-pr-kicker")) {
            current = group(element.textContent, "gold");
            left.appendChild(current.details);
        } else if (current) current.body.appendChild(element);
        else left.appendChild(element);
    });
    const aiGroup = group("AI writer, vision & connection", "violet");
    aiGroup.body.appendChild(aiPanel);
    left.appendChild(aiGroup.details);
    const workarea = make("div", "iamccs-pr-workarea");
    const tabs = make("nav", "iamccs-pr-worktabs");
    tabs.setAttribute("aria-label", "Creative workspace");
    const voiceTools = make("div", "iamccs-pr-voice-tools");
    voiceTools.appendChild(audioPanel);
    const footer = make("div", "iamccs-pr-workfooter");
    footer.append(injectStatus, injectBtn);
    workarea.append(tabs, center, voiceTools, right, footer);
    layout.appendChild(workarea);
    let selected = "direction";
    const buttons = new Map();
    for (const [key, title] of [["direction", "Direction"], ["structure", "H3 fields"], ["shots", "Shots"], ["voices", "Voices / transcript"], ["review", "Final / checks"]]) {
        const button = make("button", "iamccs-pr-btn iamccs-pr-worktab", title);
        button.type = "button";
        button.dataset.view = key;
        button.onclick = () => { selected = key; refresh(); };
        tabs.appendChild(button);
        buttons.set(key, button);
    }
    function refresh() {
        const hasShots = Boolean(center.querySelector('[data-workspace="shots"]'));
        buttons.get("shots").disabled = !hasShots;
        if (selected === "shots" && !hasShots) selected = "direction";
        modePicker.value = String(Math.max(0, modeButtons.findIndex(button => button.classList.contains("active"))));
        for (const [key, button] of buttons) {
            button.classList.toggle("active", key === selected);
            button.setAttribute("aria-pressed", String(key === selected));
        }
        center.classList.toggle("iamccs-pr-hidden", selected === "review");
        right.classList.toggle("iamccs-pr-hidden", selected !== "review");
        voiceTools.classList.toggle("iamccs-pr-hidden", selected !== "voices");
        for (const child of center.children) {
            const view = child.dataset.workspace || "direction";
            child.classList.toggle("iamccs-pr-hidden", view === "tags" ? !["direction", "structure", "shots"].includes(selected) : view !== selected);
        }
    }
    refresh();
    return { refresh, showReview() { selected = "review"; refresh(); } };
}

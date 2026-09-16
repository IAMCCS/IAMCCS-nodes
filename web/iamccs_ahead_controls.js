import { app } from '/scripts/app.js';

export function bindAheadControls(node) {
    for(const name of ['freeze_tail_max_frames','freeze_tail_threshold']) {
        const item=node.widgets?.find(w=>w.name===name);if(!item||item._aheadBound)continue;
        item._aheadBound=true;const previous=item.callback;
        item.callback=function(value){const result=previous?.apply(this,arguments);const cut=node.widgets?.find(w=>w.name==='cut_plan');if(cut){cut.value='{}';cut.callback?.('{}');}document.dispatchEvent(new CustomEvent('iamccs:ahead-controls',{detail:{node}}));return result;};
    }
}

export function setAhead(node, name, value) {
    const item=node.widgets?.find(w=>w.name===name);if(!item)return;
    item.value=value;item.callback?.(value);
    // Editing the automatic policy deliberately supersedes previously saved manual OUTs.
    if(name==='freeze_tail_max_frames'||name==='freeze_tail_threshold') {
        const cut=node.widgets?.find(w=>w.name==='cut_plan');if(cut){cut.value='{}';cut.callback?.('{}');}
    }
    node.setDirtyCanvas?.(true,true);app.graph?.change?.();
    document.dispatchEvent(new CustomEvent('iamccs:ahead-controls',{detail:{node}}));
}

export function mountFreezeControls(root,node) {
    bindAheadControls(node);
    const box=document.createElement('div');box.style.cssText='padding:10px;border:1px solid #658b92;display:flex;gap:8px;flex-wrap:wrap;align-items:center';
    const title=document.createElement('strong');title.textContent='FREEZE-AWARE · next generation';box.append(title);
    const fields=[];
    for(const [name,label,max,step] of [['freeze_tail_max_frames','Max tail frames',72,1],['freeze_tail_threshold','Static threshold',0.02,0.0005]]) {
        const widget=node.widgets?.find(w=>w.name===name);if(!widget)continue;
        const row=document.createElement('label');row.textContent=label+' ';const input=document.createElement('input');input.type='number';input.min=0;input.max=max;input.step=step;input.value=widget.value;input.style.width='80px';input.onchange=()=>setAhead(node,name,Number(input.value));row.append(input);box.append(row);fields.push([widget,input]);
    }
    for(const [label,max] of [['OFF',0],['SAFE 6f',6],['LONG FREEZE 72f',72]]){const b=document.createElement('button');b.textContent=label;b.onclick=()=>setAhead(node,'freeze_tail_max_frames',max);box.append(b);}
    const note=document.createElement('small');note.textContent='Only a terminal run of at least 2 static frames is removed. 0 disables auto trim. Changing this policy clears manual OUTs; saving OUT overrides auto only for that clip. Values are shared with Settings Control Room.';box.append(note);root.append(box);
    const sync=()=>{for(const [w,input]of fields)if(document.activeElement!==input)input.value=w.value;};
    document.addEventListener('iamccs:ahead-controls',sync);
    const timer=setInterval(()=>{if(!box.isConnected){clearInterval(timer);document.removeEventListener('iamccs:ahead-controls',sync);return;}sync();},400);
}

export function mountFlashControls(root,node) {
    const policy=node.widgets?.find(w=>w.name==='flash_guard');
    const sensitivity=node.widgets?.find(w=>w.name==='flash_guard_sensitivity');
    const radius=node.widgets?.find(w=>w.name==='flash_guard_radius');
    if(!policy||!sensitivity||!radius)return;
    const box=document.createElement('section');box.style.cssText='margin-top:12px;padding:14px;border:1px solid #826b9c;background:#1a2031;border-radius:8px';
    const title=document.createElement('h3');title.textContent='H3 PERIODIC FLASH GUARD · delivery only';title.style.margin='0 0 7px';
    const note=document.createElement('p');note.textContent='Optional post-processing: replaces selected frames with a dissolve between nearby frames. This can hide a flash but can also create ghosting or suppress motion. It does not correct the sampler. RAW frames and audio timing are preserved.';
    const mode=document.createElement('select');
    for(const [value,label] of [['auto','AUTO · repair detected spikes'],['off','OFF · decoded frames untouched'],['all_periodic','ALL PERIODIC · diagnostic / strong repair']])mode.add(new Option(label,value));
    const ratio=document.createElement('select');for(const n of [1.4,1.6,1.8,2,2.4,3])ratio.add(new Option(`${n.toFixed(1)}× local motion`,n));
    const size=document.createElement('select');for(const n of [1,2,3,4,5,6])size.add(new Option(`±${n} frames`,n));
    const row=document.createElement('div');row.style.cssText='display:grid;grid-template-columns:2fr 1fr 1fr;gap:9px';row.append(mode,ratio,size);
    const warning=document.createElement('small');warning.textContent='Default: OFF. Compare against RAW before using this treatment. ALL PERIODIC also replaces frames when no flash was detected.';
    const sync=()=>{mode.value=String(policy.value||'off');ratio.value=String(Number(sensitivity.value||2));size.value=String(Number(radius.value||2));ratio.disabled=size.disabled=mode.value==='off';};
    mode.onchange=()=>{setAhead(node,'flash_guard',mode.value);sync();};ratio.onchange=()=>setAhead(node,'flash_guard_sensitivity',String(ratio.value));size.onchange=()=>setAhead(node,'flash_guard_radius',String(size.value));
    document.addEventListener('iamccs:ahead-controls',sync);const timer=setInterval(()=>{if(!box.isConnected){clearInterval(timer);document.removeEventListener('iamccs:ahead-controls',sync);return;}sync();},400);
    box.append(title,note,row,warning);root.append(box);sync();
}

export function mountDeliveryControls(root,node,{editor=false}={}) {
    const blend=node.widgets?.find(w=>w.name==='join_blend');
    const frames=node.widgets?.find(w=>w.name==='blend_frames');
    const audio=node.widgets?.find(w=>w.name==='audio_join');
    const smoothing=node.widgets?.find(w=>w.name==='audio_smoothing_ms');
    const box=document.createElement('section');
    box.className='ahead-delivery-controls';
    box.style.cssText='padding:14px;border:1px solid #658b92;background:#132630;border-radius:8px';
    const heading=document.createElement('h3');heading.textContent='DELIVERY JOIN · video + audio';heading.style.margin='0 0 8px';box.append(heading);
    const intro=document.createElement('p');intro.textContent='These settings affect only the assembled delivery. Latent history, interval checkpoints and the RAW master remain unchanged.';box.append(intro);
    const cards=document.createElement('div');cards.style.cssText='display:grid;grid-template-columns:repeat(3,minmax(150px,1fr));gap:9px';box.append(cards);
    const modes=[
        ['none','CUT · preserve every frame','No video overlap and no duration loss. Recommended with CLICK-SAFE audio.'],
        ['linear','LINEAR DISSOLVE','Constant-rate AV overlap. It can reveal double edges.'],
        ['smoothstep','SMOOTHSTEP DISSOLVE','Eased AV overlap with a softer start and end.'],
    ];
    const cardButtons=[];
    for(const [value,label,note] of modes){
        const card=document.createElement('button');card.type='button';card.dataset.mode=value;card.style.cssText='text-align:left;white-space:normal;padding:12px;border-radius:7px;color:#eff;background:#233f4b;border:2px solid #54737c';
        const strong=document.createElement('strong');strong.textContent=label;const small=document.createElement('small');small.textContent=note;small.style.cssText='display:block;margin-top:6px;line-height:1.35';card.append(strong,small);card.onclick=()=>setAhead(node,'join_blend',value);cards.append(card);cardButtons.push(card);
    }
    const grid=document.createElement('div');grid.style.cssText='display:grid;grid-template-columns:repeat(2,minmax(220px,1fr));gap:12px;margin-top:14px';box.append(grid);
    const field=(label,control,note)=>{const wrap=document.createElement('label');wrap.style.display='block';const title=document.createElement('strong');title.textContent=label;const p=document.createElement('small');p.textContent=note;p.style.cssText='display:block;margin-top:5px;line-height:1.35';wrap.append(title,control,p);grid.append(wrap);return wrap;};
    const count=document.createElement('select');for(const n of [3,6,9,12,18,24])count.add(new Option(`${n} frames · ${(n/24).toFixed(3)} s`,n));count.value=String(frames?.value??9);count.onchange=()=>setAhead(node,'blend_frames',Number(count.value));
    const countField=field('Dissolve length',count,'Applied at every video join. Each join shortens the delivery by this duration. 9 frames matches the common Wan 2.2 overlap.');
    const audioSelect=document.createElement('select');for(const [value,label] of [['click_safe','CLICK-SAFE · recommended'],['hard_cut','HARD CUT · decoded samples unchanged']])audioSelect.add(new Option(label,value));audioSelect.value=audio?.value||'click_safe';audioSelect.onchange=()=>setAhead(node,'audio_join',audioSelect.value);
    const audioField=field('Audio boundary',audioSelect,'CLICK-SAFE applies a short fade-out/fade-in without changing duration. With a video dissolve, audio follows the full overlap.');
    const ms=document.createElement('select');for(const n of [0,5,10,20,40,80,120])ms.add(new Option(`${n} ms per side`,n));ms.value=String(smoothing?.value??20);ms.onchange=()=>setAhead(node,'audio_smoothing_ms',Number(ms.value));
    const msField=field('Audio click protection',ms,'20 ms per side removes most sample-edge clicks without a noticeable pause. Use 40–80 ms only for a strong transient.');
    const diagram=document.createElement('div');diagram.style.cssText='margin-top:16px;padding:12px;background:#0b171e;border-radius:7px;font:13px/1.4 Segoe UI,sans-serif';box.append(diagram);
    const sync=()=>{
        const mode=String(blend?.value||'none');
        for(const card of cardButtons){const on=card.dataset.mode===mode;card.style.borderColor=on?'#7de2c4':'#54737c';card.style.background=on?'#245f53':'#233f4b';}
        count.value=String(frames?.value??9);audioSelect.value=String(audio?.value||'click_safe');ms.value=String(smoothing?.value??20);
        count.disabled=mode==='none';countField.style.opacity=mode==='none'?'0.55':'1';
        audioSelect.disabled=mode!=='none';audioField.style.opacity=mode!=='none'?'0.55':'1';
        ms.disabled=audioSelect.value!=='click_safe'||mode!=='none';msField.style.opacity=ms.disabled?'0.55':'1';
        const n=Number(frames?.value||9),a=String(audio?.value||'click_safe');
        diagram.textContent=mode==='none'
            ? `PREVIOUS CLIP | CUT | NEXT CLIP · every frame stays · audio: ${a==='click_safe'?`${smoothing?.value??20} ms smoothing per side, duration unchanged`:'unaltered hard cut'}`
            : `PREVIOUS TAIL [ ${n} FRAME ${mode.toUpperCase()} OVERLAP ] NEXT HEAD · successor frame 0 enters the first mixed frame · ${(n/24).toFixed(3)} s removed per join`;
    };
    document.addEventListener('iamccs:ahead-controls',sync);
    audioSelect.addEventListener('change',sync);sync();
    const timer=setInterval(()=>{if(!box.isConnected){clearInterval(timer);document.removeEventListener('iamccs:ahead-controls',sync);return;}sync();},400);
    if(editor){const guide=document.createElement('p');guide.textContent='Choose CUT when latent continuity is already good. Try SMOOTHSTEP 9f when motion still has a small visual groove. If a dissolve shows double contours, return to CUT and use the local Seam Export tab for diagnosis.';guide.style.cssText='padding:10px;border-left:4px solid #f0ca72';box.append(guide);}
    root.append(box);
}

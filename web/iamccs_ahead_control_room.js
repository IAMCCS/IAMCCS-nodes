import { mountFreezeControls, mountDeliveryControls, mountFlashControls, setAhead } from './iamccs_ahead_controls.js';
import { openCutEditor } from './iamccs_ahead_cut_editor.js';
// SPDX-License-Identifier: GPL-3.0-or-later
import { api } from '/scripts/api.js';
import { publishAheadResult } from './iamccs_ahead_result_preview.js';

const urlFor = (run, filename) => api.apiURL('/view?'+new URLSearchParams({filename,subfolder:`IAMCCS/LatentGoAhead/${run}`,type:'output'}));
const button = (text, action) => {const b=document.createElement('button'); b.textContent=text;b.onclick=action;return b;};
const help = text => {const p=document.createElement('p');p.textContent=text;return p;};
export function mountAheadRoom(root, node) {
    root.replaceChildren();
    root.append(help('AHEAD CONTROL ROOM · Motion controls apply to the next generation. Seam edits apply only to a separate exported copy.'));
    if(!node){root.append(help('Connect one LatentGoAhead branch to use these controls.'));return;}
    for(const [name,label,note] of [
        ['video_context_frames','Motion history','18 / 35 / 52 frames. More history may help motion but does not guarantee a seamless join.'],
]) {
        const w=node.widgets?.find(w=>w.name===name); if(!w)continue;
        const box=document.createElement('label');box.style.cssText='display:block;margin:14px 0';
        box.append(document.createTextNode(label+' '));
        const control=document.createElement(name==='video_context_frames'?'select':'input');
        if(control.tagName==='SELECT') for(const value of ['18','35','52'])control.add(new Option(`${value} f · ${(Number(value)/24).toFixed(2)} s`,value));
        else {control.type='number';control.min='0';control.max=name==='freeze_tail_threshold'?'0.02':'12';control.step=name==='freeze_tail_threshold'?'0.0005':'1';}
        control.value=w.value;control.onchange=()=>{w.value=control.tagName==='SELECT'?control.value:Number(control.value);w.callback?.(w.value);node.setDirtyCanvas?.(true,true);};
        box.append(control,help(note));root.append(box);
    }
    mountDeliveryControls(root,node);
    mountFlashControls(root,node);
    mountFreezeControls(root,node);
    root.append(button('OPEN EDITOR · TRIM / SEAMS / DELIVERY JOIN',()=>openAheadEditor(node,'cut')));
    root.append(help('Live values above belong to the connected LatentGoAhead node and are saved in the workflow. Audio context is kept unchanged.'));
}

export async function openAheadEditor(node, initialTab='cut') {
    const shell=document.createElement('dialog');shell.style.cssText='width:96vw;height:94vh;max-width:none;max-height:none;padding:14px;background:#0c1923;color:#e3f4f1;border:1px solid #76baac;box-sizing:border-box;overflow:auto';
    const tabs=document.createElement('nav');tabs.style.cssText='display:flex;gap:10px;align-items:center;position:sticky;top:0;background:#0c1923;z-index:2;padding:8px';
    const title=document.createElement('strong');title.textContent='AHEAD / EDITOR';title.style.marginRight='auto';
    const body=document.createElement('div');let activeHost=null;
    const choose=tab=>{if(activeHost)activeHost.dispatchEvent(new Event('ahead-dispose'));body.replaceChildren();activeHost=document.createElement('div');body.append(activeHost);for(const b of tabs.querySelectorAll('[data-tab]'))b.style.background=b.dataset.tab===tab?'#287361':'#263d4b';if(tab==='cut')openCutEditor(node,activeHost);else if(tab==='seam')openSeamEditor(node,activeHost);else openDeliveryEditor(node,activeHost);};
    tabs.append(title);for(const [id,label] of [['cut','1 · TRIM / SET OUT'],['seam','2 · LOCAL SEAM / EXPORT'],['delivery','3 · DELIVERY JOIN / AUDIO']]){const b=button(label,()=>choose(id));b.dataset.tab=id;b.style.cssText='padding:10px;color:white;border:1px solid #76baac;border-radius:6px';tabs.append(b);}
    tabs.append(button('CLOSE',()=>shell.close()));shell.append(tabs,body);shell.addEventListener('close',()=>{activeHost?.dispatchEvent(new Event('ahead-dispose'));shell.remove();},{once:true});document.body.append(shell);shell.showModal();choose(initialTab);
}

function openDeliveryEditor(node, host) {
    const page=document.createElement('section');page.style.cssText='max-width:1150px;margin:18px auto;padding:20px;background:#101f29;border:1px solid #527f7a;border-radius:10px';
    const title=document.createElement('h2');title.textContent='DELIVERY JOIN / AUDIO';
    const guide=help('This tab sets the automatic join used after the next generation. It never modifies the latent chain. CUT preserves all frames; LINEAR and SMOOTHSTEP overlap the tail and head and therefore shorten the final delivery.');
    page.append(title,guide);mountDeliveryControls(page,node,{editor:true});mountFlashControls(page,node);
    page.append(help('RAW MASTER is always written before this treatment. The returned FINAL file uses the selection above, so you can compare both outputs in the LatentGoAhead preview.'));
    host.append(page);
}

async function openSeamEditor(node, host) {
    const run=node.properties?.iamccs_lga_run;
    if(!run){const empty=help('LOCAL SEAM / EXPORT is ready. Complete one run from this branch first; the editor will attach only that new master and will not reuse an older Shotboard run.');empty.style.cssText='margin:24px;padding:20px;background:#13232c;border:1px solid #6ba99c';host.append(empty);return;}
    const dialog=document.createElement('section');
    dialog.style.cssText='width:min(1200px,94vw);max-height:94vh;overflow:auto;background:#111c23;color:#dfedea;border:1px solid #468579;padding:20px;border-radius:12px';
    const style=document.createElement('style');style.textContent='.ahead-room button,.ahead-room select{background:#27423f;color:#eff;border:1px solid #628b83;padding:8px;margin:4px;border-radius:5px}.ahead-room p{font-size:13px;line-height:1.45}.ahead-room input{accent-color:#64d9bb;max-width:100%}.ahead-strip{width:100%;height:90px;touch-action:none;cursor:ew-resize;background:#070c10}.ahead-monitor{width:100%;height:min(40vh,380px);object-fit:contain;background:black}.ahead-room:fullscreen{max-height:none;width:100vw;height:100vh;box-sizing:border-box}';
    dialog.className='ahead-room';dialog.append(style);
    let closed=false, playing=false, serial=0, playToken=0;
    const close=()=>{closed=true;playing=false;host.closest('dialog')?.close();};
    host.addEventListener('ahead-dispose',()=>{closed=true;playing=false;serial++;playToken++;},{once:true});
    dialog.append(button('CLOSE',close),help('LOCAL SEAM REVIEW · Changes replace a few video frames without shortening the movie. Audio is copied unchanged on export. Blending can create double edges: compare RAW before accepting.'));
    host.append(dialog);
    const status=help('Loading completed master…');dialog.append(status);
    try {
        const response=await api.fetchApi(`/iamccs/ahead/seams/${run}`);const info=await response.json();if(closed)return;if(!response.ok)throw Error(info.error || 'Restart ComfyUI to load the seam editor endpoint.');
        const select=document.createElement('select');select.setAttribute('aria-label','Select video join');info.boundaries.forEach((b,i)=>select.add(new Option(`JOIN ${i+1} / ${info.boundaries.length} · ${(b/info.fps).toFixed(2)} s`,i)));
        let seam=0,position=info.boundaries[0];
        node.properties ||= {};const saved=node.properties.iamccs_ahead_seams;
        let edits=info.boundaries.map((_,i)=>({seam:i,radius:0,offset:0,method:'smoothstep'}));
        if(saved?.run===run && saved.edits?.length===edits.length)edits=saved.edits.map(e=>({...e}));
        let treated=true, comparison='playback', opacity=0.5, span=12;
        const monitor=document.createElement('canvas');monitor.className='ahead-monitor';
        const video=document.createElement('video');video.muted=true;video.preload='auto';
        const ready=new Promise((resolve,reject)=>{video.onloadeddata=resolve;video.onerror=()=>reject(Error('Cannot decode master'));});video.src=urlFor(run,'final_film.mp4');host.addEventListener('ahead-dispose',()=>{video.pause();video.removeAttribute('src');video.load();},{once:true});await ready;if(closed)return;
        monitor.width=video.videoWidth;monitor.height=video.videoHeight;
        const cache=new Map();const cacheLimit=Math.max(2,Math.min(70,Math.floor(96*1024*1024/(monitor.width*monitor.height*4))));let work=Promise.resolve();
        const frame=(n)=>{n=Math.max(0,Math.min(info.frames-1,Math.round(n)));const job=async()=>{
            if(cache.has(n))return cache.get(n);
            const t=(n+0.2)/info.fps;
            if(Math.abs(video.currentTime-t)>0.00001)await new Promise((resolve,reject)=>{const timeout=setTimeout(()=>reject(Error('Video seek timed out')),10000);video.onseeked=()=>{clearTimeout(timeout);resolve();};video.currentTime=t;});
            const c=document.createElement('canvas');c.width=monitor.width;c.height=monitor.height;c.getContext('2d').drawImage(video,0,0);
            if(cache.size>=cacheLimit)cache.delete(cache.keys().next().value);cache.set(n,c);return c;
        };const result=work.then(job);work=result.catch(()=>{});return result;};
        const slider=document.createElement('input');slider.type='range';slider.step='1';slider.style.width='100%';
        const radius=document.createElement('input');radius.type='range';radius.min=0;radius.max=6;radius.step=1;
        const offset=document.createElement('input');offset.type='range';offset.min=-6;offset.max=6;offset.step=1;
        const method=document.createElement('select');method.setAttribute('aria-label','Blend curve');
        for(const [value,label] of [['linear','LINEAR · constant mix'],['smoothstep','SMOOTHSTEP · soft ends'],['cosine','COSINE · film dissolve']])method.add(new Option(label,value));
        const labels=help('');const tail=document.createElement('canvas'),head=document.createElement('canvas');
        const compare=document.createElement('select');compare.setAttribute('aria-label','Monitor comparison');
        for(const [value,label] of [['playback','PLAYBACK'],['split','SPLIT · tail / head'],['onion','ONION SKIN · tail + head']])compare.add(new Option(label,value));
        const alpha=document.createElement('input');alpha.type='range';alpha.min=0;alpha.max=1;alpha.step=0.05;alpha.value=opacity;alpha.setAttribute('aria-label','Onion skin opacity');
        const timeline=document.createElement('canvas');timeline.width=1000;timeline.height=70;timeline.style.cssText='width:100%;height:70px;touch-action:none;cursor:ew-resize';
        const diagnostic=help('Visual checks are advisory, not proof of motion continuity. Analyse the join to compare image change and temporal change.');
        const zoom=document.createElement('select');zoom.setAttribute('aria-label','Timeline zoom');for(const v of [12,24,48])zoom.add(new Option(`Timeline ±${v} frames`,v));
        let mixImage=null;
        function paintTimeline(){const ctx=timeline.getContext('2d'),e=edits[seam],b=info.boundaries[seam],min=Number(slider.min),max=Number(slider.max),x=n=>(n-min)/(max-min)*1000;ctx.fillStyle='#08151c';ctx.fillRect(0,0,1000,70);if(mixImage)ctx.drawImage(mixImage,0,0);ctx.fillStyle='rgba(172,114,232,0.55)';const l=x(b+e.offset-e.radius),r=x(b+e.offset+e.radius);ctx.fillRect(l,16,Math.max(2,r-l),42);ctx.fillStyle='#fff';for(const p of [l,r])ctx.fillRect(p-4,16,8,42);ctx.strokeStyle='#f6c76a';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(x(b),0);ctx.lineTo(x(b),60);ctx.stroke();ctx.fillStyle='#fff';ctx.beginPath();ctx.moveTo(x(position),8);ctx.lineTo(x(position)-6,0);ctx.lineTo(x(position)+6,0);ctx.fill();ctx.fillRect(x(position),8,1,50);ctx.font='12px sans-serif';ctx.fillText(`${min} f`,5,66);ctx.fillText(`${max} f`,930,66);ctx.fillStyle='#f6c76a';ctx.fillText(`JOIN ${seam+1}`,Math.min(920,x(b)+8),13);
            for(const [c,start] of [[tail,b-10],[head,b]]){const cursor=c.parentElement?.querySelector('.ahead-cursor');if(cursor){cursor.style.display=position>=start&&position<start+10?'block':'none';cursor.style.left=`${(position-start+0.5)*10}%`;}}
        }
        for(const c of [tail,head]){c.className='ahead-strip';c.width=1000;c.height=90;}
        const curve=(value,kind)=>kind==='smoothstep'?value*value*(3-2*value):kind==='cosine'?(1-Math.cos(Math.PI*value))/2:value;
        function save(){node.properties.iamccs_ahead_seams={run,edits:edits.map(e=>({...e,method:['linear','smoothstep','cosine'].includes(e.method)?e.method:'smoothstep'}))};node.setDirtyCanvas?.(true,true);}
        async function draw(){const ticket=++serial,n=position,e=edits[seam],b=info.boundaries[seam]+e.offset,lo=b-e.radius-1,hi=b+e.radius;
            const ctx=monitor.getContext('2d');const raw=await frame(n);let left,right;
            if(comparison!=='playback'){left=await frame(lo);right=await frame(hi);if(closed||ticket!==serial)return;ctx.globalAlpha=1;ctx.drawImage(left,0,0);if(comparison==='onion'){ctx.globalAlpha=opacity;ctx.drawImage(right,0,0);ctx.globalAlpha=1;}else{ctx.save();ctx.beginPath();ctx.rect(monitor.width/2,0,monitor.width/2,monitor.height);ctx.clip();ctx.drawImage(right,0,0);ctx.restore();ctx.fillStyle='#66ffd8';ctx.fillRect(monitor.width/2,0,2,monitor.height);}labels.textContent=`${comparison.toUpperCase()} · endpoints ${lo} / ${hi} · diagnostic only; export is unchanged by this view.`;paintTimeline();return;}
            if(treated&&e.radius&&n>lo&&n<hi){left=await frame(lo);right=await frame(hi);}
            if(closed||ticket!==serial)return;ctx.globalAlpha=1;ctx.drawImage(left||raw,0,0);
            if(right){ctx.globalAlpha=curve((n-lo)/(hi-lo),e.method||'smoothstep');ctx.drawImage(right,0,0);ctx.globalAlpha=1;}
            slider.value=n;labels.textContent=`${treated?'TREATED':'RAW'} · frame ${n} · ${2*e.radius} replaced frames (${(2*e.radius/info.fps*1000).toFixed(0)} ms) · window offset ${e.offset} f. Duration and audio unchanged.`;
            paintTimeline();
        }
        async function strips(){const ticket=seam,zoomSpan=span,b=info.boundaries[seam];for(const [c,start]of[[tail,b-10],[head,b]]){const ctx=c.getContext('2d');for(let j=0;j<10;j++){const n=Math.max(0,Math.min(info.frames-1,start+j));const image=await frame(n);if(closed||ticket!==seam||zoomSpan!==span)return;ctx.drawImage(image,j*100,0,100,70);ctx.fillStyle='#bdffe4';ctx.fillText(String(n),j*100+5,84);}}
            const strip=document.createElement('canvas');strip.width=1000;strip.height=70;const ctx=strip.getContext('2d'),min=Number(slider.min),max=Number(slider.max);
            for(let j=0;j<12;j++){const n=Math.round(min+(j+0.5)/12*(max-min));const image=await frame(n);if(closed||ticket!==seam||zoomSpan!==span)return;ctx.drawImage(image,j*1000/12,16,1000/12,42);}
            mixImage=strip;paintTimeline();
        }
        function refresh(){const b=info.boundaries[seam],e=edits[seam];e.method=['linear','smoothstep','cosine'].includes(e.method)?e.method:'smoothstep';radius.value=e.radius;offset.value=e.offset;method.value=e.method;slider.min=Math.max(0,b-span);slider.max=Math.min(info.frames-1,b+span);position=b;draw().catch(e=>status.textContent=e.message);strips().catch(e=>status.textContent=e.message);}
        const toggle=button('RAW / TREATED',()=>{treated=!treated;draw();});
        dialog.append(help('1 · Compare the endpoints. SPLIT and ONION are visual guides only; use PLAYBACK to judge the actual edit.'),compare,help('Onion opacity'),alpha);
        dialog.append(select,toggle,button('◀ FRAME',()=>{position=Math.max(Number(slider.min),position-1);draw();}),button('FRAME ▶',()=>{position=Math.min(Number(slider.max),position+1);draw();}),button('PLAY / PAUSE LOOP',()=>{playing=!playing;loop();}),monitor,slider,labels,help('TAIL · drag to move the treatment window. Click a frame to inspect it.'),tail,help('HEAD · same shared clock; no independent trim that could desynchronise audio.'),head,help('Blend size · zero = original'),radius,help('Move treatment window · frames'),offset);
        dialog.append(help('2 · Final timeline. White handles resize the blend symmetrically; drag the green area to move it. Gold line = original join. No clip duration or audio changes.'),zoom,timeline,diagnostic);
        method.onchange=()=>{edits[seam].method=method.value;save();draw();};
        compare.onchange=()=>{playing=false;playToken++;comparison=compare.value;draw();};alpha.oninput=()=>{opacity=Number(alpha.value);draw();};zoom.onchange=()=>{span=Number(zoom.value);refresh();};
        let timelineDrag=null;
        const frameAt=ev=>Number(slider.min)+(ev.clientX-timeline.getBoundingClientRect().left)/timeline.clientWidth*(Number(slider.max)-Number(slider.min));
        timeline.onpointerdown=ev=>{const f=frameAt(ev),e=edits[seam],center=info.boundaries[seam]+e.offset,tolerance=(Number(slider.max)-Number(slider.min))*10/timeline.clientWidth;timelineDrag=Math.abs(f-(center-e.radius))<tolerance||Math.abs(f-(center+e.radius))<tolerance?'resize':(f>=center-e.radius&&f<=center+e.radius?'move':'scrub');timeline.setPointerCapture(ev.pointerId);playing=false;};
        timeline.onpointermove=ev=>{if(!timelineDrag)return;const f=frameAt(ev),e=edits[seam],b=info.boundaries[seam];if(timelineDrag==='resize')e.radius=Math.min(6,Math.max(0,Math.round(Math.abs(f-b-e.offset))));else if(timelineDrag==='move')e.offset=Math.max(-6,Math.min(6,Math.round(f-b)));else position=Math.max(Number(slider.min),Math.min(Number(slider.max),Math.round(f)));radius.value=e.radius;offset.value=e.offset;draw();};
        timeline.onpointerup=()=>{timelineDrag=null;save();};timeline.onpointercancel=()=>{timelineDrag=null;};
        const thumb=async n=>{const c=document.createElement('canvas');c.width=64;c.height=36;const ctx=c.getContext('2d',{willReadFrequently:true});ctx.drawImage(await frame(n),0,0,64,36);return ctx.getImageData(0,0,64,36).data;};
        const difference=(a,b)=>{let sum=0;for(let i=0;i<a.length;i++)if(i%4!==3)sum+=Math.abs(a[i]-b[i]);return sum/(a.length/4*3*255);};
        const analyse=button('ANALYSE RAW JOIN',async()=>{analyse.disabled=true;const selected=seam;diagnostic.textContent='Analysing four raw frames…';try{const b=info.boundaries[selected];const a=await thumb(b-2),p=await thumb(b-1),q=await thumb(b),r=await thumb(b+1);if(selected!==seam||closed)return;const gap=difference(p,q),before=difference(a,p),after=difference(q,r),ratio=gap/Math.max(0.001,(before+after)/2);diagnostic.textContent=`Raw image change: ${(100*gap).toFixed(1)}% · change across join / neighbouring changes: ${ratio.toFixed(2)}×. ${ratio>2?'Join changes more than nearby frames: inspect for a jump.':gap<0.001?'Near-identical frames: inspect for a freeze.':'No large discontinuity in this simple pixel comparison.'} This is downsampled RGB difference, NOT optical flow or a continuity guarantee.`;}catch(e){diagnostic.textContent=e.message;}finally{analyse.disabled=false;}});
        dialog.append(analyse,help('3 · Review every join in PLAYBACK. Start with 2 frames; use zero if ghosting is worse. Export only after the comparison.'));
        async function loop(token=++playToken){if(!playing||closed||token!==playToken)return;position=position>=Number(slider.max)?Number(slider.min):position+1;try{await draw();}catch(e){playing=false;status.textContent=e.message;return;}setTimeout(()=>loop(token),1000/info.fps);}
        for(const c of [tail,head]){let start=null,initial=0,moved=false;c.onpointerdown=ev=>{start=ev.clientX;initial=edits[seam].offset;moved=false;c.setPointerCapture(ev.pointerId);};c.onpointermove=ev=>{if(start===null)return;const delta=Math.round((ev.clientX-start)/Math.max(1,c.clientWidth/10));moved ||= delta!==0;edits[seam].offset=Math.max(-6,Math.min(6,initial+delta));offset.value=edits[seam].offset;draw();};c.onpointerup=ev=>{if(!moved){const b=info.boundaries[seam];position=Math.max(0,Math.min(info.frames-1,b+(c===tail?-10:0)+Math.min(9,Math.floor((ev.clientX-c.getBoundingClientRect().left)/c.clientWidth*10))));draw();}start=null;save();};}
        select.onchange=()=>{playing=false;playToken++;seam=Number(select.value);diagnostic.textContent='Selected a different join. Both filmstrips and the timeline now refer to this boundary; settings are independent per join.';refresh();};slider.oninput=()=>{playing=false;position=Number(slider.value);draw();};
        radius.oninput=()=>{edits[seam].radius=Number(radius.value);save();draw();};offset.oninput=()=>{edits[seam].offset=Number(offset.value);save();draw();};
        const exportButton=button('EXPORT TREATED COPY · all joins',async()=>{exportButton.disabled=true;status.textContent='Exporting video; original audio packets will be copied…';try{const r=await api.fetchApi('/iamccs/ahead/seams/export',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run,edits})});const data=await r.json();if(!r.ok)throw Error(data.error);const a=document.createElement('a');a.textContent='OPEN EXPORTED FILM';a.href=urlFor(run,data.filename);a.target='_blank';status.replaceChildren(a);node.properties.iamccs_ahead_last_export=data;publishAheadResult(node,data);node.setDirtyCanvas?.(true,true);}catch(e){status.textContent=e.message;}finally{exportButton.disabled=false;}});
        const presets=document.createElement('select');presets.setAttribute('aria-label','Join preset');
        presets.add(new Option('APPLY PRESET · selected join',''));
        for(const [v,t] of [['0','CUT · untouched'],['1','MICRO · 2 frames'],['2','SOFT · 4 frames']])presets.add(new Option(t,v));
        presets.onchange=()=>{if(presets.value==='')return;edits[seam]={seam,radius:Number(presets.value),offset:0,method:edits[seam].method||'smoothstep'};save();refresh();presets.value='';};
        const best=button('BEST MIX · suggest selected join',async()=>{
            best.disabled=true;playing=false;playToken++;const selected=seam;
            diagnostic.textContent='Comparing boundary change with neighbouring motion…';
            try{
                const b=info.boundaries[selected];const a=await thumb(b-2),p=await thumb(b-1),q=await thumb(b),r=await thumb(b+1);
                if(closed||selected!==seam)return;
                const gap=difference(p,q),local=(difference(a,p)+difference(q,r))/2;
                // Conservative appearance heuristic, not motion estimation: never extend a freeze.
                const radius=gap>=0.001&&gap<0.08&&gap>Math.max(0.004,local*1.5)?1:0;
                edits[selected]={seam:selected,radius,offset:0,method:'smoothstep'};save();refresh();
                diagnostic.textContent=radius?'Suggested MICRO: 2 frames. Compare RAW/TREATED for ghosting; this is an appearance heuristic, not a proven optimum.':'Suggested CUT: no frames replaced. '+(gap<0.001?'Near-identical boundary frames: inspect a possible source freeze; blending cannot repair it.':gap>=0.08?'Large visual change: blending risks double images.':'No strong isolated jump detected by the pixel heuristic.');
            }catch(e){diagnostic.textContent=e.message;}finally{best.disabled=false;}
        });
        dialog.append(presets,best,exportButton);
        // Reparent existing controls: the same handlers and saved settings, no new edit path.
        const buttons=[...dialog.querySelectorAll('button')];
        const findButton=text=>buttons.find(b=>b.textContent===text);
        const header=document.createElement('header');header.className='ahead-header';
        const title=document.createElement('strong');title.textContent='IAMCCS / AHEAD CONTROL ROOM';
        const joinLabel=document.createElement('label');joinLabel.textContent='SELECT JOIN ';joinLabel.append(select);
        exportButton.textContent='APPLY & EXPORT · ALL JOINS';
        exportButton.title='Apply the current settings for every join to a new video. The original is preserved.';
        exportButton.style.cssText='background:#246c59;border:1px solid #89e6c6;font-weight:700;flex-shrink:0';
        header.append(title,joinLabel,exportButton,findButton('CLOSE'));
        const leftPanel=document.createElement('aside'),rightPanel=document.createElement('aside'),center=document.createElement('main');
        leftPanel.className='ahead-left';rightPanel.className='ahead-right';center.className='ahead-center';
        leftPanel.append(help('01 / INSPECT'),compare,help('Onion opacity · diagnostic only'),alpha,analyse,diagnostic);
        rightPanel.append(help('02 / TREAT SELECTED JOIN'),presets,best,help('Blend curve · updates the monitor in realtime'),method,help('Blend radius · 1 = 2 frames'),radius,help('Window offset · frames'),offset,help('White = playhead · Gold = join\nPurple = treatment. Drag its handles to resize. CUT adds no blend; source freezes remain.\nUse APPLY & EXPORT above to create the finished copy.'),status);
        const transport=document.createElement('nav');transport.append(toggle,findButton('◀ FRAME'),findButton('FRAME ▶'),findButton('PLAY / PAUSE LOOP'),zoom);
        const wrapStrip=(canvas,label)=>{const wrap=document.createElement('div');wrap.className='ahead-track';const caption=document.createElement('span');caption.textContent=label;const track=document.createElement('div');track.className='ahead-track-image';const cursor=document.createElement('i');cursor.className='ahead-cursor';track.append(canvas,cursor);wrap.append(caption,track);return wrap;};
        center.append(monitor,transport,slider,labels,wrapStrip(tail,'TAIL / previous chunk'),wrapStrip(head,'HEAD / next chunk'),wrapStrip(timeline,'MIX / final timeline'));
        dialog.replaceChildren(style,header,leftPanel,center,rightPanel);
        dialog.style.cssText='width:100%;height:78vh;max-width:none;max-height:none;margin:auto;padding:12px;box-sizing:border-box;overflow:hidden;background:#0b131b;color:#dce9ee;border:1px solid #426674;border-radius:12px';
        style.textContent+=`.ahead-room{display:grid;grid-template-columns:190px minmax(0,1fr) 220px;grid-template-rows:44px minmax(0,1fr);gap:10px}.ahead-header{grid-column:1/-1;display:flex;align-items:center;gap:10px}.ahead-header strong{flex:1;color:#93e2d2;font:600 14px sans-serif}.ahead-room aside{min-width:0;background:#14222d;padding:10px;border:1px solid #2d4353;border-radius:8px;display:flex;flex-direction:column;gap:7px}.ahead-room aside p{margin:2px 0;font-size:11px;line-height:1.3;white-space:pre-line}.ahead-room aside select,.ahead-room aside button{width:100%;margin:0;box-sizing:border-box;white-space:normal}.ahead-room button,.ahead-room select{font-size:11px;padding:6px;min-width:0}.ahead-center{min-height:0;min-width:0;display:grid;grid-template-rows:minmax(60px,1fr) auto 14px 30px repeat(3,66px);gap:5px}.ahead-room .ahead-monitor{height:100%;width:100%;min-height:0}.ahead-center nav{display:flex;flex-wrap:wrap;gap:2px}.ahead-center p{margin:0;font-size:11px}.ahead-track{display:grid;grid-template-columns:110px minmax(0,1fr);gap:6px;align-items:center;min-height:0}.ahead-track>span{font:10px sans-serif;color:#8eb9c9}.ahead-track-image{position:relative;height:100%;min-width:0}.ahead-room .ahead-strip,.ahead-track canvas{width:100%!important;height:100%!important}.ahead-cursor{position:absolute;top:0;bottom:0;width:2px;background:white;pointer-events:none;display:none;filter:drop-shadow(0 0 2px black)}.ahead-cursor:before{content:'';position:absolute;top:0;left:-4px;border-left:5px solid transparent;border-right:5px solid transparent;border-top:7px solid white}.ahead-room.ahead-focus{grid-template-columns:minmax(0,1fr)}.ahead-focus aside{display:none}.ahead-focus .ahead-center{grid-column:1}.ahead-room input{width:100%;box-sizing:border-box}@media(max-height:720px){.ahead-center{grid-template-rows:minmax(40px,1fr) auto 12px 28px repeat(3,45px)}.ahead-room aside{gap:4px;padding:6px}.ahead-room aside p{font-size:10px}}@media(max-width:1000px){.ahead-room{grid-template-columns:150px minmax(0,1fr) 175px}.ahead-header strong{font-size:11px}.ahead-track{grid-template-columns:70px minmax(0,1fr)}}`;
        status.textContent='CUT is the default for new joins. Saved edits are kept. Export is manual.';refresh();
    }catch(e){status.textContent=e.message;}
}

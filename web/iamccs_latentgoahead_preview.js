import { bindAheadControls } from './iamccs_ahead_controls.js';
import { app } from '/scripts/app.js';
import { api } from '/scripts/api.js';
import { openAheadEditor } from './iamccs_ahead_control_room.js';
import { mountAheadResult } from './iamccs_ahead_result_preview.js';
import { findRuntimeNode } from './iamccs_graph_nodes.js';

function panel(node) {
    bindAheadControls(node);
    for(const name of ['cut_plan','join_blend','blend_frames','audio_join','audio_smoothing_ms','flash_guard','flash_guard_sensitivity','flash_guard_radius']){
        const internal=node.widgets?.find(w=>w.name===name);
        if(internal){internal.type='hidden';internal.hidden=true;internal.computeSize=()=>[0,0];internal.draw=()=>{};if(internal.inputEl)internal.inputEl.style.display='none';}
    }
    const blend=node.widgets?.find(w=>w.name==='join_blend');
    if(blend && !['none','linear','smoothstep'].includes(blend.value))blend.value='smoothstep';
    const count=node.widgets?.find(w=>w.name==='blend_frames');if(count && !Number.isFinite(Number(count.value)))count.value=9;
    if(count && Number(count.value)<1)count.value=9;
    const audio=node.widgets?.find(w=>w.name==='audio_join');
    if(audio && !['click_safe','hard_cut'].includes(audio.value))audio.value='click_safe';
    const smoothing=node.widgets?.find(w=>w.name==='audio_smoothing_ms');
    if(smoothing && (String(smoothing.value??'').trim()===''||!Number.isFinite(Number(smoothing.value))||Number(smoothing.value)<0))smoothing.value='20';
    if(node._lgaPreview)return node._lgaPreview;
    const root=document.createElement('div');
    root.style.cssText='box-sizing:border-box;width:100%;height:auto;overflow:visible;background:#101c20;color:#dceeed;padding:8px;display:flex;flex-direction:column;gap:8px';
    const title=document.createElement('div');title.textContent='LATENTGOAHEAD · INTERVALS / FINAL';root.append(title);
    const editor=document.createElement('button');editor.textContent='OPEN EDITOR · TRIM / SEAMS / DELIVERY JOIN';editor.onclick=()=>openAheadEditor(node);root.append(editor);
    const select=document.createElement('select');select.style.cssText='width:100%;min-width:0;background:#20343a;color:white;padding:6px';root.append(select);
    const video=document.createElement('video');video.controls=true;video.preload='metadata';video.style.cssText='display:block;width:100%;height:170px;min-height:0;flex-shrink:0;object-fit:contain;background:black';root.append(video);
    const link=document.createElement('a');link.textContent='Open selected video';link.target='_blank';link.style.color='#8ed9ca';root.append(link);
    const items=()=>node.properties?.iamccs_lga_previews || [];
    function show(){const item=items()[Number(select.value)];if(!item)return;const url=api.apiURL('/view?'+new URLSearchParams({filename:item.filename,subfolder:item.subfolder,type:'output'}));video.src=url;link.href=url;}
    function refresh(){select.replaceChildren();items().forEach((item,i)=>{const option=document.createElement('option');option.value=i;option.textContent=item.label;select.append(option);});select.value=String(Math.max(0,items().length-1));show();}
    select.onchange=show;
    node.addDOMWidget('latentgoahead_previews','custom',root,{serialize:false,hideOnZoom:false,getMinHeight:()=>320,getMaxHeight:()=>360});
    node._lgaPreview={refresh};mountAheadResult(node);refresh();node.setSize?.(node.computeSize());return node._lgaPreview;
}
app.registerExtension({name:'IAMCCS.LatentGoAhead.Previews',
    nodeCreated(node){if(String(node.comfyClass).startsWith('IAMCCS_MiniMaxH3LatentGoAhead'))panel(node);},
    loadedGraphNode(node){if(String(node.comfyClass).startsWith('IAMCCS_MiniMaxH3LatentGoAhead')){panel(node).refresh();mountAheadResult(node).refresh();}}
});
api.addEventListener('iamccs-latentgoahead-video',event=>{
    const item=event.detail;const node=findRuntimeNode(item.node,n=>String(n.comfyClass||n.type).startsWith('IAMCCS_MiniMaxH3LatentGoAhead'));if(!node)return;
    node.properties ||= {};
    if(node.properties.iamccs_lga_run!==item.run){node.properties.iamccs_lga_run=item.run;node.properties.iamccs_lga_previews=[];}
    const records=node.properties.iamccs_lga_previews ||= [];
    if(!records.some(r=>r.filename===item.filename))records.push(item);
    panel(node).refresh();node.setDirtyCanvas?.(true,true);
});

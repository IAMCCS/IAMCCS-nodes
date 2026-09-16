import { api } from '/scripts/api.js';
import { app } from '/scripts/app.js';
import { allGraphNodes } from './iamccs_graph_nodes.js';

export function mountAheadResult(node) {
    if(node._aheadResult)return node._aheadResult;
    const root=document.createElement('div');
    root.style.cssText='box-sizing:border-box;width:100%;height:100%;padding:8px;background:#18202c;border:1px solid #9471bc;color:#eee;display:flex;flex-direction:column;gap:6px;overflow:hidden';
    const title=document.createElement('strong');title.textContent='VIDEO REFACTOR AHEAD';title.style.cssText='font-size:12px;color:#d7b7ff';
    const video=document.createElement('video');video.controls=true;video.preload='metadata';video.style.cssText='width:100%;flex:1;min-height:0;object-fit:contain;background:#000';
    const link=document.createElement('a');link.target='_blank';link.style.cssText='color:#cba3ff;font-size:11px';
    const state=document.createElement('small');state.style.color='#eeb7b7';
    video.onerror=()=>{video.hidden=true;state.textContent='Preview unavailable in this panel; use the output link below.';};
    video.onloadeddata=()=>{state.textContent='';video.hidden=false;};
    root.append(title,video,state,link);
    function refresh(){
        const item=node.properties?.iamccs_ahead_last_export;
        if(!item?.filename){video.hidden=true;link.removeAttribute('href');link.textContent='Apply & Export in Ahead to show the treated video here.';return;}
        const current=node.properties?.iamccs_lga_run;
        const matches=!current||item.subfolder?.endsWith('/'+current);
        const url=api.apiURL('/view?'+new URLSearchParams({filename:item.filename,subfolder:item.subfolder,type:'output'}));
        video.hidden=false;if(video.getAttribute('src')!==url)video.src=url;
        link.href=url;link.textContent=matches?'Open VIDEO REFACTOR AHEAD':'Previous run · open last Ahead export';
    }
    node.addDOMWidget('iamccs_ahead_result','custom',root,{serialize:false,hideOnZoom:false,getMinHeight:()=>220,getMaxHeight:()=>260});
    node._aheadResult={refresh};refresh();return node._aheadResult;
}

export function publishAheadResult(source,data){
    const run=source.properties?.iamccs_lga_run;
    const targets=new Set([source,...allGraphNodes().filter(n=>
        ['IAMCCS_AheadControlRoom','IAMCCS_MiniMaxH3LatentGoAhead','IAMCCS_MiniMaxH3LatentGoAheadBranch'].includes(n.comfyClass||n.type)&&n.properties?.iamccs_lga_run===run)]);
    for(const node of targets){node.properties ||= {};node.properties.iamccs_ahead_last_export=data;mountAheadResult(node).refresh();node.setDirtyCanvas?.(true,true);}
}

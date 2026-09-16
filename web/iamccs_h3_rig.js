// Explicit user-triggered media scaffolding. Never replaces a connected socket.
export function rigMedia(settings, mode, connected) {
    const graph=settings.graph, LG=globalThis.LiteGraph;
    if(!graph||!LG?.createNode)return 'RIG unavailable: graph API not ready.';
    const types=n=>n.comfyClass||n.type;
    const pool=connected.filter(n=>n.mode!==2&&n.mode!==4);
    const select=(type,input)=>pool.filter(n=>(!type||types(n)===type)&&n.inputs?.some(i=>i.name===input));
    let targets=[];
    if(mode==='v2va_controlnet')targets=[['IAMCCS_CineH3FunControlInput','source_video','VHS_LoadVideo','IMAGE']];
    else if(mode==='v2va_face_swap')targets=[['IAMCCS_H3FaceSwapInput','source_video','VHS_LoadVideo','IMAGE'],['IAMCCS_H3FaceSwapInput','reference_face','LoadImage','IMAGE'],['IAMCCS_H3FaceSwapInput','reference_face_2','LoadImage','IMAGE']];
    else if(mode==='ref2va'||mode==='ref2vid_lipsync')targets=[['IAMCCS_CineInfoH3','reference_image_1','LoadImage','IMAGE']];
    else if(mode==='v2va_object_swap')targets=[['IAMCCS_CineInfoH3','reference_video','VHS_LoadVideo','IMAGE']];
    else return 'RIG: this mode uses Shotboard slots or text. No loaders added.';
    const tasks=[];
    for(const [type,input,loader,out]of targets){const candidates=select(type,input);if(candidates.length!==1)return `RIG stopped: expected one connected ${type}. Found ${candidates.length}. Add the correct branch first; no wiring changed.`;const target=candidates[0],index=target.inputs.findIndex(i=>i.name===input);if(target.inputs[index].link!=null)continue;
        if(!LG.registered_node_types?.[loader])return `RIG stopped: ${loader} is not installed.`;
        tasks.push({target,index,loader,out});}
    let count=0;graph.beforeChange?.();
    try{for(const task of tasks){const loader=LG.createNode(task.loader);if(!loader)throw Error('Cannot create '+task.loader);const output=loader.outputs?.findIndex(o=>o.type===task.out);if(output<0)throw Error('Loader has no IMAGE output');loader.title='RIG · SELECT MEDIA · '+task.target.inputs[task.index].name;loader.pos=[task.target.pos[0]-380,task.target.pos[1]+count*240];graph.add(loader);loader.connect(output,task.target,task.index);count++;}
        if(mode==='v2va_controlnet'){
            const control=select('IAMCCS_CineH3FunControlInput','source_video')[0];
            const pre=control.widgets?.find(w=>w.name==='preprocessor');if(count&&pre?.value==='already_preprocessed'){pre.value='from_iamccs_settings';pre.callback?.(pre.value);}
            const output=control.outputs?.findIndex(o=>o.name==='control_preview');
            if(output>=0&&!control.outputs[output].links?.length&&LG.registered_node_types?.PreviewImage){const preview=LG.createNode('PreviewImage');preview.title='RIG · ACTUAL CONTROLNET PREVIEW';preview.pos=[control.pos[0]+420,control.pos[1]+200];graph.add(preview);control.connect(output,preview,0);}
        }
    }catch(e){return 'RIG partial setup: '+e.message+' — inspect new nodes before queueing.';}finally{graph.afterChange?.();graph.setDirtyCanvas?.(true,true);}
    return `RIG ready: ${count} missing media loader(s) added. Select your files, confirm source FPS and model settings before queueing. Existing connections preserved.`;
}

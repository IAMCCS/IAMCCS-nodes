import { app } from '/scripts/app.js';
import { openAheadEditor } from './iamccs_ahead_control_room.js';
import { mountAheadResult } from './iamccs_ahead_result_preview.js';

app.registerExtension({name:'IAMCCS.Ahead.ControlRoom',
    beforeRegisterNodeDef(type,data){
        if(data.name!=='IAMCCS_AheadControlRoom')return;
        const created=type.prototype.onNodeCreated;
        type.prototype.onNodeCreated=function(){
            created?.apply(this,arguments);
            this.addWidget('button','OPEN AHEAD · SEAM EDITOR',null,()=>openAheadEditor(this));
            this.addWidget('button','HELP · how to connect',null,()=>alert('Connect LatentGoAhead report → generation_report. Run the workflow normally. After completion, open this editor; choose JOIN, compare RAW/TREATED, then export manually. No automatic blending or resampling. Saved runs remain reviewable without queueing again.'));
            mountAheadResult(this);
            this.size=[380,380];
        };
        const executed=type.prototype.onExecuted;
        type.prototype.onExecuted=function(message){
            executed?.apply(this,arguments);
            const run=message?.iamccs_ahead_run?.[0];
            if(run){this.properties ||= {};this.properties.iamccs_lga_run=run;mountAheadResult(this).refresh();this.setDirtyCanvas?.(true,true);}
        };
    },
    loadedGraphNode(node){if((node.comfyClass||node.type)==='IAMCCS_AheadControlRoom')mountAheadResult(node).refresh();}
});

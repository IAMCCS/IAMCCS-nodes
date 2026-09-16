"""Queue-time workflow sidecars; never consulted as generation inputs."""
import copy
import json
import logging
import platform
from datetime import datetime, timezone

def clean(value):
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    if isinstance(value,dict):
        return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):
        return [clean(v) for v in value]
    return {'omitted_runtime_type':type(value).__name__}

def snapshot(prompt, extra, plan, node_id, controls):
    extra=extra or {}
    workflow = extra.get('workflow') or {}
    scopes = [workflow] + list((workflow.get('definitions') or {}).get('subgraphs', []))
    ui_controls = [{
        'scope':scope.get('id','root'), 'node_id':node.get('id'), 'type':node.get('type'),
        'widgets_values':node.get('widgets_values'),
        'widgets_values_named':node.get('widgets_values_named'),
        'properties':node.get('properties',{}),
    } for scope in scopes for node in scope.get('nodes', [])]
    return {'workflow':copy.deepcopy(extra.get('workflow')), 'metadata':clean({
        'schema':1,'captured_utc':datetime.now(timezone.utc).isoformat(),
        'node_id':node_id,'python':platform.python_version(),'api_prompt':prompt,
        'extra_pnginfo':{k:v for k,v in extra.items() if k!='workflow'},
        'resolved_shotplan':plan,'continuation_controls':controls,
        'all_ui_node_settings':ui_controls, 'effective_sampling':plan.get('sampling',{}),
        'controlroom':{'generation':controls,'post_seams':[v for v in ui_controls if any(str(k).startswith('iamccs_ahead') for k in v['properties'])]},
        'limitations':'External media, models and custom node code are not embedded. Seeds and queued values are recorded; bit-identical regeneration is not guaranteed.'})}

def write_sidecar(video, data, label, run):
    try:
        workflow=copy.deepcopy(data['workflow'])
        metadata=copy.deepcopy(data['metadata'])
        metadata.update(video=video.name,label=label,run=run)
        if isinstance(workflow,dict) and isinstance(workflow.get('nodes'),list):
            workflow.setdefault('extra',{})['iamccs_generation_provenance']=metadata
            target=video.with_suffix('.workflow.json')
            target.write_text(json.dumps(workflow,ensure_ascii=False,indent=2),encoding='utf-8')
        else:
            video.with_suffix('.metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding='utf-8')
            logging.warning('LatentGoAhead sidecar: queue did not include UI workflow; saved API metadata only for %s',video.name)
    except Exception:
        logging.exception('LatentGoAhead sidecar write failed for %s; generated video retained',video)

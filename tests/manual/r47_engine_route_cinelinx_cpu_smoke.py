from pathlib import Path
import types, sys, importlib.util, inspect, json, ast
root=Path('/mnt/data/r47_engine_fix/IAMCCS-nodes')
pkg=types.ModuleType('iamccs_r47_testpkg');pkg.__path__=[str(root)];sys.modules[pkg.__name__]=pkg
folder=types.ModuleType('folder_paths');folder.get_filename_list=lambda kind:[];folder.get_full_path=lambda *x:None;folder.get_output_directory=lambda:'/tmp';folder.get_input_directory=lambda:'/tmp';sys.modules['folder_paths']=folder

def stub(modname, attrs):
 m=types.ModuleType(f'{pkg.__name__}.{modname}')
 for k,v in attrs.items():setattr(m,k,v)
 sys.modules[m.__name__]=m

stub('iamccs_h3_advisor', dict(describe_asset=lambda *a,**kw:{}, recipes=lambda *a,**kw:{},validate_settings=lambda *a,**kw:None))
stub('iamccs_h3_face_swap',dict(settings_schema=lambda:{},face_swap_settings=lambda *a,**kw:{},validate_plan=lambda *a,**kw:None))
stub('iamccs_prompter', dict(SUPERNODE_LINX_TYPE='IAMCCS_LINX',apply_prompter_to_minimax=lambda linx,g,t:(g,t,{'applied':False})))
stub('audio.dialogue_tag_editor', dict(apply_dialogue_to_minimax=lambda *a,**kw:{}))
stub('iamccs_supernodes_linx', dict(build_stage_linx_payload=lambda *a,**kw:{'resources':{},'outputs':{},'payload':{}}))
# All bottom-of-file backend registrations are import-only; fake them for a
# CPU planner wiring test. Keep the actual core and restore implementation real.
source=(root/'iamccs_minimax_h3_shotboard.py').read_text()
for node in ast.walk(ast.parse(source)):
 if not isinstance(node,ast.ImportFrom) or node.level!=1 or not node.module:
  continue
 if node.module in {'iamccs_minimax_h3_shotboard_core','iamccs_h3_advisor','iamccs_h3_face_swap','iamccs_prompter','iamccs_supernodes_linx','audio.dialogue_tag_editor','iamccs_minimax_h3_timeline_director_restore'}:
  continue
 name=f'{pkg.__name__}.{node.module}'
 if name not in sys.modules:
  stub(node.module,{})
 for x in node.names:
  if not hasattr(sys.modules[name], x.name):
   setattr(sys.modules[name],x.name,{} if x.name in ('NODE_CLASS_MAPPINGS','NODE_DISPLAY_NAME_MAPPINGS') else (lambda *a,**kw:None))
spec=importlib.util.spec_from_file_location(f'{pkg.__name__}.iamccs_minimax_h3_shotboard',root/'iamccs_minimax_h3_shotboard.py')
module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
print('IMPORT_OK',True)
# Validates live Planner integration without CUDA or ComfyUI dependencies.
rows=[{'id':'a','type':'image','start':0,'length':300,'imageFile':'a.png','use_guide':True,'prompt':'walking'}, {'id':'b','type':'image','start':300,'length':420,'imageFile':'b.png','use_guide':True,'prompt':'turning'}]
module._build_minimax_cine_linx=lambda linx,plan,timeline,prompt,report:{'resources':{'iamccs_minimax_h3_shotplan':plan}}
for eng in ['pianosequenza_1_native_tail','extended_av_continuous','pianosequenza_evolving','pianosequenza_2_drift','pianosequenza_3_phase']:
 t={'fps':24,'duration_seconds':30.0,'rows':rows,'h3_saved_settings':{'longvid_continuous_engine':'pianosequenza_1_native_tail','flf_continuity_tail_frames':'22','extended_av_profile':'safe_8_12gb'}}
 if eng=='pianosequenza_evolving': t['extended_prompt_blocks']=[{'start_frame':0,'end_frame':720,'prompt':'sustain movement'}]
 try:
  plan=module.IAMCCS_MiniMaxH3ShotPlanner().plan(global_prompt='one take',timeline_data=json.dumps(t),duration_seconds=30,task_mode='longvid_keyframe_continuous',audio_mode='h3_native_generated',cine_linx={'resources':{'iamccs_minimax_h3_settings':{'schema':'iamccs.minimax_h3.settings_cine_linx','settings':{'longvid_continuous_engine':eng,'flf_continuity_tail_frames':'39','extended_av_profile':'safe_8_12gb'}}}},prompt_mapping='global_plus_local',upscale_mode='off',width=960,height=544)[0]['resources']['iamccs_minimax_h3_shotplan']
  assert plan.get('requested_task_mode') == 'longvid_keyframe_continuous', (eng, plan.get('requested_task_mode'))
  assert plan.get('longvid_engine') == eng, (eng, plan.get('longvid_engine'))
  if eng == 'extended_av_continuous':
   assert plan['task_mode'] == 'fl2va_extended_av' and plan['extended_av']['overlap_frames'] == 39
   assert plan['chunks'][0]['task_mode'] == 'i2va' and plan['chunks'][1]['extended_av_trim_head_frames'] == 39
   assert plan['mode_contract']['extended_av_trimmed_prefix'] is True
  else:
   assert plan['task_mode'] == 'longvid_guides' and plan['mode_contract']['overlap_frames'] == 39
   assert plan['mode_contract']['sampled_native_video_audio_tail'] is True
   if eng == 'pianosequenza_evolving':
    assert plan['conditioning_schedule']['beat_count'] == 1
    assert '[EVOLVING TIMED BEATS]' in plan['chunks'][0]['prompt']
   if eng == 'pianosequenza_2_drift':
    assert plan['mode_contract']['terminal_endpoint_mode'] == 'pianosequenza_drift'
   if eng == 'pianosequenza_3_phase':
    assert plan['mode_contract']['terminal_endpoint_mode'] == 'pianosequenza_phase'
  print('ROUTE',eng,'plan_task',plan['task_mode'],'backend',plan.get('backend_variant'),'overlap',plan.get('mode_contract',{}).get('overlap_frames'), 'total',len(plan['chunks']), 'evolving',plan.get('conditioning_schedule'))
 except Exception as e:
  import traceback; print('ERROR',eng,e);traceback.print_exc()

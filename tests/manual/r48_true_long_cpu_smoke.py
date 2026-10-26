from pathlib import Path

fixture = Path(__file__).with_name('r47_engine_route_cinelinx_cpu_smoke.py')
source = fixture.read_text(encoding='utf-8').split("for eng in [")[0]
source = source.replace("root=Path('/mnt/data/r47_engine_fix/IAMCCS-nodes')", "root=Path(__file__).resolve().parents[2]")
exec(compile(source, str(fixture), 'exec'))
timeline = {'fps':24, 'duration_seconds':10.0, 'rows':[
    {'id':'a','type':'image','start':0,'length':5,'imageFile':'a.png','use_guide':True},
    {'id':'b','type':'image','start':5,'length':5,'imageFile':'b.png','use_guide':True},
    {'id':'c','type':'image','start':10,'length':1,'imageFile':'c.png','use_guide':True},
]}
for public_mode in ('pianosequenza_true_long', 'longvid_keyframe_continuous'):
    saved_timeline = dict(timeline)
    saved_timeline['h3_saved_settings'] = {
        'longvid_continuous_engine': 'extended_av_continuous',
    }
    plan = module.IAMCCS_MiniMaxH3ShotPlanner().plan(
        global_prompt='One uninterrupted camera move.', timeline_data=json.dumps(saved_timeline),
        duration_seconds=10, task_mode=public_mode, audio_mode='h3_native_generated',
        prompt_mapping='global_plus_local', upscale_mode='off', width=640, height=384,
    )[0]['resources']['iamccs_minimax_h3_shotplan']
    assert plan['longvid_engine'] == 'pianosequenza_true_long'
    assert plan['requested_task_mode'] == public_mode
    assert plan['task_mode'] == 'longvid_keyframe_continuous'
    assert plan['iamccs_continuous_av']['segment_checkpoints']
    assert plan['mode_contract']['selected_engine'] == 'pianosequenza_true_long'
    assert plan['mode_contract']['engine_selector_ignored']
    assert len(plan['chunks']) >= 2
    assert not plan.get('timeline_director_native')
print('PASS: Settings/Planner -> continuous AV -> PSkf True_long', len(plan['chunks']))

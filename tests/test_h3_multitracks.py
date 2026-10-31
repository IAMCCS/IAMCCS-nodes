import importlib.util,sys,types,unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
pkg=types.ModuleType("multitracks_tests");pkg.__path__=[str(ROOT)];sys.modules[pkg.__name__]=pkg

def load(name):
 spec=importlib.util.spec_from_file_location(pkg.__name__+'.'+name,ROOT/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
A=load('iamccs_minimax_h3_audio_timeline');M=load('iamccs_h3_multitracks')
stub=types.ModuleType(pkg.__name__+'.iamccs_minimax_h3_atomic_backend');stub._resolve_shotplan=A._resolve_shotplan;sys.modules[stub.__name__]=stub

def fixture(task):
 frames=124;duration=frames/24
 plan={'schema':'iamccs.minimax_h3.shotplan','task_mode':task,'audio_mode':'h3_custom_audio_drive','fps':24,'duration_seconds':duration,'requested_duration_seconds':5,'chunks':[{'index':0,'frame_count':frames,'duration_seconds':duration,'timeline_start_seconds':0,'timeline_start_frame':0,'unique_frames':frames,'slot_id':'shot_1'}]}
 rows=[{'id':str(i),'audio_input':i+1,'track':i,'start_seconds':0,'duration_seconds':duration,'gain':1,'pan':0} for i in range(3)]
 audios=[{'waveform':torch.full((1,2,round(duration*32000)),.1*(i+1)),'sample_rate':32000} for i in range(3)]
 return {'resources':{'iamccs_minimax_h3_shotplan':plan,'cine_audio_tracks':{'segments':rows,'shotboard_segments':rows},'iamccs_cine_h3_audio_bus_audio_lanes':audios}}
class Tests(unittest.TestCase):
 def test_three_tracks_retain_isolation_and_master(self):
  for task in ['t2va','fl2va','ref2va']:
   with self.subTest(task=task):
    bus=fixture(task);original=A._resolve_shotplan(bus).copy();out,*result=M.IAMCCS_H3MultitrackAudioRouter().route(bus)
    tracks=result[:3];master=result[3]
    for i,a in enumerate(tracks):self.assertAlmostEqual(a['waveform'].mean().item(),.1*(i+1),places=4)
    self.assertAlmostEqual(master['waveform'].mean().item(),.6,places=4)
    self.assertEqual(A._resolve_shotplan(bus),original)
    self.assertEqual(out['resources']['iamccs_h3_multitracks']['family'],'ref2va' if task=='ref2va' else 'fl2va')
    if task=='ref2va':self.assertEqual(sum(t['waveform'].shape[-1] for t in tracks),15*32000)
 def test_missing_track_fails_explicitly(self):
  bus=fixture('t2va');bus['resources']['cine_audio_tracks']['shotboard_segments']=bus['resources']['cine_audio_tracks']['segments'][:2]
  with self.assertRaisesRegex(ValueError,'Track 3'):M.IAMCCS_H3MultitrackAudioRouter().route(bus)
 def test_exact_transcript_required_and_retained(self):
  with self.assertRaisesRegex(ValueError,'transcript'):M.prompt_context({'enabled':True,'lipsync':True},'fl2va')
  words='[Italian] Seguimi nel bosco.'
  text=M.prompt_context({'enabled':True,'lipsync':True,'transcript':words},'fl2va')
  self.assertIn(words,text);self.assertIn('one locked custom audio',text)
  ref=M.prompt_context({'enabled':True},'ref2va');self.assertIn('Track 3 = <Audio 3>',ref)
 def test_disabled_preset_is_noop(self):self.assertEqual(M.prompt_context({},'t2va'),'')
 def test_clip_solo_also_mutes_other_reference_stems(self):
  bus=fixture('ref2va');bus['resources']['cine_audio_tracks']['segments'][0]['solo']=True
  _,one,two,three,master,_=M.IAMCCS_H3MultitrackAudioRouter().route(bus)
  self.assertAlmostEqual(one['waveform'].mean().item(),.1,places=4)
  self.assertEqual(two['waveform'].abs().max().item(),0)
  self.assertEqual(three['waveform'].abs().max().item(),0)
  self.assertAlmostEqual(master['waveform'].mean().item(),.1,places=4)
if __name__=='__main__':unittest.main()

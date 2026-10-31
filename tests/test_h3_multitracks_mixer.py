import importlib.util,sys,types,unittest,json
from pathlib import Path
ns={'__name__':'fixture_module','__file__':str(Path(__file__).parent/'test_h3_multitracks.py')}
exec(compile((Path(__file__).parent/'test_h3_multitracks.py').read_text(),'<fixtures>','exec'),ns)
A=ns['A'];ROOT=ns['ROOT'];fixture=ns['fixture'];torch=ns['torch'];pkg=ns['pkg']

audio=types.ModuleType(pkg.__name__+'.audio');audio.__path__=[str(ROOT/'audio')];sys.modules[audio.__name__]=audio
def load_audio(name):
 spec=importlib.util.spec_from_file_location(audio.__name__+'.'+name,ROOT/'audio'/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m);return m
B=load_audio('audio_board_arranger');X=load_audio('audio_board_mixer')
def mixed(bus,settings,master=1):
 out,_,_=X.IAMCCS_AudioBoardMixer().mix(json.dumps({'audioTrackCount':3,'trackSettings':settings,'masterAudioGain':master}),'mixer_overrides_cine_linx',bus)
 return A.mix_audio_timeline(out,0,[],mapping_policy='absolute_timeline',coverage_policy='allow_silence',headroom='none')[1]['waveform']
class MixerTests(unittest.TestCase):
 def test_fader_is_recomputed_from_clip_gain_once(self):
  bus=fixture('t2va');t=bus['resources']['cine_audio_tracks'];raw=t['segments'];baked=B.IAMCCS_AudioBoardArranger._segments_with_track_mix(raw,[{'volume':.5}]*3,3);t.update(segments=baked,all_segments=baked,shotboard_segments=baked)
  w=mixed(bus,[{'volume':.25}]*3);self.assertAlmostEqual(w.mean().item(),.15,places=4)
 def test_master_can_reach_exact_silence(self):self.assertEqual(mixed(fixture('t2va'),[{}]*3,0).abs().max().item(),0)
 def test_unmute_restores_original_clip(self):
  bus=fixture('t2va');t=bus['resources']['cine_audio_tracks'];baked=B.IAMCCS_AudioBoardArranger._segments_with_track_mix(t['segments'],[{'mute':True}]*3,3);t.update(segments=baked,all_segments=baked,shotboard_segments=baked)
  self.assertAlmostEqual(mixed(bus,[{'mute':False}]*3).mean().item(),.6,places=4)
 def test_solo_has_one_authority(self):self.assertAlmostEqual(mixed(fixture('t2va'),[{}, {'solo':True},{}]).mean().item(),.2,places=4)
 def test_pan_reaches_the_rendered_channels(self):
  w=mixed(fixture('t2va'),[{'pan':1}]*3)
  self.assertAlmostEqual(w[:,0].abs().max().item(),0,places=7)
  self.assertAlmostEqual(w[:,1].mean().item(),.6 * 2**.5,places=4)
 def test_clip_mute_survives_track_unmute(self):
  bus=fixture('t2va');t=bus['resources']['cine_audio_tracks'];t['segments'][0]['mute']=True
  baked=B.IAMCCS_AudioBoardArranger._segments_with_track_mix(t['segments'],[{'mute':True}]*3,3)
  t.update(segments=baked,all_segments=baked,shotboard_segments=baked)
  self.assertAlmostEqual(mixed(bus,[{'mute':False}]*3).mean().item(),.5,places=4)
 def test_source_trim_is_preserved(self):
  bus=fixture('t2va');source=bus['resources']['iamccs_cine_h3_audio_bus_audio_lanes'][0]['waveform']
  source[:]=torch.linspace(0,.1,source.shape[-1]);bus['resources']['cine_audio_tracks']['segments'][0]['trimStart']=24
  w=mixed(bus,[{'solo':True},{},{}])
  self.assertAlmostEqual(w[0,0,0].item(),source[0,0,32000].item(),places=5)
 def test_read_only_monitor_preserves_effective_mix(self):
  bus=fixture('t2va')
  out,_,_=X.IAMCCS_AudioBoardMixer().mix(json.dumps({'trackSettings':[{'volume':0}]*3}),'read_only_monitor',bus)
  w=A.mix_audio_timeline(out,0,[],mapping_policy='absolute_timeline',coverage_policy='allow_silence',headroom='none')[1]['waveform']
  self.assertAlmostEqual(w.mean().item(),.6,places=4)
 def test_fade_envelope_continues_across_chunk_cut(self):
  full=A._fade(torch.ones(1,2,100),40,40)
  left=A._fade(torch.ones(1,2,50),40,40,offset=0,total_length=100)
  right=A._fade(torch.ones(1,2,50),40,40,offset=50,total_length=100)
  self.assertTrue(torch.allclose(torch.cat([left,right],-1),full))
if __name__=='__main__':unittest.main()

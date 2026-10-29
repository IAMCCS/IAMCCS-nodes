import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_prompter_native_qwen import prompter


class OfficialTagAuthoringTests(unittest.TestCase):
    def test_global_request_creates_speech_tags_even_when_input_has_none(self):
        system, user = prompter._assistant_instruction(
            't2va', {}, 'Compile Global', ['scene', 'dialogue'],
            request_text='Una donna dice: Aspettami qui.',
        )
        self.assertIn('even when the rough request has no tags', system)
        self.assertIn('<d>[Language] exact words</d>', system)
        self.assertIn('independently of subject numbering', system)
        self.assertIn('<scenetrans>', system)
        self.assertIn('<cutoff>', system)
        self.assertIn('Global describes output scope', system)
        self.assertEqual(json.loads(user)['request'], 'Una donna dice: Aspettami qui.')

    def test_reference_grammar_distinguishes_assets_content_and_speakers(self):
        keys=[key for key,_ in prompter.MODE_SECTIONS['ref2va']]
        system,_=prompter._assistant_instruction('ref2va',{},'',keys,request_text='The second character speaks.')
        for tag in ['<Subject N>', '<Picture N>', '<Video N>', '<Audio N>']:
            self.assertIn(tag,system)
        self.assertIn('Do not populate unused slots',system)
        self.assertIn('full-reference modes keep their six sections',system)

    def test_compact_retry_retains_tag_rules(self):
        system,_=prompter._compact_assistant_instruction('i2va',{},'She says hello.',['dialogue'])
        self.assertIn('<d>[Language] exact words</d>',system)

    def test_specialized_edit_grammars_are_not_overwritten(self):
        for mode in ['v2va_face_swap_lora','ref2va_vfx_edit']:
            self.assertIn('native aligned-edit grammar',prompter._official_h3_tag_rules(mode))
            self.assertNotIn('Format every supplied',prompter._official_h3_tag_rules(mode))

    def test_reference_output_preserves_independent_subject_and_speaker_ids(self):
        line='<Subject 2> (S1) says <d>[Italian] Aspettami qui.</d>'
        payload={'subject_definitions':'<Subject 2> is the woman in <Picture 1>.','detailed_description':'[Shot 1] '+line}
        reply={'choices':[{'message':{'content':json.dumps(payload)}}]}
        with patch.object(prompter,'_http_json',return_value=reply):
            fields,_=prompter.rewrite_sections_with_ai('lm_studio','','local','','ref2va',{},target_keys=list(payload),request_text='Una donna dice: Aspettami qui.')
        self.assertIn(line,fields['detailed_description'])
        self.assertNotIn('<Subject 1>',fields['detailed_description'])

    def test_global_and_locals_endpoint_uses_the_same_tag_contract(self):
        payload={'global_prompt':'A quiet room.','shots':[{'slot':1,'local_prompt':'(S1) says <d>[Italian] Resta.</d>'}]}
        with patch.object(prompter,'_multimodal_json_chat',return_value=payload) as chat:
            plan,_=prompter.build_visual_story_plan_with_ai('lm_studio','','local','','Prompt 1: una donna dice Resta.','t2va',[])
        self.assertIn('even when the rough request has no tags',chat.call_args.args[4])
        self.assertIn('<d>[Italian] Resta.</d>',plan['shots'][0]['local_prompt'])

    def test_request_utterance_with_says_and_cutoff_survives_a_bad_rewrite(self):
        line='<Subject 2> (S1) says <d>[Italian] Non andare!</d><cutoff>'
        reply={'choices':[{'message':{'content':json.dumps({'dialogue':'The woman says go.'})}}]}
        with patch.object(prompter,'_http_json',return_value=reply):
            fields,_=prompter.rewrite_sections_with_ai('lm_studio','','local','','t2va',{},target_keys=['dialogue'],request_text=line)
        self.assertIn(line,fields['dialogue'])
        self.assertNotIn('The woman says go.',fields['dialogue'])

    def test_standalone_voice_and_cross_cut_markers_are_preserved(self):
        line='(S2) speaks off-screen <d>[English] Stay here.</d><scenetrans>'
        self.assertEqual(prompter._authored_h3_dialogue_lines(line),[line])

    def test_keyframe_global_does_not_absorb_timed_dialogue_from_request(self):
        line='(S1) says <d>[Italian] Resta.</d>'
        reply={'choices':[{'message':{'content':json.dumps({'reference_use':'Keep the same opening frame.'})}}]}
        with patch.object(prompter,'_http_json',return_value=reply):
            fields,_=prompter.rewrite_sections_with_ai('lm_studio','','local','','fl2va',{},target_keys=['reference_use'],request_text=line)
        self.assertNotIn('<d>',fields['reference_use'])


if __name__=='__main__':unittest.main()

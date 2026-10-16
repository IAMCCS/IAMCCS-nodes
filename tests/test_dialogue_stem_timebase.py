import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch


ROOT = Path(__file__).parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "audio" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


INFO = load("cine_audio_info")
EDITOR = load("dialogue_tag_editor")


class StemTimebaseTests(unittest.TestCase):
    def dialogue(self, local=False):
        return {"speakers": [{"id": "A"}, {"id": "B"}],
                "settings": {"audio_lane_mode": "speaker_stems_timeline", "speaker_stem_srt_local_zero": local},
                "speaker_stem_start_frames": {"A": 48, "B": 120},
                "lines": [{"speaker": "A", "start": 2, "end": 3, "text": "One."},
                          {"speaker": "B", "start": 5, "end": 6, "text": "Two."},
                          {"speaker": "A", "start": 8, "end": 9, "text": "Three."}]}

    def inject(self, dialogue):
        segments = [{"speaker": line["speaker"], "start": int(line["start"] * 24), "length": 24,
                     "track": 0 if line["speaker"] == "A" else 1} for line in dialogue["lines"]]
        linx = {"resources": {"dialogue_tag_editor": dialogue, "cine_audio_tracks": {"segments": segments}}}
        original = copy.deepcopy(linx)
        with patch.object(INFO, "_save_audio_to_input", side_effect=[("A.wav", 10., 48000, 480000),
                                                                    ("B.wav", 10., 48000, 480000)]):
            result = INFO.IAMCCS_CineAudioInfo().run("inject_speaker_stems", 24, "plain_dialogue",
                                                  "speaker_full_timeline_clips", "test", "test", cine_linx=linx,
                                                  generated_audio=object(), generated_audio_b=object())
        self.assertEqual(linx, original)
        return json.loads(result[3])["audioSegments"]

    def test_absolute_srt_wavs_start_at_zero_even_with_legacy_offsets(self):
        d = self.dialogue(False)
        self.assertEqual(INFO._dialogue_speaker_stem_start_frames(d, 24), {"A": 0, "B": 0})
        self.assertIn("00:00:05,000", INFO._export_speaker_srts(d, [], 24, "plain_dialogue")["B"])
        clips = self.inject(d)
        self.assertEqual([clip["start"] for clip in clips], [0, 0])
        self.assertEqual([clip["trimStart"] for clip in clips], [0, 0])
        self.assertEqual([clip["length"] for clip in clips], [240, 240])
        self.assertEqual([clip["audioFile"] for clip in clips], ["A.wav", "B.wav"])

    def test_local_srt_wavs_keep_authored_lane_offsets(self):
        d = self.dialogue(True)
        self.assertEqual(INFO._dialogue_speaker_stem_start_frames(d, 24), {"A": 48, "B": 120})
        self.assertIn("00:00:00,000", INFO._export_speaker_srts(d, [], 24, "plain_dialogue")["B"])
        self.assertEqual([clip["start"] for clip in self.inject(d)], [48, 120])

    def test_template_only_absolute_timebase_is_respected(self):
        d = self.dialogue()
        d["settings"] = {}
        d["audio_board_template"] = {"speakerStemSrtLocalZero": False, "speakerStemStartFrames": {"B": 120}}
        self.assertEqual(INFO._dialogue_speaker_stem_start_frames(d, 24), {"A": 0, "B": 0})

    def test_explicit_rebased_stems_still_start_at_zero(self):
        d = self.dialogue(True)
        d["settings"]["speaker_stems_zero_start"] = True
        self.assertEqual([clip["start"] for clip in self.inject(d)], [0, 0])

    def test_editor_exports_timebase_and_visual_positions_consistently(self):
        for local, expected in [(False, {"A": 0, "B": 0}), (True, {"A": 48, "B": 120})]:
            d = self.dialogue(local)
            export = EDITOR._build_dialogue_export(d, 24, 130, .2, 0, "speaker_stems_for_overlap", "metadata_only")
            self.assertEqual(export["speaker_stem_start_frames"], expected)
            self.assertEqual(export["audio_board_template"]["speakerStemStartFrames"], expected)
            self.assertEqual([s["start"] for s in export["shotboard_timeline"]["segments"]], [48, 120, 192])


if __name__ == "__main__":
    unittest.main()

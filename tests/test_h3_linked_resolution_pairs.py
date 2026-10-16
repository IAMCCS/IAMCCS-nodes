import ast
from pathlib import Path
import re
import textwrap
import unittest

ROOT = Path(__file__).parents[1]
SOURCE = (ROOT / 'iamccs_minimax_h3_shotboard.py').read_text(encoding='utf-8')
TREE = ast.parse(SOURCE)
PAIR_ASSIGNMENT = next(n for n in TREE.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'H3_LINKED_STAGE_PAIRS' for t in n.targets))
SCOPE = {}
exec(compile(ast.Module(body=[PAIR_ASSIGNMENT], type_ignores=[]), 'pairs', 'exec'), SCOPE)
PAIRS = SCOPE['H3_LINKED_STAGE_PAIRS']


class LinkedResolutionTests(unittest.TestCase):
    def test_ui_and_backend_have_identical_pairs(self):
        js = (ROOT / 'web/iamccs_h3_settings_pro_ui.js').read_text(encoding='utf-8')
        base = ast.literal_eval(re.search(r'const H3_FAST_2PASS_PAIRS = Object.freeze\((\[[\s\S]*?\])\.flatMap', js).group(1))
        expected = {tuple(pair) for pair in base}
        expected.update((h,w,hh,hw) for w,h,hw,hh in base)
        expected.add((768,768,1536,1536))
        self.assertEqual(set(PAIRS.values()), expected)
        self.assertEqual(len(PAIRS),49)

    def test_legal_dimensions_and_saved_legacy_pairs(self):
        for dims in PAIRS.values():
            self.assertTrue(all(256 <= n <= 5760 and n % 32 == 0 for n in dims))
            self.assertGreaterEqual(dims[2],dims[0])
            self.assertGreaterEqual(dims[3],dims[1])
        for key in ['640x384 -> 1280x768','768x448 -> 1536x896','896x512 -> 1792x1024','1024x576 -> 2048x1152','1280x736 -> 2560x1472']:
            self.assertIn(key,PAIRS)

    def test_new_portrait_pair_reaches_generation_plan_without_fallback(self):
        start=SOURCE.index('        linked_stage_pairs = H3_LINKED_STAGE_PAIRS')
        end=SOURCE.index('        width = _h3_legal_dimension',start)
        scope=dict(H3_LINKED_STAGE_PAIRS=PAIRS,upscale_link_factor='576x960 -> 1152x1920',
                   longvid_pianosequenza_2stage_enabled=True,longvid_guides_requested=True,upscale_link_to_native=True)
        exec(textwrap.dedent(SOURCE[start:end]),scope)
        self.assertEqual((scope['stage_low_width'],scope['stage_low_height']),(576,960))
        self.assertEqual((scope['width'],scope['height']),(1152,1920))


if __name__ == '__main__':
    unittest.main()

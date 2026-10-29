"""Retired Repair compatibility and read-only diagnostics; no inference."""
import importlib.util
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
spec = importlib.util.spec_from_file_location("guard_test_module", ROOT / "iamccs_h3_continuity_guard.py")
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


def test_legacy_repair_disables_all_interventions(tmp_path):
    config = guard.configuration({"h3_guard_mode": "repair", "h3_guard_reference_memory": True})
    assert config['mode'] == 'off'
    samples, frames = object(), object()
    result = guard.process({'mode': 'repair'}, samples, frames, directory=tmp_path,
                           index=1, head_frames=0, visible_frames=1)
    assert result[0] is samples and result[1] is frames
    assert result[2]['status'] == 'off'
    assert list(tmp_path.iterdir()) == []


def test_monitor_preserves_video_audio_mask_and_frames(tmp_path):
    from unittest.mock import patch
    samples = {'samples': object(), 'noise_mask': object(), 'metadata': {'seed': 42}}
    frames = torch.rand(8, 12, 12, 3)
    before = frames.clone()
    config = guard.configuration({'h3_guard_mode': 'monitor'})
    with patch.object(guard, 'measure', return_value={'detail': .1, 'luminance': .5, 'motion': .1}):
        for index in (0, 1):
            result, returned, report = guard.process(config, samples, frames,
                directory=tmp_path, index=index, head_frames=2, visible_frames=6)
            for key, value in samples.items():
                assert result[key] is value
            assert returned is frames and torch.equal(frames, before)
            assert report['repair_attempts'] == 0


def test_darkening_or_motion_change_requires_review_not_automatic_repair():
    config = guard.configuration({})
    baseline = {'detail': .1, 'luminance': .5, 'motion': .1}
    assert guard.assess({'detail': .04, 'luminance': .5, 'motion': .1}, baseline, config)['texture_loss_detected']
    for changed in ({'detail': .04, 'luminance': .2, 'motion': .1}, {'detail': .04, 'luminance': .5, 'motion': .005}):
        verdict = guard.assess(changed, baseline, config)
        assert verdict['review_recommended'] and not verdict['texture_loss_detected']


def test_metrics_detect_synthetic_blur():
    import torch.nn.functional as F
    torch.manual_seed(5)
    texture = torch.rand(1, 3, 144, 256)
    movie = torch.cat([torch.roll(texture, i, -1) for i in range(8)]).movedim(1, -1)
    blurred = F.avg_pool2d(movie.movedim(-1, 1), 7, stride=1, padding=3).movedim(1, -1)
    a, b = guard.measure(movie), guard.measure(blurred)
    assert b['detail'] < a['detail'] * .65


def test_storage_does_not_use_render_id_as_path(tmp_path):
    path = guard.storage(tmp_path, '../../outside')
    assert path.is_relative_to(tmp_path)


def test_invalid_configuration_is_rejected():
    import unittest
    with unittest.TestCase().assertRaises(ValueError):
        guard.configuration({'h3_guard_detail_ratio': float('nan')})


def test_blank_root_identity_binds_to_numbered_checkpoint_and_continuation(tmp_path):
    import json
    provisional = guard.conditioning_identity('', 0)
    source = guard.storage(tmp_path, provisional)
    baseline = {'detail': .1, 'motion': .1, 'luminance': .5}
    guard.write_report(source / 'baseline.json', baseline)
    report = {'storage_key': source.name, 'segment_index': 0}
    guard.write_report(source / 'segment_0000.json', report)
    guard.bind_checkpoint(tmp_path, report, 'soldier_0042')
    continuation_id = guard.conditioning_identity('soldier_0042', 1)
    destination = guard.storage(tmp_path, continuation_id)
    assert json.loads((destination / 'baseline.json').read_text()) == baseline
    guard.bind_checkpoint(tmp_path, {'storage_key': destination.name, 'segment_index': 1}, continuation_id)
    assert (source / 'baseline.json').is_file()
    assert guard.conditioning_identity('', 0) != provisional


def test_missing_continuation_identity_and_invalid_binding_key_fail_closed(tmp_path):
    import unittest
    with unittest.TestCase().assertRaises(ValueError):
        guard.conditioning_identity('', 1)
    with unittest.TestCase().assertRaises(ValueError):
        guard.bind_checkpoint(tmp_path, {'storage_key': '../../outside', 'segment_index': 0}, 'safe')


if __name__ == '__main__':
    import tempfile
    import inspect
    tests = [(name, value) for name, value in globals().copy().items() if name.startswith('test_')]
    for name, test in tests:
        with tempfile.TemporaryDirectory() as directory:
            test(Path(directory)) if inspect.signature(test).parameters else test()
        print('PASS', name)

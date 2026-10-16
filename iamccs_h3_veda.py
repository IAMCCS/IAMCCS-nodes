# SPDX-License-Identifier: GPL-3.0-or-later
"""Optional, explicit VeDA attention patch. No downloads or silent fallback."""
import logging

T2VA_PREDICTOR = 'minimax_h3_t2va_veda_8nfe_600step_preview_fp8.safetensors'
R2VA_PREDICTOR = 'minimax_h3_r2va_veda_preview_fp8.safetensors'


def veda_plan(settings):
    return {'enabled': bool(settings.get('h3_veda_enabled', False)),
            'predictor': str(settings.get('h3_veda_predictor', 'auto')),
            'verbose': bool(settings.get('h3_veda_verbose', False))}


def apply_veda(model, plan, task):
    options = plan.get('veda') or {}
    if not options.get('enabled'):
        return model
    # The resolved atomic task, not the top-level LongVid label, chooses weights.
    if task not in {'t2va', 'i2va', 'fl2va', 'ref2va'}:
        raise ValueError(f'VeDA is not validated for {task}. Disable VeDA for this task.')
    import nodes
    node = nodes.NODE_CLASS_MAPPINGS.get('VedaSparseAttention')
    if node is None:
        raise ValueError('Install Veda-on-ComfyUI and restart ComfyUI before enabling VeDA.')
    predictor = options.get('predictor', 'auto')
    if predictor == 'auto':
        predictor = R2VA_PREDICTOR if task == 'ref2va' else T2VA_PREDICTOR
    elif predictor not in {T2VA_PREDICTOR, R2VA_PREDICTOR}:
        raise ValueError('Unknown VeDA predictor. Choose Auto or an installed supported predictor.')
    elif (predictor == R2VA_PREDICTOR) != (task == 'ref2va'):
        raise ValueError('VeDA predictor does not match the resolved reference/first-last task. Choose Auto.')
    guide_options = {}
    if plan.get('task_mode') == 'longvid_multishot_hardcut':
        # Interior shot anchors must remain visible when the spatial tile grid
        # changes. Keep every guide tile and establish the take with a dense step.
        guide_options = {'reference_sparsity': '0%', 'full_attention_steps': '0'}
    result = node.execute(model=model, predictor=predictor, verbose=bool(options.get('verbose', False)), **guide_options)
    patched = result.result[0] if hasattr(result, 'result') else result[0]
    logging.getLogger(__name__).info('IAMCCS VeDA enabled | task=%s | predictor=%s | kernel self-test required', task, predictor)
    if guide_options:
        logging.getLogger(__name__).info('IAMCCS VeDA multishot guides | reference attention=full | first sampling step=dense')
    return patched

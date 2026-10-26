"""IAMCCS KittenTTS 2 AUDIO node. Model paths come from extra_model_paths.yaml."""
import json
from pathlib import Path
import tempfile
import folder_paths
import torch
from . import iamccs_kitten_tts2_runtime as runtime

folder_paths.folder_names_and_paths.setdefault('kitten_tts2',
    ([str(Path(folder_paths.models_dir) / 'kitten_tts2')], set()))


def model_roots():
    found = {}
    for base in folder_paths.get_folder_paths('kitten_tts2'):
        parent = Path(base)
        if not parent.is_dir():
            continue
        for directory in [parent, *sorted(p for p in parent.iterdir() if p.is_dir())]:
            config = directory / 'config.json'
            try:
                if json.loads(config.read_text(encoding='utf-8')).get('type') == 'KITTEN2':
                    name = directory.name
                    if name in found and found[name] != directory.resolve():
                        raise ValueError('Duplicate Kitten model directory name: ' + name)
                    found[name] = directory.resolve()
            except (OSError, json.JSONDecodeError):
                continue
    return found


def speakers():
    voices = set()
    for root in model_roots().values():
        config = json.loads((root / 'config.json').read_text(encoding='utf-8'))
        voices.update(json.loads((root / config.get('voices', 'voices/voices.json')).read_text(encoding='utf-8')))
    return sorted(voices) or ['Bruno']


class IAMCCSKittenTTS2:
    CATEGORY = 'IAMCCS/Audio/KittenTTS 2'
    RETURN_TYPES = ('AUDIO',)
    FUNCTION = 'generate'
    DESCRIPTION = 'Powered by KittenML. Native emotion tags, emphasis, speakers and reference cloning. Community model license; see bundled notice.'

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': (list(model_roots()) or ['not installed'],),
            'text': ('STRING', {'multiline': True, 'default': '[tender] Welcome to the cinema.'}),
            'speaker': (speakers(), {'default': 'Bruno'}),
            'preset': (['stable', 'expressive'],),
            'device': (['auto', 'cuda', 'cpu', 'cpu-fast'],),
            'decoder': (['student_w4', 'default'],),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffffffffffff}),
            'max_new_tokens': ('INT', {'default': 1000, 'min': 50, 'max': 4000}),
            'normalize_english': ('BOOLEAN', {'default': False}),
            'keep_loaded': ('BOOLEAN', {'default': False}),
        }, 'optional': {'reference': ('AUDIO',), 'reference_text': ('STRING', {'multiline': True, 'default': ''})}}

    def generate(self, model, text, speaker, preset, device, decoder, seed,
                 max_new_tokens, normalize_english, keep_loaded, reference=None, reference_text=''):
        import soundfile as sf
        import comfy.model_management as mm
        if model not in model_roots():
            raise ValueError('Install/select the KittenTTS 2 model and refresh ComfyUI.')
        if not text.strip() or len(text) > 20000:
            raise ValueError('Kitten needs 1–20000 characters of dialogue.')
        if reference is not None and not reference_text.strip():
            raise ValueError('Enter the exact reference transcript. Automatic Whisper downloads are disabled.')
        root = model_roots()[model]
        # External generators do not participate in Comfy image-model management.
        mm.unload_all_models()
        mm.soft_empty_cache()
        with runtime.LOCK, tempfile.TemporaryDirectory(prefix='iamccs-kitten-', dir=folder_paths.get_temp_directory()) as stage:
            request = dict(root=str(root), text=text, speaker=speaker, preset=preset, device=device,
                           decoder=decoder, seed=seed, max_new_tokens=max_new_tokens,
                           normalize_english=normalize_english, keep_loaded=keep_loaded,
                           output=str(Path(stage) / 'take.wav'), reference_text=reference_text)
            if reference is not None:
                wave = reference['waveform'].detach().float().cpu()
                if wave.ndim != 3 or wave.shape[0] != 1:
                    raise ValueError('Kitten expects one reference clip, not an audio batch.')
                seconds = wave.shape[-1] / reference['sample_rate']
                if seconds < 5 or seconds > 30:
                    raise ValueError('Use a clean 5–30 second reference recording with its exact transcript.')
                request['reference'] = str(Path(stage) / 'reference.wav')
                sf.write(request['reference'], wave[0].mean(dim=0).numpy(), reference['sample_rate'])
            runtime.run(request, mm.throw_exception_if_processing_interrupted)
            audio, rate = sf.read(request['output'], dtype='float32', always_2d=True)
            return ({'waveform': torch.from_numpy(audio.T.copy()).unsqueeze(0), 'sample_rate': rate},)


NODE_CLASS_MAPPINGS = {'IAMCCS_KittenTTS2': IAMCCSKittenTTS2}
NODE_DISPLAY_NAME_MAPPINGS = {'IAMCCS_KittenTTS2': 'IAMCCS KittenTTS 2 · Voice / Emotion / Clone'}

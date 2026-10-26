# SPDX-License-Identifier: GPL-3.0-or-later
"""IAMCCS CUDA execution contracts for the separately installed Drama 3 runtime.

The upstream runtime owns model math, quantization, codec and expert streaming.
No CPU inference fallback, implicit downloads or machine-specific model paths.
"""
import logging
import os

import torch
import nodes
import folder_paths
import comfy.model_management

LOGGER = logging.getLogger(__name__)
folder_paths.add_model_folder_path('asr', os.path.join(folder_paths.models_dir, 'asr'))
PROFILES = {'8 GB · RAM experts': (True, .5), '12 GB · RAM experts': (True, 1.5),
            '24 GB · RAM experts': (True, 8.0), 'VRAM experts · advanced': (False, 0.0)}


def cuda_device():
    device = comfy.model_management.get_torch_device()
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('DRAMA 3 GPU requires CUDA. Start the selected ComfyUI on its NVIDIA GPU, not --cpu.')
    return device


def native(name, **inputs):
    cls = nodes.NODE_CLASS_MAPPINGS.get(name)
    if cls is None:
        raise RuntimeError(f'Install the pinned ComfyUI-Drama3 runtime and restart ComfyUI: missing {name}.')
    return cls.execute(**inputs).result


def choices(category, prefix):
    return [name for name in folder_paths.get_filename_list(category)
            if name.replace('\\', '/').rsplit('/', 1)[-1].startswith(prefix)]


class IAMCCS_Drama3GPUModelLoader:
    CATEGORY = 'IAMCCS/AUDIA/Drama 3'
    FUNCTION = 'load'
    RETURN_TYPES = ('DRAMA3_MODEL',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model_name': (choices('diffusion_models', 'drama3_'),),
                             'memory_profile': (list(PROFILES), {'default': '12 GB · RAM experts'})}}

    def load(self, model_name, memory_profile):
        device = cuda_device()
        if memory_profile not in PROFILES:
            raise ValueError('Choose an advertised DRAMA 3 memory profile.')
        # Resolve the untrusted combo again before handing it to the runtime.
        folder_paths.get_full_path_or_raise('diffusion_models', model_name)
        experts_in_ram, cache = PROFILES[memory_profile]
        result = native('Drama3Loader', model_name=model_name,
                        experts='in RAM' if experts_in_ram else 'in VRAM', expert_cache_gb=cache,
                        second_gpu='none', split_layer=0)
        if result[0].patcher.load_device.type != 'cuda':
            raise RuntimeError('DRAMA 3 did not select a CUDA execution device; CPU fallback refused.')
        LOGGER.info('DRAMA 3 execution=%s; expert storage=%s; cache limit=%.1f GiB',
                    device, 'RAM' if experts_in_ram else 'VRAM', cache)
        return (result[0],)


class IAMCCS_Drama3GPUCodecLoader:
    CATEGORY = 'IAMCCS/AUDIA/Drama 3'
    FUNCTION = 'load'
    RETURN_TYPES = ('DRAMA3_CODEC',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'codec_name': (choices('vae', 'drama3_codec'),)}}

    def load(self, codec_name):
        cuda_device()
        folder_paths.get_full_path_or_raise('vae', codec_name)
        result = native('Drama3CodecLoader', codec_name=codec_name)
        if result[0].patcher.load_device.type != 'cuda':
            raise RuntimeError('DRAMA 3 codec requires CUDA; CPU fallback refused.')
        return (result[0],)


class IAMCCS_Drama3GPUEncodeVoice:
    CATEGORY = 'IAMCCS/AUDIA/Drama 3'
    FUNCTION = 'encode'
    RETURN_TYPES = ('DRAMA3_VOICE', 'STRING')
    RETURN_NAMES = ('voice', 'reference_transcript')

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'codec': ('DRAMA3_CODEC',), 'audio': ('AUDIO',),
                             'transcript': ('STRING', {'default': '', 'multiline': True}),
                             'asr_model': (['none'] + choices('asr', 'qwen3_asr_'),
                                           {'default': 'none'})}}

    def encode(self, codec, audio, transcript, asr_model):
        cuda_device()
        text = transcript.strip()
        if not text:
            if asr_model == 'none':
                raise ValueError('Enter the exact reference transcript or choose the installed Qwen3-ASR model.')
            folder_paths.get_full_path_or_raise('asr', asr_model)
            asr = native('Qwen3ASRLoader', model_name=asr_model)[0]
            text = native('Qwen3ASRTranscribe', model=asr, audio=audio, language='auto',
                          context='', max_new_tokens=4096)[0].strip()
            if not text:
                raise ValueError('Qwen3-ASR returned no reference words. Use a clean speech clip and review its transcript.')
        voice = native('Drama3EncodeVoice', codec=codec, audio=audio, transcript=text)[0]
        return (voice, text)


class IAMCCS_Drama3GenerateGPU:
    """One performance/turn; PAN keeps separate speaker turns in its arranger.

For native multi-speaker generation connect this GPU loader to Drama3Generate,
whose upstream Autogrow input supports up to 16 encoded voices.
"""
    CATEGORY = 'IAMCCS/AUDIA/Drama 3'
    FUNCTION = 'generate'
    RETURN_TYPES = ('AUDIO',)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'model': ('DRAMA3_MODEL',), 'codec': ('DRAMA3_CODEC',),
            'text': ('STRING', {'multiline': True, 'default': ''}),
            'seed': ('INT', {'default': 0, 'min': 0, 'max': 0xffffffffffffffff}),
            'temperature': ('FLOAT', {'default': 1., 'min': .01, 'max': 2., 'step': .05}),
            'top_p': ('FLOAT', {'default': .9, 'min': .01, 'max': 1., 'step': .01}),
            'top_k': ('INT', {'default': 0, 'min': 0, 'max': 4096}),
            'max_frames': ('INT', {'default': 600, 'min': 1, 'max': 8192}),
            'split_long_text': ('BOOLEAN', {'default': True}),
            'shorten_long_pauses': ('BOOLEAN', {'default': True}),
        }, 'optional': {'voice': ('DRAMA3_VOICE',)}}

    def generate(self, model, codec, text, seed, temperature, top_p, top_k, max_frames,
                 split_long_text=True, shorten_long_pauses=True, voice=None):
        cuda_device()
        if model.patcher.load_device.type != 'cuda' or codec.patcher.load_device.type != 'cuda':
            raise RuntimeError('DRAMA 3 model and codec must use CUDA; CPU fallback refused.')
        codes = native('Drama3Generate', model=model, text=text, voices={'voice0': voice} if voice else {},
                       seed=seed, temperature=temperature, top_p=top_p, top_k=top_k, max_frames=max_frames,
                       split_long_text=split_long_text, speakers='one take per line')[0]
        return (native('Drama3Decode', codec=codec, codes=codes, shorten_long_pauses=shorten_long_pauses)[0],)


NODE_CLASS_MAPPINGS = {cls.__name__: cls for cls in (
    IAMCCS_Drama3GPUModelLoader, IAMCCS_Drama3GPUCodecLoader,
    IAMCCS_Drama3GPUEncodeVoice, IAMCCS_Drama3GenerateGPU)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'IAMCCS_Drama3GPUModelLoader': 'IAMCCS Drama 3 · CUDA Model',
    'IAMCCS_Drama3GPUCodecLoader': 'IAMCCS Drama 3 · CUDA Codec',
    'IAMCCS_Drama3GPUEncodeVoice': 'IAMCCS Drama 3 · Voice ID + Qwen ASR',
    'IAMCCS_Drama3GenerateGPU': 'IAMCCS Drama 3 · GPU Performance',
}

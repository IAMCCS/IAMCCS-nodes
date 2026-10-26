"""Owned, isolated Kitten runtime. JSON lines on stdout; SDK logs on stderr."""
import contextlib
import json
import os
from pathlib import Path
import sys
import traceback
import threading


def watch_parent():
    # Windows venv launchers can own this second Python process. Its lifetime
    # must end with the owned launcher even when inference is being cancelled.
    if sys.platform != 'win32':
        return
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x00100000, False, os.getppid())
    if handle:
        def wait():
            kernel.WaitForSingleObject(handle, 0xffffffff)
            kernel.CloseHandle(handle)
            os._exit(0)
        threading.Thread(target=wait, daemon=True).start()


watch_parent()

model = None
identity = None


def generate(request):
    global model, identity
    root = Path(request['root']).resolve()
    cache = root.parent / '.hf'
    os.environ.update(HF_HOME=str(cache), HF_HUB_CACHE=str(cache / 'hub'),
                      TORCH_HOME=str(root.parent / '.torch'), HF_HUB_OFFLINE='1')
    os.environ.pop('TRANSFORMERS_CACHE', None)
    import numpy as np
    import soundfile as sf
    import torch
    device = request['device']
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cpp = device == 'cpu-fast'
    if cpp and request.get('reference'):
        raise ValueError('Fast CPU currently supports built-in speakers only. Choose auto, cuda or cpu for raw-reference cloning.')
    key = (str(root), device, request['decoder'])
    if identity != key:
        model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if cpp:
            from kitten_tts_cpp import KittenTTS
            model = KittenTTS(str(root), device='cpu', decoder=request['decoder'],
                              offline=True, threads=min(8, os.cpu_count() or 4))
        else:
            from kittenml import KittenTTS
            model = KittenTTS(str(root), device=device, decoder=request['decoder'],
                              cache_dir=str(cache / 'hub'), weights='packed')
        identity = key
    options = dict(preset=request['preset'], max_new_tokens=request['max_new_tokens'],
                   normalize=request['normalize_english'], advanced={'seed': request['seed']})
    if request.get('reference'):
        transcript = request.get('reference_text', '').strip()
        if not transcript:
            raise ValueError('Exact reference transcript required; automatic Whisper downloads are disabled.')
        options.update(reference=request['reference'], reference_text=transcript)
    else:
        options['voice'] = request['speaker']
    samples = np.asarray(model.generate(request['text'], **options), dtype=np.float32)
    if not samples.size or not np.isfinite(samples).all():
        raise ValueError('Kitten returned empty or non-finite audio.')
    sf.write(request['output'], samples, model.sample_rate, subtype='PCM_24')
    return {'ok': True, 'sample_rate': model.sample_rate,
            'seconds': len(samples) / model.sample_rate, 'device': device}


for line in sys.stdin:
    try:
        request = json.loads(line)
        with contextlib.redirect_stdout(sys.stderr):
            result = generate(request)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        result = {'ok': False, 'error': str(error)}
    print(json.dumps(result), flush=True)

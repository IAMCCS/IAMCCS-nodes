"""Persistent child runtime; does not replace Comfy's torch or download models."""
import atexit
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time

_process = None
_key = None
_responses = None
_log = None
LOCK = threading.Lock()


def close():
    global _process, _key, _log
    if _process is not None:
        _process.terminate()
        try:
            _process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _process.kill()
        _process = None
    if _log:
        _log.close()
        _log = None
    _key = None


atexit.register(close)


def run(request, interrupt=lambda: None):
    global _process, _key, _responses, _log
    root = Path(request['root'])
    config_path = root.parent / 'pan-runtime.json'
    if not config_path.is_file():
        raise RuntimeError('Kitten runtime is not configured. Run the PAN KittenTTS 2 installer.')
    config = json.loads(config_path.read_text(encoding='utf-8-sig'))
    executable = config.get('cpu_python' if request['device'] == 'cpu-fast' else 'python')
    if not executable or not Path(executable).is_file():
        raise RuntimeError('The configured Kitten Python runtime is missing. Run the PAN installer.')
    key = (executable, request['root'], request['device'], request['decoder'])
    if _key != key or _process is None or _process.poll() is not None:
        close()
        log_path = root.parent / 'pan-runtime.log'
        if log_path.exists() and log_path.stat().st_size > 2_000_000:
            log_path.write_text('', encoding='utf-8')
        _log = log_path.open('a', encoding='utf-8')
        _responses = queue.Queue()
        _process = subprocess.Popen([executable, '-u', str(Path(__file__).with_name('iamccs_kitten_tts2_worker.py'))],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=_log,
                                    text=True, encoding='utf-8', creationflags=0x08000000 if sys.platform=='win32' else 0)
        responses, process = _responses, _process
        def read():
            for line in process.stdout:
                responses.put(line)
            responses.put(None)
        threading.Thread(target=read, daemon=True).start()
        _key = key
    try:
        _process.stdin.write(json.dumps(request) + '\n')
        _process.stdin.flush()
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            interrupt()
            try:
                line = _responses.get(timeout=.2)
            except queue.Empty:
                continue
            if line is None:
                raise RuntimeError('Kitten runtime exited. See pan-runtime.log beside the model directory.')
            result = json.loads(line)
            if not result.get('ok'):
                raise RuntimeError('KittenTTS 2: ' + result.get('error', 'generation failed'))
            return result
        raise TimeoutError('Kitten generation exceeded 30 minutes; the owned runtime was stopped.')
    except BaseException:
        close()
        raise
    finally:
        if not request.get('keep_loaded', False):
            close()

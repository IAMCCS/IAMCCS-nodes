"""Decode container audio without requiring TorchCodec or libsndfile codecs."""
import shutil
import subprocess


def decode_audio_with_ffmpeg(path, sample_rate=48000):
    import numpy as np
    import torch
    executable = shutil.which("ffmpeg")
    if not executable:
        import imageio_ffmpeg
        executable = imageio_ffmpeg.get_ffmpeg_exe()
    result = subprocess.run(
        [executable, "-v", "error", "-nostdin", "-i", str(path),
         "-map", "0:a:0", "-vn", "-ac", "2", "-ar", str(sample_rate),
         "-f", "f32le", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode or not result.stdout or len(result.stdout) % 8:
        detail = result.stderr.decode("utf-8", errors="replace")[-1500:]
        raise ValueError(f"FFmpeg could not decode an audio stream from {path!r}: {detail}")
    data = np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2).T.copy()
    return torch.from_numpy(data).unsqueeze(0).contiguous(), int(sample_rate)

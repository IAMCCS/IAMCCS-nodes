# SPDX-License-Identifier: GPL-3.0-or-later
"""Bound the existing H3 -> LTX AV refinement sampler's working set."""

from __future__ import annotations

from copy import copy
import json
import logging
import math
from pathlib import Path
import time

import folder_paths
import torch
import comfy.nested_tensor
import comfy.utils
from comfy_extras.nodes_custom_sampler import SamplerCustomAdvanced


LOG = logging.getLogger("IAMCCS.MiniMaxH3.LTXTiled")
SAMPLER = "IAMCCS_MiniMaxH3LTXTiledSampler"
MAX_TILE_TOKENS = 4096


def _ranges(length, size, overlap):
    size = min(length, size)
    overlap = min(max(0, overlap), size // 2)
    count = math.ceil((length - size) / (size - overlap)) + 1
    starts = [round(i * (length - size) / (count - 1)) for i in range(count)] if count > 1 else [0]
    return [(start, start + size) for start in starts]


def _tile_size(frames, height, width, horizontal, vertical, overlap):
    h = min(height, math.ceil((height + (vertical - 1) * overlap) / vertical))
    w = min(width, math.ceil((width + (horizontal - 1) * overlap) / horizontal))
    while frames * h * w > MAX_TILE_TOKENS:
        if w >= h and w > 1:
            w = max(1, min(w - 1, math.ceil((w + overlap) / 2)))
        elif h > 1:
            h = max(1, min(h - 1, math.ceil((h + overlap) / 2)))
        else:
            break
    return h, w


def _axis_weight(ranges, index, dtype):
    start, end = ranges[index]
    weight = torch.ones(end - start, dtype=dtype, device="cpu")
    if index:
        overlap = max(0, ranges[index - 1][1] - start)
        weight[:overlap] *= torch.arange(1, overlap + 1, dtype=dtype) / (overlap + 1)
    if index + 1 < len(ranges):
        overlap = max(0, end - ranges[index + 1][0])
        if overlap:
            weight[-overlap:] *= torch.arange(overlap, 0, -1, dtype=dtype) / (overlap + 1)
    return weight


class _TileNoise:
    def __init__(self, seed, video, audio):
        self.seed, self.video, self.audio = seed, video, audio

    def generate_noise(self, latent):
        return comfy.nested_tensor.NestedTensor((self.video, self.audio))


def _sample_tiled(noise, guider, sampler, sigmas, latent_image, settings, fps):
    samples = latent_image["samples"]
    if not samples.is_nested or len(samples.unbind()) != 2:
        raise ValueError("H3 LTX tiled delivery requires the existing joint LTX AV latent")
    video, audio = samples.unbind()
    video, audio = video.cpu(), audio.cpu()
    frames, height, width = video.shape[2:]
    # LTX's video VAE has an 8-frame temporal stride; metadata stays in pixels.
    window = min(frames, max(2, min(9, int(settings.get("temporal_tile_size", 80)) // 8)))
    temporal_overlap = min(window // 2, max(0, int(settings.get("temporal_overlap", 24)) // 8))
    spatial_overlap = max(1, int(settings.get("spatial_overlap", 1)))
    tile_h, tile_w = _tile_size(window, height, width,
                                max(1, int(settings.get("horizontal_tiles", 1))),
                                max(1, int(settings.get("vertical_tiles", 1))), spatial_overlap)
    times = _ranges(frames, window, temporal_overlap)
    ys = _ranges(height, tile_h, spatial_overlap)
    xs = _ranges(width, tile_w, spatial_overlap)
    pieces = len(times) * len(ys) * len(xs)
    generated_noise = noise.generate_noise(latent_image)
    video_noise = generated_noise.unbind()[0].cpu()
    del generated_noise
    masks = latent_image.get("noise_mask")
    video_mask = masks.unbind()[0] if masks is not None and masks.is_nested else masks
    if video_mask is not None:
        video_mask = comfy.utils.reshape_mask(video_mask.cpu(), video.shape)
    output = torch.zeros_like(video, device="cpu")
    denoised = torch.zeros_like(video, device="cpu")
    weights = torch.zeros((1, 1, frames, height, width), dtype=video.dtype, device="cpu")
    progress = comfy.utils.ProgressBar(pieces)
    LOG.info("LTX tiled refinement | video=%s | windows=%d spatial_tiles=%dx%d pieces=%d | max_tokens=%d | audio=preserved",
             tuple(video.shape), len(times), len(xs), len(ys), pieces, MAX_TILE_TOKENS)
    piece = 0
    for ti, (t0, t1) in enumerate(times):
        # Match each AV piece to its time interval, including the causal first frame.
        start_frame = max(0, t0 * 8 - 7)
        end_frame = (t1 - 1) * 8 + 1
        total_frames = (frames - 1) * 8 + 1
        a0 = min(audio.shape[2] - 1, start_frame * audio.shape[2] // total_frames)
        a1 = min(audio.shape[2], max(a0 + 1, math.ceil(end_frame * audio.shape[2] / total_frames)))
        tw = _axis_weight(times, ti, video.dtype)
        for yi, (y0, y1) in enumerate(ys):
            yw = _axis_weight(ys, yi, video.dtype)
            for xi, (x0, x1) in enumerate(xs):
                piece += 1
                started = time.monotonic()
                region = (slice(None), slice(None), slice(t0, t1), slice(y0, y1), slice(x0, x1))
                tile_video = video[region].clone()
                tile_audio = audio[:, :, a0:a1].clone()
                tile_mask = video_mask[region].clone() if video_mask is not None else torch.ones_like(tile_video)
                tile = {"samples": comfy.nested_tensor.NestedTensor((tile_video, tile_audio)),
                        "noise_mask": comfy.nested_tensor.NestedTensor((tile_mask, torch.zeros_like(tile_audio)))}
                tile_guider = copy(guider)
                tile_guider.original_conds = {
                    key: [dict(cond, frame_rate=fps) for cond in values]
                    for key, values in guider.original_conds.items()
                }
                tile_noise = _TileNoise(noise.seed, video_noise[region].clone(),
                                        torch.zeros_like(tile_audio))
                LOG.info("LTX refine | piece=%d/%d START | latent_t=%d:%d y=%d:%d x=%d:%d tokens=%d",
                         piece, pieces, t0, t1, y0, y1, x0, x1, tile_video.shape[2] * tile_video.shape[3] * tile_video.shape[4])
                result = SamplerCustomAdvanced.execute(noise=tile_noise, guider=tile_guider,
                                                       sampler=sampler, sigmas=sigmas, latent_image=tile)
                sampled = result[0]["samples"].unbind()[0].cpu()
                clean = result[1]["samples"].unbind()[0].cpu()
                xw = _axis_weight(xs, xi, video.dtype)
                weight = tw[None, None, :, None, None] * yw[None, None, None, :, None] * xw[None, None, None, None, :]
                output[region] += sampled * weight
                denoised[region] += clean * weight
                weights[:, :, t0:t1, y0:y1, x0:x1] += weight
                del result, sampled, clean, tile, tile_video, tile_audio, tile_mask, tile_noise, tile_guider
                progress.update_absolute(piece, pieces)
                LOG.info("LTX refine | piece=%d/%d DONE | elapsed=%.1fs", piece, pieces, time.monotonic() - started)
    output.div_(weights)
    denoised.div_(weights)
    results = []
    for value in (output, denoised):
        item = dict(latent_image)
        item["samples"] = comfy.nested_tensor.NestedTensor((value, audio))
        results.append(item)
    return tuple(results)


class IAMCCS_MiniMaxH3LTXTiledSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"noise": ("NOISE",), "guider": ("GUIDER",), "sampler": ("SAMPLER",),
                             "sigmas": ("SIGMAS",), "latent_image": ("LATENT",)},
                "optional": {"cine_linx": ("IAMCCS_SUPERNODE_LINX",)}}

    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"
    CATEGORY = "IAMCCS/MiniMax H3/Delivery"

    def sample(self, noise, guider, sampler, sigmas, latent_image, cine_linx=None):
        from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan

        plan = _resolve_shotplan(cine_linx) if cine_linx is not None else {}
        settings = plan.get("upscale_settings", {}).get("ltx_looper", {})
        return _sample_tiled(noise, guider, sampler, sigmas, latent_image, settings, int(plan.get("fps", 24)))


def upgrade_ltx_prompt(data):
    """Upgrade only the existing IAMCCS H3 LTX delivery graph at queue ingress."""
    prompt = data.get("prompt", {})
    for node in list(prompt.values()):
        if not isinstance(node, dict) or node.get("class_type") != "SamplerCustomAdvanced":
            continue
        ancestors = {}
        pending = list(node.get("inputs", {}).values())
        while pending:
            link = pending.pop()
            if not isinstance(link, list) or len(link) != 2 or str(link[0]) in ancestors:
                continue
            parent = prompt.get(str(link[0]))
            if not isinstance(parent, dict):
                continue
            ancestors[str(link[0])] = parent
            pending.extend(parent.get("inputs", {}).values())
        classes = {parent.get("class_type") for parent in ancestors.values()}
        if "IAMCCS_MiniMaxH3LTXDenoiseStackLoaderV3" not in classes or "LTXVLatentUpsampler" not in classes:
            continue
        control = next((parent for parent in ancestors.values()
                        if parent.get("class_type") == "IAMCCS_MiniMaxH3PostUpscaleControlV2"), None)
        if control is None or "cine_linx" not in control.get("inputs", {}):
            continue
        node["class_type"] = SAMPLER
        node["inputs"]["cine_linx"] = control["inputs"]["cine_linx"]
        LOG.info("H3 LTX delivery sampler upgraded to bounded temporal/spatial AV tiles")
    return data


class IAMCCS_MiniMaxH3LTXResumeSource:
    """Read a saved native checkpoint and its plan without loading H3 models."""
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"native_file": ("STRING", {"default": ""}),
                             "upscale_prompt": ("STRING", {"default": "Preserve the source video's exact shot order, hard cuts, framing, faces, clothing, lighting, lip motion and performance. Recover fine detail without redesigning subjects or changing camera motion.", "multiline": True})}}

    RETURN_TYPES = ("IMAGE", "AUDIO", "IAMCCS_SUPERNODE_LINX", "BOOLEAN", "STRING", "INT")
    RETURN_NAMES = ("native_frames", "native_audio", "cine_linx", "master_ready", "render_id", "fps")
    FUNCTION = "load"
    CATEGORY = "IAMCCS/MiniMax H3/Delivery"

    @staticmethod
    def _path(native_file):
        root = Path(folder_paths.get_output_directory()).resolve()
        path = Path(native_file).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ValueError("LTX resume source must be a native checkpoint under ComfyUI/output")
        return path

    @classmethod
    def IS_CHANGED(cls, native_file, **kwargs):
        path = cls._path(native_file)
        return path.stat().st_mtime_ns, path.with_suffix(path.suffix + ".iamccs.json").stat().st_mtime_ns

    def load(self, native_file, upscale_prompt):
        from .iamccs_minimax_h3_shotboard import _decode_video_master, _build_locked_audio_wave_master

        path = self._path(native_file)
        root = Path(folder_paths.get_output_directory()).resolve()
        metadata_path = path.with_suffix(path.suffix + ".iamccs.json").resolve(strict=True)
        if not metadata_path.is_relative_to(root):
            raise ValueError("LTX checkpoint metadata is outside ComfyUI/output")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        generation = metadata["generation"]
        plan = generation["shotplan"]
        fps = int(metadata["fps"])
        frames, audio = _decode_video_master(path, fps=fps)
        if len(frames) != int(metadata["frame_count"]):
            raise ValueError("LTX resume checkpoint frame count differs from its metadata")
        segments = [(path.parent / name).resolve(strict=True) for name in generation.get("segment_files", [])]
        if any(not segment.is_relative_to(root) for segment in segments):
            raise ValueError("LTX checkpoint audio segments are outside ComfyUI/output")
        if segments and all(segment.with_suffix(segment.suffix + ".locked_audio.npz").is_file() for segment in segments):
            for segment in segments:
                for suffix in (".iamccs.json", ".locked_audio.npz"):
                    sidecar = segment.with_suffix(segment.suffix + suffix).resolve(strict=True)
                    if not sidecar.is_relative_to(root):
                        raise ValueError("LTX checkpoint audio sidecar is outside ComfyUI/output")
            counts = [int(json.loads(segment.with_suffix(segment.suffix + ".iamccs.json").read_text(encoding="utf-8"))["frame_count"])
                      for segment in segments]
            if sum(counts) != len(frames):
                raise ValueError("LTX resume audio segment durations differ from the saved master")
            wave, rate, _ = _build_locked_audio_wave_master(segments, counts)
            audio = {"waveform": torch.from_numpy(wave).unsqueeze(0), "sample_rate": rate}
        plan["upscale_mode"], plan["upscale_enabled"] = "ltx23", True
        plan["upscale_settings"]["prompt"] = str(upscale_prompt)
        run = f"{generation['render_id']}_ltx_resume_{time.time_ns()}"
        LOG.info("LTX resume | native=%s | frames=%d | H3 models=not required", path.name, len(frames))
        return frames, audio, plan, True, run, fps


NODE_CLASS_MAPPINGS = {SAMPLER: IAMCCS_MiniMaxH3LTXTiledSampler,
                       "IAMCCS_MiniMaxH3LTXResumeSource": IAMCCS_MiniMaxH3LTXResumeSource}
NODE_DISPLAY_NAME_MAPPINGS = {SAMPLER: "MiniMax H3 → LTX · Tiled AV Refinement",
                              "IAMCCS_MiniMaxH3LTXResumeSource": "MiniMax H3 → LTX · Resume Saved Native"}

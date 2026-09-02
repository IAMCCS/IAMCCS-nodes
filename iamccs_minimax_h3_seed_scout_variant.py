# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later

"""Isolated R40 Seed Scout / selected-x0 / second-pass delivery variant.

R39 nodes are intentionally not subclassed or patched in-place.  The seed
candidate nodes are upstream of the selector, while the selector and Stage-2
seed live in a separate downstream control node.  That graph boundary is what
lets ComfyUI reuse cached candidates when only the selected take or refine seed
changes.
"""

from __future__ import annotations

import copy
import gc
import logging
from pathlib import Path
from typing import Any

import folder_paths
import torch

from .iamccs_minimax_h3_atomic_backend import (
    H3_FPS,
    SUPERNODE_LINX_TYPE,
    _accelerate,
    _apply_secondary_lora,
    _apply_turbo_lora,
    _audit_lipsync_audio_lock,
    _chunk,
    _clean_vram_before_decode,
    _h3_lora_compatibility,
    _release_conditioning_models,
    _resolve_shotplan,
    _turbo_sampler,
    _turbo_settings,
)
from .iamccs_minimax_h3_latent_upres_variant import _delivery_size, _grid_up, _safe_render_id, _upres_settings


LOG = logging.getLogger("IAMCCS.MiniMaxH3.SeedScoutR40")
CATEGORY = "IAMCCS/MiniMax H3/R40 Shot Lab"
R40_SHOT_LAB = "IAMCCS_H3_R40_SHOT_LAB"
R40_TAKE_SELECT = "IAMCCS_H3_R40_TAKE_SELECT"


def _scout_settings(plan: dict[str, Any]) -> dict[str, Any]:
    value = plan.get("seed_scout_settings")
    return value if isinstance(value, dict) else {}


def _node_output(result: Any, index: int = 0):
    values = getattr(result, "result", result)
    return values[index]


def _apply_sparse(model, enabled: bool, budget: float, denser_edges: bool):
    if not enabled:
        return model, "off"
    try:
        import nodes as comfy_nodes

        node_cls = comfy_nodes.NODE_CLASS_MAPPINGS.get("H3SparseAttention")
        if node_cls is None:
            raise RuntimeError("H3SparseAttention is not registered")
        result = node_cls.execute(
            model=model,
            video_budget=max(0.05, min(1.0, float(budget))),
            denser_early_late_steps=bool(denser_edges),
            layer_video_budgets="",
        )
        return _node_output(result), f"H3SparseAttention budget={float(budget):.2f} edges={bool(denser_edges)}"
    except Exception as exc:
        raise RuntimeError(
            "R40 Sparse Attention was requested but the current H3-Optimizations node could not be applied. "
            "Disable R40 sparse mode or update H3-Optimizations."
        ) from exc


def _active_sampling_model(model, plan: dict[str, Any]):
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

    sampling = plan.get("sampling") if isinstance(plan.get("sampling"), dict) else {}
    shift_video = float(sampling.get("shift_video", 12.0))
    shift_audio = float(sampling.get("shift_audio", 3.0))
    turbo_model, turbo_report = _apply_turbo_lora(model, plan)
    lora_model, secondary_report = _apply_secondary_lora(turbo_model, plan)
    turbo = _turbo_settings(plan)
    turbo_name = str(turbo.get("lora_name", "") or "").strip()
    compatible, _ = _h3_lora_compatibility(turbo_name)
    turbo_enabled = (
        str(turbo.get("mode", "off") or "off").lower() != "off"
        and bool(turbo.get("enabled", True))
        and bool(turbo_name)
        and bool(folder_paths.get_full_path("loras", turbo_name))
        and compatible
    )
    if turbo_enabled:
        accelerated, acceleration_report = _accelerate(lora_model, plan)
        active = MiniMaxH3SigmaShift.execute(
            model=accelerated, shift_video=shift_video, shift_audio=shift_audio
        )[0]
    else:
        shifted = MiniMaxH3SigmaShift.execute(
            model=lora_model, shift_video=shift_video, shift_audio=shift_audio
        )[0]
        active, acceleration_report = _accelerate(shifted, plan)
    scout = _scout_settings(plan)
    active, sparse_report = _apply_sparse(
        active,
        bool(scout.get("sparse_enabled", False)),
        float(scout.get("sparse_video_budget", 0.30)),
        bool(scout.get("sparse_denser_edges", True)),
    )
    return active, turbo_enabled, turbo_report, secondary_report, acceleration_report, sparse_report


class IAMCCS_MiniMaxH3SeedCandidateR40:
    """One cacheable low-resolution seed candidate; no candidate selection input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "chunk_index": ("INT", {"forceInput": True}),
                "candidate_index": ("INT", {"default": 0, "min": 0, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "BOOLEAN", "STRING")
    RETURN_NAMES = ("sampled_latent", "denoised_x0", "candidate_seed", "active", "report")
    FUNCTION = "sample"
    CATEGORY = CATEGORY

    def sample(self, model, positive, latent, cine_linx, chunk_index, candidate_index):
        from comfy_extras.nodes_custom_sampler import (
            BasicGuider,
            BasicScheduler,
            KSamplerSelect,
            RandomNoise,
            SamplerCustomAdvanced,
        )

        plan = _resolve_shotplan(cine_linx)
        scout = _scout_settings(plan)
        enabled = bool(scout.get("enabled", False))
        count = max(1, min(4, int(scout.get("candidate_count", 1) or 1))) if enabled else 1
        index = max(0, min(3, int(candidate_index)))
        sampling = plan.get("sampling") if isinstance(plan.get("sampling"), dict) else {}
        base_seed = int(sampling.get("seed", 0) or 0)
        chunk_stride = int(sampling.get("seed_stride", 1) or 1)
        scout_stride = max(1, int(scout.get("seed_stride", 1_000_003) or 1_000_003))
        actual_seed = (base_seed + int(chunk_index) * chunk_stride + index * scout_stride) & 0xFFFFFFFFFFFFFFFF
        if index >= count:
            inactive_sampled = copy.copy(latent)
            inactive_x0 = copy.copy(latent)
            inactive_sampled["iamccs_r40_candidate_active"] = False
            inactive_x0["iamccs_r40_candidate_active"] = False
            report = f"R40 candidate {index + 1} inactive | configured candidates={count}"
            return inactive_sampled, inactive_x0, actual_seed, False, report

        _audit_lipsync_audio_lock(plan, latent, int(chunk_index))
        cleanup_report = _release_conditioning_models(plan)
        active_model, turbo_enabled, turbo_report, secondary_report, acceleration_report, sparse_report = (
            _active_sampling_model(model, plan)
        )
        noise = RandomNoise.execute(noise_seed=actual_seed)[0]
        guider = BasicGuider.execute(model=active_model, conditioning=positive)[0]
        turbo = _turbo_settings(plan)
        if turbo_enabled and str(turbo.get("sampler_mode", "audio_fixed") or "audio_fixed").lower() == "audio_fixed":
            sampler, sampler_report = _turbo_sampler(plan)
        else:
            sampler_name = str(sampling.get("sampler_name", "res_multistep") or "res_multistep")
            sampler = KSamplerSelect.execute(sampler_name=sampler_name)[0]
            sampler_report = sampler_name
        sigmas = BasicScheduler.execute(
            model=active_model,
            scheduler=str(sampling.get("scheduler", "simple") or "simple"),
            steps=max(1, int(sampling.get("steps", 8) or 8)),
            denoise=max(0.0, min(1.0, float(sampling.get("denoise", 1.0) or 0.0))),
        )[0]
        result = SamplerCustomAdvanced.execute(
            noise=noise, guider=guider, sampler=sampler, sigmas=sigmas, latent_image=latent
        )
        sampled, denoised = result[0], result[1]
        sampled = copy.copy(sampled)
        denoised = copy.copy(denoised)
        for value, kind in ((sampled, "sampled_output"), (denoised, "denoised_x0")):
            value["iamccs_r40_candidate_active"] = True
            value["iamccs_r40_candidate_index"] = index
            value["iamccs_r40_candidate_seed"] = actual_seed
            value["iamccs_r40_latent_kind"] = kind
        report = (
            f"R40 candidate {index + 1}/{count} | seed={actual_seed} | sampler={sampler_report} | "
            f"turbo={turbo_report} | lora2={secondary_report} | acceleration={acceleration_report} | "
            f"sparse={sparse_report} | conditioning_cleanup={cleanup_report}"
        )
        LOG.info(report)
        return sampled, denoised, actual_seed, True, report


class IAMCCS_MiniMaxH3ShotLabControlR40:
    """Stage-2-only controls: changing these invalidates only delivery."""

    @classmethod
    def INPUT_TYPES(cls):
        loras = [
            name for name in folder_paths.get_filename_list("loras")
            if "h3" in name.lower() or "minimax" in name.lower()
        ]
        loras = [""] + sorted(set(loras), key=str.lower)
        return {
            "required": {
                "stage2_seed": ("INT", {"default": 2026082801, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "display_name": "STAGE 2 · refine seed"}),
                "stage2_seed_stride": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "step": 1, "display_name": "STAGE 2 · seed stride per segment"}),
                "stage2_sigma_preset": (["seedhunter_3", "seedhunter_4", "seedhunter_5", "authored_scheduler"], {"default": "seedhunter_3", "display_name": "STAGE 2 · sigma schedule"}),
                "stage2_lora_enabled": ("BOOLEAN", {"default": False, "display_name": "STAGE 2 LORA · enable"}),
                "stage2_lora_name": (loras, {"default": "", "display_name": "STAGE 2 LORA · model", "tooltip": "Delivery/refine LoRA only. It never changes the selected native candidate or its saved H3 audio."}),
                "stage2_lora_strength": ("FLOAT", {"default": 0.6, "min": -2.0, "max": 2.0, "step": 0.01, "display_name": "STAGE 2 LORA · strength"}),
                "stage2_sparse_enabled": ("BOOLEAN", {"default": False, "display_name": "STAGE 2 SPARSE · enable"}),
                "stage2_sparse_video_budget": ("FLOAT", {"default": 0.30, "min": 0.05, "max": 1.0, "step": 0.05, "display_name": "STAGE 2 SPARSE · video budget"}),
                "stage2_sparse_denser_edges": ("BOOLEAN", {"default": True, "display_name": "STAGE 2 SPARSE · denser edges"}),
            }
        }

    RETURN_TYPES = (R40_SHOT_LAB, "STRING")
    RETURN_NAMES = ("shot_lab", "helper")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, **kwargs):
        control = dict(kwargs)
        control["schema"] = "iamccs.minimax_h3.r40.shot_lab"
        control["schema_version"] = 1
        helper = (
            "R40 Stage-2-only control: seed, sigma schedule, refine LoRA and sparse delivery are outside CineLinX and outside Take Select. "
            "Changing them reuses the candidate bank, selected native decode and native checkpoint."
        )
        return control, helper


class IAMCCS_MiniMaxH3TakeSelectControlR40:
    """Take choice and Stage-2 source; independent from the Stage-2 reroll seed."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "selected_candidate": (["1", "2", "3", "4"], {"default": "1", "display_name": "WINNING TAKE · candidate"}),
                "stage2_latent_source": (["denoised_x0", "sampled_output"], {"default": "denoised_x0", "display_name": "STAGE 2 · source from winning take"}),
            }
        }

    RETURN_TYPES = (R40_TAKE_SELECT, "STRING")
    RETURN_NAMES = ("take_select", "helper")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    def build(self, selected_candidate, stage2_latent_source):
        control = {
            "schema": "iamccs.minimax_h3.r40.take_select",
            "schema_version": 1,
            "selected_candidate": max(1, min(4, int(selected_candidate))),
            "stage2_latent_source": str(stage2_latent_source),
        }
        return control, "Changing the take reuses all cacheable candidates; denoised x0 is Stage-2 only."


class IAMCCS_MiniMaxH3SeedSelectR40:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {}
        for index in range(1, 5):
            optional[f"sampled_{index}"] = ("LATENT", {"lazy": True})
            optional[f"x0_{index}"] = ("LATENT", {"lazy": True})
        return {"required": {"take_select": (R40_TAKE_SELECT,)}, "optional": optional}

    RETURN_TYPES = ("LATENT", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("native_sampled_latent", "stage2_latent", "selected_seed", "report")
    FUNCTION = "select"
    CATEGORY = CATEGORY

    @staticmethod
    def _names(take_select):
        index = max(1, min(4, int(take_select.get("selected_candidate", 1))))
        source = str(take_select.get("stage2_latent_source", "denoised_x0") or "denoised_x0")
        return index, f"sampled_{index}", f"x0_{index}" if source == "denoised_x0" else f"sampled_{index}"

    def check_lazy_status(self, take_select, **kwargs):
        _, native_name, stage2_name = self._names(take_select)
        return [name for name in dict.fromkeys((native_name, stage2_name)) if kwargs.get(name) is None]

    def select(self, take_select, **kwargs):
        index, native_name, stage2_name = self._names(take_select)
        native = kwargs.get(native_name)
        stage2 = kwargs.get(stage2_name)
        if not isinstance(native, dict) or not bool(native.get("iamccs_r40_candidate_active", False)):
            raise ValueError(
                f"R40 candidate {index} is not active. Increase Candidate Count in IAMCCS H3 Settings or select an available take."
            )
        if not isinstance(stage2, dict):
            raise ValueError(f"R40 selected Stage-2 latent {stage2_name} is not connected")
        seed = int(native.get("iamccs_r40_candidate_seed", 0) or 0)
        report = (
            f"R40 selected candidate {index} | native=sampled_output | "
            f"stage2={take_select.get('stage2_latent_source', 'denoised_x0')} | seed={seed} | cached scout reusable=yes"
        )
        return native, stage2, seed, report


class IAMCCS_MiniMaxH3SeedPreviewR40:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "video_vae": ("VAE",),
                "trim_frames": ("INT", {"forceInput": True}),
                "max_preview_frames": ("INT", {"default": 49, "min": 1, "max": 209, "step": 1}),
                "max_preview_width": ("INT", {"default": 512, "min": 128, "max": 1024, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview", "report")
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, latent, video_vae, trim_frames, max_preview_frames, max_preview_width):
        if not bool(latent.get("iamccs_r40_candidate_active", False)):
            return torch.zeros((1, 144, 256, 3), dtype=torch.float32), "R40 candidate inactive"
        import nodes as comfy_nodes

        frames = comfy_nodes.VAEDecode().decode(vae=video_vae, samples=latent)[0]
        trim = max(0, min(int(trim_frames), max(0, int(frames.shape[0]) - 1)))
        frames = frames[trim:]
        maximum = max(1, int(max_preview_frames))
        if int(frames.shape[0]) > maximum:
            indices = torch.linspace(0, int(frames.shape[0]) - 1, maximum).round().long()
            frames = frames.index_select(0, indices)
        width = int(frames.shape[2])
        limit = max(128, int(max_preview_width))
        if width > limit:
            import torch.nn.functional as functional

            height = max(64, round(int(frames.shape[1]) * limit / width))
            frames = functional.interpolate(
                frames.permute(0, 3, 1, 2), size=(height, limit), mode="bilinear", align_corners=False
            ).permute(0, 2, 3, 1).contiguous()
        report = (
            f"R40 candidate {int(latent.get('iamccs_r40_candidate_index', 0)) + 1} preview | "
            f"seed={latent.get('iamccs_r40_candidate_seed', 0)} | frames={int(frames.shape[0])}"
        )
        return frames, report


class IAMCCS_MiniMaxH3SelectedDecodeR40:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampled_latent": ("LATENT",),
                "video_vae": ("VAE",),
                "audio_vae": ("VAE",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "chunk_index": ("INT", {"forceInput": True}),
                "trim_frames": ("INT", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("native_frames", "native_audio", "bridge_last_frame", "sampled_latent", "native_fps", "report")
    FUNCTION = "decode"
    CATEGORY = CATEGORY

    def decode(self, sampled_latent, video_vae, audio_vae, cine_linx, chunk_index, trim_frames):
        import nodes as comfy_nodes
        from comfy_extras.nodes_audio import VAEDecodeAudio
        from .iamccs_minimax_h3_motion_context_variant import _fit_audio, _provider_node

        plan = _resolve_shotplan(cine_linx)
        chunk = _chunk(plan, int(chunk_index))
        cleanup = "disabled"
        if bool(plan.get("vram_clean_before_decode", True)):
            cleanup = _clean_vram_before_decode()
        frames = comfy_nodes.VAEDecode().decode(vae=video_vae, samples=sampled_latent)[0]
        audio = VAEDecodeAudio.execute(vae=audio_vae, samples=sampled_latent)[0]
        trim = max(0, int(trim_frames))
        if trim:
            frames, audio = _provider_node("MiniMaxH3AutoChainMotionContextTrim")().trim(
                images=frames, trim_frames=trim, audio=audio, fps=float(H3_FPS), match_tail=True
            )
        visible = max(1, int(chunk.get("visible_frame_count", int(frames.shape[0])) or int(frames.shape[0])))
        if int(frames.shape[0]) < visible:
            raise RuntimeError(f"R40 decoded {int(frames.shape[0])} frames, planner requires {visible}")
        frames = frames[:visible]
        audio = _fit_audio(audio, visible, float(H3_FPS))
        if isinstance(audio, dict):
            audio = dict(audio)
            audio["iamccs_flf_locked_audio_handles"] = bool(
                str(plan.get("continuation_mode", "")) == "flf_image_center_bridges"
                and str(plan.get("audio_mode", "")) == "h3_custom_audio_drive"
            )
        bridge = frames[-1:].detach().clone()
        report = (
            f"R40 selected decode | candidate={int(sampled_latent.get('iamccs_r40_candidate_index', 0)) + 1} | "
            f"seed={sampled_latent.get('iamccs_r40_candidate_seed', 0)} | trim={trim}f | "
            f"visible={visible}f | {cleanup}"
        )
        return frames, audio, bridge, sampled_latent, int(H3_FPS), report


def _stage2_lora(model, shot_lab):
    if not bool(shot_lab.get("stage2_lora_enabled", False)):
        return model, "off"
    name = str(shot_lab.get("stage2_lora_name", "") or "").strip()
    strength = float(shot_lab.get("stage2_lora_strength", 0.0) or 0.0)
    if not name or abs(strength) < 1e-8:
        return model, "off"
    path = folder_paths.get_full_path("loras", name)
    if not path:
        return model, f"off (missing {name})"
    compatible, reason = _h3_lora_compatibility(name)
    if not compatible:
        return model, f"off ({reason})"
    import nodes as comfy_nodes

    patched = comfy_nodes.LoraLoaderModelOnly().load_lora_model_only(
        model=model, lora_name=name, strength_model=strength
    )[0]
    return patched, f"Stage-2 only {name}@{strength:.2f}"


class IAMCCS_MiniMaxH3LatentUpresSamplingR40:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "segment_index": ("INT", {"forceInput": True}),
                "shot_lab": (R40_SHOT_LAB,),
            }
        }

    RETURN_TYPES = ("MODEL", "NOISE", "SAMPLER", "SIGMAS", "STRING")
    RETURN_NAMES = ("model", "noise", "sampler", "sigmas", "report")
    FUNCTION = "prepare"
    CATEGORY = CATEGORY

    def prepare(self, model, cine_linx, segment_index, shot_lab):
        from comfy_extras.nodes_custom_sampler import BasicScheduler, KSamplerSelect, RandomNoise
        from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift

        plan = _resolve_shotplan(cine_linx)
        upres = _upres_settings(plan)
        sampling = plan.get("sampling") if isinstance(plan.get("sampling"), dict) else {}
        seed = (int(shot_lab.get("stage2_seed", 0)) + int(segment_index) * int(shot_lab.get("stage2_seed_stride", 1))) & 0xFFFFFFFFFFFFFFFF
        base, turbo_report = _apply_turbo_lora(model, plan)
        base, lora2_report = _apply_secondary_lora(base, plan)
        base, stage2_lora_report = _stage2_lora(base, shot_lab)
        shifted = MiniMaxH3SigmaShift.execute(
            model=base,
            shift_video=float(sampling.get("shift_video", 12.0)),
            shift_audio=float(sampling.get("shift_audio", 3.0)),
        )[0]
        active, acceleration_report = _accelerate(shifted, plan)
        active, sparse_report = _apply_sparse(
            active,
            bool(shot_lab.get("stage2_sparse_enabled", False)),
            float(shot_lab.get("stage2_sparse_video_budget", 0.30)),
            bool(shot_lab.get("stage2_sparse_denser_edges", True)),
        )
        noise = RandomNoise.execute(noise_seed=seed)[0]
        sampler_name = str(upres.get("sampler", "er_sde") or "er_sde")
        sampler = KSamplerSelect.execute(sampler_name=sampler_name)[0]
        preset = str(shot_lab.get("stage2_sigma_preset", "seedhunter_3") or "seedhunter_3")
        manual = {
            "seedhunter_3": [0.9035, 0.6316, 0.3158, 0.0],
            "seedhunter_4": [0.9035, 0.8000, 0.6316, 0.3158, 0.0],
            "seedhunter_5": [0.9231, 0.8780, 0.8000, 0.6316, 0.3158, 0.0],
        }
        if preset in manual:
            sigmas = torch.tensor(manual[preset], dtype=torch.float32)
            sigma_report = f"manual {preset} ({len(manual[preset]) - 1} steps)"
        else:
            sigmas = BasicScheduler.execute(
                model=active,
                scheduler=str(upres.get("scheduler", "simple") or "simple"),
                steps=max(1, int(upres.get("steps", 2) or 2)),
                denoise=max(0.0, min(1.0, float(upres.get("denoise", 0.25) or 0.0))),
            )[0]
            sigma_report = "authored scheduler"
        report = (
            f"R40 Stage-2 | isolated seed={seed} | sigmas={sigma_report} | sampler={sampler_name} | "
            f"turbo={turbo_report} | lora2={lora2_report} | {stage2_lora_report} | "
            f"acceleration={acceleration_report} | sparse={sparse_report}"
        )
        LOG.info(report)
        return active, noise, sampler, sigmas, report


NODE_CLASS_MAPPINGS = {
    "IAMCCS_MiniMaxH3SeedCandidateR40": IAMCCS_MiniMaxH3SeedCandidateR40,
    "IAMCCS_MiniMaxH3ShotLabControlR40": IAMCCS_MiniMaxH3ShotLabControlR40,
    "IAMCCS_MiniMaxH3TakeSelectControlR40": IAMCCS_MiniMaxH3TakeSelectControlR40,
    "IAMCCS_MiniMaxH3SeedSelectR40": IAMCCS_MiniMaxH3SeedSelectR40,
    "IAMCCS_MiniMaxH3SeedPreviewR40": IAMCCS_MiniMaxH3SeedPreviewR40,
    "IAMCCS_MiniMaxH3SelectedDecodeR40": IAMCCS_MiniMaxH3SelectedDecodeR40,
    "IAMCCS_MiniMaxH3LatentUpresSamplingR40": IAMCCS_MiniMaxH3LatentUpresSamplingR40,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3SeedCandidateR40": "MiniMax H3 R40 · Cacheable Seed Candidate",
    "IAMCCS_MiniMaxH3ShotLabControlR40": "IAMCCS H3 R40 · Downstream Shot Lab Control",
    "IAMCCS_MiniMaxH3TakeSelectControlR40": "IAMCCS H3 R40 · Cached Take Select Control",
    "IAMCCS_MiniMaxH3SeedSelectR40": "MiniMax H3 R40 · Cached Candidate Selector",
    "IAMCCS_MiniMaxH3SeedPreviewR40": "MiniMax H3 R40 · Candidate Preview",
    "IAMCCS_MiniMaxH3SelectedDecodeR40": "MiniMax H3 R40 · Selected AV Decode",
    "IAMCCS_MiniMaxH3LatentUpresSamplingR40": "MiniMax H3 R40 · Isolated Stage-2 Sampling",
}

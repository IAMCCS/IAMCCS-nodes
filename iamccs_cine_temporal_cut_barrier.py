"""Optional Stage-1 temporal self-attention barrier for LTX video models.

This node is intentionally isolated from Shotboard V3 and its backend.  When
disabled it returns the input model unchanged.  When enabled it prevents noisy
video tokens from attending across selected Shotboard cut boundaries while
keeping guide tokens shared and leaving audio/cross-modal attention untouched.
"""

from __future__ import annotations

import json
import math
import types
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import comfy.ldm.modules.attention


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _as_dict(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_int(value: Any, fallback: int = 0) -> int:
    try:
        return int(round(float(value)))
    except Exception:
        return fallback


def _is_cut_marker(segment: Dict[str, Any]) -> bool:
    text = " ".join(
        str(segment.get(key, "") or "").strip().lower()
        for key in ("transition", "camera", "cut", "edit", "note")
    )
    markers = ("hard_cut", "hard cut", "cut_to", "cut to", "direct_cut", "direct cut")
    return any(marker in text for marker in markers)


def _largest_remainder_lengths(pixel_lengths: Sequence[int], latent_frames: int) -> List[int]:
    """Map pixel-frame island lengths to causal latent-frame island lengths."""
    clean = [max(1, int(length)) for length in pixel_lengths]
    if not clean or latent_frames <= 0:
        return []
    exact = [length * latent_frames / sum(clean) for length in clean]
    result = [int(value) for value in exact]
    remaining = latent_frames - sum(result)
    order = sorted(range(len(exact)), key=lambda index: -(exact[index] - result[index]))
    for index in range(max(0, remaining)):
        result[order[index % len(order)]] += 1
    for index, value in enumerate(result):
        if value >= 1:
            continue
        donor = max(range(len(result)), key=lambda candidate: result[candidate])
        if result[donor] > 1:
            result[donor] -= 1
            result[index] = 1
    return result


def _extract_plan(cine_linx: Any, boundary_mode: str) -> Dict[str, Any]:
    root = _as_dict(cine_linx)
    resources = _as_dict(root.get("resources"))
    outputs = _as_dict(root.get("outputs"))
    payload = _as_dict(resources.get("cine_payload"))

    segments = payload.get("visual_segments")
    if not isinstance(segments, list):
        raw_segments = resources.get("cine_visual_segments_json")
        try:
            segments = json.loads(raw_segments) if isinstance(raw_segments, str) else []
        except Exception:
            segments = []
    segments = [
        dict(segment)
        for segment in segments
        if isinstance(segment, dict)
        and str(segment.get("type", "image") or "image").lower() != "audio"
        and not bool(segment.get("placeholder", False))
    ]
    segments.sort(key=lambda item: _as_int(item.get("start", item.get("frame", 0)), 0))

    max_frames = _as_int(
        resources.get("cine_max_frames", outputs.get("max_frames", payload.get("max_frames", 0))),
        0,
    )
    width = _as_int(
        resources.get("cine_image_width", outputs.get("width", payload.get("image_width", 0))),
        0,
    )
    height = _as_int(
        resources.get("cine_image_height", outputs.get("height", payload.get("image_height", 0))),
        0,
    )

    if max_frames <= 0 or width <= 0 or height <= 0 or len(segments) < 2:
        return {
            "valid": False,
            "reason": "cine_linx needs max_frames, image dimensions, and at least two visual slots",
        }

    boundaries = [0]
    for segment in segments[1:]:
        start = max(0, min(max_frames, _as_int(segment.get("start", segment.get("frame", 0)), 0)))
        if start <= 0 or start >= max_frames:
            continue
        if boundary_mode == "every_visual_slot" or _is_cut_marker(segment):
            boundaries.append(start)
    boundaries.append(max_frames)
    boundaries = sorted(set(boundaries))
    pixel_lengths = [
        boundaries[index + 1] - boundaries[index]
        for index in range(len(boundaries) - 1)
        if boundaries[index + 1] > boundaries[index]
    ]
    if len(pixel_lengths) < 2:
        return {"valid": False, "reason": "no active cut boundaries found"}

    # LTX causal temporal compression and spatial patch geometry.
    latent_frames = max(1, ((max_frames - 1) // 8) + 1)
    tokens_per_frame = max(1, math.ceil(width / 32) * math.ceil(height / 32))
    latent_lengths = _largest_remainder_lengths(pixel_lengths, latent_frames)
    return {
        "valid": True,
        "max_frames": max_frames,
        "width": width,
        "height": height,
        "latent_frames": latent_frames,
        "tokens_per_frame": tokens_per_frame,
        "latent_lengths": latent_lengths,
        "pixel_boundaries": boundaries,
    }


def _slice_mask(mask: Any, q_start: int, q_end: int, k_start: int, k_end: int, guide_start: int):
    if mask is None:
        return None
    row_slice = mask[..., q_start:q_end, :]
    shot_columns = row_slice[..., k_start:k_end]
    if guide_start < mask.shape[-1]:
        guide_columns = row_slice[..., guide_start:]
        return torch.cat((shot_columns, guide_columns), dim=-1)
    return shot_columns


def _attention(q, k, v, module, mask, transformer_options):
    if mask is None:
        return comfy.ldm.modules.attention.optimized_attention(
            q,
            k,
            v,
            module.heads,
            attn_precision=module.attn_precision,
            transformer_options=transformer_options,
        )
    return comfy.ldm.modules.attention.optimized_attention_masked(
        q,
        k,
        v,
        module.heads,
        mask,
        attn_precision=module.attn_precision,
        transformer_options=transformer_options,
    )


def _temporal_barrier_forward(
    module,
    original_forward,
    plan,
    barrier_strength,
    x,
    context=None,
    mask=None,
    pe=None,
    k_pe=None,
    transformer_options=None,
):
    transformer_options = transformer_options or {}
    if context is not None:
        return original_forward(
            x,
            context=context,
            mask=mask,
            pe=pe,
            k_pe=k_pe,
            transformer_options=transformer_options,
        )

    latent_frames = int(plan["latent_frames"])
    tokens_per_frame = int(plan["tokens_per_frame"])
    noisy_tokens = latent_frames * tokens_per_frame
    total_tokens = int(x.shape[1])
    if noisy_tokens <= 0 or total_tokens < noisy_tokens:
        if not plan["runtime"].get("warned_geometry"):
            print(
                "[IAMCCS TemporalCutBarrier] fallback: runtime token geometry "
                f"total={total_tokens} expected_noisy={noisy_tokens}"
            )
            plan["runtime"]["warned_geometry"] = True
        return original_forward(
            x,
            context=context,
            mask=mask,
            pe=pe,
            k_pe=k_pe,
            transformer_options=transformer_options,
        )

    from comfy.ldm.lightricks.model import apply_rotary_emb

    q = module.q_norm(module.to_q(x))
    k = module.k_norm(module.to_k(x))
    v = module.to_v(x)
    if pe is not None:
        q = apply_rotary_emb(q, pe)
        k = apply_rotary_emb(k, pe)

    guide_start = noisy_tokens
    guide_k = k[:, guide_start:]
    guide_v = v[:, guide_start:]
    isolated = torch.empty_like(v)
    cursor_frames = 0
    for length in plan["latent_lengths"]:
        frame_end = min(latent_frames, cursor_frames + int(length))
        q_start = cursor_frames * tokens_per_frame
        q_end = frame_end * tokens_per_frame
        if q_end <= q_start:
            continue
        shot_k = k[:, q_start:q_end]
        shot_v = v[:, q_start:q_end]
        if guide_start < total_tokens:
            shot_k = torch.cat((shot_k, guide_k), dim=1)
            shot_v = torch.cat((shot_v, guide_v), dim=1)
        shot_mask = _slice_mask(mask, q_start, q_end, q_start, q_end, guide_start)
        isolated[:, q_start:q_end] = _attention(
            q[:, q_start:q_end],
            shot_k,
            shot_v,
            module,
            shot_mask,
            transformer_options,
        )
        cursor_frames = frame_end

    # Any rounding remainder belongs to the final island.
    remainder_start = cursor_frames * tokens_per_frame
    if remainder_start < noisy_tokens:
        shot_k = k[:, remainder_start:noisy_tokens]
        shot_v = v[:, remainder_start:noisy_tokens]
        if guide_start < total_tokens:
            shot_k = torch.cat((shot_k, guide_k), dim=1)
            shot_v = torch.cat((shot_v, guide_v), dim=1)
        shot_mask = _slice_mask(mask, remainder_start, noisy_tokens, remainder_start, noisy_tokens, guide_start)
        isolated[:, remainder_start:noisy_tokens] = _attention(
            q[:, remainder_start:noisy_tokens],
            shot_k,
            shot_v,
            module,
            shot_mask,
            transformer_options,
        )

    # Guide tokens still see the complete video so reference-image conditioning
    # remains available to every shot island.
    if guide_start < total_tokens:
        guide_mask = mask[..., guide_start:, :] if mask is not None else None
        isolated[:, guide_start:] = _attention(
            q[:, guide_start:],
            k,
            v,
            module,
            guide_mask,
            transformer_options,
        )

    if module.to_gate_logits is not None:
        gate_logits = module.to_gate_logits(x)
        batch, token_count, _ = isolated.shape
        isolated = isolated.view(batch, token_count, module.heads, module.dim_head)
        isolated = isolated * (2.0 * torch.sigmoid(gate_logits)).unsqueeze(-1)
        isolated = isolated.view(batch, token_count, module.heads * module.dim_head)
    isolated = module.to_out(isolated)

    if barrier_strength < 0.999:
        regular = original_forward(
            x,
            context=context,
            mask=mask,
            pe=pe,
            k_pe=k_pe,
            transformer_options=transformer_options,
        )
        isolated = torch.lerp(regular, isolated, float(barrier_strength))

    if not plan["runtime"].get("logged"):
        guide_tokens = total_tokens - noisy_tokens
        print(
            "[IAMCCS TemporalCutBarrier] ACTIVE "
            f"islands={plan['latent_lengths']} noisy_tokens={noisy_tokens} "
            f"guide_tokens={guide_tokens} strength={barrier_strength:.3f}"
        )
        plan["runtime"]["logged"] = True
    return isolated


class _TemporalBarrierPatch:
    def __init__(self, original_forward, plan, barrier_strength):
        self.original_forward = original_forward
        self.plan = plan
        self.barrier_strength = barrier_strength

    def __get__(self, obj, objtype=None):
        original_forward = self.original_forward
        plan = self.plan
        barrier_strength = self.barrier_strength

        def wrapped(self_module, *args, **kwargs):
            return _temporal_barrier_forward(
                self_module,
                original_forward,
                plan,
                barrier_strength,
                *args,
                **kwargs,
            )

        return types.MethodType(wrapped, obj)


class IAMCCS_CineTemporalCutBarrier:
    """Optional single-generation hard-cut experiment for Stage 1."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "enabled": ("BOOLEAN", {"default": False}),
                "boundary_mode": (
                    ["marked_cuts", "every_visual_slot"],
                    {"default": "marked_cuts"},
                ),
                "barrier_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
                "first_block": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
                "last_block": ("INT", {"default": -1, "min": -1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/Cine/03 Experimental"

    def apply(
        self,
        model,
        cine_linx,
        enabled=False,
        boundary_mode="marked_cuts",
        barrier_strength=1.0,
        first_block=0,
        last_block=-1,
    ):
        if not enabled:
            return (model, "Temporal Cut Barrier: OFF (model unchanged)")

        plan = _extract_plan(cine_linx, str(boundary_mode))
        if not plan.get("valid"):
            reason = str(plan.get("reason", "invalid cine_linx plan"))
            print(f"[IAMCCS TemporalCutBarrier] bypass: {reason}")
            return (model, f"Temporal Cut Barrier: BYPASS - {reason}")

        model_clone = model.clone()
        diffusion_model = model_clone.get_model_object("diffusion_model")
        blocks = getattr(diffusion_model, "transformer_blocks", None)
        key_prefix = "diffusion_model.transformer_blocks"
        if blocks is None and hasattr(diffusion_model, "transformer"):
            blocks = getattr(diffusion_model.transformer, "transformer_blocks", None)
            key_prefix = "diffusion_model.transformer.transformer_blocks"
        if blocks is None:
            reason = f"unsupported diffusion model {type(diffusion_model).__name__}"
            print(f"[IAMCCS TemporalCutBarrier] bypass: {reason}")
            return (model, f"Temporal Cut Barrier: BYPASS - {reason}")

        plan["runtime"] = {}
        start = max(0, int(first_block))
        end = len(blocks) - 1 if int(last_block) < 0 else min(len(blocks) - 1, int(last_block))
        patched = 0
        for index, block in enumerate(blocks):
            if index < start or index > end:
                continue
            attn = getattr(block, "attn1", None)
            if attn is None:
                continue
            key = f"{key_prefix}.{index}.attn1.forward"
            if key in getattr(model_clone, "object_patches", {}):
                reason = f"attn1 already patched at block {index}"
                print(f"[IAMCCS TemporalCutBarrier] bypass: {reason}")
                return (model, f"Temporal Cut Barrier: BYPASS - {reason}")
            original_forward = attn.forward
            descriptor = _TemporalBarrierPatch(original_forward, plan, float(barrier_strength))
            model_clone.add_object_patch(key, descriptor.__get__(attn, attn.__class__))
            patched += 1

        if patched == 0:
            return (model, "Temporal Cut Barrier: BYPASS - no compatible video attention blocks")

        report = (
            "Temporal Cut Barrier: ACTIVE | "
            f"mode={boundary_mode} | cuts={len(plan['latent_lengths']) - 1} | "
            f"latent_islands={plan['latent_lengths']} | blocks={start}-{end} | "
            f"strength={float(barrier_strength):.2f}"
        )
        print(f"[IAMCCS TemporalCutBarrier] configured {report}")
        return (model_clone, report)


import logging
import sys
from collections import OrderedDict

import comfy.sd
import comfy.hooks

from .iamccs_wan_lora_schedule import IAMCCS_WanLoRASchedule


_log = logging.getLogger("IAMCCS.LoRA.Hooks")

_ANSI_GREEN = "\033[92m"
_ANSI_YELLOW = "\033[93m"
_ANSI_RESET = "\033[0m"
_MODEL_BANK_TYPE = "IAMCCS_WAN_MODEL_BANK"

# Cache: (id(model_patcher), lora_fingerprint) → patched_model_patcher
# Avoids re-cloning and re-patching when the LoRA stack is identical across loop iterations,
# which ensures model management reuses the same loaded model (no unload/reload overhead).
_MODEL_CACHE: OrderedDict = OrderedDict()
_MAX_MODEL_CACHE = 8


def _flush_log_handlers():
    seen_handlers = set()
    manager = getattr(logging.Logger, "manager", None)
    logger_dict = getattr(manager, "loggerDict", {}) if manager is not None else {}

    for logger_name in ("IAMCCS.LoRA.Hooks", None):
        logger = logging.getLogger(logger_name) if logger_name else logging.getLogger()
        for handler in getattr(logger, "handlers", []):
            seen_handlers.add(handler)

    for logger in logger_dict.values():
        if isinstance(logger, logging.Logger):
            for handler in getattr(logger, "handlers", []):
                seen_handlers.add(handler)

    for handler in seen_handlers:
        try:
            handler.flush()
        except Exception:
            pass

    for stream in (getattr(sys, "stdout", None), getattr(sys, "stderr", None)):
        if stream is None:
            continue
        try:
            stream.flush()
        except Exception:
            pass


def _lora_fingerprint(lora_stack):
    """Hashable fingerprint of a lora stack: (name, strength) pairs sorted."""
    return tuple(sorted(
        (str(e.get("name") or ""), float(e.get("strength") or 0.0))
        for e in (lora_stack or [])
    ))

# ─── helpers shared with stack ────────────────────────────────────────────────

def _entry_summary(entry):
    name = entry.get("name", "unnamed")
    strength = entry.get("strength", 0.0)
    origin = entry.get("origin", "manual")
    gen = entry.get("generation_index")
    slot = entry.get("slot")
    rule = entry.get("rule") or ""
    extras = [origin]
    if gen is not None:
        extras.append(f"gen={gen}")
    if slot is not None:
        extras.append(f"slot={int(slot):02d}")
    if rule:
        extras.append(rule)
    return f"{name}({strength}) [{ ' | '.join(extras) }]"


def _stack_summary(lora_stack):
    if not lora_stack:
        return "none"
    parts = []
    for entry in lora_stack:
        name = entry.get("name", "unnamed")
        strength = entry.get("strength", 0.0)
        origin = entry.get("_iamccs_lora_origin", "manual")
        gen = entry.get("_iamccs_generation_index")
        slot = entry.get("_iamccs_schedule_slot")
        rule = entry.get("_iamccs_schedule_rule") or ""
        extras = [origin]
        if gen is not None:
            extras.append(f"gen={gen}")
        if slot is not None:
            extras.append(f"slot={int(slot):02d}")
        if rule:
            extras.append(rule)
        parts.append(f"{name}({strength}) [{' | '.join(extras)}]")
    return "; ".join(parts)


def _origin_names(lora_stack, origin):
    names = []
    for entry in lora_stack or []:
        if str(entry.get("_iamccs_lora_origin") or "manual") != origin:
            continue
        name = entry.get("name", "unnamed")
        strength = entry.get("strength", 0.0)
        slot = entry.get("_iamccs_schedule_slot")
        rule = entry.get("_iamccs_schedule_rule") or origin
        if slot is not None:
            names.append(f"slot={int(slot):02d}:{name}({strength}) [{rule}]")
        else:
            names.append(f"{name}({strength})")
    return "; ".join(names) if names else "none"


def _origin_count(lora_stack, origin):
    return sum(
        1
        for entry in (lora_stack or [])
        if str(entry.get("_iamccs_lora_origin") or "manual") == origin
    )


def _inject_debug_report(lora_stack):
    total_count = len(lora_stack or [])
    default_summary = _origin_names(lora_stack, "default")
    scheduled_summary = _origin_names(lora_stack, "scheduled")
    manual_summary = _origin_names(lora_stack, "manual")
    return {
        "total_count": total_count,
        "default_count": _origin_count(lora_stack, "default"),
        "scheduled_count": _origin_count(lora_stack, "scheduled"),
        "manual_count": _origin_count(lora_stack, "manual"),
        "default_summary": default_summary,
        "scheduled_summary": scheduled_summary,
        "manual_summary": manual_summary,
        "stack_summary": _stack_summary(lora_stack),
        "has_default": default_summary != "none",
        "has_scheduled": scheduled_summary != "none",
    }


def _stack_context(lora_stack):
    if not lora_stack:
        return ""
    prompt_ids = sorted({str(e.get("_iamccs_prompt_id") or "") for e in lora_stack if e.get("_iamccs_prompt_id")})
    node_ids = sorted({str(e.get("_iamccs_schedule_node_id") or "") for e in lora_stack if e.get("_iamccs_schedule_node_id")})
    log_prefixes = sorted({str(e.get("_iamccs_schedule_log_prefix") or "") for e in lora_stack if e.get("_iamccs_schedule_log_prefix")})
    gen_idxs = sorted({int(e["_iamccs_generation_index"]) for e in lora_stack if e.get("_iamccs_generation_index") is not None})
    parts = []
    if prompt_ids:
        parts.append(f"prompt={','.join(prompt_ids)}")
    if node_ids:
        parts.append(f"schedule_node={','.join(node_ids)}")
    if log_prefixes:
        parts.append(f"schedule={','.join(log_prefixes)}")
    if gen_idxs:
        parts.append(f"generation={','.join(str(g) for g in gen_idxs)}")
    return " | ".join(parts)


def _prompt_scope_from_lora_stack(lora_stack):
    prompt_ids = sorted({
        str(entry.get("_iamccs_prompt_id") or "")
        for entry in (lora_stack or [])
        if entry.get("_iamccs_prompt_id")
    })
    return ",".join(prompt_ids) if prompt_ids else "no-prompt"


def _extract_generation_index_from_conditioning(*conditioning_values):
    for conditioning in conditioning_values:
        if not isinstance(conditioning, list):
            continue
        for item in conditioning:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            meta = item[1]
            if not isinstance(meta, dict):
                continue
            for key in (
                "_iamccs_generation_index",
                "_iamccs_segment_index",
                "generation_index",
                "segment_index",
            ):
                value = meta.get(key)
                if value is None:
                    continue
                try:
                    return max(0, int(value)), key
                except Exception:
                    continue
    return 0, "default:0"


def _conditioning_flags(*conditioning_values):
    flags = {
        "clip_vision": False,
        "reference_latents": False,
        "concat_latent_image": False,
        "concat_mask": False,
    }
    for conditioning in conditioning_values:
        if not isinstance(conditioning, list):
            continue
        for item in conditioning:
            if not isinstance(item, (list, tuple)) or len(item) < 2:
                continue
            meta = item[1]
            if not isinstance(meta, dict):
                continue
            flags["clip_vision"] = flags["clip_vision"] or (meta.get("clip_vision_output") is not None)
            flags["reference_latents"] = flags["reference_latents"] or bool(meta.get("reference_latents"))
            flags["concat_latent_image"] = flags["concat_latent_image"] or (meta.get("concat_latent_image") is not None)
            flags["concat_mask"] = flags["concat_mask"] or (meta.get("concat_mask") is not None)
    return flags


def _select_model_from_bank(model_bank, generation_index, log_prefix):
    generation_count = int((model_bank or {}).get("generation_count") or 1)
    generation_index = max(0, min(int(generation_index), generation_count - 1))
    generation_to_fingerprint = (model_bank or {}).get("generation_to_fingerprint") or {}
    fingerprints_to_model = (model_bank or {}).get("fingerprints_to_model") or {}
    reports = (model_bank or {}).get("reports") or {}

    fingerprint = generation_to_fingerprint.get(generation_index)
    if fingerprint is None:
        error = (
            f"[{log_prefix}][selector-check] generation={generation_index} missing fingerprint in model bank"
        )
        _log.error(error)
        _flush_log_handlers()
        raise RuntimeError(error)

    model_out = fingerprints_to_model.get(fingerprint)
    if model_out is None:
        error = (
            f"[{log_prefix}][selector-check] generation={generation_index} missing model variant for fingerprint={fingerprint}"
        )
        _log.error(error)
        _flush_log_handlers()
        raise RuntimeError(error)

    return generation_index, model_out, reports.get(generation_index, {})


def _apply_lora_stack_to_model(model, lora_stack):
    prompt_scope = _prompt_scope_from_lora_stack(lora_stack)
    cache_key = (prompt_scope, id(model), _lora_fingerprint(lora_stack))
    inject_report = _inject_debug_report(lora_stack)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        _log.info(
            "[IAMCCS_ApplyLoRAHooksToConditioning] cache hit — reusing patched model (prompt=%s, stack unchanged)",
            prompt_scope,
        )
        _log.info(
            "[IAMCCS_LoRAInjectCheck] mode=model_bank_cache_hit | prompt=%s | total=%s | always_on_present=%s | always_on=%s | scheduled_present=%s | scheduled=%s | stack=%s | backend=load_lora_for_models",
            prompt_scope,
            inject_report["total_count"],
            "YES" if inject_report["has_default"] else "NO",
            inject_report["default_summary"],
            "YES" if inject_report["has_scheduled"] else "NO",
            inject_report["scheduled_summary"],
            inject_report["stack_summary"],
        )
        _flush_log_handlers()
        return cached, True

    model_out = model
    patched_names = []
    for entry in lora_stack or []:
        sd = entry.get("state_dict")
        if not isinstance(sd, dict) or not sd:
            _log.warning("[IAMCCS_ApplyLoRAHooksToConditioning] skipped '%s': no state_dict", entry.get("name"))
            continue
        strength = float(entry.get("strength", 1.0) or 1.0)
        model_out, _ = comfy.sd.load_lora_for_models(model_out, None, sd, strength, 0.0)
        patched_names.append(f"{entry.get('name')}({strength:.3f})")

    if patched_names:
        _log.info(
            "[IAMCCS_ApplyLoRAHooksToConditioning] patched %d LoRA(s) for prompt=%s: %s",
            len(patched_names),
            prompt_scope,
            "; ".join(patched_names),
        )
        _log.info(
            "[IAMCCS_LoRAInjectCheck] mode=model_patch_apply | prompt=%s | total=%s | always_on_present=%s | always_on=%s | scheduled_present=%s | scheduled=%s | stack=%s | backend=load_lora_for_models",
            prompt_scope,
            inject_report["total_count"],
            "YES" if inject_report["has_default"] else "NO",
            inject_report["default_summary"],
            "YES" if inject_report["has_scheduled"] else "NO",
            inject_report["scheduled_summary"],
            inject_report["stack_summary"],
        )
    else:
        _log.warning("[IAMCCS_ApplyLoRAHooksToConditioning] no LoRAs were patched (empty/invalid stack)")
    _flush_log_handlers()

    _MODEL_CACHE[cache_key] = model_out
    _MODEL_CACHE.move_to_end(cache_key)
    while len(_MODEL_CACHE) > _MAX_MODEL_CACHE:
        _MODEL_CACHE.popitem(last=False)
    return model_out, False


# ─── Schedule node (outputs LORA stack) ───────────────────────────────────────

class IAMCCS_WanLoRAHookSchedule:
    """Schedule wrapper that outputs a LORA stack (same as IAMCCS_WanLoRASchedule).
    Kept as a separate node type so JS UI / workflows can reference it by name."""

    @classmethod
    def INPUT_TYPES(cls):
        return IAMCCS_WanLoRASchedule.INPUT_TYPES()

    RETURN_TYPES = ("LORA", "INT", "STRING", "IAMCCS_WAN_LORA_LINX")
    RETURN_NAMES = ("lora", "active_slots", "report", "linx")
    FUNCTION = "schedule"
    CATEGORY = "IAMCCS/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check_lazy_status(self, generation_index, **kwargs):
        # Two-pass lazy protocol: lazy=True in INPUT_TYPES breaks the static
        # validation cycle (execution.py L878). This method forces the executor
        # to evaluate the upstream (forLoopStart) before calling schedule(),
        # so generation_index is always the real iteration index, never None.
        if generation_index is None:
            return ["generation_index"]
        return []

    def schedule(self, generation_index, log_prefix, model_type="flow", default_lora=None, linx=None, unique_id=None, **kwargs):
        return IAMCCS_WanLoRASchedule().schedule(
            generation_index=generation_index,
            log_prefix=log_prefix,
            model_type=model_type,
            default_lora=default_lora,
            linx=linx,
            unique_id=unique_id,
            **kwargs,
        )


# ─── Apply node (MODEL + LORA → MODEL + CONDITIONING passthrough) ─────────────

class IAMCCS_ApplyLoRAHooksToConditioning:
    """Apply LoRA schedule to MODEL (add_patches style, not hooks) and pass through conditioning.

    Uses comfy.sd.load_lora_for_models — identical to the LoRA Loader node in ComfyUI core.
    This avoids the per-step hook weight patching that causes ~15 min stalls on 14B offloaded models.
    The MODEL output should go to the KSampler model input; positive/negative are passed through unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "lora": ("LORA",),
            },
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply(self, model, positive, negative, lora):
        stack_summary = _stack_summary(lora)
        scheduled_active = "YES" if any(
            str(e.get("_iamccs_lora_origin") or "manual") == "scheduled" for e in (lora or [])
        ) else "NO"
        default_summary = _origin_names(lora, "default")
        scheduled_summary = _origin_names(lora, "scheduled")
        ctx = _stack_context(lora)

        if ctx:
            _log.info("[IAMCCS_ApplyLoRAHooksToConditioning] apply request | %s", ctx)
        _log.info("[IAMCCS_ApplyLoRAHooksToConditioning] active_stack=%s", stack_summary)
        _log.info("[IAMCCS_ApplyLoRAHooksToConditioning] scheduled_active=%s", scheduled_active)
        _log.info("[IAMCCS_ApplyLoRAHooksToConditioning] applied_default=%s", default_summary)
        _log.info("[IAMCCS_ApplyLoRAHooksToConditioning] applied_scheduled=%s", scheduled_summary)
        _flush_log_handlers()

        model_out, _cached = _apply_lora_stack_to_model(model, lora)

        report = (
            f"scheduled_active={scheduled_active} | "
            f"applied_default={default_summary} | "
            f"applied_scheduled={scheduled_summary}"
        )
        return (model_out, positive, negative, report)


class IAMCCS_BuildScheduledWanModelBank:
    """Build all scheduled WAN model variants outside the loop.

    The loop then selects a prebuilt model by generation metadata instead of
    patching or scheduling LoRAs inside the model branch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = IAMCCS_WanLoRASchedule.INPUT_TYPES()
        base_required = dict(base.get("required", {}))
        base_required.pop("generation_index", None)

        required = {
            "model": ("MODEL",),
            "generation_count": ("INT", {"default": 4, "min": 1, "max": 512, "step": 1}),
        }
        required.update(base_required)

        optional = dict(base.get("optional", {}))
        hidden = {"unique_id": "UNIQUE_ID"}
        return {"required": required, "optional": optional, "hidden": hidden}

    RETURN_TYPES = (_MODEL_BANK_TYPE, "STRING", "IAMCCS_WAN_LORA_LINX")
    RETURN_NAMES = ("model_bank", "report", "linx")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def build(self, model, generation_count, log_prefix, model_type="flow", default_lora=None, linx=None, unique_id=None, **kwargs):
        generation_count = max(1, int(generation_count or 1))
        fingerprints_to_model = {}
        generation_to_fingerprint = {}
        generation_reports = {}
        linx_payload_out = None

        for generation_index in range(generation_count):
            lora_stack, _active_slots, schedule_report, linx_payload = IAMCCS_WanLoRASchedule().schedule(
                generation_index=generation_index,
                log_prefix=log_prefix,
                model_type=model_type,
                default_lora=default_lora,
                linx=linx,
                unique_id=unique_id,
                **kwargs,
            )
            linx_payload_out = linx_payload
            fingerprint = _lora_fingerprint(lora_stack)
            inject_report = _inject_debug_report(lora_stack)
            generation_to_fingerprint[generation_index] = fingerprint
            generation_reports[generation_index] = {
                "schedule": schedule_report,
                "default": inject_report["default_summary"],
                "scheduled": _origin_names(lora_stack, "scheduled"),
                "stack": _stack_summary(lora_stack),
                "always_on_present": inject_report["has_default"],
                "scheduled_present": inject_report["has_scheduled"],
                "fingerprint": repr(fingerprint),
            }
            _log.info(
                "[IAMCCS_BuildScheduledWanModelBank][inject-check] generation=%s/%s | log_prefix=%s | always_on_present=%s | always_on=%s | scheduled_present=%s | scheduled=%s | stack=%s | fingerprint=%s",
                generation_index,
                generation_count - 1,
                log_prefix,
                "YES" if inject_report["has_default"] else "NO",
                inject_report["default_summary"],
                "YES" if inject_report["has_scheduled"] else "NO",
                inject_report["scheduled_summary"],
                inject_report["stack_summary"],
                repr(fingerprint),
            )
            if fingerprint not in fingerprints_to_model:
                model_variant, _cached = _apply_lora_stack_to_model(model, lora_stack)
                fingerprints_to_model[fingerprint] = model_variant

        bank = {
            "generation_count": generation_count,
            "generation_to_fingerprint": generation_to_fingerprint,
            "fingerprints_to_model": fingerprints_to_model,
            "reports": generation_reports,
            "log_prefix": str(log_prefix or "WAN LoRA schedule"),
        }

        summary_parts = []
        for generation_index in range(generation_count):
            info = generation_reports[generation_index]
            summary_parts.append(
                f"g{generation_index}:always_on={info['default']} | scheduled={info['scheduled'] if info['scheduled'] != 'none' else 'default_only'}"
            )
        report = (
            f"generation_count={generation_count} | unique_variants={len(fingerprints_to_model)} | "
            f"schedule_map={' ; '.join(summary_parts)}"
        )
        _log.info("[IAMCCS_BuildScheduledWanModelBank] %s", report)
        _flush_log_handlers()
        return (bank, report, linx_payload_out)


class IAMCCS_SelectScheduledWanModelFromConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_bank": (_MODEL_BANK_TYPE,),
                "positive": ("CONDITIONING", {"lazy": True}),
                "negative": ("CONDITIONING", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("model", "positive", "negative", "generation_index", "report")
    FUNCTION = "select"
    CATEGORY = "IAMCCS/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check_lazy_status(self, positive=None, negative=None, **kwargs):
        needed = []
        if positive is None:
            needed.append("positive")
        if negative is None:
            needed.append("negative")
        return needed

    def select(self, model_bank, positive, negative):
        generation_index, generation_source = _extract_generation_index_from_conditioning(positive, negative)
        log_prefix = str((model_bank or {}).get("log_prefix") or "WAN LoRA schedule")
        generation_index, model_out, info = _select_model_from_bank(model_bank, generation_index, log_prefix)
        cond_flags = _conditioning_flags(positive, negative)
        scheduled = info.get("scheduled", "none")
        always_on = info.get("default", "none")
        color = _ANSI_GREEN if scheduled != "none" else _ANSI_YELLOW
        report = (
            f"generation_index={generation_index} | source={generation_source} | "
            f"always_on={always_on} | scheduled={scheduled} | schedule={info.get('schedule', 'unknown')}"
        )
        _log.info(
            "%s[IAMCCS_SelectScheduledWanModelFromConditioning] generation=%s source=%s | clip_vision=%s reference_latents=%s concat_latent=%s concat_mask=%s | always_on=%s | scheduled=%s%s",
            color,
            generation_index,
            generation_source,
            "YES" if cond_flags["clip_vision"] else "NO",
            "YES" if cond_flags["reference_latents"] else "NO",
            "YES" if cond_flags["concat_latent_image"] else "NO",
            "YES" if cond_flags["concat_mask"] else "NO",
            always_on,
            scheduled,
            _ANSI_RESET,
        )
        _flush_log_handlers()
        return (model_out, positive, negative, generation_index, report)


class IAMCCS_SelectScheduledWanModelPairFromConditioning:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "high_model_bank": (_MODEL_BANK_TYPE,),
                "low_model_bank": (_MODEL_BANK_TYPE,),
                "positive": ("CONDITIONING", {"lazy": True}),
                "negative": ("CONDITIONING", {"lazy": True}),
            }
        }

    RETURN_TYPES = ("MODEL", "MODEL", "CONDITIONING", "CONDITIONING", "INT", "STRING")
    RETURN_NAMES = ("high_model", "low_model", "positive", "negative", "generation_index", "report")
    FUNCTION = "select_pair"
    CATEGORY = "IAMCCS/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def check_lazy_status(self, positive=None, negative=None, **kwargs):
        needed = []
        if positive is None:
            needed.append("positive")
        if negative is None:
            needed.append("negative")
        return needed

    def select_pair(self, high_model_bank, low_model_bank, positive, negative):
        generation_index, generation_source = _extract_generation_index_from_conditioning(positive, negative)
        high_log_prefix = str((high_model_bank or {}).get("log_prefix") or "WAN LoRA schedule HIGH")
        low_log_prefix = str((low_model_bank or {}).get("log_prefix") or "WAN LoRA schedule LOW")
        generation_index, high_model_out, high_info = _select_model_from_bank(
            high_model_bank,
            generation_index,
            high_log_prefix,
        )
        generation_index, low_model_out, low_info = _select_model_from_bank(
            low_model_bank,
            generation_index,
            low_log_prefix,
        )

        cond_flags = _conditioning_flags(positive, negative)
        high_default = high_info.get("default", "none")
        low_default = low_info.get("default", "none")
        high_scheduled = high_info.get("scheduled", "none")
        low_scheduled = low_info.get("scheduled", "none")
        report = (
            f"generation_index={generation_index} | source={generation_source} | "
            f"high_always_on={high_default} | low_always_on={low_default} | "
            f"high_scheduled={high_scheduled} | low_scheduled={low_scheduled}"
        )
        _log.info(
            "%s[IAMCCS_SelectScheduledWanModelPairFromConditioning] generation=%s source=%s | clip_vision=%s reference_latents=%s concat_latent=%s concat_mask=%s | high_always_on=%s | low_always_on=%s | high_scheduled=%s | low_scheduled=%s%s",
            _ANSI_GREEN if (high_scheduled != "none" or low_scheduled != "none") else _ANSI_YELLOW,
            generation_index,
            generation_source,
            "YES" if cond_flags["clip_vision"] else "NO",
            "YES" if cond_flags["reference_latents"] else "NO",
            "YES" if cond_flags["concat_latent_image"] else "NO",
            "YES" if cond_flags["concat_mask"] else "NO",
            high_default,
            low_default,
            high_scheduled,
            low_scheduled,
            _ANSI_RESET,
        )
        _flush_log_handlers()
        return (high_model_out, low_model_out, positive, negative, generation_index, report)


class IAMCCS_ApplyScheduledWanLoRAFromConditioning:
    """Resolve generation index from conditioning metadata, then apply scheduled WAN LoRAs.

    This avoids wiring easy forLoopStart.index into the MODEL branch entirely.
    The effective generation index comes from conditioning metadata emitted by
    IAMCCS_WanIndexedPromptEncode and therefore stays synchronized with the
    prompt/segment that actually reaches the sampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        base = IAMCCS_WanLoRASchedule.INPUT_TYPES()
        base_required = dict(base.get("required", {}))
        base_required.pop("generation_index", None)

        required = {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
        }
        required.update(base_required)

        optional = dict(base.get("optional", {}))
        hidden = {"unique_id": "UNIQUE_ID"}
        return {"required": required, "optional": optional, "hidden": hidden}

    RETURN_TYPES = ("MODEL", "CONDITIONING", "CONDITIONING", "INT", "STRING", "IAMCCS_WAN_LORA_LINX")
    RETURN_NAMES = ("model", "positive", "negative", "generation_index", "report", "linx")
    FUNCTION = "apply_scheduled"
    CATEGORY = "IAMCCS/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def apply_scheduled(self, model, positive, negative, log_prefix, model_type="flow", default_lora=None, linx=None, unique_id=None, **kwargs):
        generation_index, generation_source = _extract_generation_index_from_conditioning(positive, negative)
        cond_flags = _conditioning_flags(positive, negative)
        lora, _active_slots, schedule_report, linx_payload = IAMCCS_WanLoRASchedule().schedule(
            generation_index=generation_index,
            log_prefix=log_prefix,
            model_type=model_type,
            default_lora=default_lora,
            linx=linx,
            unique_id=unique_id,
            **kwargs,
        )
        model_out, positive_out, negative_out, apply_report = IAMCCS_ApplyLoRAHooksToConditioning().apply(
            model=model,
            positive=positive,
            negative=negative,
            lora=lora,
        )
        report = (
            f"generation_index={int(generation_index)} | "
            f"schedule={schedule_report} | "
            f"apply={apply_report}"
        )
        scheduled_summary = _origin_names(lora, "scheduled")
        color = _ANSI_GREEN if scheduled_summary != "none" else _ANSI_YELLOW
        _log.info(
            "%s[IAMCCS_ApplyScheduledWanLoRAFromConditioning] generation=%s source=%s | clip_vision=%s reference_latents=%s concat_latent=%s concat_mask=%s | scheduled=%s%s",
            color,
            int(generation_index),
            generation_source,
            "YES" if cond_flags["clip_vision"] else "NO",
            "YES" if cond_flags["reference_latents"] else "NO",
            "YES" if cond_flags["concat_latent_image"] else "NO",
            "YES" if cond_flags["concat_mask"] else "NO",
            scheduled_summary,
            _ANSI_RESET,
        )
        _flush_log_handlers()
        return (model_out, positive_out, negative_out, int(generation_index), report, linx_payload)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAHookSchedule": IAMCCS_WanLoRAHookSchedule,
    "IAMCCS_ApplyLoRAHooksToConditioning": IAMCCS_ApplyLoRAHooksToConditioning,
    "IAMCCS_ApplyScheduledWanLoRAFromConditioning": IAMCCS_ApplyScheduledWanLoRAFromConditioning,
    "IAMCCS_BuildScheduledWanModelBank": IAMCCS_BuildScheduledWanModelBank,
    "IAMCCS_SelectScheduledWanModelFromConditioning": IAMCCS_SelectScheduledWanModelFromConditioning,
    "IAMCCS_SelectScheduledWanModelPairFromConditioning": IAMCCS_SelectScheduledWanModelPairFromConditioning,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAHookSchedule": "LoRA Schedule (WAN, hooks)",
    "IAMCCS_ApplyLoRAHooksToConditioning": "Apply LoRA Hooks to Conditioning",
    "IAMCCS_ApplyScheduledWanLoRAFromConditioning": "Apply Scheduled WAN LoRA From Conditioning",
    "IAMCCS_BuildScheduledWanModelBank": "Build Scheduled WAN Model Bank",
    "IAMCCS_SelectScheduledWanModelFromConditioning": "Select Scheduled WAN Model From Conditioning",
    "IAMCCS_SelectScheduledWanModelPairFromConditioning": "Select Scheduled WAN Model Pair From Conditioning",
}


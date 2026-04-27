import logging

import folder_paths

from .iamccs_wan_lora_schedule import (
    _DEFAULT_MODEL_TYPE,
    _LINX_TYPE,
    _annotate_lora_entry,
    _coerce_lora_stack,
    _load_internal_lora_entry,
    _preset_matches,
    _resolve_linx_name,
    _stack_names,
)
from .iamccs_wan_lora_stack import (
    IAMCCS_ModelWithLoRA,
    _lora_stack_debug_context,
    _lora_stack_debug_summary,
)


_log = logging.getLogger("IAMCCS.LoRA.RuntimeBridge")


def _runtime_slot_items(linx_payload):
    if not isinstance(linx_payload, dict):
        return []
    slot_map = linx_payload.get("slot_map") or {}
    if not isinstance(slot_map, dict):
        return []
    return sorted(slot_map.items())


def _coerce_generation_index(value) -> int:
    try:
        return max(0, int(value))
    except Exception:
        return 0


def _resolved_model_type(model_type_override: str, linx_payload) -> str:
    if model_type_override and model_type_override != "inherit":
        return str(model_type_override)
    if isinstance(linx_payload, dict):
        linx_model_type = str(linx_payload.get("model_type") or "").strip()
        if linx_model_type:
            return linx_model_type
    return _DEFAULT_MODEL_TYPE


class IAMCCS_WanLoRARuntimeBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "generation_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "inject_tag": ("STRING", {"default": "inject"}),
                "log_prefix": ("STRING", {"default": "WAN LoRA runtime bridge"}),
                "model_type_override": (["inherit", "wan2x", "flow", "standard"], {"default": "inherit"}),
            },
            "optional": {
                "default_lora": ("LORA",),
                "linx": (_LINX_TYPE,),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LORA", "INT", "STRING", "INT")
    RETURN_NAMES = ("lora", "active_slots", "report", "generation_index")
    FUNCTION = "resolve"
    CATEGORY = "IAMCCS/LoRA"

    def resolve(self, generation_index, inject_tag, log_prefix, model_type_override="inherit", default_lora=None, linx=None, unique_id=None):
        generation_index = _coerce_generation_index(generation_index)
        inject_tag = str(inject_tag or "inject")
        log_prefix = str(log_prefix or "WAN LoRA runtime bridge")
        model_type = _resolved_model_type(model_type_override, linx)
        available_loras = set(folder_paths.get_filename_list("loras"))

        resolved_stack = []
        active_slot_labels = []
        load_cache = {}

        for entry in _coerce_lora_stack(default_lora):
            resolved_stack.append(
                _annotate_lora_entry(
                    entry,
                    origin=str(entry.get("_iamccs_lora_origin") or "default"),
                    generation_index=generation_index,
                    log_prefix=log_prefix,
                    prompt_id=entry.get("_iamccs_prompt_id"),
                    unique_id=unique_id,
                    slot=entry.get("_iamccs_schedule_slot"),
                    rule=entry.get("_iamccs_schedule_rule"),
                )
            )

        for slot_key, slot_data in _runtime_slot_items(linx):
            if not isinstance(slot_data, dict):
                continue

            try:
                slot = int(str(slot_key).split("_")[-1])
            except Exception:
                continue

            local_name = str(slot_data.get("name") or "no")
            resolved_name = _resolve_linx_name(local_name, slot, linx, available_loras)
            matched, preset_report = _preset_matches(
                generation_index,
                str(slot_data.get("preset") or "custom range"),
                int(slot_data.get("start", slot - 1) or 0),
                int(slot_data.get("end", slot - 1) or 0),
            )
            if not matched:
                continue

            lora_entry = _load_internal_lora_entry(
                resolved_name,
                float(slot_data.get("strength") or 0.0),
                model_type,
                load_cache,
            )
            if not lora_entry:
                continue

            resolved_stack.append(
                _annotate_lora_entry(
                    lora_entry,
                    origin="runtime scheduled",
                    generation_index=generation_index,
                    log_prefix=log_prefix,
                    prompt_id="",
                    unique_id=unique_id,
                    slot=slot,
                    rule=preset_report,
                )
            )
            active_slot_labels.append(f"{slot:02d}:{resolved_name}")

        report = (
            f"{inject_tag}: generation_index={generation_index}"
            f" | active_slots={len(active_slot_labels)}"
            f" | model_type={model_type}"
            f" | entries={len(resolved_stack)}"
            f" | active={', '.join(active_slot_labels) if active_slot_labels else 'default_only'}"
        )

        if resolved_stack:
            _log.info(
                "[%s][%s] generation=%s resolved_stack=%s | %s",
                log_prefix,
                inject_tag,
                generation_index,
                _lora_stack_debug_summary(resolved_stack),
                _lora_stack_debug_context(resolved_stack),
            )
        else:
            _log.warning(
                "[%s][%s] generation=%s resolved no active LoRA entries",
                log_prefix,
                inject_tag,
                generation_index,
            )

        return (resolved_stack, len(active_slot_labels), report, generation_index)


class IAMCCS_ModelWithLoRA_RuntimeBridge:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "lora": ("LORA",),
                "generation_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
                "inject_tag": ("STRING", {"default": "inject"}),
                "log_prefix": ("STRING", {"default": "WAN LoRA runtime apply"}),
            }
        }

    RETURN_TYPES = ("MODEL", "STRING")
    RETURN_NAMES = ("model", "report")
    FUNCTION = "apply"
    CATEGORY = "IAMCCS/LoRA"

    def apply(self, model, lora, generation_index, inject_tag, log_prefix):
        generation_index = _coerce_generation_index(generation_index)
        inject_tag = str(inject_tag or "inject")
        log_prefix = str(log_prefix or "WAN LoRA runtime apply")
        report = (
            f"{inject_tag}: generation_index={generation_index}"
            f" | entries={len(lora or [])}"
            f" | stack={_stack_names(lora or [])}"
        )

        if lora:
            _log.info(
                "[%s][%s] generation=%s apply request | %s",
                log_prefix,
                inject_tag,
                generation_index,
                _lora_stack_debug_summary(lora),
            )
        else:
            _log.warning(
                "[%s][%s] generation=%s apply request with empty LoRA stack",
                log_prefix,
                inject_tag,
                generation_index,
            )

        model_out = IAMCCS_ModelWithLoRA().apply(model, lora)[0]
        return (model_out, report)
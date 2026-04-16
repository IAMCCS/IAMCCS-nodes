import logging
from collections import OrderedDict

import comfy.utils
import folder_paths

from .iamccs_wan_lora_stack import standardize_wan_lora_keys

_log = logging.getLogger("IAMCCS.LoRA.Schedule")

_PRESET_OPTIONS = [
    "custom range",          # applies from Start gen to End gen (-1 = forever)
    "manual_range",          # legacy alias kept for prompt validation compatibility
    "all generations",       # always active
    "gen 0 only",            # first generation only
    "gen 1 onwards",         # skip gen 0, apply to all subsequent
    "even gens (0,2,4...)",  # every even-numbered generation
    "odd gens (1,3,5...)",   # every odd-numbered generation
    "every 2nd gen",         # every 2 gens starting from Start gen
    "every 3rd gen",         # every 3 gens starting from Start gen
]
_DEFAULT_MODEL_TYPE = "flow"
_SLOT_COUNT = 64
_LINX_TYPE = "IAMCCS_WAN_LORA_LINX"
_AUTO_INDEX_STATE: OrderedDict[tuple[str, str], int] = OrderedDict()
_MAX_AUTO_INDEX_STATE = 64


def _annotate_lora_entry(entry: dict, *, origin: str, generation_index: int, log_prefix: str, prompt_id: str | None, unique_id, slot: int | None = None, rule: str | None = None) -> dict:
    annotated = dict(entry)
    annotated["_iamccs_lora_origin"] = str(origin or "unknown")
    annotated["_iamccs_generation_index"] = int(generation_index)
    annotated["_iamccs_schedule_log_prefix"] = str(log_prefix or "WAN LoRA schedule")
    annotated["_iamccs_prompt_id"] = str(prompt_id or "")
    annotated["_iamccs_schedule_node_id"] = "" if unique_id is None else str(unique_id)
    if slot is not None:
        annotated["_iamccs_schedule_slot"] = int(slot)
    if rule:
        annotated["_iamccs_schedule_rule"] = str(rule)
    return annotated


def _cache_put(cache: OrderedDict, key, value, max_size: int):
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max_size:
        cache.popitem(last=False)


def _get_current_prompt_id() -> str | None:
    try:
        import server  # Imported lazily to avoid hard import issues during discovery.

        prompt_server = getattr(server.PromptServer, "instance", None)
        prompt_queue = getattr(prompt_server, "prompt_queue", None)
        currently_running = getattr(prompt_queue, "currently_running", None)
        if not currently_running:
            return None
        current = next(iter(currently_running.values()))
        if len(current) >= 2:
            return str(current[1])
    except Exception:
        return None
    return None


def _next_auto_generation_index(unique_id) -> int:
    prompt_id = _get_current_prompt_id()
    if not prompt_id or unique_id is None:
        return 0

    key = (prompt_id, str(unique_id))
    current = int(_AUTO_INDEX_STATE.get(key, 0) or 0)
    _cache_put(_AUTO_INDEX_STATE, key, current + 1, _MAX_AUTO_INDEX_STATE)
    return current


def _coerce_lora_stack(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [entry for entry in value if isinstance(entry, dict)]
    return []


def _range_matches(index: int, start: int, end: int) -> bool:
    start_i = max(0, int(start))
    end_i = int(end)
    if end_i < 0:
        return index >= start_i
    if end_i < start_i:
        end_i = start_i
    return start_i <= index <= end_i


def _preset_matches(index: int, preset: str, start: int, end: int) -> tuple[bool, str]:
    start_i = max(0, int(start))
    end_i = int(end)
    preset = str(preset or "custom range")

    # Friendly names (new) + underscore aliases (backward compat with old saved workflows)
    if preset in ("all generations", "all_generations"):
        return True, "all generations"
    if preset in ("gen 0 only", "only_first"):
        return index == 0, "gen 0 only"
    if preset in ("gen 1 onwards", "all_nonfirst"):
        return index >= 1, "gen 1 onwards"
    if preset in ("even gens (0,2,4...)", "even_generations"):
        return (index % 2) == 0, "even gens"
    if preset in ("odd gens (1,3,5...)", "odd_generations"):
        return (index % 2) == 1, "odd gens"
    if preset in ("every 2nd gen", "every_2_from_start"):
        return index >= start_i and ((index - start_i) % 2) == 0, f"every 2nd gen (from {start_i})"
    if preset in ("every 3rd gen", "every_3_from_start"):
        return index >= start_i and ((index - start_i) % 3) == 0, f"every 3rd gen (from {start_i})"

    # custom range / manual_range fallback
    if end_i < 0:
        return index >= start_i, f"range {start_i}..forever"
    return _range_matches(index, start_i, end_i), f"range {start_i}..{max(start_i, end_i)}"


def _stack_names(stack: list[dict]) -> str:
    names = []
    for entry in stack:
        name = str(entry.get("name") or "unnamed")
        strength = entry.get("strength", 0.0)
        names.append(f"{name}({strength})")
    return ", ".join(names) if names else "empty"


def _candidate_low_names(name: str) -> list[str]:
    source = str(name or "")
    if not source or source == "no":
        return []

    replacements = [
        ("_HN_", "_LN_"),
        ("-HN_", "-LN_"),
        ("_HN-", "_LN-"),
        ("_HIGH_", "_LOW_"),
        ("-HIGH_", "-LOW_"),
        ("_HIGH-", "_LOW-"),
        ("HN", "LN"),
        ("Hn", "Ln"),
        ("hn", "ln"),
        ("HIGH", "LOW"),
        ("High", "Low"),
        ("high", "low"),
    ]
    out: list[str] = []
    seen: set[str] = set()
    for old, new in replacements:
        if old not in source:
            continue
        candidate = source.replace(old, new)
        if candidate not in seen:
            out.append(candidate)
            seen.add(candidate)
    if source not in seen:
        out.append(source)
    return out


def _resolve_linx_name(local_name: str, slot: int, linx_payload, available_loras: set[str]) -> str:
    local = str(local_name or "no")
    if local != "no":
        return local
    if not isinstance(linx_payload, dict):
        return local

    slot_map = linx_payload.get("slot_map") or {}
    slot_data = slot_map.get(f"slot_{slot:02d}") or {}
    suggested = str(slot_data.get("suggested_low_name") or "")
    if suggested and suggested in available_loras:
        return suggested

    source_name = str(slot_data.get("name") or "")
    for candidate in _candidate_low_names(source_name):
        if candidate in available_loras:
            return candidate
    return local


def _build_linx_payload(model_type: str, kwargs: dict, available_loras: set[str]) -> dict:
    slot_map = {}
    for slot in range(1, _SLOT_COUNT + 1):
        name = str(kwargs.get(f"slot_{slot:02d}_lora_name") or "no")
        suggested_low_name = ""
        for candidate in _candidate_low_names(name):
            if candidate in available_loras:
                suggested_low_name = candidate
                break
        slot_map[f"slot_{slot:02d}"] = {
            "name": name,
            "strength": float(kwargs.get(f"slot_{slot:02d}_strength", 0.0) or 0.0),
            "preset": str(kwargs.get(f"slot_{slot:02d}_preset", "custom range") or "custom range"),
            "start": int(kwargs.get(f"slot_{slot:02d}_start", slot - 1) or 0),
            "end": int(kwargs.get(f"slot_{slot:02d}_end", slot - 1) or 0),
            "suggested_low_name": suggested_low_name,
        }
    return {
        "type": _LINX_TYPE,
        "model_type": str(model_type or _DEFAULT_MODEL_TYPE),
        "slot_map": slot_map,
    }


def _load_internal_lora_entry(name: str, strength: float, model_type: str, cache: dict) -> dict | None:
    if not name or name == "no":
        return None

    strength_f = float(strength)
    if strength_f == 0.0:
        return None

    cache_key = (str(name), str(model_type or _DEFAULT_MODEL_TYPE))
    if cache_key in cache:
        state_dict = cache[cache_key]
    else:
        path = folder_paths.get_full_path_or_raise("loras", name)
        state_dict = comfy.utils.load_torch_file(path, safe_load=True)
        if str(model_type or _DEFAULT_MODEL_TYPE) != "standard":
            state_dict = standardize_wan_lora_keys(state_dict)
        cache[cache_key] = state_dict

    return {
        "name": str(name),
        "strength": strength_f,
        "state_dict": state_dict,
    }


class IAMCCS_WanLoRASchedule:
    @classmethod
    def INPUT_TYPES(cls):
        lora_list = folder_paths.get_filename_list("loras") + ["no"]
        required = {
            # NOTE: Must NOT use lazy:True here.  With lazy:True, ComfyUI caches the
            # node output after the first call (gen_idx=None→0) and skips re-execution
            # on subsequent loop iterations when only generation_index changes, causing
            # the schedule to always behave as if generation_index==0.
            "generation_index": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            "log_prefix": ("STRING", {"default": "WAN LoRA schedule"}),
            "model_type": (["wan2x", "flow", "standard"], {"default": _DEFAULT_MODEL_TYPE}),
        }

        optional = {
            "default_lora": ("LORA",),
            "linx": (_LINX_TYPE,),
        }

        hidden = {
            "unique_id": "UNIQUE_ID",
        }

        for slot in range(1, _SLOT_COUNT + 1):
            required[f"slot_{slot:02d}_lora_name"] = (lora_list, {"default": "no"})
            required[f"slot_{slot:02d}_strength"] = (
                "FLOAT",
                {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.01},
            )
            required[f"slot_{slot:02d}_preset"] = (_PRESET_OPTIONS, {"default": "custom range"})
            required[f"slot_{slot:02d}_start"] = (
                "INT",
                {"default": slot - 1, "min": 0, "max": 1000000, "step": 1},
            )
            required[f"slot_{slot:02d}_end"] = (
                "INT",
                {"default": slot - 1, "min": -1, "max": 1000000, "step": 1},
            )

        return {"required": required, "optional": optional, "hidden": hidden}

    RETURN_TYPES = ("LORA", "INT", "STRING", _LINX_TYPE)
    RETURN_NAMES = ("lora", "active_slots", "report", "linx")
    FUNCTION = "schedule"
    CATEGORY = "IAMCCS/LoRA"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # This node may use the internal loop-safe auto counter mode; cached reuse
        # would freeze the LoRA selection on the first iteration.
        return float("nan")

    def schedule(self, generation_index, log_prefix, model_type=_DEFAULT_MODEL_TYPE, default_lora=None, linx=None, unique_id=None, **kwargs):
        # Loop-safe mode: when generation_index is left at its hidden default 0,
        # use a per-prompt internal counter instead of wiring the easy-use loop
        # index into the MODEL branch, which creates a prompt-validation cycle.
        generation_source = "input"
        if generation_index is None:
            generation_index = _next_auto_generation_index(unique_id)
            generation_source = "auto:none"
        else:
            generation_index = int(generation_index)
            if generation_index == 0:
                generation_index = _next_auto_generation_index(unique_id)
                generation_source = "auto:zero"
        generation_index = max(0, generation_index)
        log_prefix = str(log_prefix or "WAN LoRA schedule")
        model_type = str(model_type or _DEFAULT_MODEL_TYPE)
        prompt_id = _get_current_prompt_id()
        available_loras = set(folder_paths.get_filename_list("loras"))
        out = [
            _annotate_lora_entry(
                entry,
                origin="default",
                generation_index=generation_index,
                log_prefix=log_prefix,
                prompt_id=prompt_id,
                unique_id=unique_id,
            )
            for entry in _coerce_lora_stack(default_lora)
        ]
        active_slots = []
        cache: dict = {}
        linx_payload = _build_linx_payload(model_type, kwargs, available_loras)

        if out:
            _log.info(
                "[%s] prompt=%s node=%s generation=%s source=%s | default_lora=%s",
                log_prefix,
                prompt_id or "unknown",
                unique_id if unique_id is not None else "unknown",
                generation_index,
                generation_source,
                _stack_names(out),
            )

        for slot in range(1, _SLOT_COUNT + 1):
            preset = kwargs.get(f"slot_{slot:02d}_preset", "custom range")
            start = kwargs.get(f"slot_{slot:02d}_start", slot - 1)
            end = kwargs.get(f"slot_{slot:02d}_end", slot - 1)
            matched, preset_report = _preset_matches(generation_index, preset, start, end)
            if not matched:
                continue

            slot_stack = []
            slot_name = _resolve_linx_name(
                str(kwargs.get(f"slot_{slot:02d}_lora_name") or "no"),
                slot,
                linx,
                available_loras,
            )
            entry = _load_internal_lora_entry(
                slot_name,
                float(kwargs.get(f"slot_{slot:02d}_strength", 0.0) or 0.0),
                model_type,
                cache,
            )
            if entry is not None:
                slot_stack.append(
                    _annotate_lora_entry(
                        entry,
                        origin="scheduled",
                        generation_index=generation_index,
                        log_prefix=log_prefix,
                        prompt_id=prompt_id,
                        unique_id=unique_id,
                        slot=slot,
                        rule=preset_report,
                    )
                )

            if not slot_stack:
                continue

            out.extend(slot_stack)
            active_slots.append(f"slot_{slot:02d}:{preset_report} => {_stack_names(slot_stack)}")

        report = (
            f"generation_index={generation_index} | active={' + '.join(active_slots) if active_slots else 'default_only'} | "
            f"entries={len(out)}"
        )

        if out:
            _log.info("[%s] %s", log_prefix, report)
            _log.info(
                "[%s] generation=%s resolved_stack=%s",
                log_prefix,
                generation_index,
                _stack_names(out),
            )
        else:
            _log.warning("[%s] No LoRA active | generation_index=%s", log_prefix, generation_index)

        return (out, len(active_slots), report, linx_payload)


NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRASchedule": IAMCCS_WanLoRASchedule,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRASchedule": "LoRA Schedule (WAN, ranged)",
}

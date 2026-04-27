from __future__ import annotations

from typing import Any

try:
    from .iamccs_flexible_inputs import FlexibleOptionalInputType, any_type
except ImportError:
    from iamccs_flexible_inputs import FlexibleOptionalInputType, any_type


_SUPERNODE_CONTRACT_TYPE = "IAMCCS_SUPERNODE_CONTRACT"
_SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
_MODULE_INPUT_SLOTS = 6


def _parse_contract(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return dict(payload)

    parsed: dict[str, Any] = {}
    if payload is None:
        return parsed

    for item in str(payload).split(";"):
        item = item.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def _build_dynamic_optional_slots(type_name: str) -> FlexibleOptionalInputType:
    data = {f"module_{index:02d}": (type_name,) for index in range(1, _MODULE_INPUT_SLOTS + 1)}
    return FlexibleOptionalInputType(type_name, data=data)


def _collect_named_inputs(kwargs: dict[str, Any], prefix: str) -> dict[str, Any]:
    collected: dict[str, Any] = {}
    for key, value in kwargs.items():
        if not key.startswith(prefix):
            continue
        if value is None:
            continue
        collected[key] = value
    return collected


def _build_linx_payload(
    existing_linx: Any,
    *,
    pipeline_kind: str,
    node_role: str,
    node_label: str,
    unique_id: Any,
    upstream_contract: dict[str, Any],
    module_inputs: dict[str, Any],
) -> dict[str, Any]:
    if isinstance(existing_linx, dict):
        linx_payload = dict(existing_linx)
        nodes = list(existing_linx.get("nodes") or [])
    else:
        linx_payload = {}
        nodes = []

    linx_payload["type"] = _SUPERNODE_LINX_TYPE
    linx_payload["pipeline_kind"] = pipeline_kind
    linx_payload["root_contract"] = upstream_contract.get("pipeline_key", pipeline_kind)
    linx_payload["nodes"] = nodes
    linx_payload["nodes"].append(
        {
            "id": str(unique_id or node_label or node_role),
            "role": node_role,
            "label": node_label,
            "module_keys": sorted(module_inputs.keys()),
        }
    )
    return linx_payload


class IAMCCS_SupernodeBase:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "upstream_contract": (_SUPERNODE_CONTRACT_TYPE,),
            "linx": (_SUPERNODE_LINX_TYPE,),
            "planner_payload": ("STRING",),
            "backend_payload": ("STRING",),
            "continuity_payload": ("STRING",),
            "second_stage_payload": ("STRING",),
        }
        optional.update(_build_dynamic_optional_slots(_SUPERNODE_CONTRACT_TYPE))
        return {
            "required": {
                "pipeline_kind": (["v2v", "i2v_flf", "au_img2vid", "audio_concat", "wan_flf", "wan_continuity"],),
                "surface_profile": (["compact", "progressive", "debug_surface"], {"default": "compact"}),
                "backend_binding": (["ltx_v2v_disk", "ltx_audio_guided_lowram", "ltx_i2v_flf", "wan_continuity", "wan_flf", "custom"], {"default": "custom"}),
                "node_label": ("STRING", {"default": "Base Supernode"}),
                "notes": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": optional,
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (_SUPERNODE_CONTRACT_TYPE, _SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("contract", "linx", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Supernodes"

    def build(
        self,
        pipeline_kind,
        surface_profile,
        backend_binding,
        node_label,
        notes="",
        upstream_contract=None,
        linx=None,
        planner_payload=None,
        backend_payload=None,
        continuity_payload=None,
        second_stage_payload=None,
        unique_id=None,
        **kwargs,
    ):
        upstream = _parse_contract(upstream_contract)
        module_inputs = _collect_named_inputs(kwargs, "module_")
        contract = {
            "type": _SUPERNODE_CONTRACT_TYPE,
            "pipeline_key": f"{pipeline_kind}:{node_label}",
            "pipeline_kind": pipeline_kind,
            "surface_profile": surface_profile,
            "backend_binding": backend_binding,
            "role": "base",
            "label": str(node_label or "Base Supernode"),
            "notes": str(notes or ""),
            "payloads": {
                "planner_payload": planner_payload,
                "backend_payload": backend_payload,
                "continuity_payload": continuity_payload,
                "second_stage_payload": second_stage_payload,
            },
            "module_inputs": sorted(module_inputs.keys()),
            "upstream": upstream,
        }
        linx_payload = _build_linx_payload(
            linx,
            pipeline_kind=pipeline_kind,
            node_role="base",
            node_label=str(node_label or "Base Supernode"),
            unique_id=unique_id,
            upstream_contract=contract,
            module_inputs=module_inputs,
        )
        report = (
            f"supernode_base pipeline={pipeline_kind} | surface={surface_profile} | backend_binding={backend_binding} | "
            f"linked_modules={','.join(sorted(module_inputs.keys())) or 'none'} | upstream={upstream.get('pipeline_key', 'none')}"
        )
        return (contract, linx_payload, report)


class IAMCCS_SupernodeModule:
    @classmethod
    def INPUT_TYPES(cls):
        optional = {
            "parent_contract": (_SUPERNODE_CONTRACT_TYPE,),
            "linx": (_SUPERNODE_LINX_TYPE,),
            "payload": ("STRING",),
        }
        optional.update(_build_dynamic_optional_slots(any_type))
        return {
            "required": {
                "pipeline_kind": (["v2v", "i2v_flf", "au_img2vid", "audio_concat", "wan_flf", "wan_continuity"],),
                "module_role": (["planner", "backend", "continuity", "audio", "concat", "loop", "second_stage", "output", "custom"],),
                "module_mode": (["consume", "augment", "branch"], {"default": "augment"}),
                "node_label": ("STRING", {"default": "Module"}),
                "module_key": ("STRING", {"default": "module"}),
            },
            "optional": optional,
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = (_SUPERNODE_CONTRACT_TYPE, _SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("contract", "linx", "report")
    FUNCTION = "build"
    CATEGORY = "IAMCCS/Supernodes"

    def build(
        self,
        pipeline_kind,
        module_role,
        module_mode,
        node_label,
        module_key,
        parent_contract=None,
        linx=None,
        payload=None,
        unique_id=None,
        **kwargs,
    ):
        parent = _parse_contract(parent_contract)
        module_inputs = _collect_named_inputs(kwargs, "module_")
        contract = {
            "type": _SUPERNODE_CONTRACT_TYPE,
            "pipeline_key": parent.get("pipeline_key", pipeline_kind),
            "pipeline_kind": pipeline_kind,
            "role": str(module_role or "custom"),
            "mode": str(module_mode or "augment"),
            "label": str(node_label or module_role or "Module"),
            "module_key": str(module_key or "module"),
            "payload": payload,
            "module_inputs": sorted(module_inputs.keys()),
            "parent": parent,
        }
        linx_payload = _build_linx_payload(
            linx,
            pipeline_kind=pipeline_kind,
            node_role=str(module_role or "custom"),
            node_label=str(node_label or module_role or "Module"),
            unique_id=unique_id,
            upstream_contract=contract,
            module_inputs=module_inputs,
        )
        report = (
            f"supernode_module pipeline={pipeline_kind} | role={module_role} | mode={module_mode} | key={module_key} | "
            f"linked_inputs={','.join(sorted(module_inputs.keys())) or 'none'} | parent={parent.get('label', 'none')}"
        )
        return (contract, linx_payload, report)
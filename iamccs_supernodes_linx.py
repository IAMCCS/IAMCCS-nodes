SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _clone_chain(existing_linx):
    if not isinstance(existing_linx, dict):
        return []
    chain = existing_linx.get("chain") or []
    return [dict(item) for item in chain if isinstance(item, dict)]


def _clone_dict(existing_linx, key):
    if not isinstance(existing_linx, dict):
        return {}
    value = existing_linx.get(key) or {}
    if not isinstance(value, dict):
        return {}
    return dict(value)


def _clone_resources(existing_linx):
    if not isinstance(existing_linx, dict):
        return {}
    value = existing_linx.get("resources") or {}
    if not isinstance(value, dict):
        return {}
    return dict(value)


_RESOURCE_TYPE_HINTS = {
    "audio": "AUDIO",
    "audio_raw": "AUDIO",
    "audio_conditioning_single": "AUDIO",
    "audio_conditioning_segmented": "AUDIO",
    "audio_duration_source": "AUDIO",
    "conditioning_duration_audio": "AUDIO",
    "model": "MODEL",
    "clip": "CLIP",
    "vae": "VAE",
    "audio_vae": "VAE",
    "taeltx_vae": "VAE",
    "second_stage_model": "MODEL",
    "video_latent": "LATENT",
    "first_stage_video_latent": "LATENT",
    "rendered_images": "IMAGE",
    "first_stage_preview_images": "IMAGE",
    "taeltx_first_stage_preview_images": "IMAGE",
    "taeltx_first_stage_preview_report": "STRING",
    "fps": "FLOAT",
    "decode_mode": "STRING",
    "output_root": "STRING",
    "planner_payload": "STRING",
    "second_stage_payload": "STRING",
}


def _type_hint_for_key(key, value=None):
    key = str(key)
    if key in _RESOURCE_TYPE_HINTS:
        return _RESOURCE_TYPE_HINTS[key]
    if value is None:
        return "PYTHON_OBJECT"
    return type(value).__name__


def _normalize_contract_map(value):
    if not isinstance(value, dict):
        return {}
    normalized = {}
    for key, item in value.items():
        if isinstance(item, dict):
            normalized[str(key)] = dict(item)
        elif isinstance(item, (list, tuple, set)):
            normalized[str(key)] = sorted(str(entry) for entry in item)
        else:
            normalized[str(key)] = str(item)
    return normalized


def linx_contract(existing_linx, stage_name=None, default=None):
    if not isinstance(existing_linx, dict):
        return default
    contracts = existing_linx.get("contracts") or {}
    if not isinstance(contracts, dict):
        return default
    if stage_name is None:
        return contracts
    return contracts.get(str(stage_name), default)


def linx_missing_resources(existing_linx, required_keys):
    resources = _clone_resources(existing_linx)
    return [str(key) for key in required_keys if resources.get(str(key)) is None]


def linx_resource(existing_linx, key, default=None):
    if not isinstance(existing_linx, dict):
        return default
    resources = existing_linx.get("resources") or {}
    if not isinstance(resources, dict):
        return default
    return resources.get(key, default)


def linx_output(existing_linx, key, default=None):
    if not isinstance(existing_linx, dict):
        return default
    outputs = existing_linx.get("outputs") or {}
    if not isinstance(outputs, dict):
        return default
    return outputs.get(key, default)


def linx_policy(existing_linx, key, default=None):
    if not isinstance(existing_linx, dict):
        return default
    policies = existing_linx.get("policies") or {}
    if not isinstance(policies, dict):
        return default
    return policies.get(key, default)


def build_supernode_linx_payload(existing_linx, node_role, payload, report, unique_id=None):
    if isinstance(existing_linx, dict):
        linx_payload = dict(existing_linx)
        chain = _clone_chain(existing_linx)
    else:
        linx_payload = {}
        chain = []

    linx_payload["type"] = SUPERNODE_LINX_TYPE
    linx_payload["chain"] = chain
    linx_payload["chain"].append(
        {
            "id": str(unique_id or node_role),
            "role": str(node_role),
            "payload_preview": str(payload or "")[:240],
            "report_preview": str(report or "")[:240],
        }
    )
    return linx_payload


def build_stage_linx_payload(
    existing_linx,
    stage_name,
    stage_kind,
    payload,
    report,
    unique_id=None,
    slot_map=None,
    downstream_stages=None,
    policies=None,
    outputs=None,
    resources=None,
    requires=None,
):
    linx_payload = build_supernode_linx_payload(existing_linx, stage_name, payload, report, unique_id=unique_id)
    stage_entry = {
        "id": str(unique_id or stage_name),
        "name": str(stage_name),
        "kind": str(stage_kind),
        "payload": dict(payload) if isinstance(payload, dict) else {"preview": str(payload or "")[:240]},
        "report_preview": str(report or "")[:240],
    }
    normalized_requires = _normalize_contract_map(requires)
    if normalized_requires:
        stage_entry["requires"] = normalized_requires

    stages = list(linx_payload.get("stages") or [])
    stages.append(stage_entry)
    linx_payload["stages"] = stages
    linx_payload["stage_count"] = len(stages)

    merged_slot_map = _clone_dict(linx_payload, "slot_map")
    if isinstance(slot_map, dict):
        merged_slot_map.update(slot_map)
    if merged_slot_map:
        linx_payload["slot_map"] = merged_slot_map

    merged_policies = _clone_dict(linx_payload, "policies")
    if isinstance(policies, dict):
        merged_policies.update(policies)
    if merged_policies:
        linx_payload["policies"] = merged_policies

    merged_outputs = _clone_dict(linx_payload, "outputs")
    if isinstance(outputs, dict):
        merged_outputs.update(outputs)
    if merged_outputs:
        linx_payload["outputs"] = merged_outputs
        output_sources = _clone_dict(linx_payload, "output_sources")
        if isinstance(outputs, dict):
            for output_key in outputs.keys():
                output_sources[str(output_key)] = str(stage_name)
        if output_sources:
            linx_payload["output_sources"] = output_sources

    merged_resources = _clone_resources(linx_payload)
    if isinstance(resources, dict):
        for resource_key, resource_value in resources.items():
            if resource_value is not None:
                merged_resources[str(resource_key)] = resource_value
    if merged_resources:
        linx_payload["resources"] = merged_resources
        linx_payload["resource_keys"] = sorted(str(item) for item in merged_resources.keys())
        linx_payload["resource_types"] = {
            str(key): _type_hint_for_key(key, value)
            for key, value in merged_resources.items()
        }
        resource_sources = _clone_dict(linx_payload, "resource_sources")
        if isinstance(resources, dict):
            for resource_key, resource_value in resources.items():
                if resource_value is not None:
                    resource_sources[str(resource_key)] = str(stage_name)
        if resource_sources:
            linx_payload["resource_sources"] = resource_sources

    provided = {}
    if isinstance(outputs, dict) and outputs:
        provided["outputs"] = {
            str(key): _type_hint_for_key(key, value)
            for key, value in outputs.items()
        }
    if isinstance(resources, dict):
        resource_contract = {
            str(key): _type_hint_for_key(key, value)
            for key, value in resources.items()
            if value is not None
        }
        if resource_contract:
            provided["resources"] = resource_contract
    if isinstance(slot_map, dict) and slot_map:
        provided["slots"] = dict(slot_map)
    if isinstance(policies, dict) and policies:
        provided["policies"] = {
            str(key): _type_hint_for_key(key, value)
            for key, value in policies.items()
        }
    stage_contract = {}
    if normalized_requires:
        stage_contract["requires"] = normalized_requires
    if provided:
        stage_contract["provides"] = provided
    if stage_contract:
        stages = list(linx_payload.get("stages") or [])
        if stages:
            stages[-1].update(stage_contract)
            linx_payload["stages"] = stages
        contracts = _clone_dict(linx_payload, "contracts")
        contracts[str(stage_name)] = stage_contract
        linx_payload["contracts"] = contracts

    if downstream_stages is not None:
        linx_payload["downstream_stages"] = [str(item) for item in downstream_stages if str(item or "").strip()]

    linx_payload["active_stage"] = str(stage_name)
    linx_payload["active_stage_kind"] = str(stage_kind)
    return linx_payload

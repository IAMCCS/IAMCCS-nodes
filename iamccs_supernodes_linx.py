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
):
    linx_payload = build_supernode_linx_payload(existing_linx, stage_name, payload, report, unique_id=unique_id)
    stage_entry = {
        "id": str(unique_id or stage_name),
        "name": str(stage_name),
        "kind": str(stage_kind),
        "payload": dict(payload) if isinstance(payload, dict) else {"preview": str(payload or "")[:240]},
        "report_preview": str(report or "")[:240],
    }

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

    merged_resources = _clone_resources(linx_payload)
    if isinstance(resources, dict):
        for resource_key, resource_value in resources.items():
            if resource_value is not None:
                merged_resources[str(resource_key)] = resource_value
    if merged_resources:
        linx_payload["resources"] = merged_resources
        linx_payload["resource_keys"] = sorted(str(item) for item in merged_resources.keys())

    if downstream_stages is not None:
        linx_payload["downstream_stages"] = [str(item) for item in downstream_stages if str(item or "").strip()]

    linx_payload["active_stage"] = str(stage_name)
    linx_payload["active_stage_kind"] = str(stage_kind)
    return linx_payload
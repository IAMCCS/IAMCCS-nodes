# SPDX-License-Identifier: GPL-3.0-or-later
"""Incremental H3 LoRA discovery for the UI, outside ComfyUI schema generation."""
from __future__ import annotations

import asyncio
from pathlib import Path

from .iamccs_h3_advisor import classify_fasth3_dense_file, describe_asset, preferred_sla_defaults


_entries = {}
_refresh_task: asyncio.Task | None = None


def is_native_pdd_candidate(name: str) -> bool:
    name = name.lower()
    return "comfy" in name and (
        ("pdd" in name and "acc" in name)
        or ("h3" in name and "acc" in name and "8step" in name)
    )


def _refresh_catalog():
    import folder_paths
    import nodes
    import torch

    names = sorted(folder_paths.get_filename_list("loras"), key=str.lower)
    current = {}
    assets = []
    fasth3 = []
    choices = []
    for name in names:
        path = folder_paths.get_full_path("loras", name)
        if not path:
            continue
        try:
            stat = Path(path).stat()
        except OSError:
            continue
        fingerprint = (str(path), stat.st_mtime_ns, stat.st_size)
        previous = _entries.get(name)
        if previous is not None and previous[0] == fingerprint:
            _, compatible, asset = previous
        else:
            compatible = Path(path).suffix.lower() == ".safetensors" and classify_fasth3_dense_file(*fingerprint)[0]
            asset = describe_asset(name, path) if compatible or "h3" in name.lower() else None
        current[name] = (fingerprint, compatible, asset)
        if compatible:
            fasth3.append(name)
        if asset is not None:
            choices.append(name)
            assets.append(asset)

    # Drop removed files and old fingerprints. Unchanged files need no reads,
    # even when the active registry exceeds the generic header cache size.
    _entries.clear()
    _entries.update(current)
    return {
        "choices": choices,
        "pdd": [name for name in names if name in current and is_native_pdd_candidate(name)],
        "fasth3": fasth3,
        "assets": assets,
        "sla_defaults": preferred_sla_defaults(
            assets, "auto_from_timeline", nodes.NODE_CLASS_MAPPINGS if torch.cuda.is_available() else {},
        ),
    }


async def lora_catalog():
    global _refresh_task
    if _refresh_task is None or _refresh_task.done():
        _refresh_task = asyncio.create_task(asyncio.to_thread(_refresh_catalog))
    # One disconnected client must not cancel work shared by other clients.
    return await asyncio.shield(_refresh_task)

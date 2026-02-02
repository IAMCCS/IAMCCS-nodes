"""IAMCCS MultiSwitch

Dynamic-input switch node:
- Starts with 5 inputs named input_01..input_05.
- Frontend adds more inputs when the last one is connected.
- Backend accepts any input_* keys via FlexibleOptionalInputType.

Selection:
- If `bus` is set (1-based), prefer that input if it is non-empty.
- Otherwise return the first non-empty input_* in numeric order.

Receiver:
- Takes `bus` from the main MultiSwitch and selects the same index among its own inputs.
"""

from __future__ import annotations

import re
from typing import Any

from .iamccs_flexible_inputs import FlexibleOptionalInputType, any_type


_INPUT_RE = re.compile(r"^input_(\d+)$")


def _is_none_like(value: Any) -> bool:
    if value is None:
        return True

    # Treat "empty" context-ish dicts as none-like (best-effort).
    if isinstance(value, dict) and "model" in value and "clip" in value:
        if not value.get("model") and not value.get("clip"):
            return True

    return False


def _iter_input_keys_sorted(kwargs: dict) -> list[str]:
    pairs: list[tuple[int, str]] = []
    for k in kwargs.keys():
        m = _INPUT_RE.match(k)
        if not m:
            continue
        try:
            i = int(m.group(1))
        except Exception:
            continue
        pairs.append((i, k))
    pairs.sort(key=lambda x: x[0])
    return [k for _, k in pairs]


def _pick_value(bus: Any, kwargs: dict) -> tuple[Any, int]:
    bus_i = 0
    if bus is not None:
        try:
            bus_i = int(bus or 0)
        except Exception:
            bus_i = 0

    # Prefer BUS-selected slot if valid and non-empty.
    if bus_i > 0:
        k = f"input_{bus_i:02d}"
        v = kwargs.get(k, None)
        if not _is_none_like(v):
            return v, bus_i

    # Fallback: first non-empty in order.
    for k in _iter_input_keys_sorted(kwargs):
        v = kwargs.get(k, None)
        if not _is_none_like(v):
            m = _INPUT_RE.match(k)
            out_bus = int(m.group(1)) if m else 0
            return v, out_bus

    return None, 0


class IAMCCS_MultiSwitch:
    NAME = "IAMCCS_MultiSwitch"
    CATEGORY = "IAMCCS/Utilities"

    @classmethod
    def INPUT_TYPES(cls):
        data: dict[str, object] = {}
        for i in range(1, 6):
            data[f"input_{i:02d}"] = (any_type,)
        return {"required": {}, "optional": FlexibleOptionalInputType(any_type, data=data)}

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("output",)
    FUNCTION = "switch"

    def switch(self, **kwargs):
        value, _out_bus = _pick_value(None, kwargs)
        return (value,)

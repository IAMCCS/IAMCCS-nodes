from __future__ import annotations

from typing import Any, Optional, Union


class AnyType(str):
    """A string type that compares equal to any other type string.

    Used to support flexible inputs/outputs in ComfyUI.
    """

    def __ne__(self, __value: object) -> bool:  # noqa: D401
        return False


class FlexibleOptionalInputType(dict):
    """Dict-like helper for dynamic optional inputs.

    - Appears to contain any key (so ComfyUI will accept dynamically created input names).
    - For unknown keys, returns a (type,) tuple.
    - For known keys (provided via `data`), returns the stored tuple.
    """

    def __init__(self, type_: Any, data: Union[dict, None] = None):
        super().__init__()
        self.type = type_
        self.data: Optional[dict] = data
        if self.data is not None:
            for k, v in self.data.items():
                self[k] = v

    def __getitem__(self, key):
        if self.data is not None and key in self.data:
            return self.data[key]
        return (self.type,)

    def __contains__(self, key):
        return True


any_type = AnyType("*")

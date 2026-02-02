from __future__ import annotations

import json
from typing import Any

from .iamccs_hw_probe import recommend_settings


class IAMCCS_HWProbeRecommendations:
    """Expose IAMCCS hardware probe recommendations as a workflow node.

    This mirrors the `/api/iamccs/hw_probe` endpoint, but lets you drive the
    recommendations inside ComfyUI graphs (e.g. for logging, branching, or
    pre-filling other nodes).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8}),
                "frames": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "fps": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 240.0, "step": 0.5}),
                "pretty": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (
        "STRING",
        "STRING",
        "FLOAT",
        "INT",
        "INT",
        "INT",
        "INT",
    )
    RETURN_NAMES = (
        "report_json",
        "profile",
        "reserved_vram_effective_gb",
        "vae_tile_size",
        "vae_overlap",
        "vae_temporal_size",
        "vae_temporal_overlap",
    )
    FUNCTION = "probe"
    CATEGORY = "IAMCCS/HW"

    def probe(self, width: int, height: int, frames: int, fps: float, pretty: bool):
        w = int(width) if int(width) > 0 else None
        h = int(height) if int(height) > 0 else None
        f = int(frames) if int(frames) > 0 else None
        r = float(fps) if float(fps) > 0 else None

        data: dict[str, Any] = recommend_settings(width=w, height=h, frames=f, fps=r)

        rec = (data.get("recommendations") or {})
        hw = (rec.get("hw_supporter") or {})
        vae = (rec.get("vae_decode") or {})

        profile = str(hw.get("profile") or "auto")
        reserved_eff = hw.get("reserved_vram_effective_gb")
        try:
            reserved_eff_f = float(reserved_eff) if reserved_eff is not None else 0.0
        except Exception:
            reserved_eff_f = 0.0

        def _to_int(x: Any) -> int:
            try:
                return int(x)
            except Exception:
                return 0

        tile_size = _to_int(vae.get("tile_size"))
        overlap = _to_int(vae.get("overlap"))
        temporal_size = _to_int(vae.get("temporal_size"))
        temporal_overlap = _to_int(vae.get("temporal_overlap"))

        if bool(pretty):
            report = json.dumps(data, ensure_ascii=False, indent=2)
        else:
            report = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

        return (
            report,
            profile,
            float(reserved_eff_f),
            int(tile_size),
            int(overlap),
            int(temporal_size),
            int(temporal_overlap),
        )

import copy
import json
import time


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"


def _base_linx(phase, enabled):
    return {
        "schema": "iamccs_audio_phase_lazy_gate",
        "version": 1,
        "phase": str(phase or "dialogue_to_audioboard"),
        "enabled": bool(enabled),
        "resources": {},
        "meta": {
            "created_at": time.time(),
            "truth": "IAMCCS lazy gates evaluate only the active audio/video phase and pass CineLinx/audio timeline payloads without changing their contents.",
        },
    }


def _clone_linx(cine_linx):
    if isinstance(cine_linx, dict):
        try:
            return copy.deepcopy(cine_linx)
        except Exception:
            return dict(cine_linx)
    if isinstance(cine_linx, str) and cine_linx.strip():
        try:
            parsed = json.loads(cine_linx)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    return None


class IAMCCS_AudioPhaseLazyGate:
    """
    Lazy CineLinx/audio phase gate for splitting large IAMCCS audio workflows into explicit stages.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "phase": ([
                    "dialogue_to_audioboard",
                    "publish_to_shotboard",
                    "video_from_shotboard",
                ], {"default": "dialogue_to_audioboard"}),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "ACTIVE", "label_off": "BYPASS"}),
            },
            "optional": {
                "cine_linx": (SUPERNODE_LINX_TYPE, {"lazy": True}),
                "audio_timeline_json": ("STRING", {"forceInput": True, "lazy": True}),
            },
        }

    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("cine_linx", "audio_timeline_json", "enabled", "report")
    FUNCTION = "run"
    CATEGORY = "IAMCCS/Audio Tools"

    def check_lazy_status(self, phase, enabled=True, cine_linx=None, audio_timeline_json=None):
        if not enabled:
            return []
        requested = []
        if cine_linx is None:
            requested.append("cine_linx")
        return requested

    def run(self, phase, enabled=True, cine_linx=None, audio_timeline_json=""):
        phase = str(phase or "dialogue_to_audioboard")
        if not enabled:
            payload = _base_linx(phase, False)
            report = {
                "node": "IAMCCS_AudioPhaseLazyGate",
                "phase": phase,
                "enabled": False,
                "truth": "Disabled lazy gate did not request upstream lazy inputs.",
            }
            return (payload, "", False, json.dumps(report, ensure_ascii=False, indent=2))

        payload = _clone_linx(cine_linx) or _base_linx(phase, True)
        resources = payload.setdefault("resources", {})
        timeline_json = str(audio_timeline_json or resources.get("cine_audio_timeline_json") or resources.get("audio_timeline_json") or "")
        resources["iamccs_audio_phase_lazy_gate"] = {
            "phase": phase,
            "enabled": True,
            "has_audio_timeline_json": bool(timeline_json.strip()),
            "updated_at": time.time(),
        }
        if timeline_json.strip():
            resources["cine_audio_timeline_json"] = timeline_json
            resources["audio_timeline_json"] = timeline_json
        chain = payload.setdefault("iamccs_phase_chain", [])
        if isinstance(chain, list):
            chain.append({"node": "IAMCCS_AudioPhaseLazyGate", "phase": phase, "enabled": True})
        report = {
            "node": "IAMCCS_AudioPhaseLazyGate",
            "phase": phase,
            "enabled": True,
            "has_cine_linx": cine_linx is not None,
            "has_audio_timeline_json": bool(timeline_json.strip()),
            "truth": "Enabled lazy gate passes CineLinx and audio timeline payloads unchanged except for phase metadata.",
        }
        return (payload, timeline_json, True, json.dumps(report, ensure_ascii=False, indent=2))

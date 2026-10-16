"""Delivery-file bridge for the isolated R39/R40/R42 editor workflows.

The production R39/R40 delivery nodes intentionally write an MP4 and return its
path.  The IAMCCS editor, on the other hand, consumes decoded IMAGE/AUDIO media.
This adapter keeps those contracts separate and never changes the established
R37/R39 generation or editor nodes.
"""

from pathlib import Path
from typing import Any, Dict

import torch

from comfy_api.latest import InputImpl

from .iamccs_shotboarder_exporter_pro import IAMCCS_ShotboarderAudVidExporterPRO
from .cine_multigeneration import IAMCCS_ShotboardVideoEditorV1, IAMCCS_ShotboardVideoEditorRenderV1


SUPERNODE_LINX_TYPE = "IAMCCS_SUPERNODE_LINX"
CATEGORY = "IAMCCS/MiniMax H3/Editor"


def _shotplan(cine_linx: Any) -> Dict[str, Any]:
    if not isinstance(cine_linx, dict):
        return {}
    for container_name in ("resources", "outputs", "payload"):
        container = cine_linx.get(container_name)
        if not isinstance(container, dict):
            continue
        for key in ("iamccs_minimax_h3_shotplan", "minimax_h3_shotplan", "shotplan"):
            value = container.get(key)
            if isinstance(value, dict) and str(value.get("schema", "")).startswith("iamccs.minimax_h3.shotplan"):
                return value
    return {}


def _audio_ok(audio: Any) -> bool:
    return isinstance(audio, dict) and torch.is_tensor(audio.get("waveform"))


class IAMCCS_MiniMaxH3EditorDeliveryMedia:
    """Decode the selected per-shot or one-film delivery for the editor.

    Ordinary multishot modes publish each rendered ``segment_NNNN.mp4`` to the
    chosen editor lane. LongVid and FL2VA Extended AV publish only the completed
    ``final_film.mp4`` as one asset because their chunks are technical pieces of
    a single take. This mirrors IAMCCS_MiniMaxH3EditorTakeRoute semantics while
    ensuring the editor receives the upscaled delivery when delivery is enabled.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "cine_linx": (SUPERNODE_LINX_TYPE,),
                "delivery_video_path": ("STRING", {"forceInput": True}),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "native_frames": ("IMAGE",),
                "native_audio": ("AUDIO",),
            }
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("delivery_frames", "delivery_audio", "master_ready", "resolved_path", "report")
    FUNCTION = "load"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # The path can be reused by an automatic segment loop.  Always verify
        # what exists on disk rather than caching an intermediate readiness flag.
        return float("nan")

    def load(self, cine_linx, delivery_video_path, current_segment, total_segments, native_frames, native_audio):
        plan = _shotplan(cine_linx)
        if not plan:
            raise ValueError("H3 Editor Delivery requires an IAMCCS H3 shotplan in cine_linx")

        index = max(0, int(current_segment))
        total = max(1, int(total_segments))
        task_mode = str(plan.get("task_mode", "") or "").strip().lower()
        upscale_enabled = bool(plan.get("upscale_enabled", False))
        upscale_mode = str(plan.get("upscale_mode", "off") or "off").strip().lower()
        editorial_per_shot = upscale_enabled and upscale_mode == "ltx23_per_chunk"
        if task_mode == "viggle_animation":
            if not torch.is_tensor(native_frames) or native_frames.ndim != 4 or int(native_frames.shape[0]) < 1:
                raise ValueError("H3 editor delivery received no Viggle IMAGE frames")
            if not _audio_ok(native_audio):
                raise ValueError("H3 editor delivery received no Viggle source AUDIO")
            return (
                native_frames,
                native_audio,
                True,
                "",
                f"H3 editor delivery | Viggle in-memory asset | frames={int(native_frames.shape[0])}",
            )
        # A normal LongVid remains one editorial asset.  The explicit LTX
        # per-shot contract is the exception: every generated H3 chunk is a
        # separately decoded roll so pre/post-roll and cut decisions remain
        # available inside the Video Editor.
        longvid = task_mode.startswith("longvid") and not editorial_per_shot
        extended_av = task_mode == "fl2va_extended_av" and not editorial_per_shot
        one_master = bool(longvid or extended_av)
        authored = Path(str(delivery_video_path or "").strip())

        if one_master and index + 1 < total:
            return (
                native_frames,
                native_audio,
                False,
                "",
                f"H3 editor delivery waiting for {'Extended AV' if extended_av else 'LongVid'} master | completed={index + 1}/{total}",
            )

        if one_master:
            resolved = authored
        else:
            # On the final pass the delivery node returns final_film.mp4.  The
            # editor still needs the current shot, not a duplicate concatenated
            # programme, so resolve the deterministic per-segment asset.
            resolved = authored.parent / f"segment_{index + 1:04d}.mp4"
            if not resolved.is_file() and authored.name.lower() != "final_film.mp4":
                resolved = authored

        if not resolved.is_file():
            raise FileNotFoundError(f"H3 editor delivery checkpoint not found: {resolved}")

        components = InputImpl.VideoFromFile(str(resolved)).get_components()
        frames = components.images
        audio = components.audio if _audio_ok(components.audio) else native_audio
        if not torch.is_tensor(frames) or frames.ndim != 4 or int(frames.shape[0]) < 1:
            raise ValueError(f"H3 editor delivery decoded no frames from: {resolved}")
        if not _audio_ok(audio):
            raise ValueError(f"H3 editor delivery decoded no audio from: {resolved}")

        master_ready = bool(one_master and index + 1 >= total)
        delivery_kind = (
            "Extended AV final master" if extended_av
            else ("LongVid final master" if longvid
                  else (f"LTX editorial roll {index + 1}/{total}" if editorial_per_shot else f"slot {index + 1}/{total}"))
        )
        report = (
            f"H3 editor delivery checkpoint ready | {delivery_kind} | "
            f"frames={int(frames.shape[0])} | fps={float(components.frame_rate):.3f} | path={resolved}"
        )
        return frames, audio, master_ready, str(resolved), report


class IAMCCS_MiniMaxH3DeliveryPreviewStripR42:
    """Expose a lightweight visual result for every automatic R42 pass.

    The delivery/editor adapter may hold hundreds of decoded frames. Sending
    the full batch to PreviewImage creates unnecessary temp files and makes a
    long-video workflow hard to inspect. This node selects only first, middle
    and last, while retaining the segment/master provenance in its UI report.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "delivery_frames": ("IMAGE",),
                "current_segment": ("INT", {"forceInput": True}),
                "total_segments": ("INT", {"forceInput": True}),
                "resolved_path": ("STRING", {"forceInput": True}),
                "delivery_report": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("preview_frames", "preview_report")
    FUNCTION = "build"
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        # Automatic segment queues reuse the same graph and node ids.
        return float("nan")

    def build(self, delivery_frames, current_segment, total_segments, resolved_path, delivery_report):
        if not torch.is_tensor(delivery_frames) or delivery_frames.ndim != 4:
            raise ValueError("R42 Delivery Preview expects an IMAGE batch")
        count = int(delivery_frames.shape[0])
        if count < 1:
            raise ValueError("R42 Delivery Preview received an empty IMAGE batch")
        indexes = sorted({0, count // 2, count - 1})
        preview = delivery_frames[indexes]
        index = max(0, int(current_segment))
        total = max(1, int(total_segments))
        source = str(resolved_path or "").strip()
        scope = "DELIVERY MASTER" if source and index + 1 >= total else "CURRENT CHUNK"
        report = (
            f"R42 {scope} PREVIEW | segment={index + 1}/{total} | "
            f"source_frames={count} | shown={','.join(str(value) for value in indexes)} | "
            f"path={source or 'native in-memory checkpoint'}"
        )
        if delivery_report:
            report = f"{report} | {delivery_report}"
        return {"ui": {"text": [report]}, "result": (preview, report)}



class IAMCCS_ShotboardVideoEditorOptionalR49(IAMCCS_ShotboardVideoEditorV1):
    OUTPUT_NODE = False


class IAMCCS_ShotboardVideoEditorRenderOptionalR49(IAMCCS_ShotboardVideoEditorRenderV1):
    OUTPUT_NODE = False


class IAMCCS_ShotboarderAudVidExporterPROLazyR45(IAMCCS_ShotboarderAudVidExporterPRO):
    """Non-terminal exporter used only behind the R45 optional Editor gate.

    The original exporter remains an OUTPUT_NODE for every historical workflow.
    This subclass is deliberately non-terminal so ComfyUI does not evaluate the
    Editor branch unless IAMCCS_MiniMaxH3EditorOptionalFinalR45 requests it.
    """

    OUTPUT_NODE = False


def _find_named_setting(value: Any, key: str):
    """Find one named Settings-PRO value without changing legacy shotplan truth."""
    if isinstance(value, dict):
        if key in value:
            return value.get(key)
        for child_key in ("settings", "resources", "outputs", "payload", "cine_payload"):
            child = value.get(child_key)
            found = _find_named_setting(child, key)
            if found is not None:
                return found
        for child in value.values():
            if isinstance(child, dict):
                found = _find_named_setting(child, key)
                if found is not None:
                    return found
    return None


class IAMCCS_MiniMaxH3EditorOptionalFinalR45:
    """Final lazy gate: direct joined master by default, Editor only when enabled.

    This gate lives *after* Universal delivery/master assembly. It never changes
    chunk joins, LongVid seams, audio stitching or queue continuation. With the
    Editor disabled, the exact normal final master remains the output.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"cine_linx": (SUPERNODE_LINX_TYPE,)},
            "optional": {
                "direct_video_path": ("STRING", {"lazy": True}),
                "editor_video_path": ("STRING", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("STRING", "BOOLEAN", "STRING")
    RETURN_NAMES = ("video_path", "editor_enabled", "report")
    FUNCTION = "select"
    OUTPUT_NODE = True
    CATEGORY = CATEGORY

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

    @staticmethod
    def _enabled(cine_linx: Any) -> bool:
        value = _find_named_setting(cine_linx, "h3_editor_enabled")
        # Missing means legacy topology/settings; this new gate is only used by
        # R45 workflows, where OFF is the explicit production default.
        return bool(value) if value is not None else False

    def check_lazy_status(self, cine_linx, direct_video_path=None, editor_video_path=None, **kwargs):
        if _shotplan(cine_linx).get("task_mode") == "reference_image":
            return []
        name = "editor_video_path" if self._enabled(cine_linx) else "direct_video_path"
        value = editor_video_path if name == "editor_video_path" else direct_video_path
        return [name] if value is None else []

    def select(self, cine_linx, direct_video_path=None, editor_video_path=None):
        if _shotplan(cine_linx).get("task_mode") == "reference_image":
            return {"ui": {}, "result": ("", False, "Video delivery idle · Image Generator selected")}
        enabled = self._enabled(cine_linx)
        path = str(editor_video_path if enabled else direct_video_path or "").strip()
        route = "EDITOR" if enabled else "DIRECT JOINED MASTER"
        report = (
            f"R45 optional editor gate | route={route} | "
            f"master={'ready' if path else 'waiting'} | {path or 'no final path yet'}"
        )
        return {"ui": {"text": [report]}, "result": (path, enabled, report)}

NODE_CLASS_MAPPINGS = {
    "IAMCCS_ShotboardVideoEditorOptionalR49": IAMCCS_ShotboardVideoEditorOptionalR49,
    "IAMCCS_ShotboardVideoEditorRenderOptionalR49": IAMCCS_ShotboardVideoEditorRenderOptionalR49,
    "IAMCCS_MiniMaxH3EditorDeliveryMedia": IAMCCS_MiniMaxH3EditorDeliveryMedia,
    "IAMCCS_MiniMaxH3DeliveryPreviewStripR42": IAMCCS_MiniMaxH3DeliveryPreviewStripR42,
    "IAMCCS_ShotboarderAudVidExporterPROLazyR45": IAMCCS_ShotboarderAudVidExporterPROLazyR45,
    "IAMCCS_MiniMaxH3EditorOptionalFinalR45": IAMCCS_MiniMaxH3EditorOptionalFinalR45,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_MiniMaxH3EditorDeliveryMedia": "MiniMax H3 · Delivery Checkpoint → Editor Chunk",
    "IAMCCS_MiniMaxH3DeliveryPreviewStripR42": "R42 Delivery Preview · First / Middle / Last",
    "IAMCCS_ShotboarderAudVidExporterPROLazyR45": "R45 Video Editor Exporter · Lazy",
    "IAMCCS_MiniMaxH3EditorOptionalFinalR45": "R45 Final Master / Optional Editor Gate",
}

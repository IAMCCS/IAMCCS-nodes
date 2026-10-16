"""SwapChar adapter for VHS sources with an optional audio stream."""
from collections.abc import Mapping
import logging


def optional_video_audio(audio):
    if audio is None or isinstance(audio, dict):
        return audio
    if isinstance(audio, Mapping):
        # VHS returns a lazy Mapping even for files without any audio stream.
        # Resolve it before ComfyUI's cache traversal attempts to iterate it.
        import av
        with av.open(audio.file) as container:
            if not container.streams.audio:
                return None
        return dict(audio)
    raise TypeError("The SwapChar video loader returned an invalid audio output")


class IAMCCS_H3SwapVideoLoader:
    ORIGINAL = "VHS_LoadVideo"
    RETURN_TYPES = ("IMAGE", "INT", "AUDIO", "VHS_VIDEOINFO")
    RETURN_NAMES = ("IMAGE", "frame_count", "audio", "video_info")
    FUNCTION = "load_video"
    CATEGORY = "IAMCCS/MiniMax H3/Internal"

    @classmethod
    def original(cls):
        import nodes
        return nodes.NODE_CLASS_MAPPINGS[cls.ORIGINAL]

    @classmethod
    def INPUT_TYPES(cls):
        original = cls.original()
        cls.RETURN_TYPES = original.RETURN_TYPES
        return original.INPUT_TYPES()

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return cls.original().IS_CHANGED(**kwargs)

    @classmethod
    def VALIDATE_INPUTS(cls, video):
        return cls.original().VALIDATE_INPUTS(video)

    def load_video(self, **kwargs):
        frames, count, audio, info = self.original()().load_video(**kwargs)
        audio = optional_video_audio(audio)
        logging.getLogger(__name__).info("H3 SwapChar video loaded | frames=%d | source audio=%s", count, "present" if audio is not None else "absent (silent video)")
        return frames, count, audio, info


class IAMCCS_H3SwapVideoPathLoader(IAMCCS_H3SwapVideoLoader):
    ORIGINAL = "VHS_LoadVideoPath"


LOADERS = {"VHS_LoadVideo": "IAMCCS_H3SwapVideoLoader", "VHS_LoadVideoPath": "IAMCCS_H3SwapVideoPathLoader"}
NODE_CLASS_MAPPINGS = {
    "IAMCCS_H3SwapVideoLoader": IAMCCS_H3SwapVideoLoader,
    "IAMCCS_H3SwapVideoPathLoader": IAMCCS_H3SwapVideoPathLoader,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_H3SwapVideoLoader": "H3 SwapChar · Video + Optional Audio",
    "IAMCCS_H3SwapVideoPathLoader": "H3 SwapChar · Video Path + Optional Audio",
}


def upgrade_swap_video_prompt(data):
    prompt = data.get("prompt", {})
    if not any(item.get("inputs", {}).get("task_mode") == "v2va_face_swap_lora"
               for item in prompt.values() if isinstance(item, dict)):
        return data
    sources = set()

    def visit(link, seen):
        if not isinstance(link, list) or len(link) != 2:
            return
        key = str(link[0])
        if key in seen:
            return
        seen.add(key)
        item = prompt.get(key, {})
        if item.get("class_type") in LOADERS:
            sources.add(key)
            return
        for value in item.get("inputs", {}).values():
            visit(value, seen)

    for item in prompt.values():
        if isinstance(item, dict) and item.get("class_type") == "IAMCCS_H3FaceSwapInput":
            visit(item.get("inputs", {}).get("source_video"), set())
    for key in sources:
        prompt[key]["class_type"] = LOADERS[prompt[key]["class_type"]]
    if sources:
        logging.getLogger(__name__).info("H3 SwapChar source loader upgraded | optional audio supported | loaders=%d", len(sources))
    return data

import importlib
import logging


LOGGER = logging.getLogger(__name__)


def _patch_seed_control_after_generate(node_cls):
    if node_cls is None or getattr(node_cls, "_iamccs_seed_control_after_generate_compat", False):
        return False

    original_input_types = getattr(node_cls, "INPUT_TYPES", None)
    if not callable(original_input_types):
        return False

    @classmethod
    def compat_input_types(cls):
        input_types = original_input_types()
        if not isinstance(input_types, dict):
            return input_types

        required = dict(input_types.get("required", {}))
        seed_spec = required.get("seed")
        if not (isinstance(seed_spec, tuple) and len(seed_spec) >= 2 and isinstance(seed_spec[1], dict)):
            return input_types

        seed_meta = dict(seed_spec[1])
        if seed_meta.get("control_after_generate") is True:
            return input_types

        seed_meta["control_after_generate"] = True
        updated_seed_spec = list(seed_spec)
        updated_seed_spec[1] = seed_meta
        required["seed"] = tuple(updated_seed_spec)

        patched = dict(input_types)
        patched["required"] = required
        return patched

    node_cls.INPUT_TYPES = compat_input_types
    node_cls._iamccs_seed_control_after_generate_compat = True
    node_cls._iamccs_original_input_types = original_input_types
    return True


def _apply_qwen_multigen_seed_widget_compat_patch():
    try:
        from . import iamccs_qwen_multigen as local_module

        patched = _patch_seed_control_after_generate(getattr(local_module, "IAMCCS_QwenMultiGen", None))
        if patched:
            LOGGER.info("IAMCCS compat: enabled local Qwen Multi-Gen seed widget patch")
            return
    except Exception as exc:
        LOGGER.debug("IAMCCS local Qwen Multi-Gen compat patch skipped: %s", exc)

    module_names = [
        "IAMCCS_QE_prompt_enhancer.nodes.qe_multi_gen",
        "custom_nodes.IAMCCS_QE_prompt_enhancer.nodes.qe_multi_gen",
    ]

    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            LOGGER.debug("IAMCCS Qwen Multi-Gen compat patch skipped for %s: %s", module_name, exc)
            continue

        patched = _patch_seed_control_after_generate(getattr(module, "IAMCCS_QwenMultiGen", None))
        if patched:
            LOGGER.info("IAMCCS compat: enabled Qwen Multi-Gen seed widget patch")
            return


def _apply_transformers_safetensors_metadata_compat_patch():
    try:
        import transformers.modeling_utils as modeling_utils
    except Exception as exc:
        LOGGER.debug("IAMCCS safetensors compat patch skipped: transformers unavailable: %s", exc)
        return

    original_safe_open = getattr(modeling_utils, "safe_open", None)
    if original_safe_open is None:
        return

    if getattr(modeling_utils, "_iamccs_safe_open_metadata_compat", False):
        return

    class _SafeOpenMetadataCompatWrapper:
        def __init__(self, handle):
            self._handle = handle

        def __enter__(self):
            entered = self._handle.__enter__()
            self._handle = entered
            return self

        def __exit__(self, exc_type, exc, tb):
            return self._handle.__exit__(exc_type, exc, tb)

        def metadata(self):
            metadata = self._handle.metadata()
            if metadata is None:
                return None
            if metadata.get("format") is not None:
                return metadata

            patched = dict(metadata)
            patched["format"] = "pt"
            return patched

        def __getattr__(self, name):
            return getattr(self._handle, name)

    def compat_safe_open(*args, **kwargs):
        return _SafeOpenMetadataCompatWrapper(original_safe_open(*args, **kwargs))

    modeling_utils.safe_open = compat_safe_open
    modeling_utils._iamccs_safe_open_metadata_compat = True
    modeling_utils._iamccs_original_safe_open = original_safe_open


def apply_iamccs_comfy_compat_patches():
    _apply_transformers_safetensors_metadata_compat_patch()
    _apply_qwen_multigen_seed_widget_compat_patch()

    try:
        import comfy.model_management as model_management
    except Exception as exc:
        LOGGER.debug("IAMCCS compat patch skipped: comfy.model_management unavailable: %s", exc)
        return

    loaded_model_cls = getattr(model_management, "LoadedModel", None)
    if loaded_model_cls is None:
        return

    if not getattr(loaded_model_cls, "_iamccs_tts_audio_suite_compat", False):
        original_model_mmap_residency = loaded_model_cls.model_mmap_residency
        original_is_dead = loaded_model_cls.is_dead

        def safe_model_mmap_residency(self, free=False):
            model = getattr(self, "model", None)
            if model is None:
                return 0, 0

            residency_fn = getattr(model, "model_mmap_residency", None)
            if callable(residency_fn):
                return residency_fn(free=free)

            LOGGER.debug(
                "IAMCCS compat: %s has no model_mmap_residency(); returning zero residency",
                type(model).__name__,
            )
            return 0, 0

        def safe_is_dead(self):
            real_model_ref = getattr(self, "real_model", None)
            if real_model_ref is None or not callable(real_model_ref):
                return False

            try:
                return real_model_ref() is not None and self.model is None
            except Exception:
                return False

        loaded_model_cls.model_mmap_residency = safe_model_mmap_residency
        loaded_model_cls.is_dead = safe_is_dead
        loaded_model_cls._iamccs_tts_audio_suite_compat = True
        loaded_model_cls._iamccs_original_model_mmap_residency = original_model_mmap_residency
        loaded_model_cls._iamccs_original_is_dead = original_is_dead

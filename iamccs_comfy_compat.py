import logging


LOGGER = logging.getLogger(__name__)


def apply_iamccs_comfy_compat_patches():
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

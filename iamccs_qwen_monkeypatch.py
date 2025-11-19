import logging


def _patch_qwen_wrapper_device_sync():
    try:
        from importlib import import_module
        mod = import_module("ComfyUI-QwenImageLoraLoader.wrappers.qwenimage")
        ComfyQwenImageWrapper = getattr(mod, "ComfyQwenImageWrapper", None)
        if ComfyQwenImageWrapper is None:
            return False
    except Exception:
        return False

    if getattr(ComfyQwenImageWrapper, "_iamccs_device_sync_patched", False):
        return True

    orig_execute = getattr(ComfyQwenImageWrapper, "_execute_model", None)
    if orig_execute is None:
        return False

    def _execute_model_patched(self, x, timestep, context, guidance, control, transformer_options, **kwargs):
        # Ensure all params/buffers are on the same device as input x (prevents CPU/CUDA mix)
        try:
            dev = getattr(x, "device", None)
            if dev is not None and dev.type in ("cuda", "cpu") and hasattr(self, "model") and self.model is not None:
                needs_move = False
                # Quick scan: if any param/buffer is on a different device, move whole model once
                for _, p in self.model.named_parameters(recurse=True):
                    if p.device.type != dev.type:
                        needs_move = True
                        break
                if not needs_move:
                    for _, b in self.model.named_buffers(recurse=True):
                        if b.device.type != dev.type:
                            needs_move = True
                            break
                if needs_move:
                    try:
                        self.model.to(dev)
                    except Exception:
                        pass
        except Exception:
            pass
        return orig_execute(self, x, timestep, context, guidance, control, transformer_options, **kwargs)

    setattr(ComfyQwenImageWrapper, "_execute_model", _execute_model_patched)
    setattr(ComfyQwenImageWrapper, "_iamccs_device_sync_patched", True)
    logging.getLogger(__name__).info("[IAMCCS] Applied device-sync monkeypatch to ComfyQwenImageWrapper")
    return True


# Execute at import time
try:
    _patch_qwen_wrapper_device_sync()
except Exception:
    pass

# SPDX-FileCopyrightText: 2026 Carmine Cristallo Scalzi (IAMCCS)
# SPDX-License-Identifier: GPL-3.0-or-later
"""Pika's released six-NFE H3 adapters and native Comfy sampling contract."""

PROFILE = "pika_h3_flex_6step"
SIGMAS = "1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0"
VALUES = dict(steps=6, sampler_name="lcm", scheduler="simple", denoise=1.0,
              shift_video=12.0, shift_audio=3.0)


def active(plan):
    return bool(plan.get("pika_flex", {}).get("enabled"))


def schema(names):
    result = {}
    for family in ("fl2va", "ref2va"):
        choices = [""] + [name for name in names if "pika" in name.lower()
                          and "flex" in name.lower() and family in name.lower()]
        result[f"pika_flex_{family}_lora"] = (choices, {
            "default": next(iter(choices[1:]), ""),
            "display_name": f"PIKA FLEX · {family.upper()} ADAPTER",
            "tooltip": "Native Comfy-format Pika Flex adapter. Its six-step recipe replaces other speed adapters.",
        })
    return result


def configure(plan, settings):
    enabled = str(plan.get("acceleration")) == PROFILE
    plan["pika_flex"] = {"enabled": enabled}
    if not enabled:
        return
    supported = {"t2va", "i2va", "fl2va", "ref2va", "v2va_face_swap_lora", "ref2va_vfx_edit", "ref2vid_lipsync", "auto_from_timeline"}
    if str(plan.get("task_mode")) not in supported:
        raise ValueError("Pika Flex: this continuation/experimental engine has no validated six-NFE branch. Use T2VA, I2VA, FL2VA or REF2VA.")
    if settings.get("h3_omni_enabled") or str(settings.get("turbo_mode", "off")) != "off":
        raise ValueError("Pika Flex: disable FastH3 Omni and additional Turbo; select PIKA FLEX · 6 STEP.")
    plan["pika_flex"].update({
        "adapters": {family: str(settings.get(f"pika_flex_{family}_lora", "") or "")
                     for family in ("fl2va", "ref2va")},
        "strength": 1.0, "sigmas": SIGMAS, "sampler": "lcm",
        "source": "https://huggingface.co/Pika-Labs/Pika-H3-Flex/blob/main/comfy/README.md",
    })
    plan["sampling"].update(VALUES)


def apply_adapter(model, plan, task):
    if not active(plan):
        return model, "off"
    import folder_paths
    import nodes
    family = "ref2va" if str(task).startswith("ref2") or str(task).startswith("v2va_face_swap") else "fl2va"
    name = plan["pika_flex"]["adapters"][family]
    if not name or not folder_paths.get_full_path("loras", name):
        raise ValueError(f"Pika Flex: select an installed {family.upper()} Comfy adapter in Settings PRO > SPEED.")
    patched = nodes.LoraLoaderModelOnly().load_lora_model_only(
        model=model, lora_name=name, strength_model=1.0)[0]
    return patched, f"Pika Flex {family} | {name}@1.0 | LCM / manual sigmas / 6 NFE"


class IAMCCS_MiniMaxH3PikaFlexLazyRouter:
    @classmethod
    def INPUT_TYPES(cls):
        types = ("IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING")
        return {"required": {"cine_linx": ("IAMCCS_SUPERNODE_LINX",)},
                "optional": {f"{prefix}_{name}": (kind, {"lazy": True})
                             for prefix in ("legacy", "variant")
                             for name, kind in zip(("frames", "audio", "bridge", "latent", "fps", "report"), types)}}

    RETURN_TYPES = ("IMAGE", "AUDIO", "IMAGE", "LATENT", "INT", "STRING")
    RETURN_NAMES = ("frames", "audio", "bridge", "sampled_latent", "fps", "report")
    FUNCTION = "select"
    CATEGORY = "IAMCCS/MiniMax H3/Pika Flex"

    @staticmethod
    def names(cine_linx):
        from .iamccs_minimax_h3_atomic_backend import _resolve_shotplan
        prefix = "variant" if active(_resolve_shotplan(cine_linx)) else "legacy"
        return [f"{prefix}_{name}" for name in ("frames", "audio", "bridge", "latent", "fps", "report")]

    def check_lazy_status(self, cine_linx, **kwargs):
        return [name for name in self.names(cine_linx) if kwargs.get(name) is None]

    def select(self, cine_linx, **kwargs):
        names = self.names(cine_linx)
        if any(kwargs.get(name) is None for name in names):
            raise ValueError("Pika Flex router: selected generation branch is incomplete.")
        return tuple(kwargs[name] for name in names)

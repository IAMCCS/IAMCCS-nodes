import logging

from .iamccs_supernodes_backend_panels import append_backend_panel_to_payload_and_report, merge_backend_panel_inputs
from .iamccs_supernodes_linx import SUPERNODE_LINX_TYPE, build_stage_linx_payload, linx_resource

_log = logging.getLogger("IAMCCS.SuperNodes.SecondStage")

try:
    import folder_paths  # type: ignore

    _LATENT_UPSCALE_MODEL_NAMES = tuple(folder_paths.get_filename_list("latent_upscale_models"))
except Exception:
    _LATENT_UPSCALE_MODEL_NAMES = (
        "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
    )


def _count_sigma_steps(sigmas_text):
    values = []
    for item in str(sigmas_text or "").replace("\n", ",").split(","):
        try:
            values.append(float(item.strip()))
        except Exception:
            pass
    return max(0, len(values) - 1)


class IAMCCS_SuperNodes_SecondStage:
    CATEGORY = "IAMCCS/SuperNodes"
    FUNCTION = "build_second_stage"
    RETURN_TYPES = (SUPERNODE_LINX_TYPE, "STRING")
    RETURN_NAMES = ("linx", "report")

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                **merge_backend_panel_inputs({
                    "second_stage_mode": (["off", "latent_upscale_refine", "latent_upscale_refine_x2_beta"],),
                    "stage2_model_policy": (["replace_stage1_if_connected", "stage2_model_if_connected", "prefer_stage2_else_primary", "keep_stage1_model"],),
                    "second_stage_upscale_model": (_LATENT_UPSCALE_MODEL_NAMES, {"default": _LATENT_UPSCALE_MODEL_NAMES[0]}),
                    "second_stage_reinject_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "second_stage_cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.1}),
                    "second_stage_manual_sigmas": ("STRING", {"default": "0.909375, 0.725, 0.421875, 0.0"}),
                }),
            },
            "optional": {
                "stage2_model": ("MODEL", {"lazy": True}),
                "linx": (SUPERNODE_LINX_TYPE, {"lazy": True}),
                "model": ("MODEL", {"lazy": True}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    def check_lazy_status(self, second_stage_mode, stage2_model_policy="replace_stage1_if_connected", stage2_model=None, model=None, linx=None, **kwargs):
        if str(second_stage_mode) == "off":
            return []
        if str(stage2_model_policy) == "keep_stage1_model":
            return [] if model is not None or linx is not None else ["model", "linx"]
        needed = []
        if stage2_model is None and model is None:
            needed.append("stage2_model")
        return needed

    def build_second_stage(
        self,
        second_stage_mode,
        stage2_model_policy,
        second_stage_upscale_model,
        second_stage_reinject_strength,
        second_stage_cfg,
        second_stage_manual_sigmas,
        backend_panel_mode="compact",
        backend_profile_preset="inherit",
        backend_memory_mode="inherit",
        backend_quality_mode="inherit",
        backend_settings_override="",
        stage2_model=None,
        linx=None,
        model=None,
        unique_id=None,
    ):
        second_stage_enabled = str(second_stage_mode) != "off"
        if not second_stage_enabled:
            second_stage_payload = "second_stage_mode=off; second_stage_enabled=false"
            report = "Second stage. mode=off | inactive | no model/resource exported | stage1 render path unchanged."
            second_stage_payload, report = append_backend_panel_to_payload_and_report(
                second_stage_payload,
                report,
                backend_panel_mode,
                backend_profile_preset,
                backend_memory_mode,
                backend_quality_mode,
                backend_settings_override,
            )
            linx_payload = build_stage_linx_payload(
                linx,
                "second_stage",
                "second_stage",
                {
                    "second_stage_mode": "off",
                    "second_stage_enabled": False,
                    "second_stage_payload": second_stage_payload,
                },
                report,
                unique_id=unique_id,
                slot_map={
                    "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
                },
                downstream_stages=["render", "vae"],
                policies={
                    "second_stage_enabled": False,
                    "stage2_model_policy": str(stage2_model_policy),
                    "second_stage_model_source": "disabled",
                    "second_stage_scale_mode": "off",
                },
                outputs={
                    "second_stage_payload": second_stage_payload,
                    "second_stage_enabled": False,
                    "second_stage_steps": 0,
                    "second_stage_scale_mode": "off",
                    "stage2_model_connected": False,
                    "second_stage_model_source": "disabled",
                    "anchor_refresh_requires_second_stage": False,
                },
            )
            _log.info("[IAMCCS Second Stage] %s", report)
            return (linx_payload, report)

        primary_model = linx_resource(linx, "model")
        connected_stage2_model = model if model is not None else stage2_model
        selected_model = primary_model
        policy = str(stage2_model_policy)
        if connected_stage2_model is not None:
            if policy in {"replace_stage1_if_connected", "stage2_model_if_connected", "prefer_stage2_else_primary"}:
                selected_model = connected_stage2_model
        elif selected_model is None:
            selected_model = connected_stage2_model

        stage2_steps = _count_sigma_steps(second_stage_manual_sigmas)
        model_source = "external_detailer_model" if connected_stage2_model is not None and selected_model is connected_stage2_model else "stage1_model"
        scale_mode = "x2_latent_upscale_beta" if str(second_stage_mode) == "latent_upscale_refine_x2_beta" else "same_resolution_refine"

        second_stage_payload = (
            f"second_stage_mode={second_stage_mode}; stage2_model_policy={stage2_model_policy}; "
            f"second_stage_upscale_model={second_stage_upscale_model}; second_stage_scale_mode={scale_mode}; "
            f"second_stage_reinject_strength={second_stage_reinject_strength}; "
            f"second_stage_cfg={second_stage_cfg}; second_stage_manual_sigmas={second_stage_manual_sigmas}; "
            f"second_stage_steps={int(stage2_steps)}; second_stage_model_source={model_source}"
        )
        report = (
            f"Second stage. mode={second_stage_mode} | scale={scale_mode} | policy={stage2_model_policy} | stage2_steps={int(stage2_steps)} | "
            f"stage1_model_in_linx={'yes' if primary_model is not None else 'no'} | detailer_model_connected={'yes' if connected_stage2_model is not None else 'no'} | "
            f"model_source={model_source} | anchor_refresh_note=render anchor refresh affects the per-segment second-stage reinjection path."
        )
        _log.info("[IAMCCS Second Stage] %s", report)
        second_stage_payload, report = append_backend_panel_to_payload_and_report(
            second_stage_payload,
            report,
            backend_panel_mode,
            backend_profile_preset,
            backend_memory_mode,
            backend_quality_mode,
            backend_settings_override,
        )
        linx_payload = build_stage_linx_payload(
            linx,
            "second_stage",
            "second_stage",
            {
                "second_stage_mode": str(second_stage_mode),
                "stage2_model_policy": str(stage2_model_policy),
                "second_stage_upscale_model": str(second_stage_upscale_model),
                "second_stage_scale_mode": str(scale_mode),
                "second_stage_reinject_strength": float(second_stage_reinject_strength),
                "second_stage_cfg": float(second_stage_cfg),
                "second_stage_manual_sigmas": str(second_stage_manual_sigmas),
                "second_stage_payload": second_stage_payload,
                "second_stage_steps": int(stage2_steps),
                "second_stage_model_source": model_source,
                "anchor_refresh_requires_second_stage": False,
            },
            report,
            unique_id=unique_id,
            slot_map={
                "linx": {"type": SUPERNODE_LINX_TYPE, "role": "stage_linx"},
            },
            downstream_stages=["render", "vae"],
            policies={
                "second_stage_enabled": str(second_stage_mode) != "off",
                "stage2_model_policy": str(stage2_model_policy),
                "second_stage_model_source": model_source,
                "second_stage_scale_mode": str(scale_mode),
            },
            outputs={
                "second_stage_payload": second_stage_payload,
                "second_stage_enabled": str(second_stage_mode) != "off",
                "second_stage_steps": int(stage2_steps),
                "second_stage_scale_mode": str(scale_mode),
                "stage2_model_connected": bool(connected_stage2_model is not None),
                "second_stage_model_source": model_source,
                "anchor_refresh_requires_second_stage": False,
            },
            resources={
                "second_stage_payload": second_stage_payload,
                "second_stage_model": selected_model,
            },
        )
        return (linx_payload, report)

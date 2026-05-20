from __future__ import annotations

import torch
from comfy_api.latest import io as comfy_io

try:
    from comfy_extras.nodes_lt import _append_guide_attention_entry
except Exception:  # pragma: no cover - depends on ComfyUI/KJ/LTX version
    _append_guide_attention_entry = None

from .iamccs_wdc_ltx_port import IAMCCS_CineFLFEngineSimple


class IAMCCS_CineFLFEngineSimpleDyno(IAMCCS_CineFLFEngineSimple):
    """Timeline/manual FLF engine with locked guide frames plus dynamic attention."""

    @classmethod
    def define_schema(cls):
        inputs = [
            comfy_io.Conditioning.Input("positive", tooltip="Positive conditioning to which guide keyframe info will be added"),
            comfy_io.Conditioning.Input("negative", tooltip="Negative conditioning to which guide keyframe info will be added"),
            comfy_io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images"),
            comfy_io.Latent.Input("latent", tooltip="Video latent, guides are added to this latent"),
            comfy_io.Image.Input("multi_input", tooltip="Batched images from the Cine Shotboard or Reference Board"),
        ]

        inputs.append(comfy_io.Int.Input(
            "num_images",
            default=0,
            min=0,
            max=50,
            step=1,
            display_name="images_loaded",
            tooltip="How many reference guide slots to read.",
        ))
        inputs.append(comfy_io.Combo.Input(
            "insert_mode",
            options=["frames", "seconds"],
            default="frames",
            tooltip="How manual_keyframes are interpreted.",
        ))
        inputs.append(comfy_io.Int.Input(
            "frame_rate",
            default=24,
            min=1,
            max=120,
            step=1,
            tooltip="FPS used when manual_keyframes/timeline_data use seconds.",
        ))

        inputs.append(comfy_io.String.Input(
            "timeline_data",
            default="",
            multiline=True,
            tooltip="Optional synced Cine Shotboard FLF timeline. When connected, this overrides manual keyframes.",
        ))
        inputs.append(comfy_io.String.Input(
            "manual_keyframes",
            default="0 | 1 | 1.00\n96 | 2 | 0.92\n184 | 3 | 0.88\n288 | 4 | 0.84\n-1 | 5 | 0.88",
            multiline=True,
            tooltip="Compact fallback. Frames mode: frame | ref | strength. Seconds mode: second | ref | strength.",
        ))

        return comfy_io.Schema(
            node_id="IAMCCS_CineFLFEngineSimpleDyno",
            display_name="IAMCCS Cine FLF Engine Simple Dyno",
            category="IAMCCS/Cine/02 Single Generation",
            description=(
                "Compact IAMCCS technical FLF engine. Dyno keeps append_keyframe image locking "
                "and also adds legacy guide attention entries for motion-heavy shots."
            ),
            inputs=inputs,
            outputs=[
                comfy_io.Conditioning.Output(display_name="positive"),
                comfy_io.Conditioning.Output(display_name="negative"),
                comfy_io.Latent.Output(display_name="latent", tooltip="Video latent with added FLF guides"),
            ],
        )

    @classmethod
    def _append_dyno_attention(cls, positive, negative, encoded, strength):
        if _append_guide_attention_entry is None:
            return positive, negative, False
        pre_filter_count = int(encoded.shape[2] * encoded.shape[3] * encoded.shape[4])
        guide_latent_shape = [int(v) for v in encoded.shape[2:]]
        positive, negative = _append_guide_attention_entry(
            positive,
            negative,
            pre_filter_count,
            guide_latent_shape,
            strength=float(strength),
        )
        return positive, negative, True

    @classmethod
    def execute(cls, positive, negative, vae, latent, multi_input, num_images, **kwargs) -> comfy_io.NodeOutput:
        scale_factors = vae.downscale_index_formula
        latent_image = latent["samples"].clone()

        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, _, _ = latent_image.shape
            noise_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=latent_image.device,
            )

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        batch_size = multi_input.shape[0] if multi_input is not None else 0
        insert_mode = kwargs.get("insert_mode", "frames")
        frame_rate = kwargs.get("frame_rate", 24)
        try:
            image_limit = max(0, min(50, int(num_images)))
        except Exception:
            image_limit = 0

        if image_limit <= 0 or batch_size <= 0:
            return comfy_io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

        synced_keyframes = cls._parse_synced_timeline(kwargs.get("timeline_data", ""), frame_rate)
        if synced_keyframes:
            guide_items = synced_keyframes
        else:
            guide_items = cls._parse_manual_keyframes(
                kwargs.get("manual_keyframes", ""),
                insert_mode,
                frame_rate,
                image_limit,
            )

        max_ref = min(batch_size, image_limit)
        applied = 0
        attention_entries = 0
        for guide in guide_items:
            i = int(guide.get("reference_index", 1))
            if i > max_ref:
                continue
            img = multi_input[i - 1:i]
            if img is None:
                continue

            f_idx = int(guide.get("frame", 0))
            strength = cls._clamp_float(guide.get("strength", 1.0), 0.0, 1.0, 1.0)
            image_1, encoded = cls.encode(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)
            assert latent_idx + encoded.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                encoded,
                strength,
                scale_factors,
            )
            positive, negative, did_attention = cls._append_dyno_attention(positive, negative, encoded, strength)
            applied += 1
            if did_attention:
                attention_entries += 1

        print(
            "[IAMCCS CineFLFEngineSimpleDyno] "
            f"guides={applied} guide_attention_available={_append_guide_attention_entry is not None} "
            f"guide_attention_entries={attention_entries}"
        )
        return comfy_io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})

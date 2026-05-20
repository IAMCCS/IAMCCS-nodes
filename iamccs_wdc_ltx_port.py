import io
import json
import os

import comfy.utils
import folder_paths
import numpy as np
import torch
from comfy_api.latest import io as comfy_io
from comfy_extras.nodes_lt import LTXVAddGuide

from PIL import Image, ImageOps


class IAMCCS_WDC_MultiImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_paths": ("STRING", {"default": "", "multiline": True}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "upscale_method": (["lanczos", "bilinear", "nearest-exact"],),
                "divisible_by": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 100, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",) * 51
    RETURN_NAMES = ("multi_output",) + tuple(f"image_{i + 1}" for i in range(50))
    FUNCTION = "load_images"
    CATEGORY = "IAMCCS/Cine/Legacy Compatibility"

    def load_images(self, image_paths, width, height, upscale_method, divisible_by, img_compression):
        results = []
        valid_paths = [p.strip() for p in str(image_paths or "").split("\n") if p.strip()]
        first_target_w, first_target_h = None, None

        for path in valid_paths:
            try:
                full_path = path
                if not os.path.exists(full_path):
                    full_path = os.path.join(folder_paths.get_input_directory(), path)
                if not os.path.exists(full_path):
                    print(f"Warning: Image path not found: {path}")
                    continue

                image = Image.open(full_path)
                image = ImageOps.exif_transpose(image)
                image = image.convert("RGB")

                orig_w, orig_h = image.size
                target_w, target_h = int(width), int(height)
                if target_w == 0 and target_h == 0:
                    target_w, target_h = orig_w, orig_h
                elif target_w == 0:
                    target_w = int(orig_w * (target_h / orig_h))
                elif target_h == 0:
                    target_h = int(orig_h * (target_w / orig_w))

                target_w = max(1, (target_w // int(divisible_by)) * int(divisible_by))
                target_h = max(1, (target_h // int(divisible_by)) * int(divisible_by))

                if first_target_w is None:
                    first_target_w, first_target_h = target_w, target_h
                else:
                    target_w, target_h = first_target_w, first_target_h

                if target_w != orig_w or target_h != orig_h:
                    resample = Image.LANCZOS if upscale_method == "lanczos" else Image.BILINEAR
                    image = image.resize((target_w, target_h), resample=resample)

                if int(img_compression) > 0:
                    img_byte_arr = io.BytesIO()
                    image.save(img_byte_arr, format="JPEG", quality=max(1, 100 - int(img_compression)))
                    image = Image.open(img_byte_arr)

                image_np = np.array(image).astype(np.float32) / 255.0
                results.append(torch.from_numpy(image_np)[None,])
            except Exception as e:
                print(f"Error loading {path}: {e}")

        if len(results) > 0:
            multi_output = torch.cat(results, dim=0)
        else:
            multi_output = torch.zeros((1, 64, 64, 3))
            results = [multi_output]

        padded_results = results + [torch.zeros((1, 64, 64, 3))] * (50 - len(results))
        return (multi_output, *padded_results[:50])


class IAMCCS_WDC_LTXKeyframer(comfy_io.ComfyNode):
    @classmethod
    def define_schema(cls):
        inputs = [
            comfy_io.Vae.Input("vae", tooltip="Video VAE used to encode the images"),
            comfy_io.Latent.Input("latent", tooltip="Video latent to insert images into"),
            comfy_io.Image.Input("multi_input", tooltip="Batched images from IAMCCS Cine Reference Board"),
        ]

        inputs.append(comfy_io.Int.Input(
            "num_images",
            default=1,
            min=0,
            max=50,
            step=1,
            display_name="images_loaded",
            tooltip="Select how many index/strength widgets to configure.",
        ))

        for i in range(1, 51):
            inputs.extend([
                comfy_io.Int.Input(
                    f"insert_frame_{i}",
                    default=0,
                    min=-9999,
                    max=9999,
                    step=1,
                    tooltip=f"Frame insert_frame for image {i} in pixel space.",
                    optional=True,
                ),
                comfy_io.Float.Input(
                    f"strength_{i}",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=f"Strength for image {i}.",
                    optional=True,
                ),
            ])

        return comfy_io.Schema(
            node_id="IAMCCS_WDC_LTXKeyframer",
            display_name="IAMCCS Cine LTX Keyframer (legacy alias)",
            category="IAMCCS/Cine/Legacy Compatibility",
            description="Legacy IAMCCS Cine-compatible LTX keyframer.",
            inputs=inputs,
            outputs=[
                comfy_io.Latent.Output(display_name="latent", tooltip="Latent with images inserted and noise mask updated."),
            ],
        )

    @classmethod
    def execute(cls, vae, latent, multi_input, num_images, **kwargs) -> comfy_io.NodeOutput:
        samples = latent["samples"].clone()
        scale_factors = vae.downscale_index_formula
        time_scale_factor, height_scale_factor, width_scale_factor = scale_factors

        batch, _, latent_frames, latent_height, latent_width = samples.shape
        width = latent_width * width_scale_factor
        height = latent_height * height_scale_factor

        if "noise_mask" in latent:
            conditioning_latent_frames_mask = latent["noise_mask"].clone()
        else:
            conditioning_latent_frames_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=samples.device,
            )

        batch_size = multi_input.shape[0] if multi_input is not None else 0

        for i in range(1, int(num_images) + 1):
            if i > batch_size:
                continue
            image = multi_input[i - 1:i]
            if image is None:
                continue

            insert_frame = kwargs.get(f"insert_frame_{i}")
            if insert_frame is None:
                continue
            strength = kwargs.get(f"strength_{i}", 1.0)

            if image.shape[1] != height or image.shape[2] != width:
                pixels = comfy.utils.common_upscale(
                    image.movedim(-1, 1),
                    width,
                    height,
                    "bilinear",
                    "center",
                ).movedim(1, -1)
            else:
                pixels = image
            encode_pixels = pixels[:, :, :, :3]
            t = vae.encode(encode_pixels)

            pixel_frame_count = (latent_frames - 1) * time_scale_factor + 1
            if insert_frame < 0:
                insert_frame = pixel_frame_count + insert_frame
            latent_idx = int(insert_frame) // time_scale_factor
            latent_idx = max(0, min(latent_idx, latent_frames - 1))
            end_index = min(latent_idx + t.shape[2], latent_frames)

            samples[:, :, latent_idx:end_index] = t[:, :, :end_index - latent_idx]
            conditioning_latent_frames_mask[:, :, latent_idx:end_index] = 1.0 - float(strength)

        return comfy_io.NodeOutput({"samples": samples, "noise_mask": conditioning_latent_frames_mask})


class IAMCCS_WDC_LTXSequencer(LTXVAddGuide):
    @classmethod
    def define_schema(cls):
        inputs = [
            comfy_io.Conditioning.Input("positive", tooltip="Positive conditioning to which guide keyframe info will be added"),
            comfy_io.Conditioning.Input("negative", tooltip="Negative conditioning to which guide keyframe info will be added"),
            comfy_io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images"),
            comfy_io.Latent.Input("latent", tooltip="Video latent, guides are added to the end of this latent"),
            comfy_io.Image.Input("multi_input", tooltip="Batched images from IAMCCS Cine Reference Board"),
        ]

        inputs.append(comfy_io.Int.Input(
            "num_images",
            default=1,
            min=0,
            max=50,
            step=1,
            display_name="images_loaded",
            tooltip="Select how many index/strength widgets to configure.",
        ))
        inputs.append(comfy_io.Combo.Input("insert_mode", options=["frames", "seconds"], default="frames", tooltip="Select the method for determining insertion points."))
        inputs.append(comfy_io.Int.Input("frame_rate", default=24, min=1, max=120, step=1, tooltip="Video FPS used for calculating second insertions."))

        for i in range(1, 51):
            inputs.extend([
                comfy_io.Int.Input(
                    f"insert_frame_{i}",
                    default=0,
                    min=-9999,
                    max=9999,
                    step=1,
                    tooltip=f"Frame insert_frame for image {i} in pixel space.",
                    optional=True,
                ),
                comfy_io.Float.Input(
                    f"insert_second_{i}",
                    default=0.0,
                    min=0.0,
                    max=9999.0,
                    step=0.1,
                    tooltip=f"Second insert point for image {i}.",
                    optional=True,
                ),
                comfy_io.Float.Input(
                    f"strength_{i}",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=f"Strength for image {i}.",
                    optional=True,
                ),
            ])

        return comfy_io.Schema(
            node_id="IAMCCS_WDC_LTXSequencer",
            display_name="IAMCCS Cine LTX Sequencer (legacy alias)",
            category="IAMCCS/Cine/Legacy Compatibility",
            description="Legacy IAMCCS Cine-compatible LTX sequencer.",
            inputs=inputs,
            outputs=[
                comfy_io.Conditioning.Output(display_name="positive"),
                comfy_io.Conditioning.Output(display_name="negative"),
                comfy_io.Latent.Output(display_name="latent", tooltip="Video latent with added guides"),
            ],
        )

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

        for i in range(1, int(num_images) + 1):
            if i > batch_size:
                continue
            img = multi_input[i - 1:i]
            if img is None:
                continue

            if insert_mode == "seconds":
                sec = kwargs.get(f"insert_second_{i}")
                if sec is None:
                    continue
                f_idx = int(float(sec) * float(frame_rate))
            else:
                f_idx = kwargs.get(f"insert_frame_{i}")
            if f_idx is None:
                continue
            strength = kwargs.get(f"strength_{i}", 1.0)

            image_1, t = cls.encode(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)
            assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                float(strength),
                scale_factors,
            )

        return comfy_io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


class IAMCCS_CineLTXSequencerExact(LTXVAddGuide):
    @classmethod
    def define_schema(cls):
        inputs = [
            comfy_io.Conditioning.Input("positive", tooltip="Positive conditioning to which guide keyframe info will be added"),
            comfy_io.Conditioning.Input("negative", tooltip="Negative conditioning to which guide keyframe info will be added"),
            comfy_io.Vae.Input("vae", tooltip="Video VAE used to encode the guide images"),
            comfy_io.Latent.Input("latent", tooltip="Video latent, guides are added to the end of this latent"),
            comfy_io.Image.Input("multi_input", tooltip="Batched guide images from the cine reference loader"),
        ]

        inputs.append(comfy_io.Int.Input(
            "num_images",
            default=1,
            min=0,
            max=50,
            step=1,
            display_name="images_loaded",
            tooltip="Select how many index/strength widgets to configure.",
        ))
        inputs.append(comfy_io.Combo.Input(
            "insert_mode",
            options=["frames", "seconds"],
            default="frames",
            tooltip="Select whether guide positions are read from frame or second widgets.",
        ))
        inputs.append(comfy_io.Int.Input(
            "frame_rate",
            default=24,
            min=1,
            max=120,
            step=1,
            tooltip="Video FPS used when insert_mode is seconds.",
        ))

        for i in range(1, 51):
            inputs.extend([
                comfy_io.Int.Input(
                    f"insert_frame_{i}",
                    default=0,
                    min=-9999,
                    max=9999,
                    step=1,
                    tooltip=f"Frame insert point for image {i}. Use -1 for the last frame.",
                    optional=True,
                ),
                comfy_io.Float.Input(
                    f"insert_second_{i}",
                    default=0.0,
                    min=0.0,
                    max=9999.0,
                    step=0.1,
                    tooltip=f"Second insert point for image {i}.",
                    optional=True,
                ),
                comfy_io.Float.Input(
                    f"strength_{i}",
                    default=1.0,
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    tooltip=f"Guide strength for image {i}.",
                    optional=True,
                ),
            ])

        return comfy_io.Schema(
            node_id="IAMCCS_CineLTXSequencerExact",
            display_name="IAMCCS Cine LTX Sequencer Exact",
            category="IAMCCS/Cine/FLF",
            description="Exact IAMCCS cine port of the proven LTX FLF sequencer. Use this only as the behavior baseline before adding filmmaker UI.",
            inputs=inputs,
            outputs=[
                comfy_io.Conditioning.Output(display_name="positive"),
                comfy_io.Conditioning.Output(display_name="negative"),
                comfy_io.Latent.Output(display_name="latent", tooltip="Video latent with added FLF guides"),
            ],
        )

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
        frame_rate = int(kwargs.get("frame_rate", 24) or 24)

        for i in range(1, int(num_images) + 1):
            if i > batch_size:
                continue
            img = multi_input[i - 1:i]
            if img is None:
                continue

            if insert_mode == "seconds":
                sec = kwargs.get(f"insert_second_{i}")
                if sec is None:
                    continue
                f_idx = int(float(sec) * frame_rate)
            else:
                f_idx = kwargs.get(f"insert_frame_{i}")
                if f_idx is None:
                    continue

            strength = kwargs.get(f"strength_{i}", 1.0)
            image_1, t = cls.encode(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), int(f_idx), scale_factors)
            assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                float(strength),
                scale_factors,
            )

        return comfy_io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


class IAMCCS_CineFLFEngineSimple(LTXVAddGuide):
    @classmethod
    def fingerprint_inputs(cls, *args, **kwargs):
        return float("nan")

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("nan")

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
            tooltip=(
                "Optional synced Cine Shotboard FLF timeline. When connected, this overrides the manual "
                "keyframe text so Stage 2 follows the main Shotboard."
            ),
        ))
        inputs.append(comfy_io.String.Input(
            "manual_keyframes",
            default="0 | 1 | 1.00\n96 | 2 | 0.92\n184 | 3 | 0.88\n288 | 4 | 0.84\n-1 | 5 | 0.88",
            multiline=True,
            tooltip="Compact fallback. Frames mode: frame | ref | strength. Seconds mode: second | ref | strength.",
        ))

        return comfy_io.Schema(
            node_id="IAMCCS_CineFLFEngineSimple",
            display_name="IAMCCS Cine FLF Engine Simple",
            category="IAMCCS/Cine/02 Single Generation",
            description="Compact IAMCCS technical FLF engine. Use timeline_data from Shotboard or compact manual_keyframes, not hundreds of visible widgets.",
            inputs=inputs,
            outputs=[
                comfy_io.Conditioning.Output(display_name="positive"),
                comfy_io.Conditioning.Output(display_name="negative"),
                comfy_io.Latent.Output(display_name="latent", tooltip="Video latent with added FLF guides"),
            ],
        )

    @staticmethod
    def _clamp_float(value, min_value=0.0, max_value=1.0, fallback=1.0):
        try:
            number = float(value)
        except Exception:
            number = float(fallback)
        return max(float(min_value), min(float(max_value), number))

    @staticmethod
    def _as_bool(value, default=True):
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in {"0", "false", "no", "off"}:
            return False
        if text in {"1", "true", "yes", "on"}:
            return True
        return default

    @classmethod
    def _parse_synced_timeline(cls, timeline_data, frame_rate):
        text = str(timeline_data or "").strip()
        if not text:
            return []

        rows = []
        try:
            data = json.loads(text)
            if isinstance(data, dict):
                rows = data.get("keyframes") or data.get("rows") or data.get("timeline") or data.get("shotboard") or []
            elif isinstance(data, list):
                rows = data
        except Exception:
            rows = []

        parsed = []
        if isinstance(rows, list):
            for idx, row in enumerate(rows):
                if not isinstance(row, dict):
                    continue
                if not cls._as_bool(row.get("use_guide", row.get("guide", True)), True):
                    continue
                try:
                    ref_idx = int(row.get("ref", row.get("image_ref", row.get("reference_index", idx + 1))))
                except Exception:
                    ref_idx = idx + 1
                try:
                    if row.get("frame") is not None:
                        frame_idx = int(round(float(row.get("frame"))))
                    else:
                        second = float(row.get("second", row.get("time", row.get("seconds", 0.0))))
                        frame_idx = int(round(second * max(1, int(frame_rate))))
                except Exception:
                    frame_idx = 0
                strength = cls._clamp_float(row.get("strength", row.get("force", 1.0)), 0.0, 1.0, 1.0)
                if strength <= 0.0:
                    continue
                parsed.append({
                    "reference_index": max(1, min(50, ref_idx)),
                    "frame": frame_idx,
                    "strength": strength,
                })

        if not parsed:
            for idx, raw_line in enumerate(text.splitlines()):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [part.strip() for part in line.split("|")]
                try:
                    second = float(parts[0].replace(",", "."))
                except Exception:
                    second = 0.0
                try:
                    ref_idx = int(parts[1]) if len(parts) > 1 else idx + 1
                except Exception:
                    ref_idx = idx + 1
                strength = cls._clamp_float(parts[2] if len(parts) > 2 else 1.0, 0.0, 1.0, 1.0)
                if strength <= 0.0:
                    continue
                parsed.append({
                    "reference_index": max(1, min(50, ref_idx)),
                    "frame": int(round(second * max(1, int(frame_rate)))),
                    "strength": strength,
                })

        return sorted(parsed[:50], key=lambda item: ((10**9 if int(item["frame"]) < 0 else int(item["frame"])), int(item["reference_index"])))

    @classmethod
    def _parse_manual_keyframes(cls, manual_keyframes, insert_mode, frame_rate, num_images):
        text = str(manual_keyframes or "").strip()
        if not text:
            text = "\n".join(f"{i - 1} | {i} | 1.0" for i in range(1, int(num_images) + 1))

        parsed = []
        for idx, raw_line in enumerate(text.splitlines()):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = [part.strip() for part in line.split("|")]
            try:
                first_value = float(parts[0].replace(",", "."))
            except Exception:
                first_value = float(idx)
            try:
                ref_idx = int(parts[1]) if len(parts) > 1 else idx + 1
            except Exception:
                ref_idx = idx + 1
            strength = cls._clamp_float(parts[2] if len(parts) > 2 else 1.0, 0.0, 1.0, 1.0)
            if strength <= 0.0:
                continue
            if str(insert_mode) == "seconds":
                frame_idx = int(round(first_value * max(1, int(frame_rate))))
            else:
                frame_idx = int(round(first_value))
            parsed.append({
                "reference_index": max(1, min(50, ref_idx)),
                "frame": frame_idx,
                "strength": strength,
            })
        return sorted(parsed[:50], key=lambda item: ((10**9 if int(item["frame"]) < 0 else int(item["frame"])), int(item["reference_index"])))

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
        for guide in guide_items:
            i = int(guide.get("reference_index", 1))
            if i > max_ref:
                continue
            img = multi_input[i - 1:i]
            if img is None:
                continue

            f_idx = int(guide.get("frame", 0))
            strength = cls._clamp_float(guide.get("strength", 1.0), 0.0, 1.0, 1.0)
            image_1, t = cls.encode(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = cls.get_latent_index(positive, latent_length, len(image_1), f_idx, scale_factors)
            assert latent_idx + t.shape[2] <= latent_length, "Conditioning frames exceed the length of the latent sequence."

            positive, negative, latent_image, noise_mask = cls.append_keyframe(
                positive,
                negative,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                strength,
                scale_factors,
            )

        return comfy_io.NodeOutput(positive, negative, {"samples": latent_image, "noise_mask": noise_mask})


class IAMCCS_WDC_LTXSequencerFixed5:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "multi_input": ("IMAGE",),
                "insert_frame_1": ("INT", {"default": 0, "min": -9999, "max": 9999, "step": 1}),
                "strength_1": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "insert_frame_2": ("INT", {"default": 96, "min": -9999, "max": 9999, "step": 1}),
                "strength_2": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "insert_frame_3": ("INT", {"default": 184, "min": -9999, "max": 9999, "step": 1}),
                "strength_3": ("FLOAT", {"default": 0.78, "min": 0.0, "max": 1.0, "step": 0.01}),
                "insert_frame_4": ("INT", {"default": 288, "min": -9999, "max": 9999, "step": 1}),
                "strength_4": ("FLOAT", {"default": 0.72, "min": 0.0, "max": 1.0, "step": 0.01}),
                "insert_frame_5": ("INT", {"default": -1, "min": -9999, "max": 9999, "step": 1}),
                "strength_5": ("FLOAT", {"default": 0.86, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", "STRING")
    RETURN_NAMES = ("positive", "negative", "latent", "report")
    FUNCTION = "execute"
    CATEGORY = "IAMCCS/Cine/Legacy Compatibility"

    def execute(
        self,
        positive,
        negative,
        vae,
        latent,
        multi_input,
        insert_frame_1,
        strength_1,
        insert_frame_2,
        strength_2,
        insert_frame_3,
        strength_3,
        insert_frame_4,
        strength_4,
        insert_frame_5,
        strength_5,
    ):
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
        frames = [insert_frame_1, insert_frame_2, insert_frame_3, insert_frame_4, insert_frame_5]
        strengths = [strength_1, strength_2, strength_3, strength_4, strength_5]
        batch_size = multi_input.shape[0] if multi_input is not None else 0
        applied = 0
        ops = []

        for idx, (frame, strength) in enumerate(zip(frames, strengths), start=1):
            if idx > batch_size:
                ops.append(f"skip image_{idx}: missing")
                continue
            img = multi_input[idx - 1:idx]
            if img is None:
                ops.append(f"skip image_{idx}: none")
                continue

            image_1, t = LTXVAddGuide.encode(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                positive,
                latent_length,
                len(image_1),
                int(frame),
                scale_factors,
            )
            if latent_idx + t.shape[2] > latent_length:
                ops.append(f"skip image_{idx}: frame={frame} out_of_range")
                continue

            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive,
                negative,
                int(frame_idx),
                latent_image,
                noise_mask,
                t,
                float(strength),
                scale_factors,
            )
            applied += 1
            ops.append(f"image_{idx}:f{frame_idx}->t{latent_idx},s={float(strength):.2f}")

        report = f"iamccs_cine_legacy_fixed5 applied={applied}/5 | " + "; ".join(ops)
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask}, report)



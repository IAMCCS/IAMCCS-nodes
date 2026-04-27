import logging
import math
import re

import node_helpers


log = logging.getLogger("IAMCCS.WanLongLength")


def _parse_indexed_prompt_bank(text: str) -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    pattern = re.compile(r"(?ms)^\s*\[(\d+)\]\s*\n(.*?)(?=^\s*\[\d+\]\s*\n|\Z)")
    matches = pattern.findall(normalized)
    if not matches:
        log.info("[WanPromptBank] indexed_parse found no indexed blocks")
        return []
    prompts = []
    for _, body in matches:
        cleaned = body.strip()
        if cleaned:
            prompts.append(cleaned)
    log.info("[WanPromptBank] indexed_parse count=%s", int(len(prompts)))
    return prompts


def _split_prompt_bank(text: str, separator_mode: str = "auto") -> list[str]:
    normalized = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        log.info("[WanPromptBank] split empty input mode=%s", str(separator_mode or "auto"))
        return []

    mode = str(separator_mode or "auto")
    if mode in ("auto", "indexed"):
        indexed = _parse_indexed_prompt_bank(normalized)
        if indexed:
            log.info("[WanPromptBank] split mode=%s resolved=indexed count=%s", mode, int(len(indexed)))
            return indexed
        if mode == "indexed":
            log.warning("[WanPromptBank] split mode=indexed but no indexed prompts were parsed")
            return []

    if mode == "line":
        prompts = [line.strip() for line in normalized.split("\n") if line.strip()]
        log.info("[WanPromptBank] split mode=%s resolved=line count=%s", mode, int(len(prompts)))
        return prompts

    if mode == "blank_line":
        prompts = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
        log.info("[WanPromptBank] split mode=%s resolved=blank_line count=%s", mode, int(len(prompts)))
        return prompts

    if "\n\n" in normalized:
        prompts = [part.strip() for part in re.split(r"\n\s*\n+", normalized) if part.strip()]
        log.info("[WanPromptBank] split mode=%s resolved=blank_line_auto count=%s", mode, int(len(prompts)))
        return prompts
    prompts = [line.strip() for line in normalized.split("\n") if line.strip()]
    log.info("[WanPromptBank] split mode=%s resolved=line_auto count=%s", mode, int(len(prompts)))
    return prompts


class IAMCCS_WanPromptLoopInfo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "separator_mode": (["auto", "indexed", "blank_line", "line"], {"default": "auto"}),
                "first_visible_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "continuation_visible_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "fps": ("FLOAT", {"default": 16.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "bootstrap_outside_loop": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "FLOAT", "STRING")
    RETURN_NAMES = (
        "prompt_count",
        "total_generations",
        "continuation_generations",
        "total_frames",
        "total_duration_s",
        "report",
    )
    FUNCTION = "analyze"
    CATEGORY = "IAMCCS/Wan"

    def analyze(self, text, separator_mode, first_visible_frames, continuation_visible_frames, fps, bootstrap_outside_loop):
        prompts = _split_prompt_bank(text, separator_mode)
        prompt_count = len(prompts)
        total_generations = int(prompt_count)
        fps = max(0.001, float(fps))
        first_visible_frames = max(1, int(first_visible_frames))
        continuation_visible_frames = max(1, int(continuation_visible_frames))

        if prompt_count <= 0:
            continuation_generations = 0
            total_frames = 0
        elif bootstrap_outside_loop:
            continuation_generations = max(0, prompt_count - 1)
            total_frames = first_visible_frames + max(0, prompt_count - 1) * continuation_visible_frames
        else:
            continuation_generations = prompt_count
            total_frames = prompt_count * continuation_visible_frames

        total_duration_s = float(total_frames) / fps if total_frames > 0 else 0.0
        report = (
            f"prompts={prompt_count} | total_generations={total_generations} | "
            f"continuations={continuation_generations} | total_frames={total_frames} | "
            f"duration={total_duration_s:.3f}s @ {fps:.3f}fps"
        )
        return (
            int(prompt_count),
            int(total_generations),
            int(continuation_generations),
            int(total_frames),
            float(total_duration_s),
            report,
        )


class IAMCCS_WanLongPlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "total_duration_s": ("FLOAT", {"default": 12.0, "min": 0.01, "max": 36000.0, "step": 0.01}),
                "fps": ("FLOAT", {"default": 16.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "first_visible_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "continuation_visible_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "first_motion_count": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "continuation_motion_count": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "end_overshoot_slots": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "use_end_frame_mode": (["all", "first_only", "continuations_only", "last_only", "off"], {"default": "all"}),
            }
        }

    RETURN_TYPES = (
        "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT", "INT",
        "INT", "INT", "INT", "INT", "INT", "INT", "STRING",
    )
    RETURN_NAMES = (
        "total_frames",
        "estimated_segments",
        "continuation_generations",
        "segment_index_out",
        "is_first_segment",
        "is_last_segment",
        "current_start_frame",
        "current_end_frame",
        "current_visible_frames",
        "remaining_frames_after",
        "current_motion_count",
        "use_prev_samples",
        "use_end_frame",
        "active_prompt_slot",
        "current_trim_slots",
        "report",
    )
    FUNCTION = "plan"
    CATEGORY = "IAMCCS/Wan"

    def _resolve_use_end_frame(self, mode: str, segment_index: int, estimated_segments: int) -> int:
        mode = str(mode or "all")
        is_first = segment_index == 0
        is_last = segment_index >= max(0, estimated_segments - 1)
        if mode == "off":
            return 0
        if mode == "first_only":
            return 1 if is_first else 0
        if mode == "continuations_only":
            return 0 if is_first else 1
        if mode == "last_only":
            return 1 if is_last else 0
        return 1

    def plan(
        self,
        total_duration_s: float,
        fps: float,
        first_visible_frames: int,
        continuation_visible_frames: int,
        segment_index: int,
        first_motion_count: int,
        continuation_motion_count: int,
        end_overshoot_slots: int,
        use_end_frame_mode: str,
    ):
        fps = max(0.001, float(fps))
        total_frames = max(1, int(round(max(0.01, float(total_duration_s)) * fps)))
        first_visible_frames = max(1, int(first_visible_frames))
        continuation_visible_frames = max(1, int(continuation_visible_frames))
        first_motion_count = max(0, int(first_motion_count))
        continuation_motion_count = max(0, int(continuation_motion_count))
        end_overshoot_slots = max(0, int(end_overshoot_slots))

        if total_frames <= first_visible_frames:
            estimated_segments = 1
        else:
            remaining_after_first = max(0, total_frames - first_visible_frames)
            estimated_segments = 1 + int(math.ceil(float(remaining_after_first) / float(continuation_visible_frames)))

        continuation_generations = max(0, estimated_segments - 1)
        segment_index_out = max(0, min(int(segment_index), estimated_segments - 1))
        is_first_segment = 1 if segment_index_out == 0 else 0
        is_last_segment = 1 if segment_index_out == estimated_segments - 1 else 0

        if segment_index_out == 0:
            current_start_frame = 0
            nominal_visible = first_visible_frames
        else:
            current_start_frame = first_visible_frames + (segment_index_out - 1) * continuation_visible_frames
            nominal_visible = continuation_visible_frames

        current_visible_frames = max(1, min(nominal_visible, total_frames - current_start_frame))
        current_end_frame = min(total_frames, current_start_frame + current_visible_frames)
        remaining_frames_after = max(0, total_frames - current_end_frame)
        current_motion_count = first_motion_count if is_first_segment else continuation_motion_count
        use_prev_samples = 0 if is_first_segment else 1
        use_end_frame = self._resolve_use_end_frame(use_end_frame_mode, segment_index_out, estimated_segments)
        current_trim_slots = end_overshoot_slots if use_end_frame else 0
        active_prompt_slot = min(segment_index_out + 1, 3)

        report = (
            f"total={total_frames}f @ {fps:.3f}fps | segments={estimated_segments} | "
            f"segment={segment_index_out} | range=[{current_start_frame}..{current_end_frame}) | "
            f"visible={current_visible_frames}f | remaining={remaining_frames_after}f | "
            f"motion_count={current_motion_count} | use_prev={use_prev_samples} | "
            f"use_end={use_end_frame} | overshoot={current_trim_slots}"
        )

        return (
            int(total_frames),
            int(estimated_segments),
            int(continuation_generations),
            int(segment_index_out),
            int(is_first_segment),
            int(is_last_segment),
            int(current_start_frame),
            int(current_end_frame),
            int(current_visible_frames),
            int(remaining_frames_after),
            int(current_motion_count),
            int(use_prev_samples),
            int(use_end_frame),
            int(active_prompt_slot),
            int(current_trim_slots),
            report,
        )


class IAMCCS_WanContinuityGuide:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "first_motion_count": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "continuation_motion_count": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "end_overshoot_slots": ("INT", {"default": 2, "min": 0, "max": 16, "step": 1}),
                "first_end_lock_slots": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "continuation_end_lock_slots": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "first_lock_start_slots": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "continuation_lock_start_slots": ("INT", {"default": 1, "min": 0, "max": 16, "step": 1}),
                "continuity_profile": (["demo1_default", "demo1_stronger", "soft_transition", "identity_first"], {"default": "demo1_default"}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT", "INT", "INT", "INT", "STRING")
    RETURN_NAMES = (
        "motion_latent_count",
        "use_prev_samples",
        "end_overshoot_slots_out",
        "end_lock_slots",
        "lock_start_slots",
        "prev_skip_last_slots",
        "latent_refresh_pct",
        "report",
    )
    FUNCTION = "guide"
    CATEGORY = "IAMCCS/Wan"

    def guide(
        self,
        segment_index: int,
        first_motion_count: int,
        continuation_motion_count: int,
        end_overshoot_slots: int,
        first_end_lock_slots: int,
        continuation_end_lock_slots: int,
        first_lock_start_slots: int,
        continuation_lock_start_slots: int,
        continuity_profile: str,
    ):
        is_first = int(segment_index) <= 0
        motion_latent_count = max(0, int(first_motion_count if is_first else continuation_motion_count))
        use_prev_samples = 0 if is_first else 1
        end_lock_slots = max(0, int(first_end_lock_slots if is_first else continuation_end_lock_slots))
        lock_start_slots = max(0, int(first_lock_start_slots if is_first else continuation_lock_start_slots))
        prev_skip_last_slots = 0
        latent_refresh_pct = 0

        profile = str(continuity_profile or "demo1_default")
        if profile == "demo1_stronger":
            prev_skip_last_slots = 1 if not is_first else 0
            latent_refresh_pct = 10 if not is_first else 0
        elif profile == "soft_transition":
            end_lock_slots = max(0, min(end_lock_slots, 1))
            lock_start_slots = max(0, min(lock_start_slots, 1))
        elif profile == "identity_first":
            prev_skip_last_slots = 1 if not is_first else 0
            latent_refresh_pct = 20 if not is_first else 0
            lock_start_slots = max(lock_start_slots, 1)

        report = (
            f"segment={int(segment_index)} | profile={profile} | motion_count={motion_latent_count} | "
            f"use_prev={use_prev_samples} | overshoot={int(end_overshoot_slots)} | end_lock={end_lock_slots} | "
            f"start_lock={lock_start_slots} | prev_skip_last={prev_skip_last_slots} | latent_refresh={latent_refresh_pct}%"
        )

        return (
            int(motion_latent_count),
            int(use_prev_samples),
            int(max(0, int(end_overshoot_slots))),
            int(end_lock_slots),
            int(lock_start_slots),
            int(prev_skip_last_slots),
            int(latent_refresh_pct),
            report,
        )


class IAMCCS_WanPromptPhasePlanner:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "mode": (["single", "three_phase", "cyclic_3"], {"default": "three_phase"}),
                "phase1_last_segment": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "phase2_last_segment": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("active_prompt_slot", "phase_index", "report")
    FUNCTION = "select"
    CATEGORY = "IAMCCS/Wan"

    def select(self, segment_index: int, mode: str, phase1_last_segment: int, phase2_last_segment: int):
        segment_index = max(0, int(segment_index))
        mode = str(mode or "three_phase")

        if mode == "single":
            active_prompt_slot = 1
            phase_index = 1
        elif mode == "cyclic_3":
            active_prompt_slot = (segment_index % 3) + 1
            phase_index = active_prompt_slot
        else:
            if segment_index <= int(phase1_last_segment):
                active_prompt_slot = 1
            elif segment_index <= int(phase2_last_segment):
                active_prompt_slot = 2
            else:
                active_prompt_slot = 3
            phase_index = active_prompt_slot

        report = f"segment={segment_index} | mode={mode} | prompt_slot={active_prompt_slot} | phase={phase_index}"
        return (int(active_prompt_slot), int(phase_index), report)


class IAMCCS_WanIndexedPromptEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "text": ("STRING", {"multiline": True, "default": ""}),
                "segment_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1, "lazy": True}),
                "index_offset": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1}),
                "selection_mode": (["direct_index", "three_phase", "cyclic_3", "single"], {"default": "direct_index"}),
                "phase1_last_segment": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
                "phase2_last_segment": ("INT", {"default": 1, "min": 0, "max": 100000, "step": 1}),
                "separator_mode": (["auto", "indexed", "blank_line", "line"], {"default": "auto"}),
                "fallback_mode": (["repeat_last", "wrap", "empty"], {"default": "repeat_last"}),
                "first_visible_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "continuation_visible_frames": ("INT", {"default": 81, "min": 1, "max": 8192, "step": 4}),
                "fps": ("FLOAT", {"default": 16.0, "min": 0.001, "max": 240.0, "step": 0.01}),
                "bootstrap_outside_loop": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "INT", "INT", "STRING", "INT", "INT", "INT", "FLOAT")
    RETURN_NAMES = (
        "conditioning",
        "selected_prompt",
        "selected_index",
        "prompt_count",
        "report",
        "total_generations",
        "continuation_generations",
        "total_frames",
        "total_duration_s",
    )
    FUNCTION = "encode"
    CATEGORY = "IAMCCS/Wan"

    def check_lazy_status(self, segment_index=None, **kwargs):
        # Break static validation cycles when segment_index is wired to an easy-use
        # loop index, but force the executor to resolve it before encode() runs.
        if segment_index is None:
            return ["segment_index"]
        return []

    def _pick_index(self, prompts: list[str], segment_index: int, selection_mode: str, phase1_last_segment: int, phase2_last_segment: int, fallback_mode: str) -> int:
        count = len(prompts)
        if count <= 0:
            return -1

        mode = str(selection_mode or "direct_index")
        seg = max(0, int(segment_index))
        if mode == "single":
            return 0
        if mode == "cyclic_3":
            return min((seg % 3), count - 1)
        if mode == "three_phase":
            if seg <= int(phase1_last_segment):
                return 0
            if seg <= int(phase2_last_segment):
                return min(1, count - 1)
            return min(2, count - 1)

        if seg < count:
            return seg
        if fallback_mode == "wrap":
            return seg % count
        if fallback_mode == "empty":
            return -1
        return count - 1

    def _loop_metrics(self, prompt_count: int, first_visible_frames: int, continuation_visible_frames: int, fps: float, bootstrap_outside_loop: bool):
        prompt_count = max(0, int(prompt_count))
        first_visible_frames = max(1, int(first_visible_frames))
        continuation_visible_frames = max(1, int(continuation_visible_frames))
        fps = max(0.001, float(fps))

        total_generations = prompt_count
        if prompt_count <= 0:
            continuation_generations = 0
            total_frames = 0
        elif bootstrap_outside_loop:
            continuation_generations = max(0, prompt_count - 1)
            total_frames = first_visible_frames + continuation_generations * continuation_visible_frames
        else:
            continuation_generations = prompt_count
            total_frames = prompt_count * continuation_visible_frames

        total_duration_s = float(total_frames) / fps if total_frames > 0 else 0.0
        return (int(total_generations), int(continuation_generations), int(total_frames), float(total_duration_s))

    def encode(
        self,
        clip,
        text,
        segment_index,
        index_offset,
        selection_mode,
        phase1_last_segment,
        phase2_last_segment,
        separator_mode="auto",
        fallback_mode="repeat_last",
        first_visible_frames=81,
        continuation_visible_frames=81,
        fps=16.0,
        bootstrap_outside_loop=True,
    ):
        if segment_index is None:
            segment_index = 0
        effective_index = int(segment_index) + int(index_offset)
        prompts = _split_prompt_bank(text, separator_mode)
        selected_index = self._pick_index(prompts, effective_index, selection_mode, int(phase1_last_segment), int(phase2_last_segment), str(fallback_mode))

        if selected_index < 0 or selected_index >= len(prompts):
            selected_prompt = ""
        else:
            selected_prompt = prompts[selected_index]

        tokens = clip.tokenize(selected_prompt or "")
        conditioning = clip.encode_from_tokens_scheduled(tokens)
        total_generations, continuation_generations, total_frames, total_duration_s = self._loop_metrics(
            len(prompts),
            first_visible_frames,
            continuation_visible_frames,
            fps,
            bool(bootstrap_outside_loop),
        )
        conditioning = node_helpers.conditioning_set_values(
            conditioning,
            {
                "_iamccs_generation_index": int(effective_index),
                "_iamccs_prompt_index": int(max(selected_index, 0)),
                "_iamccs_prompt_count": int(len(prompts)),
                "_iamccs_total_generations": int(total_generations),
                "_iamccs_continuation_generations": int(continuation_generations),
                "_iamccs_prompt_selection_mode": str(selection_mode or "direct_index"),
            },
        )
        report = (
            f"segment={int(segment_index)} | effective_index={effective_index} | mode={selection_mode} | prompt_index={selected_index} | "
            f"prompt_count={len(prompts)} | continuations={continuation_generations} | selected={'yes' if selected_prompt else 'empty'}"
        )
        preview = (selected_prompt or "").replace("\n", " ").strip()
        if len(preview) > 140:
            preview = preview[:137] + "..."
        log.info(
            "[WanIndexedPromptEncode] segment=%s effective_index=%s mode=%s selected_index=%s/%s total_generations=%s continuation_generations=%s preview=%r",
            int(segment_index),
            int(effective_index),
            str(selection_mode),
            int(max(selected_index, 0)),
            int(len(prompts)),
            int(total_generations),
            int(continuation_generations),
            preview,
        )
        return (
            conditioning,
            selected_prompt,
            int(max(selected_index, 0)),
            int(len(prompts)),
            report,
            int(total_generations),
            int(continuation_generations),
            int(total_frames),
            float(total_duration_s),
        )


class IAMCCS_WanImageBatchFrameSelect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "mode": (["last", "first", "index", "from_end_index"], {"default": "last"}),
                "frame_index": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT", "STRING")
    RETURN_NAMES = ("image", "selected_index", "batch_count", "report")
    FUNCTION = "select"
    CATEGORY = "IAMCCS/Wan"

    def select(self, images, mode, frame_index):
        batch_count = int(images.shape[0]) if getattr(images, "shape", None) is not None else 0
        if batch_count <= 0:
            raise ValueError("IAMCCS_WanImageBatchFrameSelect received an empty image batch")

        mode = str(mode or "last")
        raw_index = max(0, int(frame_index))
        if mode == "first":
            selected_index = 0
        elif mode == "index":
            selected_index = min(raw_index, batch_count - 1)
        elif mode == "from_end_index":
            selected_index = max(0, batch_count - 1 - raw_index)
        else:
            selected_index = batch_count - 1

        selected = images[selected_index:selected_index + 1]
        report = f"mode={mode} | selected_index={selected_index} | batch_count={batch_count}"
        return (selected, int(selected_index), int(batch_count), report)

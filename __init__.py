# ==========================================================
# __init__.py — Registro nodi IAMCCS
# ==========================================================

import logging
import os
import shutil
from pathlib import Path


def _iamccs_expose_ffmpeg():
    """Expose the bundled imageio-ffmpeg binary as ffmpeg.exe for Comfy nodes."""
    try:
        import imageio_ffmpeg

        ffmpeg_path = Path(imageio_ffmpeg.get_ffmpeg_exe())
        if not ffmpeg_path.exists():
            return
        shim_dir = Path(os.environ.get("IAMCCS_FFMPEG_SHIM_DIR", r"D:\ComfyUI\python_embeded\ffmpeg-bin"))
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / "ffmpeg.exe"
        if not shim_path.exists():
            try:
                os.link(str(ffmpeg_path), str(shim_path))
            except Exception:
                shutil.copy2(str(ffmpeg_path), str(shim_path))
        os.environ["IMAGEIO_FFMPEG_EXE"] = str(shim_path)
        os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        logging.getLogger(__name__).info("[IAMCCS] FFmpeg exposed at %s", shim_path)
    except Exception as exc:
        logging.getLogger(__name__).warning("[IAMCCS] FFmpeg auto-expose skipped: %s", exc)


_iamccs_expose_ffmpeg()

from .iamccs_comfy_compat import apply_iamccs_comfy_compat_patches

apply_iamccs_comfy_compat_patches()

# ComfyUI frontend assets
WEB_DIRECTORY = "web"

from .iamccs_wan_lora_stack import (
    IAMCCS_WanLoRAStack,
    IAMCCS_ModelWithLoRA,
)
from .iamccs_wan_lora_stack_simple import (
    IAMCCS_WanLoRAStackModelIO,
)
from .iamccs_wan_lora_schedule import (
    IAMCCS_WanLoRASchedule,
)
from .iamccs_wan_lora_hooks import (
    IAMCCS_WanLoRAHookSchedule,
    IAMCCS_ApplyLoRAHooksToConditioning,
    IAMCCS_ApplyScheduledWanLoRAFromConditioning,
    IAMCCS_BuildScheduledWanModelBank,
    IAMCCS_SelectScheduledWanModelFromConditioning,
    IAMCCS_SelectScheduledWanModelPairFromConditioning,
)
from .iamccs_wan_lora_runtime_bridge import (
    IAMCCS_WanLoRARuntimeBridge,
    IAMCCS_ModelWithLoRA_RuntimeBridge,
)

from .iamccs_ltx2_lora_stack import (
    IAMCCS_LTX2_LoRAStack,
    IAMCCS_LTX2_LoRAStackStaged,
    IAMCCS_ModelWithLoRA_LTX2,
    IAMCCS_ModelWithLoRA_LTX2_Staged,
    IAMCCS_LTX2_LoRAStackModelIO,
)

from .iamccs_ltx2_lora_stack_segmented6 import (
    IAMCCS_LTX2_LoRAStackSegmented6,
    IAMCCS_LTX2_ModelWithLoRA_Segmented6,
)

from .iamccs_ltx2_tools import (
    IAMCCS_LTX2_FrameRateSync,
    IAMCCS_LTX2_Validator,
    IAMCCS_LTX2_TimeFrameCount,
    IAMCCS_LTX2_EnsureFrames8nPlus1,
    IAMCCS_LTX2_EnsureMinFrames,
    IAMCCS_LTX2_ControlPreprocess,
    IAMCCS_LTX2_ImageBatchPadReflect,
    IAMCCS_LTX2_ImageBatchCropByPad,
    IAMCCS_SegmentPlanner,
    IAMCCS_SegmentPlannerSettings,
    IAMCCS_AudioSegmentAutoPlanner,
    IAMCCS_SegmentPlannerLinked,
    IAMCCS_SegmentPlanFromPlanner,
    IAMCCS_SourceRangeFromSegmentPlan,
    IAMCCS_TwoSegmentPlanner,
    IAMCCS_ThreeSegmentPlanner,
    IAMCCS_SegmentSwitch,
)

from .iamccs_ltx2_extension_module import (
    IAMCCS_LTX2_ExtensionModule,
    IAMCCS_LTX2_ExtensionModule_Disk,
    IAMCCS_LTX2_JointRefreshLatent,
    IAMCCS_LTX2_JointRefreshLatent_Disk,
    IAMCCS_LoadImagesFromDirLite,
    IAMCCS_ImageBatchRangeLite,
    IAMCCS_SourceFramesToDisk,
    IAMCCS_StartDirToVideoLatent,
    IAMCCS_StartImagesToVideoLatent,
    IAMCCS_VideoCombineFromDir,
    IAMCCS_LTX2_ExtensionModule_simple,
    IAMCCS_LTX2_GetImageFromBatch,
    IAMCCS_LTX2_ReferenceImageSwitch,
    IAMCCS_LTX2_ReferenceStartFramesInjector,
    IAMCCS_LTX2_FrameCountValidator,
    IAMCCS_LTX2_FirstLastFramesController,
    IAMCCS_LTX2_ContextLatent,
    IAMCCS_LTX2_MiddleFrames,
    IAMCCS_LTX2_FirstLastLatentControl,
    IAMCCS_LTX2_FirstLastLatentControl_Pro,
)

from .iamccs_ltx_guide_port import (
    IAMCCS_WDC_MultiImageLoader,
    IAMCCS_WDC_LTXKeyframer,
    IAMCCS_WDC_LTXSequencer,
    IAMCCS_CineLTXSequencerExact,
    IAMCCS_CineFLFEngineSimple,
    IAMCCS_WDC_LTXSequencerFixed5,
)

from .iamccs_cine_nodes import (
    IAMCCS_CineReferenceBoard,
    IAMCCS_CineLTXSequencer,
    IAMCCS_CineAllInOneFLFEngine,
    IAMCCS_CinePromptRelayTimeline,
    IAMCCS_CineShotboardTimelinePro,
    IAMCCS_CineShotboardPlannerPro,
    IAMCCS_CineShotboardPlannerProV2,
    IAMCCS_CineShotboardPlannerV3,
    IAMCCS_CineShotboardLite,
    IAMCCS_CineShotboardPlannerProLegacy,
    IAMCCS_CineInfo,
    IAMCCS_CineInfoV2,
    IAMCCS_CineFLFProductor,
    IAMCCS_CineFilmmaker,
    IAMCCS_CineFilmmakerBackend,
    IAMCCS_CineShotboardBackendPro,
    IAMCCS_CineFilmmakerGuide,
    IAMCCS_CineFilmmakerGuide1to1,
    IAMCCS_CineFilmmakerCropGuides1to1,
    IAMCCS_CineSwitch,
    IAMCCS_CinePromptRelayLatentShapeSync,
    IAMCCS_CineFLFLengthCompensator,
    IAMCCS_CinePromptRelaySafeEncode,
    IAMCCS_CineRelayOrBypass,
    IAMCCS_CinePromptArchitect,
    IAMCCS_BoardMaker,
    IAMCCS_CineMusicVideoPlanner,
    IAMCCS_CineShotPlanner,
    IAMCCS_CineRefLatentControl,
    IAMCCS_CineAudioPromptDirector,
    IAMCCS_CinePromptRelayAdapter,
    IAMCCS_CinePromptComposer,
    IAMCCS_CineShotLineBuilder,
    IAMCCS_CineV2VTimelineLineBuilder,
    IAMCCS_CineLineStacker,
    IAMCCS_CineMultiGenDirector,
    IAMCCS_CineShotAudioDirector,
    IAMCCS_CineV2VTimelineDirector,
    IAMCCS_CineV2VAssetSelector,
    IAMCCS_CineWorkflowInspector,
)
from .iamccs_cine_shotboard_planner_v4 import IAMCCS_CineShotboardPlannerV4
from .iamccs_cine_shotboard_planner_v5_v2v import (
    IAMCCS_CineShotboardPlannerV5V2V,
    IAMCCS_CineShotboardV5V2VRouter,
)
from .iamccs_cine_shotboard_v4_backend import (
    IAMCCS_CineShotboardV4Backend,
    IAMCCS_CineShotboardV4Guide,
    IAMCCS_CineShotboardV4CropGuides,
    IAMCCS_LTXVideoDurationCrop,
)
from .iamccs_cine_resolution_parity import IAMCCS_CineResolutionParityTranslator
from .iamccs_cine_stage_switch import IAMCCS_CineStage2BypassSwitch
from .iamccs_cine_stage2_preview_toggle import IAMCCS_CineStage2PreviewToggle
from .iamccs_cine_flf_productor_dyno import IAMCCS_CineFLFProductorDyno
from .iamccs_cine_flf_engine_simple_dyno import IAMCCS_CineFLFEngineSimpleDyno
from .iamccs_cine_duration_lock import (
    IAMCCS_CineBoardDurationLock,
    IAMCCS_CineLatentDurationCrop,
    IAMCCS_CineShotboardTailTrimPolicy,
    IAMCCS_CineShotboardFinalFrameTrim,
)
from .iamccs_shotboarder_exporter_pro import IAMCCS_ShotboarderAudVidExporterPRO
from .iamccs_cine_temporal_cut_barrier import IAMCCS_CineTemporalCutBarrier

from .iamccs_ltx2_temporal_overlap_samplers import (
    IAMCCS_LTX2_ConditionNextLatentWithPrevOverlap,
    IAMCCS_LTX2_InitLatentSampler,
    IAMCCS_LTX2_LoopingSampler,
    IAMCCS_LTX2_OneShotLowRAMLooper,
    IAMCCS_LTX2_ExtendSampler,
)

from .iamccs_wan_svipro_motion import (
    IAMCCS_WanImageMotion,
    WanImageMotionPro as WanImageMotionProPlus,
    IAMCCS_WanImageMotionPro_Simple as IAMCCS_WanImageMotionProPlus_Simple,
    IAMCCS_WanImageMotionInductive,
    IAMCCS_WanSVIToFLFBridgePro as IAMCCS_WanSVIToFLFBridgeProPlus,
    IAMCCS_WanSVIToFLFBridgePro_Simple as IAMCCS_WanSVIToFLFBridgeProPlus_Simple,
    WanMotionProTrimmer,
    IAMCCS_WanPrevTailPrep,
)

from .iamccs_wan_long_length import (
    IAMCCS_WanSviFlfTimeline,
    IAMCCS_WanSviFlfTimelinePick,
    IAMCCS_WanLongPlanner,
    IAMCCS_WanContinuityGuide,
    IAMCCS_WanPromptPhasePlanner,
    IAMCCS_WanPromptLoopInfo,
    IAMCCS_WanIndexedPromptEncode,
    IAMCCS_WanImageBatchFrameSelect,
)

from .iamccs_wan_svipro_motion_legacy import (
    WanImageMotionProLegacy,
)

from .iamccs_autolink import (
    IAMCCS_SetAutoLink,
    IAMCCS_GetAutoLink,
    IAMCCS_AutoLinkConverter,
    IAMCCS_AutoLinkArguments,
)

from .iamccs_gguf_accelerator import (
    IAMCCS_GGUF_accelerator,
)

from .iamccs_sampler_advanced_v1 import (
    IAMCCS_SamplerAdvancedVersion1,
)

from .iamccs_bus_group import (
    IAMCCS_bus_group,
)

from .iamccs_image_resize import (
    IAMCCS_ImageResizeBatchSafe,
    IAMCCS_LoadResizeSegmentFromDir,
)

from .iamccs_multiswitch import (
    IAMCCS_MultiSwitch,
)

from .iamccs_lazy_switch import (
    IAMCCS_LazyAnySwitch,
)

from .iamccs_navigator import (
    IAMCCS_Navigator,
)

from .iamccs_hw_supporter import (
    IAMCCS_HwSupporter,
    IAMCCS_HwSupporterAny,
    IAMCCS_HardMemoryPurge,
    IAMCCS_VRAMCleanup,
    IAMCCS_VRAMFlushLatent,
    IAMCCS_VAEDecodeTiledSafe,
    IAMCCS_VAEDecodeToDisk,
)

from .iamccs_hw_probe_node import (
    IAMCCS_HWProbeRecommendations,
)

from .iamccs_detail_atelier import (
    IAMCCS_DetailAtelier,
    IAMCCS_DetailAtelierAdvanced,
    IAMCCS_DetailAtelierSampler,
)

from .iamccs_qwen_vl_flf import (
    IAMCCS_QWEN_VL_FLF,
    IAMCCS_QWEN_VL_FLF_Advanced,
)

from .iamccs_move_ahead import (
    IAMCCS_MoveAhead,
    IAMCCS_MoveAheadEnforcer,
    IAMCCS_MotionScale,
    IAMCCS_MotionScaleAdvanced,
)

from .iamccs_motion_bridge import (
    IAMCCS_MotionBridgeSave,
    IAMCCS_MotionBridgeLoad,
    IAMCCS_LatentTailSlice,
)

from .iamccs_audio_extender import (
    IAMCCS_AudioExtensionMath,
    IAMCCS_AudioExtender,
    IAMCCS_AudioTimelineAssembler,
    IAMCCS_AudioTimelineGate,
)

from .iamccs_cine_audio_dialogue import (
    IAMCCS_CineSpeech1PromptCompiler,
    IAMCCS_CineAudioTranscriptPromptCompiler,
    IAMCCS_CineVideoToWooshInputs,
    IAMCCS_CineTimelineAudioMixer,
    IAMCCS_AudioBoardArranger,
    IAMCCS_CineDialogueLineRouter,
    IAMCCS_CineInfo3,
    IAMCCS_BoardMaker_DialogueFoley,
    IAMCCS_CineSpeechLength,
    IAMCCS_CineDialogueDurationPlanner,
    IAMCCS_CineAudioDurationProbe,
    IAMCCS_CineDialogueTimingReconciler,
    IAMCCS_CineWooshFoleyChunkPlanner,
    IAMCCS_CineFinalAudioMixer,
    IAMCCS_CineEmotionButtons,
    IAMCCS_CineDialoguePromptKit,
)
from .audio.audio_bus_out import IAMCCS_BusOut
from .iamccs_audioboard_direct_adapter import IAMCCS_AudioBoardDirectShotboardAdapter
from .audio.audio_board_mixer import IAMCCS_AudioBoardMixer
from .audio.audio_control_efx import IAMCCS_ControlAudEfx
from .audio.audio_control_efx_panel import IAMCCS_ControlAudEfxPanel
from .audio.dialogue_tag_editor import IAMCCS_DialogueTagEditor, IAMCCS_DialogueAudioBoardBridge
from .audio.cine_audio_info import IAMCCS_CineAudioInfo
from .audio.audio_phase_lazy import IAMCCS_AudioPhaseLazyGate
from .iamccs_ideogram_storyboard_frame_designer import (
    IAMCCS_StoryboardFrameDesigner,
    IAMCCS_StoryboardFrameDesignerV2,
    IAMCCS_IdeoInfo,
    IAMCCS_IdeoInpaintPrep,
    IAMCCS_IdeoMaskedPixels,
    IAMCCS_IdeogramJSONPreviewPass,
    IAMCCS_IdeoNoiseDebug,
    IAMCCS_IdeoRandomNoiseNoCache,
    IAMCCS_IdeoImageHashDebug,
)
from .iamccs_ideo_translate import IAMCCS_IdeoTranslate
from .iamccs_ideogram_storyboard_sheet import IAMCCS_IdeogramStoryboardSheet, IAMCCS_StoryboardCaptionSheet
from .iamccs_ideogram_sheet_builder import IAMCCS_IdeogramSheetBuilder
from .iamccs_storyboard_auto_crop import IAMCCS_StoryboardAutoCropGrid, IAMCCS_StoryboardAutoCropGridPRO
from .iamccs_target_crop import IAMCCS_TargetCrop
from .iamccs_gemma_assist import IAMCCS_GemmaAssistLazyGate, IAMCCS_GemmaAssistOutput
from .iamccs_storyboard_prompt_contact_sheet import IAMCCS_StoryboardPromptContactSheet
from .iamccs_goyai_paint import IAMCCS_GoyAICanvasPaint
from .iamccs_flashvsr_bridge import IAMCCS_FlashVSRPanelBatchPrep, IAMCCS_FlashVSRPanelBatchRestore

from .iamccs_ltx2_segment_queue import (
    IAMCCS_LTX2_BlendLatentBridge,
    IAMCCS_LTX2_LastFrameBridgeLoad,
    IAMCCS_LTX2_LastFrameBridgeSave,
    IAMCCS_LTX2_LoadLatentBridge,
    IAMCCS_LTX2_LongVideoWrapperPrep,
    IAMCCS_LTX2_LongVideoWrapperPrepDisk,
    IAMCCS_LTX2_SaveLatentBridge,
    IAMCCS_LTX2_SegmentQueueLoop,
)

from .iamccs_image_resize import (
    IAMCCS_ImageResizeBatchSafe,
)

from .iamccs_value_monitor import (
    IAMCCS_IntValueMonitor,
)

from .iamccs_flux_klein_multigen import (
    IAMCCS_FluxKleinMultiGen,
    IAMCCS_FluxKleinRefine,
    IAMCCS_ImageBatch6,
)

from .iamccs_qwen_multigen import (
    IAMCCS_QwenMultiGen,
)

from .iamccs_multiline_prompt_splitter import (
    IAMCCS_MultilinePromptSplitter8,
)

from .iamccs_supernode_modular import (
    IAMCCS_SupernodeBase,
    IAMCCS_SupernodeModule,
)

from .iamccs_auimg2vid_goal1 import (
    IAMCCS_ProjectTimelinePlanner,
    IAMCCS_Ltx2HelperModules_ProjectTimelinePlanner,
    IAMCCS_Ltx2HelperModules_Planner,
    IAMCCS_Ltx2HelperModules_AudioTimeline,
    IAMCCS_Ltx2HelperModules_KeyframeTimeline,
    IAMCCS_Ltx2HelperModules_RefreshPolicy,
    IAMCCS_Ltx2HelperModules_ReanchorLatent,
    IAMCCS_Ltx2HelperModules_DiskExtension,
    IAMCCS_Ltx2HelperModules_RuntimeBridge,
    IAMCCS_Ltx2HelperModules_Continuity,
    IAMCCS_Ltx2HelperModules_Finalize,
    IAMCCS_AUIMG2VID_ProjectTimelinePlanner,
    IAMCCS_AUIMG2VID_Planner,
    IAMCCS_AUIMG2VID_AudioTimeline,
    IAMCCS_AUIMG2VID_KeyframeTimeline,
    IAMCCS_AUIMG2VID_RefreshPolicy,
    IAMCCS_AUIMG2VID_ReanchorLatent,
    IAMCCS_AUIMG2VID_DiskExtension,
    IAMCCS_AUIMG2VID_RuntimeBridge,
    IAMCCS_AUIMG2VID_Continuity,
    IAMCCS_AUIMG2VID_Finalize,
)

from .iamccs_supernodes_exec import (
    IAMCCS_SuperNodes_AUIMG2VIDExecutablePlanner,
    IAMCCS_SuperNodes_AUIMG2VIDExecutableRender,
    IAMCCS_SuperNodes_AUIMG2VIDExecutableVAE,
    IAMCCS_SuperNodes_AUIMG2VIDExecutableFinalize,
)
from .iamccs_supernodes_second_stage import IAMCCS_SuperNodes_SecondStage

try:
    from .iamccs_scail_identity import (
        IAMCCS_ScailIdentitySeeder,
        IAMCCS_ScailIdentityTracker,
        IAMCCS_ScailMultiReference,
    )
except Exception as exc:
    logging.warning("IAMCCS SCAIL Identity nodes unavailable: %s", exc)
    IAMCCS_ScailIdentitySeeder = None
    IAMCCS_ScailIdentityTracker = None
    IAMCCS_ScailMultiReference = None

try:
    from .iamccs_scail_extends import (
        IAMCCS_ScailExtends,
        IAMCCS_ScailExtendPlan,
    )
except Exception as exc:
    logging.warning("IAMCCS SCAIL Extends nodes unavailable: %s", exc)
    IAMCCS_ScailExtends = None
    IAMCCS_ScailExtendPlan = None

try:
    from .iamccs_wananimate_extends import (
        IAMCCS_WanAnimateExtends,
        IAMCCS_WanAnimateExtendPlan,
    )
except Exception as exc:
    logging.warning("IAMCCS WanAnimate Extends nodes unavailable: %s", exc)
    IAMCCS_WanAnimateExtends = None
    IAMCCS_WanAnimateExtendPlan = None

# Nodi principali
NODE_CLASS_MAPPINGS = {
    "IAMCCS_WanLoRAStack": IAMCCS_WanLoRAStack,
    "IAMCCS_ModelWithLoRA": IAMCCS_ModelWithLoRA,
    "IAMCCS_WanLoRAStackModelIO": IAMCCS_WanLoRAStackModelIO,
    "IAMCCS_WanLoRASchedule": IAMCCS_WanLoRASchedule,
    "IAMCCS_WanLoRAHookSchedule": IAMCCS_WanLoRAHookSchedule,
    "IAMCCS_ApplyLoRAHooksToConditioning": IAMCCS_ApplyLoRAHooksToConditioning,
    "IAMCCS_ApplyScheduledWanLoRAFromConditioning": IAMCCS_ApplyScheduledWanLoRAFromConditioning,
    "IAMCCS_BuildScheduledWanModelBank": IAMCCS_BuildScheduledWanModelBank,
    "IAMCCS_SelectScheduledWanModelFromConditioning": IAMCCS_SelectScheduledWanModelFromConditioning,
    "IAMCCS_SelectScheduledWanModelPairFromConditioning": IAMCCS_SelectScheduledWanModelPairFromConditioning,
    "IAMCCS_WanLoRARuntimeBridge": IAMCCS_WanLoRARuntimeBridge,
    "IAMCCS_ModelWithLoRA_RuntimeBridge": IAMCCS_ModelWithLoRA_RuntimeBridge,
    # Backward-compatible key (kept as-is for existing workflows)
    "iamccs_ltx2_lora_stack": IAMCCS_LTX2_LoRAStack,
    # Preferred explicit names
    "IAMCCS_LTX2_LoRAStack": IAMCCS_LTX2_LoRAStack,
    "IAMCCS_LTX2_LoRAStackStaged": IAMCCS_LTX2_LoRAStackStaged,
    "IAMCCS_ModelWithLoRA_LTX2": IAMCCS_ModelWithLoRA_LTX2,
    "IAMCCS_ModelWithLoRA_LTX2_Staged": IAMCCS_ModelWithLoRA_LTX2_Staged,
    "IAMCCS_LTX2_LoRAStackModelIO": IAMCCS_LTX2_LoRAStackModelIO,
    "IAMCCS_LTX2_LoRAStackSegmented6": IAMCCS_LTX2_LoRAStackSegmented6,
    "IAMCCS_LTX2_ModelWithLoRA_Segmented6": IAMCCS_LTX2_ModelWithLoRA_Segmented6,

    "IAMCCS_LTX2_FrameRateSync": IAMCCS_LTX2_FrameRateSync,
    "IAMCCS_LTX2_Validator": IAMCCS_LTX2_Validator,
    "IAMCCS_LTX2_TimeFrameCount": IAMCCS_LTX2_TimeFrameCount,
    "IAMCCS_LTX2_EnsureFrames8nPlus1": IAMCCS_LTX2_EnsureFrames8nPlus1,
    "IAMCCS_LTX2_EnsureMinFrames": IAMCCS_LTX2_EnsureMinFrames,
    "IAMCCS_LTX2_ControlPreprocess": IAMCCS_LTX2_ControlPreprocess,
    "IAMCCS_LTX2_ImageBatchPadReflect": IAMCCS_LTX2_ImageBatchPadReflect,
    "IAMCCS_LTX2_ImageBatchCropByPad": IAMCCS_LTX2_ImageBatchCropByPad,
    "IAMCCS_SegmentPlanner": IAMCCS_SegmentPlanner,
    "IAMCCS_SegmentPlannerSettings": IAMCCS_SegmentPlannerSettings,
    "IAMCCS_AudioSegmentAutoPlanner": IAMCCS_AudioSegmentAutoPlanner,
    "IAMCCS_SegmentPlannerLinked": IAMCCS_SegmentPlannerLinked,
    "IAMCCS_SegmentPlanFromPlanner": IAMCCS_SegmentPlanFromPlanner,
    "IAMCCS_SourceRangeFromSegmentPlan": IAMCCS_SourceRangeFromSegmentPlan,
    "IAMCCS_TwoSegmentPlanner": IAMCCS_TwoSegmentPlanner,
    "IAMCCS_ThreeSegmentPlanner": IAMCCS_ThreeSegmentPlanner,
    "IAMCCS_SegmentSwitch": IAMCCS_SegmentSwitch,
    "IAMCCS_LTX2_ExtensionModule": IAMCCS_LTX2_ExtensionModule,
    "IAMCCS_LTX2_ExtensionModule_Disk": IAMCCS_LTX2_ExtensionModule_Disk,
    "IAMCCS_LTX2_JointRefreshLatent": IAMCCS_LTX2_JointRefreshLatent,
    "IAMCCS_LTX2_JointRefreshLatent_Disk": IAMCCS_LTX2_JointRefreshLatent_Disk,
    "IAMCCS_LoadImagesFromDirLite": IAMCCS_LoadImagesFromDirLite,
    "IAMCCS_ImageBatchRangeLite": IAMCCS_ImageBatchRangeLite,
    "IAMCCS_SourceFramesToDisk": IAMCCS_SourceFramesToDisk,
    "IAMCCS_StartDirToVideoLatent": IAMCCS_StartDirToVideoLatent,
    "IAMCCS_StartImagesToVideoLatent": IAMCCS_StartImagesToVideoLatent,
    "IAMCCS_VideoCombineFromDir": IAMCCS_VideoCombineFromDir,
    "IAMCCS_LTX2_ExtensionModule_simple": IAMCCS_LTX2_ExtensionModule_simple,
    "IAMCCS_LTX2_GetImageFromBatch": IAMCCS_LTX2_GetImageFromBatch,
    "IAMCCS_LTX2_ReferenceImageSwitch": IAMCCS_LTX2_ReferenceImageSwitch,
    "IAMCCS_LTX2_ReferenceStartFramesInjector": IAMCCS_LTX2_ReferenceStartFramesInjector,
    "IAMCCS_LTX2_FrameCountValidator": IAMCCS_LTX2_FrameCountValidator,
    "IAMCCS_LTX2_FirstLastFramesController": IAMCCS_LTX2_FirstLastFramesController,
    "IAMCCS_LTX2_ContextLatent": IAMCCS_LTX2_ContextLatent,
    "IAMCCS_LTX2_MiddleFrames": IAMCCS_LTX2_MiddleFrames,
    "IAMCCS_LTX2_FirstLastLatentControl": IAMCCS_LTX2_FirstLastLatentControl,
    "IAMCCS_LTX2_FirstLastLatentControl_Pro": IAMCCS_LTX2_FirstLastLatentControl_Pro,
    "IAMCCS_CineReferenceBoard": IAMCCS_CineReferenceBoard,
    "IAMCCS_CineLTXSequencer": IAMCCS_CineLTXSequencer,
    "IAMCCS_CineAllInOneFLFEngine": IAMCCS_CineAllInOneFLFEngine,
    "IAMCCS_CinePromptRelayTimeline": IAMCCS_CinePromptRelayTimeline,
    "IAMCCS_CineShotboardTimelinePro": IAMCCS_CineShotboardTimelinePro,
    "IAMCCS_CineShotboardPlannerPro": IAMCCS_CineShotboardPlannerPro,
    "IAMCCS_CineShotboardPlannerProV2": IAMCCS_CineShotboardPlannerProV2,
    "IAMCCS_CineShotboardPlannerV3": IAMCCS_CineShotboardPlannerV3,
    "IAMCCS_CineShotboardPlannerV4": IAMCCS_CineShotboardPlannerV4,
    "IAMCCS_CineShotboardPlannerV5V2V": IAMCCS_CineShotboardPlannerV5V2V,
    "IAMCCS_CineShotboardV5V2VRouter": IAMCCS_CineShotboardV5V2VRouter,
    "IAMCCS_CineShotboardV4Backend": IAMCCS_CineShotboardV4Backend,
    "IAMCCS_CineShotboardV4Guide": IAMCCS_CineShotboardV4Guide,
    "IAMCCS_CineShotboardV4CropGuides": IAMCCS_CineShotboardV4CropGuides,
    "IAMCCS_LTXVideoDurationCrop": IAMCCS_LTXVideoDurationCrop,
    "IAMCCS_CineShotboardLite": IAMCCS_CineShotboardLite,
    "IAMCCS_CineShotboardPlannerProLegacy": IAMCCS_CineShotboardPlannerProLegacy,
    "IAMCCS_CineResolutionParityTranslator": IAMCCS_CineResolutionParityTranslator,
    "IAMCCS_CineStage2BypassSwitch": IAMCCS_CineStage2BypassSwitch,
    "IAMCCS_CineStage2PreviewToggle": IAMCCS_CineStage2PreviewToggle,
    "IAMCCS_CineInfo": IAMCCS_CineInfo,
    "IAMCCS_CineInfoV2": IAMCCS_CineInfoV2,
    "IAMCCS_CineFLFProductor": IAMCCS_CineFLFProductor,
    "IAMCCS_CineFLFProductorDyno": IAMCCS_CineFLFProductorDyno,
    "IAMCCS_CineFilmmaker": IAMCCS_CineFilmmaker,
    "IAMCCS_CineFilmmakerBackend": IAMCCS_CineFilmmakerBackend,
    "IAMCCS_CineShotboardBackendPro": IAMCCS_CineShotboardBackendPro,
    "IAMCCS_CineFilmmakerGuide": IAMCCS_CineFilmmakerGuide,
    "IAMCCS_CineFilmmakerGuide1to1": IAMCCS_CineFilmmakerGuide1to1,
    "IAMCCS_CineFilmmakerCropGuides1to1": IAMCCS_CineFilmmakerCropGuides1to1,
    "IAMCCS_CineSwitch": IAMCCS_CineSwitch,
    "IAMCCS_CinePromptRelayLatentShapeSync": IAMCCS_CinePromptRelayLatentShapeSync,
    "IAMCCS_CineFLFLengthCompensator": IAMCCS_CineFLFLengthCompensator,
    "IAMCCS_CineBoardDurationLock": IAMCCS_CineBoardDurationLock,
    "IAMCCS_CineLatentDurationCrop": IAMCCS_CineLatentDurationCrop,
    "IAMCCS_CineShotboardTailTrimPolicy": IAMCCS_CineShotboardTailTrimPolicy,
    "IAMCCS_CineShotboardFinalFrameTrim": IAMCCS_CineShotboardFinalFrameTrim,
    "IAMCCS_shotboarder_aud+vid_exporter_PRO": IAMCCS_ShotboarderAudVidExporterPRO,
    "IAMCCS_CineTemporalCutBarrier": IAMCCS_CineTemporalCutBarrier,
    "IAMCCS_CinePromptRelaySafeEncode": IAMCCS_CinePromptRelaySafeEncode,
    "IAMCCS_CineRelayOrBypass": IAMCCS_CineRelayOrBypass,
    "IAMCCS_CinePromptArchitect": IAMCCS_CinePromptArchitect,
    "IAMCCS_BoardMaker": IAMCCS_BoardMaker,
    "IAMCCS_CineMusicVideoPlanner": IAMCCS_CineMusicVideoPlanner,
    "IAMCCS_CineShotPlanner": IAMCCS_CineShotPlanner,
    "IAMCCS_CineRefLatentControl": IAMCCS_CineRefLatentControl,
    "IAMCCS_CineAudioPromptDirector": IAMCCS_CineAudioPromptDirector,
    "IAMCCS_CinePromptRelayAdapter": IAMCCS_CinePromptRelayAdapter,
    "IAMCCS_CinePromptComposer": IAMCCS_CinePromptComposer,
    "IAMCCS_CineShotLineBuilder": IAMCCS_CineShotLineBuilder,
    "IAMCCS_CineV2VTimelineLineBuilder": IAMCCS_CineV2VTimelineLineBuilder,
    "IAMCCS_CineLineStacker": IAMCCS_CineLineStacker,
    "IAMCCS_CineMultiGenDirector": IAMCCS_CineMultiGenDirector,
    "IAMCCS_CineShotAudioDirector": IAMCCS_CineShotAudioDirector,
    "IAMCCS_CineV2VTimelineDirector": IAMCCS_CineV2VTimelineDirector,
    "IAMCCS_CineV2VAssetSelector": IAMCCS_CineV2VAssetSelector,
    "IAMCCS_CineWorkflowInspector": IAMCCS_CineWorkflowInspector,
    "IAMCCS_WDC_MultiImageLoader": IAMCCS_WDC_MultiImageLoader,
    "IAMCCS_WDC_LTXKeyframer": IAMCCS_WDC_LTXKeyframer,
    "IAMCCS_WDC_LTXSequencer": IAMCCS_WDC_LTXSequencer,
    "IAMCCS_CineLTXSequencerExact": IAMCCS_CineLTXSequencerExact,
    "IAMCCS_CineFLFEngineSimple": IAMCCS_CineFLFEngineSimple,
    "IAMCCS_CineFLFEngineSimpleDyno": IAMCCS_CineFLFEngineSimpleDyno,
    "IAMCCS_WDC_LTXSequencerFixed5": IAMCCS_WDC_LTXSequencerFixed5,
    "IAMCCS_LTX2_InitLatentSampler": IAMCCS_LTX2_InitLatentSampler,
    "IAMCCS_LTX2_LoopingSampler": IAMCCS_LTX2_LoopingSampler,
    "IAMCCS_LTX2_OneShotLowRAMLooper": IAMCCS_LTX2_OneShotLowRAMLooper,
    "IAMCCS_LTX2_ExtendSampler": IAMCCS_LTX2_ExtendSampler,
    "IAMCCS_LTX2_ConditionNextLatentWithPrevOverlap": IAMCCS_LTX2_ConditionNextLatentWithPrevOverlap,
    "IAMCCS_WanImageMotion": IAMCCS_WanImageMotion,
    # Backward-compat alias: workflow JSONs saved with the _AdaIN name still load.
    "IAMCCS_WanImageMotion_AdaIN": IAMCCS_WanImageMotion,
    "WanImageMotionPro": WanImageMotionProPlus,
    "IAMCCS_WanImageMotionPro_AdaIN": WanImageMotionProPlus,
    # Keep the historic key on the current implementation so existing workflows
    # load the new Plus node with continuity profiles and presets.
    "IAMCCS_WanImageMotionPro": WanImageMotionProPlus,
    # Explicit legacy entrypoint for older raw-only behavior.
    "WanImageMotionProLegacy": WanImageMotionProLegacy,
    "IAMCCS_WanImageMotionProLegacy": WanImageMotionProLegacy,
    "WanImageMotionProPlus": WanImageMotionProPlus,
    "IAMCCS_WanImageMotionProPlus": WanImageMotionProPlus,
    "IAMCCS_WanImageMotionProPlus_Simple": IAMCCS_WanImageMotionProPlus_Simple,
    "IAMCCS_WanImageMotionInductive": IAMCCS_WanImageMotionInductive,
    "IAMCCS_WanSVIToFLFBridgeProPlus": IAMCCS_WanSVIToFLFBridgeProPlus,
    "IAMCCS_WanSVIToFLFBridgeProPlus_Simple": IAMCCS_WanSVIToFLFBridgeProPlus_Simple,
    "WanMotionProTrimmer": WanMotionProTrimmer,
    "IAMCCS_WanPrevTailPrep": IAMCCS_WanPrevTailPrep,
    "IAMCCS_WanLongPlanner": IAMCCS_WanLongPlanner,
    "IAMCCS_WanSviFlfTimeline": IAMCCS_WanSviFlfTimeline,
    "IAMCCS_WanSviFlfTimelinePick": IAMCCS_WanSviFlfTimelinePick,
    "IAMCCS_WanContinuityGuide": IAMCCS_WanContinuityGuide,
    "IAMCCS_WanPromptPhasePlanner": IAMCCS_WanPromptPhasePlanner,
    "IAMCCS_WanPromptLoopInfo": IAMCCS_WanPromptLoopInfo,
    "IAMCCS_WanIndexedPromptEncode": IAMCCS_WanIndexedPromptEncode,
    "IAMCCS_WanImageBatchFrameSelect": IAMCCS_WanImageBatchFrameSelect,
    
    "IAMCCS_SetAutoLink": IAMCCS_SetAutoLink,
    "IAMCCS_GetAutoLink": IAMCCS_GetAutoLink,
    "IAMCCS_AutoLinkConverter": IAMCCS_AutoLinkConverter,
    "IAMCCS_AutoLinkArguments": IAMCCS_AutoLinkArguments,

    "IAMCCS_GGUF_accelerator": IAMCCS_GGUF_accelerator,

    "IAMCCS_SamplerAdvancedVersion1": IAMCCS_SamplerAdvancedVersion1,

    "IAMCCS_bus_group": IAMCCS_bus_group,

    "IAMCCS_MultiSwitch": IAMCCS_MultiSwitch,
    "IAMCCS_LazyAnySwitch": IAMCCS_LazyAnySwitch,
    "IAMCCS_Navigator": IAMCCS_Navigator,

    "IAMCCS_HwSupporter": IAMCCS_HwSupporter,
    "IAMCCS_HwSupporterAny": IAMCCS_HwSupporterAny,
    "IAMCCS_HardMemoryPurge": IAMCCS_HardMemoryPurge,
    "IAMCCS_VRAMCleanup": IAMCCS_VRAMCleanup,
    "IAMCCS_VRAMFlushLatent": IAMCCS_VRAMFlushLatent,
    "IAMCCS_VAEDecodeTiledSafe": IAMCCS_VAEDecodeTiledSafe,
    "IAMCCS_VAEDecodeToDisk": IAMCCS_VAEDecodeToDisk,
    "IAMCCS_HWProbeRecommendations": IAMCCS_HWProbeRecommendations,
    "IAMCCS_DetailAtelier": IAMCCS_DetailAtelier,
    "IAMCCS_DetailAtelierAdvanced": IAMCCS_DetailAtelierAdvanced,
    "IAMCCS_DetailAtelierSampler": IAMCCS_DetailAtelierSampler,

    "IAMCCS_MoveAhead": IAMCCS_MoveAhead,
    "IAMCCS_MoveAheadEnforcer": IAMCCS_MoveAheadEnforcer,
    "IAMCCS_MotionScale": IAMCCS_MotionScale,
    "IAMCCS_MotionScaleAdvanced": IAMCCS_MotionScaleAdvanced,

    "IAMCCS_MotionBridgeSave": IAMCCS_MotionBridgeSave,
    "IAMCCS_MotionBridgeLoad": IAMCCS_MotionBridgeLoad,
    "IAMCCS_LatentTailSlice":  IAMCCS_LatentTailSlice,
    "IAMCCS_AudioExtensionMath": IAMCCS_AudioExtensionMath,
    "IAMCCS_AudioExtender": IAMCCS_AudioExtender,
    "IAMCCS_AudioTimelineAssembler": IAMCCS_AudioTimelineAssembler,
    "IAMCCS_AudioTimelineGate": IAMCCS_AudioTimelineGate,
    "IAMCCS_BoardMaker_DialogueFoley": IAMCCS_BoardMaker_DialogueFoley,
    "IAMCCS_CineInfo3": IAMCCS_CineInfo3,
    "IAMCCS_CineDialogueLineRouter": IAMCCS_CineDialogueLineRouter,
    "IAMCCS_CineTimelineAudioMixer": IAMCCS_CineTimelineAudioMixer,
    "IAMCCS_AudioBoardArranger": IAMCCS_AudioBoardArranger,
    "IAMCCS_BusOut": IAMCCS_BusOut,
    "IAMCCS_AudioBoardDirectShotboardAdapter": IAMCCS_AudioBoardDirectShotboardAdapter,
    "IAMCCS_AudioBoardMixer": IAMCCS_AudioBoardMixer,
    "IAMCCS_ControlAudEfx": IAMCCS_ControlAudEfx,
    "IAMCCS_ControlAudEfxPanel": IAMCCS_ControlAudEfxPanel,
    "IAMCCS_DialogueTagEditor": IAMCCS_DialogueTagEditor,
    "IAMCCS_DialogueAudioBoardBridge": IAMCCS_DialogueAudioBoardBridge,
    "IAMCCS_CineAudioInfo": IAMCCS_CineAudioInfo,
    "IAMCCS_AudioPhaseLazyGate": IAMCCS_AudioPhaseLazyGate,
    "IAMCCS_StoryboardFrameDesigner": IAMCCS_StoryboardFrameDesigner,
    "IAMCCS_StoryboardFrameDesignerV2": IAMCCS_StoryboardFrameDesignerV2,
    "IAMCCS_IdeoInfo": IAMCCS_IdeoInfo,
    "IAMCCS_IdeoInpaintPrep": IAMCCS_IdeoInpaintPrep,
    "IAMCCS_IdeoMaskedPixels": IAMCCS_IdeoMaskedPixels,
    "IAMCCS_IdeogramJSONPreviewPass": IAMCCS_IdeogramJSONPreviewPass,
    "IAMCCS_IdeoNoiseDebug": IAMCCS_IdeoNoiseDebug,
    "IAMCCS_IdeoRandomNoiseNoCache": IAMCCS_IdeoRandomNoiseNoCache,
    "IAMCCS_IdeoImageHashDebug": IAMCCS_IdeoImageHashDebug,
    "IAMCCS_IdeoTranslate": IAMCCS_IdeoTranslate,
    "IAMCCS_IdeogramStoryboardSheet": IAMCCS_IdeogramStoryboardSheet,
    "IAMCCS_IdeogramSheetBuilder": IAMCCS_IdeogramSheetBuilder,
    "IAMCCS_StoryboardCaptionSheet": IAMCCS_StoryboardCaptionSheet,
    "IAMCCS_StoryboardAutoCropGrid": IAMCCS_StoryboardAutoCropGrid,
    "IAMCCS_StoryboardAutoCropGridPRO": IAMCCS_StoryboardAutoCropGridPRO,
    "IAMCCS_TargetCrop": IAMCCS_TargetCrop,
    "IAMCCS_GemmaAssistLazyGate": IAMCCS_GemmaAssistLazyGate,
    "IAMCCS_GemmaAssistOutput": IAMCCS_GemmaAssistOutput,
    "IAMCCS_StoryboardPromptContactSheet": IAMCCS_StoryboardPromptContactSheet,
    "IAMCCS_GoyAICanvasPaint": IAMCCS_GoyAICanvasPaint,
    "IAMCCS_FlashVSRPanelBatchPrep": IAMCCS_FlashVSRPanelBatchPrep,
    "IAMCCS_FlashVSRPanelBatchRestore": IAMCCS_FlashVSRPanelBatchRestore,
    "IAMCCS_CineVideoToWooshInputs": IAMCCS_CineVideoToWooshInputs,
    "IAMCCS_CineSpeech1PromptCompiler": IAMCCS_CineSpeech1PromptCompiler,
    "IAMCCS_CineAudioTranscriptPromptCompiler": IAMCCS_CineAudioTranscriptPromptCompiler,
    "IAMCCS_CineSpeechLength": IAMCCS_CineSpeechLength,
    "IAMCCS_CineDialogueDurationPlanner": IAMCCS_CineDialogueDurationPlanner,
    "IAMCCS_CineAudioDurationProbe": IAMCCS_CineAudioDurationProbe,
    "IAMCCS_CineDialogueTimingReconciler": IAMCCS_CineDialogueTimingReconciler,
    "IAMCCS_CineWooshFoleyChunkPlanner": IAMCCS_CineWooshFoleyChunkPlanner,
    "IAMCCS_CineFinalAudioMixer": IAMCCS_CineFinalAudioMixer,
    "IAMCCS_CineEmotionButtons": IAMCCS_CineEmotionButtons,
    "IAMCCS_CineDialoguePromptKit": IAMCCS_CineDialoguePromptKit,
    "IAMCCS_LTX2_LastFrameBridgeSave": IAMCCS_LTX2_LastFrameBridgeSave,
    "IAMCCS_LTX2_BlendLatentBridge": IAMCCS_LTX2_BlendLatentBridge,
    "IAMCCS_LTX2_LastFrameBridgeLoad": IAMCCS_LTX2_LastFrameBridgeLoad,
    "IAMCCS_LTX2_LoadLatentBridge": IAMCCS_LTX2_LoadLatentBridge,
    "IAMCCS_LTX2_LongVideoWrapperPrep": IAMCCS_LTX2_LongVideoWrapperPrep,
    "IAMCCS_LTX2_LongVideoWrapperPrepDisk": IAMCCS_LTX2_LongVideoWrapperPrepDisk,
    "IAMCCS_LTX2_SaveLatentBridge": IAMCCS_LTX2_SaveLatentBridge,
    "IAMCCS_LTX2_SegmentQueueLoop": IAMCCS_LTX2_SegmentQueueLoop,
    "IAMCCS_ImageResizeBatchSafe": IAMCCS_ImageResizeBatchSafe,
    "IAMCCS_LoadResizeSegmentFromDir": IAMCCS_LoadResizeSegmentFromDir,
    "IAMCCS_IntValueMonitor": IAMCCS_IntValueMonitor,
    "IAMCCS_QwenMultiGen": IAMCCS_QwenMultiGen,
    "IAMCCS_FluxKleinMultiGen": IAMCCS_FluxKleinMultiGen,
    "IAMCCS_FluxKleinRefine": IAMCCS_FluxKleinRefine,
    "IAMCCS_ImageBatch6": IAMCCS_ImageBatch6,
    "IAMCCS_MultilinePromptSplitter8": IAMCCS_MultilinePromptSplitter8,
    "IAMCCS_SupernodeBase": IAMCCS_SupernodeBase,
    "IAMCCS_SupernodeModule": IAMCCS_SupernodeModule,
    "IAMCCS_ProjectTimelinePlanner": IAMCCS_ProjectTimelinePlanner,
    "IAMCCS_Ltx2HelperModules_ProjectTimelinePlanner": IAMCCS_Ltx2HelperModules_ProjectTimelinePlanner,
    "IAMCCS_Ltx2HelperModules_Planner": IAMCCS_Ltx2HelperModules_Planner,
    "IAMCCS_Ltx2HelperModules_AudioTimeline": IAMCCS_Ltx2HelperModules_AudioTimeline,
    "IAMCCS_Ltx2HelperModules_KeyframeTimeline": IAMCCS_Ltx2HelperModules_KeyframeTimeline,
    "IAMCCS_Ltx2HelperModules_RefreshPolicy": IAMCCS_Ltx2HelperModules_RefreshPolicy,
    "IAMCCS_Ltx2HelperModules_ReanchorLatent": IAMCCS_Ltx2HelperModules_ReanchorLatent,
    "IAMCCS_Ltx2HelperModules_DiskExtension": IAMCCS_Ltx2HelperModules_DiskExtension,
    "IAMCCS_Ltx2HelperModules_RuntimeBridge": IAMCCS_Ltx2HelperModules_RuntimeBridge,
    "IAMCCS_Ltx2HelperModules_Continuity": IAMCCS_Ltx2HelperModules_Continuity,
    "IAMCCS_Ltx2HelperModules_Finalize": IAMCCS_Ltx2HelperModules_Finalize,
    "IAMCCS_AUIMG2VID_ProjectTimelinePlanner": IAMCCS_AUIMG2VID_ProjectTimelinePlanner,
    "IAMCCS_AUIMG2VID_Planner": IAMCCS_AUIMG2VID_Planner,
    "IAMCCS_AUIMG2VID_AudioTimeline": IAMCCS_AUIMG2VID_AudioTimeline,
    "IAMCCS_AUIMG2VID_KeyframeTimeline": IAMCCS_AUIMG2VID_KeyframeTimeline,
    "IAMCCS_AUIMG2VID_RefreshPolicy": IAMCCS_AUIMG2VID_RefreshPolicy,
    "IAMCCS_AUIMG2VID_ReanchorLatent": IAMCCS_AUIMG2VID_ReanchorLatent,
    "IAMCCS_AUIMG2VID_DiskExtension": IAMCCS_AUIMG2VID_DiskExtension,
    "IAMCCS_AUIMG2VID_RuntimeBridge": IAMCCS_AUIMG2VID_RuntimeBridge,
    "IAMCCS_AUIMG2VID_Continuity": IAMCCS_AUIMG2VID_Continuity,
    "IAMCCS_AUIMG2VID_Finalize": IAMCCS_AUIMG2VID_Finalize,
    "IAMCCS-SuperNodes AU+IMG2VID Exec Planner": IAMCCS_SuperNodes_AUIMG2VIDExecutablePlanner,
    "IAMCCS-SuperNodes AU+IMG2VID Exec Render": IAMCCS_SuperNodes_AUIMG2VIDExecutableRender,
    "IAMCCS-SuperNodes AU+IMG2VID Exec VAE": IAMCCS_SuperNodes_AUIMG2VIDExecutableVAE,
    "IAMCCS-SuperNodes Second Stage": IAMCCS_SuperNodes_SecondStage,
    **({"IAMCCS-SuperNodes AU+IMG2VID Exec Finalize": IAMCCS_SuperNodes_AUIMG2VIDExecutableFinalize} if IAMCCS_SuperNodes_AUIMG2VIDExecutableFinalize is not None else {}),

    **({
        "IAMCCS_ScailIdentitySeeder": IAMCCS_ScailIdentitySeeder,
        "IAMCCS_ScailIdentityTracker": IAMCCS_ScailIdentityTracker,
        "IAMCCS_ScailMultiReference": IAMCCS_ScailMultiReference,
    } if IAMCCS_ScailIdentityTracker is not None else {}),

    **({
        "IAMCCS_ScailExtends": IAMCCS_ScailExtends,
        "IAMCCS_ScailExtendPlan": IAMCCS_ScailExtendPlan,
    } if IAMCCS_ScailExtends is not None else {}),

    **({
        "IAMCCS_WanAnimateExtends": IAMCCS_WanAnimateExtends,
        "IAMCCS_WanAnimateExtendPlan": IAMCCS_WanAnimateExtendPlan,
    } if IAMCCS_WanAnimateExtends is not None else {}),

    # QwenVL First/Last Frame (registered only if QwenVL is installed)
    **({"IAMCCS_QWEN_VL_FLF": IAMCCS_QWEN_VL_FLF,
        "IAMCCS_QWEN_VL_FLF_Advanced": IAMCCS_QWEN_VL_FLF_Advanced,
    } if IAMCCS_QWEN_VL_FLF is not None else {}),

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_ScailExtends": "IAMCCS SCAIL Extends",
    "IAMCCS_ScailExtendPlan": "IAMCCS SCAIL Extend Plan",
    "IAMCCS_WanAnimateExtends": "IAMCCS WanAnimate Extends",
    "IAMCCS_WanAnimateExtendPlan": "IAMCCS WanAnimate Extend Plan",
    "IAMCCS_ScailIdentitySeeder": "IAMCCS SCAIL Identity Seeder",
    "IAMCCS_ScailIdentityTracker": "IAMCCS SCAIL Identity Tracker",
    "IAMCCS_ScailMultiReference": "IAMCCS SCAIL Multi-Reference (experimental)",
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
    "IAMCCS_WanLoRAStackModelIO": "LoRA Stack (Model In?Out) WAN",
    "IAMCCS_WanLoRASchedule": "LoRA Schedule (WAN, ranged)",
    "IAMCCS_WanLoRAHookSchedule": "LoRA Schedule (WAN, hooks)",
    "IAMCCS_ApplyLoRAHooksToConditioning": "Apply LoRA Hooks to Conditioning",
    "IAMCCS_ApplyScheduledWanLoRAFromConditioning": "Apply Scheduled WAN LoRA From Conditioning",
    "IAMCCS_BuildScheduledWanModelBank": "Build Scheduled WAN Model Bank",
    "IAMCCS_SelectScheduledWanModelFromConditioning": "Select Scheduled WAN Model From Conditioning",
    "IAMCCS_SelectScheduledWanModelPairFromConditioning": "Select Scheduled WAN Model Pair From Conditioning",
    "IAMCCS_WanLoRARuntimeBridge": "LoRA Runtime Bridge (WAN, loop-safe)",
    "IAMCCS_ModelWithLoRA_RuntimeBridge": "Apply LoRA to MODEL (Runtime Bridge)",
    "iamccs_ltx2_lora_stack": "iamccs_ltx2_lora_stack (3 slots)",
    "IAMCCS_LTX2_LoRAStack": "LoRA Stack (LTX-2, 3 slots)",
    "IAMCCS_LTX2_LoRAStackStaged": "LoRA Stack (LTX-2, staged: stage1+stage2) (BETA)",
    "IAMCCS_ModelWithLoRA_LTX2": "Apply LoRA to MODEL (LTX-2, quiet logs)",
    "IAMCCS_ModelWithLoRA_LTX2_Staged": "Apply LoRA to MODEL (LTX-2, staged) (BETA)",
    "IAMCCS_LTX2_LoRAStackModelIO": "LoRA Stack (Model In?Out) LTX-2",
    "IAMCCS_LTX2_LoRAStackSegmented6": "LoRA Stack (LTX-2, segmented: 3 seg × 2 stages)",
    "IAMCCS_LTX2_ModelWithLoRA_Segmented6": "Apply LoRA to MODEL (LTX-2, segmented: 3 seg × 2 stages)",

    "IAMCCS_LTX2_FrameRateSync": "LTX-2 FrameRate Sync (int+float)",
    "IAMCCS_LTX2_Validator": "LTX-2 Validator",
    "IAMCCS_LTX2_TimeFrameCount": "LTX-2 TimeFrameCount",
    "IAMCCS_LTX2_EnsureFrames8nPlus1": "LTX-2 Ensure Frames (8n + 1)",
    "IAMCCS_LTX2_EnsureMinFrames": "LTX-2 Ensure Minimum Frames",
    "IAMCCS_LTX2_ControlPreprocess": "LTX-2 Control Preprocess (aux)",
    "IAMCCS_LTX2_ImageBatchPadReflect": "LTX-2 Pad Reflect (IMAGE batch)",
    "IAMCCS_LTX2_ImageBatchCropByPad": "LTX-2 Crop By Pad (IMAGE batch)",
    "IAMCCS_SegmentPlanner": "Segment Planner (song -> LTX frames)",
    "IAMCCS_SegmentPlannerSettings": "Segment Planner Settings (shared)",
    "IAMCCS_AudioSegmentAutoPlanner": "Audio Segment Auto Planner (audio -> segmenti)",
    "IAMCCS_SegmentPlannerLinked": "Segment Planner Linked (shared inputs)",
    "IAMCCS_SegmentPlanFromPlanner": "Segment Plan From Planner (per index)",
    "IAMCCS_SourceRangeFromSegmentPlan": "Source Range From Segment Plan",
    "IAMCCS_TwoSegmentPlanner": "Two Segment Planner (stable 2SEG)",
    "IAMCCS_ThreeSegmentPlanner": "Three Segment Planner (stable 3SEG)",
    "IAMCCS_SegmentSwitch": "Segment Switch (by segment_index)",
    "IAMCCS_LTX2_ExtensionModule": "LTX-2 Extension Module ??",
    "IAMCCS_LTX2_ExtensionModule_Disk": "LTX-2 Extension Module (Disk / Low RAM) ??",
    "IAMCCS_LTX2_JointRefreshLatent": "LTX-2 Joint Refresh Latent",
    "IAMCCS_LTX2_JointRefreshLatent_Disk": "LTX-2 Joint Refresh Latent (Disk)",
    "IAMCCS_LoadImagesFromDirLite": "Load Images From Dir (Lite) ??",
    "IAMCCS_ImageBatchRangeLite": "Image Batch Range (VRAM) ???",
    "IAMCCS_SourceFramesToDisk": "Source Frames To Disk ????",
    "IAMCCS_StartDirToVideoLatent": "Start Dir To Video Latent ??",
    "IAMCCS_StartImagesToVideoLatent": "Start Images To Video Latent ??",
    "IAMCCS_VideoCombineFromDir": "Video Combine From Dir ???",
    "IAMCCS_LTX2_ExtensionModule_simple": "LTX-2 Extension Module (simple) ??",
    "IAMCCS_LTX2_GetImageFromBatch": "LTX-2 Get Images From Batch ???",
    "IAMCCS_LTX2_ReferenceImageSwitch": "LTX-2 Reference Image Switch ??",
    "IAMCCS_LTX2_ReferenceStartFramesInjector": "LTX-2 Inject Reference Into Start Frames ??",
    "IAMCCS_LTX2_FrameCountValidator": "LTX-2 Frame Count Validator ? (8n+1)",
    "IAMCCS_LTX2_FirstLastFramesController": "LTX-2 First/Last Frames Controller ??",
    "IAMCCS_LTX2_ContextLatent": "LTX-2 Context ? Latent (continue) ??",
    "IAMCCS_LTX2_MiddleFrames": "LTX-2 Middle Frames (accumulator) ??",
    "IAMCCS_LTX2_FirstLastLatentControl": "LTX-2 First/Last ? Latent (noise_mask) ??",
    "IAMCCS_LTX2_FirstLastLatentControl_Pro": "LTX-2 First/Last ? Latent (Pro, slot caps) ??",
    "IAMCCS_CineReferenceBoard": "IAMCCS Cine Reference Board",
    "IAMCCS_CineLTXSequencer": "IAMCCS Cine FLF Timeline Sequencer",
    "IAMCCS_CineAllInOneFLFEngine": "IAMCCS Cine AllInOne FLF Engine",
    "IAMCCS_CinePromptRelayTimeline": "IAMCCS Cine PromptRelay Timeline",
    "IAMCCS_CineShotboardTimelinePro": "IAMCCS Cine Shotboard Timeline Pro",
    "IAMCCS_CineShotboardPlannerPro": "IAMCCS Cine Shotboard Planner Pro",
    "IAMCCS_CineShotboardPlannerProV2": "IAMCCS Cine Shotboard Planner Pro V2",
    "IAMCCS_CineShotboardPlannerV3": "IAMCCS Cine Shotboard Planner V3",
    "IAMCCS_CineShotboardPlannerV4": "IAMCCS Cine Shotboard Planner V4",
    "IAMCCS_CineShotboardPlannerV5V2V": "IAMCCS Cine Shotboard Planner V5 V2V",
    "IAMCCS_CineShotboardV5V2VRouter": "IAMCCS Cine Shotboard V5 V2V Router",
    "IAMCCS_CineShotboardV4Backend": "IAMCCS Cine Shotboard V4 Backend",
    "IAMCCS_CineShotboardV4Guide": "IAMCCS Cine Shotboard V4 Guide",
    "IAMCCS_CineShotboardV4CropGuides": "IAMCCS Cine Shotboard V4 Crop Guides",
    "IAMCCS_LTXVideoDurationCrop": "IAMCCS LTX Video Duration Crop",
    "IAMCCS_CineShotboardLite": "IAMCCS Cine Shotboard Lite",
    "IAMCCS_CineShotboardPlannerProLegacy": "IAMCCS Cine Shotboard Planner Pro Legacy Outputs",
    "IAMCCS_CineResolutionParityTranslator": "IAMCCS Cine Resolution Parity Translator",
    "IAMCCS_CineStage2BypassSwitch": "IAMCCS Cine Stage 2 Bypass Switch",
    "IAMCCS_CineStage2PreviewToggle": "IAMCCS Cine Stage 2 Preview Toggle",
    "IAMCCS_CineInfo": "IAMCCS CineInfo",
    "IAMCCS_CineInfoV2": "IAMCCS CineInfo V2",
    "IAMCCS_CineFLFProductor": "IAMCCS Cine FLF Productor",
    "IAMCCS_CineFLFProductorDyno": "IAMCCS Cine FLF Productor Dyno",
    "IAMCCS_CineFilmmaker": "IAMCCS Cine Filmmaker",
    "IAMCCS_CineFilmmakerBackend": "IAMCCS Cine Filmmaker Backend",
    "IAMCCS_CineShotboardBackendPro": "IAMCCS Cine Shotboard Backend Pro",
    "IAMCCS_CineFilmmakerGuide": "IAMCCS Cine Filmmaker Guide",
    "IAMCCS_CineFilmmakerGuide1to1": "IAMCCS Cine Filmmaker Guide 1:1",
    "IAMCCS_CineFilmmakerCropGuides1to1": "IAMCCS Cine Filmmaker Crop Guides 1:1",
    "IAMCCS_CineSwitch": "IAMCCS CineSwitch Lazy FLF/PromptRelay",
    "IAMCCS_CinePromptRelayLatentShapeSync": "IAMCCS Cine PromptRelay Latent Shape Sync",
    "IAMCCS_CineFLFLengthCompensator": "IAMCCS Cine FLF Length Compensator",
    "IAMCCS_CineBoardDurationLock": "IAMCCS Cine Board Duration Lock",
    "IAMCCS_CineLatentDurationCrop": "IAMCCS Cine Latent Duration Crop",
    "IAMCCS_CineShotboardTailTrimPolicy": "IAMCCS Cine Shotboard Tail Trim Policy",
    "IAMCCS_CineShotboardFinalFrameTrim": "IAMCCS Cine Shotboard Final Frame Trim",
    "IAMCCS_shotboarder_aud+vid_exporter_PRO": "IAMCCS Shotboarder Aud+Vid Exporter PRO",
    "IAMCCS_CineTemporalCutBarrier": "IAMCCS Cine Temporal Cut Barrier (Experimental)",
    "IAMCCS_CinePromptRelaySafeEncode": "IAMCCS Cine PromptRelay Safe Encode",
    "IAMCCS_CineRelayOrBypass": "IAMCCS Cine Relay Or Bypass",
    "IAMCCS_CinePromptArchitect": "IAMCCS CinePrompt Architect",
    "IAMCCS_BoardMaker": "IAMCCS_BoardMaker",
    "IAMCCS_CineMusicVideoPlanner": "IAMCCS Cine Videoclip Maker Planner",
    "IAMCCS_CineShotPlanner": "IAMCCS Cine Shot Planner",
    "IAMCCS_CineRefLatentControl": "IAMCCS Cine Reference Latent Control",
    "IAMCCS_CineAudioPromptDirector": "IAMCCS Cine Audio Prompt Director",
    "IAMCCS_CinePromptRelayAdapter": "IAMCCS Cine PromptRelay Adapter",
    "IAMCCS_CinePromptComposer": "IAMCCS Cine Prompt Composer",
    "IAMCCS_CineShotLineBuilder": "IAMCCS Cine Shot Line Builder",
    "IAMCCS_CineV2VTimelineLineBuilder": "IAMCCS Cine V2V Line Builder",
    "IAMCCS_CineLineStacker": "IAMCCS Cine Line Stacker",
    "IAMCCS_CineMultiGenDirector": "IAMCCS Cine Multi-Generation Director",
    "IAMCCS_CineShotAudioDirector": "IAMCCS Cine Shot Audio Director",
    "IAMCCS_CineV2VTimelineDirector": "IAMCCS Cine V2V Timeline Director",
    "IAMCCS_CineV2VAssetSelector": "IAMCCS Cine V2V Asset Selector",
    "IAMCCS_CineWorkflowInspector": "IAMCCS Cine Workflow Inspector",
    "IAMCCS_WDC_MultiImageLoader": "IAMCCS Cine Reference Board (legacy alias)",
    "IAMCCS_WDC_LTXKeyframer": "IAMCCS Cine LTX Keyframer (legacy alias)",
    "IAMCCS_WDC_LTXSequencer": "IAMCCS Cine LTX Sequencer (legacy alias)",
    "IAMCCS_CineLTXSequencerExact": "IAMCCS Cine LTX Sequencer Exact",
    "IAMCCS_CineFLFEngineSimple": "IAMCCS Cine FLF Engine Simple",
    "IAMCCS_CineFLFEngineSimpleDyno": "IAMCCS Cine FLF Engine Simple Dyno",
    "IAMCCS_WDC_LTXSequencerFixed5": "IAMCCS Cine LTX Sequencer Fixed 5 (legacy alias)",
    "IAMCCS_LTX2_InitLatentSampler": "LTX-2 Init Latent Sampler ??",
    "IAMCCS_LTX2_LoopingSampler": "LTX-2 Looping Sampler (temporal overlap) ??",
    "IAMCCS_LTX2_OneShotLowRAMLooper": "LTX-2 One-Shot Low-RAM Looper ??",
    "IAMCCS_LTX2_ExtendSampler": "LTX-2 Extend Sampler (temporal overlap) ??",
    "IAMCCS_LTX2_ConditionNextLatentWithPrevOverlap": "LTX-2 Condition Next Latent (prev overlap) ??",
    "IAMCCS_WanImageMotion": "WanImageMotion",
    "IAMCCS_WanImageMotion_AdaIN": "WanImageMotion",
    "WanImageMotionPro": "WanImageMotionPro Plus",
    "IAMCCS_WanImageMotionPro_AdaIN": "WanImageMotionPro Plus",
    "WanImageMotionProLegacy": "WanImageMotionPro Legacy",
    "IAMCCS_WanImageMotionProLegacy": "WanImageMotionPro Legacy",
    "WanImageMotionProPlus": "WanImageMotionPro Plus",
    "IAMCCS_WanImageMotionProPlus": "WanImageMotionPro Plus",
    "IAMCCS_WanImageMotionProPlus_Simple": "WanImageMotionPro Plus Simple",
    "IAMCCS_WanImageMotionInductive": "WanImageMotion Inductive",
    "IAMCCS_WanSVIToFLFBridgeProPlus": "Wan SVI?FLF Bridge Pro Plus",
    "IAMCCS_WanSVIToFLFBridgeProPlus_Simple": "Wan SVI?FLF Bridge Pro Plus",
    "WanMotionProTrimmer": "WanMotionProTrimmer (trim overshoot tail)",
    "IAMCCS_WanPrevTailPrep": "Wan Prev Tail Prep",
    "IAMCCS_WanLongPlanner": "Wan Long Planner",
    "IAMCCS_WanSviFlfTimeline": "Wan SVI/FLF Timeline",
    "IAMCCS_WanSviFlfTimelinePick": "Wan SVI/FLF Timeline Pick",
    "IAMCCS_WanContinuityGuide": "Wan Continuity Guide",
    "IAMCCS_WanPromptPhasePlanner": "Wan Prompt Phase Planner",
    "IAMCCS_WanPromptLoopInfo": "Wan Prompt Loop Info",
    "IAMCCS_WanIndexedPromptEncode": "Wan Indexed Prompt Encode",
    "IAMCCS_WanImageBatchFrameSelect": "Wan Image Batch Frame Select",
    
    "IAMCCS_SetAutoLink": "Set AutoLink",
    "IAMCCS_GetAutoLink": "Get AutoLink",
    "IAMCCS_AutoLinkConverter": "AutoLink Converter",
    "IAMCCS_AutoLinkArguments": "AutoLink Arguments",

    "IAMCCS_GGUF_accelerator": "GGUF Accelerator (patch_on_device)",

    "IAMCCS_IntValueMonitor": "INT Value Monitor",
    "IAMCCS_QwenMultiGen": "IAMCCS Qwen Multi-Gen",
    "IAMCCS_FluxKleinMultiGen": "Flux Klein Multi-Gen",
    "IAMCCS_FluxKleinRefine": "Flux Klein Refine (Local NO PAID)",
    "IAMCCS_ImageBatch6": "IAMCCS Image Batch 6",
    "IAMCCS_StoryboardCaptionSheet": "IAMCCS Storyboard Caption Sheet",
    "IAMCCS_StoryboardAutoCropGrid": "IAMCCS Storyboard Auto Crop Grid",
    "IAMCCS_StoryboardAutoCropGridPRO": "IAMCCS Storyboard Auto Crop Grid PRO",
    "IAMCCS_MultilinePromptSplitter8": "Multiline Prompt Splitter (8 outputs)",
    "IAMCCS_SupernodeBase": "Supernode Base (contract + linx)",
    "IAMCCS_SupernodeModule": "Supernode Module (cascade contract + linx)",
    "IAMCCS_ProjectTimelinePlanner": "IAMCCS Project Timeline Planner",
    "IAMCCS_Ltx2HelperModules_ProjectTimelinePlanner": "IAMCCS_Ltx2HelperModules Project Timeline Planner",
    "IAMCCS_Ltx2HelperModules_Planner": "IAMCCS_Ltx2HelperModules Planner",
    "IAMCCS_Ltx2HelperModules_AudioTimeline": "IAMCCS_Ltx2HelperModules Audio Timeline",
    "IAMCCS_Ltx2HelperModules_KeyframeTimeline": "IAMCCS_Ltx2HelperModules Keyframe Timeline",
    "IAMCCS_Ltx2HelperModules_RefreshPolicy": "IAMCCS_Ltx2HelperModules Refresh Policy",
    "IAMCCS_Ltx2HelperModules_ReanchorLatent": "IAMCCS_Ltx2HelperModules Reanchor Latent",
    "IAMCCS_Ltx2HelperModules_DiskExtension": "IAMCCS_Ltx2HelperModules Disk Extension",
    "IAMCCS_Ltx2HelperModules_RuntimeBridge": "IAMCCS_Ltx2HelperModules Runtime Bridge",
    "IAMCCS_Ltx2HelperModules_Continuity": "IAMCCS_Ltx2HelperModules Continuity",
    "IAMCCS_Ltx2HelperModules_Finalize": "IAMCCS_Ltx2HelperModules Finalize",
    "IAMCCS_AUIMG2VID_ProjectTimelinePlanner": "AU+IMG2VID Project Timeline Planner (legacy alias)",
    "IAMCCS_AUIMG2VID_Planner": "AU+IMG2VID Planner (legacy alias)",
    "IAMCCS_AUIMG2VID_AudioTimeline": "AU+IMG2VID Audio Timeline (legacy alias)",
    "IAMCCS_AUIMG2VID_KeyframeTimeline": "AU+IMG2VID Keyframe Timeline (legacy alias)",
    "IAMCCS_AUIMG2VID_RefreshPolicy": "AU+IMG2VID Refresh Policy (legacy alias)",
    "IAMCCS_AUIMG2VID_ReanchorLatent": "AU+IMG2VID Reanchor Latent (legacy alias)",
    "IAMCCS_AUIMG2VID_DiskExtension": "AU+IMG2VID Disk Extension (legacy alias)",
    "IAMCCS_AUIMG2VID_RuntimeBridge": "AU+IMG2VID Runtime Bridge (legacy alias)",
    "IAMCCS_AUIMG2VID_Continuity": "AU+IMG2VID Continuity (legacy alias)",
    "IAMCCS_AUIMG2VID_Finalize": "AU+IMG2VID Finalize (legacy alias)",
    "IAMCCS-SuperNodes AU+IMG2VID Exec Planner": "IAMCCS-SuperNodes AU+IMG2VID Exec Planner",
    "IAMCCS-SuperNodes AU+IMG2VID Exec Render": "IAMCCS-SuperNodes AU+IMG2VID Exec Render",
    "IAMCCS-SuperNodes AU+IMG2VID Exec VAE": "IAMCCS-SuperNodes AU+IMG2VID Exec VAE",
    "IAMCCS-SuperNodes Second Stage": "IAMCCS-SuperNodes Second Stage",
    "IAMCCS-SuperNodes AU+IMG2VID Exec Finalize": "IAMCCS-SuperNodes AU+IMG2VID Exec Finalize",

    "IAMCCS_SamplerAdvancedVersion1": "Sampler Advanced v1",

    "IAMCCS_bus_group": "Bus Group (Mute + Solo) (frontend-only)",

    "IAMCCS_MultiSwitch": "MultiSwitch (dynamic inputs)",
    "IAMCCS_LazyAnySwitch": "Lazy MultiGen Switch (Qwen / Flux)",
    "IAMCCS_Navigator": "IAMCCS Navigator",
    "IAMCCS_WanSviArgs": "Wan SVI Args",
    "IAMCCS_WanSviChainRunner": "Wan SVI Chain Runner",
    "IAMCCS_WanSviSegmentPick": "Wan SVI Segment Pick",

    "IAMCCS_HwSupporter": "HW Supporter (auto VRAM/attention/torch knobs)",
    "IAMCCS_HwSupporterAny": "HW Supporter (ANY passthrough)",
    "IAMCCS_HardMemoryPurge": "Hard RAM/VRAM Purge (trim working set)",
    "IAMCCS_VRAMCleanup": "VRAM Cleanup (unload + empty cache)",
    "IAMCCS_VRAMFlushLatent": "VRAM Flush ? Latent passthrough (empty cache)",
    "IAMCCS_VAEDecodeTiledSafe": "VAE Decode Tiled (safe, optional cleanup)",
    "IAMCCS_VAEDecodeToDisk": "VAE Decode ? Disk (frames, low RAM)",
    "IAMCCS_HWProbeRecommendations": "HW Probe Recommendations (JSON)",
    "IAMCCS_DetailAtelier": "IAMCCS Detail Atelier",
    "IAMCCS_DetailAtelierAdvanced": "IAMCCS Detail Atelier Advanced",
    "IAMCCS_DetailAtelierSampler": "IAMCCS Detail Atelier",

    "IAMCCS_MoveAhead": "MoveAhead (FreeLong spectral blend) ??",
    "IAMCCS_MoveAheadEnforcer": "MoveAhead Enforcer (3-tier motion lock) ??",
    "IAMCCS_MotionScale": "MotionScale (temporal RoPE scale) ?",
    "IAMCCS_MotionScaleAdvanced": "MotionScale Advanced (RoPE + theta) ?",

    "IAMCCS_MotionBridgeSave": "Motion Bridge Save ????",
    "IAMCCS_MotionBridgeLoad": "Motion Bridge Load ????",
    "IAMCCS_LatentTailSlice":  "Latent Tail Slice ??",
    "IAMCCS_AudioExtensionMath": "Audio Extension Math (timeline sync)",
    "IAMCCS_AudioExtender": "Audio Extender (segment + overlap)",
    "IAMCCS_AudioTimelineAssembler": "Audio Timeline Assembler (full track)",
    "IAMCCS_AudioTimelineGate": "Audio Timeline Gate (continue/stop)",
    "IAMCCS_BoardMaker_DialogueFoley": "IAMCCS BoardMaker Dialogue Foley",
    "IAMCCS_CineInfo3": "IAMCCS Cine Info 3",
    "IAMCCS_CineDialogueLineRouter": "IAMCCS Dialogue Line Router",
    "IAMCCS_CineTimelineAudioMixer": "IAMCCS Timeline Audio Mixer",
    "IAMCCS_AudioBoardArranger": "IAMCCS AudioBoard Arranger",
    "IAMCCS_BusOut": "IAMCCS BusOut",
    "IAMCCS_AudioBoardDirectShotboardAdapter": "IAMCCS AudioBoard Direct -> Shotboard Adapter",
    "IAMCCS_AudioBoardMixer": "IAMCCS AudioBoard Mixer",
    "IAMCCS_ControlAudEfx": "IAMCCS ControlAudEfx",
    "IAMCCS_ControlAudEfxPanel": "IAMCCS ControlAudEfx Panel",
    "IAMCCS_DialogueTagEditor": "IAMCCS Dialogue Tag Editor",
    "IAMCCS_DialogueAudioBoardBridge": "IAMCCS Dialogue AudioBoard Bridge",
    "IAMCCS_CineAudioInfo": "IAMCCS CineAudioInfo",
    "IAMCCS_AudioPhaseLazyGate": "IAMCCS Audio Phase Lazy Gate",
    "IAMCCS_StoryboardFrameDesigner": "IAMCCS StoryboardFrame + TextInFrame Director",
    "IAMCCS_StoryboardFrameDesignerV2": "IAMCCS StoryboardFrame V2 + Image Canvas i2i",
    "IAMCCS_IdeoInfo": "IDEO_INFO",
    "IAMCCS_IdeoInpaintPrep": "IAMCCS Ideo Inpaint Prep",
    "IAMCCS_IdeoMaskedPixels": "IAMCCS Ideo Masked Pixels",
    "IAMCCS_IdeogramJSONPreviewPass": "IAMCCS Ideogram JSON Preview / Pass",
    "IAMCCS_IdeoNoiseDebug": "IDEO Noise Debug",
    "IAMCCS_IdeoRandomNoiseNoCache": "IDEO Random Noise No Cache",
    "IAMCCS_IdeoImageHashDebug": "IDEO Image Hash Debug",
    "IAMCCS_TargetCrop": "IAMCCS Target Crop",
    "IAMCCS_IdeoTranslate": "IAMCCS IdeoTranslate",
    "IAMCCS_IdeogramStoryboardSheet": "IAMCCS Ideogram Storyboard Sheet",
    "IAMCCS_IdeogramSheetBuilder": "IAMCCS Ideogram Sheet Builder",
    "IAMCCS_GemmaAssistLazyGate": "IAMCCS Gemma Assist Lazy Gate",
    "IAMCCS_GemmaAssistOutput": "IAMCCS Gemma Assist Output",
    "IAMCCS_StoryboardPromptContactSheet": "IAMCCS Storyboard Prompt Contact Sheet",
    "IAMCCS_GoyAICanvasPaint": "GoyAIcanvas Paint (Image + Mask)",
    "IAMCCS_FlashVSRPanelBatchPrep": "IAMCCS FlashVSR Panel Batch Prep",
    "IAMCCS_FlashVSRPanelBatchRestore": "IAMCCS FlashVSR Panel Batch Restore",
    "IAMCCS_CineVideoToWooshInputs": "IAMCCS Video To Woosh Inputs",
    "IAMCCS_CineSpeech1PromptCompiler": "IAMCCS Speech1 Prompt Compiler",
    "IAMCCS_CineAudioTranscriptPromptCompiler": "IAMCCS Audio Transcript Prompt Compiler",
    "IAMCCS_CineSpeechLength": "IAMCCS Speech Length Calculator",
    "IAMCCS_CineDialogueDurationPlanner": "IAMCCS Dialogue Duration Planner",
    "IAMCCS_CineAudioDurationProbe": "IAMCCS Audio Duration Probe",
    "IAMCCS_CineDialogueTimingReconciler": "IAMCCS Dialogue Timing Reconciler",
    "IAMCCS_CineWooshFoleyChunkPlanner": "IAMCCS Woosh Foley Chunk Planner",
    "IAMCCS_CineFinalAudioMixer": "IAMCCS Final Audio Mixer",
    "IAMCCS_CineEmotionButtons": "IAMCCS Emotion Buttons",
    "IAMCCS_CineDialoguePromptKit": "IAMCCS Dialogue Prompt Kit",
    "IAMCCS_LTX2_LastFrameBridgeSave": "LTX-2 Last Frame Bridge Save ?????",
    "IAMCCS_LTX2_BlendLatentBridge": "LTX-2 Blend Latent Bridge ???",
    "IAMCCS_LTX2_LastFrameBridgeLoad": "LTX-2 Last Frame Bridge Load ???",
    "IAMCCS_LTX2_LoadLatentBridge": "LTX-2 Load Latent Bridge ??",
    "IAMCCS_LTX2_LongVideoWrapperPrep": "LTX-2 Long Video Wrapper Prep ??",
    "IAMCCS_LTX2_LongVideoWrapperPrepDisk": "LTX-2 Long Video Wrapper Prep (Disk) ????",
    "IAMCCS_LTX2_SaveLatentBridge": "LTX-2 Save Latent Bridge ??",
    "IAMCCS_LTX2_SegmentQueueLoop": "LTX-2 Segment Queue Loop ??",
    "IAMCCS_ImageResizeBatchSafe": "Image Resize Batch Safe (IAMCCS)",
    "IAMCCS_LoadResizeSegmentFromDir": "Load + Resize Segment From Dir ??",

    # QwenVL FLF
    **({"IAMCCS_QWEN_VL_FLF":          "QwenVL FLF — First/Last Frame Prompt ??",
        "IAMCCS_QWEN_VL_FLF_Advanced": "QwenVL FLF — First/Last Frame Prompt (Advanced) ??",
    } if IAMCCS_QWEN_VL_FLF is not None else {}),

}

try:
    from .cine_wan_shotboard_pure import (
        NODE_CLASS_MAPPINGS as _IAMCCS_WAN_SHOTBOARD_PURE_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _IAMCCS_WAN_SHOTBOARD_PURE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_IAMCCS_WAN_SHOTBOARD_PURE_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_IAMCCS_WAN_SHOTBOARD_PURE_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _iamccs_wan_shotboard_pure_error:
    print(f"[IAMCCS WAN PURE] optional module not loaded: {_iamccs_wan_shotboard_pure_error}")

try:
    from .cine_multigeneration import (
        NODE_CLASS_MAPPINGS as _IAMCCS_MULTIGENERATION_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _IAMCCS_MULTIGENERATION_NODE_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_IAMCCS_MULTIGENERATION_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_IAMCCS_MULTIGENERATION_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _iamccs_multigeneration_error:
    print(f"[IAMCCS Multigeneration] optional module not loaded: {_iamccs_multigeneration_error}")

try:
    from .engine_v2v import (
        NODE_CLASS_MAPPINGS as _IAMCCS_ENGINE_V2V_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _IAMCCS_ENGINE_V2V_NODE_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_IAMCCS_ENGINE_V2V_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_IAMCCS_ENGINE_V2V_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as _iamccs_engine_v2v_error:
    print(f"[IAMCCS Engine V2V] optional module not loaded: {_iamccs_engine_v2v_error}")


try:
    import importlib.util as _iamccs_importlib_util

    _iamccs_cine_pp_path = os.path.join(os.path.dirname(__file__), "cine-pp", "post_aa.py")
    _iamccs_cine_pp_spec = _iamccs_importlib_util.spec_from_file_location(
        "iamccs_cine_pp_post_aa",
        _iamccs_cine_pp_path,
    )
    if _iamccs_cine_pp_spec is None or _iamccs_cine_pp_spec.loader is None:
        raise ImportError(f"Cannot load IAMCCS Cine-PP module from {_iamccs_cine_pp_path}")
    _iamccs_cine_pp_module = _iamccs_importlib_util.module_from_spec(_iamccs_cine_pp_spec)
    _iamccs_cine_pp_spec.loader.exec_module(_iamccs_cine_pp_module)
    NODE_CLASS_MAPPINGS.update(getattr(_iamccs_cine_pp_module, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_iamccs_cine_pp_module, "NODE_DISPLAY_NAME_MAPPINGS", {}))
except Exception as _iamccs_cine_pp_error:
    print(f"[IAMCCS Cine-PP] optional module not loaded: {_iamccs_cine_pp_error}")

WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


def _print_startup_banner() -> None:
    # Print once per process.
    if getattr(_print_startup_banner, "_done", False):
        return
    _print_startup_banner._done = True  # type: ignore[attr-defined]

    banner = r"""
  ___    _    __  __  ____ ____  ____   ____            _           
 |_ _|  / \  |  \/  |/ ___/ ___|/ ___| |  _ \ ___   ___| | _____    
  | |  / _ \ | |\/| | |  | |    \___ \ | |_) / _ \ / __| |/ / __|   
  | | / ___ \| |  | | |__| |___  ___) ||  __/ (_) | (__|   <\__ \   
 |___/_/   \_\_|  |_|\____\____||____/ |_|   \___/ \___|_|\_\___/   

"""
    log = logging.getLogger("IAMCCS")
    log.info("%s", banner)
    log.info("by IAMCCS (follow me on patreon.com/IAMCCS or carminecristalloscalzi.com)")

    try:
        keys = sorted(list(NODE_CLASS_MAPPINGS.keys()))
        log.info("IAMCCS nodes loaded: %d", len(keys))
        # Keep log readable: print in chunks.
        chunk = []
        for k in keys:
            chunk.append(k)
            if len(chunk) >= 10:
                log.info("- %s", ", ".join(chunk))
                chunk = []
        if chunk:
            log.info("- %s", ", ".join(chunk))
    except Exception:
        pass


def setup_api_routes() -> None:
    """IAMCCS API routes used by frontend widgets."""

    try:
        from server import PromptServer
        from aiohttp import web

        from .iamccs_hw_probe import recommend_settings

        routes = PromptServer.instance.routes

        @routes.get("/api/iamccs/cine/view_image")
        async def iamccs_cine_view_image(request):
            try:
                q = request.rel_url.query
                path = q.get("path", "")
                if not path:
                    return web.Response(status=400, text="Missing path")
                path = os.path.abspath(os.path.expanduser(path))
                allowed_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}
                ext = os.path.splitext(path)[1].lower()
                if ext not in allowed_ext:
                    return web.Response(status=400, text="Unsupported image extension")
                if not os.path.exists(path) or not os.path.isfile(path):
                    return web.Response(status=404, text="Image not found")
                response = web.FileResponse(path)
                response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
                response.headers["Pragma"] = "no-cache"
                response.headers["Expires"] = "0"
                return response
            except Exception as e:
                return web.Response(status=500, text=str(e))

        @routes.post("/api/iamccs/cine/video_editor/purge_parking")
        async def iamccs_video_editor_purge_parking(request):
            """Remove only the active Video Editor session's parked media."""
            try:
                from .cine_multigeneration import _parking_root, _safe_slug

                try:
                    payload = await request.json()
                except Exception:
                    payload = {}
                session_key = _safe_slug(payload.get("session_key") or "shotboard_video_editor_v1")
                all_sessions = bool(payload.get("all_sessions", False))
                session_dir = os.path.abspath(_parking_root(session_key))
                parking_root = os.path.abspath(os.path.dirname(session_dir))
                if os.path.commonpath([parking_root, session_dir]) != parking_root:
                    return web.json_response({"error": "Invalid parking session path."}, status=400)

                targets = [session_dir]
                if all_sessions:
                    targets = []
                    if os.path.isdir(parking_root):
                        for entry in os.scandir(parking_root):
                            if entry.is_dir() and not entry.is_symlink():
                                target = os.path.abspath(entry.path)
                                if os.path.commonpath([parking_root, target]) == parking_root:
                                    targets.append(target)

                deleted_files = 0
                deleted_bytes = 0
                failed_paths = []
                for target in targets:
                    if not os.path.isdir(target):
                        continue
                    for base, _, names in os.walk(target, followlinks=False):
                        for name in names:
                            candidate = os.path.join(base, name)
                            try:
                                if os.path.islink(candidate) or os.path.isfile(candidate):
                                    deleted_bytes += int(os.lstat(candidate).st_size)
                                    deleted_files += 1
                            except Exception:
                                pass
                    for entry in list(os.scandir(target)):
                        try:
                            if entry.is_symlink() or entry.is_file():
                                os.remove(entry.path)
                            elif entry.is_dir():
                                shutil.rmtree(entry.path)
                        except FileNotFoundError:
                            pass
                        except Exception as exc:
                            failed_paths.append({"path": entry.path, "error": str(exc)})
                    os.makedirs(target, exist_ok=True)

                remaining_files = 0
                remaining_bytes = 0
                remaining_root = parking_root if all_sessions else session_dir
                if os.path.isdir(remaining_root):
                    for base, _, names in os.walk(remaining_root, followlinks=False):
                        for name in names:
                            candidate = os.path.join(base, name)
                            try:
                                if os.path.islink(candidate) or os.path.isfile(candidate):
                                    remaining_files += 1
                                    remaining_bytes += int(os.lstat(candidate).st_size)
                            except Exception:
                                pass

                return web.json_response({
                    "ok": True,
                    "session_key": session_key,
                    "folder": parking_root if all_sessions else session_dir,
                    "all_sessions": all_sessions,
                    "sessions": len(targets),
                    "deleted_files": deleted_files,
                    "deleted_bytes": deleted_bytes,
                    "failed_files": len(failed_paths),
                    "failed_paths": failed_paths[:20],
                    "remaining_files": remaining_files,
                    "remaining_bytes": remaining_bytes,
                })
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=500)

        @routes.post("/api/iamccs/cine/video_editor/ensure_preview_video")
        async def iamccs_video_editor_ensure_preview_video(request):
            """Create the lightweight browser playback proxy for an existing parked take."""
            try:
                from .cine_multigeneration import _parking_root, _safe_slug, ensure_parked_take_preview_video

                try:
                    payload = await request.json()
                except Exception:
                    payload = {}
                session_key = _safe_slug(payload.get("session_key") or "shotboard_video_editor_v1")
                parking_path = str(payload.get("parking_tensor_path") or "").strip()
                if not parking_path:
                    return web.json_response({"error": "Missing parked take path."}, status=400)
                session_dir = os.path.abspath(_parking_root(session_key))
                parking_root = os.path.abspath(os.path.dirname(session_dir))
                source = os.path.abspath(parking_path)
                if os.path.commonpath([parking_root, source]) != parking_root:
                    return web.json_response({"error": "Preview source is outside IAMCCS parking."}, status=400)
                preview = ensure_parked_take_preview_video(source)
                return web.json_response({"ok": True, **preview})
            except Exception as exc:
                return web.json_response({"error": str(exc)}, status=500)

        @routes.post("/api/iamccs/cine/transform_reference")
        async def iamccs_cine_transform_reference(request):
            try:
                import json
                import re
                import time
                import folder_paths
                from PIL import Image, ImageOps

                def _float(value, default):
                    try:
                        return float(value)
                    except Exception:
                        return float(default)

                def _int(value, default):
                    try:
                        return int(round(float(value)))
                    except Exception:
                        return int(default)

                def _inside_dir(path, root):
                    try:
                        rel = os.path.relpath(os.path.abspath(path), os.path.abspath(root))
                        return rel != os.pardir and not rel.startswith(os.pardir + os.sep) and not os.path.isabs(rel)
                    except Exception:
                        return False

                def _input_relative(path, input_root):
                    try:
                        if not _inside_dir(path, input_root):
                            return None
                        return os.path.relpath(os.path.abspath(path), os.path.abspath(input_root)).replace(os.sep, "/")
                    except Exception:
                        return None

                def _safe_paste(canvas, layer, x, y):
                    x = int(round(x))
                    y = int(round(y))
                    src_l = max(0, -x)
                    src_t = max(0, -y)
                    dst_l = max(0, x)
                    dst_t = max(0, y)
                    w = min(layer.size[0] - src_l, canvas.size[0] - dst_l)
                    h = min(layer.size[1] - src_t, canvas.size[1] - dst_t)
                    if w <= 0 or h <= 0:
                        return
                    crop = layer.crop((src_l, src_t, src_l + w, src_t + h))
                    if crop.mode == "RGBA":
                        canvas.paste(crop.convert("RGB"), (dst_l, dst_t), crop.getchannel("A"))
                    else:
                        canvas.paste(crop.convert("RGB"), (dst_l, dst_t))

                data = await request.json()
                source = str(data.get("path") or data.get("source_path") or "").strip()
                if not source:
                    return web.json_response({"error": "Missing source path"}, status=400)

                input_dir = folder_paths.get_input_directory()
                source_is_absolute = os.path.isabs(source)
                if source_is_absolute:
                    source_path = os.path.abspath(os.path.expanduser(source))
                else:
                    source_path = os.path.abspath(os.path.join(input_dir, source.replace("/", os.sep)))
                if not os.path.exists(source_path) or not os.path.isfile(source_path):
                    return web.json_response({"error": f"Source image not found: {source}"}, status=404)

                target_w = max(64, min(8192, _int(data.get("width"), 768)))
                target_h = max(64, min(8192, _int(data.get("height"), 432)))
                fit_mode = str(data.get("fit_mode") or "cover").strip().lower()
                if fit_mode not in {"cover", "contain"}:
                    fit_mode = "cover"
                zoom = max(1.0, min(16.0, _float(data.get("zoom"), 1.0)))
                pan_x = max(-1.0, min(1.0, _float(data.get("pan_x"), 0.0)))
                pan_y = max(-1.0, min(1.0, _float(data.get("pan_y"), 0.0)))
                rotation = max(-45.0, min(45.0, _float(data.get("rotation"), 0.0)))

                resample_name = str(data.get("resample") or "lanczos").lower()
                resampling = {
                    "nearest": Image.Resampling.NEAREST,
                    "bilinear": Image.Resampling.BILINEAR,
                    "bicubic": Image.Resampling.BICUBIC,
                    "lanczos": Image.Resampling.LANCZOS,
                }.get(resample_name, Image.Resampling.LANCZOS)

                crop_box = None
                transform_mode = "composite"
                with Image.open(source_path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGBA")
                    src_w, src_h = im.size
                    fill = tuple(int(v) for v in im.convert("RGB").resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0)))

                    preview_crop_box = None
                    raw_crop_box = data.get("crop_box")
                    if abs(rotation) < 0.001 and isinstance(raw_crop_box, (list, tuple)) and len(raw_crop_box) >= 4:
                        try:
                            l = max(0.0, min(float(src_w - 1), float(raw_crop_box[0])))
                            t = max(0.0, min(float(src_h - 1), float(raw_crop_box[1])))
                            r = max(l + 1.0, min(float(src_w), float(raw_crop_box[2])))
                            b = max(t + 1.0, min(float(src_h), float(raw_crop_box[3])))
                            left_i = max(0, min(src_w - 1, int(round(l))))
                            top_i = max(0, min(src_h - 1, int(round(t))))
                            right_i = max(left_i + 1, min(src_w, int(round(r))))
                            bottom_i = max(top_i + 1, min(src_h, int(round(b))))
                            preview_crop_box = (left_i, top_i, right_i, bottom_i)
                        except Exception:
                            preview_crop_box = None

                    if preview_crop_box is not None:
                        crop_box = preview_crop_box
                        out = im.crop(crop_box).convert("RGB").resize((target_w, target_h), resampling)
                        transform_mode = "ui_preview_crop"
                    else:
                        fit_scale = min(target_w / float(src_w), target_h / float(src_h)) if fit_mode == "contain" else max(target_w / float(src_w), target_h / float(src_h))
                        scale = max(0.0001, fit_scale * zoom)
                        display_w = max(1, int(round(src_w * scale)))
                        display_h = max(1, int(round(src_h * scale)))
                        max_shift_x = max(0.0, (display_w - target_w) / 2.0)
                        max_shift_y = max(0.0, (display_h - target_h) / 2.0)
                        left = (target_w - display_w) / 2.0 + pan_x * max_shift_x
                        top = (target_h - display_h) / 2.0 + pan_y * max_shift_y
                        layer = im.resize((display_w, display_h), resampling)
                        if abs(rotation) > 0.001:
                            center_x = left + display_w / 2.0
                            center_y = top + display_h / 2.0
                            layer = layer.rotate(-rotation, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=(0, 0, 0, 0))
                            left = center_x - layer.size[0] / 2.0
                            top = center_y - layer.size[1] / 2.0
                        out = Image.new("RGB", (target_w, target_h), fill)
                        _safe_paste(out, layer, left, top)
                        crop_box = {
                            "display_left": left,
                            "display_top": top,
                            "display_width": layer.size[0],
                            "display_height": layer.size[1],
                            "source_width": src_w,
                            "source_height": src_h,
                        }

                source_dir = os.path.dirname(source_path)
                fallback_dir = os.path.join(input_dir, "IAMCCS_newimages")
                source_subdir = os.path.dirname(source.replace("/", os.sep)) if not source_is_absolute else ""
                if source_dir and os.path.isdir(source_dir):
                    out_dir = source_dir
                elif source_subdir:
                    out_dir = os.path.join(input_dir, source_subdir)
                else:
                    out_dir = fallback_dir
                try:
                    os.makedirs(out_dir, exist_ok=True)
                    test_path = os.path.join(out_dir, ".iamccs_write_test")
                    with open(test_path, "w", encoding="utf-8") as test_file:
                        test_file.write("ok")
                    try:
                        os.remove(test_path)
                    except Exception:
                        pass
                except Exception:
                    out_dir = fallback_dir
                    os.makedirs(out_dir, exist_ok=True)

                stem = os.path.splitext(os.path.basename(source_path))[0]
                stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")[:60] or "cine_ref"
                filename = f"{stem}_cineedit_{int(time.time() * 1000)}.png"
                out_path = os.path.abspath(os.path.join(out_dir, filename))
                out.save(out_path, "PNG", optimize=True)

                rel_path = _input_relative(out_path, input_dir)
                ui_path = out_path
                metadata = {
                    "source_path": source_path,
                    "edited_path": out_path,
                    "path": out_path,
                    "display_path": out_path,
                    "relative_path": rel_path,
                    "project_adjacent": os.path.abspath(out_dir) == os.path.abspath(source_dir),
                    "transform": {
                        "width": target_w,
                        "height": target_h,
                        "fit_mode": fit_mode,
                        "zoom": zoom,
                        "pan_x": pan_x,
                        "pan_y": pan_y,
                        "rotation": rotation,
                        "resample": resample_name,
                        "mode": transform_mode,
                        "crop_box": crop_box,
                        "crop_box_source": data.get("crop_box_source") or ("ui_preview" if transform_mode == "ui_preview_crop" else "backend_composite"),
                    },
                }
                with open(out_path + ".json", "w", encoding="utf-8") as meta_file:
                    json.dump(metadata, meta_file, indent=2)

                return web.json_response({
                    "ok": True,
                    "path": out_path,
                    "display_path": out_path,
                    "relative_path": rel_path,
                    "absolute_path": out_path,
                    "cache_bust": int(time.time() * 1000),
                    "metadata": metadata,
                })
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        @routes.post("/api/iamccs/cine/save_shotboard_package")
        async def iamccs_cine_save_shotboard_package(request):
            try:
                import base64
                import copy
                import json
                import re
                import shutil
                import time
                import folder_paths

                def _sanitize(value, fallback="cine_shotboard_package"):
                    clean = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", str(value or fallback).strip())
                    clean = re.sub(r"\s+", "_", clean).strip("._")
                    return (clean[:90] or fallback)

                def _split_paths(value):
                    if isinstance(value, list):
                        return [str(item).strip() for item in value if str(item or "").strip()]
                    raw = str(value or "").strip()
                    if not raw:
                        return []
                    try:
                        parsed = json.loads(raw)
                        if isinstance(parsed, list):
                            return [str(item).strip() for item in parsed if str(item or "").strip()]
                    except Exception:
                        pass
                    if "\n" in raw or "\r" in raw:
                        return [item.strip() for item in raw.splitlines() if item.strip()]
                    return [item.strip() for item in raw.split(",") if item.strip()] if "," in raw else [raw]

                def _add_path(paths, seen, value):
                    clean = str(value or "").strip()
                    if clean and clean not in seen:
                        seen.add(clean)
                        paths.append(clean)

                def _collect_paths(board):
                    paths = []
                    seen = set()

                    def _collect_payload(payload):
                        if not isinstance(payload, dict):
                            return
                        for path in _split_paths(payload.get("image_paths")):
                            _add_path(paths, seen, path)
                        for item in payload.get("images") or []:
                            if isinstance(item, dict):
                                _add_path(paths, seen, item.get("path") or item.get("original_path") or item.get("filename") or item.get("name"))
                        for seg in (payload.get("segments") or []):
                            if isinstance(seg, dict):
                                _add_path(paths, seen, seg.get("imageTruthPath") or seg.get("image_truth_path") or seg.get("imageFile") or seg.get("image_file") or seg.get("path"))
                        for row in (payload.get("rows") or []):
                            if isinstance(row, dict):
                                _add_path(paths, seen, row.get("imageTruthPath") or row.get("image_truth_path") or row.get("imageFile") or row.get("image_file") or row.get("path"))

                    def _collect_multi(container):
                        if not isinstance(container, dict):
                            return
                        multi = container.get("multiGeneration") if isinstance(container.get("multiGeneration"), dict) else container
                        visual_timelines = multi.get("visualTimelines") if isinstance(multi.get("visualTimelines"), dict) else {}
                        for visual in visual_timelines.values():
                            _collect_payload(visual)
                        for item in container.get("timelines") or []:
                            if isinstance(item, dict):
                                _collect_payload(item.get("timeline") if isinstance(item.get("timeline"), dict) else item)

                    for path in _split_paths(board.get("image_paths")):
                        _add_path(paths, seen, path)
                    for item in board.get("images") or []:
                        if isinstance(item, dict):
                            _add_path(paths, seen, item.get("path") or item.get("original_path") or item.get("filename") or item.get("name"))
                    for seg in (board.get("segments") or []):
                        if isinstance(seg, dict):
                            _add_path(paths, seen, seg.get("imageFile") or seg.get("image_file") or seg.get("path"))
                    timeline = board.get("timeline")
                    if isinstance(timeline, dict):
                        _collect_payload(timeline)
                        for seg in (timeline.get("segments") or []):
                            if isinstance(seg, dict):
                                _add_path(paths, seen, seg.get("imageFile") or seg.get("image_file") or seg.get("path"))
                    timeline_data = board.get("timeline_data")
                    if isinstance(timeline_data, str) and timeline_data.strip():
                        try:
                            parsed = json.loads(timeline_data)
                            for path in _split_paths(parsed.get("image_paths") if isinstance(parsed, dict) else None):
                                _add_path(paths, seen, path)
                            for seg in (parsed.get("segments") if isinstance(parsed, dict) else []) or []:
                                if isinstance(seg, dict):
                                    _add_path(paths, seen, seg.get("imageFile") or seg.get("image_file") or seg.get("path"))
                            if isinstance(parsed, dict):
                                _collect_payload(parsed)
                                _collect_multi(parsed)
                        except Exception:
                            pass
                    _collect_multi(board)
                    _collect_multi(timeline)
                    return paths

                def _resolve_source(path, input_dir):
                    clean = str(path or "").strip()
                    if not clean or clean.startswith("data:"):
                        return None
                    if os.path.isabs(clean):
                        return os.path.abspath(os.path.expanduser(clean))
                    return os.path.abspath(os.path.join(input_dir, clean.replace("/", os.sep)))

                def _rewrite_segments(segments, path_map):
                    if not isinstance(segments, list):
                        return
                    for seg in segments:
                        if not isinstance(seg, dict):
                            continue
                        for key in ("imageTruthPath", "image_truth_path", "imageFile", "image_file", "path"):
                            value = str(seg.get(key) or "").strip()
                            if value in path_map:
                                seg[key] = path_map[value]

                def _rewrite_board_paths(board, ordered_paths, path_map):
                    def _rewrite_payload(payload, rewrite_image_paths=False):
                        if not isinstance(payload, dict):
                            return
                        if rewrite_image_paths:
                            source_paths = _split_paths(payload.get("image_paths"))
                            if source_paths:
                                payload["image_paths"] = [path_map.get(path, path) for path in source_paths if path_map.get(path, path)]
                        if isinstance(payload.get("images"), list):
                            for item in payload["images"]:
                                if not isinstance(item, dict):
                                    continue
                                value = str(item.get("path") or item.get("original_path") or item.get("filename") or item.get("name") or "").strip()
                                if value in path_map:
                                    item["original_path"] = value
                                    item["path"] = path_map[value]
                        _rewrite_segments(payload.get("segments"), path_map)
                        _rewrite_segments(payload.get("rows"), path_map)

                    def _rewrite_multi(container):
                        if not isinstance(container, dict):
                            return
                        multi = container.get("multiGeneration") if isinstance(container.get("multiGeneration"), dict) else container
                        visual_timelines = multi.get("visualTimelines") if isinstance(multi.get("visualTimelines"), dict) else {}
                        for visual in visual_timelines.values():
                            _rewrite_payload(visual, True)
                        for item in container.get("timelines") or []:
                            if not isinstance(item, dict):
                                continue
                            _rewrite_payload(item, True)
                            if isinstance(item.get("timeline"), dict):
                                _rewrite_payload(item["timeline"], True)

                    board["image_paths"] = [path_map.get(path, path) for path in ordered_paths if path_map.get(path, path)]
                    if isinstance(board.get("images"), list):
                        for item in board["images"]:
                            if not isinstance(item, dict):
                                continue
                            value = str(item.get("path") or item.get("original_path") or item.get("filename") or item.get("name") or "").strip()
                            if value in path_map:
                                item["original_path"] = value
                                item["path"] = path_map[value]
                    _rewrite_segments(board.get("segments"), path_map)
                    _rewrite_segments(board.get("rows"), path_map)
                    _rewrite_multi(board)
                    if isinstance(board.get("timeline"), dict):
                        board["timeline"]["image_paths"] = [path_map.get(path, path) for path in ordered_paths if path_map.get(path, path)]
                        _rewrite_segments(board["timeline"].get("segments"), path_map)
                        _rewrite_segments(board["timeline"].get("rows"), path_map)
                        _rewrite_multi(board["timeline"])
                    if isinstance(board.get("timeline_data"), str) and board["timeline_data"].strip():
                        try:
                            parsed = json.loads(board["timeline_data"])
                            if isinstance(parsed, dict):
                                parsed["image_paths"] = [path_map.get(path, path) for path in ordered_paths if path_map.get(path, path)]
                                _rewrite_segments(parsed.get("segments"), path_map)
                                _rewrite_segments(parsed.get("rows"), path_map)
                                _rewrite_multi(parsed)
                                board["timeline_data"] = json.dumps(parsed, indent=2)
                        except Exception:
                            pass

                def _write_data_url(value, out_path):
                    header, _, payload = str(value or "").partition(",")
                    if not header.startswith("data:image") or not payload:
                        return False
                    with open(out_path, "wb") as fh:
                        fh.write(base64.b64decode(payload))
                    return True

                def _write_audio_data_url(value, out_path):
                    header, _, payload = str(value or "").partition(",")
                    if not header.startswith("data:audio/") or not payload:
                        return False
                    with open(out_path, "wb") as fh:
                        fh.write(base64.b64decode(payload))
                    return True

                def _collect_audio_paths(payload):
                    paths = []
                    seen = set()

                    def add(value):
                        clean = str(value or "").strip()
                        if clean and clean not in seen:
                            seen.add(clean)
                            paths.append(clean)

                    def visit(value, owner_key=""):
                        if isinstance(value, list):
                            for item in value:
                                visit(item, owner_key)
                        elif isinstance(value, dict):
                            # sourceSegment/sourceAudioFile can point to the original
                            # three-minute source. A package must carry the rendered
                            # excerpts/chunks and MA asset, never silently expand back
                            # to that source file.
                            for key in ("audioFile", "audio_file"):
                                if owner_key == "sourceSegment":
                                    continue
                                add(value.get(key))
                            for key, item in value.items():
                                visit(item, key)

                    visit(payload)
                    return paths

                def _rewrite_audio_paths(payload, path_map):
                    if isinstance(payload, list):
                        for item in payload:
                            _rewrite_audio_paths(item, path_map)
                        return
                    if not isinstance(payload, dict):
                        return
                    for key in ("audioFile", "sourceAudioFile", "audio_file", "source_audio_file"):
                        value = str(payload.get(key) or "").strip()
                        if value in path_map:
                            payload[f"original_{key}"] = value
                            payload[key] = path_map[value]
                    for item in payload.values():
                        _rewrite_audio_paths(item, path_map)

                data = await request.json()
                board = data.get("board")
                if not isinstance(board, dict):
                    return web.json_response({"error": "Missing board object"}, status=400)

                input_dir = folder_paths.get_input_directory()
                package_root = os.path.join(input_dir, "IAMCCS_shotboard_packages")
                package_name = _sanitize(data.get("package_name") or data.get("label") or f"cine_shotboard_{int(time.time())}")
                package_dir = os.path.join(package_root, package_name)
                images_dir = os.path.join(package_dir, "images")
                audio_dir = os.path.join(package_dir, "audio")
                os.makedirs(images_dir, exist_ok=True)
                os.makedirs(audio_dir, exist_ok=True)

                original_board = copy.deepcopy(board)
                ordered_paths = _collect_paths(original_board)
                path_map = {}
                manifest_images = []
                allowed_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".avif"}

                for index, original_path in enumerate(ordered_paths, start=1):
                    source_path = _resolve_source(original_path, input_dir)
                    source_name = os.path.basename(source_path or original_path) or f"ref_{index:03d}.png"
                    ext = os.path.splitext(source_name)[1].lower()
                    if ext not in allowed_ext:
                        ext = ".png"
                    clean_stem = _sanitize(os.path.splitext(source_name)[0], f"ref_{index:03d}")[:52]
                    filename = f"ref_{index:03d}_{clean_stem}{ext}"
                    target_path = os.path.join(images_dir, filename)
                    rel_path = "/".join(["IAMCCS_shotboard_packages", package_name, "images", filename])
                    entry = {
                        "ref": index,
                        "original_path": original_path,
                        "package_path": rel_path,
                        "filename": filename,
                    }
                    try:
                        if str(original_path).startswith("data:image"):
                            if not _write_data_url(original_path, target_path):
                                raise ValueError("Unsupported data URL")
                        else:
                            if not source_path or not os.path.isfile(source_path):
                                raise FileNotFoundError(f"Image not found: {original_path}")
                            shutil.copy2(source_path, target_path)
                            entry["source_path"] = source_path
                        entry["bytes"] = os.path.getsize(target_path)
                        path_map[original_path] = rel_path
                    except Exception as err:
                        entry["error"] = str(err)
                    manifest_images.append(entry)

                audio_path_map = {}
                manifest_audio = []
                allowed_audio_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".opus"}
                for index, original_path in enumerate(_collect_audio_paths(original_board), start=1):
                    source_path = _resolve_source(original_path, input_dir)
                    source_name = os.path.basename(source_path or original_path) or f"audio_{index:03d}.wav"
                    ext = os.path.splitext(source_name)[1].lower()
                    if ext not in allowed_audio_ext:
                        ext = ".wav"
                    clean_stem = _sanitize(os.path.splitext(source_name)[0], f"audio_{index:03d}")[:52]
                    filename = f"audio_{index:03d}_{clean_stem}{ext}"
                    target_path = os.path.join(audio_dir, filename)
                    rel_path = "/".join(["IAMCCS_shotboard_packages", package_name, "audio", filename])
                    entry = {
                        "index": index,
                        "original_path": original_path,
                        "package_path": rel_path,
                        "filename": filename,
                    }
                    try:
                        if str(original_path).startswith("data:audio"):
                            if not _write_audio_data_url(original_path, target_path):
                                raise ValueError("Unsupported audio data URL")
                        else:
                            if not source_path or not os.path.isfile(source_path):
                                raise FileNotFoundError(f"Audio not found: {original_path}")
                            shutil.copy2(source_path, target_path)
                            entry["source_path"] = source_path
                        entry["bytes"] = os.path.getsize(target_path)
                        audio_path_map[original_path] = rel_path
                    except Exception as err:
                        entry["error"] = str(err)
                    manifest_audio.append(entry)

                packaged_board = copy.deepcopy(original_board)
                _rewrite_board_paths(packaged_board, ordered_paths, path_map)
                _rewrite_audio_paths(packaged_board, audio_path_map)
                saved_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                packaged_board["metadata"] = {
                    **(packaged_board.get("metadata") or {}),
                    "packaged_at": saved_at,
                    "package_schema": "iamccs.cine.shotboard.package",
                }
                packaged_board["package"] = {
                    "name": package_name,
                    "root": package_dir,
                    "images_dir": "images",
                    "images": manifest_images,
                    "audio_dir": "audio",
                    "audio": manifest_audio,
                }

                manifest = {
                    "metadata": {
                        "schema": "iamccs.cine.shotboard.package",
                        "schema_version": 2,
                        "saved_at": saved_at,
                        "package_name": package_name,
                        "package_root": package_dir,
                    },
                    "board_file": "board.json",
                    "images_dir": "images",
                    "image_count": len(ordered_paths),
                    "images": manifest_images,
                    "audio_dir": "audio",
                    "audio_count": len(manifest_audio),
                    "audio": manifest_audio,
                }

                with open(os.path.join(package_dir, "board.json"), "w", encoding="utf-8") as fh:
                    json.dump(packaged_board, fh, indent=2, ensure_ascii=False)
                with open(os.path.join(package_dir, "manifest.json"), "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2, ensure_ascii=False)

                failed = sum(1 for item in manifest_images if item.get("error"))
                failed_audio = sum(1 for item in manifest_audio if item.get("error"))
                return web.json_response({
                    "ok": True,
                    "package_name": package_name,
                    "package_dir": package_dir,
                    "board_file": os.path.join(package_dir, "board.json"),
                    "manifest_file": os.path.join(package_dir, "manifest.json"),
                    "image_count": len(ordered_paths),
                    "failed_images": failed,
                    "images": manifest_images,
                    "audio_count": len(manifest_audio),
                    "failed_audio": failed_audio,
                    "audio": manifest_audio,
                })
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        @routes.post("/api/iamccs/audio/choose_package_folder")
        async def iamccs_audio_choose_package_folder(request):
            try:
                import os
                import folder_paths

                data = await request.json()
                initial_dir = str(data.get("initial_dir") or "").strip()
                if not initial_dir or not os.path.isdir(os.path.abspath(os.path.expanduser(initial_dir))):
                    initial_dir = os.path.join(folder_paths.get_input_directory(), "IAMCCS_audioboard_packages")
                os.makedirs(initial_dir, exist_ok=True)
                try:
                    import tkinter as tk
                    from tkinter import filedialog
                except Exception as err:
                    return web.json_response({"ok": False, "error": f"Folder picker unavailable: {err}"}, status=500)

                root = tk.Tk()
                root.withdraw()
                try:
                    root.attributes("-topmost", True)
                except Exception:
                    pass
                try:
                    folder = filedialog.askdirectory(
                        parent=root,
                        initialdir=initial_dir,
                        title="Choose AudioBoard package folder",
                        mustexist=False,
                    )
                finally:
                    try:
                        root.destroy()
                    except Exception:
                        pass
                if not folder:
                    return web.json_response({"ok": False, "cancelled": True})
                folder = os.path.abspath(os.path.expanduser(folder))
                os.makedirs(folder, exist_ok=True)
                return web.json_response({"ok": True, "folder": folder})
            except Exception as e:
                return web.json_response({"ok": False, "error": str(e)}, status=500)

        @routes.post("/api/iamccs/audio/save_audioboard_json")
        async def iamccs_audio_save_audioboard_json(request):
            try:
                import copy
                import json
                import os
                import re
                import time
                import folder_paths

                def _sanitize(value, fallback="audioboard"):
                    clean = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", str(value or fallback).strip())
                    clean = re.sub(r"\s+", "_", clean).strip("._")
                    return (clean[:90] or fallback)

                def _strip_embedded_audio_payloads(item):
                    if isinstance(item, dict):
                        if isinstance(item.get("packageAudioExcerpt"), dict):
                            meta = dict(item.get("packageAudioExcerpt") or {})
                            meta.pop("dataUrl", None)
                            item["packageAudioExcerpt"] = meta
                        if isinstance(item.get("masterExcerpt"), dict):
                            item["masterExcerpt"].pop("dataUrl", None)
                        for value in item.values():
                            _strip_embedded_audio_payloads(value)
                    elif isinstance(item, list):
                        for value in item:
                            _strip_embedded_audio_payloads(value)

                data = await request.json()
                board = data.get("board")
                if not isinstance(board, dict):
                    return web.json_response({"error": "Missing AudioBoard object"}, status=400)

                target_root = str(data.get("target_root") or data.get("package_root") or "").strip()
                if target_root:
                    target_root = os.path.abspath(os.path.expanduser(target_root))
                else:
                    target_root = os.path.join(folder_paths.get_input_directory(), "IAMCCS_audioboards")
                os.makedirs(target_root, exist_ok=True)

                board_name = _sanitize(data.get("board_name") or data.get("package_name") or data.get("label") or f"audioboard_{int(time.time())}")
                filename = f"{board_name}.json"
                out_path = os.path.join(target_root, filename)

                saved_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                saved_board = copy.deepcopy(board)
                _strip_embedded_audio_payloads(saved_board)
                saved_board["audioBoardPackageName"] = board_name
                saved_board["projectName"] = board_name
                saved_board["metadata"] = {
                    **(saved_board.get("metadata") or {}),
                    "saved_at": saved_at,
                    "schema": "iamccs.audio_board.metadata",
                    "schema_version": 1,
                    "save_mode": "metadata_only",
                    "audio_policy": "references_existing_audio",
                }
                saved_board["package"] = {
                    **(saved_board.get("package") or {}),
                    "kind": "audio_board_metadata",
                    "name": board_name,
                    "audio_policy": "references_existing_audio",
                    "includes_audio_files": False,
                    "saved_at": saved_at,
                }

                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(saved_board, fh, indent=2, ensure_ascii=False)

                return web.json_response({
                    "ok": True,
                    "board_name": board_name,
                    "file": out_path,
                    "folder": target_root,
                })
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        @routes.post("/api/iamccs/audio/save_audioboard_package")
        async def iamccs_audio_save_audioboard_package(request):
            try:
                import base64
                import copy
                import json
                import re
                import shutil
                import time
                import folder_paths

                def _sanitize(value, fallback="audioboard_package"):
                    clean = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", str(value or fallback).strip())
                    clean = re.sub(r"\s+", "_", clean).strip("._")
                    return (clean[:90] or fallback)

                def _resolve_audio_source(path, input_dir):
                    clean = str(path or "").strip()
                    if not clean or clean.startswith("data:"):
                        return None
                    if os.path.isabs(clean):
                        return os.path.abspath(os.path.expanduser(clean))
                    return os.path.abspath(os.path.join(input_dir, clean.replace("/", os.sep)))

                def _rewrite_audio_paths(board, path_map, segment_path_map=None):
                    segment_path_map = segment_path_map or {}
                    def _rewrite_segment(seg):
                        if not isinstance(seg, dict):
                            return
                        excerpt_meta = seg.get("packageAudioExcerpt") if isinstance(seg.get("packageAudioExcerpt"), dict) else {}
                        if excerpt_meta:
                            excerpt_meta.pop("dataUrl", None)
                            seg["packageAudioExcerpt"] = excerpt_meta
                        seg_id = str(seg.get("id") or "")
                        if seg_id and seg_id in segment_path_map:
                            rel_value = segment_path_map[seg_id]
                            for key in ("audioFile", "path"):
                                value = str(seg.get(key) or "").strip()
                                if value:
                                    seg[f"original_{key}"] = value
                                seg[key] = rel_value
                            seg["fileName"] = os.path.basename(rel_value)
                            if excerpt_meta:
                                excerpt_meta["package_path"] = rel_value
                                seg["packageAudioExcerpt"] = excerpt_meta
                            return
                        for key in ("audioFile", "fileName", "path"):
                            value = str(seg.get(key) or "").strip()
                            if value in path_map:
                                seg[f"original_{key}"] = value
                                seg[key] = path_map[value]
                        for key in ("sourceAudioFile", "source_audio_file"):
                            value = str(seg.get(key) or "").strip()
                            if value in path_map:
                                seg[f"original_{key}"] = value
                                seg[key] = path_map[value]

                    for seg in board.get("audioSegments") or []:
                        _rewrite_segment(seg)
                    multi = board.get("multiGeneration") if isinstance(board.get("multiGeneration"), dict) else {}
                    for key in ("audioSegmentsAll", "allAudioSegments", "sourceAudioSegmentsAll"):
                        for seg in multi.get(key) or []:
                            _rewrite_segment(seg)
                    for seg in (multi.get("audioByTimeline") or {}).values():
                        for item in seg if isinstance(seg, list) else []:
                            _rewrite_segment(item)
                    for key in ("sourceSegment", "masterAudioAsset", "master_audio_asset", "masterExcerpt"):
                        _rewrite_segment(multi.get(key))
                    for key in ("masterAudioAsset", "master_audio_asset", "masterExcerpt"):
                        _rewrite_segment(board.get(key))

                def _write_audio_data_url(value, out_path):
                    header, _, payload = str(value or "").partition(",")
                    if not header.startswith("data:audio/") or not payload:
                        return False
                    with open(out_path, "wb") as fh:
                        fh.write(base64.b64decode(payload))
                    return True

                def _master_excerpt_from_board(board):
                    if not isinstance(board, dict):
                        return {}
                    direct = board.get("masterExcerpt")
                    if isinstance(direct, dict):
                        return direct
                    multi = board.get("multiGeneration") if isinstance(board.get("multiGeneration"), dict) else {}
                    item = multi.get("masterExcerpt") if isinstance(multi.get("masterExcerpt"), dict) else {}
                    return item if isinstance(item, dict) else {}

                def _store_master_excerpt(board, item):
                    if not isinstance(board, dict) or not isinstance(item, dict):
                        return
                    board["masterExcerpt"] = item
                    multi = board.get("multiGeneration") if isinstance(board.get("multiGeneration"), dict) else {}
                    multi["masterExcerpt"] = item
                    board["multiGeneration"] = multi

                data = await request.json()
                board = data.get("board")
                if not isinstance(board, dict):
                    return web.json_response({"error": "Missing AudioBoard object"}, status=400)

                input_dir = folder_paths.get_input_directory()
                requested_root = str(data.get("target_root") or data.get("package_root") or "").strip()
                package_root = os.path.abspath(os.path.expanduser(requested_root)) if requested_root else os.path.join(input_dir, "IAMCCS_audioboard_packages")
                package_name = _sanitize(data.get("package_name") or data.get("label") or f"audioboard_{int(time.time())}")
                package_dir = os.path.join(package_root, package_name)
                audio_dir = os.path.join(package_dir, "audio")
                os.makedirs(audio_dir, exist_ok=True)
                input_abs = os.path.abspath(input_dir).lower()
                package_root_abs = os.path.abspath(package_root).lower()
                package_root_is_input = package_root_abs == input_abs or package_root_abs.startswith(input_abs + os.sep)

                def _package_audio_ref(filename, target_path):
                    if package_root_is_input:
                        rel_to_input = os.path.relpath(target_path, input_dir).replace(os.sep, "/")
                        return rel_to_input
                    return os.path.abspath(target_path)

                original_board = copy.deepcopy(board)
                segments = original_board.get("audioSegments") if isinstance(original_board.get("audioSegments"), list) else []
                manifest_audio = []
                path_map = {}
                segment_path_map = {}
                master_excerpt_entry = {}
                allowed_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".opus"}

                for index, seg in enumerate(segments, start=1):
                    if not isinstance(seg, dict):
                        continue
                    original_path = str(seg.get("audioFile") or seg.get("fileName") or seg.get("path") or "").strip()
                    entry = {
                        "index": index,
                        "id": str(seg.get("id") or f"audio_{index:03d}"),
                        "track": int(float(seg.get("track", 0) or 0)),
                        "start": int(float(seg.get("start", 0) or 0)),
                        "length": int(float(seg.get("length", 0) or 0)),
                        "original_path": original_path,
                    }
                    source_path = _resolve_audio_source(original_path, input_dir)
                    source_name = os.path.basename(source_path or original_path) or f"audio_{index:03d}.wav"
                    ext = os.path.splitext(source_name)[1].lower()
                    if ext not in allowed_ext:
                        ext = ".wav"
                    stem = _sanitize(os.path.splitext(source_name)[0], f"audio_{index:03d}")[:52]
                    filename = f"audio_{index:03d}_{stem}{ext}"
                    excerpt = seg.get("packageAudioExcerpt") if isinstance(seg.get("packageAudioExcerpt"), dict) else {}
                    if excerpt:
                        filename = _sanitize(excerpt.get("fileName") or f"audio_{index:03d}_{stem}_excerpt.wav", f"audio_{index:03d}_excerpt.wav")
                        if not filename.lower().endswith(".wav"):
                            filename = f"{os.path.splitext(filename)[0]}.wav"
                    target_path = os.path.join(audio_dir, filename)
                    rel_path = _package_audio_ref(filename, target_path)
                    try:
                        if excerpt:
                            if _write_audio_data_url(excerpt.get("dataUrl"), target_path):
                                entry.update({
                                    "source_path": source_path or "",
                                    "bytes": os.path.getsize(target_path),
                                    "package_path": rel_path,
                                    "package_mode": "excerpt",
                                    "trim_start_frame": int(float(excerpt.get("trimStartFrame", seg.get("trimStart", 0)) or 0)),
                                    "length_frames": int(float(excerpt.get("lengthFrames", seg.get("length", 0)) or 0)),
                                    "fps": float(excerpt.get("fps", 24) or 24),
                                    "duration_seconds": float(excerpt.get("durationSeconds", 0) or 0),
                                    "mime": "audio/wav",
                                })
                                if original_path:
                                    path_map[original_path] = rel_path
                                segment_path_map[str(seg.get("id") or f"audio_{index:03d}")] = rel_path
                            else:
                                entry["error"] = excerpt.get("error") or "Missing segment excerpt dataUrl; refused to copy full source audio"
                        elif source_path and os.path.isfile(source_path):
                            shutil.copy2(source_path, target_path)
                            entry["source_path"] = source_path
                            entry["bytes"] = os.path.getsize(target_path)
                            entry["package_path"] = rel_path
                            entry["package_mode"] = "source_copy"
                            if original_path:
                                path_map[original_path] = rel_path
                        else:
                            entry["error"] = excerpt.get("error") or f"Audio not found: {original_path}"
                    except Exception as err:
                        entry["error"] = str(err)
                    manifest_audio.append(entry)

                master_excerpt = _master_excerpt_from_board(original_board)
                if master_excerpt:
                    master_excerpt_entry = {
                        "role": "master_excerpt",
                        "source_segment_id": str(master_excerpt.get("sourceSegmentId") or ""),
                        "source_audio_file": str(master_excerpt.get("sourceAudioFile") or ""),
                        "trim_start_frame": int(float(master_excerpt.get("trimStartFrame", 0) or 0)),
                        "length_frames": int(float(master_excerpt.get("lengthFrames", 0) or 0)),
                        "fps": float(master_excerpt.get("fps", 24) or 24),
                        "duration_seconds": float(master_excerpt.get("durationSeconds", 0) or 0),
                    }
                    filename = _sanitize(master_excerpt.get("fileName") or "master_excerpt.wav", "master_excerpt.wav")
                    if not filename.lower().endswith(".wav"):
                        filename = f"{os.path.splitext(filename)[0]}.wav"
                    target_path = os.path.join(audio_dir, filename)
                    rel_path = _package_audio_ref(filename, target_path)
                    try:
                        if _write_audio_data_url(master_excerpt.get("dataUrl"), target_path):
                            master_excerpt_entry.update({
                                "filename": filename,
                                "package_path": rel_path,
                                "bytes": os.path.getsize(target_path),
                                "mime": "audio/wav",
                            })
                        else:
                            master_excerpt_entry["error"] = master_excerpt.get("error") or "Missing master excerpt dataUrl"
                    except Exception as err:
                        master_excerpt_entry["error"] = str(err)

                packaged_board = copy.deepcopy(original_board)
                _rewrite_audio_paths(packaged_board, path_map, segment_path_map)
                if master_excerpt:
                    packaged_excerpt = copy.deepcopy(master_excerpt)
                    packaged_excerpt.pop("dataUrl", None)
                    if master_excerpt_entry.get("package_path"):
                        packaged_excerpt["audioFile"] = master_excerpt_entry["package_path"]
                        packaged_excerpt["path"] = master_excerpt_entry["package_path"]
                        packaged_excerpt["packagePath"] = master_excerpt_entry["package_path"]
                    if master_excerpt_entry.get("error"):
                        packaged_excerpt["error"] = master_excerpt_entry["error"]
                    _store_master_excerpt(packaged_board, packaged_excerpt)
                saved_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")
                packaged_board["metadata"] = {
                    **(packaged_board.get("metadata") or {}),
                    "packaged_at": saved_at,
                    "package_schema": "iamccs.audio.audioboard.package",
                }
                packaged_board["package"] = {
                    "name": package_name,
                    "root": package_dir,
                    "audio_dir": "audio",
                    "audio": manifest_audio,
                    "master_excerpt": master_excerpt_entry,
                }

                manifest = {
                    "metadata": {
                        "schema": "iamccs.audio.audioboard.package",
                        "schema_version": 1,
                        "saved_at": saved_at,
                        "package_name": package_name,
                        "package_root": package_dir,
                    },
                    "board_file": "audioboard.json",
                    "audio_dir": "audio",
                    "audio_count": len(manifest_audio),
                    "failed_audio": sum(1 for item in manifest_audio if item.get("error")),
                    "audio": manifest_audio,
                    "master_excerpt": master_excerpt_entry,
                    "trackSettings": packaged_board.get("trackSettings") or [],
                    "masterBus": packaged_board.get("masterBus") or {},
                }

                with open(os.path.join(package_dir, "audioboard.json"), "w", encoding="utf-8") as fh:
                    json.dump(packaged_board, fh, indent=2, ensure_ascii=False)
                with open(os.path.join(package_dir, "manifest.json"), "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2, ensure_ascii=False)

                return web.json_response({
                    "ok": True,
                    "package_name": package_name,
                    "package_dir": package_dir,
                    "board_file": os.path.join(package_dir, "audioboard.json"),
                    "manifest_file": os.path.join(package_dir, "manifest.json"),
                    "audio_count": len(manifest_audio),
                    "failed_audio": manifest["failed_audio"],
                    "audio": manifest_audio,
                })
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        @routes.post("/api/iamccs/audio/restore_package_assets")
        async def iamccs_audio_restore_package_assets(request):
            try:
                import copy
                import os
                import re
                import shutil
                import time
                import folder_paths

                data = await request.json()
                board = data.get("board") if isinstance(data.get("board"), dict) else {}
                if not board:
                    return web.json_response({"error": "Missing AudioBoard object"}, status=400)

                def _sanitize(value, fallback="audioboard_import"):
                    clean = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", str(value or fallback).strip())
                    clean = re.sub(r"\s+", "_", clean).strip("._")
                    return clean[:80] or fallback

                def _resolve(path, input_dir):
                    clean = str(path or "").strip()
                    if not clean or clean.startswith("data:"):
                        return ""
                    if os.path.isabs(clean):
                        return os.path.abspath(os.path.expanduser(clean))
                    return os.path.abspath(os.path.join(input_dir, clean.replace("/", os.sep)))

                candidates = []
                seen = set()

                def _add(item):
                    if not isinstance(item, dict):
                        return
                    path = str(item.get("audioFile") or item.get("path") or "").strip()
                    if not path or path in seen:
                        return
                    seen.add(path)
                    candidates.append(path)

                for item in board.get("audioSegments") or []:
                    _add(item)
                multi = board.get("multiGeneration") if isinstance(board.get("multiGeneration"), dict) else {}
                for key in ("audioSegmentsAll", "allAudioSegments", "sourceAudioSegmentsAll"):
                    for item in multi.get(key) or []:
                        _add(item)
                for items in (multi.get("audioByTimeline") or {}).values():
                    for item in items if isinstance(items, list) else []:
                        _add(item)
                for key in ("masterAudioAsset", "master_audio_asset", "masterExcerpt"):
                    _add(board.get(key))
                    _add(multi.get(key))

                input_dir = folder_paths.get_input_directory()
                package_name = _sanitize(data.get("package_name") or board.get("audioBoardPackageName") or board.get("projectName") or f"audioboard_{int(time.time())}")
                target_dir = os.path.join(input_dir, "IAMCCS_imported_audioboard", package_name)
                os.makedirs(target_dir, exist_ok=True)
                mapping = {}
                restored = []
                allowed_ext = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".aiff", ".aif", ".opus"}
                for index, original in enumerate(candidates, start=1):
                    source = _resolve(original, input_dir)
                    if not source or not os.path.isfile(source):
                        continue
                    name = os.path.basename(source)
                    stem, ext = os.path.splitext(name)
                    if ext.lower() not in allowed_ext:
                        ext = ".wav"
                    filename = f"{index:03d}_{_sanitize(stem, 'audio')[:52]}{ext}"
                    target = os.path.join(target_dir, filename)
                    shutil.copy2(source, target)
                    rel = os.path.relpath(target, input_dir).replace(os.sep, "/")
                    mapping[original] = rel
                    restored.append({"original_path": original, "input_path": rel, "bytes": os.path.getsize(target)})

                return web.json_response({"ok": True, "mapping": mapping, "restored": restored})
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

        @routes.get("/api/iamccs/hw_probe")
        async def iamccs_hw_probe_endpoint(request):
            try:
                q = request.rel_url.query
                def _to_int(x):
                    try:
                        return int(float(x))
                    except Exception:
                        return None
                def _to_float(x):
                    try:
                        return float(x)
                    except Exception:
                        return None

                width = _to_int(q.get("width"))
                height = _to_int(q.get("height"))
                frames = _to_int(q.get("frames"))
                fps = _to_float(q.get("fps"))

                data = recommend_settings(width=width, height=height, frames=frames, fps=fps)
                logging.getLogger("IAMCCS.API").info(
                    "[iamccs/hw_probe] cuda=%s vram_gb=%s ram_gb=%s profile=%s vae_tile=%s frames=%s fps=%s",
                    data.get("hardware", {}).get("cuda_available"),
                    data.get("hardware", {}).get("cuda_total_vram_gb"),
                    data.get("hardware", {}).get("system_ram_gb"),
                    data.get("recommendations", {}).get("hw_supporter", {}).get("profile"),
                    data.get("recommendations", {}).get("vae_decode", {}).get("tile_size"),
                    frames,
                    fps,
                )
                return web.json_response(data)
            except Exception as e:
                return web.json_response({"error": str(e)}, status=500)

    except Exception as e:
        # Never hard-fail ComfyUI startup due to optional API endpoints.
        logging.getLogger("IAMCCS.API").warning("Could not setup IAMCCS API routes: %r", e)



# IAMCCS Shotboard V4 / Motion Sketch optional module
try:
    from .cine_shotboard_v4 import (
        NODE_CLASS_MAPPINGS as _IAMCCS_SHOTBOARD_V4_NODE_CLASS_MAPPINGS,
        NODE_DISPLAY_NAME_MAPPINGS as _IAMCCS_SHOTBOARD_V4_NODE_DISPLAY_NAME_MAPPINGS,
    )
    NODE_CLASS_MAPPINGS.update(_IAMCCS_SHOTBOARD_V4_NODE_CLASS_MAPPINGS)
    NODE_DISPLAY_NAME_MAPPINGS.update(_IAMCCS_SHOTBOARD_V4_NODE_DISPLAY_NAME_MAPPINGS)
except Exception as e:
    logging.getLogger("IAMCCS").warning("IAMCCS Shotboard V4 MotionSketch nodes unavailable: %r", e)

# Setup API routes when extension loads
setup_api_routes()

# Print banner after we are fully imported and mappings exist.
_print_startup_banner()


def _iamccs_install_ltx2_vae_encode_autofix() -> None:
    """Prevents hard-crash when LTX-2 VAE receives invalid frame counts.

    Lightricks video VAE encode requires a frame count of the form 1 + 8*x.
    Some workflows can produce off-by-a-few batches (e.g. 240 instead of 241),
    which otherwise raises ValueError and stops execution.

    This patch pads by repeating the last frame up to the next valid count.
    Opt-in via IAMCCS_LTX2_VAE_ENCODE_AUTOFIX=1.
    """

    # Default OFF: user requested workflow-level fixes without monkeypatching VAE.
    if str(os.getenv("IAMCCS_LTX2_VAE_ENCODE_AUTOFIX", "0")).strip().lower() in {"0", "false", "no", "off"}:
        return

    log = logging.getLogger("IAMCCS.LTX2.VAE")

    try:
        import torch
    except Exception:
        return

    try:
        from comfy.ldm.lightricks.vae import causal_video_autoencoder as _cvae
    except Exception:
        # ComfyUI / LTXVideo not installed or import path changed.
        return

    cls = getattr(_cvae, "CausalVideoAutoencoder", None)
    if cls is None:
        return

    orig_encode = getattr(cls, "encode", None)
    if orig_encode is None:
        return

    if getattr(orig_encode, "__iamccs_ltx2_autofix__", False):
        return

    def _round_up_8n1(frames: int) -> int:
        frames = int(frames)
        if frames <= 1:
            return 1
        rem = (frames - 1) % 8
        if rem == 0:
            return frames
        return frames + (8 - rem)

    def _is_valid_8n1(frames: int) -> bool:
        frames = int(frames)
        return frames >= 1 and (frames - 1) % 8 == 0

    def _pad_repeat_last(x: "torch.Tensor", dim: int, pad: int) -> "torch.Tensor":
        # Take last slice along `dim` (keeps dimension) and repeat it `pad` times.
        slc = [slice(None)] * x.ndim
        slc[dim] = slice(-1, None)
        last = x[tuple(slc)]
        reps = [1] * x.ndim
        reps[dim] = int(pad)
        last_rep = last.repeat(*reps)
        return torch.cat([x, last_rep], dim=dim)

    def _candidate_frame_dims(x: "torch.Tensor") -> list[int]:
        # Most common layouts:
        # - (B, C, T, H, W)  -> frames dim = 2
        # - (T, H, W, C)     -> frames dim = 0 (ComfyUI IMAGE batches)
        # We only try dims that are >1 and *not obviously channels*.
        dims: list[int] = []
        if x.ndim == 5:
            # Prefer T, then fallbacks
            dims = [2, 0, 1]
        elif x.ndim == 4:
            dims = [0]
        else:
            dims = [0]

        out: list[int] = []
        for d in dims:
            try:
                size = int(x.shape[d])
            except Exception:
                continue
            if size <= 1:
                continue
            # Heuristic: channels are usually small (1..4). Don't treat that as frames.
            if size in (1, 2, 3, 4) and x.ndim >= 4 and d in (1, 3):
                continue
            out.append(d)
        # Ensure uniqueness, preserve order
        seen = set()
        unique: list[int] = []
        for d in out:
            if d in seen:
                continue
            seen.add(d)
            unique.append(d)
        return unique

    def encode_patched(self, pixels_in: "torch.Tensor"):
        try:
            return orig_encode(self, pixels_in)
        except ValueError as e:
            msg = str(e)
            if "Invalid number of frames" not in msg:
                raise

            if not isinstance(pixels_in, torch.Tensor) or pixels_in.ndim < 4:
                raise

            # Try padding along the most likely frame dimension(s).
            last_err: Exception | None = e
            for dim in _candidate_frame_dims(pixels_in):
                frames_in = int(pixels_in.shape[dim])
                if _is_valid_8n1(frames_in):
                    continue

                frames_fixed = _round_up_8n1(frames_in)
                pad = frames_fixed - frames_in
                if pad <= 0:
                    continue

                try:
                    pixels_fixed = _pad_repeat_last(pixels_in, dim=dim, pad=pad)
                    log.warning(
                        "[LTX2 VAE encode autofix] Padded frames dim=%d %d -> %d (pad=%d) to satisfy 1+8*x rule",
                        dim,
                        frames_in,
                        frames_fixed,
                        pad,
                    )
                    return orig_encode(self, pixels_fixed)
                except Exception as ee:
                    last_err = ee
                    continue

            # If all attempts failed, re-raise the original ValueError.
            raise e

    encode_patched.__iamccs_ltx2_autofix__ = True
    setattr(cls, "encode", encode_patched)


_iamccs_install_ltx2_vae_encode_autofix()

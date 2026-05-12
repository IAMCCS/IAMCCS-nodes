# ==========================================================
# __init__.py — Registro nodi IAMCCS
# ==========================================================

import logging
import os

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

from .iamccs_wdc_ltx_port import (
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
    IAMCCS_CineShotboardLite,
    IAMCCS_CineShotboardPlannerProLegacy,
    IAMCCS_CineInfo,
    IAMCCS_CineSwitch,
    IAMCCS_CinePromptRelayLatentShapeSync,
    IAMCCS_CineFLFLengthCompensator,
    IAMCCS_CinePromptRelaySafeEncode,
    IAMCCS_CineRelayOrBypass,
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
    "IAMCCS_CineShotboardLite": IAMCCS_CineShotboardLite,
    "IAMCCS_CineShotboardPlannerProLegacy": IAMCCS_CineShotboardPlannerProLegacy,
    "IAMCCS_CineInfo": IAMCCS_CineInfo,
    "IAMCCS_CineSwitch": IAMCCS_CineSwitch,
    "IAMCCS_CinePromptRelayLatentShapeSync": IAMCCS_CinePromptRelayLatentShapeSync,
    "IAMCCS_CineFLFLengthCompensator": IAMCCS_CineFLFLengthCompensator,
    "IAMCCS_CinePromptRelaySafeEncode": IAMCCS_CinePromptRelaySafeEncode,
    "IAMCCS_CineRelayOrBypass": IAMCCS_CineRelayOrBypass,
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

    "IAMCCS_HwSupporter": IAMCCS_HwSupporter,
    "IAMCCS_HwSupporterAny": IAMCCS_HwSupporterAny,
    "IAMCCS_HardMemoryPurge": IAMCCS_HardMemoryPurge,
    "IAMCCS_VRAMCleanup": IAMCCS_VRAMCleanup,
    "IAMCCS_VRAMFlushLatent": IAMCCS_VRAMFlushLatent,
    "IAMCCS_VAEDecodeTiledSafe": IAMCCS_VAEDecodeTiledSafe,
    "IAMCCS_VAEDecodeToDisk": IAMCCS_VAEDecodeToDisk,
    "IAMCCS_HWProbeRecommendations": IAMCCS_HWProbeRecommendations,

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

    # QwenVL First/Last Frame (registered only if QwenVL is installed)
    **({"IAMCCS_QWEN_VL_FLF": IAMCCS_QWEN_VL_FLF,
        "IAMCCS_QWEN_VL_FLF_Advanced": IAMCCS_QWEN_VL_FLF_Advanced,
    } if IAMCCS_QWEN_VL_FLF is not None else {}),

}

NODE_DISPLAY_NAME_MAPPINGS = {
    "IAMCCS_WanLoRAStack": "LoRA Stack (WAN-style remap)",
    "IAMCCS_ModelWithLoRA": "Apply LoRA to MODEL (Native)",
    "IAMCCS_WanLoRAStackModelIO": "LoRA Stack (Model In→Out) WAN",
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
    "IAMCCS_LTX2_LoRAStackModelIO": "LoRA Stack (Model In→Out) LTX-2",
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
    "IAMCCS_LTX2_ExtensionModule": "LTX-2 Extension Module 🎬",
    "IAMCCS_LTX2_ExtensionModule_Disk": "LTX-2 Extension Module (Disk / Low RAM) 💾",
    "IAMCCS_LTX2_JointRefreshLatent": "LTX-2 Joint Refresh Latent",
    "IAMCCS_LTX2_JointRefreshLatent_Disk": "LTX-2 Joint Refresh Latent (Disk)",
    "IAMCCS_LoadImagesFromDirLite": "Load Images From Dir (Lite) 📁",
    "IAMCCS_ImageBatchRangeLite": "Image Batch Range (VRAM) 🎞️",
    "IAMCCS_SourceFramesToDisk": "Source Frames To Disk 📼💾",
    "IAMCCS_StartDirToVideoLatent": "Start Dir To Video Latent 🚀",
    "IAMCCS_StartImagesToVideoLatent": "Start Images To Video Latent 🚀",
    "IAMCCS_VideoCombineFromDir": "Video Combine From Dir 🎞️",
    "IAMCCS_LTX2_ExtensionModule_simple": "LTX-2 Extension Module (simple) 🎬",
    "IAMCCS_LTX2_GetImageFromBatch": "LTX-2 Get Images From Batch 🎞️",
    "IAMCCS_LTX2_ReferenceImageSwitch": "LTX-2 Reference Image Switch 🧷",
    "IAMCCS_LTX2_ReferenceStartFramesInjector": "LTX-2 Inject Reference Into Start Frames 🧬",
    "IAMCCS_LTX2_FrameCountValidator": "LTX-2 Frame Count Validator ✅ (8n+1)",
    "IAMCCS_LTX2_FirstLastFramesController": "LTX-2 First/Last Frames Controller 🧲",
    "IAMCCS_LTX2_ContextLatent": "LTX-2 Context → Latent (continue) 🧩",
    "IAMCCS_LTX2_MiddleFrames": "LTX-2 Middle Frames (accumulator) 🧷",
    "IAMCCS_LTX2_FirstLastLatentControl": "LTX-2 First/Last → Latent (noise_mask) 🎯",
    "IAMCCS_LTX2_FirstLastLatentControl_Pro": "LTX-2 First/Last → Latent (Pro, slot caps) 🎯",
    "IAMCCS_CineReferenceBoard": "IAMCCS Cine Reference Board",
    "IAMCCS_CineLTXSequencer": "IAMCCS Cine FLF Timeline Sequencer",
    "IAMCCS_CineAllInOneFLFEngine": "IAMCCS Cine AllInOne FLF Engine",
    "IAMCCS_CinePromptRelayTimeline": "IAMCCS Cine PromptRelay Timeline",
    "IAMCCS_CineShotboardTimelinePro": "IAMCCS Cine Shotboard Timeline Pro",
    "IAMCCS_CineShotboardPlannerPro": "IAMCCS Cine Shotboard Planner Pro",
    "IAMCCS_CineShotboardPlannerProV2": "IAMCCS Cine Shotboard Planner Pro V2",
    "IAMCCS_CineShotboardLite": "IAMCCS Cine Shotboard Lite",
    "IAMCCS_CineShotboardPlannerProLegacy": "IAMCCS Cine Shotboard Planner Pro Legacy Outputs",
    "IAMCCS_CineInfo": "IAMCCS CineInfo",
    "IAMCCS_CineSwitch": "IAMCCS CineSwitch Lazy FLF/PromptRelay",
    "IAMCCS_CinePromptRelayLatentShapeSync": "IAMCCS Cine PromptRelay Latent Shape Sync",
    "IAMCCS_CineFLFLengthCompensator": "IAMCCS Cine FLF Length Compensator",
    "IAMCCS_CinePromptRelaySafeEncode": "IAMCCS Cine PromptRelay Safe Encode",
    "IAMCCS_CineRelayOrBypass": "IAMCCS Cine Relay Or Bypass",
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
    "IAMCCS_WDC_LTXSequencerFixed5": "IAMCCS Cine LTX Sequencer Fixed 5 (legacy alias)",
    "IAMCCS_LTX2_InitLatentSampler": "LTX-2 Init Latent Sampler 🧱",
    "IAMCCS_LTX2_LoopingSampler": "LTX-2 Looping Sampler (temporal overlap) 🧷",
    "IAMCCS_LTX2_OneShotLowRAMLooper": "LTX-2 One-Shot Low-RAM Looper 🪶",
    "IAMCCS_LTX2_ExtendSampler": "LTX-2 Extend Sampler (temporal overlap) 🧷",
    "IAMCCS_LTX2_ConditionNextLatentWithPrevOverlap": "LTX-2 Condition Next Latent (prev overlap) 🧷",
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
    "IAMCCS_WanSVIToFLFBridgeProPlus": "Wan SVI→FLF Bridge Pro Plus",
    "IAMCCS_WanSVIToFLFBridgeProPlus_Simple": "Wan SVI→FLF Bridge Pro Plus",
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
    "IAMCCS_WanSviArgs": "Wan SVI Args",
    "IAMCCS_WanSviChainRunner": "Wan SVI Chain Runner",
    "IAMCCS_WanSviSegmentPick": "Wan SVI Segment Pick",

    "IAMCCS_HwSupporter": "HW Supporter (auto VRAM/attention/torch knobs)",
    "IAMCCS_HwSupporterAny": "HW Supporter (ANY passthrough)",
    "IAMCCS_HardMemoryPurge": "Hard RAM/VRAM Purge (trim working set)",
    "IAMCCS_VRAMCleanup": "VRAM Cleanup (unload + empty cache)",
    "IAMCCS_VRAMFlushLatent": "VRAM Flush → Latent passthrough (empty cache)",
    "IAMCCS_VAEDecodeTiledSafe": "VAE Decode Tiled (safe, optional cleanup)",
    "IAMCCS_VAEDecodeToDisk": "VAE Decode → Disk (frames, low RAM)",
    "IAMCCS_HWProbeRecommendations": "HW Probe Recommendations (JSON)",

    "IAMCCS_MoveAhead": "MoveAhead (FreeLong spectral blend) 🎬",
    "IAMCCS_MoveAheadEnforcer": "MoveAhead Enforcer (3-tier motion lock) 🎬",
    "IAMCCS_MotionScale": "MotionScale (temporal RoPE scale) ⚡",
    "IAMCCS_MotionScaleAdvanced": "MotionScale Advanced (RoPE + theta) ⚡",

    "IAMCCS_MotionBridgeSave": "Motion Bridge Save 🎬💾",
    "IAMCCS_MotionBridgeLoad": "Motion Bridge Load 🎬📂",
    "IAMCCS_LatentTailSlice":  "Latent Tail Slice ✂️",
    "IAMCCS_AudioExtensionMath": "Audio Extension Math (timeline sync)",
    "IAMCCS_AudioExtender": "Audio Extender (segment + overlap)",
    "IAMCCS_AudioTimelineAssembler": "Audio Timeline Assembler (full track)",
    "IAMCCS_AudioTimelineGate": "Audio Timeline Gate (continue/stop)",
    "IAMCCS_LTX2_LastFrameBridgeSave": "LTX-2 Last Frame Bridge Save 🖼️💾",
    "IAMCCS_LTX2_BlendLatentBridge": "LTX-2 Blend Latent Bridge 🎚️",
    "IAMCCS_LTX2_LastFrameBridgeLoad": "LTX-2 Last Frame Bridge Load 🖼️",
    "IAMCCS_LTX2_LoadLatentBridge": "LTX-2 Load Latent Bridge 🧬",
    "IAMCCS_LTX2_LongVideoWrapperPrep": "LTX-2 Long Video Wrapper Prep 🧰",
    "IAMCCS_LTX2_LongVideoWrapperPrepDisk": "LTX-2 Long Video Wrapper Prep (Disk) 💾🧰",
    "IAMCCS_LTX2_SaveLatentBridge": "LTX-2 Save Latent Bridge 💾",
    "IAMCCS_LTX2_SegmentQueueLoop": "LTX-2 Segment Queue Loop 🔁",
    "IAMCCS_ImageResizeBatchSafe": "Image Resize Batch Safe (IAMCCS)",
    "IAMCCS_LoadResizeSegmentFromDir": "Load + Resize Segment From Dir 🧰",

    # QwenVL FLF
    **({"IAMCCS_QWEN_VL_FLF":          "QwenVL FLF — First/Last Frame Prompt 🎬",
        "IAMCCS_QWEN_VL_FLF_Advanced": "QwenVL FLF — First/Last Frame Prompt (Advanced) 🎬",
    } if IAMCCS_QWEN_VL_FLF is not None else {}),

}

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
                return web.FileResponse(path)
            except Exception as e:
                return web.Response(status=500, text=str(e))

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

                data = await request.json()
                source = str(data.get("path") or data.get("source_path") or "").strip()
                if not source:
                    return web.json_response({"error": "Missing source path"}, status=400)

                input_dir = folder_paths.get_input_directory()
                if os.path.isabs(source):
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

                with Image.open(source_path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    if abs(rotation) > 0.001:
                        fill = tuple(int(v) for v in im.resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0)))
                        im = im.rotate(rotation, resample=Image.Resampling.BICUBIC, expand=True, fillcolor=fill)
                    src_w, src_h = im.size
                    crop_box = None
                    preview_crop_box = None
                    raw_crop_box = data.get("crop_box")
                    if isinstance(raw_crop_box, (list, tuple)) and len(raw_crop_box) >= 4:
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
                        out = im.crop(crop_box).resize((target_w, target_h), resampling)
                    elif fit_mode == "contain":
                        bg = Image.new("RGB", (target_w, target_h), tuple(int(v) for v in im.resize((1, 1), Image.Resampling.BILINEAR).getpixel((0, 0))))
                        fitted = ImageOps.contain(im, (target_w, target_h), method=resampling)
                        paste_x = int(round((target_w - fitted.size[0]) / 2.0))
                        paste_y = int(round((target_h - fitted.size[1]) / 2.0))
                        bg.paste(fitted, (paste_x, paste_y))
                        out = bg
                    else:
                        aspect = target_w / float(target_h)
                        if src_w / float(src_h) >= aspect:
                            base_h = src_h
                            base_w = base_h * aspect
                        else:
                            base_w = src_w
                            base_h = base_w / aspect
                        crop_w = max(1.0, min(src_w, base_w / zoom))
                        crop_h = max(1.0, min(src_h, base_h / zoom))
                        max_shift_x = max(0.0, (src_w - crop_w) / 2.0)
                        max_shift_y = max(0.0, (src_h - crop_h) / 2.0)
                        center_x = src_w / 2.0 + pan_x * max_shift_x
                        center_y = src_h / 2.0 + pan_y * max_shift_y
                        left = max(0.0, min(src_w - crop_w, center_x - crop_w / 2.0))
                        top = max(0.0, min(src_h - crop_h, center_y - crop_h / 2.0))
                        crop_box = (int(round(left)), int(round(top)), int(round(left + crop_w)), int(round(top + crop_h)))
                        out = im.crop(crop_box).resize((target_w, target_h), resampling)

                out_dir = os.path.join(input_dir, "IAMCCS_newimages")
                os.makedirs(out_dir, exist_ok=True)
                stem = os.path.splitext(os.path.basename(source_path))[0]
                stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")[:60] or "cine_ref"
                filename = f"{stem}_cinecrop_{int(time.time() * 1000)}.png"
                out_path = os.path.join(out_dir, filename)
                out.save(out_path, "PNG", optimize=True)

                rel_path = "IAMCCS_newimages/" + filename
                metadata = {
                    "source_path": source_path,
                    "edited_path": out_path,
                    "relative_path": rel_path,
                    "transform": {
                        "width": target_w,
                        "height": target_h,
                        "fit_mode": fit_mode,
                        "zoom": zoom,
                        "pan_x": pan_x,
                        "pan_y": pan_y,
                        "rotation": rotation,
                        "resample": resample_name,
                        "crop_box_after_rotation": crop_box,
                        "crop_box_source": data.get("crop_box_source") or ("ui_preview" if preview_crop_box is not None else "backend_formula"),
                    },
                }
                with open(out_path + ".json", "w", encoding="utf-8") as meta_file:
                    json.dump(metadata, meta_file, indent=2)

                return web.json_response({
                    "ok": True,
                    "path": rel_path,
                    "absolute_path": out_path,
                    "metadata": metadata,
                })
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

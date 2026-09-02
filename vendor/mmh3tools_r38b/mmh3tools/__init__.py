from . import patch_guide_origin as _patch_guide_origin
from . import patch_ref_labels as _patch_ref_labels
from .nodes_imagelist import MMH3ImageList
from .nodes_loop import MMH3SeedOverlap, MMH3ConcatAV, MMH3FindDivergence, MMH3JoinAV, MMH3PackAV
from .nodes_align import MMH3ForcedAlign
from .nodes_lyricwindows import MMH3LyricsToWindows
from .nodes_motion import MMH3MotionOverload
from .nodes_schedule import MMH3ChunkSchedule, MMH3ChunkScheduleFrames
from .nodes_music_analysis import MMH3MusicAnalysis
from .nodes_musicscene import MMH3MusicScenePlanPrompt
from .nodes_stylepack import MMH3LoadSkill
from .nodes_controlnet import MMH3CondSetApplyControl
from .nodes_encode import MMH3StreamingEncode
from .nodes_lint import MMH3PromptLint
from .nodes_music import (MMH3LyricsSectionize, MMH3MusicCaptionSplit,
                          MMH3MusicCaptionSystemPrompt)
from .nodes_looping_sampler import MMH3KeyframePlanner, MMH3LoopingSampler
from .nodes_save import MMH3SizeCappedCopy, MMH3StreamingSave
from .nodes_upscale import MMH3ChunkedPixelUpscale
from .nodes_tokens import MMH3OfficialTokens
from .nodes_trim import (
    MMH3OutpaintLatent,
    MMH3SplitAV,
    MMH3TrimAV,
)
from .nodes_multiprompt import (
    MMH3CondSelect,
    MMH3CondToSet,
    MMH3CondSetSpread,
    MMH3CondSetStripText,
    MMH3ReferenceMultiPrompt,
)
from .nodes_prompt import (
    MMH3AssetPlan,
    MMH3PromptAccumulate,
    MMH3ReplaceSection,
    MMH3TaskSystemPrompt,
)
from .nodes_refprobe import MMH3RefAttentionMap, MMH3RefAttentionProbe
from .nodes_scene import (
    MMH3PromptPart,
    MMH3ScenePlanPrompt,
)
from .nodes_refs import (
    MMH3ImageKeyframe,
    MMH3ImageToRef,
    MMH3LatentKeyframe,
    MMH3LatentToRef,
    MMH3ReferenceFromLatent,
    MMH3Regenerate2KReference,
)
from .nodes_whisper import WhisperAlignmentToText
from .nodes_windows import (
    MMH3ContextWindows,
    MMH3SplitAudioToWindows,
    MMH3WindowContext,
    MMH3WindowPlan,
)
from .nodes_util import (
    MMH3AdaLNRefPatch,
    MMH3DimensionCalculator,
    MMH3FrameCalculator,
    MMH3LatentInfo,
    MMH3Regenerate2KDims,
    MMH3ReframePads,
    MMH3UpscaleLadder,
)

_patch_guide_origin.apply()
_patch_ref_labels.apply()

NODES = [
    # MMH3Tools
    MMH3DimensionCalculator,
    MMH3FrameCalculator,
    # MMH3Tools/sampling
    MMH3LoopingSampler,
    MMH3ContextWindows,
    # MMH3Tools/calculators
    MMH3ChunkSchedule,
    MMH3ImageList,
    MMH3CondSetApplyControl,
    WhisperAlignmentToText,
    MMH3RefAttentionProbe,
    MMH3RefAttentionMap,
    MMH3ChunkScheduleFrames,
    MMH3WindowPlan,
    MMH3KeyframePlanner,
    MMH3UpscaleLadder,
    MMH3Regenerate2KDims,
    # MMH3Tools/model
    MMH3AdaLNRefPatch,
    # MMH3Tools/prompt
    MMH3AssetPlan,
    MMH3TaskSystemPrompt,
    MMH3ScenePlanPrompt,
    MMH3MusicScenePlanPrompt,
    MMH3LoadSkill,
    MMH3WindowContext,
    MMH3PromptAccumulate,
    MMH3PromptPart,
    MMH3ReplaceSection,
    MMH3PromptLint,
    MMH3MusicCaptionSystemPrompt,
    MMH3MusicCaptionSplit,
    MMH3LyricsSectionize,
    # MMH3Tools/conditioning
    MMH3ReferenceMultiPrompt,
    MMH3Regenerate2KReference,
    MMH3CondSelect,
    MMH3CondToSet,
    MMH3CondSetSpread,
    MMH3CondSetStripText,
    # MMH3Tools/reference
    MMH3ReferenceFromLatent,
    MMH3ImageToRef,
    MMH3LatentToRef,
    MMH3ImageKeyframe,
    MMH3LatentKeyframe,
    # MMH3Tools/latent
    MMH3PackAV,
    MMH3SplitAV,
    MMH3JoinAV,
    MMH3ConcatAV,
    MMH3TrimAV,
    MMH3SeedOverlap,
    MMH3ChunkedPixelUpscale,
    # MMH3Tools/audio
    MMH3SplitAudioToWindows,
    MMH3ForcedAlign,
    MMH3LyricsToWindows,
    MMH3MusicAnalysis,
    MMH3OfficialTokens,
    # MMH3Tools/utils
    MMH3LatentInfo,
    MMH3MotionOverload,
    MMH3FindDivergence,
    MMH3ReframePads,
    MMH3OutpaintLatent,
    MMH3StreamingEncode,
    MMH3StreamingSave,
    MMH3SizeCappedCopy,
]

__all__ = ["NODES"]








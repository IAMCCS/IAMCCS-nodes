from .iamccs_supernodes_auimg2vid_exec_backend import (
    IAMCCS_GC_AUIMG2VIDExecutableFinalize as _GCFinalize,
    IAMCCS_GC_AUIMG2VIDExecutablePlanner as _GCPlanner,
    IAMCCS_GC_AUIMG2VIDExecutableRender as _GCRender,
    IAMCCS_GC_AUIMG2VIDExecutableVAE as _GCVAE,
)


class IAMCCS_SuperNodes_AUIMG2VIDExecutablePlanner(_GCPlanner):
    CATEGORY = "IAMCCS/SuperNodes"


class IAMCCS_SuperNodes_AUIMG2VIDExecutableRender(_GCRender):
    CATEGORY = "IAMCCS/SuperNodes"


class IAMCCS_SuperNodes_AUIMG2VIDExecutableVAE(_GCVAE):
    CATEGORY = "IAMCCS/SuperNodes"


if _GCFinalize is not None:
    class IAMCCS_SuperNodes_AUIMG2VIDExecutableFinalize(_GCFinalize):
        CATEGORY = "IAMCCS/SuperNodes"
else:
    IAMCCS_SuperNodes_AUIMG2VIDExecutableFinalize = None
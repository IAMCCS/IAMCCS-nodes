# IAMCCS Bus Group - frontend-only utility node
# The behavior is implemented in web/iamccs_bus_group.js


class IAMCCS_bus_group:
    """Bus Group controller (frontend-only).

    This node exists so workflows can serialize it; all muting/solo logic is handled
    by the ComfyUI frontend extension.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "noop"
    CATEGORY = "IAMCCS/Utilities"

    def noop(self):
        return ()

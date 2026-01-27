# IAMCCS AutoLink - Wireless node connections
# 1:1 copy of KJ nodes functionality (Set/Get/Converter)

# Questi nodi sono PURAMENTE FRONTEND - non fanno nulla in Python
# Tutta la logica è in JavaScript

class IAMCCS_SetAutoLink:
    """Set AutoLink - Virtual node (frontend only)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "IAMCCS/AutoLink"

    def noop(self):
        # Questo non viene mai eseguito - il nodo è virtuale
        return ()


class IAMCCS_GetAutoLink:
    """Get AutoLink - Virtual node (frontend only)"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "IAMCCS/AutoLink"

    def noop(self):
        # Questo non viene mai eseguito - il nodo è virtuale
        return ()


class IAMCCS_AutoLinkConverter:
    """AutoLink Converter - UI tool"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "arg": ("AUTOLINK_ARG",),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    CATEGORY = "IAMCCS/AutoLink"

    def noop(self, arg=None):
        return ()


class IAMCCS_AutoLinkArguments:
    """AutoLink Arguments - Configuration node"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("AUTOLINK_ARG",)
    FUNCTION = "noop"
    CATEGORY = "IAMCCS/AutoLink"

    def noop(self):
        return (None,)


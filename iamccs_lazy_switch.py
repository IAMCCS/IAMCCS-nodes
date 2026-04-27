import logging


_log = logging.getLogger("IAMCCS.LazySwitch")


class IAMCCS_LazyAnySwitch:
    """
    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    By Carmine Cristallo Scalzi AI research (IAMCCS) - patreon.com/IAMCCS - carminecristalloscalzi.com
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "select": (["Qwen", "Flux"], {"default": "Qwen"}),
            },
            "optional": {
                "input1": ("*", {"lazy": True}),
                "input2": ("*", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("output",)
    FUNCTION = "run"
    CATEGORY = "IAMCCS/Logic"

    @staticmethod
    def _normalize_selection(select):
        if isinstance(select, str):
            value = select.strip().lower()
            if value in ("qwen", "input1", "1"):
                return 1
            if value in ("flux", "input2", "2"):
                return 2
        try:
            return 1 if int(select) <= 1 else 2
        except Exception:
            return 1

    def check_lazy_status(self, select, input1=None, input2=None, **kwargs):
        selected = self._normalize_selection(select)
        if selected == 1 and input1 is None:
            return ["input1"]
        if selected == 2 and input2 is None:
            return ["input2"]
        return []

    def run(self, select, input1=None, input2=None):
        selected = self._normalize_selection(select)
        if selected == 1:
            if input1 is None:
                raise ValueError("IAMCCS_LazyAnySwitch: input1 is not connected or not available")
            _log.info("[IAMCCS_LazyAnySwitch] selecting Qwen/input1")
            return (input1,)

        if input2 is None:
            raise ValueError("IAMCCS_LazyAnySwitch: input2 is not connected or not available")
        _log.info("[IAMCCS_LazyAnySwitch] selecting Flux/input2")
        return (input2,)
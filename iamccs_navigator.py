from __future__ import annotations


class IAMCCS_Navigator:
    """Lightweight visual bookmark marker for large ComfyUI workflows."""

    CATEGORY = "IAMCCS/Navigation"
    RETURN_TYPES = ()
    FUNCTION = "noop"
    DESCRIPTION = "Canvas bookmark marker used by the IAMCCS Navigator frontend."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bookmark_name": (
                    "STRING",
                    {"default": "Bookmark", "multiline": False},
                ),
                "category": (
                    "STRING",
                    {"default": "General", "multiline": False},
                ),
                "color": (
                    [
                        "teal",
                        "amber",
                        "green",
                        "red",
                        "violet",
                        "blue",
                        "gray",
                        "custom",
                    ],
                    {"default": "teal"},
                ),
                "custom_color": (
                    "STRING",
                    {"default": "#22c7a9", "multiline": False},
                ),
                "icon": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "note": (
                    "STRING",
                    {"default": "", "multiline": False},
                ),
                "zoom_mode": (
                    ["keep current zoom", "restore saved zoom"],
                    {"default": "keep current zoom"},
                ),
                "saved_zoom": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.1, "max": 3.0, "step": 0.05},
                ),
                "order": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999, "step": 1},
                ),
                "show_in_index": ("BOOLEAN", {"default": True}),
            }
        }

    def noop(
        self,
        bookmark_name,
        category,
        color,
        custom_color,
        icon,
        note,
        zoom_mode,
        saved_zoom,
        order,
        show_in_index,
    ):
        return ()

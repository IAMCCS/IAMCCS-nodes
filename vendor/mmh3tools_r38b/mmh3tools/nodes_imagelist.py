"""Collect many reference images WITHOUT conforming them to one frame.

A batch is a single tensor and a tensor cannot be ragged, so every batching node in
reach -- core's `ImageBatch` and `BatchImagesNode`, KJNodes' `ImageBatchMulti` --
resizes and CENTRE-CROPS every image to the first one's shape before anything
downstream sees it:

    # resize all images to be the same size as the first image
    comfy.utils.common_upscale(..., first_image_shape[2], first_image_shape[1],
                               "bilinear", "center")

References of different shapes therefore arrive already cropped, and nothing
downstream can detect it -- by then they genuinely do share one frame.

A Python LIST has no such constraint. KJNodes' `ImageTensorList` returns one, but takes
exactly two inputs, so N references need N-1 chained nodes. This is the same idea with
an Autogrow input: one node, one socket per reference, each keeping its native size and
getting its own aspect-correct target inside MMH3 Reference (Multi-Prompt).

Nothing here resizes, pads or reorders. Socket order is `<Picture i>` order.
"""

import logging

from comfy_api.latest import io

MAX_IMAGES = 50
# Past this the token cost is worth a word; references are attended at EVERY step.
COST_WARN_AT = 9


class MMH3ImageList(io.ComfyNode):
    """Many reference images, each at its own size."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ImageList",
            display_name="MMH3 Image List",
            category="MMH3Tools/reference",
            description=(
                "Collect reference images into a LIST rather than a batch, so each keeps "
                "its native size.\n\n"
                "Batching cannot preserve differing shapes -- a tensor cannot be ragged, "
                "so core's Batch Images and KJNodes' ImageBatchMulti both resize and "
                "CENTRE-CROP everything to the first image. Feed this to MMH3 Reference "
                "(Multi-Prompt)'s `ref_images` and each reference gets its own "
                "aspect-correct target instead.\n\n"
                "Socket order is <Picture i> order. Nothing is resized, padded or "
                "reordered. Empty sockets are skipped, so you can leave gaps."
            ),
            inputs=[
                io.Autogrow.Input(
                    "images",
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("image"), prefix="image_",
                        min=1, max=MAX_IMAGES)),
            ],
            outputs=[
                io.Image.Output(display_name="images"),
                io.Int.Output(display_name="count"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, images: io.Autogrow.Type) -> io.NodeOutput:
        # dict order is socket order, which is <Picture i> order
        out = [img for img in images.values() if img is not None]
        if not out:
            raise ValueError(
                "MMH3ImageList: no images wired. Every socket is empty, and an empty "
                "list would silently give the reference node nothing to work with.")

        lines = ["MMH3 Image List -- %d reference%s"
                 % (len(out), "" if len(out) == 1 else "s"), ""]
        sizes = []
        for i, img in enumerate(out):
            if hasattr(img, "shape") and img.ndim == 4:
                n, h, w = int(img.shape[0]), int(img.shape[1]), int(img.shape[2])
                sizes.append((w, h, n))
                lines.append("  <Picture %d>  %dx%d%s"
                             % (i + 1, w, h,
                                "  (a BATCH of %d -- every frame becomes its own "
                                "<Picture>)" % n if n > 1 else ""))
        distinct = {(w, h) for w, h, _n in sizes}
        lines.append("")
        if len(distinct) > 1:
            lines.append("  %d distinct shapes, all preserved. Batching these would have "
                         "cropped every one to %dx%d."
                         % (len(distinct), sizes[0][0], sizes[0][1]))
        else:
            lines.append("  all one shape, so a batch would have been equivalent here.")

        total = sum(n for _w, _h, n in sizes)
        if total > COST_WARN_AT:
            lines.append("")
            lines.append("  ! %d references. They are attended at EVERY sampling step, "
                         "so this is paid on every chunk of every pass. The hosted API "
                         "caps at 9; the open model has no such limit, only the cost."
                         % total)
        logging.info("[MMH3ImageList] %d reference(s), %d distinct shape(s)",
                     len(out), len(distinct))
        return io.NodeOutput(out, total, "\n".join(lines))

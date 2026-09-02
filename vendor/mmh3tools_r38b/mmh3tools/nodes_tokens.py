"""H3's seven added special tokens, which ComfyUI's shared tokenizer does not have.

ComfyUI routes H3 text through `qwen25_tokenizer/`, whose added_tokens_decoder stops
at 151668. H3 adds seven ids on top of stock Qwen3-VL, so ComfyUI tokenizes `<d>` and
its siblings as ordinary subwords that merge with neighbouring whitespace, language
tags and punctuation. This node adds them to a COPY of the tokenizer.
"""
import copy
import logging

from comfy_api.latest import io

# The ids from the H3 tokenizer config, in order. They are contiguous from the end of
# the stock added-token range, so appending in this order reproduces them exactly --
# but the node VERIFIES rather than trusting that, since a core tokenizer change would
# silently shift the base and hand the model seven wrong rows.
H3_TOKENS = [
    ("<d>", 151669),
    ("</d>", 151670),
    ("<|cutoff|>", 151671),
    ("<|lyrics_start|>", 151672),
    ("<|lyrics_end|>", 151673),
    ("<|caption_start|>", 151674),
    ("<|caption_end|>", 151675),
]

_PROBE = "he says <d>[English] we need to leave now.</d>"


def _inner(clip):
    """The transformers tokenizer inside a CLIP, or None if this is not H3."""
    tk = getattr(clip, "tokenizer", None)
    sd = getattr(tk, "qwen3vl_32b", None)
    return getattr(sd, "tokenizer", None) if sd is not None else None


class MMH3OfficialTokens(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3OfficialTokens",
            display_name="MMH3 Official H3 Tokens",
            category="MMH3Tools/conditioning",
            description=(
                "Add H3's seven special tokens to the CLIP's tokenizer, so <d>, "
                "</d> and the lyrics/caption markers encode as single reserved ids "
                "instead of ordinary subwords. Patches a COPY -- the incoming CLIP "
                "is untouched, so bypassing this node reverts it."
            ),
            inputs=[
                io.Clip.Input(
                    "clip",
                    tooltip="The H3 text encoder, from CLIPLoader. Wire this node "
                            "between the loader and whatever encodes prompts."),
                io.Boolean.Input(
                    "enabled", default=True,
                    tooltip="Off passes the CLIP through unchanged, so the node can "
                            "stay wired while A/B testing."),
            ],
            outputs=[
                io.Clip.Output(display_name="clip"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, clip, enabled=True) -> io.NodeOutput:
        if not enabled:
            return io.NodeOutput(clip, "disabled -- CLIP passed through unchanged")

        inner = _inner(clip)
        if inner is None:
            raise ValueError(
                "MMH3OfficialTokens: this CLIP is not a MiniMax H3 text encoder -- no "
                "`qwen3vl_32b` tokenizer on it. The seven tokens belong to H3 only, and "
                "adding them to another model's tokenizer would shift ids it does have. "
                "Wire the H3 CLIPLoader here.")

        if inner.convert_tokens_to_ids("<d>") == H3_TOKENS[0][1]:
            return io.NodeOutput(clip, "already patched -- <d> is %d, nothing to do"
                                 % H3_TOKENS[0][1])

        before = len(inner(_PROBE, add_special_tokens=False)["input_ids"])

        # A COPY, because CLIP.clone() shares the tokenizer by reference: patching in
        # place would follow the loaded model around and survive bypassing this node.
        from transformers.tokenization_utils import AddedToken
        out = clip.clone()
        out.tokenizer = copy.deepcopy(clip.tokenizer)
        tok = _inner(out)

        base = len(tok)
        added = tok.add_tokens(
            [AddedToken(t, special=True, normalized=False, lstrip=False, rstrip=False)
             for t, _ in H3_TOKENS],
            special_tokens=True)

        wrong = [(t, want, tok.convert_tokens_to_ids(t))
                 for t, want in H3_TOKENS if tok.convert_tokens_to_ids(t) != want]
        if wrong:
            raise ValueError(
                "MMH3OfficialTokens: the tokens did not land on their H3 ids, so the "
                "model would be handed the wrong embedding rows. Expected the vocab to "
                "end at %d and it ends at %d.\n%s\nThe CLIP has NOT been modified."
                % (H3_TOKENS[0][1], base,
                   "\n".join("  %-20s want %d, got %s" % w for w in wrong)))

        after = len(tok(_PROBE, add_special_tokens=False)["input_ids"])
        report = (
            "added %d token%s at %d-%d\n%s\nprobe %r: %d ids -> %d"
            % (added, "" if added == 1 else "s", H3_TOKENS[0][1], H3_TOKENS[-1][1],
               "\n".join("  %-20s %d" % (t, i) for t, i in H3_TOKENS),
               _PROBE, before, after))
        logging.info("[MMH3OfficialTokens] added %d tokens at %d-%d (probe %d -> %d ids)",
                     added, H3_TOKENS[0][1], H3_TOKENS[-1][1], before, after)
        return io.NodeOutput(out, report)

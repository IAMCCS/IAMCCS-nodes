"""Multi-prompt reference conditioning: encode the references ONCE, the prompts N times.

For a text-driven sequence with locked identity, every chunk shares the same
references and differs only in its prompt. Stock MiniMaxH3ReferenceToVideo does
the reference resize, vae.encode and audio_vae.encode in the SAME execute() as the
text encode, so N chunks means N copies of all of it -- and, worse, N model swap
cycles, because Qwen3-VL-32B and a 33B DiT cannot be resident together. ComfyUI
resolves outputs depth-first, so a naive N-chunk graph runs
load TE -> cond -> evict -> load DiT -> sample -> evict -> load TE -> ...

Doing every encode inside ONE node execution collapses that to a single swap.

What is still paid per prompt: clip.tokenize re-presents the references to Qwen and
the vision tower plus 50 layers run again. That is inherent -- references are
emitted BEFORE the prompt text, and while comfy/text_encoders/llama.py does thread
past_key_values through every layer, the CLIP API exposes no way to hand it a
cached prefix. Cheap for still images; the thing to avoid for video references.
"""

import hashlib
import logging
import math

import torch

import node_helpers
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    CANVAS_MULTIPLE,
    FPS,
    REF_IMAGE_SHORT_EDGE,
    MiniMaxH3ReferenceToVideo,
    _empty_av_latent,
    _encode_ref_audio,
    _resize,
    adapt_canvas,
)

from .common import evict_text_encoder
from .nodes_windows import _plan, _window_frame_spans

MMH3CondSet = io.Custom("MMH3_COND_SET")

# (prompt, ref fingerprint) -> conditioning. Editing ONE prompt re-executes the
# whole node, so without this a one-word change costs every prompt's Qwen pass.
_CACHE = {}
_CACHE_MAX = 64


def _hash_tensor(h, t):
    """Strided sample of a tensor into a hash. O(4096) regardless of size."""
    h.update(str(tuple(t.shape)).encode())
    flat = t.detach().flatten()
    step = max(1, flat.numel() // 4096)
    h.update(flat[::step].to(torch.float32).cpu().numpy().tobytes())


def _hash_input(h, obj):
    """Hash a raw reference input: IMAGE tensor, or AUDIO {waveform, sample_rate}."""
    if obj is None:
        h.update(b"none")
    elif isinstance(obj, dict) and "waveform" in obj:
        h.update(str(obj.get("sample_rate")).encode())
        _hash_tensor(h, obj["waveform"])
    elif isinstance(obj, (list, tuple)):
        # a reference LIST: hash each entry, since repr() of a tensor list is
        # truncated and two different sets can share one
        h.update(b"list:%d" % len(obj))
        for item in obj:
            _hash_input(h, item)
    elif hasattr(obj, "shape"):
        _hash_tensor(h, obj)
    else:
        h.update(repr(obj).encode())


def _fingerprint(ref_blocks, raw_inputs, width, height, length, ref_image_size):
    """Identify a reference set cheaply but honestly.

    Hashes BOTH the raw inputs and the encoded blocks. The blocks alone would be
    tempting -- they are already in hand and capture every sizing decision -- but
    that makes cache validity depend on the VAE mapping different references to
    different latents. That holds for a real VAE and is exactly the assumption you
    do not want load-bearing when the failure mode is a stale encode with no
    visible symptom: the wrong reference, silently, in every chunk. Hashing the
    inputs costs a strided read of a few images and removes the assumption.
    """
    h = hashlib.sha256()
    h.update(("%d|%d|%d|%s" % (width, height, length, ref_image_size)).encode())
    for obj in raw_inputs:
        _hash_input(h, obj)
    for b in ref_blocks:
        h.update(("%s|%s|%s|%s|%s" % (b.get("kind"), b.get("latent_h"), b.get("latent_w"),
                                      b.get("latent_t"), b.get("ref_audio_t"))).encode())
        for key in ("latent", "audio_latent"):
            t = b.get(key)
            if t is not None:
                _hash_tensor(h, t)
    return h.hexdigest()


def _embedding_rows(name):
    """How many token slots this embedding occupies, from the file header alone."""
    try:
        import folder_paths
        from safetensors import safe_open
        path = folder_paths.get_full_path("embeddings", name)
        if path is None:
            for ext in (".safetensors", ".pt", ".bin"):
                path = folder_paths.get_full_path("embeddings", name + ext)
                if path is not None:
                    break
        if path is None or not str(path).endswith(".safetensors"):
            return None
        with safe_open(path, "pt") as f:
            for k in f.keys():
                shape = f.get_slice(k).get_shape()
                return int(shape[0]) if len(shape) == 2 else None
    except Exception:
        return None
    return None


def _known_embeddings():
    try:
        import folder_paths
        return set(folder_paths.get_filename_list("embeddings"))
    except Exception:
        return set()


def _parse_embeddings(spec, n_chunks):
    """Lines of `name` or `name: all|N|A-B` -> the names each chunk should carry.

    A bare name means EVERY chunk, which is the ordinary case: one look applied to
    the whole piece. The range form exists because prompts are per chunk anyway, so
    scheduling an effect is only a question of which prompts get the token.
    Indices are 1-based, matching how the chunks are talked about everywhere else.
    """
    per = [[] for _ in range(max(0, n_chunks))]
    for raw in (spec or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        # strip the marker BEFORE splitting: partition() takes the FIRST colon, so
        # `embedding:name: all` would otherwise parse as name="embedding"
        if line.startswith("embedding:"):
            line = line[len("embedding:"):].strip()
        name, _, rng = line.partition(":")
        name = name.strip()
        if not name:
            continue
        rng = rng.strip().lower()
        if not rng or rng == "all":
            idx = range(n_chunks)
        elif "-" in rng:
            a, _, b = rng.partition("-")
            idx = range(max(0, int(a) - 1), min(n_chunks, int(b)))
        else:
            k = int(rng) - 1
            idx = [k] if 0 <= k < n_chunks else []
        for i in idx:
            if name not in per[i]:
                per[i].append(name)
    return per


def _embeddings_resolve(clip, name):
    """Does THIS core actually splice `embedding:` in an H3 prompt?

    Before #15808 the H3 tokenizer never looked for the marker and the words went
    through as ordinary text -- no error, no embedding. Probed rather than inferred
    from a version, because the failure is invisible in the output.
    """
    try:
        out = clip.tokenize("embedding:%s" % name)
        for entries in out.values():
            for batch in entries:
                for tok, *_rest in batch:
                    if hasattr(tok, "shape"):
                        return True
    except Exception:
        return False
    return False


def _window_media(frames, soundtrack, span, fps=FPS):
    """A reference video and its audio cut to `span` = (first_frame, last_frame).

    The audio is cut on the SAME clock rather than by latent arithmetic: the two
    streams run on independent grids (24 fps against 40 Hz) and the conversion is not
    additive, so cutting each from seconds is exact where cutting one from the other
    accumulates drift.
    """
    a, b = span
    out_frames = frames[a:b + 1] if frames is not None else None
    out_audio = soundtrack
    if soundtrack is not None and soundtrack.get("waveform") is not None:
        sr = int(soundtrack.get("sample_rate") or 0)
        wf = soundtrack["waveform"]
        if sr > 0 and wf.ndim == 3:
            s0 = int(round(a / float(fps) * sr))
            s1 = int(round((b + 1) / float(fps) * sr))
            s0 = max(0, min(s0, int(wf.shape[-1])))
            s1 = max(s0, min(s1, int(wf.shape[-1])))
            out_audio = dict(soundtrack)
            out_audio["waveform"] = wf[..., s0:s1].contiguous()
    return out_frames, out_audio


def _ref_windows(total_frames, chunk_frames, overlap_frames):
    """(first, last) per chunk, from the SAME plan the sampler runs.

    `standard_static` and the argument order are copied from the sampler's own call,
    not chosen here -- if these drifted apart, reference window i would stop being the
    span chunk i renders and every chunk would be conditioned on somebody else's
    footage, silently.
    """
    _length, _overlap, total_f, _total_t, windows = _plan(
        int(total_frames), int(chunk_frames), int(overlap_frames), "standard_static")
    return _window_frame_spans(windows, total_f)


def _build_refs(vae, audio_vae, width, height, frame_count, ref_image_size,
                ref_images, ref_videos, ref_video_audios, ref_audios,
                ref_window=None):
    """Reference items (for the tokenizer) and blocks (for the DiT), built once.

    DUPLICATED FROM comfy_extras/nodes_minimax_h3.py, deliberately: upstream runs
    this inline in the same execute() as the text encode, so there is no seam to
    call. Re-sync if that file changes its sizing, its block keys, or -- most
    fragile -- the emission ORDER, since the tokenizer assigns <Picture i>,
    <Audio j> and <Video k> labels by counting items in the order given. A video's
    soundtrack must be appended BEFORE the video itself or every label after it
    shifts and the prompt's tags stop matching.
    """
    ref_items = []
    ref_blocks = []

    # Every reference is its own <Picture i>, in the order given. Accepts either
    # form:
    #   BATCH  [N,H,W,C] -- one tensor, so every image ALREADY shares H,W. Whatever
    #                       batched them (core's ImageBatch, KJNodes'
    #                       ImageBatchMulti) resized and centre-cropped them to the
    #                       first one's frame before this node ran. Nothing here can
    #                       undo that, so it is reported instead.
    #   LIST   [t1, t2]  -- each entry keeps its NATIVE size and gets its own
    #                       aspect-correct target, which is what core's per-socket
    #                       node does. KJNodes' ImageTensorList emits this.
    #
    # Core's node slices img[:1] per socket, so a batch wired into one of its slots
    # contributes only its first frame; this expands instead.
    if ref_images is not None:
        if isinstance(ref_images, (list, tuple)):
            # A list entry may itself hold several frames -- one socket fed from a
            # LoadImage of a multi-page file, say. Expand those too, so N frames is
            # N references however they arrived. Entries keep their own size either
            # way; only entries batched TOGETHER were ever conformed.
            frames = []
            for f in ref_images:
                if f is None:
                    continue
                if hasattr(f, "shape") and f.ndim == 4 and int(f.shape[0]) > 1:
                    frames.extend(f[i:i + 1] for i in range(int(f.shape[0])))
                else:
                    frames.append(f)
        else:
            frames = [ref_images[i:i + 1] for i in range(int(ref_images.shape[0]))]
            if len(frames) > 1:
                logging.info(
                    "[MMH3ReferenceMultiPrompt] %d references arrived as one BATCH, so "
                    "they all share %dx%d -- whatever batched them conformed the rest "
                    "to the first. Wire a LIST (KJNodes ImageTensorList) to keep each "
                    "reference's own size.", len(frames),
                    int(ref_images.shape[2]), int(ref_images.shape[1]))
        sizes = []
        for img in frames:
            h, w = img.shape[1], img.shape[2]
            if ref_image_size == "match":
                scale = min(1.0, math.sqrt((width * height) / (w * h)))
            else:
                scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(w, h))
            tw = max(CANVAS_MULTIPLE, round(w * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            th = max(CANVAS_MULTIPLE, round(h * scale / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            resized = _resize(img, tw, th, "disabled")
            z = vae.encode(resized)
            ref_items.append({"type": "image", "data": resized})
            ref_blocks.append({"kind": "image", "latent_h": th // 16,
                               "latent_w": tw // 16, "latent": z})
            sizes.append("%dx%d->%dx%d" % (w, h, tw, th))
        if sizes:
            logging.info("[MMH3ReferenceMultiPrompt] <Picture 1..%d>: %s",
                         len(sizes), "  ".join(sizes))

    ref_video_audios = ref_video_audios or {}
    for name, video_frames in (ref_videos or {}).items():
        if video_frames is None:
            continue
        soundtrack = ref_video_audios.get("ref_video_audio_" + name.rsplit("_", 1)[-1])
        if ref_window is not None:
            # cut BEFORE the resize and the encode: windowing after would pay the
            # full reference's VAE and vision cost to then throw most of it away
            video_frames, soundtrack = _window_media(video_frames, soundtrack,
                                                     ref_window)
        vh, vw = video_frames.shape[1], video_frames.shape[2]
        cw, ch = adapt_canvas(vw, vh)
        if vw * vh < cw * ch:
            cw = max(CANVAS_MULTIPLE, round(vw / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
            ch = max(CANVAS_MULTIPLE, round(vh / CANVAS_MULTIPLE) * CANVAS_MULTIPLE)
        frames = _resize(video_frames, cw, ch, "disabled")
        if frames.shape[0] > frame_count:
            frames = frames[:frame_count]
        n = frames.shape[0]
        if n < 5:
            raise ValueError("MiniMax H3 reference videos need at least 5 frames (~0.2s at 24 fps)")
        while n % 17 != 5:
            n -= 1
        frames = frames[:n]
        z = vae.encode(frames)
        audio_latent, ref_audio_t = (None, 0)
        if soundtrack is not None:
            audio_latent, ref_audio_t = _encode_ref_audio(
                audio_vae, _to_stereo(soundtrack, "reference video soundtrack"))
            ref_items.append({"type": "audio"})
        sample_idx = list(range(0, frames.shape[0], FPS // 2))
        ref_items.append({"type": "video", "data": frames[sample_idx],
                          "timestamps": [i / 2.0 for i in range(len(sample_idx))]})
        ref_blocks.append({"kind": "video_audio" if ref_audio_t else "video",
                           "latent_t": z.shape[2], "latent_h": ch // 16, "latent_w": cw // 16,
                           "ref_audio_t": ref_audio_t, "latent": z, "audio_latent": audio_latent})

    for name, audio in (ref_audios or {}).items():
        if audio is None:
            continue
        audio = _to_stereo(audio, "reference audio %s" % name)
        audio_latent, ref_audio_t = _encode_ref_audio(audio_vae, audio)
        # Not trimmed: a reference is chosen, not derived, so its length is the
        # user's call. It is still worth saying what it costs -- reference tokens
        # are attended at EVERY step.
        if ref_audio_t > 40 * 30:
            logging.warning(
                "[MMH3ReferenceMultiPrompt] reference audio %s is %.0fs (%d latents). "
                "References are attended at every step, so long ones are paid for on "
                "every chunk of every pass.", name, ref_audio_t / 40.0, ref_audio_t)
        ref_items.append({"type": "audio"})
        ref_blocks.append({"kind": "audio", "ref_audio_t": ref_audio_t,
                           "audio_latent": audio_latent})

    return ref_items, ref_blocks


AUDIO_LATENT_HZ = 40


def _to_stereo(audio, label):
    """H3's audio VAE expects STEREO. Mono is accepted here and quietly wrong.

    Core's `_encode_ref_audio` hands the waveform straight to the VAE, so a [B,1,L]
    track encodes without complaint and the model gets something it was not trained
    on; sglang refuses the same input outright. Reported by fredbliss 2026-08-22.

    One channel is duplicated rather than resampled -- the content is identical on
    both sides, which is what a mono source means. More than two is downmixed to a
    stereo pair rather than truncated, so a 5.1 upload does not silently lose its
    centre channel.
    """
    wf = audio.get("waveform")
    if wf is None or wf.ndim != 3:
        return audio
    ch = int(wf.shape[1])
    if ch == 2:
        return audio
    if ch == 1:
        wf = wf.repeat(1, 2, 1)
        logging.warning(
            "[MMH3ReferenceMultiPrompt] %s is MONO; duplicated to stereo. H3's audio "
            "VAE expects two channels and encodes one without complaining, so this "
            "would have been wrong rather than loud.", label)
    else:
        # Every channel into both sides. An even/odd split looks more like a real
        # downmix and is worse: in WAV order channel 2 is CENTRE, so it would land
        # in the left channel alone and dialogue would sit off to one side. A proper
        # 5.1 fold needs per-layout coefficients, which is not worth carrying for an
        # input this rare -- collapse it, and say so loudly enough to fix upstream.
        mono = wf.mean(dim=1, keepdim=True)
        wf = mono.repeat(1, 2, 1)
        logging.warning(
            "[MMH3ReferenceMultiPrompt] %s has %d channels; every channel was summed "
            "into both sides, so the stereo image is GONE. Downmix it yourself if the "
            "placement matters -- H3's VAE takes stereo only.", label, ch)
    out = dict(audio)
    out["waveform"] = wf.contiguous()
    return out


def _trim_audio(audio, want_latents, label):
    """Cut the waveform to what the clip needs BEFORE encoding it.

    `_use_input_audio` used to encode the whole track and drop the latents past the
    end. The result was identical and the work was not: a five-minute track for a
    sixty-second render is five minutes of VAE encode to keep one. A small margin is
    left on so the trim never decides the final length -- the latent-side cut still
    does, and it is the one that lands on the grid.
    """
    wf = audio.get("waveform")
    if wf is None or wf.ndim != 3 or want_latents <= 0:
        return audio
    sr = int(audio.get("sample_rate") or 0)
    if sr <= 0:
        return audio
    keep = int((want_latents + 8) / float(AUDIO_LATENT_HZ) * sr)   # +8 latents of margin
    have = int(wf.shape[-1])
    if have <= keep:
        return audio
    out = dict(audio)
    out["waveform"] = wf[..., :keep].contiguous()
    logging.info("[MMH3ReferenceMultiPrompt] %s trimmed %.1fs -> %.1fs before encoding "
                 "(the clip holds %.1fs)", label, have / sr, keep / sr,
                 want_latents / float(AUDIO_LATENT_HZ))
    return out


def _use_input_audio(latent, audio_vae, audio):
    """Swap the empty audio half for a real track, masked so it is left alone.

    The half is sized round(frames / 24 * 40) and an encode will not land on that
    exactly, so it is cut or padded. Padding is silence at the END -- looping a
    short track would put a seam somewhere no prompt describes.
    """
    video, empty = latent["samples"].unbind()
    want = int(empty.shape[-1])
    # stereo first, then trim to the clip: both BEFORE the encode, which is the
    # whole point -- the old order encoded minutes of audio to keep seconds.
    audio = _trim_audio(_to_stereo(audio, "input audio"), want, "input audio")
    z, have = _encode_ref_audio(audio_vae, audio)
    z = z.to(dtype=empty.dtype, device=empty.device)

    if have > want:
        z = z[..., :want]
    elif have < want:
        z = torch.cat([z, torch.zeros(list(z.shape[:-1]) + [want - have],
                                      dtype=z.dtype, device=z.device)], dim=-1)
    logging.info("[MMH3ReferenceMultiPrompt] input audio: %d latents for %d (%+.2fs)",
                 have, want, (have - want) / 40.0)

    latent["samples"] = NestedTensor((video, z.contiguous()))
    # 1 generates, 0 preserves: video free, audio pinned
    latent["noise_mask"] = NestedTensor((
        torch.ones([video.shape[0], 1] + list(video.shape[2:]), dtype=torch.float32),
        torch.zeros([z.shape[0], 1, z.shape[2], z.shape[3]], dtype=torch.float32)))


class MMH3ReferenceMultiPrompt(io.ComfyNode):
    """One reference set, N prompts, one model swap."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3ReferenceMultiPrompt",
            display_name="MiniMax H3 Reference (Multi-Prompt)",
            category="MMH3Tools/conditioning",
            description=(
                "MiniMaxH3ReferenceToVideo with N prompts. References are resized and "
                "encoded ONCE, and every text encode happens in one node execution, so "
                "the text encoder and the DiT each load once for the whole sequence "
                "instead of once per chunk. Feed the output to MMH3 Cond Select."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.Int.Input("width", default=1344, min=32, max=16384, step=32),
                io.Int.Input("height", default=768, min=32, max=16384, step=32),
                io.Int.Input("length", default=192, min=5, max=3600, step=17,
                             tooltip="Frames at 24 fps for the latent this node emits, "
                                     "shared by every prompt. Snapped to the 17j+5 grid.\n\n"
                                     "MMH3 Looping Sampler and MMH3 Context Windows both "
                                     "take this latent as the WHOLE clip and slice it, so "
                                     "wire the total length here. Chunk and window size are "
                                     "set on those nodes.\n\n"
                                     "192 is the only whole-second duration in the trained "
                                     "range."),
                io.Combo.Input(
                    "ref_image_size", options=["match", "max"], default="match",
                    tooltip="'match' scales each reference to the generation's pixel area; "
                            "'max' uses a 2048px short edge for best identity fidelity. "
                            "Reference tokens ride through every sampling step of every "
                            "chunk, so 'max' is paid N times over.",
                ),
                io.String.Input(
                    "prompts", multiline=True, dynamic_prompts=True,
                    tooltip="Every prompt in one string, PIPE separated, in chunk order. "
                            "A loop that accumulates one prompt per window wires straight "
                            "in here -- no socket per chunk, so the graph is the same size "
                            "whatever N is.\n\n"
                            "Keep subject_definitions and retention_analysis byte-identical "
                            "across all of them; only detailed_description should vary, or "
                            "the character drifts. A literal | inside a prompt WILL "
                            "over-split silently -- watch the `count` output, which is "
                            "how many prompts this actually found.",
                ),
                io.Image.Input(
                    "ref_images", optional=True,
                    tooltip="Reference stills -- each becomes its own <Picture i>, "
                            "numbered in the order given. One image is the ordinary "
                            "case.\n\n"
                            "A BATCH forces every image to share one frame: a tensor "
                            "cannot be ragged, so core's ImageBatch and KJNodes' "
                            "ImageBatchMulti both resize and CENTRE-CROP everything to "
                            "the first image before this node sees it. If your "
                            "references are different shapes, that crop already "
                            "happened and nothing here can undo it.\n\n"
                            "Wire a LIST instead to keep each reference at its native "
                            "size, each getting its own aspect-correct target -- "
                            "KJNodes' ImageTensorList emits one and chains to any "
                            "depth.",
                ),
                io.Autogrow.Input(
                    "ref_videos", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Image.Input("ref_video"), prefix="ref_video_", min=0, max=3)),
                io.Autogrow.Input(
                    "ref_video_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_video_audio"),
                        prefix="ref_video_audio_", min=0, max=3)),
                io.Autogrow.Input(
                    "ref_audios", optional=True,
                    template=io.Autogrow.TemplatePrefix(
                        input=io.Audio.Input("ref_audio"), prefix="ref_audio_", min=0, max=3)),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip="The TARGET's soundtrack, encoded here and written into the "
                            "latent's audio half in place of silence, masked so the "
                            "sampler leaves it alone. A ref_audio is a voice to imitate; "
                            "this IS the audio."),
                io.Boolean.Input(
                    "use_input_audio", default=False,
                    tooltip="Off leaves the audio half empty and the model generates it."),
                io.Boolean.Input(
                    "unload_text_encoder", default=True,
                    tooltip="Evict the text encoder from VRAM once every prompt is "
                            "encoded. Unloads THIS clip's patcher and its clones only, "
                            "not every model, so the VAEs stay resident.\n\n"
                            "H3's text encoder is large and this node is the last thing "
                            "that needs it. Left loaded, it occupies room the diffusion "
                            "model then cannot get, and the sampler falls back to system "
                            "RAM.\n\n"
                            "The cost is a reload the next time any node needs the "
                            "encoder, including a re-run with one prompt edited."),
                io.Boolean.Input(
                    "window_ref_video", default=False, optional=True,
                    tooltip="Cut the REFERENCE VIDEO (and its soundtrack) to each "
                            "chunk's own span, so chunk i is conditioned on the footage "
                            "it is actually rendering instead of the whole reference "
                            "every time.\n\n"
                            "Costs one text-encode PER CHUNK rather than one for the "
                            "sequence -- still a single text-encoder load, so it is N "
                            "forward passes, not N model swaps. The vision work is "
                            "partitioned rather than duplicated: each chunk encodes "
                            "1/N of the frames. At sampling time it is cheaper, since "
                            "reference tokens are attended at EVERY step and each chunk "
                            "now carries only its window.\n\n"
                            "Needs `chunk_frames` and `overlap_frames` wired -- the "
                            "SAME values the sampler gets, or the windows stop matching "
                            "the chunks. Off, nothing changes."),
                io.Int.Input(
                    "chunk_frames", default=0, min=0, max=100000, step=1, optional=True,
                    tooltip="Only read when `window_ref_video` is on. Wire the sampler's "
                            "own `chunk_frames` -- from MMH3 Chunk Schedule, not typed "
                            "again."),
                io.Int.Input(
                    "overlap_frames", default=0, min=0, max=100000, step=1, optional=True,
                    tooltip="Only read when `window_ref_video` is on. Wire the sampler's "
                            "own `overlap_frames`."),
                io.String.Input(
                    "embeddings", default="", multiline=True, optional=True,
                    tooltip="H3 text embeddings to PREPEND to every chunk's prompt, one "
                            "per line, by filename from `models/embeddings/` (the "
                            "extension is optional).\n\n"
                            "    minimaxh3_bullet_time\n"
                            "    minimaxh3_storm_magic: 4-6\n"
                            "    minimaxh3_four_seasons: all\n\n"
                            "A bare name goes on EVERY chunk. `N` or `A-B` (1-based) "
                            "schedules it, which works because each chunk has its own "
                            "prompt anyway. Several lines stack: their cost is exactly "
                            "additive, and so is their effect.\n\n"
                            "They are NOT free -- 50 to 142 token slots each, attended "
                            "at every sampling step of every chunk. The report prints "
                            "the per-chunk total.\n\n"
                            "Needs a core that parses `embedding:` in H3 prompts "
                            "(#15808, merged 2026-08-22). On an older one the marker "
                            "tokenizes as ordinary words and nothing is spliced, so "
                            "this refuses rather than pretending."),
            ],
            outputs=[
                MMH3CondSet.Output(display_name="cond_set"),
                io.Latent.Output(display_name="latent"),
                io.Int.Output(display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, audio_vae, width, height, length, ref_image_size,
                prompts=None, ref_images=None, ref_videos=None, ref_video_audios=None,
                ref_audios=None, audio=None, use_input_audio=False,
                window_ref_video=False, chunk_frames=0, overlap_frames=0,
                embeddings="", unload_text_encoder=True) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)
        if use_input_audio:
            if audio is None:
                raise ValueError("MMH3ReferenceMultiPrompt: use_input_audio is on but "
                                 "no audio is wired.")
            _use_input_audio(latent, audio_vae, audio)

        # Pipe separated, in chunk order. Empty pieces are dropped rather than
        # encoded, so a trailing | or a blank line between prompts costs nothing.
        texts = [p.strip() for p in (prompts or "").split("|") if p.strip()]
        if not texts:
            raise ValueError(
                "MMH3ReferenceMultiPrompt needs at least one prompt. Prompts go in one "
                "string separated by | , in chunk order.")

        # Embeddings are prepended to the prompts BEFORE anything else looks at
        # them, so the cache key, the fingerprint and the encode all see the real
        # string rather than one that grows later.
        if (embeddings or "").strip():
            per_chunk = _parse_embeddings(embeddings, len(texts))
            wanted = sorted({n for row in per_chunk for n in row})
            if not wanted:
                raise ValueError(
                    "MMH3ReferenceMultiPrompt: `embeddings` has no usable names. One "
                    "filename per line, optionally `name: all|N|A-B`.")
            known = _known_embeddings()
            if known:
                missing = [n for n in wanted
                           if n not in known
                           and not any(k.startswith(n + ".") for k in known)]
                if missing:
                    raise ValueError(
                        "MMH3ReferenceMultiPrompt: no such embedding(s) in "
                        "models/embeddings: %s.\nA name that does not resolve is "
                        "dropped with only a log line, so this stops instead."
                        % ", ".join(missing))
            if not _embeddings_resolve(clip, wanted[0]):
                raise ValueError(
                    "MMH3ReferenceMultiPrompt: this ComfyUI does not splice "
                    "`embedding:` into H3 prompts -- the marker tokenizes as ordinary "
                    "words and nothing is added. Needs #15808 (merged 2026-08-22). "
                    "Update, or clear `embeddings`.")
            costs = {n: _embedding_rows(n) for n in wanted}
            texts = [(" ".join("embedding:%s" % n for n in per_chunk[i]) + " " + t).strip()
                     if per_chunk[i] else t
                     for i, t in enumerate(texts)]
            spread = ["chunk %d: %s (%s slots)"
                      % (i + 1, ", ".join(row) or "none",
                         sum(costs.get(n) or 0 for n in row))
                      for i, row in enumerate(per_chunk)]
            logging.info("[MMH3ReferenceMultiPrompt] embeddings prepended -- %s",
                         "; ".join(spread[:4]) + (" ..." if len(spread) > 4 else ""))

        raw_inputs = [ref_images]
        for group in (ref_videos, ref_video_audios, ref_audios):
            raw_inputs.extend((group or {}).values())

        # Per-chunk reference windows, or one reference set for every prompt.
        windows = None
        if window_ref_video and (ref_videos or {}):
            if int(chunk_frames) <= 0:
                raise ValueError(
                    "MMH3ReferenceMultiPrompt: `window_ref_video` is on but "
                    "`chunk_frames` is 0. Wire the sampler's own chunk_frames and "
                    "overlap_frames, or the reference windows will not be the spans "
                    "the sampler renders.")
            windows = _ref_windows(frame_count, int(chunk_frames), int(overlap_frames))
            if len(windows) != len(texts):
                logging.warning(
                    "[MMH3ReferenceMultiPrompt] %d prompt(s) against %d reference "
                    "window(s). Prompt i is paired with window i and the shorter list "
                    "decides; check that chunk_frames/overlap_frames match the sampler.",
                    len(texts), len(windows))
            logging.info("[MMH3ReferenceMultiPrompt] windowing the reference video into "
                         "%d span(s): %s", len(windows),
                         ", ".join("%d-%d" % w for w in windows[:6])
                         + (" ..." if len(windows) > 6 else ""))

        shared = None
        if windows is None:
            shared = _build_refs(
                vae, audio_vae, width, height, frame_count, ref_image_size,
                ref_images, ref_videos, ref_video_audios, ref_audios)
            ref_items, ref_blocks = shared
            fp = _fingerprint(ref_blocks, raw_inputs, width, height, length,
                              ref_image_size)

        conds = []
        hits = 0
        for pi, text in enumerate(texts):
            if windows is not None:
                span = windows[min(pi, len(windows) - 1)]
                ref_items, ref_blocks = _build_refs(
                    vae, audio_vae, width, height, frame_count, ref_image_size,
                    ref_images, ref_videos, ref_video_audios, ref_audios,
                    ref_window=span)
                fp = _fingerprint(ref_blocks, raw_inputs + [span], width, height,
                                  length, ref_image_size)
            key = (text, fp)
            cached = _CACHE.get(key)
            if cached is not None:
                conds.append(cached)
                hits += 1
                continue
            tokens = clip.tokenize(text, minimax_ref_items=ref_items)
            cond = clip.encode_from_tokens_scheduled(tokens)
            if ref_blocks:
                cond = node_helpers.conditioning_set_values(cond, {"minimax_refs": ref_blocks})
            if len(_CACHE) >= _CACHE_MAX:
                _CACHE.pop(next(iter(_CACHE)))
            _CACHE[key] = cond
            conds.append(cond)

        # Every prompt is encoded by now, so the encoder has no further use in this run
        if unload_text_encoder:
            evict_text_encoder(clip, "MMH3ReferenceMultiPrompt")

        logging.info("[MMH3ReferenceMultiPrompt] %d prompts, %d refs, %d frames "
                     "(%d encodes reused)", len(conds), len(ref_blocks), frame_count, hits)
        return io.NodeOutput({"conds": conds, "prompts": texts, "fingerprint": fp},
                             latent, len(conds))


class MMH3CondSelect(io.ComfyNode):
    """Pull one chunk's conditioning out of a cond_set."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondSelect",
            display_name="MMH3 Cond Select",
            category="MMH3Tools/conditioning",
            description="Select one prompt's conditioning from a MiniMax H3 cond_set.",
            inputs=[
                MMH3CondSet.Input("cond_set"),
                io.Int.Input("index", default=0, min=0, max=31, step=1,
                             tooltip="0-based. Out of range is an error rather than a wrap, "
                                     "because silently rendering the wrong chunk is worse "
                                     "than a stopped queue."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.String.Output(display_name="prompt"),
            ],
        )

    @classmethod
    def execute(cls, cond_set, index) -> io.NodeOutput:
        conds = cond_set["conds"]
        i = int(index)
        if i >= len(conds):
            raise ValueError(
                "index %d is out of range: the cond_set holds %d prompt%s (0-%d)."
                % (i, len(conds), "" if len(conds) == 1 else "s", len(conds) - 1))
        # `prompts` is the display half of the contract and not every producer can
        # fill it -- MMH3Regenerate2KReference builds conds from encoded conditioning
        # and may never see the text. Guard rather than index: the conditioning is
        # what callers actually wire, and losing a label is not worth an exception.
        texts = cond_set.get("prompts") or []
        return io.NodeOutput(conds[i], texts[i] if i < len(texts) else "")


class MMH3CondToSet(io.ComfyNode):
    """Wrap a plain CONDITIONING as a cond_set.

    The inverse of MMH3CondSelect. The looping sampler REQUIRES a cond_set and
    ignores the guider's conditioning, and every producer of one goes through the
    text encoder -- so a refine pass whose conditioning is a zero-out (no prompt,
    no CLIP anywhere in the graph) had no way to reach the sampler without loading
    a 20 GB encoder to tokenize an empty string. This is that missing edge.

    `prompts` is empty strings -- the display half of the contract, unfillable
    here for the same reason as MMH3Regenerate2KReference: the text was never
    seen. `fingerprint` is None; nothing reads it.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondToSet",
            display_name="MMH3 Cond To Set",
            category="MMH3Tools/conditioning",
            description=(
                "Wrap an already-encoded CONDITIONING as a cond_set for the looping "
                "sampler. No text encoder involved. Every chunk receives this same "
                "conditioning; count only changes how many entries formally exist."
            ),
            inputs=[
                io.Conditioning.Input("conditioning"),
                io.Int.Input(
                    "count", default=1, min=1, max=32, step=1,
                    tooltip="How many entries the set holds, all of them this same "
                            "conditioning. The looping sampler reuses the last entry "
                            "for chunks past the end, so 1 already covers any chunk "
                            "count.",
                ),
            ],
            outputs=[MMH3CondSet.Output(display_name="cond_set")],
        )

    @classmethod
    def execute(cls, conditioning, count) -> io.NodeOutput:
        n = int(count)
        return io.NodeOutput({"conds": [conditioning] * n,
                              "prompts": [""] * n,
                              "fingerprint": None})


def _strip_text_from_cond(cond, mode):
    """One CONDITIONING -> (stripped, kept_rows, total_rows). Media survives whole.

    Text and reference media live in DIFFERENT halves of a conditioning entry, which
    is what makes this possible at all: the prompt is the TENSOR `t[0]`, while
    `minimax_refs` / `minimax_keyframes` are keys in the DICT `t[1]`. Copying the
    dict and replacing only the tensor keeps every reference exactly as it was --
    the same thing core's ConditioningZeroOut does, applied per entry.

    `minimax_token_tags` marks each position: 1 = text, 0 = a vision block (the
    flanking <|vision_start|>/<|vision_end|> included). 'vision only' keeps the
    zeros, and MUST slice the tags in lockstep with the embedding -- a tag vector
    that no longer lines up with its tokens is worse than leaving the text alone,
    because the DiT reads modality from it.
    """
    out = []
    kept = total = 0
    for t in cond:
        emb, d = t[0], t[1].copy()
        n = int(emb.shape[1]) if emb.ndim >= 2 else 0
        total += n
        tags = d.get("minimax_token_tags", None)
        if mode == "vision only" and tags is not None and n:
            keep = (tags.to(emb.device) == 0)
            if int(keep.shape[0]) == n:
                emb = emb[:, keep]
                d["minimax_token_tags"] = tags[keep.to(tags.device)]
            else:
                # tags and embedding disagree -- do not guess which is right
                logging.warning("[MMH3CondSetStripText] token tags are %d long against "
                                "%d embedding rows; zeroing instead of slicing",
                                int(keep.shape[0]), n)
                emb = torch.zeros_like(emb)
        elif mode == "vision only":
            # nothing marks the vision blocks: references appended after encoding
            # (MMH3ImageToRef / MMH3LatentToRef / MMH3Regenerate2KReference) never
            # register with the tokenizer, so there is no text-side copy to keep
            emb = emb[:, :0]
            if tags is not None:
                d["minimax_token_tags"] = tags[:0]
        else:
            emb = torch.zeros_like(emb)
        kept += int(emb.shape[1]) if emb.ndim >= 2 else 0
        if d.get("pooled_output", None) is not None:
            d["pooled_output"] = torch.zeros_like(d["pooled_output"])
        out.append([emb, d])
    return out, kept, total


class MMH3CondSetStripText(io.ComfyNode):
    """Drop the prompt from every entry of a cond_set, keeping the reference media.

    FOR REFINE PASSES WITH SMALL WINDOWS. A chunk-level prompt describes a whole
    chunk, but a refine pass windows that chunk into pieces and core picks each
    window's region from its MIDPOINT -- so a window covering a fraction of the
    timeline is handed text describing all of it, and asked to render the whole
    script into its slice. At low denoise that is pure confounding: nothing is
    being invented, the content is already in the latent, and the only thing worth
    conditioning on is identity. Stripping the text leaves exactly that.

    Two halves, two fates. The prompt is the conditioning's TENSOR; the references
    are keys in its DICT. This rewrites the first and copies the second, so
    `minimax_refs` and `minimax_keyframes` arrive at the sampler untouched.

    'zero' keeps the text span's LENGTH and zeroes its values, which is what
    ConditioningZeroOut does and is always valid. 'vision only' instead keeps just
    the vision-block positions and drops the prose, shrinking text_len -- which
    also pulls the target closer, since references lay out from a cursor starting
    at text_len.

    ⚠ 'vision only' on conditioning whose references were appended AFTER encoding
    leaves the text span EMPTY (text_len 0). PackedLayout accepts that -- the whole
    sequence just shifts down -- but no encoder can produce it, so it is untested
    against real weights. The node reports it rather than preventing it.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondSetStripText",
            display_name="MMH3 Cond Set Strip Text",
            category="MMH3Tools/conditioning",
            description=(
                "Remove the prompt from every entry of a cond_set while keeping the "
                "reference media. For refine passes whose windows are smaller than the "
                "chunk the prompt was written for, where the text confounds rather "
                "than helps."
            ),
            inputs=[
                MMH3CondSet.Input("cond_set"),
                io.Combo.Input(
                    "mode", options=["zero", "vision only"], default="zero",
                    tooltip="'zero' blanks the text values and keeps its length -- "
                            "always valid, and what references appended after encoding "
                            "will get anyway. 'vision only' keeps just the image tokens "
                            "and drops the prose, shortening text_len; it needs "
                            "conditioning built through the text encoder, and leaves an "
                            "EMPTY text span if there is none.",
                ),
            ],
            outputs=[
                MMH3CondSet.Output(display_name="cond_set"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, cond_set, mode) -> io.NodeOutput:
        conds = list(cond_set.get("conds", []))
        if not conds:
            raise ValueError(
                "MMH3CondSetStripText: the cond_set is empty. It should carry one "
                "conditioning per chunk.")

        stripped, lines, empties = [], [], 0
        for i, c in enumerate(conds):
            s, kept, total = _strip_text_from_cond(c, mode)
            stripped.append(s)
            if kept == 0 and mode == "vision only":
                empties += 1
            lines.append("  entry %d: %d -> %d text rows" % (i, total, kept))

        report = "%d entr%s, mode '%s'\n%s" % (
            len(conds), "y" if len(conds) == 1 else "ies", mode, "\n".join(lines))
        if empties:
            report += ("\n  ! %d entr%s left with an EMPTY text span (text_len 0). The "
                       "layout accepts it, but no encoder produces it -- untested "
                       "against real weights. Use 'zero' if this misbehaves."
                       % (empties, "y" if empties == 1 else "ies"))
            logging.warning("[MMH3CondSetStripText] %d entry/entries have text_len 0; "
                            "references were appended after encoding, so there were no "
                            "vision tokens to keep", empties)
        logging.info("[MMH3CondSetStripText] %s", report.splitlines()[0])

        # prompts are blanked: they would otherwise print in the sampler's report
        # describing text this conditioning no longer carries
        return io.NodeOutput({"conds": stripped,
                              "prompts": [""] * len(stripped),
                              "fingerprint": None}, report)


class MMH3CondSetSpread(io.ComfyNode):
    """Flatten a cond_set into ONE conditioning holding every prompt, in order.

    This is the input shape `split_conds_to_windows` wants. Core decides which prompt
    a window uses from the window's own midpoint:

        center_ratio = (min(index_list) + max(index_list)) / (2 * total_frames)
        region       = int(center_ratio * len(cond_in))

    so entry 0 covers the start of the timeline and entry N-1 the end. Without this,
    every window sees the same single conditioning and the model is asked to render
    the whole script into each one -- which is what "it looks like it's doing the
    entire conditioning per window" was.

    MMH3CondSelect takes ONE prompt for ONE chunk; this takes all of them for one
    windowed pass. The references are shared either way, because the cond_set encoded
    them once, so identity does not shift as the region changes -- only the prompt does.
    """

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3CondSetSpread",
            display_name="MMH3 Cond Set Spread",
            category="MMH3Tools/conditioning",
            description=(
                "Flatten a cond_set into a single conditioning containing every prompt "
                "in order, for MMH3 Context Windows with split_conds_to_windows on. "
                "Each window then uses the prompt for its own region of the timeline."
            ),
            inputs=[
                MMH3CondSet.Input("cond_set"),
            ],
            outputs=[
                io.Conditioning.Output(display_name="conditioning"),
                io.Int.Output(display_name="regions"),
                io.String.Output(display_name="report"),
            ],
        )

    @classmethod
    def execute(cls, cond_set) -> io.NodeOutput:
        conds = cond_set["conds"]
        prompts = cond_set.get("prompts") or []

        # each cond_set entry is a full CONDITIONING (a list); the region split works on
        # the ENTRIES of one conditioning, so concatenate rather than nest
        flat = []
        for c in conds:
            flat.extend(c)

        if len(flat) != len(conds):
            logging.info("[MMH3CondSetSpread] %d prompts expanded to %d entries; regions "
                         "are per ENTRY, so they will not line up with prompts",
                         len(conds), len(flat))

        # A keyframe re-projects into EVERY window. The layout is rebuilt per window
        # from the window's own latent_t, and a first-frame anchor is placed at the
        # target origin -- which is that window's frame 0, not the clip's. An i2v start
        # image would therefore be re-imposed at every window boundary. A last-frame
        # anchor is worse: minimax_frame_count is not patched per window, so the index
        # check still matches the clip while the POSITION comes from the window.
        # Entry 0 is the exception, not an offender: region 0 IS the first window, so a
        # start frame anchored there lands where it belongs. Anywhere else is a repeat.
        with_kf = [i for i, e in enumerate(flat) if e[1].get("minimax_keyframes")]
        misplaced = [i for i in with_kf if i > 0]
        kf_note = ""
        if misplaced and len(flat) > 1:
            kf_note = ("\n  ! entr%s %s carr%s keyframes. Under split_conds_to_windows a "
                       "keyframe is re-anchored to ITS OWN window's start or end, not the "
                       "clip's, so this repeats at every window boundary. Keep keyframes on "
                       "entry 0 only -- region 0 is the first window, the one place a start "
                       "frame belongs."
                       % ("y" if len(misplaced) == 1 else "ies",
                          ", ".join(str(i) for i in misplaced),
                          "ies" if len(misplaced) == 1 else "y"))
        elif with_kf == [0] and len(flat) > 1:
            kf_note = "\n  keyframe on entry 0 only -- anchored to the first window, correct"

        lines = []
        for i, text in enumerate(prompts[:len(flat)]):
            lo, hi = i / len(flat), (i + 1) / len(flat)
            first = (text or "").strip().splitlines()
            lines.append("  %d  %.0f%%-%.0f%%  %s"
                         % (i, lo * 100, hi * 100, (first[0][:60] if first else "(empty)")))
        report = "%d region%s across the clip:\n%s" % (
            len(flat), "" if len(flat) == 1 else "s", "\n".join(lines))
        if len(flat) == 1:
            report += ("\n  ! one prompt means split_conds_to_windows does nothing -- core "
                       "only splits when a conditioning holds more than one entry")
        report += kf_note
        logging.info("[MMH3CondSetSpread] " + report)
        return io.NodeOutput(flat, len(flat), report)

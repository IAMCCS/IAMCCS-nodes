"""Streaming export: decode in chunks straight into ffmpeg, never hold the video.

A full decode allocates the whole pixel tensor at once, and that is the wall you hit
before any other one:

    1344x768, 362f     4.48 GB
    2048x1152, 362f   10.25 GB
    2048x1152, 750f   21.23 GB

against 0.48 GB for a single 17-frame clip at 2K.

WHY THIS IS NOT THE LTX PATTERN
-------------------------------
LTXAVStreamingSave decodes a slice with LEFT context and trims it, because the LTX
video VAE is causal -- past context only. H3's decoder is neither causal nor
independent:

    t_end_idx = t_start_idx + tokens_chunk_size + token_overlap      # 5 + 2
    clip_dec_chunk = self.blend(dec_overlap, clip_dec_chunk, self.frame_overlap)

Each chunk reads TWO latents of LOOKAHEAD, and carries `dec_overlap` forward to blend
5 frames into the next chunk's output. So a slice decoded on its own is wrong at BOTH
ends: its first written part skips a blend it should have had, and its last chunk pads
instead of reading real lookahead.

(Note this is the opposite of MMH3StreamingEncode, where clips encode independently
and chunking is bit-identical for free. Encode and decode are not symmetric here.)

THE SCHEME
----------
Work in groups of `tokens_chunk_size` (5) latents; each group is exactly 17 output
frames, which is the 5j+2 <-> 17j+5 grid seen from the VAE's side.

To emit groups g0..g1-1 exactly, decode latents

    [ 5*g0 - 5 : 5*g1 + 2 ]        (the -5 omitted for g0 == 0)

The leading group is decoded only so its j=1 part becomes the `dec_overlap` that
blends correctly into group g0 -- its own output is discarded. The +2 supplies the
lookahead. Then keep 17 frames per emitted group, starting at frame 17 when there is
a context group.

AND DROP THE LAST 5 FRAMES, except on the final batch. A partial decode always ends
by writing its last chunk's j=1 part raw; in a full decode that part is never written
raw, it is blended into the chunk after it. Keeping them would put a visibly
unblended seam at every batch boundary.

Audio is not decoded here -- it is tiny. Decode it normally and pass it in to be muxed.

MMH3SizeCappedCopy
------------------
Also here: a two-pass transcode of an already-written file to a hard size ceiling.
It shares this module only for the ffmpeg discovery above; it never touches a VAE.
CRF cannot have a size ceiling by construction, so a delivery copy under an upload
limit is a separate encode rather than a setting on the first one.
"""

import json
import logging
import math
import os
import re
import shutil
import subprocess
import tempfile

import torch

import comfy.model_management
import folder_paths
from comfy_api.latest import io

from .common import unpack_av

# Fallbacks if the VAE ever stops exposing its own constants. Verified against
# comfy/ldm/minimax/vae.py: clip_length 17, token_drop 3, vae_ratio_t 4.
DEFAULTS = {"clip_length": 17, "token_drop": 3, "vae_ratio_t": 4}

# frames the FINAL decode chunk contributes beyond its groups: the last chunk's
# carried part, written raw only because there is no chunk after it to blend into.
# frame_overlap = token_overlap*vae_ratio_t - frame_pre_padding = 8 - 3 = 5.
TAIL_FRAMES = 5


def write_ffmetadata(path, tags):
    """An ffmetadata1 file for `-f ffmetadata -i path`.

    A workflow JSON runs 45-95 KB. Windows caps a command line near 32,767
    characters, so `-metadata workflow=...` cannot carry one -- the file form is
    not a tidiness choice, it is the only form that fits.

    ffmpeg's escapes are = ; # \\ and newline; everything else is literal, and
    the file is UTF-8 so emoji in node titles survive.
    """
    out = [";FFMETADATA1"]
    for key, value in tags.items():
        value = (value.replace("\\", "\\\\").replace("=", "\\=")
                      .replace(";", "\\;").replace("#", "\\#")
                      .replace("\n", "\\\n"))
        out.append("%s=%s" % (key, value))
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(out) + "\n")


def collect_metadata(prompt, extra_pnginfo):
    """ComfyUI's saved-workflow tags, or {} when metadata is disabled."""
    try:
        from comfy.cli_args import args
        if args.disable_metadata:
            return {}
    except Exception:
        pass
    tags = {}
    if prompt is not None:
        tags["prompt"] = json.dumps(prompt)
    if extra_pnginfo is not None:
        for k, v in extra_pnginfo.items():
            tags[k] = json.dumps(v)      # "workflow" is the one that drag-drops
    return tags


def vae_grid(vae):
    """(tokens_chunk_size, token_overlap, frames_per_group) read from the VAE itself.

    Read rather than hardcoded, so a checkpoint with different constants is followed
    instead of silently mis-sliced. The caller self-checks the result against the
    17j+5 grid before trusting it.
    """
    inner = getattr(vae, "first_stage_model", None)
    def get(name):
        v = getattr(inner, name, None)
        return DEFAULTS[name] if v is None else int(v)

    clip_length, token_drop, ratio_t = get("clip_length"), get("token_drop"), get("vae_ratio_t")
    tokens_chunk = int(getattr(inner, "tokens_chunk_size", 0) or math.ceil(clip_length / ratio_t))
    overlap = getattr(inner, "token_overlap", None)
    overlap = int((-token_drop) % tokens_chunk if overlap is None else overlap)
    return tokens_chunk, overlap, clip_length


class MMH3StreamingSave(io.ComfyNode):
    """Decode an H3 video latent in chunks and stream it straight into ffmpeg."""

    _ENCODERS = ["libx264", "h264_nvenc", "libopenh264", "mpeg4"]
    _probe_cache = {}

    @staticmethod
    def _encoder_args(name, crf, w, h):
        # -crf and -preset are x264-specific; every encoder needs its own mapping
        if name == "libx264":
            return ["-c:v", "libx264", "-preset", "medium",
                    "-crf", str(crf), "-pix_fmt", "yuv420p"]
        if name == "h264_nvenc":
            return ["-c:v", "h264_nvenc", "-preset", "p5", "-rc", "vbr",
                    "-cq", str(crf), "-pix_fmt", "yuv420p"]
        if name == "libopenh264":
            mbps = max(1.0, (w * h / (1280 * 720)) * (2.0 ** ((28 - crf) / 6.0)))
            return ["-c:v", "libopenh264", "-b:v", "%.1fM" % mbps, "-pix_fmt", "yuv420p"]
        return ["-c:v", "mpeg4", "-q:v", str(max(1, min(31, crf // 2))), "-pix_fmt", "yuv420p"]

    @classmethod
    def _available(cls, ffmpeg):
        if ffmpeg in cls._probe_cache:
            return cls._probe_cache[ffmpeg]
        found = set()
        try:
            out = subprocess.run([ffmpeg, "-hide_banner", "-encoders"],
                                 capture_output=True, timeout=30)
            text = (out.stdout or b"").decode("utf-8", "replace")
            for enc in cls._ENCODERS:
                if any(line.split()[1:2] == [enc] for line in text.splitlines() if line.strip()):
                    found.add(enc)
        except Exception as e:
            # Report NO encoders rather than optimistically assuming this binary is
            # fine -- otherwise a stale PATH entry shadows a working build.
            logging.warning("[MMH3StreamingSave] ffmpeg candidate unusable, skipping: %s (%s)",
                            ffmpeg, e)
        cls._probe_cache[ffmpeg] = found
        return found

    @classmethod
    def _resolve_ffmpeg(cls, explicit, want):
        """Prefer a binary that HAS a usable encoder over merely the first found.

        A conda ffmpeg built without libx264 sits on PATH and shadows a working
        imageio-ffmpeg build, so 'first on PATH' picks the broken one.
        """
        import shutil
        candidates = []
        if explicit and explicit.strip():
            candidates.append(explicit.strip().strip('"'))
        on_path = shutil.which("ffmpeg")
        if on_path:
            candidates.append(on_path)
        try:
            from imageio_ffmpeg import get_ffmpeg_exe
            candidates.append(get_ffmpeg_exe())
        except Exception:
            pass

        seen, ordered = set(), []
        for c in candidates:
            if c and c not in seen:
                seen.add(c)
                ordered.append(c)
        if not ordered:
            raise RuntimeError("[MMH3StreamingSave] no ffmpeg found (not on PATH, "
                               "imageio-ffmpeg unavailable, no ffmpeg_path given).")

        wanted = [want] if want != "auto" else cls._ENCODERS
        for binary in ordered:
            have = cls._available(binary)
            for enc in wanted:
                if enc in have:
                    return binary, enc
        logging.warning("[MMH3StreamingSave] no candidate reported %s -- trying %s anyway",
                        wanted, ordered[0])
        return ordered[0], wanted[0]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3StreamingSave",
            display_name="MMH3 Streaming Save",
            category="MMH3Tools/utils",
            description=(
                "Decode an H3 video latent in chunks straight into ffmpeg. The full "
                "pixel tensor never exists, so RAM is constant at any length - 2K at "
                "750 frames is 21 GB decoded whole. Pass decoded AUDIO to mux."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="H3 AV latent, or a plain 5D video latent."),
                io.Vae.Input("vae", tooltip="The H3 VIDEO vae."),
                io.Int.Input(
                    "groups_per_chunk", default=4, min=1, max=64, step=1,
                    tooltip="Latent GROUPS decoded per call. One group is 5 latents = 17 "
                            "frames, so 4 is ~68 frames live at a time. Each call also "
                            "decodes one extra group of context that is discarded, so "
                            "small values waste proportionally more work.",
                ),
                io.Float.Input("fps", default=24.0, min=1.0, max=120.0, step=0.01,
                               tooltip="H3 is 24. Only affects container timing."),
                io.String.Input("filename_prefix", default="MMH3/stream"),
                io.Int.Input("crf", default=19, min=0, max=51,
                             tooltip="Quality, mapped per encoder: crf (x264) / cq (nvenc) "
                                     "/ bitrate (openh264) / qscale (mpeg4)."),
                io.Audio.Input(
                    "audio", optional=True,
                    tooltip="Decoded audio to mux in. Audio decode is cheap, so it stays "
                            "outside this node. Omit for silent video.",
                ),
                io.Combo.Input(
                    "video_encoder", options=["auto"] + cls._ENCODERS, default="auto",
                    optional=True,
                    tooltip="auto probes the binary and takes the first available in "
                            "preference order.",
                ),
                io.String.Input(
                    "ffmpeg_path", default="", optional=True,
                    tooltip="Override the binary. Empty searches this path, then PATH, then "
                            "imageio-ffmpeg, taking the first that actually has a working "
                            "H.264 encoder.",
                ),
                io.Boolean.Input(
                    "save_metadata", default=True, optional=True,
                    tooltip="Embed the workflow and prompt in the mp4, so dragging the "
                            "file back into ComfyUI restores the graph. Written as an "
                            "ffmetadata file rather than a command-line argument because "
                            "a workflow is 45-95 KB and Windows caps a command line near "
                            "32,767 characters.\n\n"
                            "Turn OFF for files you are sending out: the workflow "
                            "carries every prompt and path in the graph. ComfyUI's own "
                            "--disable-metadata also wins over this.",
                ),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo],
            outputs=[io.String.Output(display_name="file_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, latent, vae, groups_per_chunk, fps, filename_prefix, crf,
                audio=None, video_encoder="auto", ffmpeg_path="",
                save_metadata=True) -> io.NodeOutput:
        video, _ = unpack_av(latent, "latent", allow_video_only=True)
        video = video[:1]
        T = int(video.shape[2])

        group, lookahead, frames_per_group = vae_grid(vae)
        n_groups = max(0, (T - 2) // group)
        if n_groups < 1:
            raise ValueError(
                "[MMH3StreamingSave] %d latents is under one group (%d + %d). Decode it "
                "normally; there is nothing to stream." % (T, group, 2))
        tail = T - (n_groups * group + 2)
        if tail:
            logging.warning("[MMH3StreamingSave] %d latents is off the %dj+2 grid by %d; "
                            "the remainder is not emitted", T, group, tail)

        # 17j+5: every group is 17 frames, plus the final chunk's carried part
        expected = frames_per_group * n_groups + TAIL_FRAMES

        ffmpeg, encoder = cls._resolve_ffmpeg(ffmpeg_path, video_encoder)
        logging.info("[MMH3StreamingSave] ffmpeg %s | encoder %s%s | %d groups, %d frames "
                     "expected", ffmpeg, encoder,
                     "" if video_encoder == "auto" else " (forced)", n_groups, expected)

        out_dir = folder_paths.get_output_directory()
        full_folder, fname, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir)
        video_tmp = os.path.join(full_folder, "%s_%05d_tmp.mp4" % (fname, counter))
        final_path = os.path.join(full_folder, "%s_%05d.mp4" % (fname, counter))

        # Attached to the FIRST pass so both endings inherit it: the silent path's
        # os.replace keeps the file as-is, and the audio mux carries the tags through
        # -c:v copy.
        meta_tmp = None
        meta_args = []
        if save_metadata:
            tags = collect_metadata(cls.hidden.prompt, cls.hidden.extra_pnginfo)
            if tags:
                meta_tmp = os.path.join(full_folder, "%s_%05d_tmp.ffmeta" % (fname, counter))
                write_ffmetadata(meta_tmp, tags)
                meta_args = ["-f", "ffmetadata", "-i", meta_tmp,
                             "-map", "0:v", "-map_metadata", "1",
                             "-movflags", "use_metadata_tags"]
                logging.info("[MMH3StreamingSave] embedding %s (%d bytes)",
                             ", ".join(sorted(tags)), os.path.getsize(meta_tmp))

        proc = None
        written = 0
        # stderr goes to a FILE, not a pipe: nothing drains a pipe during a long
        # encode, and a chatty ffmpeg would fill it and deadlock.
        err_file = tempfile.TemporaryFile()

        def err_tail():
            try:
                err_file.seek(0)
                t = err_file.read()[-2000:].decode("utf-8", "replace").strip()
            except Exception:
                t = ""
            return ("\n--- ffmpeg stderr ---\n" + t) if t else " (ffmpeg printed no output)"

        try:
            g0 = 0
            while g0 < n_groups:
                comfy.model_management.throw_exception_if_processing_interrupted()
                g1 = min(g0 + int(groups_per_chunk), n_groups)
                last = g1 >= n_groups

                lo = max(0, group * g0 - group)          # one group of context
                hi = min(T, group * g1 + 2)              # +2 lookahead
                px = vae.decode(video[:, :, lo:hi])
                if isinstance(px, tuple):
                    px = px[0]
                if px.ndim == 5:
                    px = px.reshape(-1, *px.shape[-3:])

                head = frames_per_group if g0 > 0 else 0
                keep = frames_per_group * (g1 - g0)
                # the trailing 5 are this call's last chunk's carried part, written raw
                # here but BLENDED in a full decode -- keep them only at the true end
                px = px[head:head + keep + (TAIL_FRAMES if last else 0)]

                if proc is None:
                    H, W = int(px.shape[1]), int(px.shape[2])
                    if (W % 2) or (H % 2):
                        raise ValueError(
                            "[MMH3StreamingSave] frame size %dx%d has an odd dimension; "
                            "yuv420p needs both even. H3 latents are always /16 of a "
                            "/32 canvas, so a hand-cropped latent is the usual cause."
                            % (W, H))
                    proc = subprocess.Popen(
                        [ffmpeg, "-y", "-loglevel", "error",
                         "-f", "rawvideo", "-pix_fmt", "rgb24",
                         "-s", "%dx%d" % (W, H), "-r", str(fps), "-i", "pipe:"]
                        + meta_args
                        + cls._encoder_args(encoder, crf, W, H) + [video_tmp],
                        stdin=subprocess.PIPE, stderr=err_file)

                data = (px.clamp(0, 1).mul(255).round()
                        .to(torch.uint8).cpu().contiguous().numpy().tobytes())
                try:
                    proc.stdin.write(data)
                except (BrokenPipeError, OSError):
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass
                    ret = proc.wait()
                    proc = None
                    raise RuntimeError("[MMH3StreamingSave] ffmpeg died mid-stream "
                                       "(exit %s).%s" % (ret, err_tail()))
                written += int(px.shape[0])
                logging.info("[MMH3StreamingSave] groups [%d,%d) of %d -> %d frames "
                             "(total %d)", g0, g1, n_groups, int(px.shape[0]), written)
                del px, data
                g0 = g1

            proc.stdin.close()
            ret = proc.wait()
            if ret != 0:
                raise RuntimeError("[MMH3StreamingSave] ffmpeg exited with %s.%s"
                                   % (ret, err_tail()))
            proc = None
        finally:
            if proc is not None:
                try:
                    proc.stdin.close()
                except Exception:
                    pass
                proc.kill()
            err_file.close()

        if written != expected:
            logging.warning("[MMH3StreamingSave] wrote %d frames, expected %d. The VAE's "
                            "chunking no longer matches the %dj+2 grid this slices on.",
                            written, expected, group)

        if audio is not None and audio.get("waveform") is not None:
            # RAW PCM, not torchaudio.save: since torchaudio 2.9 that call routes through
            # TorchCodec, which ComfyUI does not require and which fails to load its
            # FFmpeg DLLs on many Windows installs -- a hard error on a machine that has
            # a perfectly good ffmpeg BINARY sitting right there. This function already
            # spawns ffmpeg to write the video, so interleaved f32le needs no encoder
            # library at all and cannot break when torchaudio next moves its backend.
            pcm_tmp = os.path.join(full_folder, "%s_%05d_tmp.f32" % (fname, counter))
            wf = audio["waveform"]
            if wf.ndim == 3:
                wf = wf[0]                      # [B,C,T] -- save one take, not a batch
            wf = wf.detach().to(torch.float32).cpu()
            channels = int(wf.shape[0])
            # ffmpeg reads f32le interleaved: sample 0 of every channel, then sample 1.
            wf.t().contiguous().numpy().tofile(pcm_tmp)
            try:
                mux = subprocess.run(
                    [ffmpeg, "-y", "-loglevel", "error", "-i", video_tmp,
                     "-f", "f32le", "-ar", str(int(audio["sample_rate"])),
                     "-ac", str(channels), "-i", pcm_tmp,
                     "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
                    + (["-movflags", "use_metadata_tags"] if meta_args else [])
                    + [final_path],
                    stderr=subprocess.PIPE)
                if mux.returncode != 0:
                    t = (mux.stderr[-2000:].decode("utf-8", "replace").strip()
                         if mux.stderr else "")
                    raise RuntimeError(
                        "[MMH3StreamingSave] audio mux failed (ffmpeg exit %s); the silent "
                        "video is kept at %s%s"
                        % (mux.returncode, video_tmp,
                           ("\n--- ffmpeg stderr ---\n" + t) if t else ""))
            finally:
                try:
                    os.remove(pcm_tmp)
                except OSError:
                    pass
            os.remove(video_tmp)
        else:
            os.replace(video_tmp, final_path)

        if meta_tmp is not None:
            try:
                os.remove(meta_tmp)
            except OSError:
                pass

        logging.info("[MMH3StreamingSave] %d frames (%.2fs) -> %s",
                     written, written / float(fps), final_path)
        return io.NodeOutput(
            final_path,
            ui={"images": [{"filename": os.path.basename(final_path),
                            "subfolder": subfolder, "type": "output"}],
                "animated": (True,)},
        )


# Two-pass lands within ~1-2% of its target, and the mp4 container adds its own
# fraction of a percent on top. The video bitrate is scaled by this so a copy aimed
# at a limit ends up under it rather than a hair over, which is the only failure
# that matters when the number is an upload ceiling.
SIZE_SAFETY = 0.97

# Below this the picture is falling apart at any resolution, so the useful answer is
# to say what would fix it rather than to encode mush.
MIN_SANE_KBPS = 150


def size_capped_bitrate(target_mb, duration_s, audio_kbps, safety=SIZE_SAFETY):
    """Video kbps whose file lands at `target_mb` MiB over `duration_s` seconds.

    MiB, not MB: upload limits are quoted in binary megabytes, and at a 100 "MB"
    ceiling the two differ by 5 MB -- enough to be the whole safety margin.
    """
    if duration_s <= 0:
        raise ValueError("[MMH3SizeCappedCopy] duration must be positive, got %r"
                         % (duration_s,))
    total_kbits = float(target_mb) * 1024.0 * 1024.0 * 8.0 / 1000.0
    video_kbps = (total_kbits / float(duration_s) - float(audio_kbps)) * safety
    if video_kbps <= 0:
        raise ValueError(
            "[MMH3SizeCappedCopy] %.1f MiB over %.1fs leaves nothing for video after "
            "%d kbps of audio. Raise target_mb, lower audio_kbps, or cut the clip."
            % (target_mb, duration_s, audio_kbps))
    return video_kbps


def capped_copy_plan(src_mb, target_mb, src_height, max_height):
    """What a size-capped copy can actually achieve.

    -> (needed, effective_target_mb, reason)

    `target_mb` is a CEILING, exactly as `max_height` already is -- scale_filter
    says so in code (`min(ih, max_height)`, "a master already shorter than the cap
    is never upscaled into it") and the size budget must mean the same thing.

    It did not. `size_capped_bitrate` solves purely from target and duration and
    never looks at the source, and `-b:v` in two-pass x264 is a TARGET AVERAGE, not
    a limit -- so a source already under the ceiling was re-encoded UP to it. The
    added bits cannot add information: they go into finer quantization of an
    already-decoded picture, which spends bandwidth faithfully reproducing the
    FIRST encode's blocking and ringing. Bigger file, a generation of loss, and a
    slow two-pass encode, for nothing.

    Two guards, because there are two ways to be inside the ceiling:
      * nothing to do at all      -> caller skips the encode entirely
      * under size but too TALL   -> the encode must run, so clamp the budget to
                                     the source's own size and it still cannot grow

    An unknown height is treated as possibly-too-tall: letting the encode run costs
    time, wrongly skipping it silently ignores max_height.
    """
    cap_h = int(max_height) if max_height and int(max_height) > 0 else 0
    over_size = float(src_mb) > float(target_mb)
    over_height = cap_h > 0 and (src_height is None or int(src_height) > cap_h)
    effective = min(float(target_mb), float(src_mb))

    if not (over_size or over_height):
        reason = "%.2f MiB is already under the %.1f MiB ceiling" % (src_mb, target_mb)
        if cap_h:
            reason += " and %dp is within the %dp cap" % (int(src_height), cap_h)
        return False, effective, reason
    if over_size and over_height:
        reason = "over the size ceiling and taller than %dp" % cap_h
    elif over_size:
        reason = "over the size ceiling"
    else:
        reason = ("under the size ceiling but %s than %dp, so the encode runs with the "
                  "budget clamped to the source's own size"
                  % ("taller" if src_height else "of unknown height rather", cap_h))
    return True, effective, reason


def scale_filter(max_height):
    """`-vf` args capping height at `max_height`, or [] for native.

    min() rather than a plain height so a master already shorter than the cap is
    never upscaled into it, and -2 keeps the aspect while forcing both dimensions
    even, which yuv420p requires. The comma is escaped because an unescaped one ends
    the filter inside a filtergraph.
    """
    if not max_height or int(max_height) <= 0:
        return []
    return ["-vf", "scale=-2:min(ih\\,%d)" % int(max_height)]


class MMH3SizeCappedCopy(io.ComfyNode):
    """Transcode a finished video to a hard file-size ceiling, two-pass."""

    @staticmethod
    def _ffprobe_for(ffmpeg):
        """The ffprobe sitting beside this ffmpeg, or None.

        Beside, not on PATH: the whole point of _resolve_ffmpeg is that the binary on
        PATH may be a different build, and a duration read from one install while
        encoding with another is the kind of mismatch that produces a silently wrong
        bitrate.
        """
        d, base = os.path.dirname(ffmpeg), os.path.basename(ffmpeg)
        i = base.lower().rfind("ffmpeg")
        if i < 0:
            return None
        cand = os.path.join(d, base[:i] + "ffprobe" + base[i + 6:])
        return cand if os.path.exists(cand) else None

    @classmethod
    def _probe(cls, ffmpeg, path):
        """(duration_s, height or None). Duration is required, height best-effort.

        JSON rather than `-of default=nw=1:nk=1`: asking for a stream field and a
        format field together emits bare values whose ORDER is not guaranteed, and
        silently reading a height as a duration is exactly the sort of mismatch
        _ffprobe_for exists to avoid.
        """
        dur = height = None
        probe = cls._ffprobe_for(ffmpeg)
        if probe:
            try:
                out = subprocess.run(
                    [probe, "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "format=duration:stream=height", "-of", "json", path],
                    capture_output=True, timeout=60)
                j = json.loads((out.stdout or b"{}").decode("utf-8", "replace") or "{}")
                v = float(j.get("format", {}).get("duration", 0) or 0)
                if v > 0:
                    dur = v
                streams = j.get("streams") or []
                if streams and streams[0].get("height"):
                    height = int(streams[0]["height"])
            except Exception as e:
                logging.warning("[MMH3SizeCappedCopy] ffprobe unusable (%s); reading "
                                "the duration off ffmpeg instead", e)
        if dur is not None and height is not None:
            return dur, height

        # ffprobe is not guaranteed to ship beside ffmpeg -- imageio-ffmpeg's build has
        # none at all -- so fall back to ffmpeg's own banner.
        out = subprocess.run([ffmpeg, "-hide_banner", "-i", path],
                             capture_output=True, timeout=60)
        text = (out.stderr or b"").decode("utf-8", "replace")
        if dur is None:
            m = re.search(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)", text)
            if not m:
                raise RuntimeError(
                    "[MMH3SizeCappedCopy] could not read a duration from %s. ffmpeg "
                    "said:\n%s" % (path, text[-1500:].strip()))
            dur = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
        if height is None:
            # the resolution on the Video stream line, avoiding SAR/DAR ratios
            m = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})[\s,]", text)
            if m:
                height = int(m.group(2))
        return dur, height

    @classmethod
    def _duration(cls, ffmpeg, path):
        return cls._probe(ffmpeg, path)[0]

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3SizeCappedCopy",
            display_name="MMH3 Size Capped Copy",
            category="MMH3Tools/utils",
            description=(
                "Two-pass transcode of a finished video to a hard size ceiling, for "
                "upload limits. Chains off MMH3 Streaming Save's file_path; works on "
                "any video file. Writes beside the source and leaves it untouched."
            ),
            inputs=[
                io.String.Input(
                    "file_path", default="",
                    tooltip="Video to copy. Takes MMH3 Streaming Save's file_path "
                            "output, or an absolute path.",
                ),
                io.Float.Input(
                    "target_mb", default=95.0, min=1.0, max=4096.0, step=1.0,
                    tooltip="Ceiling in binary MB (MiB), audio included. The encode "
                            "aims %d%% under it to absorb rate-control drift. A "
                            "CEILING, not a target: a file already under it is not "
                            "re-encoded at all and the source path is returned "
                            "unchanged, with no copy written."
                            % int(round((1 - SIZE_SAFETY) * 100)),
                ),
                io.Int.Input(
                    "max_height", default=1080, min=0, max=4320, step=8,
                    tooltip="Cap on output height; 0 keeps the source size. A source "
                            "already shorter is left at its own height. Width follows "
                            "the aspect, rounded to even.",
                ),
                io.Int.Input(
                    "audio_kbps", default=128, min=32, max=320, step=8,
                    tooltip="AAC bitrate, subtracted from the budget before the video "
                            "bitrate is solved. Sources with no audio track ignore it.",
                ),
                io.Combo.Input(
                    "preset", options=["veryslow", "slower", "slow", "medium", "fast"],
                    default="slow",
                    tooltip="x264 preset. Trades encode time against compression at a "
                            "fixed bitrate; it does not change the output size.",
                ),
                io.String.Input(
                    "suffix", default="_capped", optional=True,
                    tooltip="Appended to the source's name for the copy. Cannot be "
                            "empty -- the source is never overwritten.",
                ),
                io.String.Input(
                    "ffmpeg_path", default="", optional=True,
                    tooltip="Override the binary. Empty searches this path, then PATH, "
                            "then imageio-ffmpeg, taking the first with libx264.",
                ),
            ],
            outputs=[io.String.Output(display_name="file_path")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, file_path, target_mb, max_height, audio_kbps, preset,
                suffix="_capped", ffmpeg_path="") -> io.NodeOutput:
        src = (file_path or "").strip().strip('"')
        if not src:
            raise ValueError("[MMH3SizeCappedCopy] file_path is empty.")
        if not os.path.isfile(src):
            raise ValueError("[MMH3SizeCappedCopy] no such file: %s" % src)
        suffix = (suffix or "").strip()
        if not suffix:
            raise ValueError("[MMH3SizeCappedCopy] suffix cannot be empty; the copy "
                             "would overwrite the source.")

        out_path = os.path.splitext(src)[0] + suffix + ".mp4"
        if os.path.abspath(out_path) == os.path.abspath(src):
            raise ValueError("[MMH3SizeCappedCopy] suffix %r resolves to the source "
                             "itself." % suffix)

        # libx264 specifically: this is a two-pass node, and the stats-log flow the
        # passes share is an x264 thing. Asking _resolve_ffmpeg for it also skips a
        # build that lacks the encoder rather than failing at pass 1.
        ffmpeg, _ = MMH3StreamingSave._resolve_ffmpeg(ffmpeg_path, "libx264")

        duration, src_h = cls._probe(ffmpeg, src)
        src_mb = os.path.getsize(src) / (1024.0 * 1024.0)
        needed, effective_mb, why = capped_copy_plan(src_mb, target_mb, src_h, max_height)

        if not needed:
            # Nothing a re-encode could improve, and encoding anyway would INFLATE the
            # file to the target -- see capped_copy_plan. The source path is returned
            # as-is: no copy is written, so downstream steps that expect a distinct
            # `_capped` file will receive the master itself. Do not point a destructive
            # step at this output.
            logging.info("[MMH3SizeCappedCopy] %s | %s -- no copy written, returning "
                         "the source path unchanged",
                         os.path.basename(src), why)
            return io.NodeOutput(src, ui=cls._preview_for(src))

        video_kbps = size_capped_bitrate(effective_mb, duration, audio_kbps)
        logging.info("[MMH3SizeCappedCopy] %s | %.2f MiB, %.2fs, %s -> %s "
                     "budget %.1f MiB = %d kbps video + %d kbps audio",
                     os.path.basename(src), src_mb, duration,
                     ("%dp" % src_h) if src_h else "height unknown", why,
                     effective_mb, int(video_kbps), audio_kbps)
        if video_kbps < MIN_SANE_KBPS:
            logging.warning(
                "[MMH3SizeCappedCopy] %d kbps over %.0fs will look badly degraded. "
                "Lower max_height (currently %s), drop audio_kbps, or split the video.",
                int(video_kbps), duration, max_height or "native")

        common = (["-c:v", "libx264", "-preset", preset,
                   "-b:v", "%dk" % int(video_kbps), "-pix_fmt", "yuv420p"]
                  + scale_filter(max_height))

        tmpdir = tempfile.mkdtemp(prefix="mmh3cap_")
        passlog = os.path.join(tmpdir, "pass")

        def run(stage, args):
            r = subprocess.run(args, capture_output=True)
            if r.returncode != 0:
                tail = (r.stderr or b"")[-2000:].decode("utf-8", "replace").strip()
                raise RuntimeError(
                    "[MMH3SizeCappedCopy] %s failed (ffmpeg exit %s).%s"
                    % (stage, r.returncode,
                       ("\n--- ffmpeg stderr ---\n" + tail) if tail else ""))

        try:
            # -f null rather than a real mp4 to the null device: the mp4 muxer wants
            # seekable output and complains at one. Only the stats log matters here.
            run("pass 1",
                [ffmpeg, "-y", "-loglevel", "error", "-i", src, "-map", "0:v:0"]
                + common + ["-pass", "1", "-passlogfile", passlog, "-an",
                            "-f", "null", "-"])
            # The filters must match pass 1 exactly or the stats describe a different
            # picture than the one being encoded.
            run("pass 2",
                [ffmpeg, "-y", "-loglevel", "error", "-i", src,
                 "-map", "0:v:0", "-map", "0:a:0?"]
                + common + ["-pass", "2", "-passlogfile", passlog,
                            "-c:a", "aac", "-b:a", "%dk" % int(audio_kbps),
                            "-movflags", "+faststart", out_path])
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        out_mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
        if out_mb > target_mb:
            logging.warning("[MMH3SizeCappedCopy] %.2f MiB is OVER the %.1f MiB "
                            "target. Rate control undershot; lower target_mb and "
                            "re-run.", out_mb, target_mb)
        logging.info("[MMH3SizeCappedCopy] %.2f MiB -> %.2f MiB (%.0f%%) -> %s",
                     src_mb, out_mb, 100.0 * out_mb / max(src_mb, 1e-9), out_path)

        return io.NodeOutput(out_path, ui=cls._preview_for(out_path))

    @staticmethod
    def _preview_for(path):
        """UI preview dict, or {} when the file has no URL the frontend can fetch.

        Only files inside ComfyUI's output tree are servable; the source can be
        anywhere, which matters now that the no-work path returns it directly.
        """
        try:
            rel = os.path.relpath(path, folder_paths.get_output_directory())
        except ValueError:
            return {}  # different drive on Windows
        if rel.startswith(".."):
            return {}
        return {"images": [{"filename": os.path.basename(path),
                            "subfolder": os.path.dirname(rel).replace("\\", "/"),
                            "type": "output"}],
                "animated": (True,)}

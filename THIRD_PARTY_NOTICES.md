# IAMCCS-nodes third-party notices

Copyright (C) 2025-2026 Carmine Cristallo Scalzi (IAMCCS)

## Project license

IAMCCS-nodes is distributed under the GNU General Public License, version 3
or later (GPL-3.0-or-later). The legally controlling license text is in
`LICENSE`.

## IAMCCS implementation and upstream references

IAMCCS-nodes includes original IAMCCS architecture, implementation,
integration, orchestration and production tooling. This includes the IAMCCS
Shotboard and LongVid systems; editorial timeline planning; generation-window
geometry; guide, audio and endpoint routing; AudioCon continuity; validation;
compatibility; and delivery tooling.

Some IAMCCS nodes are independent implementations informed by, adapted from,
or deduced from public methods, behavior or documentation of the upstream
projects below. This attribution does not state that upstream source code was
copied into IAMCCS-authored modules. An upstream project's license governs
upstream material only where that material is actually included or forms a
derivative work.

| Upstream project | License | IAMCCS reference |
| --- | --- | --- |
| ComfyUI-MiniMaxH3-TimelineDirector | GPL-3.0 | Latent-tail continuation, temporal masking and Drift-Control concepts used by IAMCCS Pianosequenza modes. |
| ComfyUI-MiniMaxH3-Context-Loop | GPL-3.0 | Drift-Control and schedule-matched temporal-mask lineage. |
| Herrgotts-H3-Infinite-Continuation-Suite | GPL-3.0 | Phase-aware latent continuation concepts informing the IAMCCS FL2VA and Pianosequenza work. |
| ComfyUI-Minimax-H3-Continuation | MIT | Bounded continuation-window and synchronized AV-context concepts. |
| ComfyUI-MiniMax-H3-LongMedia | Apache-2.0 | Previous-latent-tail, frozen-prefix and native continuation concepts. |
| ComfyUI-Wan-SVI2Pro-FLF | GPL-3.0 | Logic adapted in `iamccs_wan_svipro_motion.py`. |
| ComfyUI-KJNodes | GPL-3.0 | Logic adapted in `iamccs_wan_svipro_motion.py`. |
| ComfyUI | GPL-3.0 | Platform integration and compatibility reference. |

The upstream projects above are credited for their relevant technical lineage.
Their authors do not endorse, certify or necessarily support IAMCCS-nodes.

## Included vendored material

`vendor/mmh3tools_r38b` is an immutable upstream snapshot of
ComfyUI-MMH3Tools (https://github.com/ckinpdx/ComfyUI-MMH3Tools), identified
in `vendor/mmh3tools_r38b/UPSTREAM.txt` as MIT-licensed. Its upstream notices
and license obligations remain applicable to that material.

## Notice handling

Where a future change includes copied or substantially derived MIT- or
Apache-2.0 source material, retain the upstream copyright, license and NOTICE
requirements with that material. For GPL-covered derivative material, preserve
the required notices and distribute under GPL-compatible terms.

This file is an attribution notice, not legal advice.

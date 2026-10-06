# IAMCCS MiniMax H3 Face Delivery R44 — Multi Face / Crowd Safe

## Purpose
The existing Face Delivery path refined one tracked subject. R44 keeps that exact path for `single` mode and adds an opt-in sequential `multi_face` mode for crowd/procession shots.

## Runtime model
- Tracking for every candidate is performed against the untouched native visible frames.
- `select_index=0..N-1` chooses a subject once; the external H3 FaceRefine tracker then follows that subject by continuity instead of re-ranking faces every frame.
- H3 crop/refine is run sequentially per accepted subject. VRAM therefore remains bounded to approximately one face-refine pass at a time; runtime scales roughly with the number of accepted faces.
- Stitching is cumulative: subject 2 is composited onto the result containing subject 1, subject 3 onto subjects 1+2, etc.
- A conservative trajectory deduplicator rejects candidate tracks that converge on an already-refined person.
- `min_face_coverage` skips subjects detected in too little of the shot, avoiding expensive H3 passes for incidental/background detections.
- The original native audio tensor, temporal extent, LongVid context and motion state are never modified by the face passes.

## Controls on IAMCCS_MiniMaxH3FaceDeliveryR38B
- `face_mode`: `from_shotboard` (default), `single`, `multi_face`.
- `max_faces`: 1–8, default 4. This is an accepted-subject target, not a VRAM batch size.
- `min_face_coverage`: default 0.10. In multi-face mode a candidate must have detector hits on at least this fraction of source frames.

## Shotboard profile
`face_detailer_profile = multi_face` activates multi-face automatically when `face_mode = from_shotboard`.
The existing `balanced`, `small_faces`, `gentle`, `sam_face_mask`, and `wide_character_12gb` profiles retain single-face behavior.

## Compatibility
- Compatible with historical H3FaceTrackCrop 6-output contract.
- Compatible with H3FaceTrackCrop 1.1.x 7-output contract (`frame_count` appended).
- `wide_character_12gb` is now accepted by backend profile validation instead of being silently normalized to `balanced`.

## 12 GB recommendation
Start at 512×512, 4 steps, denoise 0.20, max_faces 3–4. Raising max_faces does not increase peak VRAM because passes are sequential, but increases wall-clock time nearly linearly. For a shot reporting 8 faces, use 4 first and raise toward 8 only if the background faces materially need refinement.

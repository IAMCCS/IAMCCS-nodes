IAMCCS Herrgotts intermediate keyframe preservation backup
===========================================================

Problem observed
----------------
Three chronological Shotboard guides produced two FL2VA intervals, but the
middle authored guide was absent from the final video.  The provider's
automatic no-lock fallback trimmed the final 9 frames of clip 1; those frames
contained the approach and landing on the middle guide.  Clip 2 consequently
received a context window ending before that guide.

Fix scope
---------
Only IAMCCS FL2VA Continuous AV / Herrgotts Direct AV is changed.  Explicit
intermediate Shotboard endpoints now use the provider's supported manual
landing tail=0 path.  Intermediate rendered tails are kept, continuation head
overlap is still removed, and safe-tail bridging is disabled for this strict
keyframe path.  R37 Motion Context and legacy backends are unchanged.

Rollback
--------
Copy the two *.before files in this directory back to IAMCCS-nodes using their
original names, then restart ComfyUI.

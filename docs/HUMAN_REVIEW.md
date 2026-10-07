# Minimal human contact review

The current checked-in recording is
[`assets/demo_inputs/badminton_sample.mp4`](../assets/demo_inputs/badminton_sample.mp4).
It is 24 FPS; review frames **0 through 199** (the first 8.33 seconds).
Identify every visible racket-shuttle contact, including times without a
predicted wrist peak. Existing unverified navigation hints from an earlier
sequential run were frames 23, 29, 75, 123 and 138. They are **not labels**.

Send just two things privately:

1. A sorted list of contact times in seconds, e.g. `1.75, 3.79`, or
   zero-based frame numbers, e.g. `42, 91`. We can convert timestamps to
   frame indices at 24 FPS; note that a coarse player timestamp adds timing
   uncertainty.
2. Any intervals where contact is not visible or is ambiguous, e.g.
   `3.0-3.5 s unclear`. If the whole clip cannot be reviewed, say so.

Do not guess an uncertain contact or mark the review complete from candidate
hints alone. Once the full interval is reviewed, the existing
`scripts/evaluate_shot_candidates.py --template` workflow binds the annotation
to the video's SHA-256 and computes one-to-one Precision/Recall/F1 and timing
error. The 60-frame YOLO bounding-box audit is deferred: it is not needed for
the software closed-loop demo, and this project is not starting another ML
experiment. No event accuracy number is available before this review.

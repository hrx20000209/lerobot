# Physical inference validation — 2026-07-02

## VLA-JEPA

- Checkpoint: `/home/hrx/Projects/models/three_cubes_1/vla_jepa`
- The checkpoint uses 16-step chunks, historical visual observations, and relative actions on the first five joints.
- Fixed checkpoint-config compatibility, PEFT `ModulesToSaveWrapper` action-head dispatch, relative-action mask/state selection, stable camera paths, and `pathlib.Path` camera opening on OpenCV 4.8.
- Physical run executed 519 actions and 72 model chunks. Median policy prediction time was 262.5 ms.
- 312/519 actions were changed by the 5-degree robot safety clamp. The arm moved away from the cubes and did not complete the task.
- Evidence: `logs/async_timeline/vla_jepa_fixed2_20260702/`.

## Giga World

- Checkpoint: `/home/hrx/Projects/models/three_cubes_1/giga_world`
- Fixed the runtime dependency mismatch by pinning `diffusers==0.36.0`; 0.30.2 lacked `AutoencoderKLWan`, and 0.35.2 lacked `diffusers.models._modeling_parallel`.
- Physical run executed 380 actions and generated 17 model chunks of 48 actions. Median policy prediction time was about 1.49 s.
- 163/380 actions were changed by the 5-degree robot safety clamp. The arm approached the cube area but did not complete a grasp in the test window.
- Median source-observation age at action execution was about 2.34 s, so closed-loop corrections are substantially stale.
- Evidence: `logs/async_timeline/giga_world_fixed_20260702_run3/`.

## Final hardware state

After both tests, the arm was reset with the SO-101 leader. The final measured follower state was:

`[1.187, -104.923, 88.659, 74.330, -2.418, 1.085]`

Both cameras connected using stable `/dev/v4l/by-path/...` identifiers (front MJPG, wrist YUYV).

# Task: render the Newton overlay videos from the raw camera images

For the Claude instance on the main machine (the one that made the other folders in
`clips36_final_v1/prediction_videos/`). The owner talks German; code, logs and commit messages stay in English.

## Goal

Make overlay videos of the Newton fit that look like the other approaches, for example `genesis_12.7/`.

`prediction_videos/newton_7.6/` already exists, but it was rendered on the cluster, where the raw images are missing. It
is drawn on `check_videos/`, so the colours and the debug text differ. The numbers are fine. Only the base images and the
look need to change.

**Do not simulate again.** The rollouts are in the repo, so the videos need no GPU and do not disturb the main run. The
simulation is not fully deterministic: a new run would change per-clip values by up to 0.16 mm/node, and one clip has
already scored 10.2 and 15.4 with the same parameters.

## Get the data

Everything is on the branch `newton-t36-results-tinygpu-2080ti`, not on `main`. Use a fresh clone, because the history of
`main` has been rewritten before. Do not work inside the folder the main run uses.

```
git clone --filter=blob:none --no-checkout https://github.com/ludwigstr/wire-harness-approaches.git
cd wire-harness-approaches
git sparse-checkout set newton_fit_t36/results newton_fit_t36/data/tracks36 clips36_final_v1/prediction_videos/newton_7.6 clips36_final_v1/prediction_videos/genesis_12.7
git checkout newton-t36-results-tinygpu-2080ti
```

| Path | Content |
|---|---|
| `newton_fit_t36/results/tinygpu_2080ti/rollouts/<clip>.npz` | `pred` (T, 4, 15, 2) float32, predicted node xy in board metres, indexed frame, cable, node, xy; `errs` (T), mm/node per frame |
| `newton_fit_t36/data/tracks36/<clip>.npz` | measured `c0..c3` (T, 16, 2) in board metres, `source_frames` (T), `fps` |
| `clips36_final_v1/prediction_videos/newton_7.6/results.json` | per-clip `reproduced` (mean of `errs`) and `reported` (fresh-process validation) |
| `newton_fit_t36/results/tinygpu_2080ti/newton_overlay.py` | how the cluster version was made; it uses `check_videos`, so do not reuse its pixel map |

The raw images are not in the repo. Use the ones you used for the other folders.

## Alignment

- Rollout frame `f` belongs to raw frame `source_frames[f]`, which is the `<n>` in `<k>_frame_<n>/01_raw.png` of the export.
  In a replay source video from `track_replay_folder.py`, it is simply index `f`, as `t0` is 0.
- Map board to pixels as in the other folders (`find_board` on the clip's first frame, then `tf.to_px(xy_m * 1000)`).
- `pred` has 15 nodes, because the score compares 15 body origins with the first 15 tracked nodes. Draw all 15 predicted
  nodes and all 16 measured nodes.

## Output

- Write to a **new** folder, `clips36_final_v1/prediction_videos/newton_7.6_raw/`. Never overwrite or delete `newton_7.6/`.
  The owner decides which one stays.
- Match the other folders: one video per clip, stills for the worst and middle frames, `results.json`, measured solid,
  prediction dashed, same colours, same label band.
- Label band: title `Newton per-cable fit, 36 clips (30 generations, mass pinned)`, then clip, `t = f / fps`, the frame's
  `errs[f]`, the clip mean, and `solid = measured, dashed = Newton`.
- Worst frame is `argmax(errs)`, middle frame is `T // 2`.
- Keep the `reproduced` and `reported` fields from the existing `results.json`. Their difference is not 0.0 (largest 0.159,
  within the 0.5 tolerance), and the mean is 7.562 in both. Do not edit the values to make them match.

## Checks before pushing

1. **All 36 clips** rendered, each video with exactly T frames. Never use a subset or clips from outside `tracks36`.
2. **Values:** recompute each clip's mean from `pred` and `tracks36` (first 15 nodes, mean over cables, every frame). It must
   equal `reproduced` to within 1e-6.
3. **Alignment:** look at frames from at least three clips across the three recording days (21, 26 and 28 Oct). The measured
   lines must lie on the cables in the image. If they do not, the frame mapping is wrong: stop and report.

## Hand back

- Commit to the branch `newton-t36-results-tinygpu-2080ti`. Do not merge into `main`; that is the owner's decision.
- Add a row for the new folder to `prediction_videos/README.md`.
- Commit messages have no `Co-Authored-By` trailer (owner's explicit wish). Never print or write GitHub tokens.
- Never stop or change the main run. Never run `pkill -f`. Report every deviation to the owner.
- Tell the owner: how many clips, the mean of `reproduced`, the result of check 3, the folder size, and that they can now
  choose between `newton_7.6/` and `newton_7.6_raw/`.

Do not change the table values. They come from the fresh-process validation: mean 7.56, median 6.53 mm/node.

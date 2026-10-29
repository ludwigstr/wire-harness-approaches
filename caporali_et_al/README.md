# Caporali et al. (RA-L 2024), one step per video frame: snapshot of the current scripts

This is a snapshot of the Caporali pipeline from the main project (Wire_Model_S). It was copied
as-is on 2026-10-28 so colleagues can use it. **It has not been adapted to this repo or tested here.**
The paths are still those of the main project; the ones to change are listed below.

It replaces the 1 s step (4 video frames per transition) of `approaches/09_caporali.py`. There
the step length was an arbitrary choice; here the MLP predicts one 0.25 s video frame per step.

**Update 2026-10-29:** the training data is now built from the 14 evaluation clips (`--tracks data/tracks14`)
instead of the 103 X-Planar clips: `scripts/run_train_on_14.sh`, checkpoint
`checkpoints/bench_film_c14_step1.pt`, result 14.54 / median 11.96 (`results_caporali_c14.json`).
The X-Planar version below (20.58) is kept for reference.

## Method

1. `scripts/make_dataset_real.py --mode step --stride 1 --speed 0.8 1.2` builds the synthetic
   training data from the analytical mass-spring simulator (`wire_model/dlo_model.py`).
   - Every chain starts from a real tracked X-Planar cable state.
   - The cable ends follow the real end paths, with speed-only variation ×0.8–1.2 and no
     spatial scaling.
   - One transition is one frame: 0.15 s of end motion plus 0.1 s of settling (the 1 s
     protocol scaled by 1/4).
   - Sampled parameters: m 0.02–0.10 kg, k_b 0.2–3.0 N m/rad, k_d 5–40 N s/m,
     k_c 0.05–0.4 N/node. The rest shape is the frame-0 turning angles ×0.6–1.2.
   - Result: 709,635 transitions.
2. `scripts/bench_models.py --models film` trains the Fig. 3 FiLM MLP: width 256, 0.42 M
   weights, 25 epochs. Synthetic one-step error 0.81 mm/node.
3. `scripts/render_neural_video.py --variant online --stride 1` runs the evaluation and renders
   the video. For each clip:
   - The network is frozen.
   - m, k_b, k_d per cable are estimated online, starting from 0.04 / 1.0 / 20 with one Adam
     step per frame.
   - The rollout is open loop from frame 0, with the ends set to the measured ends.
   - The rest shape is taken from frame 0, capped at 0.6 rad.
   - Each frame is forecast before it is used for estimation.
4. `scripts/caporali_overlay_video.py` renders overlay videos: measured shape against the
   per-frame prediction (`--frame-only`), optionally with the old per-second prediction as well.

The full chain with the exact arguments is `scripts/run_caporali_step1.sh`.

## Result on the 14 clips

20.58 mm/node mean, median 14.55. Per-clip numbers are in `results_caporali_step1.json`; the
per-second model gave 18.4 / median 14.95. The difference comes almost entirely from two clips
where the open-loop rollout collapses:

| Clip | per second (mm/node) | per frame (mm/node) |
|---|---|---|
| 102713 | 38.0 | 54.7 |
| 134613 | 25.6 | 40.3 |

Trained checkpoint: `checkpoints/bench_film_real_step1.pt`.

## Paths to adapt before running elsewhere

- The `cd` line in `run_caporali_step1.sh`.
- `DIR` / `WIRE_VIDEO_DIR` points to the raw videos. It is used in `render_neural_video.py`,
  `long_horizon_all.py` and `long_horizon_guided_sms.py`. The videos are needed only for
  drawing; the scores use the cached tracks.
- Cached tracks: `--tracks-npz` uses `data/tracks14/<clip>.npz` (the repo's top-level
  `data/tracks14`).
- X-Planar tracks for the dataset: `--tracks` of `make_dataset_real.py` (main project:
  `X-Planar_vids/mujoco_videos/tracks`, not shipped here).
- `render_neural_video.py` loads `data/transitions_1s_all.pt`, which is included here under
  `data/`. With `--epochs 0` it does not fine-tune on it.

`wire_model/` here is a full copy of the main project's package. Run the scripts from inside
`caporali_et_al/`.

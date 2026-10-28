# RopeDreamer (Missal et al. 2026) trained on simulation, as in the paper: snapshot of the current scripts

This is a snapshot of the RopeDreamer pipeline from the main project (Wire_Model_S). It was copied
as-is on 2026-10-28 so colleagues can use it. **It has not been adapted to this repo or tested here.**
The paths are still those of the main project; the ones to change are listed below.

The paper trains on MuJoCo simulation only and scores on the simulation test split. The model in
`approaches/08_ropedreamer.py` (16.8) instead trained on real footage. This snapshot follows the
paper: the training data are MuJoCo simulations around twins of the 14 evaluation clips.

## Method

1. `scripts/ropedreamer_twins14.py` builds the twins. Each of the 14 clips is replayed in the
   MuJoCo general-fit environment (`data/mujoco_general_fit3.npz`, summary-sheet row 7) with the
   real mover paths and timing. The twins' tracks are in `data/twin_tracks/`.
2. `scripts/yang_make_dataset.py --env general --randomize --motion speed --allow-eval-clips --twin-reference`
   generates variations around the twins:
   - **Motion:** the real mover paths, with speed-only variation (one shared time warp).
   - **Parameters:** domain randomization around the fit (`DR_RANGES`):
     - bend and twist ×0.5–2, with a per-cable jitter of ×0.85–1.18
     - damping ×1/3–3
     - mass ×0.7–1.4, clipped to 0.01–0.5 kg/m
     - mu ×0.5–2
     - rest angles ×0.8–1.2
   - **Rejection:** rollouts that add mover–cable contacts not present in the real recording
     are rejected.
   - **Result:** 5173 rollouts (557 rejected), 219k frames. Generation took 6.1 h on 16 workers.
3. `scripts/ropedreamer_train.py` trains RopeDreamer size S (5.6 M weights) with the paper's
   settings (beta 1, lr 1e-4, batch 32, paper loss only).
   - Validation is split at rollout level (10 %), with early stopping after 10 epochs; this run
     stopped at epoch 104 after 5.2 h.
   - Tracking-noise augmentation uses `data/ropedreamer_twins14_noise_params_v3.json`, calibrated
     with `ropedreamer_noise_calibrate.py`.
4. `scripts/ropedreamer_eval.py` evaluates with a 5-frame warm-up, open loop and both end nodes
   pinned.
5. `scripts/ropedreamer_video.py` renders the overlay videos. With `--tracks data/twin_tracks`
   the solid line is the MuJoCo twin.

The full chain with the exact arguments is in `scripts/run_ropedreamer_twins14.sh` (dataset) and
`scripts/run_ropedreamer_twins14_noise_cont.sh` (training with noise + evaluation).

## Result

| Scored on | Mean (mm/node) | Median | Results file |
|---|---|---|---|
| The MuJoCo twins of the 14 clips (the paper's protocol) | **17.24** | 6.85 | `ropedreamer_twins14_S_noise_twins14_results.json` |
| The real tracks of the 14 clips | 29.76 | 19.31 | `ropedreamer_twins14_S_noise_tracks14_results.json` |

Checkpoint: `checkpoints/ropedreamer_twins14_S_noise.pt`.

## Paths to adapt before running elsewhere

- The `cd` lines in both `run_*.sh` scripts.
- `DIR` (raw videos) in `mujoco_scene.py`, `mujoco_full_loco.py`, `render_neural_video.py`,
  `long_horizon_*.py` and `--video-dir` of `ropedreamer_video.py`. The videos are needed only
  for drawing.
- Paths inside `mujoco_general_fit_cmaes.py`.
- `--tracks data/tracks14`: the repo's top-level `data/tracks14`.

`wire_model/` here is a full copy of the main project's package. Run the scripts from inside
`ropedreamer_twins14/`.

# Yang et al. (RAS 2022) on Newton — snapshot of the current scripts

Snapshot of the Yang et al. pipeline from the main project (Wire_Model_S), copied as-is on
2026-10-28 so colleagues can use it. **Not yet adapted to this repo and not tested here**:
paths are still those of the main project (see below).

Method (Yang, Stork, Stoyanov 2022, with Newton instead of AGX Dynamics):
1. `scripts/yang_identify.py --engine newton` — differential evolution on the one-step error
   (Eq. 1) over 8 global Newton parameters (k_stretch, d_stretch, k_bend, d_bend, k_twist,
   d_twist, mass_per_m, mu), on all 14 clips (734 transitions).
2. `scripts/yang_sim_control.py` — the identified simulator alone on the 14 clips.
3. `scripts/yang_make_dataset.py --engine newton --env yang --motion speed --randomize` —
   synthetic data from the identified simulator: real X-Planar mover paths at speed x0.65-1.4
   (one shared time warp), parameters randomized around the identified set. Additions to the
   paper: the randomization and the speed-only motion variation.
4. `scripts/yang_train.py --val-by-clip --gap-ref-tol 0.010` — interaction network + bi-LSTM
   (wire_model/yang/model.py, 1.16 M weights), 60 epochs, clip-level validation. A transition is
   dropped only if its stretch gap exceeds that of the real-timing rollout of its clip + 10 mm
   (so violations the real recording already has are tolerated).
5. `scripts/yang_eval.py` — evaluation on the 14 clips (5-frame warm-up, open loop, ends pinned).
6. `scripts/yang_video.py` — overlay videos (measured solid, prediction dashed); they reproduce
   every clip's number exactly (`results/video_reproduction.json`).

The chain with the exact arguments: `scripts/run_yang_newton_paper3_chain.sh` (identification,
simulator control, dataset), then `scripts/run_yang_newton_paper3_cont.sh` (training + evaluation).

## Result on the 14 clips

Identified (DE, 40 generations, 4920 evaluations, one-step error 3.98 mm/node; all values inside
the bounds): k_stretch 4367 N/m, d_stretch 0.016, k_bend 3.66 N m/rad, d_bend 7.45 N m s/rad,
k_twist 0.053 N m/rad, d_twist 0.14, mass 0.087 kg/m, mu 4.48.

| Run | Mean (mm/node, 5-frame warm-up) | Median |
|---|---|---|
| Identified Newton simulator alone | 11.15 | 10.10 |
| **IN + bi-LSTM, paper protocol (`yang_newton_paper3b`)** | **14.38** | **11.25** |
| IN + bi-LSTM without `--val-by-clip --gap-ref-tol` (first chain) | 17.04 | 13.04 |
| Previous run, DE on 3 clips | 20.09 | 14.72 |

The first chain trained without the tolerant gap filter; with the soft identified stretch it
dropped 71.9 % of the transitions. The continuation keeps 93.9 % (437k transitions).

Fitting time, wall clock: DE 17.5 h, dataset 1.7 h (711 rollouts, 113 rejected), training 0.4 h.

Shipped: `checkpoints/yang_newton_paper3b_inbilstm.pt`, `data/yang_newton_paper3_synth/`
(the synthetic dataset), `results/` (identification, simulator control, evaluations, logs).

Paths to adapt before running elsewhere:
- `TRACKS` in yang_identify.py / yang_eval.py / yang_sim_control.py / mujoco_general_fit_cmaes.py
  -> this repo's `data/tracks14`
- `DIR` in mujoco_scene.py / mujoco_full_loco.py (raw videos; not needed when cached tracks are used)
- `cd` line in run_yang_newton_paper3_chain.sh
- X-Planar tracks for the dataset: `--tracks` of yang_make_dataset.py (main project:
  X-Planar_vids/mujoco_videos/tracks, not shipped here)

`wire_model/` here is a full copy of the main project's package (the repo's top-level
`wire_model/` is older and lacks `wire_model/yang`); run the scripts from inside `yang_et_al/`.

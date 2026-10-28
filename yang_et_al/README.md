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
4. `scripts/yang_train.py` — interaction network + bi-LSTM (wire_model/yang/model.py), 60 epochs,
   clip-level validation.
5. `scripts/yang_eval.py` — evaluation on the 14 clips (5-frame warm-up, open loop, ends pinned).

The full chain with the exact arguments: `scripts/run_yang_newton_paper3_chain.sh`.

Status: the run with identification on all 14 clips is in progress. The previous run (DE on 3
clips) gave 20.09 mm/node mean (median 14.72) on the 14 clips.

Paths to adapt before running elsewhere:
- `TRACKS` in yang_identify.py / yang_eval.py / yang_sim_control.py / mujoco_general_fit_cmaes.py
  -> this repo's `data/tracks14`
- `DIR` in mujoco_scene.py / mujoco_full_loco.py (raw videos; not needed when cached tracks are used)
- `cd` line in run_yang_newton_paper3_chain.sh
- X-Planar tracks for the dataset: `--tracks` of yang_make_dataset.py (main project:
  X-Planar_vids/mujoco_videos/tracks, not shipped here)

`wire_model/` here is a full copy of the main project's package (the repo's top-level
`wire_model/` is older and lacks `wire_model/yang`); run the scripts from inside `yang_et_al/`.

# PhysWorld (Yang et al. 2025) with its own MPM twin, as in the paper: snapshot of the current scripts

This is a snapshot of the PhysWorld rebuild from the main project (Wire_Model_S). It was copied as-is on
2026-10-29 so colleagues can use it. **It has not been adapted to this repo or tested here.** The paths
are still those of the main project; the ones to change are listed below.

`approaches/06_physworld.py` (21.8) built its world model on PhysTwin's spring-mass twins, which the
paper does not do. This rebuild follows the paper: each clip gets its own differentiable MPM digital
twin (Sec. 3.1, App. B.2). The twin generates the demonstrations, a GNN is trained on them, and then
per-node Phi is fitted on the real clip.

## Method (one model per clip, as in the paper)

1. **Twin fit:** `scripts/physworld_mpm_fit.py --model elastoplastic --global-iters 10 --local-iters 6`
   fits the MPM twin of the clip with Warp.
   - The global stage fits homogeneous E, rho, mu (board friction), Poisson nu and yield stress.
   - The local stage fits per-particle E, rho and mu on top of that, about 23k particles per clip;
     nu and the yield stress stay global.
   - The loss is the paper's: Chamfer + 0.1 smooth-L1 track loss over all frames.
   - `--frac 0.33` only sets which share of the frames gives the gradient in each iteration. All
     frames are simulated and all of them count in the loss.
   - Checks of the gradient handling: `physworld_mpm_gradcheck*.py`.
2. **Demos:** `scripts/physworld_mpm_demos.py --n 40` generates 40 demonstrations per clip from the
   twin, 560 in total with 29.6k frames.
3. **GNN and Phi:** `scripts/physworld_mpm_gnn.py`.
   - The GNN (0.86 M weights) is trained on the demos with history h = 3, the multi-step loss of
     Eq. 8 and noise on the history and on Phi.
   - Phi is the per-node (log E, log rho), 2 × 16 × 4 = 128 values per clip. It is fine-tuned on
     the real clip with the network frozen. The objective is the open-loop rollout, MSE +
     Chamfer, and Phi stays within ±1.5 (log) of the twin's value.
   - Evaluation: 5-frame warm-up, then open loop with the measured end motions, ends pinned.
4. **Summary:** `scripts/physworld_mpm_summary.py` writes `physworld_mpm_14.json`.
5. **Videos:** `scripts/physworld_mpm_video.py` renders the overlay videos. They reproduce every
   clip's number exactly (`video_reproduction.json`).

The chain per clip is in `scripts/physworld_mpm_chains/stream.sh`. The other chain scripts are
tests and variants (no-yaw, one-step Phi objective, box 2.5).

## Result on the 14 clips

| Variant | Mean (mm/node) | Median |
|---|---|---|
| MPM twin alone | 11.31 | 10.60 |
| PhysWorld before Phi fine-tuning | 13.47 | 11.33 |
| **PhysWorld after Phi fine-tuning** | **9.52** | **7.12** |

All numbers are in-sample: Phi is fitted on the frames it is scored on, as for the physics rows.

Fitting time, wall clock: twins 14.2 h, demos 12.8 h, GNN 2.7 h, Phi 1.3 h.

Shipped data:
- `data/twins/`: the fitted twins
- `data/demos/`: the demonstrations
- `data/gnn/`: the GNN checkpoints, plus a JSON per clip with the fitted Phi and the per-frame errors

## Paths to adapt before running elsewhere

- The `cd` line in the chain scripts.
- `DIR` / `WIRE_VIDEO_DIR` in `render_neural_video.py` and `long_horizon_*.py`: the raw videos.
  They are needed only for drawing.
- The scripts read and write `data/wm/physworld_mpm_{twins,demos,gnn}/`. Here those files are
  shipped under `data/{twins,demos,gnn}/`.
- Tracks: `data/tracks14` (the repo's top-level `data/tracks14`).

`wire_model/` here is a full copy of the main project's package, including
`wire_model/physworld_mpm/` (MPM solver, twin and GNN). Run the scripts from inside `physworld_mpm/`.
Requires NVIDIA Warp.

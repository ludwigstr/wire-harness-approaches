# Wire-harness cable dynamics: ten approaches

Predicting how the four cables of a wire harness move on the assembly board,
from tracked video. Every approach gets the same input -- the cable's tracked
shape at the start and the measured motion of the two cable ends (hub and plug)
-- and predicts the shape of every cable over the rest of the clip. The score
is the mean distance between predicted and tracked cable nodes (16 per cable),
in mm per node, averaged over the 14 evaluation clips.

One script per approach, in the order of the approaches table:

| # | Script | Approach | Learns from | Table rows | Best (mm/node) |
|---|---|---|---|---|---|
| 01 | `approaches/01_mujoco.py` | MuJoCo cable composites in the measured scene | fit on the 14 clips | 1-4 | 8.5 |
| 02 | `approaches/02_genesis.py` | Genesis 1.3.3 hinge chains | fit on the 14 clips | 5-7 | 18.4 |
| 03 | `approaches/03_newton.py` | NVIDIA Newton rods + measured cable height | fit on the 14 clips | 8-14 | 7.8 |
| 04 | `approaches/04_dlolab.py` | DLO-Lab Cosserat rod, gradient fit | fit on the clips | 15-18 | 16.56 |
| 05 | `approaches/05_phystwin.py` | PhysTwin spring-mass twin (Jiang et al. 2025) | one twin per clip | world 19 | 19.8 |
| 06 | `approaches/06_physworld.py` | PhysWorld (Yang et al. 2025): twin-synthesised demos -> GNN | twins + per-node fit | world 20 | 21.8 |
| 07 | `approaches/07_deformnet.py` | DeformNet (Li et al. 2024): PointNet + RSSM in embedding space | tracked videos | world 21 | 30.6 |
| 08 | `approaches/08_ropedreamer.py` | RopeDreamer (Missal et al. 2026): RSSM over a quaternion chain | tracked videos | world 22 | 16.8 |
| 09 | `approaches/09_caporali.py` | Caporali et al. (RA-L 2024): MLP + online parameter estimation | simulator only (real-anchored data) | world 23 | 18.4 |
| 10 | `approaches/10_adaptigraph.py` | AdaptiGraph (Zhang et al. 2024): property-conditioned graph dynamics | simulator (real-anchored) + real fine-tune | world 24 | 24.4 |
| 11 | `yang_et_al/` | Yang et al. 2022 (IN + bi-LSTM), Newton instead of AGX: DE identification -> synthetic data -> network; snapshot of the current scripts, not yet adapted/tested in this repo | DE on 14 clips + net | summary row 4 | 13.3 (data from the 14 clips) |
| 12 | `caporali_et_al/` | Caporali et al. (RA-L 2024), one step per video frame (0.25 s) instead of 1 s: retrained MLP + online estimation every frame; snapshot of the current scripts, not yet adapted/tested in this repo | simulator only (data from the 14 clips, speed x0.8-1.2) | summary row 5 | 14.5 |
| 13 | `ropedreamer_twins14/` | RopeDreamer as in the paper: trained on MuJoCo simulations only (twins of the 14 clips + speed/parameter variations), scored on the twins; snapshot of the current scripts, not yet adapted/tested in this repo | simulator only (MuJoCo twins) | summary row 1 | 18.3 (real tracks; 14-frame windows) |
| 14 | `physworld_mpm/` | PhysWorld as in the paper: own differentiable MPM twin per clip (global then per-particle E, rho, mu; nu, yield) -> 40 demos -> GNN; per-node Phi (E, rho) fitted on the real clip; snapshot of the current scripts, not yet adapted/tested in this repo | MPM twins + Phi on real | summary row 3 | 9.5 |
| 15 | `adaptigraph_c14/` | AdaptiGraph with simulator data built from the 14 clips (speed-only variation) + real fine-tuning; snapshot of the current scripts, not yet adapted/tested in this repo | simulator (14 clips) + real fine-tune | summary row 6 | 21.1 |

All numbers are **in-sample**: every approach is fitted or fine-tuned on the
same 14 clips it is scored on (the physics rows fit their parameters there; the
learned world models are fine-tuned there). Out-of-sample numbers on the 103
X-Planar clips are in the main project's X-Planar sheet; RopeDreamer, for
example, scores 62.4 on the 14 clips without fine-tuning on them.

## Quick start

```bash
pip install -r requirements.txt
python approaches/08_ropedreamer.py eval       # -> results/ropedreamer_14.json, 16.8
python approaches/01_mujoco.py eval --row 4    # -> results/mujoco_row4.json, 8.5
```

Every script has `eval` (score the shipped fitted parameters / checkpoints on
`data/tracks14`) and a fit or training step (`fit`, `train`, `pretrain` /
`finetune`, `all`); `--help` explains each. Results land in `results/`, fits and
training intermediates in `work/`.

## Repository layout

```
approaches/      the ten scripts + common.py (clips, metric, result format),
                 rssm_common.py (RopeDreamer/DeformNet), synthdata.py (simulator data)
engines/         the physics-engine code the scripts call (scene building,
                 rollouts, the original fit scripts)
wire_model/      library: analytical mass-spring simulator, Caporali MLP,
                 RopeDreamer, DeformNet, AdaptiGraph, PhysTwin, PhysWorld, perception
data/tracks14/   tracked cables of the 14 evaluation clips (npz: c0..c3 (T,16,2) m)
data/tracks_xplanar/  tracked cables of the 103 X-Planar clips (pre-training / synthesis)
data/*.npz|json  fitted parameters of every physics row, depth targets for Newton
checkpoints/     trained networks and PhysTwin twins behind the table numbers
patches/         the DLO-Lab per-joint patch
```

## Data and protocol

* **Clips.** 14 recordings, trimmed where a hand enters the scene
  (`approaches/common.py: CLIP_TRIMS`), 4 fps, 19-125 frames. The tracks are the
  output of the project's harness tracker; the physics scripts read them through
  `engines/mujoco_scene.track_clip`, which only re-tracks from video when no
  cached track exists (then it needs the videos in `$WIRE_VIDEO_DIR` and the
  segmentation masks in `$WIRE_SEG_DIR`; neither is in this repository).
* **Scored frames** differ by family and are stated in each script: physics
  engines and PhysTwin score every frame from frame 0; Caporali scores every
  4th frame (its 1 s steps); the RSSM and graph world models score every frame
  after a 5-frame warm-up with the measured states.
* **Fits that need the video.** Genesis (row 7) and DLO-Lab (rows 16-18) fit
  against the cable masks, so their `fit` step needs the videos and masks.
  Their `eval` runs from the shipped parameters without them.

## Environments

* MuJoCo, Newton, DLO-Lab, the world models: one Python 3.11 environment with
  `requirements.txt` (Newton and DLO-Lab need CUDA).
* Genesis: **genesis-world 1.3.3 exactly**, in its own environment (CPU torch is
  enough); export `GENESIS_PYTHON=/path/to/that/python`. Other versions order
  the links differently and give different numbers.
* DLO-Lab: `./setup_dlolab.sh` clones UMass-Embodied-AGI/DLO-Lab at `c5026a9`
  and applies `patches/dlo-lab-perjoint.patch` (per-element stiffness scales and
  rest-curvature offsets, behaviour-neutral at scale 1).

## Reproduction status

Each script's `eval` was run on this repository's data and checkpoints and
compared with the approaches table (2026-10-23; results in `results/`):

| Approach | Row | Table | This repo | |
|---|---|---|---|---|
| MuJoCo | 1 | 20.3 | 19.45 | table value predates the unified rollout function; current code gives 19.45 |
| MuJoCo | 2 / 3 / 4 | 20.3 / 11.6 / 8.5 | 20.26 / 11.62 / 8.54 | exact |
| Genesis | 5 | 20.5 | 20.47 | exact |
| Genesis | 6 | 24.4 | 18.93 | table value from an older Genesis scene; current code gives 18.93 |
| Genesis | 7 | 18.4 | 19.05 | under investigation |
| Newton | 8 / 9 / 10 / 11 | 14.1 / 8.6 / 8.1 / 7.8 | 14.08 / 8.64 / 8.07 / 7.76 | exact |
| Newton | 12 / 13 / 14 | 7.9 / 8.0 / 7.7 | 7.88 / 8.02 / 7.66 | exact |
| DLO-Lab | 15-18 | 33.9 / 18.6 / 17.0 / 16.56 | 56.7 / 44.1 / 42.8 / 30.9 | the main project's own code gives the same higher numbers today; under investigation |
| PhysTwin | world 19 | 19.8 | 19.82 | exact |
| PhysWorld | world 20 | 21.8 | 21.75 | exact (paper Bezier-only demos: 31.41, also exact) |
| DeformNet | world 21 | 30.6 | 30.63 | exact |
| RopeDreamer | world 22 | 16.8 | 16.80 | exact |
| Caporali | world 23 | 18.4 | 18.44 | exact (original data: 56.80) |
| AdaptiGraph | world 24 | 24.4 | 24.42 | exact (sim only 28.41; original data 63.86 / 26.08) |

Not part of the tables: rows 16 and 17 (DLO-Lab) were fitted on clip 20260602_170108 only.

## Approaches in one paragraph each

**MuJoCo, Genesis, Newton, DLO-Lab** simulate the measured scene with a physics
engine and fit the cable parameters (stiffness, damping, mass, friction, rest
shape -- per joint where the row says so) so that the simulated cables follow
the tracked ones. The single most effective parameter in every engine is a
per-joint rest angle (the cable's plastic set).

**PhysTwin** fits a planar spring-mass twin per clip (CMA-ES on global
parameters, then gradient descent on a stiffness per spring). **PhysWorld**
uses those twins to synthesise demonstrations and trains a graph network on
them, adapting a per-node property on the real clip.

**DeformNet** and **RopeDreamer** are Dreamer-style recurrent state-space
models learned from the tracked videos only (pre-training on 103 X-Planar
clips, fine-tuning on the 14).

**Caporali** and **AdaptiGraph** train a network on the analytical mass-spring
simulator only and adapt a few physical parameters on the real clip (online
gradient estimation of mass/stiffness/damping, or a few-shot stiffness variable
phi). `--data real` regenerates their training data from real tracked cable
states and end motions (simulator targets) instead of artificial arcs.

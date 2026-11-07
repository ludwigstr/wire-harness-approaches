# Prediction overlay videos on the 36 clips (approaches whose refit is finished)

Each video shows the camera image with the measured cables (solid) and the approach's prediction (dashed). The
label band at the top gives the clip, the time, the error of the current frame and the clip mean in mm/node.
Each folder holds one video per clip plus two stills per clip (worst frame, middle frame) and `results.json`.
`results.json` records the reproduced per-clip values. They match the table values exactly (difference 0.0).

| folder | approach | mean over the 36 clips (mm/node) | scored |
|---|---|---|---|
| `phystwin_22.0/` | PhysTwin, one spring-mass twin per clip | 22.03 | 14 inner nodes, every frame |
| `caporali_15.8/` | Caporali et al., FiLM MLP + online estimation of (m, k_b, k_d), one step per frame | 15.78 | 16 nodes, every frame incl. frame 0 |
| `adaptigraph_10.3/` | AdaptiGraph, rebuilt (3-step training as in the paper, rest-shape and end-rotation inputs, input noise), trained on simulation data only; 4 phi values, one per cable, shared by all clips and chosen on all scored frames | 10.26 | 16 nodes, frames 5..T-1 |
| `adaptigraph_29.4/` | AdaptiGraph, first implementation (one-step training, fine-tuned on the real clips, phi per cable from the first 5 frames); superseded by `adaptigraph_10.3/` | 29.42 | 16 nodes, frames 5..T-1 |
| `physworld_4.5/` | PhysWorld, MPM twin per clip, one GNN on the demos of all 36 clips (the paper uses one per scene), per-node Phi fitted on the scored clip | 4.51 | 16 nodes, frames 5..T-1 |
| `genesis_12.7/` | Genesis general fit (185 parameters), stopped by the owner after generation 33; value from a fresh-process validation | 12.68 | 16 nodes, every frame |
| `yang_16.0/` | Yang et al., interaction network + bi-LSTM trained on simulated Newton rollouts; parameters from differential evolution, stopped by the owner after 14 of 41 generations | 16.00 | 16 nodes, frames 5..T-1 |
| `newton_7.6/` | Newton per-cable fit (231 parameters, mass pinned), 30 generations; value from a fresh-process validation | 7.56 | 15 body origins vs the first 15 tracked nodes, every frame |

The values are not defined identically (nodes and frames scored differ, see the last column).
Videos for the other approaches follow when their fits are finished.

The recorder's own yellow debug text is in the export images. It shows faintly under the label band.

`newton_7.6/` differs from the other folders in two ways:

- **Base images:** the raw camera images were not available on the machine that rendered it, so the prediction is drawn
  on `check_videos/`. The measured skeleton there has its own colours (red, blue, green, magenta), and the board-to-pixel
  map was fitted per clip against those burned-in lines (mean residual under 1 px on every clip).
- **Values:** the rollouts were recomputed for the videos and do not match the table value exactly per clip, because
  the simulation is not fully deterministic. The largest clip difference is 0.16 mm/node; the mean is 7.562 in both.
  See `newton_fit_t36/results/tinygpu_2080ti/REPORT.md`.

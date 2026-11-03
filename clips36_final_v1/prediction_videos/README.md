# Prediction overlay videos on the 36 clips (approaches whose refit is finished)

Each video shows the camera image with the measured cables (solid) and the approach's prediction (dashed). The
label band at the top gives the clip, the time, the error of the current frame and the clip mean in mm/node.
Each folder holds one video per clip plus two stills per clip (worst frame, middle frame) and `results.json`.
`results.json` records the reproduced per-clip values. They match the table values exactly (difference 0.0).

| folder | approach | mean over the 36 clips (mm/node) | scored |
|---|---|---|---|
| `phystwin_22.0/` | PhysTwin, one spring-mass twin per clip | 22.03 | 14 inner nodes, every frame |
| `caporali_15.8/` | Caporali et al., FiLM MLP + online estimation of (m, k_b, k_d), one step per frame | 15.78 | 16 nodes, every frame incl. frame 0 |
| `adaptigraph_29.4/` | AdaptiGraph, simulator-trained + fine-tuned on the real clips, phi per cable from the first 5 frames | 29.42 | 16 nodes, frames 5..T-1 |

The three values are not defined identically (nodes and frames scored differ, see the last column).
Videos for the other approaches follow when their fits are finished.

The recorder's own yellow debug text is in the export images. It shows faintly under the label band.

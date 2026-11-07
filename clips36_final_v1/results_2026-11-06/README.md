# Results of 2026-11-06 on the 36 clips

Mean error over the 36 clips in mm/node. No videos yet for these runs.

| file | run | mean (mm/node) |
|---|---|---|
| `physworld_shared_phi_t36.json` | PhysWorld: pooled GNN fixed, one set of 128 Phi values shared by all 36 clips, 100 Adam iterations | 7.46 |
| `physworld_phi_first5_t36.json`, variant `twin` | PhysWorld: Phi per clip fitted on frames 0..4 only, started from the clip's own twin Phi; scored on frames 5..T-1 | 23.12 |
| `physworld_phi_first5_t36.json`, variant `mean` | PhysWorld: as above, started from the mean twin Phi | 21.66 |
| `adaptigraph_rebuilt_4phi_t36.json` | AdaptiGraph, rebuilt: 3-step training (K=3 BPTT) as in the paper, rest-shape and end-rotation inputs, input noise 0.001 m; trained on the simulation data of the 36 clips only, no fine-tuning on real transitions; 4 phi values, one per cable index, shared by all 36 clips and chosen on all scored frames (29-point grid) | 10.26 |
| `adaptigraph_rebuilt_training_log.json` | per-epoch log of that training (30 epochs, 2.0 h on one GPU) | - |
| `phystwin_corrected_stage2_t36.json` | PhysTwin, one spring-mass twin per clip (2026-11-07): stage 1 (CMA-ES, 5 global parameters + connection radius) reused from the first run, stage 2 (Adam on the dense per-spring stiffness and the globals) now minimises the rollout error over the whole clip from frame 0, the paper's Eq. 3; 100 iterations, best iterate kept. Videos in `../prediction_videos/phystwin_14.3/` | 14.32 |
| `ropedreamer_t36_S_noise_seq64_roll1000_tracks36_results.json` | RopeDreamer, trained on simulation, sequence length 64, rollout loss | 23.35 |
| `ropedreamer_t36_S_noise_seq120_roll1000_tracks36_results.json` | RopeDreamer, trained on simulation, sequence length 120, rollout loss | 20.41 |
| `ropedreamer_t36_real_finetune_tracks36_results.json` | RopeDreamer, the sequence-length-120 network fine-tuned on the 36 real clips | 5.69 |

Baselines in `physworld_shared_phi_t36.json`: own fitted Phi per clip 4.51 (the table value), own twin Phi 9.43,
mean fitted Phi 9.30, mean twin Phi 10.19.

AdaptiGraph: per-cable means 5.70, 19.87, 5.44, 10.03 mm/node; phi 0.95, 0.75, 0.80, 0.60. The earlier table value 29.42 came from the first implementation (one-step training, phi per cable from the first 5 frames, fine-tuned on real transitions); its videos are in `../prediction_videos/adaptigraph_29.4/`.

PhysTwin: the first run (22.03, `../prediction_videos/phystwin_22.0/`) ran stage 2 on random 8-frame windows started from the measured state; that lowered the window error while the full rollout got worse, so the best iterate was stage 1 on 17 of 36 clips. With the whole-clip objective, stage 2 improves every one of the 36 clips (stage 1 alone 22.80 on average). Per clip: `mittel` is the scored value, `stage1_mm` the stage-1 value. A step whose error is not finite or more than twice the best so far returns to the best iterate with the learning rate halved; this happened on 8 of 36 clips.

All values are in-sample. The PhysWorld Phi values and the network were fitted on the scored clips, except that the
first-5 variants fit Phi on frames that are not scored. The fine-tuned RopeDreamer was trained and scored on the same
clips, so 5.69 does not show how it predicts unseen clips.

The per-clip results of RopeDreamer are keyed by clip, with the value in `mittel`.

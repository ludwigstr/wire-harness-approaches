# Results of 2026-11-06 on the 36 clips

Mean error over the 36 clips in mm/node. No videos yet for these runs.

| file | run | mean (mm/node) |
|---|---|---|
| `physworld_shared_phi_t36.json` | PhysWorld: pooled GNN fixed, one set of 128 Phi values shared by all 36 clips, 100 Adam iterations | 7.46 |
| `physworld_phi_first5_t36.json`, variant `twin` | PhysWorld: Phi per clip fitted on frames 0..4 only, started from the clip's own twin Phi; scored on frames 5..T-1 | 23.12 |
| `physworld_phi_first5_t36.json`, variant `mean` | PhysWorld: as above, started from the mean twin Phi | 21.66 |
| `adaptigraph_rebuilt_4phi_t36.json` | AdaptiGraph, rebuilt: 3-step training (K=3 BPTT) as in the paper, rest-shape and end-rotation inputs, input noise 0.001 m; trained on the simulation data of the 36 clips only, no fine-tuning on real transitions; 4 phi values, one per cable index, shared by all 36 clips and chosen on all scored frames (29-point grid) | 10.26 |
| `adaptigraph_rebuilt_training_log.json` | per-epoch log of that training (30 epochs, 2.0 h on one GPU) | - |
| `ropedreamer_t36_S_noise_seq64_roll1000_tracks36_results.json` | RopeDreamer, trained on simulation, sequence length 64, rollout loss | 23.35 |
| `ropedreamer_t36_S_noise_seq120_roll1000_tracks36_results.json` | RopeDreamer, trained on simulation, sequence length 120, rollout loss | 20.41 |
| `ropedreamer_t36_real_finetune_tracks36_results.json` | RopeDreamer, the sequence-length-120 network fine-tuned on the 36 real clips | 5.69 |

Baselines in `physworld_shared_phi_t36.json`: own fitted Phi per clip 4.51 (the table value), own twin Phi 9.43,
mean fitted Phi 9.30, mean twin Phi 10.19.

AdaptiGraph: per-cable means 5.70, 19.87, 5.44, 10.03 mm/node; phi 0.95, 0.75, 0.80, 0.60. The earlier table value 29.42 came from the first implementation (one-step training, phi per cable from the first 5 frames, fine-tuned on real transitions); its videos are in `../prediction_videos/adaptigraph_29.4/`.

All values are in-sample. The PhysWorld Phi values and the network were fitted on the scored clips, except that the
first-5 variants fit Phi on frames that are not scored. The fine-tuned RopeDreamer was trained and scored on the same
clips, so 5.69 does not show how it predicts unseen clips.

The per-clip results of RopeDreamer are keyed by clip, with the value in `mittel`.

# Results of 2026-11-06 on the 36 clips

Mean error over the 36 clips in mm/node. No videos yet for these runs.

| file | run | mean (mm/node) |
|---|---|---|
| `physworld_shared_phi_t36.json` | PhysWorld: pooled GNN fixed, one set of 128 Phi values shared by all 36 clips, 100 Adam iterations | 7.46 |
| `physworld_phi_first5_t36.json`, variant `twin` | PhysWorld: Phi per clip fitted on frames 0..4 only, started from the clip's own twin Phi; scored on frames 5..T-1 | 23.12 |
| `physworld_phi_first5_t36.json`, variant `mean` | PhysWorld: as above, started from the mean twin Phi | 21.66 |
| `ropedreamer_t36_S_noise_seq64_roll1000_tracks36_results.json` | RopeDreamer, trained on simulation, sequence length 64, rollout loss | 23.35 |
| `ropedreamer_t36_S_noise_seq120_roll1000_tracks36_results.json` | RopeDreamer, trained on simulation, sequence length 120, rollout loss | 20.41 |
| `ropedreamer_t36_real_finetune_tracks36_results.json` | RopeDreamer, the sequence-length-120 network fine-tuned on the 36 real clips | 5.69 |

Baselines in `physworld_shared_phi_t36.json`: own fitted Phi per clip 4.51 (the table value), own twin Phi 9.43,
mean fitted Phi 9.30, mean twin Phi 10.19.

All values are in-sample. The PhysWorld Phi values and the network were fitted on the scored clips, except that the
first-5 variants fit Phi on frames that are not scored. The fine-tuned RopeDreamer was trained and scored on the same
clips, so 5.69 does not show how it predicts unseen clips.

The per-clip results of RopeDreamer are keyed by clip, with the value in `mittel`.

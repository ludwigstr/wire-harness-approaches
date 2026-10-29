#!/bin/bash
# RopeDreamer on the twins14 data, retrained with 14-frame windows (user, 2026-10-28 night: "Ja").
# Bug in the seq_len-32 run (data/ropedreamer_twins14_S_noise.pt): WindowDataset drops trajectories shorter than
# seq_len -> 0 % of clip 102713 (19 frames) and 18 % of 170108 were ever trained on (102713: 146 mm real).
# The shortest trajectory has 14 frames, so --seq-len 14 keeps 100 %. Everything else as run_ropedreamer_twins14_noise_cont.sh.
cd .
log() { echo "[$(date '+%F %T')] $*" >> data/wm/ropedreamer_twins14_seq14.log; }
log "training start"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_train.py --synth data/wm/ropedreamer_twins14_synth --size S --seq-len 14 \
    --beta 1.0 --pos-weight 0 --rollout-weight 0 --lr 1e-4 --batch 32 --val-frac 0.1 --val-by-rollout --gap-ref-tol 0.010 \
    --obs-noise data/wm/ropedreamer_twins14_noise_params_v3.json --loader-workers 8 \
    --select val_loss --out data/ropedreamer_twins14_S_noise_seq14.pt > data/ropedreamer_twins14_S_noise_seq14_train.log 2>&1 || { log "training failed"; exit 1; }
log "training done"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_eval.py --ckpt data/ropedreamer_twins14_S_noise_seq14.pt --tracks data/tracks14 \
    --out data/ropedreamer_twins14_S_noise_seq14_tracks14_results.json > data/ropedreamer_twins14_S_noise_seq14_eval.log 2>&1 || { log "eval failed"; exit 1; }
log "eval real 14: $(grep -o 'Mittel.*' data/ropedreamer_twins14_S_noise_seq14_eval.log)"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_eval.py --ckpt data/ropedreamer_twins14_S_noise_seq14.pt --tracks data/wm/ropedreamer_twins14_twin_tracks \
    --out data/ropedreamer_twins14_S_noise_seq14_twins14_results.json > data/ropedreamer_twins14_S_noise_seq14_eval_twins.log 2>&1 || { log "eval (twins) failed"; exit 1; }
log "eval clean twins: $(grep -o 'Mittel.*' data/ropedreamer_twins14_S_noise_seq14_eval_twins.log)"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_video.py --ckpt data/ropedreamer_twins14_S_noise_seq14.pt --tracks data/tracks14 \
    --video-dir data/WirePrediction \
    --truth-label "solid = measured, dashed = RopeDreamer (trained on MuJoCo sims, 14-frame windows)" \
    --out data/ropedreamer_video/twins14_seq14_real > data/ropedreamer_twins14_S_noise_seq14_video.log 2>&1 || { log "video failed"; exit 1; }
log "ALL DONE"

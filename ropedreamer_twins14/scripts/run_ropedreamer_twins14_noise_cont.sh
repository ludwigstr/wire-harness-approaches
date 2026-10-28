#!/bin/bash
# Continuation of run_ropedreamer_twins14.sh (user, 2026-10-28): its shell was stopped by PID before
# training so that training uses the tracking-noise augmentation ("Das mit dem Zittern gerne mit rein"):
# --obs-noise data/wm/ropedreamer_twins14_noise_params_v3.json (jitter + end straightening/extension +
# smooth shape gap + spatial smoothing, calibrated on the 14 real tracks vs their twins; the posterior
# sees noisy states and noisy end actions, the paper's losses stay on the clean sim states).
# Waits for the running dataset step (PID $1). seq_len 32 kept (user): 102713 and the short 170108
# rollouts are skipped by the window loader. Outputs data/ropedreamer_twins14_S_noise*. GPU 2.
cd .
log() { echo "[$(date '+%F %T')] $*"; }
while kill -0 "$1" 2>/dev/null; do sleep 60; done
grep -q YANG_DATASET_DONE data/wm/ropedreamer_twins14_synth.log || { log "dataset failed"; exit 1; }
log "dataset done: $(ls data/wm/ropedreamer_twins14_synth/*.npz | wc -l) rollouts, $(grep -c REJECTED data/wm/ropedreamer_twins14_synth.log) rejected"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_train.py --synth data/wm/ropedreamer_twins14_synth --size S \
    --beta 1.0 --pos-weight 0 --rollout-weight 0 --lr 1e-4 --batch 32 --val-frac 0.1 --val-by-rollout --gap-ref-tol 0.010 \
    --obs-noise data/wm/ropedreamer_twins14_noise_params_v3.json --loader-workers 8 \
    --select val_loss --out data/ropedreamer_twins14_S_noise.pt > data/ropedreamer_twins14_S_noise_train.log 2>&1 || { log "training failed"; exit 1; }
log "training done: $(grep -m1 load_synth data/ropedreamer_twins14_S_noise_train.log)"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_eval.py --ckpt data/ropedreamer_twins14_S_noise.pt --tracks data/tracks14 \
    --out data/ropedreamer_twins14_S_noise_tracks14_results.json > data/ropedreamer_twins14_S_noise_eval.log 2>&1 || { log "eval failed"; exit 1; }
log "eval real 14: $(grep -o 'Mittel.*' data/ropedreamer_twins14_S_noise_eval.log)"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_eval.py --ckpt data/ropedreamer_twins14_S_noise.pt --tracks data/wm/ropedreamer_twins14_twin_tracks \
    --out data/ropedreamer_twins14_S_noise_twins14_results.json > data/ropedreamer_twins14_S_noise_eval_twins.log 2>&1 || { log "eval (twins) failed"; exit 1; }
log "eval clean twins: $(grep -o 'Mittel.*' data/ropedreamer_twins14_S_noise_eval_twins.log)"
log "ALL DONE"

#!/bin/bash
# RopeDreamer trained on variations AROUND the 14 evaluation clips (user, 2026-10-28:
# "Nimm die 14 Videos und fuehre sie in MuJoCo zurueck. Dann variiere rund um diese 14
# Simulationen."). EXPLICIT user decision: the evaluation clips' mover paths are in the
# training data (--allow-eval-clips, logged in config.json).
#   dataset: MuJoCo general fit (data/mujoco_general_fit3.npz), per clip ceil(18000 / T) rollouts
#            (every clip contributes ~18k frames, ~1M cable transitions in total); rollout 000 = the
#            EXACT twin (fit parameters, real timing; verified == the fit's own eval to 0.000 mm,
#            data/wm/ropedreamer_twins14_twins.json); rollouts > 0: shared speed warp x0.65-1.4 +
#            parameter randomization, constraints/contacts tolerated as far as the clip's twin shows them
#   training: paper settings (S, beta 1, pos/rollout weight 0, lr 1e-4, batch 32, early stop 10),
#            10 % of the ROLLOUTS held out (--val-by-rollout), tolerant gap cut (twin + 10 mm). GPU 2.
#   eval: the 14 real clips.
cd .
log() { echo "[$(date '+%F %T')] $*"; }
OMP_NUM_THREADS=1 python3 scripts/yang_make_dataset.py --env general --randomize --motion speed --tracks data/tracks14 \
    --allow-eval-clips --frames-per-clip 18000 --twin-reference --workers 16 \
    --out data/wm/ropedreamer_twins14_synth > data/wm/ropedreamer_twins14_synth.log 2>&1 || { log "dataset failed"; exit 1; }
grep -q YANG_DATASET_DONE data/wm/ropedreamer_twins14_synth.log || { log "dataset failed"; exit 1; }
log "dataset done: $(ls data/wm/ropedreamer_twins14_synth/*.npz | wc -l) rollouts, $(grep -c REJECTED data/wm/ropedreamer_twins14_synth.log) rejected"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_train.py --synth data/wm/ropedreamer_twins14_synth --size S \
    --beta 1.0 --pos-weight 0 --rollout-weight 0 --lr 1e-4 --batch 32 --val-frac 0.1 --val-by-rollout --gap-ref-tol 0.010 \
    --select val_loss --out data/ropedreamer_twins14_S.pt > data/ropedreamer_twins14_S_train.log 2>&1 || { log "training failed"; exit 1; }
log "training done: $(grep -m1 load_synth data/ropedreamer_twins14_S_train.log)"
CUDA_VISIBLE_DEVICES=2 python3 scripts/ropedreamer_eval.py --ckpt data/ropedreamer_twins14_S.pt --tracks data/tracks14 \
    --out data/ropedreamer_twins14_S_tracks14_results.json > data/ropedreamer_twins14_S_eval.log 2>&1 || { log "eval failed"; exit 1; }
log "eval done: $(grep Mittel data/ropedreamer_twins14_S_eval.log)"
log "ALL DONE"

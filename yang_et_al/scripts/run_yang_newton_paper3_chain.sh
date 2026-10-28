#!/bin/bash
# Yang et al. on Newton, third run (user, 2026-10-28: "Warum wurde nur auf 3 Clips gefittet???"):
#   DE identification on ALL 14 clips (734 transitions) instead of the 3 ID_CLIPS, like every other row;
#   1. DE identification with WIDENED bounds -- the first run (yang_newton_paper_ident.json) ended
#      on two bounds (k_stretch 10^5.99 of 6, d_bend 10^-0.0005 of 0); a boundary optimum means the
#      range was too small: k_stretch up to 1e8, d_bend up to 10; everything else unchanged
#   2. simulator-alone control on the 14 clips
#   3. synthetic dataset: SPEED-ONLY variation (shared warp x0.65-1.4, hard constraints, --motion speed)
#      + domain randomization (--randomize), X-Planar starts, never the 14 clips
#   4. IN-biLSTM training (60 epochs), 5. evaluation on the 14 clips
# Outputs data/wm/yang_newton_paper3_*. GPU 1 only.
cd .
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
W=data/wm
P=$W/yang_newton_paper3
log() { echo "[$(date '+%F %T')] $*"; }

log "identification start"
python3 scripts/yang_identify.py --engine newton --maxiter 40 --popsize 15 --tol 0.01 \
    --bound k_stretch=1,8 --bound d_bend=-6,1 --clips 20260602_102713,20260602_114019,20260602_121006,20260602_134613,20260602_141210,20260602_162650,20260602_165113,20260602_170108,20260602_172435,20260602_175847,20260602_183620,20260602_193329,20260602_193953,20260605_112939 --out ${P}_ident.json > ${P}_ident.log 2>&1 || { log "identification failed"; exit 1; }
log "identification done: $(tail -n 1 ${P}_ident.log)"

python3 scripts/yang_sim_control.py --engine newton --ident ${P}_ident.json \
    --out ${P}_sim_control_14.json > ${P}_sim_control_14.log 2>&1 || { log "sim control failed"; exit 1; }
log "sim control done: $(tail -n 1 ${P}_sim_control_14.log)"

python3 scripts/yang_make_dataset.py --engine newton --env yang --ident ${P}_ident.json --motion speed --randomize \
    --per-clip 8 --batch 128 --out ${P}_synth > ${P}_synth.log 2>&1 || { log "dataset failed"; exit 1; }
grep -q YANG_DATASET_DONE ${P}_synth.log || { log "dataset failed"; exit 1; }
log "dataset done: $(ls ${P}_synth/*.npz | wc -l) rollouts; $(grep 'newton speed done' ${P}_synth.log)"

python3 scripts/yang_train.py --data ${P}_synth --out ${P}_inbilstm.pt --epochs 60 \
    > ${P}_train.log 2>&1 || { log "training failed"; exit 1; }
log "training done: $(grep -m1 'transitions' ${P}_train.log)"

python3 scripts/yang_eval.py --ckpt ${P}_inbilstm.pt --out ${P}_inbilstm_14.json --synth ${P}_synth \
    > ${P}_eval_14.log 2>&1 || { log "eval failed"; exit 1; }
log "eval done: $(tail -n 2 ${P}_eval_14.log | tr '\n' ' ')"
log "ALL DONE"

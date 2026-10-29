#!/bin/bash
# Yang paper3 continuation (2026-10-29): run_yang_newton_paper3_chain.sh trained WITHOUT --val-by-clip and
# --gap-ref-tol 0.010, unlike the table's paper2 protocol (run_yang_newton_paper2_cont2.sh); with the soft
# identified stretch (k 4367 N/m) the plain 15 mm gap filter dropped 71.9% of the transitions. Same dataset,
# same protocol as paper2_cont2. Outputs data/wm/yang_newton_paper3b_*. GPU 2.
cd .
export CUDA_VISIBLE_DEVICES=2 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
W=data/wm; S=$W/yang_newton_paper3_synth; P=$W/yang_newton_paper3b
log() { echo "[$(date '+%F %T')] $*"; }
python3 scripts/yang_train.py --data $S --out ${P}_inbilstm.pt --epochs 60 --val-by-clip --gap-ref-tol 0.010 \
    > ${P}_train.log 2>&1 || { log "training failed"; exit 1; }
log "training done: $(grep -m1 'transitions' ${P}_train.log)"
python3 scripts/yang_eval.py --ckpt ${P}_inbilstm.pt --out ${P}_inbilstm_14.json --synth $S \
    > ${P}_eval_14.log 2>&1 || { log "eval failed"; exit 1; }
log "eval done: $(tail -n 2 ${P}_eval_14.log | tr '\n' ' ')"
log "ALL DONE"

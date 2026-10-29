#!/bin/bash
# Yang et al. (summary row 4) with the synthetic data built from the 14 CLIPS (user, 2026-10-29: "Es muessen
# natuerlich die 14 Videos sein!!!" / "Ja, alles korrigieren") instead of the 103 X-Planar clips.
# Identification unchanged (yang_newton_paper3_ident.json, DE on all 14 clips). Dataset: the 14 clips' real
# mover paths, speed-only variation + domain randomization, 100 rollouts per clip (rollout 00 = real timing;
# seeds i*101+j stay distinct for j < 101). Training: tolerant gap filter as paper3b, but a ROLLOUT-level
# validation split (a clip-level split would hold out whole evaluation clips; same as the RopeDreamer twins14 set).
# Outputs data/wm/yang_newton_c14_*. GPU 1.
cd .
export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1
W=data/wm; P=$W/yang_newton_c14
log() { echo "[$(date '+%F %T')] $*"; }
python3 scripts/yang_make_dataset.py --engine newton --env yang --ident $W/yang_newton_paper3_ident.json --motion speed --randomize \
    --tracks data/tracks14 --allow-eval-clips --per-clip 100 --batch 128 --out ${P}_synth > ${P}_synth.log 2>&1 || { log "dataset failed"; exit 1; }
grep -q YANG_DATASET_DONE ${P}_synth.log || { log "dataset failed"; exit 1; }
log "dataset done: $(ls ${P}_synth/*.npz | wc -l) rollouts; $(grep 'newton speed done' ${P}_synth.log)"
python3 scripts/yang_train.py --data ${P}_synth --out ${P}_inbilstm.pt --epochs 60 --gap-ref-tol 0.010 \
    > ${P}_train.log 2>&1 || { log "training failed"; exit 1; }
log "training done: $(grep -m1 'transitions' ${P}_train.log)"
python3 scripts/yang_eval.py --ckpt ${P}_inbilstm.pt --out ${P}_inbilstm_14.json --synth ${P}_synth \
    > ${P}_eval_14.log 2>&1 || { log "eval failed"; exit 1; }
log "eval done: $(tail -n 2 ${P}_eval_14.log | tr '\n' ' ')"
log "ALL DONE"

#!/bin/bash
# Newton per-cable fit (summary row 9, mass pinned 0.044 kg/m, 231 params, CMA-ES popsize 18) on the 36-clip
# selection data/tracks36 -- the same call as scripts/run_refit_t36_newton.sh of the main project, with paths relative
# to this folder. See HANDOFF.md.
#   fresh start (gen 0 = warm start scored on the 36 clips):   bash run_newton_t36.sh
#   continue from a state snapshot copied into data/:         RESUME=1 bash run_newton_t36.sh
# Env: NEWTON_GPUS (warp device indices under CUDA_VISIBLE_DEVICES, one process pool per entry, e.g. 0,0,1),
#      NEWTON_WORKERS (total worker processes, split evenly over the pools), STOP_BY ("YYYY-mm-dd HH:MM" local time:
#      no generation is started that would end after it; default 2026-11-09 19:00), GENS (default 30).
set -u
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
P=data/newton_fit_everything_pinned_t36
log() { echo "[$(date '+%F %T')] $*"; }
if [ "${RESUME:-0}" = "1" ]; then
  [ -e $P.npz ] || { log "RESUME=1 but $P.npz is missing (copy a snapshot from state/ into data/)"; exit 1; }
else
  [ -e $P.npz ] && { log "$P.npz exists -- refusing to overwrite (RESUME=1 to continue it)"; exit 1; }
fi
log "fit start"
WIRE_TRACKS=data/tracks36 WIRE_Z_WEIGHT=0 WIRE_MASS_PIN=0.044 \
WIRE_WARM=data/newton_fit_everything_pinned.npz WIRE_WARM_FULL=1 WIRE_RESUME=${RESUME:-0} \
WIRE_OUT=$P.npz WIRE_FITLOG=${P}_log.jsonl WIRE_GPUS=${NEWTON_GPUS:-0} WIRE_WORKERS=${NEWTON_WORKERS:-4} \
WIRE_POPSIZE=18 WIRE_GENS=${GENS:-30} WIRE_STOP_BY="${STOP_BY:-2026-11-09 19:00}" \
    python3 scripts/newton_fit_everything_cmaes.py >> ${P}_stdout.log 2>&1 || { log "fit failed (see ${P}_stdout.log)"; exit 1; }
grep -q "=== DONE ===" ${P}_stdout.log || { log "fit did not finish"; exit 1; }
log "fit done: $(grep 'final result' ${P}_stdout.log | tail -n 1)"
python3 scripts/newton_validate_tracks.py --params $P.npz --tracks data/tracks36 --out ${P}_validate.json \
    --gpus ${VALIDATE_GPUS:-0} --workers ${NEWTON_WORKERS:-4} > ${P}_validate.log 2>&1 || { log "validation failed"; exit 1; }
log "fresh-process validation: $(tail -n 1 ${P}_validate.log)"
log "ALL DONE"

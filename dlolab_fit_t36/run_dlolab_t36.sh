#!/bin/bash
# DLO-Lab general fit (summary row 10, 293 params, CMA-ES popsize 20, CPU) on the 36-clip selection data/tracks36,
# continued from the main machine's best state (generation 21, 17.012 mm/node). Same call as
# scripts/run_refit_t36_dlolab.sh of the main project, with paths relative to this folder. See HANDOFF.md.
# Env: DLOLAB_WORKERS (default 12), SIGMA0 (default 0.2), STOP_BY (default "2026-11-09 19:00").
set -euo pipefail
cd "$(dirname "$0")"
W=${DLOLAB_WORKERS:-12}; SIGMA0=${SIGMA0:-0.2}; STOP_BY=${STOP_BY:-2026-11-09 19:00}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" PYTHONUNBUFFERED=1
P=data/dlolab_general_fit_t36
for f in $P.json ${P}_log.jsonl ${P}_validate.json; do
  [ -e "$f" ] && { echo "refusing to overwrite $f"; exit 1; }
done
echo "[$(date '+%F %T')] fit (start = state/dlolab_general_fit_t36_gen21.json, sigma0 $SIGMA0, $W workers, stop by $STOP_BY)"
python3 scripts/dlolab_general_fit_cmaes.py --tracks data/tracks36 --mass-max 0.5 --popsize 20 --workers "$W" \
  --resume state/dlolab_general_fit_t36_gen21.json --sigma0 "$SIGMA0" --gens 1000 --stop-by "$STOP_BY" \
  --out $P.json --log ${P}_log.jsonl
echo "[$(date '+%F %T')] fresh-process validation"
python3 scripts/general_fit_validate.py dlolab $P.json --tracks data/tracks36 --out ${P}_validate.json --workers "$W"
echo "[$(date '+%F %T')] CHAIN DONE"

#!/bin/bash
# bound check: Phi trust box widened from +-1.5 to +-2.5 (log units), rollout objective, same GNNs (evaluation only)
cd .
for f in data/wm/physworld_mpm_gnn/*.pt; do
  c=$(basename $f .pt)
  CUDA_VISIBLE_DEVICES=2 python3 scripts/physworld_mpm_gnn.py --clip $c --load $f --phi-box 2.5 --tag box25 > data/wm/physworld_mpm_logs/gnn_box25_$c.log 2>&1
done
echo BOX25_DONE

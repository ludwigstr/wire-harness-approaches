#!/bin/bash
# control variant: Phi fine-tuned on the real ONE-STEP transitions (unbounded, 200 iterations: the first
# implementation), same trained GNNs (evaluation only)
cd .
for f in data/wm/physworld_mpm_gnn/*.pt; do
  c=$(basename $f .pt)
  CUDA_VISIBLE_DEVICES=1 python3 scripts/physworld_mpm_gnn.py --clip $c --load $f --phi-objective onestep --phi-box 0 --phi-iters 200 --tag onestep > data/wm/physworld_mpm_logs/gnn_onestep_$c.log 2>&1
done
echo ONESTEP_DONE

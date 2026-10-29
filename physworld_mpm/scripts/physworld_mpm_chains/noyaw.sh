#!/bin/bash
# control variant: GNN without the module yaw in the action (action = end-node displacement only, as the old
# spring-mass PhysWorld row); same demos, same Phi fine-tuning and evaluation. Waits for each clip's demos.
cd .
for clip in "$@"; do
  until [ -f data/wm/physworld_mpm_gnn/$clip.json ]; do sleep 60; done
  CUDA_VISIBLE_DEVICES=1 python3 scripts/physworld_mpm_gnn.py --clip $clip --no-yaw --tag noyaw > data/wm/physworld_mpm_logs/gnn_noyaw_$clip.log 2>&1
  echo "$clip noyaw $(date +%H:%M)"
done
echo NOYAW_DONE

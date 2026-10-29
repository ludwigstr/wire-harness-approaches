#!/bin/bash
# usage: gnn_after.sh <pid> <gpu> <clip>: wait for the clip's demo generation (PID) to exit, then GNN + evaluation
cd .
while kill -0 $1 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=$2 python3 scripts/physworld_mpm_gnn.py --clip $3 > data/wm/physworld_mpm_logs/gnn_$3.log 2>&1
echo "$3 DONE $(date +%H:%M)"

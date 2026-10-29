#!/bin/bash
# continuation of stream1 (its chain was stopped so that 20260602_102713 can run in parallel): wait for the
# 134613 demo generation (PID given) to finish, then train the GNN
cd .
while kill -0 $1 2>/dev/null; do sleep 30; done
CUDA_VISIBLE_DEVICES=2 python3 scripts/physworld_mpm_gnn.py --clip 20260602_134613 > data/wm/physworld_mpm_logs/gnn_20260602_134613.log 2>&1
echo "20260602_134613 DONE $(date +%H:%M)"

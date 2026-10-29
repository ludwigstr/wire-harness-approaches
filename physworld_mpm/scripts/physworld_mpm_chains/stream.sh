#!/bin/bash
# usage: stream.sh <gpu> <clip> [<clip> ...]: per clip twin fit (elastoplastic, global 10 + local 6),
# 40 demonstrations, GNN + Phi fine-tuning + evaluation
cd .
GPU=$1; shift
L=data/wm/physworld_mpm_logs
for clip in "$@"; do
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/physworld_mpm_fit.py --clip $clip --model elastoplastic --global-iters 10 --local-iters 6 --frac 0.33 > $L/fit_${clip}.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/physworld_mpm_demos.py --clip $clip --n 40 --batch 4 > $L/demos_${clip}.log 2>&1
  CUDA_VISIBLE_DEVICES=$GPU python3 scripts/physworld_mpm_gnn.py --clip $clip > $L/gnn_${clip}.log 2>&1
  echo "$clip DONE $(date +%H:%M)"
done
echo STREAM_DONE

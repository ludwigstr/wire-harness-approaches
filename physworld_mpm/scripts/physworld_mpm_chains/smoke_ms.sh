#!/bin/bash
cd .
L=data/wm/physworld_mpm_logs
export CUDA_VISIBLE_DEVICES=2
python3 scripts/physworld_mpm_fit.py --clip 20260602_102713 --global-iters 2 --local-iters 1 --tag smokems > $L/smokems_fit.log 2>&1 && \
python3 scripts/physworld_mpm_demos.py --clip 20260602_102713 --twin elastoplastic_smokems --n 4 --batch 4 --out-tag _smokems > $L/smokems_demos.log 2>&1 && \
python3 scripts/physworld_mpm_gnn.py --clip 20260602_102713 --demos-tag _smokems --epochs 3 --phi-iters 10 --tag smokems > $L/smokems_gnn.log 2>&1
echo SMOKE_DONE $?

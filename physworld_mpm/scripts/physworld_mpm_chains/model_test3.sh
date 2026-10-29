#!/bin/bash
# elastic vs elastoplastic (global stage only) on two clips, GPU 1
cd .
for clip in 20260602_170108 20260602_114019; do
  for model in elastic elastoplastic; do
    CUDA_VISIBLE_DEVICES=1 python3 scripts/physworld_mpm_fit.py --clip $clip --model $model --global-iters 15 --local-iters 0 --frac 0.5 --tag modeltest3 \
      > data/wm/physworld_mpm_logs/modeltest3_${clip}_${model}.log 2>&1 &
  done
done
wait
echo MODELTEST_DONE

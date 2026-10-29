#!/bin/bash
# elastic (Neo-Hookean, no yield) global stage on two clips, to compare with the elastoplastic production twins' global stage
cd .
L=data/wm/physworld_mpm_logs
for clip in 20260602_170108 20260602_114019; do
  CUDA_VISIBLE_DEVICES=$1 python3 scripts/physworld_mpm_fit.py --clip $clip --model elastic --global-iters 10 --local-iters 0 --frac 0.33 --tag elastictest > $L/elastictest_${clip}.log 2>&1
done
echo ELASTIC_TEST_DONE

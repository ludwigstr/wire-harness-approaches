#!/bin/bash
# Caporali (summary row 5) per VIDEO FRAME instead of per 1 s (user, 2026-10-28: "Ich will Caporali per
# Step!!!! Diese Sekunde ist doch arbitraer!!!"). Same protocol as scripts/run_real_anchored.sh (caporali,
# frame0 rest), except: one transition = one 0.25 s frame (1 s move/settle protocol scaled by 1/4), ends follow
# the real X-Planar end paths with speed-only variation x0.8-1.2 (no spatial scaling), evaluation and video
# with --stride 1 on the cached data/tracks14 tracks of all 14 clips.
cd .; export OMP_NUM_THREADS=3 CUDA_VISIBLE_DEVICES=2
L=data/wm/caporali_step1.log
echo "[$(date +%H:%M:%S)] dataset" >> $L
python3 scripts/make_dataset_real.py --mode step --stride 1 --speed 0.8 1.2 --chains 60000 --batch 10000 \
  --out data/dlo_dataset_real_step1.pt > data/wm/caporali_step1_dataset.log 2>&1 || exit 1
echo "[$(date +%H:%M:%S)] training" >> $L
python3 scripts/bench_models.py --models film --data data/dlo_dataset_real_step1.pt --width 256 --epochs 25 --save \
  --save-as data/bench_film_real_step1.pt --out data/wm/caporali_step1_bench.json > data/wm/caporali_step1_train.log 2>&1 || exit 1
echo "[$(date +%H:%M:%S)] evaluation" >> $L
export WIRE_NOMINAL="0.04,1.0,20,0.15" WIRE_REST_MODE=frame0 WIRE_REST_CAP=0.6
rm -f data/neural_video_caporali_step1.json
for f in data/tracks14/*.npz; do
  clip=$(basename $f .npz)
  python3 scripts/render_neural_video.py --clip $clip --variant online --arch film --in-sample \
    --base-ckpt data/bench_film_real_step1.pt --epochs 0 --stride 1 --tracks-npz $f \
    --out data/neural_video_caporali_step1 --results-json data/neural_video_caporali_step1.json 2>&1 \
    | grep -E "online_film\]|Traceback|Error" >> $L
done
python3 -c "
import json, numpy as np
r = json.load(open('data/neural_video_caporali_step1.json')); v = np.array([x['mittel'] for x in r])
print(f'Caporali per frame (stride 1): n={len(v)} mean {v.mean():.2f} median {np.median(v):.2f}')" >> $L
echo "[$(date +%H:%M:%S)] === CAPORALI_STEP1_FERTIG ===" >> $L

#!/bin/bash
# Caporali (summary row 5) and AdaptiGraph (row 6) with synthetic training data built from the 14 CLIPS
# (user, 2026-10-29: "Es muessen natuerlich die 14 Videos sein!!!") instead of the 103 X-Planar clips.
# Same simulator and parameter ranges as before; chains start from real frames of the 14 clips and follow
# their real end paths with speed-only variation x0.8-1.2 (no spatial scaling, both rows).
# --chain-len 10 so the 19-frame clip 102713 qualifies (12 frames x 1.2 + 5 > 19 would drop it).
# AdaptiGraph then gets the same real fine-tuning as before (2712 transitions of the 14 clips, 60 epochs).
# Outputs *_c14*. GPUs 1 and 2.
cd .; export OMP_NUM_THREADS=3
TR="--tracks data/tracks14 --chain-len 10 --speed 0.8 1.2"
caporali() {
  export CUDA_VISIBLE_DEVICES=1; L=data/wm/caporali_c14.log
  echo "[$(date +%H:%M:%S)] dataset" >> $L
  python3 scripts/make_dataset_real.py --mode step --stride 1 $TR --chains 60000 --batch 10000 \
    --out data/dlo_dataset_c14_step1.pt > data/wm/caporali_c14_dataset.log 2>&1 || { echo "dataset failed" >> $L; return; }
  echo "[$(date +%H:%M:%S)] training" >> $L
  python3 scripts/bench_models.py --models film --data data/dlo_dataset_c14_step1.pt --width 256 --epochs 25 --save \
    --save-as data/bench_film_c14_step1.pt --out data/wm/caporali_c14_bench.json > data/wm/caporali_c14_train.log 2>&1 || { echo "training failed" >> $L; return; }
  echo "[$(date +%H:%M:%S)] evaluation" >> $L
  export WIRE_NOMINAL="0.04,1.0,20,0.15" WIRE_REST_MODE=frame0 WIRE_REST_CAP=0.6
  for f in data/tracks14/*.npz; do
    clip=$(basename $f .npz)
    python3 scripts/render_neural_video.py --clip $clip --variant online --arch film --in-sample \
      --base-ckpt data/bench_film_c14_step1.pt --epochs 0 --stride 1 --tracks-npz $f \
      --out data/neural_video_caporali_c14 --results-json data/neural_video_caporali_c14.json 2>&1 \
      | grep -E "online_film\]|Traceback|Error" >> $L
  done
  python3 -c "
import json, numpy as np
r = json.load(open('data/neural_video_caporali_c14.json')); v = np.array([x['mittel'] for x in r])
print(f'Caporali, data from the 14 clips: n={len(v)} mean {v.mean():.2f} median {np.median(v):.2f}')" >> $L
  echo "[$(date +%H:%M:%S)] === CAPORALI_C14_FERTIG ===" >> $L
}
adaptigraph() {
  export CUDA_VISIBLE_DEVICES=2; L=data/wm/adaptigraph_c14_chain.log
  echo "[$(date +%H:%M:%S)] dataset" >> $L
  python3 scripts/make_dataset_real.py --mode seq $TR --chains 30000 --batch 10000 \
    --out data/dlo_dataset_c14_seq.pt > data/wm/adaptigraph_c14_dataset.log 2>&1 || { echo "dataset failed" >> $L; return; }
  echo "[$(date +%H:%M:%S)] sim training" >> $L
  python3 scripts/adaptigraph_train.py --data data/dlo_dataset_c14_seq.pt --kb-range 0.2 3.0 --epochs 20 \
    --out data/wm/adaptigraph_c14.pt > data/wm/adaptigraph_c14.log 2>&1 || { echo "training failed" >> $L; return; }
  echo "[$(date +%H:%M:%S)] real fine-tuning" >> $L
  python3 scripts/adaptigraph_train.py --real --init data/wm/adaptigraph_c14.pt --epochs 60 \
    --out data/wm/adaptigraph_c14_ft.pt > data/wm/adaptigraph_c14_ft.log 2>&1 || { echo "fine-tuning failed" >> $L; return; }
  echo "[$(date +%H:%M:%S)] evaluation" >> $L
  for ck in c14 c14_ft; do
    python3 scripts/adaptigraph_eval.py --ckpt data/wm/adaptigraph_$ck.pt --tracks data/tracks14 \
      --out data/wm/adaptigraph_${ck}_14_results.json >> data/wm/adaptigraph_c14_eval.log 2>&1
  done
  tail -n 4 data/wm/adaptigraph_c14_eval.log >> $L
  echo "[$(date +%H:%M:%S)] === ADAPTIGRAPH_C14_FERTIG ===" >> $L
}
caporali & adaptigraph & wait

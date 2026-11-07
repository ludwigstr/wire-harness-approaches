# Newton fit on the 36 clips: run `tinygpu_2080ti`

Independent fresh run following `newton_fit_t36/HANDOFF.md` (231 parameters, mass pinned at 0.044 kg/m, CMA-ES popsize 18,
10 VBD iterations, all 36 clips of `data/tracks36`). The fit settings were not changed; only throughput settings differ.

## Result

| | mean (mm/node) | median (mm/node) |
|---|---:|---:|
| Fresh-process validation (`_validate.json`) | **7.562** | **6.528** |
| Final in-script result (`.npz`) | 7.699 | 6.515 |
| Generation 0 (warm start) | 10.186 | 8.728 |

All values are in-sample: the validation uses the same 36 clips as the fit. It tests reproducibility, not generalisation.

**Open point: clip `20261028_105803` does not reproduce.** It scored 15.358 in the in-script evaluation and 10.162 in the
fresh-process validation, a difference of 5.196 mm/node. This one clip causes nearly the whole gap between the
two means. Over the other 35 clips the mean difference is +0.0078 and the median absolute difference
0.0031 mm/node. The cause has not been investigated.

## Run

- **Hardware:** 4 x NVIDIA RTX 2080 Ti on the TinyGPU cluster (node tg06b, then tg06a)
- **Workers:** 28 (`NEWTON_GPUS=0,1,2,3`, `NEWTON_WORKERS=28`, `VALIDATE_GPUS=0,1,2,3`)
- **Generations:** 30 of 30, plus generation 0
- **Fit time:** 102.4 h, from start (2026-11-02 14:10) to the end of generation 30 (2026-11-06 20:36), without validations
- **Best parameters:** best_ever 7.565; K 442.6 N/m,
  ke 0.0021..1.348, kt 0.0012..1.178,
  kd 8.27e-06..0.085, rest -33.9..+44.1 deg,
  shape_mu 0.346, cable_mu 0.431, mu_tors 0.0450, mu_roll 1.58e-04, Dahl eps 0.071, tau 0.201
- **Main machine, for comparison:** about 14 generations projected by `STOP_BY`. This is an estimate, not a value read from that run.

## Checks

| Check | Result | Reference |
|---|---:|---:|
| Check 1, A100 | 17.886 | 17.87 |
| Check 1, RTX 2080 Ti | 17.898 | 17.87 |
| Check 2 (gen 0), mean | 10.186 | 10.178 |
| Check 2 (gen 0), median | 8.728 | 8.724 |
| Check 2, largest per-clip difference | 0.077 | |

## Deviations from the guide

1. **GPU:** RTX 2080 Ti instead of the A6000, with 28 workers on 4 GPUs.
2. **Python:** conda Python 3.11.18 instead of a venv, with the pinned packages.
3. **Job limit:** the 24-hour Slurm limit split the fit into a job chain (`--dependency=afterany`, `RESUME=1`), resuming from `_es.pkl` at each hand-over.
4. **Hand-overs:** after generation 6 by manually cancelling job 1859858, approved by the owner; after generations 13, 20 and 26 by an in-job watchdog that stopped the fit once another generation no longer fit into the job. No completed generation was lost.
5. **Node change:** tg06b to tg06a at the last hand-over.
6. **Reserve job:** job 1861310 started after the end and exited after 2 s without computing anything, because `_validate.json` existed.

The closing `fit done` line of `_log.jsonl` reports the time of the last chain link only (49023 s), not the whole fit.

## Overlay videos

`clips36_final_v1/prediction_videos/newton_7.6/` holds one video per clip, two stills per clip and `results.json`,
made with `newton_overlay.py` (job on 2 x RTX 2080 Ti, 14 workers, same load per GPU as the validation).

- **Third evaluation:** these rollouts give mean 7.562 and median 6.534, against 7.562 and 6.528 in the validation. The largest per-clip difference is 0.159 mm/node (`20261021_171620`), so per-clip values vary by more than the median absolute difference above suggests.
- **Clip `20261028_105803`:** 10.185 here. Two of three evaluations are near 10.2; the in-script 15.358 is the exception. The cause is still not investigated.
- **Deviation from the other folders:** drawn on `check_videos/` because the raw camera images are not on the cluster. See the README in `prediction_videos/`.

## best_ever per generation

| gen | best | mean | best_ever |
|---:|---:|---:|---:|
| 0 | 10.186 | -- | 10.186 |
| 1 | 9.284 | 11.274 | 9.284 |
| 2 | 9.335 | 9.822 | 9.284 |
| 3 | 9.053 | 9.628 | 9.053 |
| 4 | 8.938 | 9.481 | 8.938 |
| 5 | 8.623 | 9.158 | 8.623 |
| 6 | 8.652 | 9.148 | 8.623 |
| 7 | 8.604 | 9.076 | 8.604 |
| 8 | 8.633 | 9.068 | 8.604 |
| 9 | 8.395 | 8.795 | 8.395 |
| 10 | 8.561 | 8.990 | 8.395 |
| 11 | 8.068 | 8.674 | 8.068 |
| 12 | 8.284 | 8.746 | 8.068 |
| 13 | 8.313 | 8.599 | 8.068 |
| 14 | 8.157 | 8.655 | 8.068 |
| 15 | 8.148 | 8.515 | 8.068 |
| 16 | 7.958 | 8.451 | 7.958 |
| 17 | 8.120 | 8.479 | 7.958 |
| 18 | 7.880 | 8.263 | 7.880 |
| 19 | 7.840 | 8.187 | 7.840 |
| 20 | 7.630 | 8.249 | 7.630 |
| 21 | 7.891 | 8.315 | 7.630 |
| 22 | 7.683 | 8.134 | 7.630 |
| 23 | 7.682 | 8.225 | 7.630 |
| 24 | 7.794 | 8.158 | 7.630 |
| 25 | 7.789 | 8.026 | 7.630 |
| 26 | 7.615 | 8.070 | 7.615 |
| 27 | 7.667 | 8.153 | 7.615 |
| 28 | 7.565 | 7.959 | 7.565 |
| 29 | 7.688 | 7.994 | 7.565 |
| 30 | 7.566 | 7.961 | 7.565 |

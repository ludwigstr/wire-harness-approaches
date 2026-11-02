# Handoff: Newton fit on the 36 selected clips

This guide is for a Claude instance taking over the Newton fit from the instance that started it on the main machine,
the project owner's workstation. It says what to run, how to tell that your setup reproduces the original numbers, and
which rules apply. The owner talks German; code, logs and commit messages stay in English.

## 1. What this fit is

- **Model.** NVIDIA Newton 1.5.0 cable rods (VBD solver, 10 iterations) driven by the measured motion of the cable
  ends. There are 4 cables on the harness board, 16 tracked nodes each.
- **Parameters.** 231, all per cable (4 cables x 14 joints where per joint), in log space except the rest angles:

  | Parameter | Values |
  |---|---|
  | bend stiffness ke | 56 |
  | twist stiffness kt | 56 |
  | bend damping kd | 56 |
  | rest angle | 56 (rad) |
  | Dahl hysteresis eps, tau | 2 |
  | stretch stiffness K | 1 |
  | frictions (shape_mu, cable_mu, mu_tors, mu_roll) | 4 |

  The **mass is pinned at the measured 0.044 kg/m** and is not fitted.
- **Optimizer.** CMA-ES, popsize 18, per-coordinate step sizes fixed in the script (`STD_*`). The warm start is the
  finished 14-clip optimum `data/newton_fit_everything_pinned.npz` (in-sample 7.42 mm/node on its own 14 clips).
- **Objective.** Mean over clips of the mean node distance (mm/node) between simulated and tracked cables, which is
  exactly the table metric (`WIRE_Z_WEIGHT=0`: these clips have no depth targets).
- **Clips.** `data/tracks36/` holds 36 clips, 5853 frames, whole clips with no trims. This is the owner's selection of
  the best recordings (see `data/tracks36/README.txt`). Each npz has `c0..c3` (T,16,2) in metres plus `fps`.
- **Deliverable.** A summary-table row ("Newton, per cable, mass pinned") on these 36 clips. It needs the fitted
  parameters and a fresh-process validation. Hard deadline: **2026-11-10 12:00 (Europe/Berlin)**, validation
  included.

## 2. Coordination: read before starting anything

The original run **keeps running on the main machine**. The owner explicitly said nothing there may be stopped.

- It started 2026-11-02 10:49, with 9 workers on 2 RTX A6000s.
- Its generation 0 is finished. Generations take about 10.5–12 h each, and it stops starting new generations at
  2026-11-09 19:00.

Ask the owner which role you have before you spend GPU days. Possible roles:

- **A: Independent run.** A fresh start on your hardware, for example with more GPUs, and so more generations.
  Whichever run has the lower validated error goes into the table.
- **B: Continue from a snapshot.** Resume from the latest CMA-ES state in `state/` (see §5). This only helps if your
  hardware is faster or the main run is later stopped by the owner.

Never write into the main machine's files and never touch processes you did not start.

## 3. Setup

```bash
git clone <this repo>              # private repo; the owner gives you access
cd wire-harness-approaches/newton_fit_t36
python3 -m venv ~/venvs/newton_t36 && . ~/venvs/newton_t36/bin/activate
pip install -r requirements_newton.txt
```

- **Python.** The original runs on 3.11.
- **Pinned versions.** These are the exact versions behind every number here, all unmodified PyPI wheels (verified
  against their RECORD hashes). Different Newton or Warp versions change the numbers, so keep them pinned. If the
  `newton` distribution name does not resolve on your index, find the package that provides `import newton` at
  version 1.5.0 and say so in your report.
- **GPU.** CUDA is needed; each worker process uses about 0.3–2 GB of GPU memory.

The folder is self-contained: `scripts/` plus the `wire_model/` modules they import, `data/` and `state/`. Every
script resolves paths relative to this folder (`Path(__file__).parents[1]`). Two absolute path constants
(`scripts/mujoco_scene.py:44`, `wire_model/perception/seg_masks.py:39`) belong to the live-video tracking path. That
path is not used here: routes come from the npz.

## 4. Check that your setup reproduces the original (do this first)

**Check 1: one clip, about 8 min on one GPU.** Clip 20261021_163244 with the warm start must give **17.87 mm/node**
(the main machine: 17.874 in generation 0, 17.898 and 17.909 in two separate processes).

```bash
CUDA_VISIBLE_DEVICES=0 python3 - <<'PY'
import os, sys, numpy as np
os.environ.update(WIRE_TRACKS="data/tracks36", WIRE_Z_WEIGHT="0", WIRE_MASS_PIN="0.044")
sys.path[:0] = [".", "scripts"]
import newton_fit_everything_cmaes as E, newton_full_scene as N
kw = E.unpack(np.load("data/newton_fit_everything_pinned.npz", allow_pickle=True)["best"])
z = np.load("data/tracks36/20261021_163244.npz")
e = N.eval_clip("20261021_163244", shp_fps=({k: np.asarray(z[f"c{k}"], float) for k in range(4)}, float(z["fps"])),
                solver_iterations=10, device="cuda:0", **kw)
print(np.mean(e))
PY
```

- A deviation of a few hundredths is normal (the simulation is not bit-deterministic on GPU). Up to about 0.4 mm on a single clip is known run-to-run variation of
  this parameter set; the mean over clips is stable to about 0.02.
- More than 0.5 mm means a different environment. Stop and report to the owner instead of fitting.

**Check 2: generation 0 of a fresh run.** It must give **mean 10.178, median 8.724 mm/node**. Per clip values are in
`state/newton_fit_everything_pinned_t36_gen0.json`, and the run writes its own `data/..._gen0.json` for comparison.

## 5. Running

**Fresh start (role A):**

```bash
NEWTON_GPUS=0,0,1,1 NEWTON_WORKERS=12 setsid nohup bash run_newton_t36.sh > run_newton_t36.log 2>&1 < /dev/null &
```

- **`NEWTON_GPUS`** lists one process pool per entry; the entries are warp device indices under
  `CUDA_VISIBLE_DEVICES`.
- **`NEWTON_WORKERS`** is the total number of worker processes, split evenly over the pools.
- **Work per generation.** One generation is 18 candidates x 36 clips = 648 independent simulations, distributed
  longest clip first. Wall time is roughly proportional to 1 / number of workers, as long as GPUs and CPU cores are
  not saturated.
- **Measured on the main machine** (9 workers on 2 A6000s, which were shared with other jobs): generation 0, one
  candidate, took 40 min. A full generation takes about 10.5–12 h.
- **Plan for the deadline.** `STOP_BY` (default `2026-11-09 19:00`) makes the fit refuse to start a generation that
  would end later. After the fit come the in-script final validation (one candidate, about the length of generation 0)
  and the fresh-process validation (same again). Leave margin for both.
- **Logs:**
  - `data/newton_fit_everything_pinned_t36_stdout.log`: one line per generation:
    `gen N/30: best=... mean=... best_ever=...`
  - `data/..._log.jsonl`
- **Results.** The script stores the best parameters (`data/newton_fit_everything_pinned_t36.npz`, key `best`) and
  pickles the CMA-ES state (`..._es.pkl`) after every generation. A crash costs at most one generation.

**Continue (after a crash, or role B):**

1. Put `newton_fit_everything_pinned_t36.npz` (and `_es.pkl` if one exists) into `data/`: your own files after a crash,
   or the latest snapshot from `state/` for role B.
2. Run:

   ```bash
   RESUME=1 NEWTON_GPUS=... NEWTON_WORKERS=... setsid nohup bash run_newton_t36.sh >> run_newton_t36.log 2>&1 < /dev/null &
   ```

Without `_es.pkl`, the resume reuses the stored generation-0 score and starts CMA-ES fresh at the warm start, which
saves the 40-min generation 0. `state/` currently holds only the generation-0 result of the main run.

**Afterwards.** `run_newton_t36.sh` runs `scripts/newton_validate_tracks.py` in fresh processes. It writes
`data/newton_fit_everything_pinned_t36_validate.json` (`mean`, `median`, per clip). That file's mean and median are
the table numbers.

## 6. Rules (from the owner; non-negotiable)

**Fit setup**
- **All 36 clips, always.** Never fit or decide on a subset; a sweep on a subset may guide, never decide. Never add
  clips from outside `data/tracks36`.
- **Do not change the fit itself.** Keep parameter set, bounds, step sizes, popsize, objective, solver iterations,
  pinned mass and warm start. Only throughput settings (workers, GPUs) are yours. Anything else is a decision for the
  owner; ask first.
- **Report deviations.** Every deviation from this procedure goes into your report, because the table carries a
  footnote listing deviations.

**Files**
- **Never overwrite results.** The scripts refuse to overwrite; do not work around it. A new attempt gets new file
  names.

**Processes and machine**
- **Stop processes only by PID**, never by pattern (`pkill -f`). Never touch processes you did not start; other people
  run long trainings on shared GPUs.
- No `sudo`, no `curl | sh`. Treat downloaded code as untrusted.

**Git and access**
- **Commit messages: no `Co-Authored-By` trailer** (owner's explicit wish).
- **Never print or write GitHub tokens.**
- **Never change repository access.** `KishanThummar007` is a legitimate collaborator. Ask the owner before any access
  change.

**Wording and data keys**
- Never rename the JSON key `"mittel"` where it appears.
- Do not use the word "LOCO" and do not use the old 16-clip set; the reference sets are the 14 clips and these 36.

## 7. What to hand back

Push to this repo under `newton_fit_t36/results/<your-run-name>/`:

- **Result files:**
  - `newton_fit_everything_pinned_t36.npz`
  - `_es.pkl`
  - `_log.jsonl`
  - `_stdout.log`
  - `_gen0.json`
  - `_validate.json`
  - your `run_newton_t36.log`
- **A short `REPORT.md`:**
  - hardware and workers
  - check 1 and check 2 numbers
  - number of generations
  - best_ever per generation
  - final in-script result
  - fresh-process validation mean and median
  - every deviation from this guide

Tell the owner the validated mean and median (mm/node) and how many generations were run, compared with the main
machine's run.

## 8. File list

| Path | What |
|---|---|
| `run_newton_t36.sh` | the chain: fit -> in-script final validation -> fresh-process validation |
| `scripts/newton_fit_everything_cmaes.py` | the fit (env-var configured; docstring explains every mode) |
| `scripts/newton_full_scene.py` | scene building + `eval_clip` (the simulation and the metric) |
| `scripts/newton_validate_tracks.py` | fresh-process validation on a track folder |
| `scripts/newton_fit_percable_cmaes.py` | worker init, GPU assignment, `safe_savez` |
| `scripts/{clip_set,fit_clip,forecast_frames,mujoco_scene,newton_fit_stiffness,newton_pseudo_render}.py`, `wire_model/` | imported helpers |
| `data/tracks36/` | the 36 clips (+ README with the selection) |
| `data/newton_fit_everything_pinned.npz` | warm start = 14-clip optimum (`best`, 231 values, stored layout see `unpack()`) |
| `state/newton_fit_everything_pinned_t36_gen0.json` | main machine's generation 0 per clip (reference for check 2) |
| `requirements_newton.txt` | pinned packages |

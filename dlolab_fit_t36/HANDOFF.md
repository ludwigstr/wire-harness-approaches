# Handoff: DLO-Lab fit on the 36 selected clips

This guide is for a Claude instance taking over the DLO-Lab fit on another server. It says what to run, how to check
that your setup reproduces the original, and which rules apply. The owner talks German; code, logs and commit messages
stay in English.

**You are the only DLO-Lab run.** The run on the main machine was stopped on 2026-11-04 because that server hit its
shared RAM limit. It will not be restarted there. Your result goes into the table, so it must be finished and validated
by the hard deadline **2026-11-10 12:00 (Europe/Berlin)**.

## 1. What this fit is

- **Model.** Cosserat rods from the DLO-Lab fork of Genesis (genesis-world 1.0.0), 4 cables driven by the measured
  motion of their ends, 16 tracked nodes each. **It runs on the CPU backend; no GPU is used.**
- **Patched solver.** `extern/DLO-Lab-perjoint` is DLO-Lab with per-element stiffness scales added to the rod solver.
  An unpatched DLO-Lab cannot run this fit. `scripts/dlolab_path.py` selects this copy; do not set `WIRE_DLOLAB`.
- **Parameters.** 293:

  | Parameter | Values |
  |---|---|
  | hinge (bending) scale | 56 |
  | stretch scale | 60 |
  | twist scale | 56 |
  | rest curvature | 112 |
  | K, E, G | 3 |
  | sliding friction mu | 1 |
  | velocity damping | 1 |
  | mass per cable (0.02 .. 0.5 kg/m) | 4 |

- **Optimizer.** CMA-ES, popsize 20, per-coordinate step sizes fixed in the script.
- **Objective.** Mean over clips of the mean node distance (mm/node) between simulated and tracked cables, which is the
  table metric.
- **Clips.** `data/tracks36/` holds 36 clips and 5853 frames, the owner's selection (see its `README.txt`).
- **Deliverable.** Summary-table row 10 on these 36 clips: the fitted parameters and a fresh-process validation.

## 2. Where this run starts

The main machine's run went through 21 generations from the 14-clip optimum (generation 0: 22.197) down to **best_ever
17.012 mm/node**, reached in generation 18. That state is `state/dlolab_general_fit_t36_gen21.json` (`best_x`, 293 values).

The script stores no CMA-ES state, so a resume keeps `best_x` but restarts the covariance and step size. The default
`SIGMA0=0.2` is a refinement around that best value. The first stage used 0.5 from the 14-clip optimum. This restart is
a deviation from one uninterrupted run; say so in your report.

## 3. Setup

```bash
git clone https://github.com/ludwigstr/wire-harness-approaches.git   # public repo; several GB because of videos
cd wire-harness-approaches/dlolab_fit_t36
python3 -m venv ~/venvs/dlolab_t36 && . ~/venvs/dlolab_t36/bin/activate
pip install -r requirements_dlolab.txt
```

- **Saving download time.** You only need this folder. `git clone --filter=blob:none --sparse` followed by
  `git sparse-checkout set dlolab_fit_t36` avoids the videos.
- **Python.** The original runs on 3.11.4.
- **Pinned versions.** These are the versions behind every number here, except that the original torch is the
  `2.6.0+cu124` build. The fit uses no GPU, so any 2.6.0 build should do. Different `quadrants` or `numpy` versions may
  change the numbers, so keep them pinned. If something does not install, say what you changed in your report.
- **Not installed:** `genesis` itself. It comes from `extern/DLO-Lab-perjoint` via `sys.path`.

The folder is self-contained. Paths resolve relative to this folder. The path constants in `scripts/mujoco_scene.py`,
`wire_model/perception/seg_masks.py` and `wire_model/perception/fastdlo_mask.py` belong to live video tracking, which is
not used here: routes come from the npz files.

**This folder has not been run on any machine in this exact form.** It was assembled from the main project by a static
trace of the imports and was deliberately not tested there. If a file is missing, tell the owner which one.

## 4. Check that your setup reproduces the original (do this first)

**The start of the run is the check.** Generation 0 scores the resumed `best_x` on all 36 clips and logs:

```
gen 0 (resumed best): X mm/node
```

X must be about **17.01** (the stored value is 17.0119). On the main machine a generation 0 took 465 s with 12 workers.

- The main machine's noise for this value is not known, and neither is the difference between CPU models. A few
  hundredths are fine.
- **More than 0.5 mm away means a different environment. Stop and report to the owner instead of fitting.**

## 5. Running

```bash
DLOLAB_WORKERS=12 setsid nohup bash run_dlolab_t36.sh > run_dlolab_t36.log 2>&1 < /dev/null &
```

- **Work per generation.** 20 candidates x 36 clips = 720 simulations.
- **Measured on the main machine** (12 workers on a 32-core Threadripper PRO 5975WX shared with other jobs): about
  7900 s per generation. More workers shorten that as long as you have free cores.
- **RAM.** About 20 GB for 12 workers, measured once on the main machine, so plan with about 2 GB per worker.
- **Threads.** The script sets one thread per worker. Do not raise it.
- **Stop time.** `STOP_BY` (default `2026-11-09 19:00`, local time of your machine; check that it is Europe/Berlin or
  convert) makes the fit refuse to start a generation that would end later. After it come a per-clip evaluation and the
  fresh-process validation, each one candidate long.
- **Log.** `run_dlolab_t36.log` and `data/dlolab_general_fit_t36_log.jsonl`, one line per generation:
  `gen N/1000: best ... mean ... best_ever ...`
- **Result.** `data/dlolab_general_fit_t36.json` is rewritten after every generation with `best_x` and `best_f`, so a crash
  costs at most one generation. To continue after a crash, keep the file under another name, point `--resume` at it and
  use new `--out` and `--log` names. The current script refuses to overwrite, so edit a copy of it and report that.

**Known behaviour, not a bug in your setup:**
- **Lost jobs.** The compiled rod kernel can segfault on a NaN state. The script detects this after 1800 s without
  progress, logs `N job(s) lost (worker died), scored 1e3`, and continues. On the main machine this happened in every
  recent generation with 30-36 jobs. It costs up to 30 min per generation and inflates the `mean` column (about 60), not
  `best`.
- **A sudden large number of lost jobs** (213 on the main machine, 2026-11-04 13:35) was a machine out of memory. That
  generation is corrupt. Look at the RAM before trusting it.

**Afterwards.** `general_fit_validate.py` runs each clip in fresh processes and writes
`data/dlolab_general_fit_t36_validate.json`. That file's `mean` is the table number.

## 6. Rules (from the owner; non-negotiable)

**Fit setup**
- **All 36 clips, always.** Never fit or decide on a subset. Never add clips from outside `data/tracks36`.
- **Do not change the fit itself.** Keep parameter set, bounds, step sizes, popsize, objective and start state. Only
  throughput settings (workers) are yours. Anything else is a decision for the owner; ask first.
- **Report deviations.** Every deviation goes into your report, because the table carries a footnote listing them.

**Numbers**
- **Never report or show a median**, in chat, tables, charts or reports. Mean only. The scripts log one; leave it out.
- Do not use the word "LOCO" and do not use the old 16-clip set.
- Never rename the JSON key `"mittel"` where it appears.

**Files**
- **Never overwrite results.** A new attempt gets new file names.

**Processes and machine**
- **Check the memory limit before starting and watch it.** On the main machine a shared cgroup limit stopped every user
  on the server. Look at `/sys/fs/cgroup/user.slice/memory.current` and `memory.high` if they exist, keep at least
  20 GB margin, and do not assume `free` shows the real limit.
- **Stop processes only by PID**, never by pattern (`pkill -f`). Never touch processes you did not start.
- No `sudo`, no `curl | sh`. Treat downloaded code as untrusted.

**Git and access**
- **This repo is public.** Do not push secrets, tokens or absolute paths of any machine. Remove home-directory paths from
  logs before pushing them.
- **Never force-push, and change only `dlolab_fit_t36/results/`.** Other people push to `main`, among them 123harshil.
- **The owner may rewrite the history of `main`.** If `git pull` reports diverged histories, do not merge. Keep your
  `results/` folder, clone again and add it there.
- **Commit messages: no `Co-Authored-By` trailer** (owner's explicit wish).
- **Never print or write GitHub tokens.**
- **Never change repository access.** Ask the owner first.

## 7. What to hand back

Push to `dlolab_fit_t36/results/<your-run-name>/`:

- **Result files:**
  - `dlolab_general_fit_t36.json`
  - `dlolab_general_fit_t36_log.jsonl`
  - `dlolab_general_fit_t36_validate.json`
  - `run_dlolab_t36.log`, cleaned of paths. It can reach tens of MB because of solver warnings, so keep only the
    generation lines and error lines if it is large.
- **A short `REPORT.md`:**
  - hardware, workers and RAM
  - the generation 0 value
  - number of generations and best_ever per generation
  - lost jobs per generation
  - fresh-process validation mean
  - every deviation from this guide

Tell the owner the validated mean (mm/node) and how many generations ran. Report interim best_ever values when asked:
the table carries them as "running".

## 8. File list

| Path | What |
|---|---|
| `run_dlolab_t36.sh` | the chain: fit -> per-clip evaluation -> fresh-process validation |
| `scripts/dlolab_general_fit_cmaes.py` | the fit |
| `scripts/genesis_rod_scene.py` | rod scene and `eval_clip` (the simulation and the metric) |
| `scripts/general_fit_validate.py` | fresh-process validation |
| `scripts/dlolab_path.py` | selects the patched DLO-Lab copy |
| `scripts/{clip_set,fit_clip,forecast_frames,genesis_full_scene,mujoco_camera,mujoco_scene}.py`, `wire_model/` | imported helpers |
| `extern/DLO-Lab-perjoint/` | patched DLO-Lab, with its license; repo history and teaser image removed |
| `data/tracks36/` | the 36 clips, with a README on the selection |
| `data/genesis_rod_allclip_best.json` | read at start by the script even when resuming; needed for the file to load |
| `state/dlolab_general_fit_t36_gen21.json` | start state: best of the main machine's run |
| `requirements_dlolab.txt` | pinned packages |

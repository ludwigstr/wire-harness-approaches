"""Does the fit win because the solver is under-converged?

The 59-parameter twist fit scores xy 7.76 while it runs and 8.21 in its final
validation. Same parameters, different solver setting: the fit evaluates with
10 VBD iterations, the validation with 20. VBD sweeps all vertices toward the
implicit solution N times per substep, so a low N leaves residual constraint
violation standing -- the cable may stretch a little and contacts may penetrate
a little. The cheap reading is "the fit cheats"; the interesting reading is
that the real cable IS that compliant and the model has no parameter for it.

This separates the two. One parameter set, evaluated at 5 .. 40 iterations:
  - error rising monotonically with N  -> the softness is doing real work and
    belongs in a parameter, most likely K stretch, which sits pinned at 1e5 and
    has never been fitted
  - a minimum at 20 or above           -> 10 is simply under-converged and the
    in-run numbers were optimistic, nothing more

20 is the protocol value and is bracketed on both sides.
"""
from __future__ import annotations
import json, os, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
FIT = os.environ.get("WIRE_FIT", "data/newton_fit_twist.npz")
ITERS = [int(x) for x in os.environ.get("WIRE_ITERS", "5,10,15,20,30,40").split(",")]
N_JOINTS = 14


def unpack(row):
    """Layout by length: 45 = 14 ke + 14 kd + 1 mass + 2 Dahl + 14 rest,
    59 = the same plus 14 per-joint twist stiffnesses."""
    row = np.asarray(row, float)
    core = 2 * N_JOINTS + 1                      # ke, kd, mass
    has_dahl = len(row) >= core + 2              # 29 = row 8, no Dahl, no rest
    has_rest = len(row) >= core + 2 + N_JOINTS
    has_twist = len(row) >= core + 2 + 2 * N_JOINTS
    ke = np.tile(row[:N_JOINTS], (4, 1))
    kd = np.tile(row[N_JOINTS:2 * N_JOINTS], (4, 1))
    mass = np.full(4, row[2 * N_JOINTS])
    e0 = core
    eps = np.full((4, N_JOINTS), float(row[e0])) if has_dahl else None
    tau = np.full((4, N_JOINTS), float(row[e0 + 1])) if has_dahl else None
    r0 = e0 + 2
    rest = np.tile(row[r0:r0 + N_JOINTS], (4, 1)) if has_rest else None
    kt = (np.tile(row[r0 + N_JOINTS:r0 + 2 * N_JOINTS], (4, 1))
          if has_twist else None)
    return ke, kd, mass, eps, tau, rest, kt


def _one(job):
    clip, t1, gpu, iters = job
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    from newton_full_scene import eval_clip
    ke, kd, mass, eps, tau, rest, kt = unpack(
        np.load(ROOT / FIT, allow_pickle=True)["best"])
    t0 = time.time()
    try:
        errs = eval_clip(clip, t1_sec=t1, per_joint_ke=ke, per_joint_kd=kd,
                         per_joint_kt=kt, mass_per_m=mass,
                         solver_iterations=iters, dahl_eps_max=eps,
                         dahl_tau=tau, per_joint_rest=rest)
        return clip, float(np.mean(errs)), time.time() - t0, None
    except Exception as e:
        return clip, None, time.time() - t0, f"{type(e).__name__}: {e}"


def main():
    from clip_set import CLIP_TRIMS
    import multiprocessing as mp
    clips = sorted(CLIP_TRIMS)
    gpus = [0, 1, 2]
    out_path = ROOT / os.environ.get("WIRE_ITER_OUT",
                                 "data/newton_iteration_sweep.json")
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    ctx = mp.get_context("spawn")
    for iters in ITERS:
        if str(iters) in res:
            continue
        print(f"\n=== {iters} Iterationen ===", flush=True)
        jobs = [(c, CLIP_TRIMS.get(c), gpus[i % len(gpus)], iters)
                for i, c in enumerate(clips)]
        per, vals = {}, []
        with ctx.Pool(3) as pool:
            for clip, mm, dt, err in pool.imap_unordered(_one, jobs):
                if err:
                    print(f"  {clip}  FEHLER: {err}", flush=True)
                else:
                    per[clip] = mm; vals.append(mm)
        res[str(iters)] = {"mittel": float(np.mean(vals)),
                           "median": float(np.median(vals)), "per_clip": per}
        print(f"  Mittel {np.mean(vals):.3f}  Median {np.median(vals):.3f} mm",
              flush=True)
        out_path.write_text(json.dumps(res, indent=1))

    done = {int(k): v["mittel"] for k, v in res.items()}
    order = sorted(done)
    print("\n" + "  ".join(f"{k}:{done[k]:.3f}" for k in order), flush=True)
    best = min(done, key=done.get)
    rising = all(done[a] <= done[b] + 1e-9 for a, b in zip(order, order[1:]))
    print(f"bester Wert bei {best} Iterationen", flush=True)
    print("MONOTON STEIGEND -- die Weichheit leistet echte Arbeit, sie gehoert "
          "in einen Parameter (K stretch ist der Kandidat)" if rising else
          "kein monotoner Trend -- 10 war schlicht unterkonvergiert", flush=True)
    print("ITER_SWEEP_FERTIG", flush=True)


if __name__ == "__main__":
    main()

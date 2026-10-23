"""PhysTwin (Jiang et al. 2025): a spring-mass digital twin fitted to video.
Table: world sheet row 19 -- 19.8 mm/node (median 17.0), one twin per clip fitted
on all frames and resimulated from frame 0 (paper 7:3 split: 13.7 on the fitting
70 %, 30.5 on the held-out 30 %).

The twin (wire_model/phystwin/sim.py), planar:
  mass points = the 16 tracked nodes of each cable (2 g each, not fitted);
  springs     = chain neighbours + node pairs closer than `radius` segments in
                the first frame (at most 6 per node); they also carry bending,
                since a spring-mass model has no bending angle;
  rest length = the first frame (frame 0 is the cable's rest shape);
  forces      = springs, dashpots on connected pairs, drag, Coulomb board friction;
  ends        = the two end nodes move kinematically along their tracks; 1 ms steps.

Fit, per clip (the four cables share damping, drag, friction):
  stage 1  CMA-ES on [homogeneous k, dashpot gamma, drag delta, friction mu,
           radius] -- cost = mean node error of the full resimulation from frame 0;
  stage 2  Adam through the simulator on a stiffness PER SPRING (+ gamma, delta,
           mu) on random 8-frame windows; every 10 iterations the full
           resimulation is scored and the best iterate is kept (iterate 0 =
           stage 1), because the window-only optimisation diverged on long clips.
Not implemented from the paper: the Gaussian-splat rendering (we compare node
tracks, not images), 3-D, and collisions.

    python approaches/05_phystwin.py eval                       # checkpoints/phystwin_twins/*.pt -> 19.8
    python approaches/05_phystwin.py fit [--train-frac 0.7]     # ~15 min per clip on one GPU
"""
import argparse
import time

import numpy as np
import torch

from common import ROOT, load_clips, record, write_results
from wire_model.phystwin.fit import stage1, stage2
from wire_model.phystwin.sim import SpringMassTwin, build_topology

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def resimulate(P, twin, k, gl, frame_s):
    """full resimulation from frame 0 with the tracked end motions -> (T,) error over the interior nodes (mm)"""
    C = P.shape[0]
    with torch.no_grad():
        sim = twin.rollout(P[:, 0], (P[:, 1] - P[:, 0]) / frame_s, P[:, :, [0, -1]], k, gl[0].expand(C), gl[1].expand(C), gl[2].expand(C))
        return torch.nan_to_num((sim[:, :, 1:-1] - P[:, :, 1:-1]).norm(dim=-1), nan=1e3).mean((0, 2)).cpu().numpy() * 1000


def fit(tracks, twins_dir, out, train_frac=1.0, gens=20, iters=100, clips=None):
    res = {}
    for c in load_clips(tracks, clips):
        P = torch.tensor(c["P"], device=DEV); frame_s = 1.0 / c["fps"]; T = P.shape[1]
        n_fit = T if train_frac >= 1 else max(3, int(round(train_frac * T)))
        t0 = time.time(); print(f"{c['clip']}: {T} frames x 4 cables, fit on {n_fit}", flush=True)
        cost1, x1 = stage1(P, n_fit, DEV, gens=gens, frame_s=frame_s)
        twin, k, gl = stage2(P, n_fit, x1, DEV, iters=iters, frame_s=frame_s)
        ef = resimulate(P, twin, k, gl, frame_s)
        d = ROOT / twins_dir; d.mkdir(parents=True, exist_ok=True)
        torch.save({"k": k.cpu(), "gl": gl.cpu(), "radius_seg": float(x1[4]), "x1": np.asarray(x1), "adj": twin.adj.cpu(), "frame_s": frame_s,
                    "n_fit": n_fit}, d / f"{c['clip']}.pt")
        res[c["clip"]] = record(ef, fit_mm=float(ef[:n_fit].mean()), test_mm=float(ef[n_fit:].mean()) if n_fit < T else None, n_fit=n_fit,
                                stage1_mm=cost1, k_hom=float(np.exp(x1[0])), radius_seg=float(x1[4]), gamma=float(gl[0]), delta=float(gl[1]), mu=float(gl[2]))
        print(f"{c['clip']}: all frames {ef.mean():.1f} | fitting frames {ef[:n_fit].mean():.1f} ({time.time() - t0:.0f}s)", flush=True)
    write_results(res, out, "PhysTwin fit")


def evaluate(tracks, twins_dir, out):
    res = {}
    for c in load_clips(tracks):
        tw = torch.load(ROOT / twins_dir / f"{c['clip']}.pt", map_location=DEV, weights_only=False)
        P = torch.tensor(c["P"], device=DEV)
        r = round(tw["radius_seg"] * 2) / 2
        twin = SpringMassTwin(P[:, 0], build_topology(P[:, 0], r, 6), frame_s=tw["frame_s"])
        res[c["clip"]] = record(resimulate(P, twin, tw["k"].to(DEV), tw["gl"].to(DEV), tw["frame_s"]))
    write_results(res, out, "PhysTwin (stored twins)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["fit", "eval"])
    ap.add_argument("--tracks", default="data/tracks14"); ap.add_argument("--clips", default=None)
    ap.add_argument("--twins", default=None, help="twin directory (default checkpoints/phystwin_twins)")
    ap.add_argument("--train-frac", type=float, default=1.0); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    twins = a.twins or ("checkpoints/phystwin_twins" if a.step == "eval" else "work/phystwin_twins")
    if a.step == "fit":
        fit(a.tracks, twins, a.out or f"results/phystwin_fit_{a.train_frac:g}.json", a.train_frac, clips=a.clips)
    else:
        evaluate(a.tracks, twins, a.out or "results/phystwin_14.json")


if __name__ == "__main__":
    main()

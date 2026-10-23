"""Validate the per-cable fit's best row at 10 iterations in a FRESH process.

The fit's own final validation runs inside the process that was launched on
2026-10-19 22:27, which imported newton_full_scene when the iteration default
was still 20 -- the mu fit's "final 8.16" turned out to be exactly that (both
paths re-evaluated today give 7.88). This gives the per-cable set its N=10
number, the table's protocol, independent of that stale import.
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
os.environ.setdefault("WIRE_REST", "1"); os.environ.setdefault("WIRE_DAHL_MODE", "scalar")
import newton_fit_percable_cmaes as M
from clip_set import CLIP_TRIMS
from mujoco_scene import track_clip
from newton_full_scene import eval_clip

d = np.load(ROOT / "data/newton_fit_percable.npz", allow_pickle=True)
best = np.asarray(d["best"], float)
x = M._to_x(best)
assert np.allclose(M._from_x(x), best), "round trip x<->row is not exact"
parts = M._params_from_log(x)
print("param blocks:", [None if p is None else np.asarray(p).shape for p in parts], flush=True)
ke, kd, mass, eps, tau, rest = parts[:6]
extra = {}
if len(parts) > 6 and parts[6] is not None:          # kt or mu blocks, if the layout has them
    print("extra blocks present:", [np.asarray(p).shape for p in parts[6:] if p is not None])
kw = dict(per_joint_ke=ke, per_joint_kd=kd, mass_per_m=mass, dahl_eps_max=eps, dahl_tau=tau,
          per_joint_rest=rest, subdiv=M.SUBDIV)
out = {}
for c in sorted(CLIP_TRIMS):
    shp, t0, t1, fps = track_clip(c, t0=8, t1=M._clip_frame_t1(c, CLIP_TRIMS[c]))
    v = float(np.mean(eval_clip(c, shp_fps=(shp, fps), solver_iterations=10, device="cuda:0", **kw)))
    out[c] = v
    print(f"{c}  {v:7.3f}", flush=True)
res = {"mittel": float(np.mean(list(out.values()))), "median": float(np.median(list(out.values()))),
       "n_clips": len(out), "per_clip": out, "solver_iterations": 10}
(ROOT / "data/newton_percable_N10.json").write_text(json.dumps(res, indent=1))
print(f"\nMittel {res['mittel']:.3f}  Median {res['median']:.3f}  n={len(out)}")
print("PERCABLE_N10_FERTIG", flush=True)

"""Why did the mu fit's in-run best (xy 7.88) re-validate at 8.16?

Newton itself is deterministic: the same 45 parameters re-evaluated at the
same N reproduce to 0.001 mm per clip (newton_determinism_big_N10.json). So
the gap must come from the two code paths inside newton_fit_mu_cmaes.py:
  A  final validation: _params_from_log(_to_x(best)), device=None,
     solver_iterations left at the module default
  B  in-run worker:    _params_from_log(x), device given, solver_iterations=10
This evaluates the stored best row through both, per clip.
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "engines"))
os.environ.setdefault("WIRE_MU", "1"); os.environ.setdefault("WIRE_REST", "1")
os.environ.setdefault("WIRE_DAHL_MODE", "scalar")
import newton_fit_mu_cmaes as M
from clip_set import CLIP_TRIMS
from mujoco_scene import track_clip
from newton_full_scene import eval_clip

best = np.asarray(np.load(ROOT / "data/newton_fit_mu.npz", allow_pickle=True)["best"], float)
x = M._to_x(best)
assert np.allclose(M._from_x(x), best), "round trip x<->row is not exact"
ke, kd, mass, eps, tau, rest, extra, kt, mu = M._params_from_log(x)
kw = dict(per_joint_ke=ke, per_joint_kd=kd, per_joint_kt=kt, mass_per_m=mass,
          shape_mu=float(mu[0]), cable_mu=float(mu[1]), dahl_eps_max=eps, dahl_tau=tau,
          per_joint_rest=np.tile(rest, (4, 1)), subdiv=M.SUBDIV)
stored = json.loads(str(np.load(ROOT / "data/newton_fit_mu.npz", allow_pickle=True)["per_clip"]))
out = {}
for c in sorted(CLIP_TRIMS):
    shp, t0, t1, fps = track_clip(c, t0=8, t1=M._clip_frame_t1(c, CLIP_TRIMS[c]))
    a = float(np.mean(eval_clip(c, shp_fps=(shp, fps), **kw)))                       # path A
    b = float(np.mean(eval_clip(c, shp_fps=(shp, fps), solver_iterations=10,
                                device="cuda:0", **kw)))                              # path B
    out[c] = {"A_final_path": a, "B_inrun_path": b, "stored_final": stored.get(c)}
    print(f"{c}  A {a:7.3f}  B {b:7.3f}  stored {stored.get(c, float('nan')):7.3f}", flush=True)
    (ROOT / "data/newton_mu_path_probe.json").write_text(json.dumps(out, indent=1))
A = np.mean([v["A_final_path"] for v in out.values()]); B = np.mean([v["B_inrun_path"] for v in out.values()])
print(f"\nMittel  A (final path) {A:.3f}   B (in-run path) {B:.3f}   stored final 8.160   in-run best 7.88")
print("PATH_PROBE_FERTIG", flush=True)

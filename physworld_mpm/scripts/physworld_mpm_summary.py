"""Summary of the PhysWorld MPM rebuild (2026-10-28) over the 14 clips: twin alone, GNN before /
after Phi fine-tuning, parameter ranges, counts and timings -> data/wm/physworld_mpm_14.json
usage: physworld_mpm_summary.py [--twin elastoplastic] [--gnn-tag ""] [--demos-tag ""]"""
import argparse, json, math
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
WARM = 5

ap = argparse.ArgumentParser(); ap.add_argument("--twin", default="elastoplastic"); ap.add_argument("--gnn-tag", default="")
ap.add_argument("--demos-tag", default=""); ap.add_argument("--out", default="data/wm/physworld_mpm_14.json")
a = ap.parse_args()
clips = sorted(p.stem for p in (ROOT / "data/tracks14").glob("*.npz"))
res, twin_m, twin_g, gnn, gnn0 = {}, [], [], [], []
rng = {k: [] for k in ("E", "rho", "mu")}; glob = {k: [] for k in ("E", "rho", "mu", "nu", "sy")}
n_part = 0; t_fit = []; t_demo = []; n_demo = 0; n_rej = 0; rej_reasons = []; n_frames_demo = 0
t_train = []; t_phi = []; phi_all = []; gnn_params = None; bounds = {}
for c in clips:
    r = {}
    f = ROOT / f"data/wm/physworld_mpm_twins/{c}_{a.twin}.pt"
    if f.exists():
        tf = torch.load(f, weights_only=False)
        m = np.array(tf["twin_final"])[WARM:].mean(); mg = np.array(tf["twin_global"])[WARM:].mean()
        r.update(twin_mm=float(m), twin_global_mm=float(mg), twin_initial_mm=float(np.array(tf["twin_initial"])[WARM:].mean()))
        twin_m.append(m); twin_g.append(mg)
        for k in rng:
            rng[k].append((float(tf[k].min()), float(tf[k].max())))
        G = tf["glob"]
        for k, v in (("E", math.exp(G["lE"])), ("rho", math.exp(G["lrho"])), ("mu", math.exp(G["lmu"])), ("nu", float(G["nu"])),
                     ("sy", tf["sy"])):
            glob[k].append(v)
        r["global"] = {k: glob[k][-1] for k in glob}
        n_part += len(tf["E"]); t_fit.append(tf["sec_total"]); bounds[c] = tf["bounds_global"]
    f = ROOT / f"data/wm/physworld_mpm_demos/{c}{a.demos_tag}.pt"
    if f.exists():
        D = torch.load(f, weights_only=False)
        t_demo.append(D["sec"]); n_demo += len(D["demos"]); n_rej += len(D["rejected"]); n_frames_demo += D["n_frames_total"]
        rej_reasons += [x["reason"] for x in D["rejected"]]
    f = ROOT / f"data/wm/physworld_mpm_gnn{('_' + a.gnn_tag) if a.gnn_tag else ''}/{c}.json"
    if f.exists():
        g = json.loads(f.read_text())
        r.update(mittel=g["mittel"], mittel_before_phi=g["mittel_before_phi_finetune"], n_frames=g["n_frames"],
                 start=g["start"], ende=g["ende"])
        gnn.append(g["mittel"]); gnn0.append(g["mittel_before_phi_finetune"]); t_train.append(g["sec_train"]); t_phi.append(g["sec_phi"])
        phi_all.append(np.array(g["phi_fitted"]).reshape(-1, 2)); gnn_params = g["n_gnn_params"]
    res[c] = r
st = lambda v: {"mean": float(np.mean(v)), "median": float(np.median(v)), "n": len(v)} if len(v) else None
mm = lambda L: (min(x[0] for x in L), max(x[1] for x in L)) if L else None
phi = np.concatenate(phi_all) if phi_all else np.zeros((0, 2))
summary = {
    "twin_alone": st(twin_m), "twin_global_stage_only": st(twin_g), "physworld_after_phi": st(gnn), "physworld_before_phi": st(gnn0),
    "per_particle_ranges": {k: mm(v) for k, v in rng.items()}, "global_ranges": {k: (min(v), max(v)) if v else None for k, v in glob.items()},
    "global_on_bound": bounds, "n_particles_total": n_part, "n_fitted_twin_params": {"global": 5 * len(t_fit),
        "per_particle": 3 * n_part},
    "sec_twin_fit": float(np.sum(t_fit)) if t_fit else None, "sec_demos": float(np.sum(t_demo)) if t_demo else None,
    "n_demos_kept": n_demo, "n_demos_rejected": n_rej, "rejected_reasons": rej_reasons, "n_demo_frames": n_frames_demo,
    "sec_gnn_train": float(np.sum(t_train)) if t_train else None, "sec_phi": float(np.sum(t_phi)) if t_phi else None,
    "gnn_params_per_model": gnn_params, "n_phi_fitted": int(phi.size),
    "phi_fitted_ranges": {"E": (float(np.exp(phi[:, 0].min())), float(np.exp(phi[:, 0].max()))) if len(phi) else None,
                          "rho": (float(np.exp(phi[:, 1].min())), float(np.exp(phi[:, 1].max()))) if len(phi) else None},
}
(ROOT / a.out).write_text(json.dumps({"summary": summary, "clips": res}, indent=1))
print(json.dumps(summary, indent=1))
for c, r in res.items():
    print(c, {k: (round(v, 2) if isinstance(v, float) else v) for k, v in r.items() if k != "global"})

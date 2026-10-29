"""Check of the quasi-static mass scaling (PhysWorld MPM rebuild, 2026-10-28): the same
parameters simulated with true inertia (time step from CFL 0.5) and with the inertial
density raised so that dt = 1e-4 s meets CFL 0.5. Clip 20260602_170108, all frames."""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch, warp as wp
sys.path.insert(0, str(Path(__file__).resolve().parents[1])); wp.init()
from wire_model.physworld_mpm.twin import Twin
clip = sys.argv[1] if len(sys.argv) > 1 else "20260602_170108"
z = np.load(f"data/tracks14/{clip}.npz"); P = np.stack([z[f"c{k}"] for k in range(4)])
res = []
for E, rho, mu, nu, sy in [(3e5, 1556, 0.3, 0.3, 1e4), (6.55e5, 704, 0.136, 0.401, 2.79e4), (2e6, 400, 0.08, 0.40, 5e4)]:
    out = {"E": E, "rho": rho, "mu": mu, "nu": nu, "sy": sy}
    for ms in (None, 1e-4):
        tw = Twin(P, device="cuda:0", r_grasp=0.012, mass_scale_dt=ms)
        g = {"lE": torch.tensor(math.log(E)), "lrho": torch.tensor(math.log(rho)), "lmu": torch.tensor(math.log(mu)),
             "nu": torch.tensor(nu), "lsy": torch.tensor(math.log(sy))}
        pp = tw.per_particle(g, None)
        Mv = pp["E"] * (1 - pp["nu"]) / ((1 + pp["nu"]) * (1 - 2 * pp["nu"]))
        dt = min(1e-4, 0.5 * tw.dx / torch.sqrt(Mv / pp["rhoi"]).max().item() * 1.0001)
        n_sub = int(100 * math.ceil(0.25 / dt / 100)); tw.mpm.dt = 0.25 / n_sub
        t0 = time.time(); nodes, info = tw.rollout(tw.to_warp(pp)); el = time.time() - t0
        e = (nodes[..., :2] - tw.Ptorch.permute(1, 0, 2, 3)).norm(dim=-1); e[:, :, [0, -1]] = 0
        m = (e.mean((1, 2)) * 1000).cpu().numpy()
        key = "scaled" if ms else "true"
        out[key] = {"twin_mm": float(m[5:].mean()), "n_sub": n_sub, "rhoi_over_rho": float((pp["rhoi"] / pp["rho"]).max()),
                    "sec": el, "bad": info["bad"]}
        out[key + "_nodes"] = nodes[..., :2].cpu()
        print(E, rho, mu, nu, key, out[key], flush=True)
    d = (out["true_nodes"] - out["scaled_nodes"]).norm(dim=-1).mean().item() * 1000
    out["mean_node_diff_true_vs_scaled_mm"] = d; print("  node difference true vs scaled:", d, "mm", flush=True)
    del out["true_nodes"], out["scaled_nodes"]; res.append(out)
Path(f"data/wm/physworld_mpm_masscale_check_{clip}.json").write_text(json.dumps(res, indent=1))

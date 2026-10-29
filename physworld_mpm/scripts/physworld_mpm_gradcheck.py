"""Finite-difference check of the Warp MPM gradients (PhysWorld rebuild, 2026-10-28).

Tiny scene: one 12 cm cable (16 nodes, gently curved) on the board, one end
dragged 15 mm in one 0.05 s frame = 500 substeps, the whole frame on the tape
(W = n_sub). Loss = a fixed random linear functional of the particle xy
displacements (well conditioned; the paper loss is checked through the same
tape since its torch part is plain autograd). FD noise floor: float32 atomics
make the simulation non-bitwise-deterministic, measured by repeating runs. Compared: tape gradient vs central finite differences of the
full simulation for the 5 global parameters (log E, log rho, log mu, nu, log sy)
and one random per-particle direction of log E. Run elastic and elastoplastic
(sy small enough to yield). Also: truncated gradient (last 10 substeps, the
paper's setting) vs the full one (cosine)."""
import json, math, sys
from pathlib import Path
import numpy as np, torch, warp as wp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
wp.config.log_level = wp.LOG_WARNING if hasattr(wp, "LOG_WARNING") else None
wp.init()
from wire_model.physworld_mpm.twin import Twin

dev = "cuda:0"
s = np.linspace(0, 0.12, 16)
n0 = np.stack([0.3 + s, 0.4 + 0.01 * np.sin(s / 0.12 * np.pi)], 1)
n1 = n0.copy(); n1[0] += [0.010, 0.011]
P = np.stack([n0, n1])[None].astype(np.float32)            # (1, 2, 16, 2)
FS = 0.05


def run(tw, glob, loc, target, W=0, need_grad=False):
    pp = tw.per_particle(glob, loc)
    mats = tw.to_warp(pp, requires_grad=need_grad)
    acc = {k: torch.zeros(tw.NP, device=dev) for k in ("E", "rho", "mu", "nu", "sy")} if need_grad else {}
    holder = {}
    def lin(x):
        return (WLIN * (x[:, :2] - tw.sp.pos0[:, :2])).sum() * 1000.0
    def cb(t, x):
        xl = x.detach().clone().requires_grad_(True)
        L = lin(xl)
        L.backward(); holder["L"] = L.item(); return xl.grad
    nodes, info = tw.rollout(mats, W=W, grad=("trunc" if W < 500 else "frame") if need_grad else None, loss_cb=cb, acc=acc, frame_s=FS, K=100)
    if not need_grad:
        x = info["state"].x
        return lin(wp.to_torch(x).clone()).item()
    g = {"lE": (acc["E"] * pp["E"]).sum().item(), "lrho": (acc["rho"] * pp["rho"]).sum().item(),
         "lmu": (acc["mu"] * pp["mu"]).sum().item(), "nu": acc["nu"].sum().item(),
         "lsy": (acc["sy"] * pp["sy"]).sum().item()}
    return holder["L"], g, acc, pp


def main():
    out = {}
    for case, lsy in (("elastic", None), ("elastoplastic", math.log(300.0))):
        tw = Twin(P, cables=(0,), dt=1e-4, frame_s=FS, device=dev, r_grasp=0.006)
        tw.obs_dense = [[a for a in tw.obs_dense[0]]]
        global WLIN
        WLIN = torch.tensor(np.random.default_rng(3).normal(0, 1, (tw.NP, 2)), dtype=torch.float32, device=dev) / tw.NP
        target = tw.Ptorch[:, 1] + torch.tensor(np.random.default_rng(1).normal(0, 0.002, (1, 16, 2)), dtype=torch.float32, device=dev)
        glob = {"lE": torch.tensor(math.log(2e5)), "lrho": torch.tensor(math.log(1556.0)), "lmu": torch.tensor(math.log(0.4)),
                "nu": torch.tensor(0.3), "lsy": torch.tensor(lsy if lsy is not None else 0.0)}
        if lsy is None:
            glob["elastic"] = True
        dirE = torch.tensor(np.random.default_rng(2).normal(0, 1, tw.NP), dtype=torch.float32, device=dev)
        n_sub = int(round(FS / 1e-4))
        L, g, acc, pp = run(tw, glob, None, target, W=n_sub, need_grad=True)
        g["dirE"] = (acc["E"] * pp["E"] * dirE).sum().item()
        _, gt, acct, _ = run(tw, glob, None, target, W=10, need_grad=True)
        rep = [run(tw, glob, None, target) for _ in range(4)]
        res = {"loss": L, "repeat_spread": float(np.ptp(rep))}
        print(case, "loss", L, "repeat spread (noise floor)", np.ptp(rep), flush=True)
        keys = ["lE", "lrho", "lmu", "nu"] + (["lsy"] if lsy is not None else []) + ["dirE"]
        for k in keys:
            fd = {}
            for h in (3e-2, 1e-1):
                def f(sgn):
                    gl = dict(glob); loc = None
                    if k == "dirE":
                        loc = {"dlE": sgn * h * dirE, "dlrho": torch.zeros_like(dirE), "dlmu": torch.zeros_like(dirE)}
                    else:
                        gl[k] = glob[k] + sgn * h
                    return run(tw, gl, loc, target)
                fd[h] = (f(1) - f(-1)) / (2 * h)
            res[k] = {"tape": g[k], "fd_3e-2": fd[3e-2], "fd_1e-1": fd[1e-1],
                      "rel_err_3e-2": abs(g[k] - fd[3e-2]) / (abs(fd[3e-2]) + 1e-12),
                      "truncated_W10": gt.get(k) if k != "dirE" else (acct["E"] * pp["E"] * dirE).sum().item()}
            print(case, k, json.dumps(res[k]), flush=True)
        v_full = torch.cat([acc[k] for k in ("E", "rho", "mu")]); v_tr = torch.cat([acct[k] for k in ("E", "rho", "mu")])
        res["cos_full_vs_truncated_perparticle"] = torch.nn.functional.cosine_similarity(v_full, v_tr, dim=0).item()
        print(case, "cos(full, truncated W=10) per-particle grads:", res["cos_full_vs_truncated_perparticle"], flush=True)
        out[case] = res
    Path("data/wm/physworld_mpm_gradcheck.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

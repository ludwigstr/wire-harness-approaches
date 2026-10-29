"""Gradient check on a real clip (PhysWorld rebuild, 2026-10-28): paper loss summed over
frames 1..T-1 of clip 20260602_170108 (4 cables, frames 0..T-1), global parameters.
Compared: central finite differences of the full simulation vs the three gradient
modes of Twin.rollout: "trunc" (last 10 substeps per frame = paper Sec. 4.1),
"frame" (all 2500 substeps of each frame, frame-start state detached), "full"
(adjoint through the whole rollout)."""
import json, math, sys, time
from pathlib import Path
import numpy as np, torch, warp as wp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
wp.init()
from wire_model.physworld_mpm.twin import Twin

dev = "cuda:0"; CLIP = "20260602_170108"; T = int(sys.argv[1]) if len(sys.argv) > 1 else 5
z = np.load(f"data/tracks14/{CLIP}.npz"); P = np.stack([z[f"c{k}"] for k in range(4)])[:, :T]
tw = Twin(P, device=dev, r_grasp=0.012)
glob = {"lE": torch.tensor(math.log(3e5)), "lrho": torch.tensor(math.log(1556.0)), "lmu": torch.tensor(math.log(0.3)),
        "nu": torch.tensor(0.3), "lsy": torch.tensor(math.log(3e3))}
KEYS = ["lE", "lrho", "lmu", "nu", "lsy"]


def loss_of(gl, mode=None):
    pp = tw.per_particle(gl, None); mats = tw.to_warp(pp, requires_grad=mode is not None)
    acc = {k: torch.zeros(tw.NP, device=dev) for k in ("E", "rho", "mu", "nu", "sy")} if mode else {}
    tot = [0.0]
    def cb(t, x):
        xl = x.detach().clone().requires_grad_(True)
        L, _ = tw.frame_loss(xl, t); L.backward(); tot[0] += L.item(); return xl.grad
    if mode is None:
        nodes, info = tw.rollout(mats, return_x=True)
        return sum(tw.frame_loss(info["xs"][t], t)[0].item() for t in range(1, T))
    t0 = time.time()
    tw.rollout(mats, grad=mode, loss_cb=cb, acc=acc)
    g = {"lE": (acc["E"] * pp["E"]).sum().item(), "lrho": (acc["rho"] * pp["rho"]).sum().item(),
         "lmu": (acc["mu"] * pp["mu"]).sum().item(), "nu": acc["nu"].sum().item(), "lsy": (acc["sy"] * pp["sy"]).sum().item()}
    return tot[0], g, time.time() - t0


res = {}
rep = [loss_of(glob) for _ in range(3)]
res["loss_repeats"] = rep
print("loss repeats", rep, flush=True)
for mode in ("trunc", "frame", "full"):
    L, g, el = loss_of(glob, mode)
    res[mode] = g; res[mode + "_sec"] = el
    print(mode, f"{el:.0f}s", json.dumps(g), flush=True)
fd = {}
for k in KEYS:
    h = 0.05 if k != "nu" else 0.02
    gp = dict(glob); gp[k] = glob[k] + h; gm = dict(glob); gm[k] = glob[k] - h
    fd[k] = (loss_of(gp) - loss_of(gm)) / (2 * h)
    print("fd", k, fd[k], flush=True)
res["fd"] = fd
v = lambda d: np.array([d[k] for k in KEYS])
for mode in ("trunc", "frame", "full"):
    a, b = v(res[mode]), v(fd)
    res[mode + "_cos_vs_fd"] = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(mode, "cosine vs FD (5 global params):", res[mode + "_cos_vs_fd"], flush=True)
Path(f"data/wm/physworld_mpm_gradcheck_clip_T{T}.json").write_text(json.dumps(res, indent=1))

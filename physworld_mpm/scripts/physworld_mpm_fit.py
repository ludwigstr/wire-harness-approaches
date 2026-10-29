"""PhysWorld rebuild (2026-10-28): global-to-local physical-property optimisation of the
MPM digital twin of ONE clip (paper Sec. 3.1, App. B.2).

  global stage: homogeneous log E, log rho, log mu (board friction), nu, log sy (yield
                stress; elastoplastic model only), Adam on the paper loss
  local stage : per-particle log E, log rho, log mu offsets on top of the global values
                (nu, sy stay global, as in the paper), Adam on the paper loss
  loss        : sum over frames of Chamfer + 0.1 smooth-L1 track (App. B.2), xy, metres
  gradient    : "frame" mode = through all substeps of each frame with the frame-start
                state detached (the paper's last-10-substeps truncation gave a gradient
                ~1% of the true magnitude with per-particle cosine 0.005 in the tiny-scene
                check, scripts/physworld_mpm_gradcheck.py; the full-rollout gradient is ~10x
                too large on a real clip, physworld_mpm_gradcheck_clip.py); per iteration the
                gradient is taken on a random fraction --frac of the frames (all frames are
                simulated and all enter the logged loss), gradient-norm clipping (paper).
  selection   : the iterate with the lowest full loss of each stage.
Twin metric (control): resimulation from frame 0 with the measured ends, mean node
distance over frames 5..T-1 (the table's predicted frames), end nodes pinned.
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, torch, warp as wp
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
wp.config.log_level = getattr(wp, "LOG_WARNING", None) or 30
wp.init()
from wire_model.physworld_mpm.twin import Twin, BOUNDS, RHO_PHYS, max_stable_dt

WARM = 5


def log(s):
    print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)


def twin_metric(nodes, P):
    """nodes (T, G, 16, 3) sim, P (G, T, 16, 2) real -> per-frame mean node error (mm), ends pinned"""
    e = (nodes[..., :2] - P.permute(1, 0, 2, 3)).norm(dim=-1)
    e[:, :, [0, -1]] = 0.0
    return e.mean((1, 2)).cpu().numpy() * 1000


def clamp_glob(g):
    with torch.no_grad():
        for k, (lo, hi) in BOUNDS.items():
            if k in g:
                g[k].clamp_(lo, hi)


def on_bound(g):
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        if k in g:
            v = g[k].item(); out[k] = "low" if abs(v - lo) < 1e-3 else ("high" if abs(v - hi) < 1e-3 else "")
    return out


CFL = 0.5


def run_iter(tw, glob, loc, model, frac, rng, dev, need_grad=True, dt_scale=1.0):
    """one rollout over all frames; gradient of the loss on a random frame subset. Returns (loss, grads, nodes)"""
    G = {k: v for k, v in glob.items()}
    if model == "elastic":
        G["elastic"] = True
    pp = tw.per_particle(G, loc)
    # adaptive time step: the paper's 1e-4 s unless the stiffest particle needs less (CFL 0.5;
    # 0.9 blew up at E 4.7e5 / rho 1000 / nu 0.34)
    M = pp["E"] * (1 - pp["nu"]) / ((1 + pp["nu"]) * (1 - 2 * pp["nu"]))
    c_max = torch.sqrt(M / pp["rhoi"]).max().item()
    dt = dt_scale * min(1e-4, CFL * tw.dx / c_max * 1.0001)
    n_sub = int(100 * math.ceil(tw.frame_s / dt / 100))
    tw.mpm.dt = tw.frame_s / n_sub
    mats = tw.to_warp(pp, requires_grad=need_grad)
    acc = {k: torch.zeros(tw.NP, device=dev) for k in ("E", "rho", "rhoi", "mu", "nu", "sy")}
    sel = set((rng.random(tw.T) < frac).nonzero()[0].tolist()) if need_grad else set()
    tot = [0.0]
    def cb(t, x):
        xl = x.detach().clone().requires_grad_(t in sel)
        L, _ = tw.frame_loss(xl, t)
        tot[0] += L.item()
        if t not in sel:
            return None
        L.backward(); return xl.grad
    nodes, info = tw.rollout(mats, grad="frame" if need_grad else None, loss_cb=cb, acc=acc)
    if info["bad"]:
        if dt_scale > 0.3:                                       # retry once with half the time step
            log(f"    unstable at n_sub {n_sub}, retrying with dt/2")
            return run_iter(tw, glob, loc, model, frac, rng, dev, need_grad, dt_scale * 0.5)
        return float("inf"), None, nodes, info, n_sub
    if not need_grad:
        return None, None, nodes, info, n_sub
    # chain rule to the parameters
    pairs = [(pp[k], acc[k]) for k in ("E", "rho", "rhoi", "mu", "nu", "sy") if pp[k].requires_grad]
    if pairs:
        torch.autograd.backward([a for a, _ in pairs], [b for _, b in pairs])
    return tot[0], None, nodes, info, n_sub


def stage(tw, glob, loc, params, lrs, iters, model, frac, rng, dev, P, name, hist, clip_norm=1.0):
    opt = torch.optim.Adam([{"params": [p], "lr": lr} for p, lr in zip(params, lrs)])
    best = (float("inf"), None)
    for it in range(iters):
        t0 = time.time()
        for p in params:
            p.grad = None
        L, _, nodes, info, n_sub = run_iter(tw, glob, loc, model, frac, rng, dev)
        m = twin_metric(nodes, tw.Ptorch) if not info["bad"] else np.array([np.inf])
        mt = float(m[WARM:].mean()) if len(m) > WARM else float("inf")
        rec = {"stage": name, "it": it, "loss": L, "twin_mm": mt, "n_sub": n_sub, "missed": info["missed"],
               "sec": None, **{k: (float(v) if k == "nu" else float(math.exp(v))) for k, v in
                               ((k_, glob[k_].item()) for k_ in ("lE", "lrho", "lmu", "nu", "lsy"))}}
        if L < best[0]:
            best = (L, ([p.detach().clone() for p in params]), mt, it)
        if not math.isfinite(L):
            log(f"  {name} it {it}: unstable rollout, stopping stage"); hist.append(rec); break
        gn = torch.sqrt(sum((p.grad ** 2).sum() for p in params if p.grad is not None))
        if gn > clip_norm:
            for p in params:
                if p.grad is not None:
                    p.grad *= clip_norm / gn
        opt.step()
        clamp_glob(glob)
        if loc is not None:
            with torch.no_grad():
                for v in loc.values():
                    v.clamp_(-3.0, 3.0)
        rec["sec"] = time.time() - t0; rec["grad_norm"] = float(gn)
        hist.append(rec)
        log(f"  {name} it {it}: loss {L:.5f} twin {mt:.2f} mm  E {rec['lE']:.3g} rho {rec['lrho']:.4g} mu {rec['lmu']:.3g} "
            f"nu {rec['nu']:.3f} sy {rec['lsy']:.3g}  |g| {gn:.2e}  n_sub {n_sub}  {rec['sec']:.0f}s")
    with torch.no_grad():
        for p, b in zip(params, best[1]):
            p.copy_(b)
    log(f"  {name}: best loss {best[0]:.5f} at it {best[3]} (twin {best[2]:.2f} mm)")
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--model", default="elastoplastic", choices=["elastic", "elastoplastic"])
    ap.add_argument("--global-iters", type=int, default=20); ap.add_argument("--local-iters", type=int, default=20)
    ap.add_argument("--frac", type=float, default=0.33)
    ap.add_argument("--E0", type=float, default=3e5); ap.add_argument("--mu0", type=float, default=0.3)
    ap.add_argument("--sy0", type=float, default=1e4); ap.add_argument("--nu0", type=float, default=0.3)
    ap.add_argument("--r-grasp", type=float, default=0.012)
    ap.add_argument("--mass-scale-dt", type=float, default=1e-4,
                    help="quasi-static mass scaling: inertial density raised so that this time step meets CFL 0.5 "
                         "(0 = off, true inertia, time step shrinks with E/rho)")
    ap.add_argument("--tag", default=""); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = a.device
    torch.manual_seed(0); rng = np.random.default_rng(0)
    z = np.load(ROOT / f"data/tracks14/{a.clip}.npz"); P = np.stack([z[f"c{k}"] for k in range(4)])
    tw = Twin(P, device=dev, r_grasp=a.r_grasp, mass_scale_dt=a.mass_scale_dt or None)
    log(f"{a.clip}: {tw.T} frames, {tw.NP} particles ({[len(c['pos']) for c in tw.cab]}), model {a.model}, "
        f"dx {tw.dx*1000:.1f} mm, r_grasp {a.r_grasp*1000:.0f} mm")
    glob = {"lE": torch.tensor(math.log(a.E0), device=dev, requires_grad=True),
            "lrho": torch.tensor(math.log(RHO_PHYS), device=dev, requires_grad=True),
            "lmu": torch.tensor(math.log(a.mu0), device=dev, requires_grad=True),
            "nu": torch.tensor(a.nu0, device=dev, requires_grad=True),
            "lsy": torch.tensor(math.log(a.sy0), device=dev, requires_grad=a.model == "elastoplastic")}
    hist = []; t0 = time.time()
    # initial twin metric
    _, _, nodes, info, _ = run_iter(tw, glob, None, a.model, 0.0, rng, dev, need_grad=False)
    m0 = twin_metric(nodes, tw.Ptorch); log(f"  initial twin {m0[WARM:].mean():.2f} mm")
    gkeys = ["lE", "lrho", "lmu", "nu"] + (["lsy"] if a.model == "elastoplastic" else [])
    glr = {"lE": 0.2, "lrho": 0.2, "lmu": 0.2, "nu": 0.03, "lsy": 0.3}
    stage(tw, glob, None, [glob[k] for k in gkeys], [glr[k] for k in gkeys], a.global_iters, a.model, a.frac, rng, dev,
          P, "global", hist)
    t_glob = time.time() - t0
    _, _, nodes, info, _ = run_iter(tw, glob, None, a.model, 0.0, rng, dev, need_grad=False)
    m_glob = twin_metric(nodes, tw.Ptorch)
    log(f"global stage done: twin {m_glob[WARM:].mean():.2f} mm ({t_glob:.0f}s); bounds {on_bound(glob)}")
    for k in glob:
        glob[k].requires_grad_(False)
    loc = {k: torch.zeros(tw.NP, device=dev, requires_grad=True) for k in ("dlE", "dlrho", "dlmu")}
    if a.local_iters > 0:
        stage(tw, glob, loc, list(loc.values()), [0.1, 0.1, 0.1], a.local_iters, a.model, a.frac, rng, dev, P, "local",
              hist, clip_norm=1.0)
    t_all = time.time() - t0
    _, _, nodes, info, n_sub = run_iter(tw, glob, loc, a.model, 0.0, rng, dev, need_grad=False)
    m_fin = twin_metric(nodes, tw.Ptorch)
    log(f"final twin {m_fin[WARM:].mean():.2f} mm (global {m_glob[WARM:].mean():.2f}, initial {m0[WARM:].mean():.2f}); "
        f"total {t_all:.0f}s")
    G = {k: v.detach().cpu() for k, v in glob.items()}
    pp = tw.per_particle({**glob, **({"elastic": True} if a.model == "elastic" else {})}, loc)
    tag = f"_{a.tag}" if a.tag else ""
    out = ROOT / f"data/wm/physworld_mpm_twins/{a.clip}_{a.model}{tag}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"clip": a.clip, "model": a.model, "glob": G, "loc": {k: v.detach().cpu() for k, v in loc.items()},
                "E": pp["E"].detach().cpu(), "rho": pp["rho"].detach().cpu(), "mu": pp["mu"].detach().cpu(),
                "nu": float(glob["nu"]), "sy": float(pp["sy"][0]), "perm": tw.sp.perm, "r_grasp": a.r_grasp,
                "dx": tw.dx, "n_sub": n_sub, "mass_scale_dt": a.mass_scale_dt, "rhoi": pp["rhoi"].detach().cpu(),
                "twin_initial": m0.tolist(), "twin_global": m_glob.tolist(), "twin_final": m_fin.tolist(),
                "nodes_final": nodes.cpu(), "hist": hist, "sec_global": t_glob, "sec_total": t_all,
                "bounds_global": on_bound(glob)}, out)
    log(f"saved {out}")


if __name__ == "__main__":
    main()

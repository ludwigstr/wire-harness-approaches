"""Training and evaluation shared by the two RSSM world models (RopeDreamer, DeformNet).

Both learn ONLY from the tracked videos (no simulator): pre-training on the 103
X-Planar clips, then fine-tuning in-sample on the 14 evaluation clips.

Evaluation = the RopeDreamer paper protocol (Sec. V-B): warm up on the first
`warm` frames with the measured states and actions, then predict open loop from
the measured end motions only; both end nodes are pinned to the measurement;
scored = every frame after the warm-up.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from common import ROOT, record, write_results
import wire_model.ropedreamer.data as RD
from wire_model.ropedreamer.data import load_tracks, WindowDataset, state_offset, trajectory_tensors, state_stats
from wire_model.ropedreamer.model import RopeDreamer


def _p(x):
    x = Path(x)
    return x if x.is_absolute() else ROOT / x


def build(kind, size="S", beta=1.0, free_nats=0.0, pos_weight=0.0, rollout_weight=0.0, warm=5, repr_="chain", device="cuda"):
    if kind == "deformnet":
        from wire_model.deformnet import DeformNet
        return DeformNet(16, beta=beta, free_nats=free_nats, pos_weight=pos_weight, rollout_weight=rollout_weight, warm=warm).to(device)
    return RopeDreamer(16, size, beta=beta, free_nats=free_nats, pos_weight=pos_weight,
                       rollout_weight=rollout_weight, warm=warm, repr=repr_).to(device)


def open_loop_mm(model, trajs_t, off, warm, horizon, device, n_max=64):
    """mean node error (mm) of the open-loop prediction after `warm` frames, ends pinned (checkpoint selection)."""
    model.eval(); errs = []
    for t in trajs_t[:n_max]:
        T = t["s"].shape[0]; n = min(horizon, T - warm)
        if n < 1:
            continue
        s = (t["s"][None] - off).to(device); idx = t["idx"][None].to(device); dp = t["dp"][None].to(device)
        h, z = model.warmup(s[:, :warm], idx[:, :warm], dp[:, :warm])
        pred = model.imagine(h, z, idx[:, warm - 1:warm - 1 + n], dp[:, warm - 1:warm - 1 + n]) + off.to(device)
        truth = t["P"][warm:warm + n].to(device)
        if model.repr == "chain":
            pred = pred.clone(); pred[0, :, :2] = truth[:, 0]
        P = model.positions(pred[0], t["ell"].to(device))[..., :2].clone(); P[:, 0] = truth[:, 0]; P[:, -1] = truth[:, -1]
        errs.append((P - truth).norm(dim=-1).mean(-1).cpu().numpy() * 1000)
    model.train()
    return float(np.mean([e.mean() for e in errs]))


def train(kind, tracks, out, size="S", seq_len=32, epochs=200, patience=10, batch=32, lr=1e-4, beta=1.0, free_nats=0.0,
          pos_weight=0.0, rollout_weight=0.0, warm=5, val_frac=0.1, init=None, select="val_loss", horizon=50,
          repr_="chain", device="cuda", seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    if kind == "deformnet":
        repr_ = "positions"
    RD.REPR = repr_
    out = _p(out); out.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(str(out).replace(".pt", "_log.txt"), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    trajs = load_tracks([_p(d) for d in tracks])
    perm = np.random.default_rng(seed).permutation(len(trajs))
    if val_frac > 0:
        n_val = max(1, int(val_frac * len(trajs)))
        val = [trajs[i] for i in perm[:n_val]]; tr = [trajs[i] for i in perm[n_val:]]
    else:                                   # in-sample: train and select on the same clips
        val = list(trajs); tr = list(trajs)
    off = state_offset(trajs)
    dl_tr = torch.utils.data.DataLoader(WindowDataset(tr, seq_len, seed=seed), batch_size=batch, shuffle=True, drop_last=True)
    dl_va = torch.utils.data.DataLoader(WindowDataset(val, seq_len, seed=seed + 1), batch_size=batch)
    val_t = [trajectory_tensors(t["P"]) for t in val]
    model = build(kind, size, beta, free_nats, pos_weight, rollout_weight, warm, repr_, device)
    if init:
        ck0 = torch.load(_p(init), map_location=device, weights_only=False)
        model.load_state_dict(ck0["state_dict"]); off = ck0["offset"].cpu()
        if hasattr(model, "pretrain_ae"):    # DeformNet: encoder/decoder stay frozen after stage 1
            for p_ in list(model.enc.parameters()) + list(model.dec.parameters()):
                p_.requires_grad_(False)
        log(f"init from {init}")
    else:
        s_mean, s_std = state_stats(tr)
        model.set_state_stats((s_mean - off).to(device), s_std.to(device))
        if hasattr(model, "pretrain_ae"):    # DeformNet stage 1
            model.pretrain_ae(torch.cat([trajectory_tensors(t["P"])["s"] for t in tr]) - off, device, log=log)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log(f"=== {kind} {size}: {len(tr)} train / {len(val)} val trajectories, seq_len {seq_len}, beta {beta}, "
        f"free nats {free_nats}, pos {pos_weight}, rollout {rollout_weight}, select {select} ===")
    best, bad = float("inf"), 0; off_d = off.to(device)
    for ep in range(1, epochs + 1):
        t0 = time.time(); model.train(); tl = 0.0; n = 0
        for s, idx, dp, ell in dl_tr:
            o = model(s.to(device) - off_d, idx.to(device), dp.to(device), ell=ell.to(device))
            opt.zero_grad(); o["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0); opt.step()
            tl += float(o["loss"]); n += 1
        model.eval(); vl = 0.0; m = 0
        with torch.no_grad():
            for s, idx, dp, ell in dl_va:
                vl += float(model(s.to(device) - off_d, idx.to(device), dp.to(device), sample=False, ell=ell.to(device))["loss"]); m += 1
        ol = open_loop_mm(model, val_t, off, warm, horizon, device)
        score = vl / m if select == "val_loss" else ol
        improved = score < best
        if improved:
            best, bad = score, 0
            torch.save({"state_dict": model.state_dict(), "size": size, "n_nodes": 16, "offset": off, "seq_len": seq_len, "beta": beta,
                        "pos_weight": pos_weight, "rollout_weight": rollout_weight, "warm": warm, "repr": repr_, "model": kind,
                        "free_nats": free_nats, "epoch": ep, "select": select, "best_score": best, "init": init, "tracks": list(tracks)}, out)
        else:
            bad += 1
        log(f"ep {ep:3d}: train {tl / n:.4f} | val {vl / m:.4f} | open loop {ol:.1f} mm{' *' if improved else ''} ({time.time() - t0:.0f}s)")
        if bad >= patience:
            log(f"early stop after {patience} epochs without improvement"); break
    log(f"best {select} {best:.4f} -> {out}")


def evaluate(ckpt, tracks, out, warm=5, device="cuda", label=None):
    ck = torch.load(_p(ckpt), map_location=device, weights_only=False)
    RD.REPR = ck.get("repr", "chain")
    model = build(ck.get("model", "ropedreamer"), ck.get("size", "S"), ck["beta"], ck.get("free_nats", 0.0), ck.get("pos_weight", 0.0),
                  ck.get("rollout_weight", 0.0), ck.get("warm", 5), ck.get("repr", "chain"), device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    off = ck["offset"].to(device)
    by_clip = {}
    for t in load_tracks([_p(tracks)]):
        by_clip.setdefault(t["clip"], []).append(t)
    res = {}
    with torch.no_grad():
        for clip, cabs in sorted(by_clip.items()):
            errs = []
            for t in sorted(cabs, key=lambda x: x["cable"]):
                d = trajectory_tensors(t["P"]); T = d["s"].shape[0]
                s = d["s"][None].to(device) - off; idx = d["idx"][None].to(device); dp = d["dp"][None].to(device)
                h, z = model.warmup(s[:, :warm], idx[:, :warm], dp[:, :warm])
                pred = model.imagine(h, z, idx[:, warm - 1:T - 1], dp[:, warm - 1:T - 1]) + off
                truth = d["P"][warm:]
                if model.repr == "chain":          # hub driven through p0 BEFORE the forward kinematics
                    pred = pred.clone(); pred[0, :, :2] = truth[:, 0].to(device)
                P = model.positions(pred[0], d["ell"].to(device))[..., :2].cpu()
                P[:, 0] = truth[:, 0]; P[:, -1] = truth[:, -1]   # both ends = measurement
                errs.append((P - truth).norm(dim=-1).mean(-1).numpy() * 1000)
            res[clip] = record(np.mean(errs, axis=0), warmup_frames=warm)
    write_results(res, out, label or f"{ck.get('model', 'ropedreamer')} {ckpt} on {tracks}")
    return res

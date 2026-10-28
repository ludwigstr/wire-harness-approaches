"""Train RopeDreamer on the tracked clips (paper Sec. V-A protocol: lr 1e-4,
batch 32, beta = 1, up to 200 epochs, early stopping with patience 10 on the
validation loss, best checkpoint kept).

    python3 scripts/ropedreamer_train.py --size S --tracks X-Planar_vids/mujoco_videos/tracks data/tracks14 \
        --out data/ropedreamer_S.pt
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from wire_model.ropedreamer.model import RopeDreamer  # noqa: E402
from wire_model.ropedreamer.data import load_tracks, load_synth, WindowDataset, state_offset, trajectory_tensors, state_stats  # noqa: E402
from wire_model.ropedreamer.kinematics import chain_to_positions  # noqa: E402
import wire_model.ropedreamer.data as RD  # noqa: E402


def open_loop_mm(model, trajs_t, off, warm, horizon, device, n_max=64):
    """Paper-style check on full trajectories: warm up on `warm` frames, imagine
    the rest (at most `horizon` steps, fewer on short clips), ends pinned.
    Returns (mean node error over all predicted frames, mean over steps 1-10,
    mean at the last predicted step, posterior reconstruction error) in mm."""
    model.eval()
    errs, recs = [], []
    for t in trajs_t[:n_max]:
        T = t["s"].shape[0]
        n = min(horizon, T - warm)
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
        m = warm + n
        out = model(s[:, :m], idx[:, :m], dp[:, :m], sample=False, ell=t["ell"][None].to(device))
        rec = out["s_recon"][0] + off.to(device)
        tr = t["P"][:m].to(device)
        if model.repr == "chain":
            rec = rec.clone(); rec[:, :2] = tr[:, 0]
        Pr = model.positions(rec, t["ell"].to(device))[..., :2].clone(); Pr[:, 0] = tr[:, 0]; Pr[:, -1] = tr[:, -1]
        recs.append((Pr - tr).norm(dim=-1).mean().item() * 1000)
    model.train()
    all_mean = float(np.mean([e.mean() for e in errs]))
    first10 = float(np.mean([e[:10].mean() for e in errs]))
    last = float(np.mean([e[-1] for e in errs]))
    return all_mean, first10, last, float(np.mean(recs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=["X-Planar_vids/mujoco_videos/tracks", "data/tracks14"])
    ap.add_argument("--synth", nargs="*", default=[],
                    help="simulated rollouts (yang_make_dataset.py output) INSTEAD of --tracks: the paper's simulation-only training")
    ap.add_argument("--size", default="S")
    ap.add_argument("--seq-len", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--free-nats", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=float, default=0.0, help="FK position loss weight (0 = the paper's state-space loss only)")
    ap.add_argument("--rollout-weight", type=float, default=0.0, help="open-loop (evaluation-protocol) position loss weight inside each window")
    ap.add_argument("--warm", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--loader-workers", type=int, default=8, help="DataLoader workers when --obs-noise is on")
    ap.add_argument("--obs-noise", default="",
                    help="tracking-noise augmentation (default off): a JSON with {'params': ...} for "
                         "wire_model/ropedreamer/noise.track_noise; the posterior sees noisy states and the model is "
                         "driven by noisy end actions, the reconstruction/prediction targets stay clean")
    ap.add_argument("--gap-ref-tol", type=float, default=None,
                    help="synthetic data: cut a rollout only where its connector gap exceeds its clip's real-timing "
                         "rollout (<clip>_00) by more than this (m); default: the fixed 15 mm cut")
    ap.add_argument("--val-by-rollout", action="store_true",
                    help="split train/val by synthetic ROLLOUT (all 4 cables of one rollout file on the same side) "
                         "instead of by single cable trajectory")
    ap.add_argument("--val-by-clip", action="store_true",
                    help="split train/val by SOURCE clip (all rollouts and cables of one X-Planar clip on the same "
                         "side; a synthetic file <clip>_<NN>.npz belongs to <clip>) instead of by trajectory")
    ap.add_argument("--out", default="data/ropedreamer_S.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights and state statistics (e.g. an X-Planar model)")
    ap.add_argument("--select", default="val_loss", choices=["val_loss", "openloop"],
                    help="checkpoint selection: validation ELBO (paper) or the open-loop mm on the selection set")
    ap.add_argument("--horizon", type=int, default=50)
    ap.add_argument("--repr", default="chain", choices=["chain", "positions"])
    ap.add_argument("--model", default="ropedreamer", choices=["ropedreamer", "deformnet"])
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    if a.model == "deformnet":
        a.repr = "positions"
    RD.REPR = a.repr
    log_f = open(ROOT / (a.out.replace(".pt", "_log.jsonl")), "a")

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"; print(line, flush=True); log_f.write(line + "\n"); log_f.flush()

    trajs = load_synth([ROOT / d for d in a.synth], gap_ref_tol=a.gap_ref_tol) if a.synth else load_tracks([ROOT / d for d in a.tracks])
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(len(trajs))
    if a.val_frac > 0 and a.val_by_clip:
        import re as _re
        src = [_re.sub(r"_\d{2,3}$", "", t["clip"]) for t in trajs]
        clips_u = sorted(set(src))
        cperm = rng.permutation(len(clips_u))
        n_vc = max(1, int(round(a.val_frac * len(clips_u))))
        vclips = {clips_u[i] for i in cperm[:n_vc]}
        val = [t for t, c in zip(trajs, src) if c in vclips]; train = [t for t, c in zip(trajs, src) if c not in vclips]
        assert not ({c for c in src if c in vclips} & {c for c in src if c not in vclips})
        print(f"clip-level split: {len(clips_u) - n_vc} train clips / {n_vc} val clips", flush=True)
    elif a.val_frac > 0 and a.val_by_rollout:
        rolls = sorted({t["clip"] for t in trajs})
        rperm = rng.permutation(len(rolls))
        vr = {rolls[i] for i in rperm[:max(1, int(round(a.val_frac * len(rolls))))]}
        val = [t for t in trajs if t["clip"] in vr]; train = [t for t in trajs if t["clip"] not in vr]
        print(f"rollout-level split: {len(rolls) - len(vr)} train rollouts / {len(vr)} val rollouts", flush=True)
    elif a.val_frac > 0:
        n_val = max(1, int(a.val_frac * len(trajs)))
        val = [trajs[i] for i in perm[:n_val]]; train = [trajs[i] for i in perm[n_val:]]
    else:
        # in-sample, the protocol of the physics rows: train on everything,
        # select the checkpoint on the same clips
        val = list(trajs); train = list(trajs)
    off = state_offset(trajs)
    obs_noise = json.loads((ROOT / a.obs_noise).read_text())["params"] if a.obs_noise else None
    if obs_noise:
        print(f"obs-noise augmentation: {obs_noise}", flush=True)
    ds_tr = WindowDataset(train, a.seq_len, seed=a.seed, obs_noise=obs_noise)
    ds_va = WindowDataset(val, a.seq_len, seed=a.seed + 1, obs_noise=obs_noise)
    if obs_noise:
        # the noise is drawn on the CPU per window: parallel loader workers, each re-seeding the
        # dataset's generator from its own torch seed (otherwise all workers draw identical noise)
        def _wif(_):
            info = torch.utils.data.get_worker_info()
            info.dataset.rng = np.random.default_rng(info.seed % (2 ** 32))
        lkw = {"num_workers": a.loader_workers, "worker_init_fn": _wif, "persistent_workers": True}
    else:
        lkw = {}
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=a.batch, shuffle=True, drop_last=True, **lkw)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=a.batch, **lkw)
    val_t = [trajectory_tensors(t["P"]) for t in val]
    if a.model == "deformnet":
        from wire_model.deformnet import DeformNet
        model = DeformNet(16, beta=a.beta, free_nats=a.free_nats, pos_weight=a.pos_weight,
                          rollout_weight=a.rollout_weight, warm=a.warm).to(a.device)
    else:
        model = RopeDreamer(16, a.size, beta=a.beta, free_nats=a.free_nats, pos_weight=a.pos_weight,
                            rollout_weight=a.rollout_weight, warm=a.warm, repr=a.repr).to(a.device)
    if a.init:
        ck0 = torch.load(ROOT / a.init, map_location=a.device, weights_only=False)
        model.load_state_dict(ck0["state_dict"]); off = ck0["offset"].cpu()
        if hasattr(model, "pretrain_ae"):
            for p_ in list(model.enc.parameters()) + list(model.dec.parameters()):
                p_.requires_grad_(False)
        log(f"init from {a.init} (epoch {ck0.get('epoch')}, val loss {ck0.get('val_loss', float('nan')):.3f}); state statistics kept")
    else:
        s_mean, s_std = state_stats(train)
        model.set_state_stats((s_mean - off).to(a.device), s_std.to(a.device))
    if hasattr(model, "pretrain_ae") and not a.init:
        # DeformNet stage 1: encoder/decoder on every training frame, then frozen
        S = torch.cat([trajectory_tensors(t["P"])["s"] for t in train]) - off
        model.pretrain_ae(S, a.device, log=log)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    n_par = sum(p.numel() for p in model.parameters())
    log(f"=== RopeDreamer {a.size} ({n_par/1e6:.2f}M params): {len(train)} train / {len(val)} val trajectories "
        f"({sum(t['P'].shape[0] for t in trajs)} frames), seq_len {a.seq_len}, batch {a.batch}, lr {a.lr}, beta {a.beta}, pos_weight {a.pos_weight}, rollout_weight {a.rollout_weight} ===")
    best, bad, hist = float("inf"), 0, []
    off_d = off.to(a.device)
    for ep in range(1, a.epochs + 1):
        t0 = time.time(); model.train(); acc = {"loss": 0, "recon": 0, "pred": 0, "kl": 0, "roll": 0}; n = 0
        for batch in dl_tr:
            s, idx, dp, ell = batch[:4]
            s, idx, dp, ell = s.to(a.device) - off_d, idx.to(a.device), dp.to(a.device), ell.to(a.device)
            if len(batch) > 4:      # --obs-noise: noisy observations + noisy end actions in, clean targets
                out = model(s, idx, batch[5].to(a.device), ell=ell, s_obs=batch[4].to(a.device) - off_d)
            else:
                out = model(s, idx, dp, ell=ell)
            opt.zero_grad(); out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
            opt.step()
            for k in acc: acc[k] += float(out[k]); 
            n += 1
        model.eval(); va = {"loss": 0, "recon": 0, "pred": 0, "kl": 0, "roll": 0}; m = 0
        with torch.no_grad():
            for batch in dl_va:
                s, idx, dp, ell = batch[:4]
                if len(batch) > 4:
                    out = model(s.to(a.device) - off_d, idx.to(a.device), batch[5].to(a.device), sample=False,
                                ell=ell.to(a.device), s_obs=batch[4].to(a.device) - off_d)
                else:
                    out = model(s.to(a.device) - off_d, idx.to(a.device), dp.to(a.device), sample=False, ell=ell.to(a.device))
                for k in va: va[k] += float(out[k])
                m += 1
        ol, ol10, ol50, rec_mm = open_loop_mm(model, val_t, off, a.warm, a.horizon, a.device)
        rec = {"epoch": ep, **{k: v / n for k, v in acc.items()}, **{"val_" + k: v / m for k, v in va.items()},
               "val_openloop_mm": ol, "val_openloop_mm_1_10": ol10, "val_openloop_mm_50": ol50, "val_recon_mm": rec_mm}
        hist.append(rec)
        score = va["loss"] / m if a.select == "val_loss" else ol
        improved = score < best
        if improved:
            best, bad = score, 0
            torch.save({"state_dict": model.state_dict(), "size": a.size, "n_nodes": 16, "offset": off,
                        "seq_len": a.seq_len, "beta": a.beta, "pos_weight": a.pos_weight, "rollout_weight": a.rollout_weight, "warm": a.warm, "repr": a.repr, "model": a.model, "free_nats": a.free_nats, "epoch": ep, "val_loss": va["loss"] / m, "select": a.select, "best_score": best, "init": a.init, "hist": hist,
                        "tracks": a.tracks, "val_clips": sorted({(t["clip"], t["cable"]) for t in val}.__iter__())}, ROOT / a.out)
        else:
            bad += 1
        log(f"ep {ep:3d}: train loss {rec['loss']:.4f} (recon {rec['recon']:.4f} pred {rec['pred']:.4f} kl {rec['kl']:.2f} roll {rec.get('roll', 0):.5f}) | "
            f"val loss {rec['val_loss']:.4f} pred {rec['val_pred']:.4f} | recon {rec_mm:.1f} mm | open-loop 50 steps: {ol:.1f} mm (1-10: {ol10:.1f}, @50: {ol50:.1f})"
            f"{' *' if improved else ''} ({time.time()-t0:.0f}s)")
        if bad >= a.patience:
            log(f"early stop: no improvement of {a.select} for {a.patience} epochs"); break
    log(f"=== ROPEDREAMER_TRAIN_FERTIG: best {a.select} {best:.4f} -> {a.out} ===")


if __name__ == "__main__":
    main()

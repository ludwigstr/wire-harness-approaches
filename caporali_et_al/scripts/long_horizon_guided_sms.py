"""PIEGraph guidance, FULL spring-mass variant (companion to long_horizon_guided.py).

Same idea, one level less approximation. long_horizon_guided.py guides the
rollout with a Position-Based-Dynamics projection, which enforces exactly one
constraint -- constant segment length -- and does it geometrically. PIEGraph
(Orozco et al., CoRL 2025, Eq. 2/4) actually guides with the spring-mass
system itself: external forces pull each particle toward the network's
predicted setpoint and the rod's OWN dynamics -- axial springs, bending
stiffness, dry friction -- arbitrate how much of that proposal survives.

So this variant asks whether the extra physics buys anything over the cheap
length-only projection: it drives our existing MSD simulator
(wire_model/dlo_model.py, the one the base checkpoint was pretrained on)
toward the Transformer's prediction via the `f_ext` hook, instead of snapping
the shape onto the constant-length manifold.

Two properties are worth stating because they decide whether the comparison
is honest:

  * The network is trained EXACTLY as in the baseline -- same loss, same
    epochs. Guidance is applied only to its output, so it cannot corrupt the
    learned dynamics the way tonight's length-loss experiment did (weight 1.0
    cost 62% accuracy; the constraint was competing with position accuracy
    inside the optimiser).
  * Every gain in --kp is rolled out from the SAME trained weights, so
    `k_p = 0` is a within-run reproduction of the baseline rollout. That
    matters: LOCO retrains move the per-clip number by a few mm on their own,
    which is enough to manufacture or hide an effect of this size.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fit_clip import read
from wire_model.perception.harness import track_harness
from wire_model.perception.rectify import find_board
from wire_model.perception.tracking import HarnessTracker
from wire_model.dlo_model import DLOParams, DLOSimulator, NodeConstraints
from train_step_model import turning_angles
from eval_rollout import train_fold, train_fold_seq, step_action, NOMINAL
from long_horizon_all import render_lh

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIR = "data/WirePrediction"
REST_CAP = 0.35


def guide_batch(sim, p_prev, setpoint, prm, k_p, n_steps, cons_target=None):
    """Pull a batch of rods from `p_prev` toward `setpoint` and let their own
    physics arbitrate -- PIEGraph Eq. 2 with the paper's PID reduced to P,
    because the simulator already carries a damping term (k_d) that an
    integral term would only fight.

    Terminal speed under this force is k_p*err/k_d, so the approach is
    exponential with time constant k_d/k_p; n_steps*dt has to cover a few of
    those or the rod stops short and "guidance" quietly degenerates into
    "barely move at all" -- which would look like a result but is just an
    under-converged solver. At k_d = 10 and dt = 4e-5 the default 4000 steps
    (0.16 s) is ~2.4 time constants at k_p = 150 and ~6 at k_p = 400.

    Both ends are pinned to the setpoint's ends, which are the REAL measured
    plug poses driving the rollout -- guidance must not be free to drift them.
    """
    p = p_prev.clone()
    v = torch.zeros_like(p)
    B, N, _ = p.shape
    mask = torch.zeros(B, N, dtype=torch.bool, device=p.device)
    mask[:, 0] = True
    mask[:, -1] = True
    # ends follow cons_target (the real measured plug poses) even when the
    # setpoint force is switched off for the ablation
    cons = NodeConstraints(mask, setpoint if cons_target is None else cons_target)
    for _ in range(n_steps):
        f_ext = None if k_p == 0.0 else k_p * (setpoint - p)
        p, v = sim.step(p, v, prm, cons, f_ext=f_ext)
    return p


def roll_guided(model, sim, cables, n_nodes, stride, k_p, n_steps,
                exit_prior=False):
    """Roll all 4 cables of one clip forward together.

    The network step runs per cable (it is small); the guidance step batches
    all four into ONE simulation call, because the simulator's per-fine-step
    Python/autograd overhead is paid per call regardless of batch size -- the
    same reason wire_model.dataset.generate() batches, and the reason
    tonight's plasticity experiment had to be rewritten after a batch=1
    version took over an hour for a single cable.
    """
    n_frames = len(cables[0])
    idxs = sorted(set(list(range(0, n_frames - 1, stride)) + [n_frames - 1]))
    B = len(cables)
    V = np.stack([s[0] for s in cables]).astype(np.float32)
    p_par = np.tile(NOMINAL, (B, 1)).astype(np.float32)
    lengths = np.array([np.linalg.norm(np.diff(s[0], axis=0), axis=1).sum()
                        for s in cables], np.float32)
    m, k_b, k_d, k_c = NOMINAL

    for ai, bi in zip(idxs, idxs[1:]):
        acts = np.stack([step_action(s[ai], s[bi], n_nodes - 2, 1)
                         for s in cables]).astype(np.float32)
        rest = np.clip(np.stack([turning_angles(x) for x in V]),
                       -REST_CAP, REST_CAP).astype(np.float32)
        with torch.no_grad():
            pred = model(torch.tensor(V, device=DEV),
                         torch.tensor(acts, device=DEV),
                         torch.tensor(p_par, device=DEV),
                         torch.tensor(rest, device=DEV)).cpu().numpy()
        pred = pred.astype(np.float32)
        for k, s in enumerate(cables):        # driven ends are known exactly
            pred[k][0], pred[k][-1] = s[bi][0], s[bi][-1]

        if k_p == 0.0:
            V = pred
            continue

        # rest_angle tracks the CURRENT shape, so bending resists change from
        # where the wire is now rather than from a remembered t0 pose -- the
        # same reading of "plastic set is the dominant shape parameter" that
        # the network's own rest-angle input uses, recomputed every step.
        rest_prm = rest.copy()
        if exit_prior:
            # The housing prior as PHYSICS rather than as a loss: the plug
            # holds its stretch of wire straight, so the rod's REST curvature
            # at the two nodes next to the connectors is zero and its bending
            # stiffness there pulls the exit back to straight. Measured in the
            # 851 tracked shapes: 0.6 and 1.3 deg median at those nodes vs
            # 7.1 deg mid-cable, while the network predicts 14.6 and 28.1 deg.
            # Every other node keeps the current-shape rest angle, so the
            # plastic set the wire actually carries is untouched.
            rest_prm[:, 0] = 0.0
            rest_prm[:, -1] = 0.0
        prm = DLOParams(m=m, k_b=k_b, k_d=k_d, k_c=k_c,
                        length=torch.tensor(lengths, device=DEV),
                        rest_angle=torch.tensor(rest_prm, device=DEV),
                        ).to_batched(B, DEV, n_nodes=n_nodes)

        # k_p < 0 is the ABLATION, not a gain: run the identical simulation
        # with NO setpoint force at all, so the rod is merely relaxed between
        # the two real measured plug poses and the network's prediction is
        # discarded. Guidance pins those ends and lets physics fill in the
        # interior, so some of its gain could come from that alone rather than
        # from the setpoint -- and the plasticity experiment already showed
        # pure physics reaching 25.6 mm/node on clip 114019 where the network
        # rollout was far worse. Without this control, a guidance result
        # cannot be attributed to PIEGraph's mechanism at all.
        target = torch.tensor(V if k_p < 0 else pred, device=DEV)
        gain = 0.0 if k_p < 0 else k_p
        cons_target = torch.tensor(pred, device=DEV)   # ends: always the real pose
        V = guide_batch(sim, torch.tensor(V, device=DEV), target, prm, gain,
                        n_steps, cons_target=cons_target
                        ).cpu().numpy().astype(np.float32)
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default="data/transitions_1s_all.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--t0-buffer", type=int, default=8)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--kp", default="0,150,400",
                    help="Guidance-Gains; 0 = ungeguidet (Within-Run-Baseline)")
    ap.add_argument("--guide-steps", type=int, default=4000)
    ap.add_argument("--clips", type=int, default=0, help="0 = alle")
    ap.add_argument("--noise-mm", type=float, default=0.0,
                    help="PhysWorld-Rauschinjektion auf V_in im Training "
                         "(0 = unveraendert)")
    ap.add_argument("--tau", type=int, default=0,
                    help="PhysWorld-Mehrschritt-Loss ueber tau Schritte "
                         "(0 = Einschritt wie bisher)")
    ap.add_argument("--pinn-lambda", type=float, default=0.0,
                    help="C-PINN Bogenlaengen-Residuum, Endgewicht der "
                         "Curriculum-Rampe (0 = aus)")
    ap.add_argument("--pinn-warmup", type=float, default=0.3,
                    help="Anteil der Epochen ohne Physikterm")
    ap.add_argument("--exit-lambda", type=float, default=0.0,
                    help="Gehaeuse-Prior als Loss-Term (gerader Kabelaustritt)")
    ap.add_argument("--exit-prior", action="store_true",
                    help="Gehaeuse-Prior in der Guidance-Physik (rest_angle=0 "
                         "an den steckernahen Knoten)")
    ap.add_argument("--out", default="data/long_horizon_guided_sms.pt")
    args = ap.parse_args()

    kps = [float(x) for x in args.kp.split(",")]
    trans = torch.load(ROOT / args.transitions, weights_only=False)["transitions"]
    clips = sorted(set(t["clip"] for t in trans))
    if args.clips:
        clips = clips[:args.clips]
    print(f"{len(clips)} Clips, Gains {kps}, {args.guide_steps} Sim-Schritte "
          f"je Prognoseschritt\n", flush=True)

    sim = DLOSimulator(n_nodes=16, dt=4e-5, device=DEV)
    results = []
    for ci, clip in enumerate(clips):
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
        n_frames = min(n_all, args.max_frames)
        tf = find_board(read(cap, 0))
        tracks, hub_i, _ = track_harness(cap, tf, n_frames, read, clip=clip)
        if tracks is None:
            print(f"[{ci+1}/{len(clips)}] {clip}: Harness nicht gefunden")
            cap.release()
            continue
        order = [hub_i] + [i for i in range(tracks.shape[1]) if i != hub_i]
        tracks = tracks[:, order, :]
        t0 = min(args.t0_buffer, max(0, n_frames // 4))
        t1 = n_frames - 1
        if t1 - t0 < args.stride * 2:
            print(f"[{ci+1}/{len(clips)}] {clip}: zu kurz")
            cap.release()
            continue
        tracked = HarnessTracker(tf, clip=clip).run(cap, read, tracks, range(t0, t1 + 1))
        cap.release()
        if t0 not in tracked or t1 not in tracked:
            print(f"[{ci+1}/{len(clips)}] {clip}: Rand nicht getrackt")
            continue
        try:
            n_nodes = len(tracked[t0][0]["shape_mm"])
            shp = {k: np.stack([tracked[f][k]["shape_mm"] for f in range(t0, t1 + 1)])
                   for k in range(4)}
        except KeyError:
            print(f"[{ci+1}/{len(clips)}] {clip}: Luecken im Tracking")
            continue

        horizon_s = (t1 - t0) / fps
        print(f"[{ci+1}/{len(clips)}] {clip}: LOCO-Training, "
              f"Horizont {horizon_s:.0f}s ...", flush=True)
        if args.tau:
            model = train_fold_seq(trans, clip, "data/nn_twoend_transformer.pt",
                                   args.epochs, 3e-4, tau=args.tau,
                                   noise_mm=args.noise_mm)
        else:
            model = train_fold(trans, clip, "data/nn_twoend_transformer.pt",
                               args.epochs, 3e-4, noise_mm=args.noise_mm,
                               pinn_lambda=args.pinn_lambda,
                               pinn_warmup=args.pinn_warmup,
                               exit_lambda=args.exit_lambda)

        cables = [shp[k] for k in range(4)]
        entry = {"clip": clip, "t0": t0, "t1": t1, "fps": fps,
                 "horizon_s": horizon_s, "by_kp": {}}
        for k_p in kps:
            V = roll_guided(model, sim, cables, n_nodes, args.stride,
                            k_p, args.guide_steps, exit_prior=args.exit_prior)
            errs = [1000.0 * float(np.linalg.norm(V[k] - cables[k][-1], axis=1).mean())
                    for k in range(4)]
            entry["by_kp"][k_p] = {"errs": errs, "avg": float(np.mean(errs)),
                                   "pred": V}
            tag = ("nur Physik" if k_p < 0 else
                   "ungeguidet" if k_p == 0 else f"k_p={k_p:g}")
            print(f"    {tag:<14} {np.mean(errs):6.1f} mm/Knoten  "
                  f"(M0-M3: {[round(e,1) for e in errs]})", flush=True)
        results.append(entry)

        outdir = (ROOT / args.out).with_suffix("")
        outdir.mkdir(parents=True, exist_ok=True)
        pos_kps = [k for k in kps if k > 0]
        if pos_kps:
            best_kp = min(pos_kps, key=lambda k: entry["by_kp"][k]["avg"])
            V = entry["by_kp"][best_kp]["pred"]
            render_lh(clip, tf, tracks, t0, t1, fps, shp,
                      {k: V[k] for k in range(4)},
                      {k: entry["by_kp"][best_kp]["errs"][k] for k in range(4)},
                      outdir)

    torch.save({"results": results, "kps": kps,
                "guide_steps": args.guide_steps}, ROOT / args.out)

    print(f"\n{'='*68}")
    print(f"{len(results)} Clips, Within-Run-Vergleich")
    base = np.array([r["by_kp"][0.0]["avg"] for r in results]) if 0.0 in kps else None
    for k_p in kps:
        per_clip = np.array([r["by_kp"][k_p]["avg"] for r in results])
        per_cab = np.array([e for r in results for e in r["by_kp"][k_p]["errs"]])
        tag = "ungeguidet" if k_p <= 0 else f"k_p={k_p:g}"
        line = (f"{tag:<12} Pro-Clip Median {np.median(per_clip):6.1f} Mittel "
                f"{per_clip.mean():6.1f} | Pro-Kabel Median {np.median(per_cab):6.1f} "
                f"Mittel {per_cab.mean():6.1f} mm/Knoten")
        if base is not None and k_p != 0:
            line += f" | besser bei {int((per_clip < base).sum())}/{len(results)} Clips"
        print(line)
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()

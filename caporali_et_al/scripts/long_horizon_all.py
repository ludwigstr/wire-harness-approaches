"""Long-horizon forecast for EVERY clip, LEAVE-ONE-CLIP-OUT.

For each clip: train the 1 s-step model on all OTHER clips' 1 s transitions,
then roll out from early in the clip (t0, a small settle buffer) to the last
trackable frame (t1) -- the maximum honest horizon that clip offers -- driven
by the REAL per-checkpoint pose of both driven ends (mover + hub), as
established in long_horizon.py. Reports the average deviation (model forecast
vs. tracked cable) per clip and overall.
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
from wire_model.perception import seg_masks as SM
from train_step_model import turning_angles
from eval_rollout import train_fold, step_action, NOMINAL

ROOT = Path(__file__).resolve().parents[1]
DEV = "cuda" if torch.cuda.is_available() else "cpu"
DIR = "data/WirePrediction"
COLORS = ["deepskyblue", "orange", "limegreen", "magenta"]


def render_lh(clip, tf, tracks, t0, t1, fps, shp, preds, errs, outdir):
    """Ausgangsbild | Prognose+Real ueber Skeleton+Maske, wie long_horizon.py."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    f0, f1 = read(cap, t0), read(cap, t1)
    cap.release()

    corners = np.array([[70, 70], [1130, 70], [1130, 1680], [70, 1680]])
    pxc = tf.to_px(corners)
    Hn, Wn = f0.shape[:2]
    x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
    x1, y1 = min(Wn, int(pxc[:, 0].max())), min(Hn, int(pxc[:, 1].max()))
    off = np.array([x0, y0])

    def px(s):
        return tf.to_px(s * 1000.0) - off

    def maskbg(fr, idx):
        skel = SM.skeleton_frame(clip, idx, fr)
        cm = SM.class_masks(clip, idx, fr)
        seg = cm[1] if cm is not None else None
        bg = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        if seg is not None:
            bg[seg[y0:y1, x0:x1]] = (75, 75, 75)
        if skel is not None:
            sc = cv2.dilate(skel[y0:y1, x0:x1].astype(np.uint8),
                            np.ones((3, 3), np.uint8), 1)
            bg[sc.astype(bool)] = (235, 235, 235)
        return bg

    photo0 = cv2.cvtColor(f0, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
    photo1 = (0.35 * cv2.cvtColor(f1, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
              + 60).clip(0, 255).astype(np.uint8)
    box = dict(boxstyle="round,pad=0.25", fc="black", alpha=0.65, ec="none")
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 15.2))
    ax[0][0].imshow(photo0);   ax[1][0].imshow(maskbg(f0, t0))
    ax[0][1].imshow(photo1);   ax[1][1].imshow(maskbg(f1, t1))
    for k in range(4):
        col = COLORS[k]
        ax[0][0].plot(px(shp[k][0])[:, 0], px(shp[k][0])[:, 1], "--", color=col, lw=2.6)
        ax[0][1].plot(px(shp[k][-1])[:, 0], px(shp[k][-1])[:, 1], ":", color=col,
                      lw=3.4, alpha=0.95)
        ax[0][1].plot(px(preds[k])[:, 0], px(preds[k])[:, 1], "--", color=col, lw=2.6)
        m1 = tracks[t1, k + 1] - off
        ax[0][1].plot([m1[0]], [m1[1]], "*", color=col, ms=18, mec="k")
        ax[0][1].text(m1[0] + 16, m1[1] - 12, f"M{k}  {errs[k]:.0f} mm/Kn.",
                      color=col, fontsize=8.5, weight="bold", bbox=box)
    for a_, fr_t in ((ax[0][0], t0), (ax[1][0], t0), (ax[0][1], t1), (ax[1][1], t1)):
        h = tracks[fr_t, 0] - off
        a_.plot([h[0]], [h[1]], "s", color="white", ms=12, mec="k")
    ax[0][1].legend(handles=[
        Line2D([0], [0], color="0.25", ls="--", lw=2.6, label="Prognose"),
        Line2D([0], [0], color="0.25", ls=":", lw=3.4, label="real gemessen")],
        loc="lower left", fontsize=9, framealpha=0.85)
    for row in ax:
        for a_ in row:
            a_.set_xticks([]); a_.set_yticks([])
    ax[0][0].set_title(f"Ausgangsbild t = {t0/fps:.0f} s", fontsize=11)
    ax[0][1].set_title(f"Prognose t = {t1/fps:.0f} s  ({(t1-t0)/fps:.0f} s Horizont)\n"
                       f"Abweichung {np.mean(list(errs.values())):.0f} mm/Knoten",
                       fontsize=11)
    ax[1][0].set_title(f"Skeleton + Maske t = {t0/fps:.0f} s", fontsize=10)
    ax[1][1].set_title(f"Skeleton + Maske t = {t1/fps:.0f} s", fontsize=10)
    fig.suptitle(f"{clip}  |  Langzeit-Prognose {t0/fps:.0f}->{t1/fps:.0f} s "
                 f"(echte Mover-Trajektorie)", fontsize=12, y=0.995)
    fig.tight_layout()
    out = outdir / f"lh_{clip}_{t0}_{t1}.png"
    fig.savefig(out, dpi=95, bbox_inches="tight")
    plt.close(fig)
    return out


def roll_traj(model, shapes, n_nodes, stride, rest_cap=0.35):
    """Roll out shapes[0] -> shapes[-1], stepping by `stride` frames, driven by
    the REAL per-step pose of both ends (mover + hub); only the interior shape
    is forecast, the driven ends are re-pinned to their known real pose."""
    idxs = list(range(0, len(shapes) - 1, stride)) + [len(shapes) - 1]
    idxs = sorted(set(idxs))
    V = shapes[0].copy().astype(np.float32)
    p = NOMINAL[None]
    for ai, bi in zip(idxs, idxs[1:]):
        a = step_action(shapes[ai], shapes[bi], n_nodes - 2, 1)
        r = np.clip(turning_angles(V), -rest_cap, rest_cap)[None].astype(np.float32)
        with torch.no_grad():
            V = model(torch.tensor(V[None], device=DEV),
                      torch.tensor(a[None], device=DEV),
                      torch.tensor(p, device=DEV),
                      torch.tensor(r, device=DEV))[0].cpu().numpy()
        V[0], V[-1] = shapes[bi][0], shapes[bi][-1]
    return V


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transitions", default="data/transitions_1s_all.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--stride", type=int, default=4, help="frames per step (4=1s @ 4fps)")
    ap.add_argument("--t0-buffer", type=int, default=8, help="frames to skip at clip start")
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--out", default="data/long_horizon_all.pt")
    ap.add_argument("--len-weight", type=float, default=0.0,
                    help="weight of the edge-length-consistency loss term "
                         "(0 = baseline, unchanged)")
    ap.add_argument("--shape-weight", type=float, default=0.0,
                    help="weight of PIEGraph's edge-vector shape loss "
                         "(0 = baseline, unchanged)")
    args = ap.parse_args()

    trans = torch.load(ROOT / args.transitions, weights_only=False)["transitions"]
    clips = sorted(set(t["clip"] for t in trans))
    print(f"{len(clips)} Clips, {len(trans)} 1s-Transitionen insgesamt\n", flush=True)

    results = []
    for ci, clip in enumerate(clips):
        vid = f"{DIR}/{clip}.mp4"
        cap = cv2.VideoCapture(vid)
        n_all = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 4.0
        n_frames = min(n_all, args.max_frames)
        tf = find_board(read(cap, 0))
        tracks, hub_i, _ = track_harness(cap, tf, n_frames, read, clip=clip)
        if tracks is None:
            print(f"[{ci+1}/{len(clips)}] {clip}: Harness nicht gefunden, uebersprungen")
            cap.release()
            continue
        order = [hub_i] + [i for i in range(tracks.shape[1]) if i != hub_i]
        tracks = tracks[:, order, :]
        t0 = min(args.t0_buffer, max(0, n_frames // 4))
        t1 = n_frames - 1
        if t1 - t0 < args.stride * 2:
            print(f"[{ci+1}/{len(clips)}] {clip}: zu kurz ({n_frames} Frames), uebersprungen")
            cap.release()
            continue
        tracked = HarnessTracker(tf, clip=clip).run(cap, read, tracks, range(t0, t1 + 1))
        cap.release()
        if t0 not in tracked or t1 not in tracked:
            print(f"[{ci+1}/{len(clips)}] {clip}: Rand nicht getrackt, uebersprungen")
            continue
        try:
            n_nodes = len(tracked[t0][0]["shape_mm"])
            shp = {k: np.stack([tracked[f][k]["shape_mm"] for f in range(t0, t1 + 1)])
                   for k in range(4)}
        except KeyError:
            print(f"[{ci+1}/{len(clips)}] {clip}: Luecken im Tracking, uebersprungen")
            continue

        horizon_s = (t1 - t0) / fps
        print(f"[{ci+1}/{len(clips)}] {clip}: LOCO-Modell trainieren, "
              f"Horizont {horizon_s:.0f}s ({t1-t0} Frames) ...", flush=True)
        model = train_fold(trans, clip, "data/nn_twoend_transformer.pt", args.epochs, 3e-4,
                          len_weight=args.len_weight, shape_weight=args.shape_weight)

        preds, errs_d = {}, {}
        for k in range(4):
            pred = roll_traj(model, shp[k], n_nodes, args.stride)
            preds[k] = pred
            errs_d[k] = 1000.0 * float(np.linalg.norm(pred - shp[k][-1], axis=1).mean())
        errs = [errs_d[k] for k in range(4)]
        avg = float(np.mean(errs))
        results.append({"clip": clip, "t0": t0, "t1": t1, "fps": fps,
                        "horizon_s": horizon_s, "errs": errs, "avg": avg})
        # keeps the existing baseline output untouched: only the default
        # --out path maps to the existing data/long_horizon_all directory,
        # any other --out (e.g. the len-weight variant) gets its own.
        outdir = (ROOT / args.out).with_suffix("")
        outdir.mkdir(parents=True, exist_ok=True)
        out_png = render_lh(clip, tf, tracks, t0, t1, fps, shp, preds, errs_d, outdir)
        print(f"    -> {horizon_s:.0f}s Horizont, Abweichung {avg:.1f} mm/Knoten "
              f"(M0-M3: {[round(e, 1) for e in errs]})  {out_png.name}", flush=True)

    torch.save({"results": results}, ROOT / args.out)
    all_avg = np.array([r["avg"] for r in results])
    all_cable = np.array([e for r in results for e in r["errs"]])
    hz = np.array([r["horizon_s"] for r in results])
    print(f"\n{'='*60}")
    print(f"{len(results)} Clips ausgewertet (von {len(clips)})")
    print(f"Horizont: Median {np.median(hz):.0f}s  Min {hz.min():.0f}s  Max {hz.max():.0f}s")
    print(f"Pro-Clip-Mittel (4 Kabel je Clip): "
          f"Median {np.median(all_avg):.1f}  Mittel {all_avg.mean():.1f} mm/Knoten")
    print(f"Pro-Kabel gesamt ({len(all_cable)} Kabel): "
          f"Median {np.median(all_cable):.1f}  Mittel {all_cable.mean():.1f} mm/Knoten")
    print(f"-> {ROOT / args.out}")


if __name__ == "__main__":
    main()

"""Split-screen style videos for a RopeDreamer checkpoint: the footage with the
measured routes (solid) and the open-loop prediction (dashed), one video per
clip, same look as the learned rows' videos (render_neural_video.py). The
first `--warm` frames are the warm-up (shown with a "warm-up" tag), the
prediction starts afterwards; both end nodes are pinned like in the tables.

    python3 scripts/ropedreamer_video.py --ckpt data/rd14/E_init_b01_pos.pt --tracks data/tracks14 \
        --video-dir data/WirePrediction --out data/ropedreamer_video/E_14clips
"""
import argparse, json, sys
from pathlib import Path
import cv2, numpy as np, torch
import imageio.v2 as iio

sys.path.insert(0, str(Path(__file__).resolve().parents[1])); sys.path.insert(0, str(Path(__file__).resolve().parent))
ROOT = Path(__file__).resolve().parents[1]
from fit_clip import read
from wire_model.perception.rectify import find_board
from mujoco_scene import COLORS
from render_neural_video import draw_poly
from wire_model.ropedreamer.model import RopeDreamer
from wire_model.ropedreamer.data import trajectory_tensors
import wire_model.ropedreamer.data as RD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tracks", required=True); ap.add_argument("--video-dir", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--warm", type=int, default=5); ap.add_argument("--fps-out", type=float, default=4.0)
    ap.add_argument("--clips", default=None)
    ap.add_argument("--truth-label", default="", help="name of the solid reference in the caption, e.g. 'solid = MuJoCo twin'")
    a = ap.parse_args(); dev = "cuda"
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False); RD.REPR = ck.get("repr", "chain")
    m = RopeDreamer(16, ck["size"], beta=ck["beta"], pos_weight=ck.get("pos_weight", 0), free_nats=ck.get("free_nats", 0),
                    rollout_weight=ck.get("rollout_weight", 0), warm=ck.get("warm", 5), repr=RD.REPR).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval(); off = ck["offset"].to(dev)
    out_dir = ROOT / a.out; out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    tracks = sorted(Path(ROOT / a.tracks).glob("*.npz"))
    if a.clips: tracks = [t for t in tracks if t.stem in a.clips.split(",")]
    for npz in tracks:
        clip = npz.stem; z = np.load(npz); t0, fps = int(z["t0"]), float(z["fps"])
        preds, truths = [], []
        for k in range(4):
            d = trajectory_tensors(z[f"c{k}"].astype(np.float32)); T = d["s"].shape[0]
            s = d["s"][None].to(dev) - off; idx = d["idx"][None].to(dev); dp = d["dp"][None].to(dev); P = d["P"].to(dev); ell = d["ell"].to(dev)
            with torch.no_grad():
                h, zz = m.warmup(s[:, :a.warm], idx[:, :a.warm], dp[:, :a.warm])
                pred = m.imagine(h, zz, idx[:, a.warm - 1:T - 1], dp[:, a.warm - 1:T - 1]) + off
            if m.repr == "chain": pred = pred.clone(); pred[0, :, :2] = P[a.warm:, 0]
            Po = m.positions(pred[0], ell)[..., :2].clone(); Po[:, 0] = P[a.warm:, 0]; Po[:, -1] = P[a.warm:, -1]
            preds.append(Po.cpu().numpy()); truths.append(d["P"].numpy())
        cap = cv2.VideoCapture(f"{a.video_dir}/{clip}.mp4"); tf = find_board(read(cap, 0))
        pxc = tf.to_px(np.array(((70, 70), (1130, 70), (1130, 1680), (70, 1680)), dtype=float))
        H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min())); x1, y1 = min(W, int(pxc[:, 0].max())), min(H, int(pxc[:, 1].max()))
        px = lambda P: tf.to_px(P * 1000.0) - np.array([x0, y0])
        frames, errs = [], []
        T = truths[0].shape[0]
        for f in range(T):
            fr = read(cap, t0 + f); vis = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1].copy()
            e = []
            for k in range(4):
                col = COLORS[k]; bgr = (int(255 * col[2]), int(255 * col[1]), int(255 * col[0]))
                draw_poly(vis, px(truths[k][f]), bgr, thick=3, marker=3)
                if f >= a.warm:
                    V = preds[k][f - a.warm]; draw_poly(vis, px(V), bgr, thick=2, marker=3, dashed=True)
                    e.append(float(np.linalg.norm(V - truths[k][f], axis=1).mean()) * 1000)
            tag = "warm-up (measured)" if f < a.warm else f"prediction  error {np.mean(e):.0f} mm/node"
            if f >= a.warm: errs.append(float(np.mean(e)))
            cv2.putText(vis, f"RopeDreamer  t={(t0+f)/fps:.1f}s  {tag}", (16, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
            if a.truth_label:
                cv2.putText(vis, a.truth_label, (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
            frames.append(vis)
        cap.release()
        h = min(x.shape[0] for x in frames); w = min(x.shape[1] for x in frames)
        vp = out_dir / f"vergleich_{clip}.mp4"; iio.mimsave(vp, [x[:h, :w] for x in frames], fps=a.fps_out)
        results[clip] = {"start": errs[0], "ende": errs[-1], "mittel": float(np.mean(errs)), "n_frames": len(errs), "video": str(vp)}
        print(f"{clip}: {len(errs)} predicted frames, Mittel {np.mean(errs):.1f} -> {vp}", flush=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    v = np.array([r["mittel"] for r in results.values()]); print(f"{len(v)} clips, Mittel {v.mean():.2f}, Median {np.median(v):.2f}")


if __name__ == "__main__":
    main()

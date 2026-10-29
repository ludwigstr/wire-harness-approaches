"""Overlay videos for Yang et al. (summary row 4): the IN-biLSTM checkpoint rolled out exactly like
yang_eval.py's "mittel_warm5" (frames 0-4 measured, open loop from frame 4 with the measured end motions,
ends pinned), drawn on the footage: measured solid, prediction dashed. The per-clip mean must reproduce
the eval JSON's "mittel_warm5".

    python scripts/yang_video.py --ckpt data/wm/yang_newton_paper3b_inbilstm.pt \
        --eval data/wm/yang_newton_paper3b_inbilstm_14.json --out data/figures/table_videos_2026-10-28/4_yang_17.0
"""
import argparse, json, sys
from pathlib import Path
import cv2, imageio.v2 as iio, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from fit_clip import read  # noqa: E402
from wire_model.perception.rectify import find_board  # noqa: E402
from wire_model.yang.model import INBiLSTM  # noqa: E402
from render_neural_video import DIR, COLORS, draw_poly  # noqa: E402
from yang_eval import TRACKS, rollout  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--eval", required=True)
    ap.add_argument("--warm", type=int, default=5); ap.add_argument("--fps-out", type=float, default=4.0)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = a.device
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"{out} exists -- never overwrite rendered videos, pick a new --out")
    out.mkdir(parents=True)
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    net = INBiLSTM(use_rest=ck["use_rest"]).to(dev); net.load_state_dict(ck["state_dict"]); net.eval()
    ev = json.loads((ROOT / a.eval).read_text())["clips"]
    results = {}
    for clip in sorted(ev):
        z = np.load(TRACKS / f"{clip}.npz"); t0, fps = int(z["t0"]), float(z["fps"])
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        pred = rollout(net, P, ck["dt"], warm=a.warm).cpu().numpy(); real = P.cpu().numpy(); T = P.shape[1]
        cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4"); tf = find_board(read(cap, 0))
        pxc = tf.to_px(np.array(((70, 70), (1130, 70), (1130, 1680), (70, 1680)), dtype=float))
        Hh, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
        x1, y1 = min(W, int(pxc[:, 0].max())), min(Hh, int(pxc[:, 1].max()))
        px = lambda s: tf.to_px(s * 1000.0) - np.array([x0, y0])
        imgs, errs = [], []
        for f in range(T):
            vis = cv2.cvtColor(read(cap, t0 + f), cv2.COLOR_BGR2RGB)[y0:y1, x0:x1].copy()
            e = []
            for k in range(4):
                col = COLORS[k]; bgr = (int(255 * col[2]), int(255 * col[1]), int(255 * col[0]))
                draw_poly(vis, px(real[k, f]), bgr, thick=3, marker=3)
                if f >= a.warm:
                    draw_poly(vis, px(pred[k, f]), bgr, thick=2, marker=3, dashed=True)
                    e.append(float(np.linalg.norm(pred[k, f] - real[k, f], axis=1).mean()) * 1000)
            tag = "warm-up (measured)" if f < a.warm else f"prediction  error {np.mean(e):.0f} mm/node"
            if f >= a.warm:
                errs.append(float(np.mean(e)))
            cv2.putText(vis, f"Yang et al. (IN + bi-LSTM)  t={(t0 + f) / fps:.1f}s  {tag}", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(vis, "solid = measured, dashed = Yang et al.", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (20, 20, 20), 2, cv2.LINE_AA)
            imgs.append(vis)
        cap.release()
        h = min(x.shape[0] for x in imgs); w = min(x.shape[1] for x in imgs)
        vp = out / f"vergleich_{clip}.mp4"; iio.mimsave(vp, [x[:h, :w] for x in imgs], fps=a.fps_out)
        results[clip] = {"mittel": float(np.mean(errs)), "json_mittel_warm5": ev[clip]["mittel_warm5"],
                         "n_frames": len(errs), "video": str(vp)}
        print(f"{clip}: Mittel {np.mean(errs):.2f} (JSON {ev[clip]['mittel_warm5']:.2f}) -> {vp}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    v = np.array([x["mittel"] for x in results.values()])
    print(f"{len(v)} clips, Mittel {v.mean():.2f}, Median {np.median(v):.2f}")


if __name__ == "__main__":
    main()

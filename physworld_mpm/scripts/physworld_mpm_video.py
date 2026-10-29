"""Overlay videos for the PhysWorld MPM rebuild (summary row 3): the saved per-clip GNN
(data/wm/physworld_mpm_gnn/<clip>.pt) with the fitted Phi from its JSON, rolled out exactly like
physworld_mpm_gnn.finetune_and_eval's evaluation (5-frame warm-up with the measured states, open loop
driven by the measured end motions, ends pinned), drawn on the footage: measured solid, prediction
dashed. The per-clip mean must reproduce the JSON's "mittel".

    python scripts/physworld_mpm_video.py --out data/figures/table_videos_2026-10-28/8_physworld_9.5
"""
import argparse, json, sys
from pathlib import Path
import cv2, imageio.v2 as iio, numpy as np, torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]
from fit_clip import read  # noqa: E402
from wire_model.perception.rectify import find_board  # noqa: E402
from wire_model.physworld_mpm.gnn import PhysWorldGNN  # noqa: E402
from render_neural_video import DIR, COLORS, draw_poly  # noqa: E402
import physworld_mpm_gnn as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gnn", default="data/wm/physworld_mpm_gnn"); ap.add_argument("--tracks", default="data/tracks14")
    ap.add_argument("--warm", type=int, default=5); ap.add_argument("--fps-out", type=float, default=4.0)
    ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args(); dev = a.device
    out = ROOT / a.out
    if out.exists():
        raise SystemExit(f"{out} exists -- never overwrite rendered videos, pick a new --out")
    out.mkdir(parents=True)
    results = {}
    for js in sorted((ROOT / a.gnn).glob("*.json")):
        clip = js.stem; r = json.loads(js.read_text())
        ck = torch.load(ROOT / a.gnn / f"{clip}.pt", map_location=dev, weights_only=False)
        model = PhysWorldGNN(16, G.H).to(dev); model.load_state_dict(ck["state_dict"]); model.eval()
        mu, sd = ck["phi_mu"].to(dev), ck["phi_sd"].to(dev)
        z = np.load(ROOT / a.tracks / f"{clip}.npz"); t0, fps = int(z["t0"]), float(z["fps"])
        P = torch.tensor(np.stack([z[f"c{k}"] for k in range(4)]), dtype=torch.float32, device=dev)
        Y = G.real_yaw(P); T = P.shape[1]
        pred = []
        with torch.no_grad():
            for c in range(4):
                X = P[c]
                dp = G.dp_of(X[:-1], X[1:])
                dy = G.dyaw_nodes((Y[c, 1:] - Y[c, :-1]) if r["use_yaw"] else torch.zeros_like(Y[c, 1:]))
                phi = (torch.tensor(r["phi_fitted"][c], dtype=torch.float32, device=dev) - mu) / sd
                frames = [X[t] for t in range(a.warm)]
                for t in range(a.warm - 1, T - 1):
                    h = torch.stack([frames[max(0, t - G.H + j)] for j in range(G.H)])
                    pr = model(h[None], frames[t][None], phi[None], dp[t][None], dy[t][None])[0]
                    pr[0] = X[t + 1, 0]; pr[-1] = X[t + 1, -1]
                    frames.append(pr)
                pred.append(torch.stack(frames).cpu().numpy())
        real = P.cpu().numpy()
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
                    draw_poly(vis, px(pred[k][f]), bgr, thick=2, marker=3, dashed=True)
                    e.append(float(np.linalg.norm(pred[k][f] - real[k, f], axis=1).mean()) * 1000)
            tag = "warm-up (measured)" if f < a.warm else f"prediction  error {np.mean(e):.0f} mm/node"
            if f >= a.warm:
                errs.append(float(np.mean(e)))
            cv2.putText(vis, f"PhysWorld (MPM twin + GNN)  t={(t0 + f) / fps:.1f}s  {tag}", (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(vis, "solid = measured, dashed = PhysWorld", (16, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (20, 20, 20), 2, cv2.LINE_AA)
            imgs.append(vis)
        cap.release()
        h = min(x.shape[0] for x in imgs); w = min(x.shape[1] for x in imgs)
        vp = out / f"vergleich_{clip}.mp4"; iio.mimsave(vp, [x[:h, :w] for x in imgs], fps=a.fps_out)
        results[clip] = {"mittel": float(np.mean(errs)), "json_mittel": r["mittel"], "n_frames": len(errs), "video": str(vp)}
        print(f"{clip}: Mittel {np.mean(errs):.2f} (JSON {r['mittel']:.2f}) -> {vp}", flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    v = np.array([x["mittel"] for x in results.values()])
    print(f"{len(v)} clips, Mittel {v.mean():.2f}, Median {np.median(v):.2f}")


if __name__ == "__main__":
    main()

"""AdaptiGraph on a clip set: few-shot estimation of phi on the first `--warm`
frames (paper Sec. III-C: minimise the prediction error over the observed
interactions, phi in the extended range [-0.2, 1.2]; 1-D -> dense grid in
place of BO), then open-loop rollout from the warm-up with the measured end
motions and the model's own history. Same scoring as RopeDreamer (frames
after the warm-up, both ends pinned) -> X-Planar-format JSON.
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from wire_model.adaptigraph import AdaptiGraph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--tracks", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=5); ap.add_argument("--device", default="cuda")
    ap.add_argument("--phi-fixed", type=float, default=None, help="skip the estimation and use this phi")
    a = ap.parse_args(); dev = a.device
    ck = torch.load(ROOT / a.ckpt, map_location=dev, weights_only=False)
    m = AdaptiGraph(16, ck["n_hist"], radius=ck["radius"]).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
    H = ck["n_hist"]; grid = torch.linspace(-0.2, 1.2, 29)
    res, phis = {}, {}
    for npz in sorted(Path(ROOT / a.tracks).glob("*.npz")):
        z = np.load(npz); clip = npz.stem; errs_c = []; phi_c = []
        for k in range(4):
            P = torch.tensor(z[f"c{k}"], dtype=torch.float32, device=dev); T = P.shape[0]
            def hist_at(t, frames):           # the H frames before t from `frames` (padded by repeating the first)
                idx = [max(0, t - H + j) for j in range(H)]; return torch.stack([frames[i] for i in idx])
            def dp_at(t):
                d = torch.zeros(16, 2, device=dev); d[0] = P[t + 1, 0] - P[t, 0]; d[-1] = P[t + 1, -1] - P[t, -1]; return d
            with torch.no_grad():
                if a.phi_fixed is None:
                    # few-shot: one-step errors on the warm-up transitions for every phi on the grid
                    ts = list(range(0, a.warm - 1))
                    hb = torch.stack([hist_at(t, P) for t in ts]); xb = P[ts]; yb = P[[t + 1 for t in ts]]; db = torch.stack([dp_at(t) for t in ts])
                    cost = []
                    for ph in grid:
                        pred = m(hb, xb, torch.full((len(ts),), float(ph), device=dev), db)
                        cost.append((pred[:, 1:-1] - yb[:, 1:-1]).norm(dim=-1).mean().item())
                    phi = float(grid[int(np.argmin(cost))])
                else:
                    phi = a.phi_fixed
                # open loop from frame warm-1 with own predictions as history
                frames = [P[t] for t in range(a.warm)]; errs = []
                for t in range(a.warm - 1, T - 1):
                    h = hist_at(t, frames); x = frames[t]
                    pred = m(h[None], x[None], torch.tensor([phi], device=dev), dp_at(t)[None])[0]
                    pred[0] = P[t + 1, 0]; pred[-1] = P[t + 1, -1]
                    frames.append(pred); errs.append((pred - P[t + 1]).norm(dim=-1).mean().item() * 1000)
            errs_c.append(np.array(errs)); phi_c.append(phi)
        e = np.mean(errs_c, 0)
        res[clip] = {"start": float(e[0]), "ende": float(e[-1]), "mittel": float(e.mean()), "n_frames": int(len(e)), "video": "", "phi": phi_c}
        print(f"{clip}: phi {np.round(phi_c, 2).tolist()}  Mittel {e.mean():.1f}", flush=True)
    v = np.array([r["mittel"] for r in res.values()])
    (ROOT / a.out).write_text(json.dumps(res, indent=1))
    print(f"{a.ckpt} on {a.tracks}: {len(v)} clips, Mittel {v.mean():.2f}, Median {np.median(v):.2f} mm/node (frames after a {a.warm}-frame warm-up); phi estimated per cable")
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()

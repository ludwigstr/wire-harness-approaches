"""Drive the recorded plug motion through the MuJoCo scene and score the result.

This is the MuJoCo counterpart of the long-horizon evaluations: the two driven
ends of every cable follow their REAL measured trajectory (the same oracle the
neural rollouts get), nothing else is prescribed, and the wire in between is
left to the simulator. The final shapes are then compared against the tracked
ones with the same metric -- mean per-node distance in mm -- so the number sits
directly next to the 32.6 mm/node the guided Transformer reaches.

Two details decide whether the comparison is fair rather than flattering:

  * The plug pose is ramped across the 125 solver steps that make up one 4 fps
    video frame, not snapped at the frame boundary. Snapping injects a velocity
    step the real motion never contains, and the cable answers it with a whip
    that has nothing to do with the recorded dynamics.
  * The hub is driven too. It moves further than the movers do in this footage,
    so a fixed hub would be modelling a different experiment.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mujoco_scene import track_clip, build_xml

ROOT = Path(__file__).resolve().parents[1]


def cable_vertices(m, d, k, n_nodes=16):
    """The cable's n_nodes support points: 15 body origins + the last capsule
    tip. The composite makes one body per SEGMENT, so the final vertex is not a
    body at all and has to be reconstructed from the last capsule's axis --
    leaving it out shortens the curve by a segment and, after arc-length
    resampling, fakes a systematic half-segment error."""
    import mujoco
    names = ([f"Wire{k}B_first"]
             + [f"Wire{k}B_{i}" for i in range(1, n_nodes - 2)]
             + [f"Wire{k}B_last"])
    ids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in names])
    if (ids < 0).any():
        raise RuntimeError(f"cable {k}: unresolved bodies {names}")
    P = d.xpos[ids].copy()
    g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == ids[-1]][0]
    tip = P[-1] + d.xmat[ids[-1]].reshape(3, 3) @ np.array([2 * m.geom_size[g][1], 0, 0])
    return np.vstack([P, tip])


def replay(m, d, shp, n_frames, sub):
    """Step the scene along the recorded trajectory, returning the simulated
    shapes after every video frame.

    The movers are driven through their position actuators, not teleported:
    two slide servos carry each platform and a third turns the MODULE on top
    of it to the heading the footage shows. That heading is continuous -- the
    module genuinely rotates on its mover -- while the platform stays square
    to the board, which is what the photographs show and what an earlier
    mocap version got wrong in both directions at once.
    """
    import mujoco
    from mujoco_scene import HUB_CAVS, PLUG_CAV, _rotz

    names = ["hub"] + [f"plug{k}" for k in range(4)]
    jid = {n: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_{a}a")
               for a in ("x", "y", "t")] for n in names}
    base = {n: m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)][:2].copy()
            for n in names}
    yaw0 = {}
    for n in names:
        b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{n}_mod")
        q = m.body_quat[b]
        yaw0[n] = 2.0 * np.arctan2(q[3], q[0])

    def unwrap(prev, cur):
        """Servo targets must be continuous: a heading that wraps from +pi to
        -pi would command a full turn the module never makes."""
        return prev + np.arctan2(np.sin(cur - prev), np.cos(cur - prev))

    def poses(t):
        r2 = [shp[k][t] for k in range(4)]
        out = np.mean([P[1] - P[0] for P in r2], axis=0)
        hy = np.arctan2(out[1], out[0]) + np.pi / 2
        hoff = _rotz(np.array(HUB_CAVS).mean(0), hy)
        res = {"hub": (np.array([r2[0][0][0] - hoff[0], r2[0][0][1] - hoff[1]]), hy)}
        for k, P in enumerate(r2):
            v = P[-1] - P[-2]
            y = np.arctan2(v[1], v[0]) + np.pi / 2  # Modul liegt HINTER der Kavitaet (vom Kabel weg), nicht davor
            o = _rotz(np.array(PLUG_CAV), y)
            res[f"plug{k}"] = (np.array([P[-1][0] - o[0], P[-1][1] - o[1]]), y)
        return res

    seq = []
    prev = {n: poses(0)[n][1] for n in names}
    for t in range(n_frames):
        p = poses(t)
        for n in names:
            prev[n] = unwrap(prev[n], p[n][1])
        seq.append({n: (p[n][0], prev[n]) for n in names})

    def apply(ctrl_pose):
        for n in names:
            xy, yaw = ctrl_pose[n]
            ax, ay, at = jid[n]
            d.ctrl[ax] = xy[0] - base[n][0]
            d.ctrl[ay] = xy[1] - base[n][1]
            d.ctrl[at] = yaw - yaw0[n]

    apply(seq[0])
    for _ in range(300):                 # let the servos settle on frame 0
        mujoco.mj_step(m, d)

    out_shapes = []
    for t in range(n_frames - 1):
        a0, a1 = seq[t], seq[t + 1]
        for s_ in range(sub):
            f = (s_ + 1) / sub           # ramp, never snap
            apply({n: (a0[n][0] + f * (a1[n][0] - a0[n][0]),
                       a0[n][1] + f * (a1[n][1] - a0[n][1])) for n in names})
            mujoco.mj_step(m, d)
        out_shapes.append(np.stack([cable_vertices(m, d, k) for k in range(4)]))
    return np.stack(out_shapes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="20260602_114019")
    ap.add_argument("--t0", type=int, default=8)
    ap.add_argument("--bend", type=float, default=1.0e5)
    ap.add_argument("--twist", type=float, default=1.0e5)
    ap.add_argument("--damping", type=float, default=0.15)
    ap.add_argument("--mu", type=float, default=1.0)
    ap.add_argument("--out", default="data/mujoco")
    args = ap.parse_args()

    import mujoco
    shp, t0, t1, fps = track_clip(args.clip, t0=args.t0)
    n_frames = len(shp[0])
    xml = build_xml(shp, bend=args.bend, twist=args.twist,
                    damping=args.damping, mu=args.mu)
    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    xml_path = outdir / f"scene_{args.clip}.xml"
    xml_path.write_text(xml)

    m = mujoco.MjModel.from_xml_path(str(xml_path))
    d = mujoco.MjData(m)
    sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))
    print(f"{args.clip}: {n_frames} Frames ({(n_frames-1)/fps:.0f}s), "
          f"{sub} Solver-Schritte je Frame, bend={args.bend:g}\n", flush=True)

    traj = replay(m, d, shp, n_frames, sub)
    np.save(outdir / f"replay_{args.clip}.npy", traj)

    errs = []
    print("Kabel   Fehler am Ende   Laenge sim/real (mm)")
    for k in range(4):
        sim = traj[-1][k][:, :2]
        real = shp[k][-1]
        e = 1000 * np.linalg.norm(sim - real, axis=1).mean()
        errs.append(e)
        Ls = 1000 * np.linalg.norm(np.diff(traj[-1][k], axis=0), axis=1).sum()
        Lr = 1000 * np.linalg.norm(np.diff(real, axis=0), axis=1).sum()
        print(f"  M{k}     {e:6.1f} mm        {Ls:6.1f} / {Lr:6.1f}")
    print(f"\nMuJoCo-Replay: {np.mean(errs):.1f} mm/Knoten  "
          f"(gefuehrtes Prognosemodell auf allen 19 Clips: 32.6)")
    print(f"-> {xml_path}")


if __name__ == "__main__":
    main()

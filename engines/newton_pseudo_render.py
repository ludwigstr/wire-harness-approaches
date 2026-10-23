"""Two-panel comparison video for the Newton cable model, same layout as
mujoco_video.py/genesis_full_scene.py (real photo left, rendered scene
right) -- except Newton's real 3D renderer is blocked in this environment
(ViewerRTX needs the proprietary ovrtx package, which needs a newer
libstdc++ than available; ViewerGL needs a real/virtual X display, which
needs `sudo apt install xvfb`, blocked on a missing password). This fakes a
"rendered" look with matplotlib patches instead: rounded coloured mover
boxes with a drop shadow and a simple highlight bevel, thick shaded cable
lines -- not a real render, just enough shading to read as one at a glance.

Movers are NOT part of the Newton physics model here (newton_full_scene.py's
eval_clip only builds cables, driving their ends straight to the raw tracked
points) -- their poses for drawing are computed with the exact same formula
genesis_full_scene.py uses, purely for this visualisation.
"""
import sys
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mujoco_scene import (track_clip, _rotz, DIR, COLORS, BOARD_W, BOARD_H,
                          HUB_CAVS, PLUG_CAV, CAV_Z, PLATFORM_HALF, MODULE_HALF)
from newton_full_scene import make_kinematic

ROOT = Path(__file__).resolve().parents[1]
HUB_RGBA = (0.85, 0.15, 0.15)
GREY = (0.55, 0.55, 0.55)
BOARD_RGBA = (0.80, 0.79, 0.75)


def mover_poses(shp, t):
    """(pos_xy, yaw) for hub + 4 plugs at frame t -- identical formula to
    genesis_full_scene.py's poses(), just returning plain numpy here since
    this is only ever used for drawing, not driving physics."""
    r2 = [shp[k][t] for k in range(4)]
    out = np.mean([P[1] - P[0] for P in r2], axis=0)
    hy = np.arctan2(out[1], out[0]) + np.pi / 2
    hoff = _rotz(np.array(HUB_CAVS).mean(0), hy)
    res = {"hub": (np.array([r2[0][0][0] - hoff[0], r2[0][0][1] - hoff[1]]), hy)}
    for k, P in enumerate(r2):
        v = P[-1] - P[-2]
        y = np.arctan2(v[1], v[0]) + np.pi / 2
        o = _rotz(np.array(PLUG_CAV), y)
        res[f"plug{k}"] = (np.array([P[-1][0] - o[0], P[-1][1] - o[1]]), y)
    return res


def draw_box(ax, center, half, yaw, facecolor, zorder):
    from matplotlib.patches import Polygon
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    local = np.array([[-half, -half], [half, -half], [half, half], [-half, half]])
    corners = local @ R.T + center
    shadow = corners + np.array([0.006, -0.006])
    ax.add_patch(Polygon(shadow, closed=True, facecolor="black", alpha=0.28,
                         linewidth=0, zorder=zorder - 0.2))
    ax.add_patch(Polygon(corners, closed=True, facecolor=facecolor,
                         edgecolor=np.array(facecolor) * 0.5, linewidth=1.1, zorder=zorder))
    hl = 0.55 * corners[3] + 0.45 * center
    hl_pts = np.array([corners[3], 0.5 * (corners[3] + corners[0]), hl, 0.5 * (corners[3] + corners[2])])
    ax.add_patch(Polygon(hl_pts, closed=True, facecolor="white", alpha=0.20,
                         linewidth=0, zorder=zorder + 0.1))


def draw_mover(ax, name, pos, yaw):
    color = HUB_RGBA if name == "hub" else COLORS[int(name[-1])]
    draw_box(ax, pos, PLATFORM_HALF[0], 0.0, color, zorder=2)
    draw_box(ax, pos, MODULE_HALF, yaw, GREY, zorder=3)


def draw_cable(ax, pts, color, zorder):
    ax.plot(pts[:, 0], pts[:, 1], "-", color="black", alpha=0.25, linewidth=9,
           zorder=zorder - 0.1, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(pts[:, 0], pts[:, 1], "-", color=color, linewidth=6.5,
           zorder=zorder, solid_capstyle="round", solid_joinstyle="round")
    ax.plot(pts[:, 0], pts[:, 1], "-", color="white", alpha=0.35, linewidth=2.2,
           zorder=zorder + 0.1, solid_capstyle="round", solid_joinstyle="round")


def render_pseudo_video(clip, t1_sec=None, dt=1e-3, steps_per_frame=None,
                        radius=0.003, stretch_stiffness=1.0e5, bend_stiffness=0.5,
                        bend_damping=0.05, fps_out=4.0,
                        out_dir="data/newton_pseudo_video"):
    import newton
    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import imageio.v2 as iio
    from fit_clip import read
    from wire_model.perception.rectify import find_board
    from newton.utils import create_parallel_transport_cable_quaternions

    t1 = None
    if t1_sec is not None:
        cap_probe = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps_probe = (cap_probe.get(cv2.CAP_PROP_FPS) if cap_probe.isOpened() else 0) or 4.0
        cap_probe.release()
        t1 = int(round(t1_sec * fps_probe))
    shp, t0, t1, fps = track_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])
    if steps_per_frame is None:
        steps_per_frame = max(1, round((1.0 / fps) / dt))

    def route3(k, t):
        P = shp[k][t].astype(float)
        z = np.full((len(P), 1), CAV_Z)
        return np.hstack([P, z])

    builder = newton.ModelBuilder()
    builder.default_shape_cfg.mu = 1.0
    cable_bodies = []
    for k in range(4):
        positions = [wp.vec3(*p) for p in route3(k, 0)]
        bodies, _ = builder.add_rod(
            positions=positions, radius=radius, stretch_stiffness=stretch_stiffness,
            bend_stiffness=bend_stiffness, bend_damping=bend_damping,
            body_frame_origin="start", label=f"cable{k}")
        make_kinematic(builder, bodies[0])
        make_kinematic(builder, bodies[-1])
        cable_bodies.append(bodies)
    builder.color()
    model = builder.finalize()
    model.set_gravity((0.0, 0.0, -9.81))

    collision_pipeline = newton.CollisionPipeline(model)
    solver = newton.solvers.SolverVBD(model, iterations=5)
    state0, state1 = model.state(), model.state()
    control = model.control()
    contacts = collision_pipeline.contacts()

    def pin_ends(t):
        bq = state0.body_q.numpy()
        for k in range(4):
            r = route3(k, t)
            pts = [wp.vec3(*p) for p in r]
            edge_q = create_parallel_transport_cable_quaternions(pts)
            bodies = cable_bodies[k]
            bq[bodies[0]] = np.concatenate([r[0], np.array(list(edge_q[0]))])
            bq[bodies[-1]] = np.concatenate([r[-2], np.array(list(edge_q[-1]))])
        state0.body_q.assign(bq)
        state1.body_q.assign(bq)

    def step_physics():
        nonlocal state0, state1
        state0.clear_forces()
        collision_pipeline.collide(state0, contacts)
        solver.step(state0, state1, control, contacts, dt)
        state0, state1 = state1, state0

    pin_ends(0)
    for _ in range(300):
        step_physics()

    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    tf = find_board(read(cap, 0))
    corners_mm = ((70, 70), (1130, 70), (1130, 1680), (70, 1680))
    pxc = tf.to_px(np.array(corners_mm, dtype=float))
    H, W = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
    x1, y1 = min(W, int(pxc[:, 0].max())), min(H, int(pxc[:, 1].max()))

    def px(shape_m):
        return tf.to_px(shape_m * 1000.0) - np.array([x0, y0])

    crop_h, crop_w = y1 - y0, x1 - x0
    panel_h_in = 7.0
    panel_w_in = panel_h_in * crop_w / crop_h
    figsize = (2 * panel_w_in + 0.6, panel_h_in + 0.5)

    outdir = ROOT / out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / clip
    frames_dir.mkdir(exist_ok=True)

    errs, frame_paths = [], []
    for t in range(n_frames):
        pin_ends(t)
        for _ in range(steps_per_frame):
            step_physics()

        sim_shapes = []
        bq = state0.body_q.numpy()
        for k in range(4):
            sim_shapes.append(bq[cable_bodies[k]][:, :2])
        e_all = [float(np.linalg.norm(sim_shapes[k] - shp[k][t][:len(cable_bodies[k])], axis=1).mean()) * 1000
                for k in range(4)]
        errs.append(float(np.mean(e_all)))

        fr = read(cap, t0 + t)
        import cv2 as _cv2
        photo = _cv2.cvtColor(fr, _cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(photo)
        for k in range(4):
            col = COLORS[k]
            P, Q = px(shp[k][t]), px(sim_shapes[k])
            ax[0].plot(P[:, 0], P[:, 1], ":", color=col, lw=3.2, alpha=0.95)
            ax[0].plot(Q[:, 0], Q[:, 1], "--", color=col, lw=2.4)
        ax[0].legend(handles=[
            Line2D([0], [0], color="0.25", ls=":", lw=3.2, label="real gemessen"),
            Line2D([0], [0], color="0.25", ls="--", lw=2.4, label="Newton")],
            loc="lower left", fontsize=9, framealpha=0.85)
        ax[0].set_title(f"Video t = {(t0+t)/fps:.2f} s   Abweichung {errs[-1]:.0f} mm/Knoten", fontsize=11)
        ax[0].set_xticks([]); ax[0].set_yticks([])

        ax[1].add_patch(plt.Rectangle((0, 0), BOARD_W, BOARD_H, facecolor=BOARD_RGBA, zorder=0))
        mp = mover_poses(shp, t)
        for name, (pos, yaw) in mp.items():
            draw_mover(ax[1], name, pos, yaw)
        for k in range(4):
            col = COLORS[k]
            draw_cable(ax[1], sim_shapes[k], col, zorder=1)
        ax[1].set_xlim(0, BOARD_W); ax[1].set_ylim(0, BOARD_H)
        ax[1].set_aspect("equal")
        ax[1].set_facecolor("#e8e8e4")
        ax[1].set_xticks([]); ax[1].set_yticks([])
        ax[1].set_title("Newton-Pseudo-Rendering (matplotlib)", fontsize=11)

        fig.suptitle(f"{clip}  |  Frame {t+1}/{n_frames}", fontsize=12)
        fig.tight_layout()
        p = frames_dir / f"f{t:03d}.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close(fig)
        frame_paths.append(p)
    cap.release()

    pngs = [iio.imread(p) for p in frame_paths]
    h = min(p.shape[0] for p in pngs); w = min(p.shape[1] for p in pngs)
    video_path = outdir / f"vergleich_{clip}.mp4"
    iio.mimsave(video_path, [p[:h, :w] for p in pngs], fps=fps_out)

    print(f"{clip}: Start {errs[0]:.1f}  Ende {errs[-1]:.1f}  Mittel {np.mean(errs):.1f} mm/Knoten")
    print(f"-> {video_path}")
    return video_path, errs


if __name__ == "__main__":
    render_pseudo_video("20260602_170108", t1_sec=9)

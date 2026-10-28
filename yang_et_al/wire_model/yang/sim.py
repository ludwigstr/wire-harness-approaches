"""The black-box simulator of Yang, Stork, Stoyanov (RAS 158, 2022) -- MuJoCo here.

The paper identifies ten AGX Dynamics DLO parameters (Table 1: Young's modulus,
Poisson ratio and damping for stretch / bend / twist, plus mass). We have no
AGX; the stand-in is the project's MuJoCo harness scene (scripts/mujoco_scene.py:
four `elasticity.cable` composites -- a discrete elastic rod -- between a PD-driven
hub and four PD-driven plugs). One global value per parameter, as in the paper.

Mapping (THETA_NAMES, all searched in log10):
    E_bend    plugin config "bend"   (Young's modulus, Pa)        <- E_bend
    G_twist   plugin config "twist"  (shear modulus, Pa)          <- E_twist and the
              Poisson ratios: MuJoCo takes G directly, nu = E/(2G) - 1 is implied
    zeta_bend dof_damping of the two bending dofs of every cable ball joint
              (local y, z; N m s / rad)                           <- zeta_bend
    zeta_twist dof_damping of the twist dof (local x, along the cable) <- zeta_twist
    mass      capsule density, given as kg per metre of cable     <- mass
    mu        sliding friction of the cable capsules (cable-board / cable-cable);
              not in the paper's Table 1, but the cable lies on a board here
NOT adjustable: everything in stretch (E_stretch, nu_stretch, zeta_stretch). The
MuJoCo cable is a chain of rigid capsules joined by ball joints -- inextensible
by construction, so there is no stretch degree of freedom to identify.

Rest shape: the cable plugin's stress-free configuration is the compiled one,
i.e. the measured frame 0 of the clip (as in every MuJoCo row of the table).
"""
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

THETA_NAMES = ["E_bend", "G_twist", "zeta_bend", "zeta_twist", "mass_per_m", "mu"]
# log10 bounds for differential evolution
THETA_BOUNDS = [(3.0, 7.0),            # E_bend, Pa  (scene default 1e5)
                (3.0, 7.0),            # G_twist, Pa (scene default 1e5)
                (-4.0, 0.7),           # zeta_bend   (scene default 0.15)
                (-4.0, 0.7),           # zeta_twist  (scene default 0.15)
                (np.log10(0.01), np.log10(0.5)),   # kg/m (measured 0.044, scene 0.15)
                (np.log10(0.05), np.log10(5.0))]   # mu   (scene 1.0)
THETA_DEFAULT = np.log10([1e5, 1e5, 0.15, 0.15, 0.15, 1.0])
RADIUS = 0.003
N_NODES = 16
NAMES = ["hub"] + [f"plug{k}" for k in range(4)]


def theta_dict(x):
    v = 10.0 ** np.asarray(x, float)
    return dict(zip(THETA_NAMES, v.tolist()))


def make_model(shp, x, frame_idx=0):
    """compiled scene of one clip (cables built on frame `frame_idx`) with parameters x (log10)"""
    import mujoco
    from mujoco_scene import build_xml, cable_ball_joints
    p = theta_dict(x)
    xml = build_xml(shp, frame_idx=frame_idx, radius=RADIUS, bend=p["E_bend"], twist=p["G_twist"],
                    mass_per_m=p["mass_per_m"], mu=p["mu"])
    m = mujoco.MjModel.from_xml_string(xml)
    ids = cable_ball_joints(m)
    for c in range(4):
        for j in range(ids.shape[1]):
            a = m.jnt_dofadr[ids[c, j]]
            m.dof_damping[a] = p["zeta_twist"]           # local x = cable axis
            m.dof_damping[a + 1:a + 3] = p["zeta_bend"]
    return m


# ---------------------------------------------------------------- mover drive
class Drive:
    """The movers' PD drive, exactly as scripts/mujoco_full_loco.rollout_errs
    does it (same pose definition from the tracked end nodes, same unwrapping,
    same linear sub-step interpolation). Copied, not imported, because that
    function keeps it in a closure."""

    def __init__(self, m):
        import mujoco
        self.m = m
        self.act = {n: [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{n}_{a}a") for a in ("x", "y", "t")]
                    for n in NAMES}
        self.base = {n: m.body_pos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)][:2].copy() for n in NAMES}
        self.yaw0 = {}
        for n in NAMES:
            q = m.body_quat[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"{n}_mod")]
            self.yaw0[n] = 2.0 * np.arctan2(q[3], q[0])
        self.jnt = {n: [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"{n}_{a}")]
                        for a in ("x", "y", "turn")] for n in NAMES}

    @staticmethod
    def poses(frame):
        """mover poses from one frame of the 4 tracked cables, frame = list of (16, 2)"""
        from mujoco_scene import HUB_CAVS, PLUG_CAV, _rotz
        out = np.mean([P[1] - P[0] for P in frame], axis=0)
        hy = np.arctan2(out[1], out[0]) + np.pi / 2
        hoff = _rotz(np.array(HUB_CAVS).mean(0), hy)
        res = {"hub": (np.array([frame[0][0][0] - hoff[0], frame[0][0][1] - hoff[1]]), hy)}
        for k, P in enumerate(frame):
            v = P[-1] - P[-2]
            y = np.arctan2(v[1], v[0]) + np.pi / 2
            o = _rotz(np.array(PLUG_CAV), y)
            res[f"plug{k}"] = (np.array([P[-1][0] - o[0], P[-1][1] - o[1]]), y)
        return res

    @classmethod
    def pose_seq(cls, frames):
        """unwrapped pose sequence; frames = list over t of [4 x (16, 2)]"""
        seq, prev = [], {n: cls.poses(frames[0])[n][1] for n in NAMES}
        for fr in frames:
            p = cls.poses(fr)
            for n in NAMES:
                prev[n] += np.arctan2(np.sin(p[n][1] - prev[n]), np.cos(p[n][1] - prev[n]))
            seq.append({n: (p[n][0], prev[n]) for n in NAMES})
        return seq

    def apply(self, d, pose):
        for n in NAMES:
            xy, yaw = pose[n]
            ax, ay, at = self.act[n]
            d.ctrl[ax], d.ctrl[ay] = xy[0] - self.base[n][0], xy[1] - self.base[n][1]
            d.ctrl[at] = yaw - self.yaw0[n]

    def set_qpos(self, d, pose):
        """teleport the movers to `pose` (used by the one-step reset)"""
        for n in NAMES:
            xy, yaw = pose[n]
            jx, jy, jt = self.jnt[n]
            d.qpos[jx], d.qpos[jy] = xy[0] - self.base[n][0], xy[1] - self.base[n][1]
            d.qpos[jt] = yaw - self.yaw0[n]

    def step_frame(self, d, a0, a1, sub):
        import mujoco
        for s_ in range(sub):
            f = (s_ + 1) / sub
            self.apply(d, {n: (a0[n][0] + f * (a1[n][0] - a0[n][0]), a0[n][1] + f * (a1[n][1] - a0[n][1]))
                           for n in NAMES})
            mujoco.mj_step(self.m, d)


def unstable(d):
    """MuJoCo resets mjData by itself when the integration blows up (warning
    BADQACC) and carries on from qpos0 -- finite numbers, wrong state. Treat
    any such event as a failed simulation."""
    import mujoco
    return d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number > 0


def vertices(m, d):
    from mujoco_replay import cable_vertices
    return np.stack([cable_vertices(m, d, k) for k in range(4)])          # (4, 16, 3)


# ---------------------------------------------------------- one-step reset
def _quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_from_to(u, v):
    """minimal rotation taking unit vector u onto unit vector v"""
    c = float(np.dot(u, v))
    ax = np.cross(u, v)
    if c < -0.999999:                                   # antiparallel: any orthogonal axis
        ax = np.cross(u, [0, 0, 1.0]) if abs(u[2]) < 0.9 else np.cross(u, [1.0, 0, 0])
        ax /= np.linalg.norm(ax)
        return np.array([0.0, *ax])
    q = np.array([1.0 + c, *ax])
    return q / np.linalg.norm(q)


def _resample_fixed(P3, seg):
    """vertices with the model's fixed segment lengths `seg`, walking along the
    measured 3-D polyline P3 from its first node (extrapolating past its end)"""
    d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(P3, axis=0), axis=1))]
    s = np.r_[0.0, np.cumsum(seg)]
    out = np.empty((len(s), 3))
    for i, si in enumerate(s):
        j = int(np.clip(np.searchsorted(d, si) - 1, 0, len(d) - 2))
        f = (si - d[j]) / max(d[j + 1] - d[j], 1e-12)
        out[i] = P3[j] + f * (P3[j + 1] - P3[j])
    return out


class OneStep:
    """Paper Eq. (1) on the MuJoCo scene: put the simulated cables into the
    OBSERVED state x_i, apply the observed action u_i (the movers' motion from
    frame i to i+1) for one frame, compare with x_{i+1}.

    The model is compiled once per clip on frame 0 (so the rest shape is the
    clip's frame 0, as in the rollouts). Setting the state x_i: each cable is
    laid along the tracked frame-i curve with the model's own (fixed) segment
    lengths -- the cable is inextensible, the tracking is not -- and every
    segment frame is the reference frame turned by the minimal rotation onto
    its new direction (no twist: the tracking is planar). Movers are teleported
    to their frame-i pose; velocities are zero (4 fps tracking has none worth
    trusting). A short settle with the movers held at frame i absorbs the
    millimetre gaps between chain tip and connector, then one frame is simulated.
    """

    def __init__(self, shp, fps, x, settle=200):
        import mujoco
        from mujoco_scene import lift_profile
        self.shp, self.fps, self.settle = shp, fps, settle
        self.m = make_model(shp, x)
        self.d = mujoco.MjData(self.m)
        self.drive = Drive(self.m)
        self.sub = max(1, int(round((1.0 / fps) / self.m.opt.timestep)))
        self.T = len(shp[0])
        self.seq = Drive.pose_seq([[shp[k][t] for k in range(4)] for t in range(self.T)])
        self.lift = lift_profile
        m, d = self.m, self.d
        mujoco.mj_kinematics(m, d)
        bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
        self.cab = []
        for k in range(4):
            names = [f"Wire{k}B_first"] + [f"Wire{k}B_{i}" for i in range(1, 14)] + [f"Wire{k}B_last"]
            ids = np.array([bid(n) for n in names])
            g = [i for i in range(m.ngeom) if m.geom_bodyid[i] == ids[-1]][0]
            seg = np.r_[[m.body_pos[i][0] for i in ids[1:]], 2 * m.geom_size[g][1]]
            R0 = d.xquat[ids].copy()
            x0 = np.stack([_quat_rotate(q, [1.0, 0, 0]) for q in R0])
            qadr = [m.jnt_qposadr[m.body_jntadr[i]] for i in ids]
            rel = [(m.jnt_qposadr[m.body_jntadr[bid(f"hub_relief_wire{k}")]]),
                   (m.jnt_qposadr[m.body_jntadr[bid(f"plug{k}_relief_wire")]])]
            self.cab.append(dict(ids=ids, seg=seg, R0=R0, x0=x0, qadr=qadr, rel=rel))

    def reset(self, i):
        import mujoco
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        for k, c in enumerate(self.cab):
            V = _resample_fixed(self.lift(self.shp[k][i].astype(float), RADIUS), c["seg"])
            dirs = np.diff(V, axis=0)
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            R = [_quat_mul(_quat_from_to(c["x0"][j], dirs[j]), c["R0"][j]) for j in range(15)]
            a = c["qadr"][0]
            d.qpos[a:a + 3] = V[0]; d.qpos[a + 3:a + 7] = R[0]
            for j in range(1, 15):
                q = _quat_mul(_quat_conj(_quat_mul(R[j - 1], m.body_quat[c["ids"][j]])), R[j])
                a = c["qadr"][j]
                d.qpos[a:a + 4] = q / np.linalg.norm(q)
            for a, v in zip(c["rel"], (V[1], V[-2])):
                d.qpos[a:a + 3] = v; d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.drive.set_qpos(d, self.seq[i])
        d.qvel[:] = 0
        self.drive.apply(d, self.seq[i])
        for _ in range(self.settle):
            mujoco.mj_step(m, d)

    def error(self, i):
        """squared and plain node distance (m^2, m), mean over 4 cables x 16 nodes,
        after one simulated frame i -> i+1 from the observed state i"""
        self.reset(i)
        self.drive.step_frame(self.d, self.seq[i], self.seq[i + 1], self.sub)
        V = vertices(self.m, self.d)[:, :, :2]
        tgt = np.stack([self.shp[k][i + 1] for k in range(4)])
        dist = np.linalg.norm(V - tgt, axis=2)
        if not np.isfinite(dist).all() or unstable(self.d):
            return np.inf, np.inf
        return float((dist ** 2).mean()), float(dist.mean())

    def nochange(self, i):
        tgt = np.stack([self.shp[k][i + 1] for k in range(4)]); cur = np.stack([self.shp[k][i] for k in range(4)])
        dist = np.linalg.norm(cur - tgt, axis=2)
        return float((dist ** 2).mean()), float(dist.mean())

    def reset_error(self, i):
        """how far the reset state itself is from the observation (after the settle)"""
        self.reset(i)
        V = vertices(self.m, self.d)[:, :, :2]
        return float(np.linalg.norm(V - np.stack([self.shp[k][i] for k in range(4)]), axis=2).mean())


def _quat_rotate(q, v):
    qv = np.array([0.0, *v])
    return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


# --------------------------------------------------------------- full rollout
def rollout(shp, fps, x, pose_seq=None, settle=300):
    """open-loop rollout of the scene (rest = frame 0 of shp) along a mover pose
    sequence (default: the measured one). Returns the simulated vertices
    (T, 4, 16, 3) at every frame, frame 0 after the 300-step settle -- the
    protocol of mujoco_full_loco.rollout_errs."""
    import mujoco
    m = make_model(shp, x)
    d = mujoco.MjData(m)
    drv = Drive(m)
    sub = max(1, int(round((1.0 / fps) / m.opt.timestep)))
    seq = pose_seq if pose_seq is not None else Drive.pose_seq([[shp[k][t] for k in range(4)] for t in range(len(shp[0]))])
    drv.apply(d, seq[0])
    for _ in range(settle):
        mujoco.mj_step(m, d)
    out = [vertices(m, d)]
    for t in range(len(seq) - 1):
        drv.step_frame(d, seq[t], seq[t + 1], sub)
        V = vertices(m, d)
        if not np.isfinite(V).all() or unstable(d):
            break
        out.append(V)
    return np.stack(out)

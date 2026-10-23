"""Full multi-mover, multi-cable Genesis scene, rendered to look like
MuJoCo's own calibrated render (mujoco_video.py) rather than the bare
single-cable test in genesis_scene.py.

Colour turned out to be the hard part: Genesis's MJCF importer does not
carry per-geom rgba/material into its render surfaces (confirmed empirically
-- a minimal red-sphere scene with an inline rgba AND with a named <material>
both rendered flat grey), and `RigidGeom.surface`/`RigidVisGeom.surface` are
read-only once an entity exists, so per-geom colour cannot be patched in
after the fact either. The one mechanism that DOES work is `add_entity(...,
surface=gs.surfaces.Default(color=...))`, which colours an entire entity
uniformly.

So this builds each cable and each mover as its OWN entity (5 movers + 4
cables = 9 entities), each with its own colour, rather than one scene-wide
entity coloured per-geom. That also sidesteps a second problem: MJCF
<equality><connect> constraints only resolve within the ONE model a single
MJCF file compiles to, so a cable and its mover could not be equality-linked
across separate entities anyway. Instead every cable end is driven directly
to the REAL tracked position, exactly where that end's mover cavity already
sits by construction (both come from the same tracked frame) -- the same
trick genesis_scene.py's single-cable test already validated, just applied
to all 4 cables and paired with 5 driven mover entities instead of 2 bare
anchor spheres.
"""
import sys
from pathlib import Path

import os
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mujoco_scene import (track_clip, _rotz, DIR, COLORS,
                          BOARD_W, BOARD_H, HUB_CAVS, PLUG_CAV, CAV_Z,
                          PLATFORM_HALF, PLATE_HALF, PLATE_Z, MODULE_HALF,
                          MODULE_Z, PLATFORM_Z, CAV_R)

ROOT = Path(__file__).resolve().parents[1]
N_NODES = 16
HUB_RED = (0.85, 0.15, 0.15, 1)
GREY = (0.55, 0.55, 0.55, 1)
BOARD_RGBA = (0.72, 0.72, 0.70, 1)


def _mover_nocol():
    """row 6 of the table was measured with the movers NOT colliding (their geoms carried
    contype/conaffinity 0 until 2026-08-25); WIRE_GENESIS_MOVER_CONTACT=0 restores that"""
    import os
    return ' contype="0" conaffinity="0"' if os.environ.get("WIRE_GENESIS_MOVER_CONTACT", "1") == "0" else ""


def platform_xml(name):
    # Collision ENABLED (2026-08-25, user requirement): these geoms carried
    # contype="0" conaffinity="0", i.e. the movers were purely visual and could
    # not push a cable at all -- the same gap Newton had before its full-scene
    # rework. The cable capsules and the board plane always collided; only the
    # movers were excluded.
    """The mover's coloured base plate only -- x/y slide, no module. Split
    out from the module so the two can be different Genesis entities (see
    topper_xml): mujoco_scene.mover_body() renders platform=coloured,
    plate+module=grey, which needs two colours and Genesis can only colour
    a whole entity uniformly."""
    return f'''<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <body name="{name}" pos="0 0 {PLATFORM_Z:.5f}">
      <joint name="{name}_x" type="slide" axis="1 0 0" damping="8"/>
      <joint name="{name}_y" type="slide" axis="0 1 0" damping="8"/>
      <geom type="box" size="{PLATFORM_HALF[0]} {PLATFORM_HALF[1]} {PLATFORM_HALF[2]}"
            mass="2"{_mover_nocol()}/>
    </body>
  </worldbody>
</mujoco>'''


def topper_xml(name, yaw):
    """Grey plate + module, the SAME x/y target as platform_xml's body plus
    its own turn hinge for the module -- a second entity driven in lock-step
    with the platform (own x/y slide, same PD target every frame) purely so
    it can carry the grey colour the platform doesn't. Collision disabled on
    every geom here: with platform_xml's body tracking the identical x/y
    target, the two footprints sit right on top of each other, and enabling
    contact between them made the two PD controllers fight the collision
    solver hard enough to blow up ('Invalid constraint forces... nan')."""
    return f'''<mujoco>
  <compiler angle="degree"/>
  <worldbody>
    <body name="{name}" pos="0 0 {PLATFORM_Z:.5f}">
      <joint name="{name}_x" type="slide" axis="1 0 0" damping="8"/>
      <joint name="{name}_y" type="slide" axis="0 1 0" damping="8"/>
      <geom type="box" pos="0 0 {PLATE_Z:.5f}"
            size="{PLATE_HALF[0]} {PLATE_HALF[1]} {PLATE_HALF[2]}"
            mass="0.5"{_mover_nocol()}/>
      <body name="{name}_mod" pos="0 0 {MODULE_Z:.5f}" euler="0 0 {np.degrees(yaw):.4f}">
        <joint name="{name}_turn" type="hinge" axis="0 0 1" damping="0.5"/>
        <geom type="box" size="{MODULE_HALF} {MODULE_HALF} {MODULE_HALF}"
              mass="0.3"{_mover_nocol()}/>
      </body>
    </body>
  </worldbody>
</mujoco>'''


def cable_xml(route_m, stiffness=0.1, damping=0.5, radius=0.003, mass_per_m=0.0368,
             rest_angle=None,
             anchor_stiffness=200.0, weld_anchor=True):
    """Capsule-hinge chain, free at BOTH ends (root + a separate anchor body
    connected to the last link via <connect>, same as genesis_scene.py) so
    both can be driven directly to the real tracked hub/plug position -- no
    cross-entity equality needed.

    `mass_per_m` replaces the old raw `density` parameter, converted the same
    way mujoco_scene.build_xml does (density = mass_per_m / (pi r^2)) -- so
    sweeps here are directly comparable to the MuJoCo mass_per_m/mu tuning in
    [[mass-inertia-beats-friction]]. A 2-round sweep on a 5-clip subset (see
    [[genesis-mass-friction-tuning]]) found the OPPOSITE trend from MuJoCo
    (lighter flat-to-better, not heavier) and picked mass_per_m=0.08 -- but
    validating that on the full 16-clip set made the OTHER 11 clips worse
    enough to net a small regression (58.7 vs 58.1mm Mittel), so the subset
    "win" was noise from the flat plateau, not a real improvement. Reverted
    to this original density=1300-equivalent default; mass/friction tuning
    does not meaningfully help this Genesis rope model either direction.
    """
    n = len(route_m)
    density = mass_per_m / (np.pi * radius ** 2)

    # stiffness/damping may be a scalar (uniform cable, as before) or a
    # sequence with one value per hinge joint, i.e. per body seg1..seg{n-1}.
    # Per-joint control is what lets a fit put a soft spot where the real cable
    # bends and a stiff one where it stays straight -- the same degree of
    # freedom Newton's per-joint weld fit uses.
    def _per_joint(v, name):
        if np.isscalar(v):
            return lambda i: f"{float(v):.6g}"
        a = np.asarray(v, float).ravel()
        if a.size != n - 1:
            raise ValueError(f"{name}: {a.size} Werte, erwartet {n - 1}")
        return lambda i: f"{a[i - 1]:.6g}"

    _st = _per_joint(stiffness, "stiffness")
    _dp = _per_joint(damping, "damping")

    def seg_geom(i):
        if i == n - 1:
            return f'<geom type="sphere" size="{radius:.5f}"/>'
        d = route_m[i + 1] - route_m[i]
        return (f'<geom type="capsule" fromto="0 0 0 {d[0]:.5f} {d[1]:.5f} {d[2]:.5f}" '
               f'size="{radius:.5f}" density="{density:.0f}"/>')

    body = (f'<body name="seg0" pos="{route_m[0][0]:.5f} {route_m[0][1]:.5f} {route_m[0][2]:.5f}">\n'
           f'  <joint type="free"/>\n  {seg_geom(0)}')
    tail = ""
    for i in range(1, n):
        d = route_m[i] - route_m[i - 1]
        body += (f'\n  <body name="seg{i}" pos="{d[0]:.5f} {d[1]:.5f} {d[2]:.5f}">\n'
                f'    <joint name="h{i}z" type="hinge" axis="0 0 1" '
                f'stiffness="{0.0 if rest_angle is not None else _st(i)}" '
                f'damping="{_dp(i)}"/>\n'
                f'    <joint name="h{i}y" type="hinge" axis="0 1 0" '
                f'stiffness="{0.0 if rest_angle is not None else _st(i)}" '
                f'damping="{_dp(i)}"/>\n'
                f'    {seg_geom(i)}')
        tail += "</body>"
    body += tail + "</body>"
    last = n - 1
    # `connect` transmits FORCE at a point only, so the anchor could never
    # orient the cable end -- the cable left the module holder at whatever
    # angle the physics produced (user requirement 2026-08-25: the cables must
    # always leave the holders perpendicular). `weld` constrains the relative
    # POSE, so driving the anchor's rotation now carries through to seg{last}.
    _eq = (f'<weld body1="anchor" body2="seg{last}" solref="0.004 1"/>'
           if weld_anchor else
           f'<connect body1="anchor" body2="seg{last}" anchor="0 0 0" solref="0.004 1"/>')

    return f'''<mujoco>
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <worldbody>
    {body}
    <body name="anchor" pos="{route_m[-1][0]:.5f} {route_m[-1][1]:.5f} {route_m[-1][2]:.5f}">
      <joint type="free"/>
      <geom type="sphere" size="{radius*0.5:.5f}" contype="0" conaffinity="0" mass="0.0001"/>
    </body>
  </worldbody>
  <equality>
    {_eq}
  </equality>
</mujoco>'''


def board_xml(mu=1.0):
    """`mu` mirrors mujoco_scene.build_xml's own board friction parameter --
    left unset before this, which meant Genesis's MJCF importer was silently
    using the MJCF spec default (also mu=1.0), so this is a no-op at the
    default and only starts to matter once a sweep moves it."""
    return f'''<mujoco>
  <worldbody>
    <geom type="plane" size="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 0.02" pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 0"
          friction="{mu} 0.05 0.002"/>
    <light pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 2.5" dir="0.2 0.3 -1" diffuse=".7 .7 .7"/>
    <light pos="{BOARD_W/2:.3f} {BOARD_H/2:.3f} 1.5" dir="0 0 -1" diffuse=".3 .3 .3"/>
  </worldbody>
</mujoco>'''


def mover_pose(P, is_hub):
    """(pos_xy, yaw) for a mover from the tracked route P (N,2), same
    formula as mujoco_scene.build_xml."""
    if is_hub:
        raise ValueError("hub pose needs all 4 routes, see build_scene()")
    v = P[-1] - P[-2]
    yaw = np.arctan2(v[1], v[0]) + np.pi / 2
    off = _rotz(np.array(PLUG_CAV), yaw)
    return np.array([P[-1][0] - off[0], P[-1][1] - off[1]]), yaw


def _sweep_stale_pid_dirs(root, keep_below=64):
    """Drop per-PID scratch dirs whose process is gone.

    The per-process MJCF directory fixes a real race (parallel sims used to
    overwrite each other's board.xml), but nothing removed the directories
    afterwards -- one run of the per-joint fit alone left 256 of them. Only
    dirs whose PID is no longer alive are touched, so a running sibling is
    never disturbed. Skipped entirely while the count is small.
    """
    try:
        dirs = [d for d in root.iterdir() if d.is_dir() and d.name.startswith("pid")]
    except OSError:
        return
    if len(dirs) < keep_below:
        return
    import shutil
    for d in dirs:
        try:
            pid = int(d.name[3:])
        except ValueError:
            continue
        if pid == os.getpid() or Path(f"/proc/{pid}").exists():
            continue
        shutil.rmtree(d, ignore_errors=True)


def cable_link_order(entity, n_nodes=16):
    """Row indices of seg0..seg{n-1} in `entity.get_links_pos()`, BY NAME.

    Genesis does not guarantee that the links come back in declaration order:
    DLO-Lab's genesis-world 1.0.0 returns ['seg0', 'anchor', 'seg1', ...],
    i.e. the anchor -- which sits at the FAR end of the cable -- is spliced in
    at row 1 and seg15 falls off the end of a [:16] slice. Reading positionally
    then mixes the far endpoint into node 1 (measured 518 mm off) and shifts
    every later node by one (~40 mm each), inflating the traced cable from
    631 mm to 1620 mm. Selecting by name is correct under any ordering.
    """
    idx = {l.name: i for i, l in enumerate(entity.links)}
    return [idx[f"seg{i}"] for i in range(n_nodes)]



def apply_rest_angles(ents, rest, stiffness=0.1, kv=0.5):
    """Per-joint rest angles for the Genesis hinge chain.

    Genesis's MJCF parser has no springref, so a hinge spring can only pull
    toward zero -- which is why this engine was the only one in the comparison
    without a rest shape at all. Route around it: cable_xml sets the hinge
    stiffness to 0 and a per-DOF PD controller supplies both the stiffness
    (kp) and the rest angle (its target). Mechanically a spring with a
    reference angle, reusing the controller that already drives the movers.

    Must be called AFTER scene.build() and from every entry point that builds
    a scene -- an earlier version lived in render_full_video() only, so
    eval_clip() silently ran with stiffness 0 and no controller, i.e. a limp
    cable whose rest angle did nothing. Three different rest amplitudes gave
    byte-identical errors, which is what gave it away.
    """
    import numpy as _np
    if rest is None:
        return 0
    rest = _np.asarray(rest, float)
    n_set = 0
    for k in range(4):
        e = ents.get(f"cable{k}")
        if e is None:
            continue
        idx, tgt, gains = [], [], []
        r = rest[k] if rest.ndim == 2 else rest
        for i in range(1, len(r) + 1):
            for ax in ("z", "y"):
                try:
                    j = e.get_joint(f"h{i}{ax}")
                except Exception:
                    continue
                idx += list(j.dofs_idx_local)
                tgt.append(float(r[i - 1]) if ax == "z" else 0.0)
                gains.append(float(stiffness[i - 1] if _np.ndim(stiffness)
                                   else stiffness))
        if idx:
            e.set_dofs_kp(_np.asarray(gains, _np.float32), idx)
            e.set_dofs_kv(_np.full(len(idx), kv, _np.float32), idx)
            e.control_dofs_position(_np.asarray(tgt, _np.float32), idx)
            n_set += len(idx)
    return n_set


def build_scene(shp, frame_idx=0, mu=1.0, **cable_kw):
    """Assemble the 9-entity scene (board, hub, 4 plugs, 4 cables) for one
    frame's initial pose. Returns (scene, entities-dict, xml-paths) -- the
    caller builds and drives it."""
    import genesis as gs

    routes2 = [shp[k][frame_idx].astype(float) for k in range(4)]
    z = CAV_Z
    routes = [np.hstack([P, np.full((len(P), 1), z)]) for P in routes2]

    hub_xy = routes2[0][0]
    out_dir = np.mean([P[1] - P[0] for P in routes2], axis=0)
    hub_yaw = np.arctan2(out_dir[1], out_dir[0]) + np.pi / 2
    hub_off = _rotz(np.array(HUB_CAVS).mean(0), hub_yaw)
    hub_pos = np.array([hub_xy[0] - hub_off[0], hub_xy[1] - hub_off[1]])

    if not gs._initialized:
        gs.init(backend=gs.cpu, logging_level="warning")
    scene = gs.Scene(show_viewer=False, sim_options=gs.options.SimOptions(dt=1e-3))

    # Per-process scratch: the MJCF files are written and then read back by
    # Genesis, so a shared directory tears when several simulations run in
    # parallel -- measured as "ParseError: no element found" in the 5-way
    # finite-difference SysID, where every worker overwrote the same board.xml.
    xml_root = ROOT / "data" / "genesis_full"
    xml_dir = xml_root / f"pid{os.getpid()}"
    xml_dir.mkdir(parents=True, exist_ok=True)
    _sweep_stale_pid_dirs(xml_root)

    def write(name, xml):
        p = xml_dir / f"{name}.xml"
        p.write_text(xml)
        return p

    ents = {}
    p = write("board", board_xml(mu=mu))
    ents["board"] = scene.add_entity(gs.morphs.MJCF(file=str(p)),
                                     surface=gs.surfaces.Default(color=BOARD_RGBA))

    def add_mover(name, pos, yaw, color):
        p = write(name, platform_xml(name))
        ents[name] = scene.add_entity(
            gs.morphs.MJCF(file=str(p), pos=(float(pos[0]), float(pos[1]), 0)),
            surface=gs.surfaces.Default(color=color))
        p = write(f"{name}_top", topper_xml(name, yaw))
        ents[f"{name}_top"] = scene.add_entity(
            gs.morphs.MJCF(file=str(p), pos=(float(pos[0]), float(pos[1]), 0)),
            surface=gs.surfaces.Default(color=GREY))

    add_mover("hub", hub_pos, hub_yaw, HUB_RED)

    plug_yaws, plug_poss = {}, {}
    for k, P in enumerate(routes2):
        pos, yaw = mover_pose(P, is_hub=False)
        plug_yaws[k] = yaw
        plug_poss[k] = pos
        add_mover(f"plug{k}", pos, yaw, (*COLORS[k], 1))

    for k in range(4):
        p = write(f"cable{k}", cable_xml(routes[k], **cable_kw))
        ents[f"cable{k}"] = scene.add_entity(
            gs.morphs.MJCF(file=str(p)),
            surface=gs.surfaces.Default(color=(*COLORS[k], 1)))

    return scene, ents, hub_pos, hub_yaw, plug_poss, plug_yaws


def render_full_video(clip, t1_sec=None, dt=1e-3, steps_per_frame=None,
                      kp_mover=4000.0, kv_mover=120.0, kp_turn=60.0, kv_turn=4.0,
                      kp_cable=80.0, kv_cable=4.0, fps_out=4.0, mu=1.0,
                      pin_cable_ends=False,
                      out_dir="data/genesis_full_video", **cable_kw):
    """Two-panel comparison video, mujoco_video.py's own layout: left is the
    real photo with tracked (dotted) vs simulated (dashed) overlay, right is
    THIS module's full coloured multi-entity Genesis render -- movers driven
    by their x/y/turn joints exactly like mujoco_replay.replay(), cable ends
    driven directly to the same real tracked positions the mover cavities
    sit at.

    `pin_cable_ends`: if True, the cable's root and anchor dofs are driven
    with `set_dofs_position` (a direct state overwrite, zero tracking error)
    instead of `control_dofs_position` (PD control, kp_cable/kv_cable, real
    lag possible) -- for a fair comparison against Newton's kinematic-pin
    protocol, which has zero endpoint error by construction. Default False
    keeps this function's original, already-validated PD behaviour."""
    import cv2
    import genesis as gs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import imageio.v2 as iio
    from fit_clip import read
    from wire_model.perception.rectify import find_board

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

    scene, ents, hub_pos0, hub_yaw0, plug_pos0, plug_yaw0 = build_scene(
        shp, frame_idx=0, mu=mu, **cable_kw)

    cap = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
    tf = find_board(read(cap, 0))
    cam_pos, cam_lookat, cam_up, cam_fov = genesis_camera(tf)
    cam = scene.add_camera(res=(560, 784), pos=tuple(cam_pos), lookat=tuple(cam_lookat),
                           up=tuple(cam_up), fov=cam_fov)
    scene.build()


    names = ["hub"] + [f"plug{k}" for k in range(4)]
    dofs, dofs_top = {}, {}
    for n in names:
        e = ents[n]
        # entity.control_dofs_position/set_dofs_kp expect ENTITY-LOCAL dof
        # indices (they add the entity's own dof_start again internally) --
        # joint.dofs_idx is already scene-global, so passing it straight in
        # double-offsets every entity but the first one added to the scene.
        # With enough total dofs that lands on a DIFFERENT entity's dofs
        # instead of raising (unsafe=True skips the bounds check), so this
        # silently drove the wrong body -- confirmed the real cause of the
        # plug movers not tracking (hub, added first with dof_start=0, was
        # unaffected and looked fine). dofs_idx_local is the fix.
        dofs[n] = e.get_joint(f"{n}_x").dofs_idx_local + e.get_joint(f"{n}_y").dofs_idx_local
        e.set_dofs_kp(np.full(2, kp_mover), dofs[n])
        e.set_dofs_kv(np.full(2, kv_mover), dofs[n])

        et = ents[f"{n}_top"]
        xy_top = et.get_joint(f"{n}_x").dofs_idx_local + et.get_joint(f"{n}_y").dofs_idx_local
        turn_top = et.get_joint(f"{n}_turn").dofs_idx_local
        dofs_top[n] = (xy_top, turn_top)
        et.set_dofs_kp(np.full(2, kp_mover), xy_top)
        et.set_dofs_kv(np.full(2, kv_mover), xy_top)
        et.set_dofs_kp(np.full(1, kp_turn), turn_top)
        et.set_dofs_kv(np.full(1, kv_turn), turn_top)
    cable_dofs = {}
    for k in range(4):
        e = ents[f"cable{k}"]
        e.set_dofs_kp(np.full(e.n_dofs, kp_cable))
        e.set_dofs_kv(np.full(e.n_dofs, kv_cable))
        # anchor body's free joint is the LAST 6 dofs of this entity
        # 0-2 / n-6..n-3 are the TRANSLATION dofs of seg0's and the anchor's
        # free joints; 3-5 / n-3..n are their ROTATION dofs (a rotation vector
        # in radians -- verified: dof 5 = 0.524 yields a 0.523 rad z-rotation).
        # Only the translations used to be driven, so the cable sat at the right
        # POINT but left the module holder at whatever angle physics produced.
        cable_dofs[k] = (np.arange(0, 3), np.arange(e.n_dofs - 6, e.n_dofs - 3),
                         np.arange(3, 6), np.arange(e.n_dofs - 3, e.n_dofs))

    # AFTER the blanket kp_cable loop above: that loop sets EVERY dof of
    # the cable to one gain, so a per-joint assignment made earlier is
    # silently wiped. An earlier version called this right after
    # scene.build() and kp 1 .. 1000 all returned the identical
    # 26.098 mm -- every value had been overwritten with kp_cable.
    apply_rest_angles(ents, cable_kw.get("rest_angle"),
                      cable_kw.get("stiffness", 0.1))

    # Reference directions at the build frame: the bodies are created with
    # identity orientation, so the target rotation at frame t is how far the
    # tracked end direction has turned since then.
    _ref_ang = {}
    for _k in range(4):
        _r0 = shp[_k][0]
        _ref_ang[_k] = (float(np.arctan2(*(_r0[1] - _r0[0])[::-1])),
                        float(np.arctan2(*(_r0[-1] - _r0[-2])[::-1])))

    def end_angles(k, t):
        """(hub, plug) target z-rotation of the two cable ends at frame t."""
        r = shp[k][t]
        a0 = float(np.arctan2(*(r[1] - r[0])[::-1]))
        a1 = float(np.arctan2(*(r[-1] - r[-2])[::-1]))
        _w = lambda a: float(np.arctan2(np.sin(a), np.cos(a)))
        return _w(a0 - _ref_ang[k][0]), _w(a1 - _ref_ang[k][1])

    def unwrap(prev, cur):
        return prev + np.arctan2(np.sin(cur - prev), np.cos(cur - prev))

    def poses(t):
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

    base = {n: (hub_pos0 if n == "hub" else plug_pos0[int(n[-1])]) for n in names}
    base_yaw = {n: (hub_yaw0 if n == "hub" else plug_yaw0[int(n[-1])]) for n in names}
    prev_yaw = dict(base_yaw)

    def apply(pose):
        for n in names:
            xy, yaw = pose[n]
            yaw_u = unwrap(prev_yaw[n], yaw)
            prev_yaw[n] = yaw_u
            rel_xy = xy - base[n]
            rel_yaw = np.array([yaw_u - base_yaw[n]])
            ents[n].control_dofs_position(rel_xy, dofs[n])
            ents[f"{n}_top"].control_dofs_position(rel_xy, dofs_top[n][0])
            ents[f"{n}_top"].control_dofs_position(rel_yaw, dofs_top[n][1])
        for k in range(4):
            z = np.full((16, 1), CAV_Z)
            route = np.hstack([shp[k][t_cur], z])
            e = ents[f"cable{k}"]
            if pin_cable_ends:
                e.set_dofs_position(route[0], cable_dofs[k][0])
                e.set_dofs_position(route[-1], cable_dofs[k][1])
                a0, a1 = end_angles(k, t)
                e.set_dofs_position(np.array([0.0, 0.0, a0]), cable_dofs[k][2])
                e.set_dofs_position(np.array([0.0, 0.0, a1]), cable_dofs[k][3])
                e.set_dofs_velocity(np.zeros(3), cable_dofs[k][0])
                e.set_dofs_velocity(np.zeros(3), cable_dofs[k][1])
            else:
                e.control_dofs_position(route[0], cable_dofs[k][0])
                e.control_dofs_position(route[-1], cable_dofs[k][1])
                # ORIENTATION as well, not just position: the module holder turns with
                # the tracked cable direction, and the cable end now turns with it, so
                # the cable leaves the holder perpendicular instead of at whatever angle
                # it happens to settle to.
                a0, a1 = end_angles(k, t)
                e.control_dofs_position(np.array([0.0, 0.0, a0]), cable_dofs[k][2])
                e.control_dofs_position(np.array([0.0, 0.0, a1]), cable_dofs[k][3])

    t_cur = 0
    apply(poses(0))
    for _ in range(300):
        scene.step()

    outdir = ROOT / out_dir
    outdir.mkdir(parents=True, exist_ok=True)
    frames_dir = outdir / clip
    frames_dir.mkdir(exist_ok=True)

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

    errs, frame_paths = [], []
    for t in range(n_frames):
        t_cur = t
        apply(poses(t))
        for _ in range(steps_per_frame):
            scene.step()
        cam.render(rgb=True)

        sim_shapes = []
        for k in range(4):
            e = ents[f"cable{k}"]
            # NOTE: no world-link offset here, unlike genesis_scene.py's
            # single-cable context -- confirmed empirically (index 0 matches
            # the tracked route[0] target almost exactly). The old [1:17]
            # slice (carried over from that other script) compared
            # [seg1..seg15, anchor] against [real0..real15], a spurious
            # off-by-one that inflated every error number computed here.
            pos = e.get_links_pos().cpu().numpy()[cable_link_order(e)]
            sim_shapes.append(pos[:, :2])
        e_all = [float(np.linalg.norm(sim_shapes[k] - shp[k][t], axis=1).mean()) * 1000
                for k in range(4)]
        errs.append(float(np.mean(e_all)))

        fr = read(cap, t0 + t)
        photo = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)[y0:y1, x0:x1]
        img = cam.render(rgb=True)[0]

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        ax[0].imshow(photo)
        for k in range(4):
            col = COLORS[k]
            P, Q = px(shp[k][t]), px(sim_shapes[k])
            ax[0].plot(P[:, 0], P[:, 1], ":", color=col, lw=3.2, alpha=0.95)
            ax[0].plot(Q[:, 0], Q[:, 1], "--", color=col, lw=2.4)
        ax[0].legend(handles=[
            Line2D([0], [0], color="0.25", ls=":", lw=3.2, label="real gemessen"),
            Line2D([0], [0], color="0.25", ls="--", lw=2.4, label="Genesis")],
            loc="lower left", fontsize=9, framealpha=0.85)
        ax[0].set_title(f"Video t = {(t0+t)/fps:.2f} s   Abweichung {errs[-1]:.0f} mm/Knoten",
                        fontsize=11)
        ax[1].imshow(img)
        ax[1].set_title("Genesis-Rendering", fontsize=11)
        for a in ax:
            a.set_xticks([]); a.set_yticks([])
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


def eval_clip(clip, t1_sec=None, dt=1e-3, steps_per_frame=None,
              kp_mover=4000.0, kv_mover=120.0, kp_turn=60.0, kv_turn=4.0,
              kp_cable=80.0, kv_cable=4.0, mu=1.0, pin_cable_ends=False,
              nodes_out=None, nodes3_out=None, shp_fps=None, **cable_kw):
    """Same physics/driving as render_full_video, minus the camera render and
    matplotlib compositing -- for parameter sweeps, where dozens of clip x
    parameter combinations need to run and only the error numbers matter.

    `pin_cable_ends`: see render_full_video -- True switches the cable's
    root/anchor dofs from PD control to a direct set_dofs_position overwrite
    (zero endpoint error), for a fair comparison against Newton's kinematic
    pinning protocol."""
    import cv2
    from mujoco_scene import track_clip as _track_clip  # local import mirrors render_full_video

    t1 = None
    if t1_sec is not None:
        cap_probe = cv2.VideoCapture(f"{DIR}/{clip}.mp4")
        fps_probe = (cap_probe.get(cv2.CAP_PROP_FPS) if cap_probe.isOpened() else 0) or 4.0
        cap_probe.release()
        t1 = int(round(t1_sec * fps_probe))
    if shp_fps is not None:
        shp, fps = shp_fps
    else:
        shp, t0, t1, fps = track_clip(clip, t0=8, t1=t1)
    n_frames = len(shp[0])
    if steps_per_frame is None:
        steps_per_frame = max(1, round((1.0 / fps) / dt))

    scene, ents, hub_pos0, hub_yaw0, plug_pos0, plug_yaw0 = build_scene(
        shp, frame_idx=0, mu=mu, **cable_kw)
    scene.build()

    names = ["hub"] + [f"plug{k}" for k in range(4)]
    dofs, dofs_top = {}, {}
    for n in names:
        e = ents[n]
        dofs[n] = e.get_joint(f"{n}_x").dofs_idx_local + e.get_joint(f"{n}_y").dofs_idx_local
        e.set_dofs_kp(np.full(2, kp_mover), dofs[n])
        e.set_dofs_kv(np.full(2, kv_mover), dofs[n])

        et = ents[f"{n}_top"]
        xy_top = et.get_joint(f"{n}_x").dofs_idx_local + et.get_joint(f"{n}_y").dofs_idx_local
        turn_top = et.get_joint(f"{n}_turn").dofs_idx_local
        dofs_top[n] = (xy_top, turn_top)
        et.set_dofs_kp(np.full(2, kp_mover), xy_top)
        et.set_dofs_kv(np.full(2, kv_mover), xy_top)
        et.set_dofs_kp(np.full(1, kp_turn), turn_top)
        et.set_dofs_kv(np.full(1, kv_turn), turn_top)
    cable_dofs = {}
    for k in range(4):
        e = ents[f"cable{k}"]
        e.set_dofs_kp(np.full(e.n_dofs, kp_cable))
        e.set_dofs_kv(np.full(e.n_dofs, kv_cable))
        # 0-2 / n-6..n-3 are the TRANSLATION dofs of seg0's and the anchor's
        # free joints; 3-5 / n-3..n are their ROTATION dofs (a rotation vector
        # in radians -- verified: dof 5 = 0.524 yields a 0.523 rad z-rotation).
        # Only the translations used to be driven, so the cable sat at the right
        # POINT but left the module holder at whatever angle physics produced.
        cable_dofs[k] = (np.arange(0, 3), np.arange(e.n_dofs - 6, e.n_dofs - 3),
                         np.arange(3, 6), np.arange(e.n_dofs - 3, e.n_dofs))

    # AFTER the blanket kp_cable loop above: that loop sets EVERY dof of
    # the cable to one gain, so a per-joint assignment made earlier is
    # silently wiped. An earlier version called this right after
    # scene.build() and kp 1 .. 1000 all returned the identical
    # 26.098 mm -- every value had been overwritten with kp_cable.
    apply_rest_angles(ents, cable_kw.get("rest_angle"),
                      cable_kw.get("stiffness", 0.1))

    # Reference directions at the build frame: the bodies are created with
    # identity orientation, so the target rotation at frame t is how far the
    # tracked end direction has turned since then.
    _ref_ang = {}
    for _k in range(4):
        _r0 = shp[_k][0]
        _ref_ang[_k] = (float(np.arctan2(*(_r0[1] - _r0[0])[::-1])),
                        float(np.arctan2(*(_r0[-1] - _r0[-2])[::-1])))

    def end_angles(k, t):
        """(hub, plug) target z-rotation of the two cable ends at frame t."""
        r = shp[k][t]
        a0 = float(np.arctan2(*(r[1] - r[0])[::-1]))
        a1 = float(np.arctan2(*(r[-1] - r[-2])[::-1]))
        _w = lambda a: float(np.arctan2(np.sin(a), np.cos(a)))
        return _w(a0 - _ref_ang[k][0]), _w(a1 - _ref_ang[k][1])

    def unwrap(prev, cur):
        return prev + np.arctan2(np.sin(cur - prev), np.cos(cur - prev))

    def poses(t):
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

    base = {n: (hub_pos0 if n == "hub" else plug_pos0[int(n[-1])]) for n in names}
    base_yaw = {n: (hub_yaw0 if n == "hub" else plug_yaw0[int(n[-1])]) for n in names}
    prev_yaw = dict(base_yaw)

    def apply_movers(pose):
        for n in names:
            xy, yaw = pose[n]
            yaw_u = unwrap(prev_yaw[n], yaw)
            prev_yaw[n] = yaw_u
            rel_xy = xy - base[n]
            rel_yaw = np.array([yaw_u - base_yaw[n]])
            ents[n].control_dofs_position(rel_xy, dofs[n])
            ents[f"{n}_top"].control_dofs_position(rel_xy, dofs_top[n][0])
            ents[f"{n}_top"].control_dofs_position(rel_yaw, dofs_top[n][1])

    def pin_cable(t):
        # set_dofs_position is a ONE-SHOT state overwrite in Genesis, not an
        # ongoing controller like control_dofs_position -- the anchor body
        # has non-zero (if tiny) mass, so it is still dynamically simulated
        # and free-falls under gravity for the rest of the frame if this
        # isn't reasserted every substep (confirmed empirically: called only
        # once per frame, the anchor settled to the RESTING height, not the
        # target, since 249 of every 250 steps had nothing holding it there).
        for k in range(4):
            z = np.full((16, 1), CAV_Z)
            route = np.hstack([shp[k][t], z])
            e = ents[f"cable{k}"]
            e.set_dofs_position(route[0], cable_dofs[k][0])
            e.set_dofs_position(route[-1], cable_dofs[k][1])
            e.set_dofs_velocity(np.zeros(3), cable_dofs[k][0])
            e.set_dofs_velocity(np.zeros(3), cable_dofs[k][1])

    def pin_cable_pd(t):
        for k in range(4):
            z = np.full((16, 1), CAV_Z)
            route = np.hstack([shp[k][t], z])
            e = ents[f"cable{k}"]
            e.control_dofs_position(route[0], cable_dofs[k][0])
            e.control_dofs_position(route[-1], cable_dofs[k][1])
            # ORIENTATION as well, not just position: the module holder turns with
            # the tracked cable direction, and the cable end now turns with it, so
            # the cable leaves the holder perpendicular instead of at whatever angle
            # it happens to settle to.
            a0, a1 = end_angles(k, t)
            e.control_dofs_position(np.array([0.0, 0.0, a0]), cable_dofs[k][2])
            e.control_dofs_position(np.array([0.0, 0.0, a1]), cable_dofs[k][3])

    def step(t):
        if pin_cable_ends:
            pin_cable(t)
        scene.step()

    apply_movers(poses(0))
    if not pin_cable_ends:
        pin_cable_pd(0)
    else:
        pin_cable(0)
    for _ in range(300):
        step(0)

    errs = []
    for t in range(n_frames):
        apply_movers(poses(t))
        if not pin_cable_ends:
            pin_cable_pd(t)
        for _ in range(steps_per_frame):
            step(t)
        sim_shapes, sim3 = [], []
        for k in range(4):
            e = ents[f"cable{k}"]
            # NOTE: no world-link offset here, unlike genesis_scene.py's
            # single-cable context -- confirmed empirically (index 0 matches
            # the tracked route[0] target almost exactly). The old [1:17]
            # slice (carried over from that other script) compared
            # [seg1..seg15, anchor] against [real0..real15], a spurious
            # off-by-one that inflated every error number computed here.
            pos = e.get_links_pos().cpu().numpy()[cable_link_order(e)]
            sim_shapes.append(pos[:, :2])
            if nodes3_out is not None:
                sim3.append(pos[:, :3])
        e_all = [float(np.linalg.norm(sim_shapes[k] - shp[k][t], axis=1).mean()) * 1000
                for k in range(4)]
        errs.append(float(np.mean(e_all)))
        if nodes3_out is not None:
            # full 3D per node -- the 2D nodes_out cannot show whether a cable
            # sinks INTO a mover, which is what the collision check needs
            nodes3_out.append(np.stack(sim3))
        if nodes_out is not None:
            # (4, 16, 2) in metres -- the DLO-Lab-style soft-mask SysID needs
            # the simulated polyline itself, not just its distance to the
            # tracked one, so it can splat it and compare against the real mask.
            nodes_out.append(np.stack(sim_shapes))

    return errs


def genesis_camera(tf, height_m=15.0, pad_mm=60.0):
    """Long-lens pinhole approximation of MuJoCo's calibrated orthographic
    camera (Genesis has no ortho mode): same lookat/height as
    mujoco_camera.board_camera, placed far above the board with a narrow FOV
    so perspective distortion stays small.

    Roll is NOT copied from board_camera's up_3d/right_3d: those encode the
    REAL camera's incidental few-degree mounting rotation for MuJoCo's
    explicit xyaxes convention. Genesis only takes a single `up` vector and
    derives `right` itself (right = forward x up), a different convention
    that does not reproduce the same roll from the same up_3d -- confirmed
    empirically, up_3d visibly over-rotated the board while world-axis-
    aligned up=(0,1,0) rendered it flush square, which is what we want here:
    this is a fresh render of the board's own axis-aligned geometry, not a
    re-creation of the real camera's mount angle."""
    from mujoco_camera import board_camera, BOARD_H_MM
    pos, right_3d, up_3d = board_camera(tf, height_m=height_m)
    fovy_m = BOARD_H_MM / 1000.0 + 2 * pad_mm / 1000.0
    fov_deg = 2 * np.degrees(np.arctan(fovy_m / (2 * height_m)))
    lookat = pos.copy(); lookat[2] = 0.0
    up = np.array([0.0, 1.0, 0.0])
    return pos, lookat, up, fov_deg

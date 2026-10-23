"""A MuJoCo camera calibrated exactly to the real camera's own geometry.

wire_model/perception/rectify.py measured the real camera and documented it
precisely: "Perspective is negligible... vertical vanishing point ~72
frame-heights away, i.e. effectively parallel" -- the real footage is, to
within the measured residual (1-2 px rms), an orthographic top-down view of
the board, rotated ~2.7 degrees in-plane. `BoardTransform` stores exactly that
(a similarity: rotate, scale, translate) -- no lens/perspective model, because
none is needed.

That means an ORTHOGRAPHIC MuJoCo camera, aimed straight down at the board
centre and using this SAME rotation and scale, is not an approximation of the
real camera's geometry -- it reproduces it, to the same precision the real
camera itself was characterised to. This replaces the earlier oblique-angle
render, which was a deliberate artistic choice (elevation -62..-72 degrees, to
show the cables lifting off the board) rather than a match to the footage.
"""
import numpy as np

BOARD_W_MM = 1200.0
BOARD_H_MM = 1680.0


def board_camera(tf, height_m=2.0, lookat_mm=None):
    """Camera pos/xyaxes/fovy for a fixed MJCF <camera>, aimed straight down
    at the board centre, oriented and scaled to match `tf` (a BoardTransform)
    exactly: 1 rendered pixel == 1 real-video pixel at the board plane.

    Returns (pos_xyz, right_3d, up_3d, fovy_m_per_full_height_placeholder) --
    actually returns (pos, right, up); fovy is computed by the caller once the
    render height in pixels is chosen (fovy_m = render_h_px * tf.mm_per_px /
    1000), since it depends on the output resolution, not on the camera pose.

    Derivation: `to_px(mm) = origin + (mm/mm_per_px) @ [e_x | e_y]`, i.e. the
    2x2 matrix M = [e_x | e_y] (columns) maps a millimetre offset to a pixel
    offset. The world direction that appears as "one pixel rightward" in the
    rendered image is therefore M^{-1} @ (1, 0); "one pixel upward" is
    M^{-1} @ (0, -1) (image rows grow downward, so "up" is -row). M is a
    reflection (det = -1, since board-mm-y-up meets image-row-down), hence
    its own inverse -- computed here numerically rather than assumed, and
    verified against a real tracked point before being trusted anywhere else
    in the pipeline.
    """
    M = np.array([tf.e_x, tf.e_y]).T
    Minv = np.linalg.inv(M)
    right = Minv @ np.array([1.0, 0.0])
    up = Minv @ np.array([0.0, -1.0])
    right_3d = np.array([right[0], right[1], 0.0])
    up_3d = np.array([up[0], up[1], 0.0])
    if lookat_mm is None:
        lookat_mm = (BOARD_W_MM / 2.0, BOARD_H_MM / 2.0)
    lookat = np.array([lookat_mm[0] / 1000.0, lookat_mm[1] / 1000.0, 0.0])
    pos = lookat + np.array([0.0, 0.0, height_m])
    return pos, right_3d, up_3d


def camera_xml(tf, render_h_px, name="video_match", lookat_mm=None):
    """<camera> element + the fovy (metres) the caller must render at.

    `lookat_mm` centres the view on an arbitrary board-mm point instead of the
    board's own centre -- pass the same crop centre the photo panel uses (via
    `crop_camera` below) to get a render that shows literally the same real-
    world region as that photo, not merely the same orientation and scale.
    """
    pos, right, up = board_camera(tf, lookat_mm=lookat_mm)
    fovy_m = render_h_px * tf.mm_per_px / 1000.0
    xml = (f'<camera name="{name}" mode="fixed" projection="orthographic" '
          f'fovy="{fovy_m:.6f}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
          f'xyaxes="{right[0]:.6f} {right[1]:.6f} {right[2]:.6f} '
          f'{up[0]:.6f} {up[1]:.6f} {up[2]:.6f}"/>')
    return xml, fovy_m


def crop_camera(tf, frame_shape, corners_mm=((70, 70), (1130, 70), (1130, 1680), (70, 1680)),
                name="video_match", scale=1.0):
    """Camera + resolution that render EXACTLY the same crop the photo panels
    use elsewhere in this project (mujoco_video.py, mujoco_frames.py,
    mujoco_figure.py all compute this identical x0:x1, y0:y1 window from the
    same corners). Returns (camera_xml, render_h_px, render_w_px, off), where
    `off` is the crop origin in raw-frame pixels -- the same value those
    scripts already subtract from `tf.to_px(...)` for the photo overlay, so
    passing it through keeps both panels pixel-registered to each other, not
    just individually correct.
    """
    H, W = frame_shape[:2]
    pxc = tf.to_px(np.array(corners_mm, dtype=float))
    x0, y0 = max(0, int(pxc[:, 0].min())), max(0, int(pxc[:, 1].min()))
    x1, y1 = min(W, int(pxc[:, 0].max())), min(H, int(pxc[:, 1].max()))
    off = np.array([x0, y0])
    rw, rh = x1 - x0, y1 - y0
    centre_px = np.array([(x0 + x1) / 2.0, (y0 + y1) / 2.0])
    centre_mm = tf.to_mm(centre_px)
    pos, right, up = board_camera(tf, lookat_mm=tuple(centre_mm))
    # fovy is fixed from the NATIVE crop height, so the real-world EXTENT
    # shown never depends on `scale` -- only how many pixels are spent
    # rendering it. Computing fovy from the scaled-down resolution instead
    # (an earlier version of this function did exactly that) shrinks the
    # field of view along with the pixel count, silently cropping the render
    # tighter than the photo: 55% resolution rendered 55% of the board's
    # height, not the full board at 55% of the pixels.
    fovy_m = rh * tf.mm_per_px / 1000.0
    rh_out, rw_out = max(1, round(rh * scale)), max(1, round(rw * scale))
    xml = (f'<camera name="{name}" mode="fixed" projection="orthographic" '
          f'fovy="{fovy_m:.6f}" pos="{pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}" '
          f'xyaxes="{right[0]:.6f} {right[1]:.6f} {right[2]:.6f} '
          f'{up[0]:.6f} {up[1]:.6f} {up[2]:.6f}"/>')
    return xml, rh_out, rw_out, off

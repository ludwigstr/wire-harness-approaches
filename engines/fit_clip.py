"""Video frame access for the engine scripts (the project's fit_clip module minus
everything but read(); only needed when tracking from video, see mujoco_scene.track_clip)."""
import cv2


def read(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, fr = cap.read()
    return fr if ok else None

"""The evaluation clip set -- deliberately free of any simulator import.

This used to live in newton_fit_stiffness.py, which pulls in warp/newton. That
made Genesis-only code fail with `ModuleNotFoundError: No module named 'warp'`
whenever it ran in an environment that has Genesis but not Newton -- exactly
the case for the genesis-world 1.3.3 venv (see the genesis-133-restored note).
Import the clip set from here; newton_fit_stiffness re-exports it unchanged so
existing callers keep working.
"""

# Clips whose TRACKING is demonstrably broken, so their routes cannot serve as
# a reference (user decision, 2026-08-25): 110528's yellow route drops to 8.7%
# mask coverage -- it runs across empty board -- and 172926's green route hops
# onto the neighbouring yellow strand, swinging the plug yaw by 161 deg.
EXCLUDED_TRACKING = {"20260602_110528", "20260602_172926"}

ALL_CLIP_TRIMS = {
    "20260602_102713": None, "20260602_110528": None, "20260602_134613": None,
    "20260605_112939": None, "20260602_183620": 20, "20260602_172435": 19,
    "20260602_170108": 9, "20260602_175847": 14, "20260602_172926": 20,
    "20260602_193329": 15, "20260602_141210": 13, "20260602_114019": 13,
    "20260602_162650": 12, "20260602_165113": 13, "20260602_193953": 12,
    "20260602_121006": 11,
}

# The evaluation set every script iterates over: 14 clips.
CLIP_TRIMS = {k: v for k, v in ALL_CLIP_TRIMS.items()
              if k not in EXCLUDED_TRACKING}

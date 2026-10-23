"""RopeDreamer (Missal et al., arXiv 2604.28161, 2026): a Dreamer-style recurrent
state-space model whose state is a kinematic chain (base position + one unit
quaternion per link), so a decoded state can never stretch the cable.
Table: world sheet row 22 -- 16.8 mm/node in-sample on the 14 clips (62.4 with
the X-Planar model and no fine-tuning).

Data: the tracked videos only (no simulator).
  1. pretrain  on the 103 X-Planar clips (10 % validation, paper loss, beta 1,
               free nats 3 -- without them the posterior collapses on our data)
  2. finetune  in-sample on the 14 clips: beta 0.1, FK position loss and an
               open-loop rollout loss (both 1000), open-loop checkpoint selection
  3. eval      paper protocol: 5-frame warm-up, open loop, both ends pinned

Deviations from the paper and why: docs in the main project (ROPEDREAMER.md) --
two driven ends instead of one grasp, quaternion loss up to sign, ELBO as a
sum over z-standardised dimensions, free nats, FK position + rollout loss.

    python approaches/08_ropedreamer.py eval                      # checkpoints/ropedreamer_14.pt -> 16.8
    python approaches/08_ropedreamer.py all                       # pretrain + finetune + eval from scratch
"""
import argparse

import common  # noqa: F401  (sets the import path)
from rssm_common import train, evaluate

XP, T14 = "data/tracks_xplanar", "data/tracks14"
PRE, FT = "checkpoints/ropedreamer_xplanar.pt", "checkpoints/ropedreamer_14.pt"


def pretrain(a):
    train("ropedreamer", [XP], a.pre_out, size="S", seq_len=32, beta=1.0, free_nats=3.0, lr=1e-4, val_frac=0.1, device=a.device)


def finetune(a):
    train("ropedreamer", [T14], a.ft_out, size="S", seq_len=16, epochs=400, patience=30, beta=0.1, free_nats=3.0, lr=1e-4,
          pos_weight=1000.0, rollout_weight=1000.0, val_frac=0.0, select="openloop", init=a.pre_out, device=a.device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["pretrain", "finetune", "eval", "all"])
    ap.add_argument("--pre-out", default=PRE); ap.add_argument("--ft-out", default=FT)
    ap.add_argument("--ckpt", default=None, help="eval: checkpoint (default: the fine-tuned one)")
    ap.add_argument("--tracks", default=T14); ap.add_argument("--out", default="results/ropedreamer_14.json")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.step in ("pretrain", "all"):
        pretrain(a)
    if a.step in ("finetune", "all"):
        finetune(a)
    if a.step in ("eval", "all"):
        evaluate(a.ckpt or a.ft_out, a.tracks, a.out, device=a.device, label="RopeDreamer")


if __name__ == "__main__":
    main()

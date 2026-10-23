"""DeformNet (Li et al. 2024): a PointNet encoder turns the observation into an
embedding, a Dreamer-v2 RSSM (32 x 32 categorical latent, KL balancing 0.8)
learns the dynamics IN EMBEDDING SPACE, a decoder maps back.
Table: world sheet row 21 -- 30.6 mm/node in-sample on the 14 clips (84.4 with
the X-Planar model and no fine-tuning).

Adapted to tracked cables: the observation is the set of 16 node positions
(read by the PointNet as an unordered set, as in the paper); the paper's
conditional NeRF decoder is replaced by a decoder to the 16 ordered positions;
actions are the measured displacements of the two driven ends.

Data: the tracked videos only (no simulator).
  1. pretrain  on the 103 X-Planar clips (10 % validation); stage 1 trains
               encoder/decoder, then they are frozen and the RSSM is trained;
               loss = paper loss + position loss + open-loop rollout loss (1000
               each) -- with the paper loss alone DeformNet scores ~200 mm here
  2. finetune  in-sample on the 14 clips, open-loop checkpoint selection
  3. eval      RopeDreamer-paper protocol: 5-frame warm-up, open loop, ends pinned

    python approaches/07_deformnet.py eval                       # checkpoints/deformnet_14.pt -> 30.6
    python approaches/07_deformnet.py all
"""
import argparse

import common  # noqa: F401
from rssm_common import train, evaluate

XP, T14 = "data/tracks_xplanar", "data/tracks14"
PRE, FT = "checkpoints/deformnet_xplanar.pt", "checkpoints/deformnet_14.pt"


def pretrain(a):
    train("deformnet", [XP], a.pre_out, seq_len=32, beta=1.0, free_nats=0.0, lr=1e-4, pos_weight=1000.0, rollout_weight=1000.0,
          val_frac=0.1, device=a.device)


def finetune(a):
    train("deformnet", [T14], a.ft_out, seq_len=16, epochs=400, patience=30, beta=1.0, free_nats=0.0, lr=1e-4,
          pos_weight=1000.0, rollout_weight=1000.0, val_frac=0.0, select="openloop", init=a.pre_out, device=a.device)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["pretrain", "finetune", "eval", "all"])
    ap.add_argument("--pre-out", default=PRE); ap.add_argument("--ft-out", default=FT)
    ap.add_argument("--ckpt", default=None); ap.add_argument("--tracks", default=T14)
    ap.add_argument("--out", default="results/deformnet_14.json"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.step in ("pretrain", "all"):
        pretrain(a)
    if a.step in ("finetune", "all"):
        finetune(a)
    if a.step in ("eval", "all"):
        evaluate(a.ckpt or a.ft_out, a.tracks, a.out, device=a.device, label="DeformNet")


if __name__ == "__main__":
    main()

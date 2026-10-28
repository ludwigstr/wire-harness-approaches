"""RopeDreamer (Missal, Domingues et al., arXiv 2604.28161, 2026): a Recurrent
State Space Model over a quaternionic kinematic chain, reimplemented for this
project's cables. See kinematics.py (state representation), model.py (RSSM
with dual decoders), data.py (sequences from the tracked clips)."""
from .kinematics import positions_to_chain, chain_to_positions, state_dim
from .model import RopeDreamer, SIZES

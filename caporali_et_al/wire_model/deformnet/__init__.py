"""DeformNet (Li, Ai et al., arXiv 2402.07648, 2024) for the harness cables:
PointNet encoder over the node set, Dreamer-v2 RSSM (categorical latents, KL
balancing) over the embedding, decoder back to node positions in place of the
conditional NeRF (we score positions, not pixels). See model.py."""
from .model import DeformNet

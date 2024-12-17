import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange
from quantum_transformers.quantum_layer import QuantumLayer
from typing import Optional, Callable

class SwinTransformer(nn.Module):
    hidden_dim: int
    layers: tuple
    heads: tuple
    channels: int
    num_classes: int
    head_dim: int
    window_size: int
    downscaling_factors: tuple
    relative_pos_embedding: bool
    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

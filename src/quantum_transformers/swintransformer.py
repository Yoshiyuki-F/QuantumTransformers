import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange
from src.quantum_transformers.quantum_layer import QuantumLayer
from typing import Optional, Callable

class CyclicShift(nn.Module):
    displacement: int

    @nn.compact
    def __call__(self, x):
        return jnp.roll(x, shift=(self.displacement, self.displacement), axis=(1, 2))

class Residual(nn.Module):
    fn: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    fn: nn.Module

    @nn.compact
    def __call__(self, x, **kwargs):
        x = nn.LayerNorm()(x)
        return self.fn(x, **kwargs)

from src.quantum_transformers.common_layers import FeedForward

def create_mask(window_size, displacement, upper_lower, left_right):
    mask = jnp.zeros((window_size ** 2, window_size ** 2), dtype=jnp.float32)

    if upper_lower:
        mask[-displacement * window_size:, :-displacement * window_size] = float('-inf')
        mask[:-displacement * window_size, -displacement * window_size:] = float('-inf')

    if left_right:
        mask = rearrange(mask, '(h1 w1) (h2 w2) -> h1 w1 h2 w2', h1=window_size, h2=window_size)
        mask[:, -displacement:, :, :-displacement] = float('-inf')
        mask[:, :-displacement, :, -displacement:] = float('-inf')
        mask = rearrange(mask, 'h1 w1 h2 w2 -> (h1 w1) (h2 w2)')

    return jnp.array(mask)

def get_relative_distances(window_size):
    indices = jnp.array([[x, y] for x in range(window_size) for y in range(window_size)])
    distances = indices[None, :, :] - indices[:, None, :]
    return distances

class WindowAttention(nn.Module):
    dim: int
    heads: int
    head_dim: int
    shifted: bool
    window_size: int
    relative_pos_embedding: bool
    quantum_w_shape: tuple = (1,)
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic=True):
        inner_dim = self.head_dim * self.heads
        scale = self.head_dim ** -0.5
        b, h, w, c = x.shape

        if self.quantum_circuit is None:
            q = nn.Dense(features=inner_dim, use_bias=False)(x)
            k = nn.Dense(features=inner_dim, use_bias=False)(x)
            v = nn.Dense(features=inner_dim, use_bias=False)(x)
        else:
            q = QuantumLayer(num_qubits=inner_dim, w_shape=self.quantum_w_shape,
                             circuit=self.quantum_circuit)(x)
            k = QuantumLayer(num_qubits=inner_dim, w_shape=self.quantum_w_shape,
                             circuit=self.quantum_circuit)(x)
            v = QuantumLayer(num_qubits=inner_dim, w_shape=self.quantum_w_shape,
                             circuit=self.quantum_circuit)(x)

        assert inner_dim % self.heads == 0, f'Feature dimension ({inner_dim}) must be divisible by number of heads ({self.heads})'

        # Reshape and compute attention
        q = rearrange(q, 'b h w (n d) -> b n h w d', n=self.heads)
        k = rearrange(k, 'b h w (n d) -> b n h w d', n=self.heads)
        v = rearrange(v, 'b h w (n d) -> b n h w d', n=self.heads)

        attn = (q @ jnp.swapaxes(k, -2, -1)) * scale
        attn = nn.softmax(attn, axis=-1)

        out = attn @ v
        out = rearrange(out, 'b n h w d -> b h w (n d)')

        if self.quantum_circuit is None:
            out = nn.Dense(features=self.dim)(out)
        else:
            out = QuantumLayer(num_qubits=self.dim, w_shape=self.quantum_w_shape,
                               circuit=self.quantum_circuit)(out)

        return out

class SwinBlock(nn.Module):
    dim: int
    heads: int
    head_dim: int
    mlp_dim: int
    shifted: bool
    window_size: int
    relative_pos_embedding: bool
    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = Residual(PreNorm(fn=WindowAttention(
            dim=self.dim,
            heads=self.heads,
            head_dim=self.head_dim,
            shifted=self.shifted,
            window_size=self.window_size,
            relative_pos_embedding=self.relative_pos_embedding,
            quantum_w_shape=self.quantum_w_shape,
            quantum_circuit=self.quantum_attn_circuit
        )))(x, deterministic=deterministic)

        x = Residual(PreNorm(fn=FeedForward(
            dim=self.dim,
            hidden_dim=self.mlp_dim,
            quantum_w_shape=self.quantum_w_shape,
            quantum_circuit=self.quantum_mlp_circuit
        )))(x)
        return x

class PatchMerging(nn.Module):
    in_channels: int
    out_channels: int
    downscaling_factor: int

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape

        # Ensure input dimensions are divisible by downscaling factor
        assert h % self.downscaling_factor == 0, f'Input height ({h}) must be divisible by downscaling_factor ({self.downscaling_factor})'
        assert w % self.downscaling_factor == 0, f'Input width ({w}) must be divisible by downscaling_factor ({self.downscaling_factor})'

        # Calculate new dimensions
        new_h = h // self.downscaling_factor
        new_w = w // self.downscaling_factor

        # Use unfold-like operation to implement patch merging
        # First reorganize in height direction
        x = x.reshape(b, new_h, self.downscaling_factor, w, c)
        # Then reorganize in width direction
        x = x.reshape(b, new_h, self.downscaling_factor, new_w, self.downscaling_factor, c)
        # Rearrange dimensions to combine all patch dimensions
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # Flatten all patch features
        x = x.reshape(b, new_h, new_w, self.downscaling_factor * self.downscaling_factor * c)

        # Project to required output channels
        x = nn.Dense(features=self.out_channels)(x)
        return x

class StageModule(nn.Module):
    in_channels: int
    hidden_dimension: int
    layers: int
    downscaling_factor: int
    num_heads: int
    head_dim: int
    window_size: int
    relative_pos_embedding: bool
    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic=True):
        assert self.layers % 2 == 0, 'Stage layers need to be divisible by 2 for regular and shifted block.'

        x = PatchMerging(
            in_channels=self.in_channels,
            out_channels=self.hidden_dimension,
            downscaling_factor=self.downscaling_factor
        )(x)

        layers = []
        for _ in range(self.layers // 2):
            layers.append([
                SwinBlock(
                    dim=self.hidden_dimension,
                    heads=self.num_heads,
                    head_dim=self.head_dim,
                    mlp_dim=self.hidden_dimension * 4,
                    shifted=False,
                    window_size=self.window_size,
                    relative_pos_embedding=self.relative_pos_embedding,
                    quantum_w_shape=self.quantum_w_shape,
                    quantum_attn_circuit=self.quantum_attn_circuit,
                    quantum_mlp_circuit=self.quantum_mlp_circuit
                ),
                SwinBlock(
                    dim=self.hidden_dimension,
                    heads=self.num_heads,
                    head_dim=self.head_dim,
                    mlp_dim=self.hidden_dimension * 4,
                    shifted=True,
                    window_size=self.window_size,
                    relative_pos_embedding=self.relative_pos_embedding,
                    quantum_w_shape=self.quantum_w_shape,
                    quantum_attn_circuit=self.quantum_attn_circuit,
                    quantum_mlp_circuit=self.quantum_mlp_circuit
                )
            ])

        for regular_block, shifted_block in layers:
            x = regular_block(x, deterministic=deterministic)
            x = shifted_block(x, deterministic=deterministic)
        return jnp.transpose(x, [0, 3, 1, 2])

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

    @nn.compact
    def __call__(self, img, train=False):
        # Check input dimensions
        b, h, w, c = img.shape
        assert c == self.channels, f'Input channels ({c}) doesn\'t match expected channels ({self.channels})'

        # Initial convolution layer - for 28x28 input, use (2, 2) kernel and stride
        x = nn.Conv(features=self.hidden_dim, kernel_size=(2, 2), strides=(2, 2))(img)  # 14x14

        # Stage modules
        num_stages = len(self.layers)
        for i in range(num_stages):
            # Calculate current feature map dimensions
            curr_h = h // (2 * (2 ** sum(1 for j in range(i) if self.downscaling_factors[j] > 1)))
            curr_w = w // (2 * (2 ** sum(1 for j in range(i) if self.downscaling_factors[j] > 1)))

            # If current size is too small, don't downsample further
            actual_downscaling = self.downscaling_factors[i] if curr_h > 7 and curr_w > 7 else 1

            x = StageModule(
                in_channels=self.hidden_dim * 2 ** max(i - 1, 0),
                hidden_dimension=self.hidden_dim * 2 ** i,
                layers=self.layers[i],
                downscaling_factor=actual_downscaling,  # Use adjusted downsampling factor
                num_heads=self.heads[i],
                head_dim=self.head_dim,
                window_size=self.window_size,
                relative_pos_embedding=self.relative_pos_embedding,
                quantum_w_shape=self.quantum_w_shape,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit
            )(x, deterministic=not train)

        x = jnp.mean(x, axis=[2, 3])
        x = nn.Dense(features=self.num_classes)(x)
        return x

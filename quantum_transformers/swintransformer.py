import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange
from quantum_transformers.quantum_layer import QuantumLayer
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


class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    quantum_w_shape: tuple = (1,)
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape

        if self.quantum_circuit is None:
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.gelu(x)
            x = nn.Dense(features=self.dim)(x)
        else:
            # 先使用Dense层
            x = nn.Dense(features=self.hidden_dim)(x)
            x = nn.gelu(x)

            # 应用量子层
            flat_batch_size = b * h * w
            quantum_dim = min(8, self.hidden_dim)  # 限制量子比特数量

            # 确保维度正确
            x = x.reshape(flat_batch_size, -1)[:, :quantum_dim]
            x = QuantumLayer(num_qubits=quantum_dim,
                             w_shape=self.quantum_w_shape,
                             circuit=self.quantum_circuit)(x)

            # 使用Dense层恢复维度
            x = nn.Dense(features=self.dim)(x)
            x = x.reshape(b, h, w, self.dim)

        return x


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
            # 先使用Dense层调整维度
            q = nn.Dense(features=inner_dim, use_bias=False)(x)
            k = nn.Dense(features=inner_dim, use_bias=False)(x)
            v = nn.Dense(features=inner_dim, use_bias=False)(x)

            # 然后应用量子层
            quantum_dim = min(8, inner_dim)  # 限制量子比特数量

            # 计算正确的batch_size
            flat_batch_size = b * h * w

            # 确保维度正确
            q = q.reshape(flat_batch_size, -1)[:, :quantum_dim]  # 只取前quantum_dim个特征
            k = k.reshape(flat_batch_size, -1)[:, :quantum_dim]
            v = v.reshape(flat_batch_size, -1)[:, :quantum_dim]

            # 应用量子层
            q = QuantumLayer(num_qubits=quantum_dim, w_shape=self.quantum_w_shape,
                             circuit=self.quantum_circuit)(q)
            k = QuantumLayer(num_qubits=quantum_dim, w_shape=self.quantum_w_shape,
                             circuit=self.quantum_circuit)(k)
            v = QuantumLayer(num_qubits=quantum_dim, w_shape=self.quantum_w_shape,
                             circuit=self.quantum_circuit)(v)

            # 使用Dense层恢复到所需维度
            q = nn.Dense(features=inner_dim)(q)
            k = nn.Dense(features=inner_dim)(k)
            v = nn.Dense(features=inner_dim)(v)

            # 恢复原始形状
            q = q.reshape(b, h, w, inner_dim)
            k = k.reshape(b, h, w, inner_dim)
            v = v.reshape(b, h, w, inner_dim)

        # 确保特征维度可以被头数整除
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
            # 先使用Dense层
            out = nn.Dense(features=self.dim)(out)

            # 然后应用量子层
            b, h, w, c = out.shape
            flat_batch_size = b * h * w
            quantum_dim = min(8, self.dim)

            # 确保维度正确
            out = out.reshape(flat_batch_size, -1)[:, :quantum_dim]
            out = QuantumLayer(num_qubits=quantum_dim, w_shape=self.quantum_w_shape,
                               circuit=self.quantum_circuit)(out)

            # 使用Dense层恢复到所需维度
            out = nn.Dense(features=self.dim)(out)
            out = out.reshape(b, h, w, self.dim)

        return out


# 继续转换其他类...
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

        # 确保输入维度可以被下采样因子整除
        assert h % self.downscaling_factor == 0, f'Input height ({h}) must be divisible by downscaling_factor ({self.downscaling_factor})'
        assert w % self.downscaling_factor == 0, f'Input width ({w}) must be divisible by downscaling_factor ({self.downscaling_factor})'

        # 计算新的维度
        new_h = h // self.downscaling_factor
        new_w = w // self.downscaling_factor

        # 使用unfold-like操作来实现patch合并
        # 首先在高度方向上重组
        x = x.reshape(b, new_h, self.downscaling_factor, w, c)
        # 然后在宽度方向上重组
        x = x.reshape(b, new_h, self.downscaling_factor, new_w, self.downscaling_factor, c)
        # 重新排列维度，将所有patch维度组合在一起
        x = x.transpose(0, 1, 3, 2, 4, 5)
        # 将所有patch特征展平
        x = x.reshape(b, new_h, new_w, self.downscaling_factor * self.downscaling_factor * c)

        # 投影到所需的输出通道数
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
    def __call__(self, img, deterministic=True):
        # 检查输入维度
        b, h, w, c = img.shape
        assert c == self.channels, f'Input channels ({c}) doesn\'t match expected channels ({self.channels})'

        # 初始卷积层 - 对于28x28的输入，使用(2, 2)的kernel和stride
        x = nn.Conv(features=self.hidden_dim, kernel_size=(2, 2), strides=(2, 2))(img)  # 14x14

        # Stage模块
        num_stages = len(self.layers)
        for i in range(num_stages):
            # 计算当前特征图的尺寸
            curr_h = h // (2 * (2 ** sum(1 for j in range(i) if self.downscaling_factors[j] > 1)))
            curr_w = w // (2 * (2 ** sum(1 for j in range(i) if self.downscaling_factors[j] > 1)))

            # 如果当前尺寸太小，则不再进行下采样
            actual_downscaling = self.downscaling_factors[i] if curr_h > 7 and curr_w > 7 else 1

            x = StageModule(
                in_channels=self.hidden_dim * 2 ** max(i - 1, 0),
                hidden_dimension=self.hidden_dim * 2 ** i,
                layers=self.layers[i],
                downscaling_factor=actual_downscaling,  # 使用调整后的下采样因子
                num_heads=self.heads[i],
                head_dim=self.head_dim,
                window_size=self.window_size,
                relative_pos_embedding=self.relative_pos_embedding,
                quantum_w_shape=self.quantum_w_shape,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit
            )(x, deterministic=deterministic)

        x = jnp.mean(x, axis=[2, 3])
        x = nn.Dense(features=self.num_classes)(x)
        return x



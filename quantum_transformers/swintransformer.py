import flax.linen as nn
import jax.numpy as jnp
from typing import Optional, Callable
from quantum_transformers.transformers import MultiHeadSelfAttention, TransformerBlock

class PatchEmbed(nn.Module):
    img_size: int = 224
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 96
    norm_layer: Optional[Callable] = None

    @nn.compact
    def __call__(self, x):
        _, H, W, C = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."

        # padding
        pad_input = (H % self.patch_size != 0) or (W % self.patch_size != 0)
        if pad_input:
            x = jnp.pad(x, (
            (0, 0), (0, self.patch_size - H % self.patch_size), (0, self.patch_size - W % self.patch_size), (0, 0)))
        _, H, W, C = x.shape

        x = nn.Conv(features=self.embed_dim, kernel_size=self.patch_size, strides=self.patch_size, name='proj')(x)
        if self.norm_layer is not None:
            x = self.norm_layer(name='norm')(x)

        Hp = H // self.patch_size
        Wp = W // self.patch_size

        x = jnp.reshape(x, (-1, Hp * Wp, self.embed_dim))
        return x, (Hp, Wp)

class PatchMerging(nn.Module):
    dim: int
    norm_layer: Callable = nn.LayerNorm

    @nn.compact
    def __call__(self, x, Hp, Wp):
        B, L, C = x.shape
        H, W = Hp, Wp
        assert L == H * W, "input feature has wrong size"

        x = jnp.reshape(x, (B, H, W, C))

        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = jnp.pad(x, ((0, 0), (0, H % 2), (0, W % 2), (0, 0)))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = jnp.concatenate([x0, x1, x2, x3], axis=-1)  # B H/2 W/2 4*C
        x = jnp.reshape(x, (B, -1, 4 * C))  # B H/2*W/2 4*C

        x = self.norm_layer(name='norm')(x)
        x = nn.Dense(features=2 * C, use_bias=False, name='reduction')(x)

        return x

class WindowAttention(nn.Module):
    dim: int
    window_size: tuple
    num_heads: int
    qkv_bias: bool = True
    attn_drop: float = 0.
    proj_drop: float = 0.
    pretrained_window_size: tuple = (0, 0)

    quantum_w_shape: tuple = (1,)
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, mask, deterministic):
        B, N, C = x.shape
        head_dim = C // self.num_heads

        # Linear transformations for Q, K, V
        qkv = nn.Dense(features=self.dim * 3, use_bias=self.qkv_bias, name='qkv')(x)
        qkv = jnp.reshape(qkv, (B, N, 3, self.num_heads, head_dim))
        qkv = jnp.transpose(qkv, (2, 0, 3, 1, 4))
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)

        # Scaled dot-product attention
        attn = (q @ jnp.swapaxes(k, -2, -1)) * (head_dim ** -0.5)

        # Apply mask if provided
        if mask is not None:
            nW = mask.shape[0]
            attn = jnp.reshape(attn, (B // nW, nW, self.num_heads, N, N)) + mask.unsqueeze(1).unsqueeze(0)
            attn = jnp.reshape(attn, (-1, self.num_heads, N, N))

        # Softmax
        attn = nn.softmax(attn, axis=-1)
        attn = nn.Dropout(rate=self.attn_drop)(attn, deterministic=deterministic)

        # Output
        x = jnp.reshape(jnp.transpose((attn @ v), (0, 2, 1, 3)), (B, N, C))
        x = nn.Dense(features=self.dim, name='proj')(x)
        x = nn.Dropout(rate=self.proj_drop)(x, deterministic=deterministic)
        return x

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = jnp.reshape(x, (B, H // window_size, window_size, W // window_size, window_size, C))
    windows = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(-1, window_size, window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = jnp.reshape(windows, (B, H // window_size, W // window_size, window_size, window_size, -1))
    x = jnp.transpose(x, (0, 1, 3, 2, 4, 5)).reshape(B, H, W, -1)
    return x

class SwinTransformerBlock(nn.Module):
    dim: int
    num_heads: int
    window_size: int
    shift_size: int
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    drop: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.
    act_layer: Callable = nn.gelu
    norm_layer: Callable = nn.LayerNorm
    pretrained_window_size: int = 0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, Hp, Wp, deterministic):
        B, L, C = x.shape
        H, W = Hp, Wp
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm_layer(name='norm1')(x)
        x = jnp.reshape(x, (B, H, W, C))

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = jnp.pad(x, ((0, 0), (0, pad_b), (0, pad_r), (0, 0)))
        _, Hp, Wp, _ = x.shape

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = jnp.roll(x, shift=(-self.shift_size, -self.shift_size), axis=(1, 2))
            img_mask = jnp.zeros((1, Hp, Wp, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask = img_mask.at[:, h, w, :].set(cnt)
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = jnp.reshape(mask_windows, (-1, self.window_size * self.window_size))
            attn_mask = jnp.expand_dims(mask_windows, 1) - jnp.expand_dims(mask_windows, 2)
            attn_mask = jnp.where(attn_mask != 0, -100.0, attn_mask)
            attn_mask = jnp.where(attn_mask == 0, 0.0, attn_mask)
        else:
            shifted_x = x
            attn_mask = None

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = jnp.reshape(x_windows, (-1, self.window_size * self.window_size, C))

        # W-MSA/SW-MSA
        attn_windows = WindowAttention(
            dim=self.dim,
            window_size=(self.window_size, self.window_size),
            num_heads=self.num_heads,
            qkv_bias=self.qkv_bias,
            attn_drop=self.attn_drop,
            proj_drop=self.drop,
            pretrained_window_size=(self.pretrained_window_size, self.pretrained_window_size),
            quantum_w_shape=self.quantum_w_shape,
            quantum_circuit=self.quantum_attn_circuit
        )(x_windows, mask=attn_mask, deterministic=deterministic)

        # merge windows
        attn_windows = jnp.reshape(attn_windows, (-1, self.window_size, self.window_size, C))
        shifted_x = window_reverse(attn_windows, self.window_size, Hp, Wp)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = jnp.roll(shifted_x, shift=(self.shift_size, self.shift_size), axis=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :]

        x = jnp.reshape(x, (B, H * W, C))

        # FFN
        x = shortcut + x
        # Use TransformerBlock from transformers.py
        x = TransformerBlock(
            hidden_size=self.dim,
            num_heads=self.num_heads,
            mlp_hidden_size=int(self.dim * self.mlp_ratio),
            dropout=self.drop,
            quantum_attn_circuit=self.quantum_attn_circuit,
            quantum_mlp_circuit=self.quantum_mlp_circuit
        )(x, deterministic=deterministic)

        return x, H, W

class BasicLayer(nn.Module):
    dim: int
    depth: int
    num_heads: int
    window_size: int
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    drop: float = 0.
    attn_drop: float = 0.
    drop_path: float = 0.
    norm_layer: Callable = nn.LayerNorm
    downsample: Optional[Callable] = None
    use_checkpoint: bool = False
    pretrained_window_size: int = 0

    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, Hp, Wp, deterministic):
        for i in range(self.depth):
            x, Hp, Wp = SwinTransformerBlock(
                dim=self.dim,
                num_heads=self.num_heads,
                window_size=self.window_size,
                shift_size=0 if (i % 2 == 0) else self.window_size // 2,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop,
                attn_drop=self.attn_drop,
                drop_path=self.drop_path[i] if isinstance(self.drop_path, list) else self.drop_path,
                norm_layer=self.norm_layer,
                pretrained_window_size=self.pretrained_window_size,
                quantum_w_shape=self.quantum_w_shape,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit
            )(x, Hp, Wp, deterministic)

        if self.downsample is not None:
            x = self.downsample(dim=self.dim, norm_layer=self.norm_layer, name='downsample')(x, Hp, Wp)
            Hp, Wp = (Hp + 1) // 2, (Wp + 1) // 2

        return x, Hp, Wp

class SwinTransformer(nn.Module):
    num_classes: int = 1000
    patch_size: int = 4
    in_chans: int = 3
    embed_dim: int = 96
    depths: tuple = (2, 2, 6, 2)
    num_heads: tuple = (3, 6, 12, 24)
    window_size: int = 7
    mlp_ratio: float = 4.
    qkv_bias: bool = True
    drop_rate: float = 0.
    attn_drop_rate: float = 0.
    drop_path_rate: float = 0.1
    norm_layer: Callable = nn.LayerNorm
    ape: bool = False
    patch_norm: bool = True
    use_checkpoint: bool = False
    pretrained_window_sizes: tuple = (0, 0, 0, 0)
    quantum_w_shape: tuple = (1,)
    quantum_attn_circuit: Optional[Callable] = None
    quantum_mlp_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, train):
        x, (Hp, Wp) = PatchEmbed(
            img_size=self.window_size * 4,
            patch_size=self.patch_size,
            in_chans=self.in_chans,
            embed_dim=self.embed_dim,
            norm_layer=self.norm_layer if self.patch_norm else None
        )(x)

        if self.ape:
            x += self.param('absolute_pos_embed', nn.initializers.zeros, (1, Hp * Wp, self.embed_dim))
        x = nn.Dropout(rate=self.drop_rate)(x, deterministic=not train)

        # stochastic depth
        dpr = [x.item() for x in jnp.linspace(0, self.drop_path_rate, sum(self.depths))]  # stochastic depth decay rule

        # build layers
        for i_layer in range(len(self.depths)):
            x, Hp, Wp = BasicLayer(
                dim=int(self.embed_dim * 2 ** i_layer),
                depth=self.depths[i_layer],
                num_heads=self.num_heads[i_layer],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=self.qkv_bias,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[sum(self.depths[:i_layer]):sum(self.depths[:i_layer + 1])],
                norm_layer=self.norm_layer,
                downsample=PatchMerging if (i_layer < len(self.depths) - 1) else None,
                use_checkpoint=self.use_checkpoint,
                pretrained_window_size=self.pretrained_window_sizes[i_layer],
                quantum_w_shape=self.quantum_w_shape,
                quantum_attn_circuit=self.quantum_attn_circuit,
                quantum_mlp_circuit=self.quantum_mlp_circuit
            )(x, Hp, Wp, deterministic=not train)

        x = self.norm_layer(name='norm')(x)  # B L C
        x = jnp.mean(x, axis=1)  # B C
        x = nn.Dense(features=self.num_classes, name='head')(x)

        return x
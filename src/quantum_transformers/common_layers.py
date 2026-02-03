from typing import Optional, Callable
import flax.linen as nn
from src.quantum_transformers.quantum_layer import QuantumLayer

class FeedForward(nn.Module):
    dim: int
    hidden_dim: int
    dropout: float = 0.0
    quantum_w_shape: tuple = (1,)
    quantum_circuit: Optional[Callable] = None

    @nn.compact
    def __call__(self, x, deterministic=True):
        x = nn.Dense(features=self.hidden_dim)(x)
        if self.quantum_circuit is not None:
            x = QuantumLayer(num_qubits=self.hidden_dim,
                             w_shape=self.quantum_w_shape,
                             circuit=self.quantum_circuit)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)
        x = nn.gelu(x)
        x = nn.Dense(features=self.dim)(x)
        return x

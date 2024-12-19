import jax.numpy as jnp
from src.quantum_transformers.quantum_layer_pennylane import get_circuit

# テスト用データ
x = jnp.ones((128, 8))  # 入力データ (512サンプル, 8特徴量)
w = jnp.ones((1, 8))    # 重み (1重みベクトル, 8特徴量)

# 回路の取得
qpred_batch = get_circuit()

# 結果を取得
output = qpred_batch(x, w)

print("出力の形状:", output.shape)  # (512, 8) が期待される
print("出力:", output)

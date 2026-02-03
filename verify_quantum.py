import jax
import jax.numpy as jnp
from src.quantum_transformers.quantum_layer import QuantumLayer, get_circuit
import tensorcircuit as tc

def verify():
    print("Verifying QuantumLayer with TensorCircuit...")
    
    # Check backend
    print(f"TensorCircuit Backend: {tc.backend.name}")
    
    # Initialize layer
    num_qubits = 4
    circuit = get_circuit()
    layer = QuantumLayer(num_qubits=num_qubits, w_shape=(3,), circuit=circuit)
    
    # Init parameters
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, (2, num_qubits)) # Batch size 2
    params = layer.init(key, x)
    
    # Forward pass
    y = layer.apply(params, x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("Output values:", y)
    
    assert y.shape == x.shape
    print("Verification complete.")

if __name__ == "__main__":
    verify()

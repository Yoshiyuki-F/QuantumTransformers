import pennylane as qml
from pennylane import numpy as pnp
import jax
import jax.numpy as jnp
from typing import Callable, Optional
import flax.linen as nn

def angle_embedding(inputs, wires):
    qml.AngleEmbedding(inputs, wires=wires, rotation='X')

def basic_vqc(weights, wires):
    qml.BasicEntanglerLayers(weights, wires=wires)

def get_circuit(embedding: Callable = angle_embedding, vqc: Callable = basic_vqc, num_qubits: int = None):
    """
    Creates a quantum circuit. num_qubits is now a required argument.
    """
    if num_qubits is None:
        raise ValueError("num_qubits must be provided to get_circuit().")

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface='jax')
    def qpred(inputs, weights):
        embedding(inputs, wires=range(num_qubits))
        vqc(weights, wires=range(num_qubits))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    def quantum_function(inputs, weights):
        # Ensure inputs are in the correct format (float32) and shape
        inputs = jnp.asarray(inputs, dtype=jnp.float32)

        # Process each input in the batch
        results = []
        for inp in inputs:
            result = qpred(inp, weights)
            results.append(jnp.stack(result))

        # Stack results
        results = jnp.stack(results)
        return results

    return quantum_function

class QuantumLayer(nn.Module):
    circuit: Callable
    num_qubits: int
    w_shape: tuple = (1,)

    def setup(self):
        # Initialize weights for the quantum circuit
        self.key = jax.random.PRNGKey(0)
        self.weights = self.param('weights', jax.nn.initializers.normal(), (self.w_shape + (self.num_qubits,)))

    @nn.compact
    def __call__(self, inputs):
        # Call the quantum function, passing num_qubits
        quantum_out = self.circuit(inputs, self.weights)

        # Flatten the quantum output to (batch_size, num_qubits)
        batch_size = inputs.shape[0]
        flattened_output = jnp.reshape(quantum_out, (batch_size, self.num_qubits))

        # Dynamically determine the output features based on input shape
        output_features = inputs.shape[-1]

        # Add a dense layer to project the output to the correct dimension
        output = nn.Dense(features=output_features)(flattened_output)

        return output

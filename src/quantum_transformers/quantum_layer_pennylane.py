import pennylane as qml
import jax
import jax.numpy as jnp
import logging
logging.basicConfig(level=logging.INFO)  # ログレベルを設定



def preprocess_inputs(inputs):
    if not hasattr(inputs, 'shape'):
        raise ValueError(f"Input is not a tensor: {inputs}")
    if inputs.shape == ():
        raise ValueError(f"Unexpected scalar input: {inputs}. Expected 1D or 2D tensor.")
    return inputs


def angle_embedding(inputs, num_qubits):
    qml.AngleEmbedding(inputs, wires=range(num_qubits), rotation='X')

def basic_vqc(weights, num_qubits):
    qml.BasicEntanglerLayers(weights, wires=range(num_qubits))



def get_quantum_layer_circuit(dev, num_qubits):
    @qml.qnode(dev, diff_method="adjoint", interface='jax')
    def circuit(inputs, weights):
        angle_embedding(inputs, num_qubits)
        basic_vqc(weights, num_qubits)
        # return jnp.real(jnp.array([qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]))
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return circuit


def get_circuit():
    def qpred(inputs, weights):
        inputs = preprocess_inputs(inputs)
        num_qubits = inputs.shape[-1]
        dev = qml.device("lightning.gpu", wires=num_qubits)
        circuit = get_quantum_layer_circuit(dev, num_qubits)
        batched_circuit = jax.vmap(circuit, in_axes=(0, None))

        return jnp.swapaxes(jnp.array(batched_circuit(inputs, weights)), 0, 1)

    return qpred



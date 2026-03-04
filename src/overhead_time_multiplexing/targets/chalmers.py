import qiskit
from qiskit.circuit import Parameter

# Dummy parameter for single-qubit parameterized gates
theta = Parameter("ϴ")

# Single-qubit gates
chalmers_single_qubit_gates = [
    (qiskit.circuit.library.standard_gates.RXGate(theta), "rx"),
    (qiskit.circuit.library.standard_gates.RYGate(theta), "ry"),
    (qiskit.circuit.library.standard_gates.RZGate(theta), "rz"),
    (qiskit.circuit.library.standard_gates.HGate(), "h"),
    (qiskit.circuit.library.standard_gates.IGate(), "i"),
]

# Two-qubit gates
chalmers_two_qubit_gates = [
    (qiskit.circuit.library.standard_gates.CZGate(), "cz"),
    (qiskit.circuit.library.standard_gates.iSwapGate(), "iswap"),
]

# Add native gates
chalmers_native_gates = chalmers_single_qubit_gates + chalmers_two_qubit_gates

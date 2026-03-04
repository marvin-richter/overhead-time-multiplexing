from typing import Tuple

import numpy as np
import qiskit
from qiskit import QuantumCircuit

EXCLUDE_GATES = {
    "measure",
    "barrier",
    "delay",
    "reset",
    "snapshot",
    "save",
    "save_statevector",
    "sdel",
    "sw2",
}


def calculate_avg_operands(qc: qiskit.QuantumCircuit) -> float:
    """Calculate average operands for non-measurement gates."""
    non_measure_gates = [gate for gate in qc.data if gate.name not in EXCLUDE_GATES]

    if not non_measure_gates:
        return 0.0

    return float(np.mean([len(gate.qubits) for gate in non_measure_gates]))


def single_qubit_gate_filter(instruction_tuple):
    # instruction, qubits, clbits = instruction_tuple
    name, qubits = instruction_tuple.name, instruction_tuple.qubits
    # Only count actual gates (not measurements, barriers, etc.) on single qubits
    return len(qubits) == 1 and name.lower() not in EXCLUDE_GATES


def two_qubit_gate_filter(instruction_tuple):
    name, qubits = instruction_tuple.name, instruction_tuple.qubits
    # Only count actual gates (not measurements, barriers, etc.) on two qubits
    return len(qubits) == 2 and name.lower() not in EXCLUDE_GATES


def compute_actual_densities(
    circuit: QuantumCircuit, filtered: bool = True
) -> Tuple[float, float, float]:
    """
    Compute the actual gate densities of a circuit.

    Returns
    -------
    tuple of (rho_1, rho_2, rho_total)
    """
    n = circuit.num_qubits
    D = circuit.depth()

    if n == 0 or D == 0:
        return 0.0, 0.0, 0.0

    if filtered:
        n_1q = circuit.size(single_qubit_gate_filter)
        n_2q = circuit.size(two_qubit_gate_filter)
    else:
        n_1q = sum(1 for inst in circuit.data if inst.operation.num_qubits == 1)
        n_2q = sum(1 for inst in circuit.data if inst.operation.num_qubits == 2)

    rho_1 = n_1q / (n * D)
    rho_2 = 2 * n_2q / (n * D)
    rho_total = rho_1 + rho_2

    return {"rho_1": rho_1, "rho_2": rho_2, "rho_total": rho_total}

"""
Test script for  SerializeGatesSwitchPass on multiple circuits.

Optionally check compilation equivalence using mqt.qcec if available.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import PassManager

from overhead_time_multiplexing import (
    SerializeGatesSwitchPass,
    generate_layout,
)
from overhead_time_multiplexing.targets import construct_chalmers_target

try:
    from mqt import qcec

    QCEC_AVAILABLE = True
except ImportError:
    QCEC_AVAILABLE = False


@pytest.fixture
def target():
    """Create a target for testing."""

    target = construct_chalmers_target(num_rows=5, num_cols=5, full_coupling=False)

    return target


@pytest.fixture
def layout(target):
    controller_layout = generate_layout(
        target=target,
        layout_type="trivial",
        qpg=4,
        hardware_label="chalmers_25",
    )

    return controller_layout


@pytest.fixture
def test_circuits():
    """Create a list of test circuits for demonstration."""
    n = 6
    circuits = []

    # Circuit 1: Simple manual circuit
    qc1 = QuantumCircuit(n, name="manual_circuit")
    qc1.h(0)
    qc1.cx(0, 1)
    qc1.rz(0.5, 2)
    qc1.x(3)
    qc1.y(4)
    qc1.z(5)
    qc1.cx(1, 2)
    qc1.measure_all()
    circuits.append(qc1)

    # Circuit 2: Random circuit
    qc2 = random_circuit(
        num_qubits=6,
        depth=4,
        max_operands=2,
        seed=1241244,
        measure=False,
    )
    qc2.measure_all()
    qc2.name = "random_circuit_seed1241244"
    circuits.append(qc2)

    # Circuit 3: Another random circuit with different parameters
    qc3 = random_circuit(
        num_qubits=6,
        depth=6,
        max_operands=2,
        seed=1234,
        measure=False,
    )
    qc3.measure_all()
    qc3.name = "random_circuit_seed1234"
    circuits.append(qc3)

    # Circuit 4: Bell states circuit
    qc4 = QuantumCircuit(n, name="bell_states")
    for i in range(0, n - 1, 2):
        qc4.h(i)
        if i + 1 < n:
            qc4.cx(i, i + 1)
    qc4.measure_all()
    circuits.append(qc4)

    # Circuit 5: Rz and Rx heavy circuit
    qc5 = QuantumCircuit(n, name="rz_heavy_circuit")
    rng = np.random.default_rng(123454664)  # For reproducibility
    for i in range(5):
        # chose qubits for rx,
        qubits_rx = rng.choice(n, size=2, replace=False)

        for q in qubits_rx:
            qc5.rx(0.1 * (i + 1), q)

        # chose qubits for rz,
        qubits_rz = rng.choice(n, size=2, replace=False)
        for q in qubits_rz:
            qc5.rz(0.2 * (i + 1), q)

        # place some two-qubit gates
        if i % 2 == 0:
            qc5.cx(qubits_rx[0], qubits_rz[0])
        else:
            qc5.cx(qubits_rx[1], qubits_rz[1])

    qc5.measure_all()
    circuits.append(qc5)

    return circuits


class TestSerializeGatesSwitchPass:
    """Test class for SerializeGatesSwitchPass."""

    @pytest.mark.parametrize("circuit_idx", range(5))
    def test_switch_pass_execution(self, test_circuits, circuit_idx, layout):
        """Test that SerializeGatesSwitchPass executes without errors."""
        qc = test_circuits[circuit_idx]

        pm_switch = PassManager()
        pm_switch.append(
            SerializeGatesSwitchPass(
                qubit_to_group=layout.qubit_to_group,
                virtual_rz=True,
                topord_method="prio_two",
            )
        )

        # This should not raise an exception
        out_circ_switch = pm_switch.run(qc)

        # Basic validations
        assert out_circ_switch is not None, f"Switch pass failed to produce output for {qc.name}"
        assert isinstance(out_circ_switch, QuantumCircuit), "Output should be a QuantumCircuit"
        assert out_circ_switch.num_qubits == qc.num_qubits, "Number of qubits should be preserved"

    def test_switch_pass_properties(self, test_circuits, layout):
        """Test that SerializeGatesSwitchPass sets expected properties."""
        qc = test_circuits[0]  # Use first test circuit

        pm_switch = PassManager()
        pm_switch.append(
            SerializeGatesSwitchPass(
                qubit_to_group=layout.qubit_to_group,
                virtual_rz=True,
                topord_method="prio_two",
            )
        )

        _ = pm_switch.run(qc)

        # Check that properties are set
        switch_gates = pm_switch.property_set.get("serial_switch_gates", 0)
        switch2_gates = pm_switch.property_set.get("serial_switch2_gates", 0)

        assert isinstance(switch_gates, int), "serial_switch_gates should be an integer"
        assert isinstance(switch2_gates, int), "serial_switch2_gates should be an integer"
        assert switch_gates >= 0, "serial_switch_gates should be non-negative"
        assert switch2_gates >= 0, "serial_switch2_gates should be non-negative"


class TestPassEquivalence:
    """Test class for verifying compilation equivalence."""

    @pytest.mark.skipif(not QCEC_AVAILABLE, reason="mqt.qcec not available")
    @pytest.mark.parametrize("circuit_idx", range(5))
    def test_switch_pass_equivalence(self, test_circuits, circuit_idx, layout):
        """Test that SerializeGatesSwitchPass preserves circuit equivalence."""
        qc = test_circuits[circuit_idx]

        pm_switch = PassManager()
        pm_switch.append(
            SerializeGatesSwitchPass(
                qubit_to_group=layout.qubit_to_group,
                virtual_rz=True,
                topord_method="prio_two",
            )
        )

        out_circ_switch = pm_switch.run(qc)

        # Verify equivalence
        results = qcec.verify_compilation(qc, out_circ_switch, optimization_level=1)
        # Check if the circuits are equivalent (result should be a boolean or have .equivalent attribute)
        if hasattr(results, "equivalent"):
            equivalent = results.equivalent
        elif hasattr(results, "equivalence"):
            equivalent = results.equivalence
        else:
            equivalent = bool(results)

        assert equivalent, f"Switch pass broke equivalence for {qc.name}"

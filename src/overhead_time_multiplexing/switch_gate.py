"""Custom switch gate."""

from typing import Optional
from qiskit.circuit.gate import Gate
from qiskit.circuit._utils import with_gate_array
from qiskit.circuit.library.standard_gates.i import IGate


@with_gate_array([[1, 0], [0, 1]])  # Identity matrix
class SwitchGate(Gate):
    r"""Single-qubit controller-switch delay gate.

    Acts as the identity operation but carries a non-zero duration (defined in
    the hardware target) that models the time the target qubit must wait while
    the controller switches over to it.

    **Circuit symbol:**

    .. code-block:: text

             ┌────┐
        q_0: ┤ SW ├
             └────┘

    **Matrix representation:**

    .. math::

        SW = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}
    """

    def __init__(self, label: Optional[str] = None):
        """Create new switch gate.

        Args:
            label: An optional label for the gate [Default: None]
        """
        super().__init__("SDel", 1, [], label=label)

    def _define(self):
        """Define the gate as an identity operation."""
        from qiskit.circuit import QuantumCircuit, QuantumRegister

        q = QuantumRegister(1, "q")
        qc = QuantumCircuit(q, name=self.name)
        rules = [(IGate(), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return the inverse gate (itself, since it's identity).

        Args:
            annotated: when set to True, this is typically used to return an
                AnnotatedOperation with an inverse modifier set instead of a concrete
                Gate. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            Switchgate: inverse gate (self-inverse).
        """
        return SwitchGate(label=self.label)

    def power(self, exponent: float, annotated: bool = False):
        """Return the gate raised to a power (still identity for any power)."""
        return SwitchGate(label=self.label)

    def __eq__(self, other):
        """Check equality with another gate."""
        return isinstance(other, SwitchGate) and self.label == other.label


@with_gate_array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
class Switch2Gate(Gate):
    r"""Two-qubit controller-switch dependency gate.

    Encodes the dependency between the qubit that was previously driven by a
    controller and the qubit it is switching to. Like :class:`SwitchGate`, it
    acts as the identity but carries a duration in the hardware target.

    **Circuit symbol:**

    .. code-block:: text

        q_0: ─[sw2]─
        q_1: ─[sw2]─

    **Matrix representation:**

    .. math::

        SW_2 = \begin{pmatrix} 1&0&0&0 \\ 0&1&0&0 \\ 0&0&1&0 \\ 0&0&0&1 \end{pmatrix}
    """

    def __init__(self, label: Optional[str] = None):
        """Create new two-qubit switch gate.

        Args:
            label: An optional label for the gate [Default: None]
        """
        super().__init__("sw2", 2, [], label=label)

    def _define(self):
        """Define the gate as an identity operation on two qubits."""
        from qiskit.circuit import QuantumCircuit, QuantumRegister

        q = QuantumRegister(2, "q")
        qc = QuantumCircuit(q)
        rules = [(IGate(), [q[0]], []), (IGate(), [q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool = False):
        """Return the inverse gate (itself, since it's identity).

        Args:
            annotated: when set to True, this is typically used to return an
                AnnotatedOperation with an inverse modifier set instead of a concrete
                Gate. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            Switch2Gate: inverse gate (self-inverse).
        """
        return Switch2Gate(label=self.label)


def add_switch2_gate_method(self, qubit1, qubit2, label: Optional[str] = None):
    """Add a two-qubit switching gate to the circuit.

    Args:
        qubit1: The first qubit to apply the switching gate to
        qubit2: The second qubit to apply the switching gate to
        label: Optional label for the gate

    Returns:
        QuantumCircuit: The circuit with the two-qubit switching gate added
    """
    return self.append(Switch2Gate(label=label), [qubit1, qubit2], [])


def add_switch_gate_method(self, qubit, label: Optional[str] = None):
    """Add a switch gate to the circuit.

    Args:
        qubit: The qubit to apply the switch gate to
        label: Optional label for the gate

    Returns:
        QuantumCircuit: The circuit with the switch gate added
    """
    return self.append(SwitchGate(label=label), [qubit], [])

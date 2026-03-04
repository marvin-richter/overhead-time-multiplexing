from qiskit.transpiler import CouplingMap, InstructionProperties, Target
from qiskit.circuit import Measure
from .. import SwitchGate, Switch2Gate
from .chalmers import chalmers_native_gates


def _add_chalmers_gates(
    target: Target,
    coupling_map: CouplingMap,
    single_qubit_duration: float,
    two_qubit_duration: float,
    switch_duration: float,
    virtual_rz: bool,
    with_switch: bool,
):
    """Add Chalmers native gates to target."""
    connected_qubits = {x for edge in coupling_map.get_edges() for x in edge}

    for gate, name in chalmers_native_gates:
        props = {}
        if gate.num_qubits == 1:
            duration = 0.0 if virtual_rz and name == "rz" else single_qubit_duration
            for qubit in connected_qubits:
                props[(qubit,)] = InstructionProperties(duration=duration, error=0.000)
        elif gate.num_qubits == 2:
            for edge in coupling_map.get_edges():
                props[edge] = InstructionProperties(duration=two_qubit_duration, error=0.00)

        target.add_instruction(gate, props, name=name)

    # Add measure instruction
    target.add_instruction(
        Measure(),
        {(q,): InstructionProperties(duration=0.0, error=0.00) for q in range(target.num_qubits)},
        name="measure",
    )

    # Add switch gates if requested
    if with_switch:
        _add_switch_gates(target, switch_duration)


def _add_switch_gates(target, switch_duration):
    """Add switch2 gates to target."""

    # Add single-qubit switch (delay) gate
    # This is a dummy gate that does nothing but is used for time multiplexing
    # It has switch_duration as its duration

    props = {}
    for qubit in range(target.num_qubits):
        props[(qubit,)] = InstructionProperties(duration=switch_duration, error=0.00)
    target.add_instruction(SwitchGate(), props, name="SDel")

    # Add two-qubit switch gate
    # This is a dummy gate that does nothing but is used for time multiplexing, causing
    # serialization
    # It has zero duration

    props = {}
    for q1 in range(target.num_qubits - 1):
        for q2 in range(q1 + 1, target.num_qubits):
            props[(q1, q2)] = InstructionProperties(duration=0.0, error=0.00)
            props[(q2, q1)] = InstructionProperties(duration=0.0, error=0.00)

    target.add_instruction(Switch2Gate(), props, name="sw2")


def update_coupling_map(
    original_target: Target,
    coupling_map: CouplingMap,
):
    """Update the coupling map of a target. Go iteratively through the target's
    instructions and update the properties based on the coupling map (e.g. if full)."""

    # Create a new target with same basic properties
    new_target = Target(
        description=f"Fully connected version of: {original_target.description or 'Original target'}",
        num_qubits=original_target.num_qubits,
        dt=original_target.dt,
        granularity=original_target.granularity,
        min_length=original_target.min_length,
        pulse_alignment=original_target.pulse_alignment,
        acquire_alignment=original_target.acquire_alignment,
        qubit_properties=original_target.qubit_properties,
        concurrent_measurements=original_target.concurrent_measurements,
    )

    def calculate_average_properties(gate_properties):
        """
        Helper function to calculate average properties from existing gate properties.
        """
        if not gate_properties:
            return None

        # Get all non-None properties
        valid_props = [props for props in gate_properties.values() if props is not None]

        if not valid_props:
            return None

        # Calculate averages
        avg_duration = sum(
            props.duration for props in valid_props if props.duration is not None
        ) / len(valid_props)
        avg_error = sum(props.error for props in valid_props if props.error is not None) / len(
            valid_props
        )

        return InstructionProperties(duration=avg_duration, error=avg_error)

    # If no specific gates specified, find all two-qubit gates in the original target
    two_qubit_gates = []
    for gate_name in original_target.operation_names:
        qargs = original_target.qargs_for_operation_name(gate_name)

        if qargs is not None:
            for qargs in original_target.qargs_for_operation_name(gate_name):
                if qargs is not None and len(qargs) == 2:
                    two_qubit_gates.append(gate_name)
                    break
    # Remove duplicates
    two_qubit_gates = list(set(two_qubit_gates))

    # get all pairs from coupling map
    all_qubit_pairs = coupling_map.get_edges()

    # Copy all instructions from original target
    for gate_name in original_target.operation_names:
        operation = original_target.operation_from_name(gate_name)
        original_qargs_set = original_target.qargs_for_operation_name(gate_name)
        original_props = original_target.get(gate_name)

        has_global_properties = any(qubits is None for qubits in original_props.keys())

        if has_global_properties:
            # For instructions with global properties, add without properties
            # These are typically control flow instructions
            new_target.add_instruction(operation, name=gate_name)
            continue

        # Check if this is a two-qubit gate we want to make fully connected
        if gate_name in two_qubit_gates:
            # Create fully connected properties for this gate
            new_props = {}

            for qargs in all_qubit_pairs:
                if qargs in original_qargs_set:
                    # Use existing properties if available
                    existing_props = original_target[gate_name].get(qargs)
                    new_props[qargs] = existing_props
                else:
                    #  Use average properties from existing connections
                    # This is more sophisticated but requires calculating averages
                    new_props[qargs] = calculate_average_properties(original_target[gate_name])

            new_target.add_instruction(operation, new_props)
        else:
            # For non-two-qubit gates or gates not in our list, copy as-is
            original_props = {}
            for qargs in original_qargs_set:
                original_props[qargs] = original_target[gate_name].get(qargs)

            new_target.add_instruction(operation, original_props)

    return new_target

from qiskit.transpiler import CouplingMap, Target
from .utils import _add_chalmers_gates, _add_switch_gates, update_coupling_map
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2


def construct_chalmers_target(
    num_rows: int,
    num_cols: int,
    full_coupling: bool = False,
    with_switch: bool = False,
    virtual_rz: bool = True,
    single_qubit_duration: float = 20.0,
    two_qubit_duration: float = 200.0,
    switch_duration: float = 2.0,
) -> Target:
    """Factory function to create targets based on name.
    If name starts with "chalmers", create a target with Chalmers native gates and specified properties.
    If name starts with "ibm", create a target based on IBM hardware properties.
    """
    num_qubits = num_rows * num_cols
    target = Target(num_qubits=num_qubits)

    if full_coupling:
        coupling_map = CouplingMap.from_full(num_qubits)
    else:
        coupling_map = CouplingMap.from_grid(num_rows, num_cols)

    _add_chalmers_gates(
        target=target,
        coupling_map=coupling_map,
        single_qubit_duration=single_qubit_duration,
        two_qubit_duration=two_qubit_duration,
        switch_duration=switch_duration,
        virtual_rz=virtual_rz,
        with_switch=with_switch,
    )
    return target


def construct_ibm_target(
    fake_label: str = "fake_washington",
    full_coupling: bool = False,
    with_switch: bool = False,
    switch_duration: float = 2.0,
) -> Target:
    # get the fake target from qiskit
    provider = FakeProviderForBackendV2()

    if not fake_label.startswith("fake_"):
        fake_label = "fake_" + fake_label

    try:
        backend = provider.backend(fake_label)
    except Exception as e:
        raise ValueError(f"IBM backend '{fake_label}' not found in FakeProvider") from e

    target = backend.target
    num_qubits = target.num_qubits
    if full_coupling:
        coupling_map = CouplingMap.from_full(num_qubits)
        # update target to have full coupling
        target = update_coupling_map(target, coupling_map)
    else:
        coupling_map = target.build_coupling_map()

    # add switch gates if needed
    if with_switch:
        _add_switch_gates(target, switch_duration)

    return target

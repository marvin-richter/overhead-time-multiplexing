from mqt.bench import BenchmarkLevel, get_benchmark
from qiskit import QuantumCircuit
from qiskit.transpiler import Target

from overhead_time_multiplexing.experiments.models import (
    ExperimentConfig,
    HardwareConfig,
    HardwareKind,
)
from overhead_time_multiplexing.experiments.random_circuits import (
    random_circuit_custom_pattern,
    random_circuit_fixed_density,
    random_circuit_native,
)
from overhead_time_multiplexing.layouts import generate_layout
from overhead_time_multiplexing.targets import construct_chalmers_target, construct_ibm_target


def construct_target_from_hw_config(hw_cfg: HardwareConfig) -> Target:
    # Step 1 parse config and get the qiskit target
    # NOTE: the native target is used for layout generation, it should not have switch gates.
    # The full target is used for duration measurements, it needs to have switch gates.

    if hw_cfg.kind == HardwareKind.CH_GRID:
        target_native = construct_chalmers_target(
            num_rows=hw_cfg.num_rows,
            num_cols=hw_cfg.num_cols,
            full_coupling=False,
            with_switch=False,
            virtual_rz=hw_cfg.virtual_rz,
            single_qubit_duration=hw_cfg.t1,
            two_qubit_duration=hw_cfg.t2,
            switch_duration=hw_cfg.tsw,
        )

        target_full = construct_chalmers_target(
            num_rows=hw_cfg.num_rows,
            num_cols=hw_cfg.num_cols,
            full_coupling=True,
            with_switch=True,
            virtual_rz=hw_cfg.virtual_rz,
            single_qubit_duration=hw_cfg.t1,
            two_qubit_duration=hw_cfg.t2,
            switch_duration=hw_cfg.tsw,
        )

    elif hw_cfg.kind == HardwareKind.IBM:
        target_native = construct_ibm_target(
            fake_label=hw_cfg.id,
            full_coupling=False,
            with_switch=False,
            switch_duration=hw_cfg.tsw,
        )
        target_full = construct_ibm_target(
            fake_label=hw_cfg.id,
            full_coupling=True,
            with_switch=True,
            switch_duration=hw_cfg.tsw,
        )

    # Step 2 get the controller_layout, important: use the NATIVE target
    controller_layout = generate_layout(
        target=target_native,
        layout_type=hw_cfg.layout.strategy,
        qpg=hw_cfg.layout.k,
        hardware_label=hw_cfg.hardware_label,
        random_seed=hw_cfg.layout.seed,
    )

    return {
        "target_full": target_full,
        "target_native": target_native,
        "layout": controller_layout,
    }


def construct_circuit_from_config(exp_config: ExperimentConfig, targets) -> QuantumCircuit:
    if exp_config.circuit.source == "random":
        target_circuit = random_circuit_native(
            num_qubits=exp_config.circuit.num_qubits,
            num_gates=exp_config.circuit.num_gates,
            seed=exp_config.circuit.seed,
            gate_weights=(
                exp_config.circuit.random_weight_1q,
                1.0 - exp_config.circuit.random_weight_1q,
            ),
            native_gates=targets["target_full"].operations,
        )
        target_circuit.measure_all()
    elif exp_config.circuit.source == "random_densities":
        if exp_config.circuit.connectivity == "full":
            edges = [
                e for e in targets["target_full"].build_coupling_map().get_edges() if e[0] < e[1]
            ]
        elif exp_config.circuit.connectivity == "native":
            edges = [
                e for e in targets["target_native"].build_coupling_map().get_edges() if e[0] < e[1]
            ]
        else:
            edges = None

        target_circuit = random_circuit_fixed_density(
            num_qubits=exp_config.circuit.num_qubits,
            depth=exp_config.circuit.depth,
            rho_1=exp_config.circuit.rho_1,
            rho_2=exp_config.circuit.rho_tot - exp_config.circuit.rho_1,
            seed=exp_config.circuit.seed,
            native_gates=targets["target_full"].operations,
            coupling_map=edges,
        )
        target_circuit.measure_all()
    elif exp_config.circuit.source == "random_pattern":
        pattern = exp_config.circuit.id
        target_circuit = random_circuit_custom_pattern(
            num_qubits=exp_config.circuit.num_qubits,
            pattern=pattern,
            reps=exp_config.circuit.num_gates,
            seed=exp_config.circuit.seed,
            native_gates=targets["target_full"].operations,
            add_barriers=False,
        )
        target_circuit.measure_all()
    elif exp_config.circuit.source == "mqt":
        target_circuit = get_benchmark(
            exp_config.circuit.id,
            level=BenchmarkLevel["INDEP"],
            circuit_size=exp_config.circuit.num_qubits,
        )
    else:
        raise ValueError(f"Unknown source: {exp_config.circuit.source}")
    return target_circuit

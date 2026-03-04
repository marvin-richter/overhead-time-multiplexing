from overhead_time_multiplexing.experiments import (
    ExperimentConfig,
    flatten_config,
    construct_circuit_from_config,
    construct_target_from_hw_config,
)

from qiskit.transpiler import PassManager, generate_preset_pass_manager
from overhead_time_multiplexing import SerializeGatesSwitchPass
from overhead_time_multiplexing.experiments.utils import (
    single_qubit_gate_filter,
    two_qubit_gate_filter,
    calculate_avg_operands,
    compute_actual_densities,
)
import time


def worker(exp_config: ExperimentConfig):
    targets = construct_target_from_hw_config(exp_config.hardware)
    target_full, target_native, controller_layout = (
        targets["target_full"],
        targets["target_native"],
        targets["layout"],
    )
    qc = construct_circuit_from_config(exp_config, targets)

    pm_trans = generate_preset_pass_manager(
        target=target_full,
        optimization_level=exp_config.circuit.optimization_level,
        seed_transpiler=exp_config.circuit.seed,
    )
    qc_trans = pm_trans.run(qc)

    pm_route = generate_preset_pass_manager(
        target=target_native,
        optimization_level=exp_config.circuit.optimization_level,
        seed_transpiler=exp_config.circuit.seed,
    )
    qc_routed = pm_route.run(qc_trans)

    pm_serial_trans = PassManager()
    pm_serial_trans.append(
        [
            SerializeGatesSwitchPass(
                qubit_to_group=controller_layout.qubit_to_group,
                virtual_rz=exp_config.hardware.virtual_rz,
                topord_method=exp_config.serialization.topord_method,
                delay_check=exp_config.serialization.delay_check,
                operation_durations=target_full.durations(),
                switch_duration=exp_config.hardware.tsw,
            )
        ]
    )
    pm_serial_routed = PassManager()
    pm_serial_routed.append(
        [
            SerializeGatesSwitchPass(
                qubit_to_group=controller_layout.qubit_to_group,
                virtual_rz=exp_config.hardware.virtual_rz,
                topord_method=exp_config.serialization.topord_method,
                delay_check=exp_config.serialization.delay_check,
                operation_durations=target_full.durations(),
                switch_duration=exp_config.hardware.tsw,
            )
        ]
    )

    qc_trans_serial = pm_serial_trans.run(qc_trans)
    qc_routed_serial = pm_serial_routed.run(qc_routed)

    qc_trans_rho = compute_actual_densities(qc_trans)
    qc_trans_serial_rho = compute_actual_densities(qc_trans_serial)
    qc_routed_rho = compute_actual_densities(qc_routed)
    qc_routed_serial_rho = compute_actual_densities(qc_routed_serial)

    result = {
        "T_trans": qc_trans.estimate_duration(target_full),
        "T_routed": qc_routed.estimate_duration(target_full),
        "T_trans_serial": qc_trans_serial.estimate_duration(target_full),
        "T_routed_serial": qc_routed_serial.estimate_duration(target_full),
        #
        "qc_trans_num_qubits": qc_trans.num_qubits,
        "qc_trans_num_gates": qc_trans.size(),
        "qc_trans_depth": qc_trans.depth(),
        "qc_trans_avg_op": calculate_avg_operands(qc_trans),
        "qc_trans_n1": qc_trans.size(single_qubit_gate_filter),
        "qc_trans_n2": qc_trans.size(two_qubit_gate_filter),
        "qc_trans_rho_1": qc_trans_rho["rho_1"],
        "qc_trans_rho_2": qc_trans_rho["rho_2"],
        "qc_trans_rho_total": qc_trans_rho["rho_total"],
        #
        "qc_routed_num_qubits": qc_routed.num_qubits,
        "qc_routed_num_gates": qc_routed.size(),
        "qc_routed_depth": qc_routed.depth(),
        "qc_routed_avg_op": calculate_avg_operands(qc_routed),
        "qc_routed_n1": qc_routed.size(single_qubit_gate_filter),
        "qc_routed_n2": qc_routed.size(two_qubit_gate_filter),
        "qc_routed_rho_1": qc_routed_rho["rho_1"],
        "qc_routed_rho_2": qc_routed_rho["rho_2"],
        "qc_routed_rho_total": qc_routed_rho["rho_total"],
        #
        "qc_trans_serial_num_qubits": qc_trans_serial.num_qubits,
        "qc_trans_serial_num_gates": qc_trans_serial.size(),
        "qc_trans_serial_depth": qc_trans_serial.depth(),
        "qc_trans_serial_avg_op": calculate_avg_operands(qc_trans_serial),
        "qc_trans_serial_n1": qc_trans_serial.size(single_qubit_gate_filter),
        "qc_trans_serial_n2": qc_trans_serial.size(two_qubit_gate_filter),
        "qc_trans_serial_rho_1": qc_trans_serial_rho["rho_1"],
        "qc_trans_serial_rho_2": qc_trans_serial_rho["rho_2"],
        "qc_trans_serial_rho_total": qc_trans_serial_rho["rho_total"],
        #
        "qc_routed_serial_num_qubits": qc_routed_serial.num_qubits,
        "qc_routed_serial_num_gates": qc_routed_serial.size(),
        "qc_routed_serial_depth": qc_routed_serial.depth(),
        "qc_routed_serial_avg_op": calculate_avg_operands(qc_routed_serial),
        "qc_routed_serial_n1": qc_routed_serial.size(single_qubit_gate_filter),
        "qc_routed_serial_n2": qc_routed_serial.size(two_qubit_gate_filter),
        "qc_routed_serial_rho_1": qc_routed_serial_rho["rho_1"],
        "qc_routed_serial_rho_2": qc_routed_serial_rho["rho_2"],
        "qc_routed_serial_rho_total": qc_routed_serial_rho["rho_total"],
        #
        "T_trans_serial_pass": pm_serial_trans.property_set.get("serial_max_duration"),
        "T_routed_serial_pass": pm_serial_routed.property_set.get("serial_max_duration"),
        #
        "trans_switch2_gates": pm_serial_trans.property_set.get("serial_switch2_gates", 0),
        "trans_delay_gates": pm_serial_trans.property_set.get("serial_switch_del_gates", 0),
        #
        "routed_switch2_gates": pm_serial_routed.property_set.get("serial_switch2_gates", 0),
        "routed_delay_gates": pm_serial_routed.property_set.get("serial_switch_del_gates", 0),
        #
        "time": time.strftime("%Y%m%d-%H%M%S"),
        #
    }

    result.update(flatten_config(exp_config))
    return result

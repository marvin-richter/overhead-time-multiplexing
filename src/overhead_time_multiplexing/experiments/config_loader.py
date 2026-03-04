"""
Config loader: YAML -> list[ExperimentConfig]
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Iterator, List, Literal
import numpy as np

import yaml
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2

from overhead_time_multiplexing.layouts import get_balanced_group_sizes

from overhead_time_multiplexing.experiments.models import (
    ExperimentConfig,
    HardwareConfig,
    LayoutConfig,
    CircuitConfig,
    SerializationConfig,
    CircuitSource,
)


def load_experiments(config_path: Path) -> list[ExperimentConfig]:
    """
    Load YAML config and expand into ExperimentConfig instances.

    This is the main entry point. Add new sources by registering
    them in SOURCE_EXPANDERS.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    source = raw["source"]
    expander = SOURCE_EXPANDERS.get(source)
    if not expander:
        raise ValueError(f"Unknown source: {source}. Available: {list(SOURCE_EXPANDERS.keys())}")

    return list(expander(raw))


# =============================================================================
# Hardware expansion
# =============================================================================


def expand_hardware_configs(raw: dict[str, Any]) -> Iterator[HardwareConfig]:
    """
    Expand hardware section into HardwareConfig instances.

    Handles the Cartesian product of:
    - hardware entries (kind, id)
    - gate durations (t1, t2, tsw)
    - layouts
    - k values
    """
    durations = raw.get("generic_gate_durations", {})
    t1_list = durations.get("single_qubit_gate", [20e-9])
    t2_list = durations.get("two_qubit_gate", [200e-9])
    tsw_list = durations.get("switch_gate", [2e-9])

    layouts = raw.get("layouts", ["trivial"])
    virtual_rz = raw.get("virtual_rz", True)

    layout_seed_value = raw.get("layout_seed", "fix")
    num_layout_seeds = raw.get("num_layout_seeds", 1)

    if layout_seed_value == "fix":
        layout_seeds = list(range(num_layout_seeds))
    elif layout_seed_value == "random":
        seed = int(time.time())
        layout_seeds = list(
            np.random.default_rng(seed=seed).integers(0, 1_000_000, size=num_layout_seeds)
        )
    elif isinstance(layout_seed_value, list):
        layout_seeds = layout_seed_value
    else:
        raise ValueError(
            f"Unknown layout_seed: {layout_seed_value}. Use 'fix', 'random', or a list of integers."
        )

    for hw_entry in raw["hardware"]:
        kind = hw_entry["kind"]
        hw_id = hw_entry["id"]

        # Parse grid dimensions
        num_rows, num_cols, num_qubits = _parse_hardware_dimensions(kind, hw_id)

        # Expand k values
        k_list = _expand_k_values(raw.get("k", [2]), num_qubits)

        for t1 in t1_list:
            for t2 in t2_list:
                for tsw in tsw_list:
                    for layout_strategy in layouts:
                        for k in k_list:
                            if k > num_qubits:
                                continue
                            for layout_seed in layout_seeds:
                                yield HardwareConfig(
                                    num_rows=num_rows,
                                    num_cols=num_cols,
                                    num_qubits=num_qubits,
                                    kind=kind,
                                    id=hw_id,
                                    t1=t1,
                                    t2=t2,
                                    tsw=tsw,
                                    virtual_rz=virtual_rz,
                                    layout=LayoutConfig(
                                        strategy=layout_strategy,
                                        k=k,
                                        seed=layout_seed,
                                    ),
                                )


def _parse_hardware_dimensions(kind: str, hw_id: str) -> tuple[int, int, int]:
    """Parse hardware dimensions from kind and id."""
    if kind == "chalmers_grid":
        num_rows, num_cols = map(int, hw_id.split("x"))
        num_qubits = num_rows * num_cols
        return num_rows, num_cols, num_qubits
    elif kind == "ibm":
        provider = FakeProviderForBackendV2()

        try:
            if hw_id.startswith("fake_"):
                backend = provider.backend(hw_id)
            else:
                backend = provider.backend("fake_" + hw_id)
        except Exception as e:
            raise ValueError(f"IBM backend '{hw_id}' not found in FakeProvider") from e
        return None, None, backend.num_qubits
    else:
        raise ValueError(f"Unknown hardware kind: {kind}")


def _expand_k_values(k_spec: int | str | list, num_qubits: int) -> list[int]:
    """Expand k specification into concrete list of k values."""
    if isinstance(k_spec, int):
        return [k_spec]
    elif isinstance(k_spec, list):
        return k_spec
    elif k_spec == "all":
        return list(range(1, num_qubits))
    elif k_spec == "balanced":
        return list(
            {
                tuple(get_balanced_group_sizes(num_qubits=num_qubits, qpg=k)): k
                for k in range(num_qubits, 0, -1)
            }.values()
        )
    elif k_spec == "max":
        return [num_qubits]
    else:
        raise ValueError(f"Unknown k specification: {k_spec}")


def _parse_num_gates(spec: int | list | dict | None) -> list[int]:
    """
    Parse num_gates specification into list.

    Accepts:
        25000           -> [25000]
        [100, 500]      -> [100, 500]
        {start: 100, stop: 500}          -> [100, 200, 300, 400]  (step=100 default)
        {start: 100, stop: 500, step: 50} -> [100, 150, 200, ...]
    """
    if spec is None:
        return [100]  # default
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, list):
        return spec
    if isinstance(spec, dict):
        start = spec["start"]
        stop = spec["stop"]
        step = spec.get("step", 100)  # sensible default
        return list(range(start, stop, step))
    raise ValueError(f"Invalid num_gates: {spec}")


def _compute_scaled_depth(spec: str, num_qubits: int) -> int:
    """
    Compute depth from a scaling formula.

    Supported formulas:
        "n"         -> num_qubits
        "n^2"       -> num_qubits^2
        "n^3"       -> num_qubits^3
        "n*log(n)"  -> num_qubits * log2(num_qubits)
        "2^n"       -> 2^num_qubits
        "c*n"       -> c * num_qubits (e.g., "3*n", "0.5*n")
        "c*n^2"     -> c * num_qubits^2
    """
    import math
    import re

    n = num_qubits
    spec = spec.strip().replace(" ", "")

    # Try to match patterns with coefficient: c*n, c*n^2, c*n^3, c*n*log(n)
    coeff_pattern = re.match(r"^([\d.]+)\*(.+)$", spec)
    if coeff_pattern:
        coeff = float(coeff_pattern.group(1))
        base_spec = coeff_pattern.group(2)
    else:
        coeff = 1.0
        base_spec = spec

    # Evaluate base formula
    if base_spec == "n":
        value = n
    elif base_spec == "n^2":
        value = n**2
    elif base_spec == "n^3":
        value = n**3
    elif base_spec in ("n*log(n)", "nlog(n)", "n*logn", "nlogn"):
        value = n * math.log2(n) if n > 1 else n
    elif base_spec == "2^n":
        value = 2**n
    elif base_spec == "sqrt(n)":
        value = math.sqrt(n)
    else:
        raise ValueError(
            f"Unknown depth scaling formula: '{spec}'. "
            f"Supported: 'n', 'n^2', 'n^3', 'n*log(n)', '2^n', 'sqrt(n)', "
            f"or with coefficient like '3*n', '0.5*n^2'"
        )

    return max(1, int(coeff * value))


def _parse_depth_spec(spec: int | str | list | None) -> list[int | str]:
    """
    Parse depth specification into a list of fixed depths or scaling formulas.

    Accepts:
        100             -> [100]           (fixed depth)
        [30, 50, 100]   -> [30, 50, 100]   (list of fixed depths)
        "n"             -> ["n"]           (scaling formula)
        "n^2"           -> ["n^2"]         (scaling formula)
        [30, "n", "n^2"] -> [30, "n", "n^2"]  (mixed)

    Scaling formulas are evaluated later when num_qubits is known.
    """
    if spec is None:
        return [100]  # default
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, str):
        return [spec]
    if isinstance(spec, list):
        return spec
    raise ValueError(f"Invalid depth specification: {spec}")


def _resolve_depth(depth_spec: int | str, num_qubits: int) -> int:
    """
    Resolve a single depth specification to a concrete integer.

    If depth_spec is an int, return it directly.
    If depth_spec is a string, compute the scaling formula.
    """
    if isinstance(depth_spec, int):
        return depth_spec
    elif isinstance(depth_spec, str):
        return _compute_scaled_depth(depth_spec, num_qubits)
    else:
        raise ValueError(f"Invalid depth spec type: {type(depth_spec)}")


# =============================================================================
# Source-specific expanders
# =============================================================================


def expand_random_experiments(raw: dict[str, Any]) -> Iterator[ExperimentConfig]:
    """Expand config for source='random'."""
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    counter = 0

    circuits_cfg = raw.get("circuits", {})
    num_circuits = circuits_cfg.get("num_circuits", 1)
    num_gates_list = _parse_num_gates(circuits_cfg.get("num_gates"))

    weight_1q_list = raw.get("random_weight_1q", [0.7])
    if isinstance(weight_1q_list, (int, float)):
        weight_1q_list = [weight_1q_list]

    optimization_level = raw.get("optimization_level", 2)
    num_serializations = raw.get("num_serializations", 1)
    needs_translation = raw.get("needs_translation", False)
    needs_routing = raw.get("needs_routing", True)

    output_dir = Path(raw.get("output_dir", "results"))

    delay_checks = raw.get("delay_check", [True])
    topord_methods = raw.get("topord_method", ["prio_two"])

    for hw in expand_hardware_configs(raw):
        for num_gates in num_gates_list:
            for weight_1q in weight_1q_list:
                for circuit_id in range(num_circuits):
                    for serial_seed in range(num_serializations):
                        for delay_check in delay_checks:
                            for topord_method in topord_methods:
                                yield ExperimentConfig(
                                    exp_name=raw.get("experiment_name", "unnamed"),
                                    exp_id=f"{run_timestamp}{counter:06d}",
                                    hardware=hw,
                                    circuit=CircuitConfig(
                                        source=CircuitSource.RANDOM,
                                        id=f"random_{circuit_id}",
                                        num_gates=num_gates,
                                        num_qubits=hw.num_qubits,
                                        random_weight_1q=weight_1q,
                                        seed=circuit_id,
                                        optimization_level=optimization_level,
                                        needs_translation=needs_translation,
                                        needs_routing=needs_routing,
                                    ),
                                    serialization=SerializationConfig(
                                        delay_check=delay_check,
                                        topord_method=topord_method,
                                        seed=serial_seed,
                                    ),
                                    path_output=output_dir,
                                )
                                counter += 1


def expand_random_densities_experiments(raw: dict[str, Any]) -> Iterator[ExperimentConfig]:
    """Expand config for source='random_densities'."""
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    counter = 0

    circuits_cfg = raw.get("circuits", {})
    rho_1_list = circuits_cfg.get("rho_1", [0.5])
    rho_tot_list = circuits_cfg.get("rho_tot", [0.5])
    depth_specs = _parse_depth_spec(circuits_cfg.get("depths", [100]))
    num_circuits = circuits_cfg.get("num_circuits", 1)
    connectivity = circuits_cfg.get("connectivity", "native")

    optimization_level = raw.get("optimization_level", 2)
    num_serializations = raw.get("num_serializations", 1)
    needs_translation = raw.get("needs_translation", False)
    needs_routing = raw.get("needs_routing", True)

    output_dir = Path(raw.get("output_dir", "results"))

    delay_checks = raw.get("delay_check", [True])
    topord_methods = raw.get("topord_method", ["prio_two"])

    for hw in expand_hardware_configs(raw):
        for rho_1 in rho_1_list:
            for rho_tot in rho_tot_list:
                if rho_1 > rho_tot:
                    continue
                for depth_spec in depth_specs:
                    # Resolve depth: either fixed int or scaling formula
                    depth = _resolve_depth(depth_spec, hw.num_qubits)
                    # Use depth_spec in ID to distinguish scaling vs fixed
                    depth_label = depth_spec if isinstance(depth_spec, str) else depth

                    for circuit_id in range(num_circuits):
                        for serial_seed in range(num_serializations):
                            for delay_check in delay_checks:
                                for topord_method in topord_methods:
                                    circ_id = f"density_r1{rho_1}_rt{rho_tot}_d{depth_label}_{circuit_id}"

                                    yield ExperimentConfig(
                                        exp_name=raw.get("experiment_name", "unnamed"),
                                        exp_id=f"{run_timestamp}{counter:06d}",
                                        hardware=hw,
                                        circuit=CircuitConfig(
                                            source=CircuitSource.RANDOM_DENSITIES,
                                            id=circ_id,
                                            num_gates=None,
                                            num_qubits=hw.num_qubits,
                                            rho_1=rho_1,
                                            rho_tot=rho_tot,
                                            connectivity=connectivity,
                                            depth=depth,
                                            seed=circuit_id,
                                            optimization_level=optimization_level,
                                            needs_translation=needs_translation,
                                            needs_routing=needs_routing,
                                        ),
                                        serialization=SerializationConfig(
                                            delay_check=delay_check,
                                            topord_method=topord_method,
                                            seed=serial_seed,
                                        ),
                                        path_output=output_dir,
                                    )
                                    counter += 1


def expand_mqt_experiments(raw: dict[str, Any]) -> Iterator[ExperimentConfig]:
    """Expand config for source='mqt'."""
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    counter = 0

    circuits_cfg = raw.get("circuits", {})
    circuit_names = circuits_cfg.get("names", [])
    num_transpiler_seeds = circuits_cfg.get("num_transpiler_seeds", 1)

    bench_num_qubits = raw.get("bench_num_qubits", "max")
    optimization_level = raw.get("optimization_level", 2)
    num_serializations = raw.get("num_serializations", 1)
    needs_translation = raw.get("needs_translation", True)
    needs_routing = raw.get("needs_routing", True)

    output_dir = Path(raw.get("output_dir", "results"))

    delay_checks = raw.get("delay_check", [True])
    topord_methods = raw.get("topord_method", ["prio_two"])

    for hw in expand_hardware_configs(raw):
        circuits = _generate_mqt_configurations(
            hw=hw,
            mqt_names=circuit_names,
            bench_num_qubits=bench_num_qubits,
        )

        for circuit_name, num_qubits in circuits:
            for transpiler_seed in range(num_transpiler_seeds):
                for serial_seed in range(num_serializations):
                    for delay_check in delay_checks:
                        for topord_method in topord_methods:
                            yield ExperimentConfig(
                                exp_name=raw.get("experiment_name", "unnamed"),
                                exp_id=f"{run_timestamp}{counter:06d}",
                                hardware=hw,
                                circuit=CircuitConfig(
                                    source=CircuitSource.MQT,
                                    id=circuit_name,
                                    num_gates=None,
                                    num_qubits=num_qubits,
                                    random_weight_1q=None,
                                    seed=transpiler_seed,
                                    optimization_level=optimization_level,
                                    needs_translation=needs_translation,
                                    needs_routing=needs_routing,
                                ),
                                serialization=SerializationConfig(
                                    delay_check=delay_check,
                                    topord_method=topord_method,
                                    seed=serial_seed,
                                ),
                                path_output=output_dir,
                            )
                            counter += 1


def expand_random_pattern_experiments(raw: dict[str, Any]) -> Iterator[ExperimentConfig]:
    """Expand config for source='random_pattern'."""
    run_timestamp = time.strftime("%Y%m%d-%H%M%S")
    counter = 0

    pattern = raw.get("random_pattern", ["1q", "1q", "1q", "2q"])

    circuits_cfg = raw.get("circuits", {})
    if isinstance(circuits_cfg, list):
        circuits_cfg = {k: v for d in circuits_cfg for k, v in d.items()}

    num_circuits = circuits_cfg.get("num_circuits", raw.get("random_num_circuits", 1))
    num_gates_list = _parse_num_gates(
        circuits_cfg.get("num_gates", raw.get("random_num_gates", [100]))
    )

    optimization_level = raw.get("optimization_level", 2)
    num_serializations = raw.get("num_serializations", 1)
    needs_translation = raw.get("needs_translation", False)
    needs_routing = raw.get("needs_routing", True)

    output_dir = Path(raw.get("output_dir", "results"))

    delay_checks = raw.get("delay_check", [True])
    topord_methods = raw.get("topord_method", ["prio_two"])

    for hw in expand_hardware_configs(raw):
        for num_gates in num_gates_list:
            for circuit_id in range(num_circuits):
                for serial_seed in range(num_serializations):
                    for delay_check in delay_checks:
                        for topord_method in topord_methods:
                            yield ExperimentConfig(
                                exp_name=raw.get("experiment_name", "unnamed"),
                                exp_id=f"{run_timestamp}{counter:06d}",
                                hardware=hw,
                                circuit=CircuitConfig(
                                    source=CircuitSource.RANDOM_PATTERN,
                                    id=f"pattern_{circuit_id}",
                                    num_gates=num_gates,
                                    num_qubits=hw.num_qubits,
                                    random_weight_1q=None,
                                    seed=circuit_id,
                                    optimization_level=optimization_level,
                                    needs_translation=needs_translation,
                                    needs_routing=needs_routing,
                                ),
                                serialization=SerializationConfig(
                                    delay_check=delay_check,
                                    topord_method=topord_method,
                                    seed=serial_seed,
                                ),
                                path_output=output_dir,
                                extra={"pattern": pattern},
                            )
                            counter += 1


# =============================================================================
# Source registry - add new sources here
# =============================================================================

SOURCE_EXPANDERS = {
    "random": expand_random_experiments,
    "random_densities": expand_random_densities_experiments,
    "random_pattern": expand_random_pattern_experiments,
    "mqt": expand_mqt_experiments,
}


# =============================================================================
# Helper functions for MQT expansion (e.g. parsing num_gates, depth specifications, k values)
# =============================================================================


def _generate_mqt_configurations(
    hw: HardwareConfig,
    mqt_names: List[str] = None,
    bench_num_qubits: Literal["all", "max"] = "max",
) -> List:
    """Generate configurations for MQT benchmarks."""

    def get_available_qubits_for_benchmarks(max_num_qubits=127):
        """
        Returns all available num_qubits for each benchmark between 2 and max_num_qubits.

        Args:
            max_num_qubits (int): Maximum number of qubits to consider (default: 127).

        Returns:
            dict: Dictionary mapping benchmark names to lists of available qubit counts.

        Benchmark-specific qubit constraints:
            shor: Fixed sizes [18, 42, 58] only. 74 is excluded due to OOM problems.
            half_adder: Odd integers >= 3.
            hrs_cumulative_multiplier: Integers >= 5 where (num_qubits - 1) % 4 == 0.
            rg_qft_multiplier: Multiples of 4, >= 4.
            vbe_ripple_carry_adder: Integers >= 4 where (num_qubits - 1) % 3 == 0.
            cdkm_ripple_carry_adder: Even integers >= 4.
            full_adder: Even integers >= 4.
            modular_adder: Even integers >= 2.
            draper_qft_adder: Even integers >= 2.
            bmw_quark_copula: Even integers >= 2.

        Excluded benchmarks:
            ae, grover: Take too long to load.
            multiplier: Disabled (would require multiples of 4, >= 4).
            qwalk: Disabled due to OOM issues beyond ~30 qubits.

        All other benchmarks accept any integer >= 2.
        """

        available_qubits = {}

        available_qubits["shor"] = [n for n in [18, 42, 58] if n <= max_num_qubits]

        available_qubits["half_adder"] = [q for q in range(3, max_num_qubits + 1, 2)]

        available_qubits["hrs_cumulative_multiplier"] = [
            q for q in range(5, max_num_qubits + 1) if (q - 1) % 4 == 0
        ]

        available_qubits["rg_qft_multiplier"] = [q for q in range(4, max_num_qubits + 1, 4)]

        available_qubits["vbe_ripple_carry_adder"] = [
            q for q in range(4, max_num_qubits + 1) if (q - 1) % 3 == 0
        ]

        available_qubits["cdkm_ripple_carry_adder"] = [q for q in range(4, max_num_qubits + 1, 2)]

        available_qubits["full_adder"] = [q for q in range(4, max_num_qubits + 1, 2)]

        available_qubits["modular_adder"] = [q for q in range(2, max_num_qubits + 1, 2)]

        available_qubits["draper_qft_adder"] = [q for q in range(2, max_num_qubits + 1, 2)]

        available_qubits["bmw_quark_copula"] = [q for q in range(2, max_num_qubits + 1, 2)]

        general_benchmarks = [
            "bmw_quark_cardinality",
            "bv",
            "dj",
            "ghz",
            "graphstate",
            "hhl",
            "qaoa",
            "qft",
            "qftentangled",
            "qnn",
            "qpeexact",
            "qpeinexact",
            "randomcircuit",
            "vqe_real_amp",
            "vqe_su2",
            "vqe_two_local",
            "wstate",
        ]

        for benchmark in general_benchmarks:
            available_qubits[benchmark] = list(range(2, max_num_qubits + 1))

        return available_qubits

    def get_largest_available_qubits(max_num_qubits=127):
        """
        Get the largest available qubit count for a specific benchmark.

        Args:
            max_num_qubits (int): Maximum number of qubits to consider (default: 100)

        Returns:
            dict: Dictionary mapping benchmark names to their largest available qubit count
        """
        largest_qubits = {}
        available_qubits = get_available_qubits_for_benchmarks(max_num_qubits)

        for benchmark, qubits in available_qubits.items():
            if qubits:
                largest_qubits[benchmark] = [max(qubits)]
            else:
                largest_qubits[benchmark] = []  # No available qubits for this benchmark

        return largest_qubits

    if bench_num_qubits == "max":
        bench_name_with_num_qubits = get_largest_available_qubits(hw.num_qubits)

    elif bench_num_qubits == "all":
        bench_name_with_num_qubits = get_available_qubits_for_benchmarks(hw.num_qubits)
    else:
        raise ValueError(f"Invalid mqt_sizes: {bench_num_qubits}. Must be 'all' or 'max'.")

    # filter
    if mqt_names:
        bench_name_with_num_qubits = {
            k: v for k, v in bench_name_with_num_qubits.items() if k in mqt_names
        }

    circuits = []
    for bench_name, num_qubits in bench_name_with_num_qubits.items():
        for num_qubit in num_qubits:
            circuits.append((bench_name, num_qubit))
    return circuits

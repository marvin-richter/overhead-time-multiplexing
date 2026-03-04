"""
Random circuit generation utilities.

This module provides functions for generating random quantum circuits
using native gate sets and various parameters.
"""

import math
import random
import numpy as np
from typing import Tuple, Set, Optional, List

import qiskit
from qiskit import QuantumCircuit


from ..targets import chalmers_native_gates


def _filter_gates_by_qubits(native_gates):
    """Filter and categorize gates by qubit count, excluding problematic gates."""
    # If only a list of native gates is provided, convert to tuples (operation, name)
    if not isinstance(native_gates[0], tuple):
        native_gates = [
            (gate, gate.name)
            for gate in native_gates
            if isinstance(gate, qiskit.circuit.operation.Operation)
        ]

    # Gates to exclude from single-qubit selection
    excluded_gates = {
        "i",
        "id",
        "delay",
        "measure",
        "reset",
        "barrier",
        "snapshot",
        "save",
        "save_statevector",
        "SDel",
        "sw2",
    }

    single_qubit_gates = []
    two_qubit_gates = []

    for operation, name in native_gates:
        if operation.name not in excluded_gates:
            if operation.num_qubits == 1:
                single_qubit_gates.append((operation, operation.name))
            elif operation.num_qubits == 2:
                two_qubit_gates.append((operation, operation.name))

    return single_qubit_gates, two_qubit_gates


def _create_parameterized_gate(gate, num_params):
    """Create a parameterized gate with random angles."""
    if num_params == 0:
        return gate

    angles = [random.uniform(0, 2 * math.pi) for _ in range(num_params)]

    # Instead of reconstructing, copy and update parameters
    new_gate = gate.copy()
    new_gate.params = angles
    return new_gate


def random_circuit_native_optimized(
    num_qubits: int,
    num_gates: int,
    seed: int = None,
    gate_weights: Tuple[float, float] = (0.7, 0.3),
    native_gates=chalmers_native_gates,
) -> qiskit.QuantumCircuit:
    """
    Optimized version of random_circuit_native for better performance with large gate counts.

    Key optimizations:
    - Batch random number generation
    - Minimize object creation in main loop
    - Pre-compute gate type choices
    """
    # Input validation
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    if num_gates < 0:
        num_gates = 0
    if num_qubits == 0 and num_gates > 0:
        raise ValueError("Cannot place gates on a circuit with 0 qubits")

    if seed is not None:
        random.seed(seed)

    # Create the circuit
    qc = qiskit.QuantumCircuit(num_qubits)

    # Early return for empty circuit
    if num_gates == 0:
        return qc

    # Filter and categorize gates (happens once)
    single_qubit_gates, two_qubit_gates = _filter_gates_by_qubits(native_gates)

    # Pre-compute valid edges if needed
    valid_edges = None
    if len(two_qubit_gates) > 0 and num_qubits > 1:
        valid_edges = [(i, j) for i in range(num_qubits) for j in range(num_qubits) if i != j]

    # Normalize gate weights
    single_weight, two_weight = gate_weights
    total_weight = single_weight + two_weight
    if total_weight <= 0:
        raise ValueError("Gate weights must be positive")

    # Determine gate type availability
    single_available = len(single_qubit_gates) > 0 and num_qubits > 0
    two_available = len(two_qubit_gates) > 0 and valid_edges and len(valid_edges) > 0

    if not (single_available or two_available):
        return qc

    # OPTIMIZATION 1: Pre-compute all gate type choices
    gate_types = []
    if single_available and two_available:
        normalized_single_weight = single_weight / total_weight
        # Generate all random choices at once (much faster than repeated random.choices)
        gate_types = [
            "single" if random.random() < normalized_single_weight else "two"
            for _ in range(num_gates)
        ]
    elif single_available:
        gate_types = ["single"] * num_gates
    else:
        gate_types = ["two"] * num_gates

    # OPTIMIZATION 2: Pre-compute all random indices
    single_gate_indices = [
        random.randrange(len(single_qubit_gates)) for _ in range(gate_types.count("single"))
    ]
    two_gate_indices = [
        random.randrange(len(two_qubit_gates)) for _ in range(gate_types.count("two"))
    ]

    single_qubit_choices = [
        random.randrange(num_qubits) for _ in range(gate_types.count("single"))
    ]
    two_qubit_choices = (
        [random.randrange(len(valid_edges)) for _ in range(gate_types.count("two"))]
        if valid_edges
        else []
    )

    # OPTIMIZATION 3: Use iterators to avoid index tracking
    single_gate_iter = iter(single_gate_indices)
    two_gate_iter = iter(two_gate_indices)
    single_qubit_iter = iter(single_qubit_choices)
    two_qubit_iter = iter(two_qubit_choices)

    # Main loop - now just applies pre-computed choices
    for gate_type in gate_types:
        if gate_type == "single":
            gate_idx = next(single_gate_iter)
            qubit = next(single_qubit_iter)
            gate = single_qubit_gates[gate_idx][0]
            qubits = [qubit]
        else:  # "two"
            gate_idx = next(two_gate_iter)
            edge_idx = next(two_qubit_iter)
            gate = two_qubit_gates[gate_idx][0]
            qubits = list(valid_edges[edge_idx])

        # Create and apply the gate
        if gate.params:
            op = _create_parameterized_gate(gate, len(gate.params))
        else:
            op = gate

        qc.append(op, qubits)

    return qc


def random_circuit_native(
    num_qubits: int,
    num_gates: int,
    seed: int = None,
    gate_weights: Tuple[float, float] = (0.7, 0.3),
    native_gates=chalmers_native_gates,
) -> qiskit.QuantumCircuit:
    """
    Generate a random quantum circuit using only Chalmers native gates.

    Args:
        num_qubits: Number of qubits in the circuit
        num_gates: Number of gates to add to the circuit
        seed: Random seed for reproducibility
        gate_weights: Tuple of (single_qubit_weight, two_qubit_weight) for gate type selection.
                     Default is (0.7, 0.3) for 70% single-qubit, 30% two-qubit gates.
                     Weights are automatically normalized.

    Returns:
        QuantumCircuit with random native gates

    Raises:
        ValueError: If num_qubits is negative or if attempting to place gates on 0 qubits
    """
    # Input validation
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")

    if num_gates < 0:
        num_gates = 0  # Treat negative gates as zero gates

    if num_qubits == 0 and num_gates > 0:
        raise ValueError("Cannot place gates on a circuit with 0 qubits")

    if seed is not None:
        random.seed(seed)

    # Create the circuit
    qc = qiskit.QuantumCircuit(num_qubits)

    # Early return for empty circuit
    if num_gates == 0:
        return qc

    # Filter and categorize gates
    single_qubit_gates, two_qubit_gates = _filter_gates_by_qubits(native_gates)

    # All-to-all connectivity for two-qubit gates
    valid_edges = [(i, j) for i in range(num_qubits) for j in range(num_qubits) if i != j]

    # Normalize gate weights
    single_weight, two_weight = gate_weights
    total_weight = single_weight + two_weight
    if total_weight <= 0:
        raise ValueError("Gate weights must be positive")
    normalized_weights = [single_weight / total_weight, two_weight / total_weight]

    for _ in range(num_gates):
        # Determine available gate types
        single_available = len(single_qubit_gates) > 0 and num_qubits > 0
        two_available = len(two_qubit_gates) > 0 and len(valid_edges) > 0

        # Choose gate type based on availability and weights
        if single_available and two_available:
            gate_choice = random.choices(["single", "two"], weights=normalized_weights)[0]
        elif single_available:
            gate_choice = "single"
        elif two_available:
            gate_choice = "two"
        else:
            # No valid gates available
            break

        # Select gate and qubits
        if gate_choice == "single":
            gate, _ = random.choice(single_qubit_gates)
            qubits = [random.randint(0, num_qubits - 1)]
        else:  # gate_choice == "two"
            gate, _ = random.choice(two_qubit_gates)
            qubits = list(random.choice(valid_edges))

        # Create and apply the gate (parameterized or not)
        op = _create_parameterized_gate(gate, len(gate.params))
        qc.append(op, qubits)

    return qc


def random_circuit_maximally_dense(
    num_qubits: int,
    num_layers: int,
    seed: int = None,
    gate_weights: Tuple[float, float] = (0.3, 0.7),  # (single, two) - favor 2q for density
    native_gates=chalmers_native_gates,
    connectivity="full",
    add_barriers: bool = True,
) -> qiskit.QuantumCircuit:
    """
    Create a maximally dense random circuit by filling each layer with
    the maximum number of non-conflicting gates.

    Args:
        num_qubits: Number of qubits
        num_layers: Number of parallel layers (circuit depth)
        gate_weights: (single_qubit_weight, two_qubit_weight)
        native_gates: List of available gates
        connectivity: "full", "linear", or "grid"
        add_barriers: Whether to add barriers between layers
    """
    # Input validation
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    if num_layers < 0:
        raise ValueError("num_layers must be non-negative")
    if num_qubits == 0:
        return qiskit.QuantumCircuit(0)

    if seed is not None:
        random.seed(seed)

    qc = qiskit.QuantumCircuit(num_qubits)

    if num_layers == 0:
        return qc

    # Handle default gates if not provided
    if native_gates is None:
        raise ValueError("native_gates must be provided")

    # Filter and categorize gates
    single_qubit_gates, two_qubit_gates = _filter_gates_by_qubits(native_gates)

    if not single_qubit_gates and not two_qubit_gates:
        return qc

    # Define connectivity
    valid_edges = _get_connectivity_edges(num_qubits, connectivity)

    # Normalize weights
    single_weight, two_weight = gate_weights
    if single_weight < 0 or two_weight < 0:
        raise ValueError("Gate weights must be non-negative")

    total_weight = single_weight + two_weight
    if total_weight == 0:
        raise ValueError("At least one gate weight must be positive")

    two_qubit_prob = two_weight / total_weight

    # Generate each layer with maximum density
    for layer in range(num_layers):
        used_qubits: Set[int] = set()

        # Strategy 1: Place two-qubit gates first (they constrain more qubits)
        if two_qubit_gates and valid_edges and num_qubits >= 2:
            available_edges = [
                edge
                for edge in valid_edges
                if edge[0] not in used_qubits and edge[1] not in used_qubits
            ]

            # Greedy placement of two-qubit gates
            while available_edges and random.random() < two_qubit_prob:
                edge = random.choice(available_edges)
                gate_info = random.choice(two_qubit_gates)
                gate = gate_info[0]

                # Create parameterized gate if needed
                if hasattr(gate, "params") and gate.params:
                    op = _create_parameterized_gate(gate, len(gate.params))
                else:
                    op = gate

                qc.append(op, [edge[0], edge[1]])
                used_qubits.update(edge)

                # Update available edges
                available_edges = [
                    e
                    for e in available_edges
                    if e[0] not in used_qubits and e[1] not in used_qubits
                ]

        # Strategy 2: Fill all remaining qubits with single-qubit gates
        if single_qubit_gates:
            unused_qubits = [q for q in range(num_qubits) if q not in used_qubits]

            for qubit in unused_qubits:
                gate_info = random.choice(single_qubit_gates)
                gate = gate_info[0]

                # Create parameterized gate if needed
                if hasattr(gate, "params") and gate.params:
                    op = _create_parameterized_gate(gate, len(gate.params))
                else:
                    op = gate

                qc.append(op, [qubit])

        # Add barrier between layers (except after last layer)
        if add_barriers and layer < num_layers - 1:
            qc.barrier()

    return qc


def _get_connectivity_edges(num_qubits: int, connectivity: str) -> list[Tuple[int, int]]:
    """Generate valid edges based on connectivity pattern."""
    if connectivity == "full":
        return [(i, j) for i in range(num_qubits) for j in range(num_qubits) if i != j]

    elif connectivity == "linear":
        edges = []
        for i in range(num_qubits - 1):
            edges.extend([(i, i + 1), (i + 1, i)])
        return edges

    elif connectivity == "grid":
        if num_qubits < 4:
            # Fall back to linear for small systems
            return _get_connectivity_edges(num_qubits, "linear")

        # Find best square grid that fits
        side = int(num_qubits**0.5)
        while side * side > num_qubits:
            side -= 1

        edges = []
        for i in range(side):
            for j in range(side):
                qubit = i * side + j
                if qubit >= num_qubits:
                    break

                # Right neighbor
                if j < side - 1 and qubit + 1 < num_qubits:
                    edges.extend([(qubit, qubit + 1), (qubit + 1, qubit)])

                # Bottom neighbor
                if i < side - 1 and qubit + side < num_qubits:
                    edges.extend([(qubit, qubit + side), (qubit + side, qubit)])

        return edges

    else:
        raise ValueError(f"Unknown connectivity pattern: {connectivity}")


def random_circuit_custom_pattern(
    num_qubits: int,
    pattern: list[str],
    reps: int,
    seed: int = None,
    native_gates=chalmers_native_gates,
    connectivity: str = "brick",
    add_barriers: bool = True,
) -> qiskit.QuantumCircuit:
    """
    Create a circuit with custom layer pattern that repeats.

    Args:
        num_qubits: Number of qubits
        pattern: List of layer types, e.g., ["1q", "1q", "1q", "2q"]
                 "1q" = single-qubit layer (all qubits)
                 "2q" = two-qubit layer (maximally dense)
        reps: Number of times to repeat the pattern
        seed: Random seed
        native_gates: Available gates
        connectivity: Connectivity for two-qubit gates
        add_barriers: Add barriers between layers

    Example:
        # 3 single-qubit + 1 two-qubit pattern
        pattern = ["1q", "1q", "1q", "2q"]

        # Alternating pattern
        pattern = ["1q", "2q", "1q", "2q"]
    """
    # Input validation
    if num_qubits < 0:
        raise ValueError("num_qubits must be non-negative")
    if reps < 0:
        raise ValueError("reps must be non-negative")
    if not pattern:
        raise ValueError("Pattern cannot be empty")
    if num_qubits == 0:
        return qiskit.QuantumCircuit(0)

    valid_layer_types = {"1q", "2q"}
    for layer_type in pattern:
        if layer_type not in valid_layer_types:
            raise ValueError(f"Invalid layer type '{layer_type}'. Must be '1q' or '2q'")

    if seed is not None:
        random.seed(seed)

    qc = qiskit.QuantumCircuit(num_qubits)

    if reps == 0:
        return qc

    if native_gates is None:
        raise ValueError("native_gates must be provided")

    # Filter and categorize gates
    single_qubit_gates, two_qubit_gates = _filter_gates_by_qubits(native_gates)

    total_layers = len(pattern) * reps
    current_layer = 0

    for rep in range(reps):
        for pattern_idx, layer_type in enumerate(pattern):
            current_layer += 1

            if layer_type == "1q":
                # Single-qubit layer: all qubits get a gate
                for qubit in range(num_qubits):
                    if single_qubit_gates:
                        gate_info = random.choice(single_qubit_gates)
                        gate = gate_info[0]

                        if hasattr(gate, "params") and gate.params:
                            op = _create_parameterized_gate(gate, len(gate.params))
                        else:
                            op = gate

                        qc.append(op, [qubit])

            elif layer_type == "2q":
                # Two-qubit layer: maximally dense placement
                if num_qubits >= 2 and two_qubit_gates:
                    used_qubits: Set[int] = set()

                    # Use brick pattern alternating by rep
                    if rep % 2 == 0:
                        pairs = [(i, i + 1) for i in range(0, num_qubits - 1, 2)]
                    else:
                        pairs = [
                            (i + 1, (i + 2) % num_qubits) for i in range(0, num_qubits - 1, 2)
                        ]

                    # Place two-qubit gates
                    for pair in pairs:
                        gate_info = random.choice(two_qubit_gates)
                        gate = gate_info[0]

                        if hasattr(gate, "params") and gate.params:
                            op = _create_parameterized_gate(gate, len(gate.params))
                        else:
                            op = gate

                        qc.append(op, [pair[0], pair[1]])
                        used_qubits.update(pair)

                    # Fill unused qubits with single-qubit gates
                    unused_qubits = [q for q in range(num_qubits) if q not in used_qubits]
                    for qubit in unused_qubits:
                        if single_qubit_gates:
                            gate_info = random.choice(single_qubit_gates)
                            gate = gate_info[0]

                            if hasattr(gate, "params") and gate.params:
                                op = _create_parameterized_gate(gate, len(gate.params))
                            else:
                                op = gate

                            qc.append(op, [qubit])

            # Add barrier (except after last layer)
            if add_barriers and current_layer < total_layers:
                qc.barrier()

    return qc


def _generate_square_grid_edges(num_qubits: int) -> List[Tuple[int, int]]:
    """
    Generate nearest-neighbor edges for a square grid.

    Assumes qubits are arranged in a square (or near-square) grid
    with row-major indexing.
    """
    # Determine grid dimensions
    side = int(math.ceil(math.sqrt(num_qubits)))

    edges = []
    for i in range(num_qubits):
        _, col = i // side, i % side

        # Right neighbor
        if col + 1 < side:
            j = i + 1
            if j < num_qubits:
                edges.append((i, j))

        # Down neighbor
        j = i + side
        if j < num_qubits:
            edges.append((i, j))

    return edges


def random_circuit_fixed_density(
    num_qubits: int,
    depth: int,
    rho_1: float,
    rho_2: float,
    seed: Optional[int] = None,
    native_gates=chalmers_native_gates,
    coupling_map: Optional[List[Tuple[int, int]]] = None,
) -> QuantumCircuit:
    """
    Generate a random circuit with fixed depth and specified gate densities.

    The circuit is built layer by layer to ensure controlled depth.
    Gate densities follow the definitions from the manuscript:
        rho_1 = N_1 / (n * D)       -- single-qubit gate density
        rho_2 = 2 * N_2 / (n * D)   -- two-qubit gate density

    Parameters
    ----------
    num_qubits : int
        Number of qubits in the circuit.
    depth : int
        Exact circuit depth (number of layers).
    rho_1 : float
        Target single-qubit gate density, should be in [0, 1].
        Each qubit has probability rho_1 of receiving a 1q gate per layer.
    rho_2 : float
        Target two-qubit gate density, should be in [0, 1].
        The probability per edge is derived to achieve this density.
    seed : int, optional
        Random seed for reproducibility.
    native_gates : list
        List of native gates to use.
    coupling_map : list of tuples, optional
        List of allowed edges (i, j) for two-qubit gates.
        If None, uses nearest-neighbor square grid connectivity.

    Returns
    -------
    QuantumCircuit
        The generated random circuit with approximately the specified densities.

    Notes
    -----
    The actual densities may differ slightly from targets due to:
    - Discrete gate placement (can't place fractional gates)
    - Conflict resolution (two-qubit gates block their qubits)

    For the controlled experiment, this ensures that circuits on different
    grid sizes have the SAME density, isolating the effect of system size N.
    """
    # Input validation
    if num_qubits < 1:
        raise ValueError("num_qubits must be at least 1")
    if depth < 1:
        raise ValueError("depth must be at least 1")
    if not 0 <= rho_1 <= 1:
        raise ValueError("rho_1 must be in [0, 1]")
    if not 0 <= rho_2 <= 1:
        raise ValueError("rho_2 must be in [0, 1]")

    if seed is not None:
        random.seed(seed)

    # Create the circuit
    qc = QuantumCircuit(num_qubits)

    # Filter and categorize gates
    single_qubit_gates, two_qubit_gates = _filter_gates_by_qubits(native_gates)

    if not single_qubit_gates and not two_qubit_gates:
        return qc

    # Set up coupling map (default: square grid nearest-neighbor)
    if coupling_map is None and num_qubits > 1:
        coupling_map = _generate_square_grid_edges(num_qubits)
    elif num_qubits == 1:
        coupling_map = []

    # Convert to set for faster lookup, and make bidirectional
    valid_edges = set()
    for edge in coupling_map:
        valid_edges.add(tuple(edge))
        valid_edges.add((edge[1], edge[0]))  # Add reverse direction
    valid_edges_list = list(valid_edges)

    # Calculate probability per edge to achieve target rho_2
    # rho_2 = 2 * N_2 / (n * D)
    # N_2 = rho_2 * n * D / 2
    # If we place 2q gates with prob p_edge per edge per layer:
    # Expected N_2 = p_edge * |edges| * D (counting each undirected edge once)
    # So: p_edge = rho_2 * n / (2 * |undirected_edges|)
    n_undirected_edges = len(valid_edges) // 2  # Each edge appears twice
    if n_undirected_edges > 0 and two_qubit_gates:
        p_2q_per_edge = (rho_2 * num_qubits) / (2 * n_undirected_edges)
        p_2q_per_edge = min(p_2q_per_edge, 1.0)  # Cap at 1
    else:
        p_2q_per_edge = 0.0

    p_1q_adjusted = rho_1 / (1 - rho_2) if rho_2 < 1 else 0.0
    p_1q_adjusted = min(p_1q_adjusted, 1.0)  # Cap at 1

    rng = np.random.default_rng(seed)

    # Target gate counts (total across circuit)
    target_n1 = int(round(rho_1 * num_qubits * depth))
    target_n2 = int(round(rho_2 * num_qubits * depth / 2))

    # Distribute with guarantee: every layer gets at least 1 gate total
    n1_per_layer, n2_per_layer = _distribute_counts_joint(target_n1, target_n2, depth, rng)

    for layer in range(depth):
        used_qubits: Set[int] = set()
        gates_placed_this_layer = 0

        # Place two-qubit gates
        if two_qubit_gates and valid_edges_list and n2_per_layer[layer] > 0:
            placed = 0
            placed_edges: Set[Tuple[int, int]] = set()
            n2_to_place = n2_per_layer[layer]

            def try_place_edge(edge):
                """Returns True if gate was placed, False if edge was invalid/conflicting."""
                if edge[0] in used_qubits or edge[1] in used_qubits:
                    return False
                if (edge[1], edge[0]) in placed_edges:
                    return False

                gate = two_qubit_gates[rng.integers(len(two_qubit_gates))][0]
                op = _create_parameterized_gate(gate, len(gate.params)) if gate.params else gate
                qc.append(op, list(edge))

                used_qubits.add(edge[0])
                used_qubits.add(edge[1])
                placed_edges.add(edge)
                return True

            # Sample edges efficiently instead of shuffling entire list
            # Oversample by 5x to account for conflicts (still O(1) not O(n²))
            edges_needed = min(n2_to_place * 5, len(valid_edges_list))
            candidate_indices = rng.choice(len(valid_edges_list), size=edges_needed, replace=False)
            sampled_indices = set(candidate_indices)

            for idx in candidate_indices:
                if placed >= n2_to_place:
                    break
                if try_place_edge(valid_edges_list[idx]):
                    placed += 1
                    gates_placed_this_layer += 1

            # Fallback: if we didn't place enough gates, try remaining edges
            if placed < n2_to_place:
                remaining_indices = set(range(len(valid_edges_list))) - sampled_indices
                for idx in remaining_indices:
                    if placed >= n2_to_place:
                        break
                    if try_place_edge(valid_edges_list[idx]):
                        placed += 1
                        gates_placed_this_layer += 1

        # Place single-qubit gates
        if single_qubit_gates:
            available_qubits = [q for q in range(num_qubits) if q not in used_qubits]
            n1_this_layer = min(n1_per_layer[layer], len(available_qubits))

            # Ensure at least 1 gate per layer if nothing placed yet
            if gates_placed_this_layer == 0 and n1_this_layer == 0 and available_qubits:
                n1_this_layer = 1  # Force one gate to maintain depth

            if n1_this_layer > 0:
                selected = rng.choice(available_qubits, size=n1_this_layer, replace=False)
                for qubit in selected:
                    gate = single_qubit_gates[rng.integers(len(single_qubit_gates))][0]
                    op = (
                        _create_parameterized_gate(gate, len(gate.params)) if gate.params else gate
                    )
                    qc.append(op, [int(qubit)])

    return qc


def _distribute_counts_joint(
    n1_total: int, n2_total: int, bins: int, rng
) -> Tuple[List[int], List[int]]:
    """
    Distribute 1q and 2q gate counts across layers, ensuring no layer is empty.
    """
    n1_counts = [0] * bins
    n2_counts = [0] * bins

    # First, ensure every layer has at least one gate
    total_gates = n1_total + n2_total

    if total_gates >= bins:
        # Assign 1 gate to each layer first (prefer 1q gates as they're always placeable)
        guaranteed = min(n1_total, bins)
        for i in range(guaranteed):
            n1_counts[i] = 1
        remaining_n1 = n1_total - guaranteed

        # If not enough 1q gates, use 2q gates for remaining layers
        if guaranteed < bins:
            for i in range(guaranteed, bins):
                n2_counts[i] = 1
            remaining_n2 = n2_total - (bins - guaranteed)
        else:
            remaining_n2 = n2_total
    else:
        # Not enough gates to fill all layers - distribute what we have
        remaining_n1 = n1_total
        remaining_n2 = n2_total

    # Distribute remaining gates randomly
    if remaining_n1 > 0:
        extra_n1 = _distribute_counts_simple(remaining_n1, bins, rng)
        n1_counts = [a + b for a, b in zip(n1_counts, extra_n1)]

    if remaining_n2 > 0:
        extra_n2 = _distribute_counts_simple(remaining_n2, bins, rng)
        n2_counts = [a + b for a, b in zip(n2_counts, extra_n2)]

    return n1_counts, n2_counts


def _distribute_counts_simple(total: int, bins: int, rng) -> List[int]:
    """Distribute count across bins randomly."""
    counts = [0] * bins
    base = total // bins
    remainder = total % bins

    for i in range(bins):
        counts[i] = base

    if remainder > 0:
        for i in rng.choice(bins, size=remainder, replace=False):
            counts[i] += 1

    return counts

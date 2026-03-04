"""This module initializes the layouts. It should be run initially to set up layouts.
It creates four layouts for each coupling map and qpg (qubits per group): "trivial",
"k-distance", "clustered", and "random".

The trivial layout is a simple layout, where the groups are filled in the order of the qubits.

The k-distance layout is a layout where the groups are filled in a way that the distance between
the qubits in each group is at least k. The distance is defined as the number of edges in the coupling map between two qubits.

The clustered layout is a layout where the groups are filled in a way that the qubits in each group are as close as possible to each other.

The random layout is a layout where the qubits are randomly assigned to groups with balanced group sizes.

The layouts are stored in layouts/ as jsons and will be accessed by the ControlllerAwareTarget.
"""

from collections import defaultdict
from dataclasses import asdict, dataclass, field
from functools import cached_property
import gzip
import json
import logging
from pathlib import Path
import random
from typing import Literal, Optional, TypeAlias, get_args

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
from qiskit.transpiler import Target

from .algorithms import (
    cluster_qubits_bfs,
)
from .k_distance import exact_k_distance_coloring, heuristic_distance_coloring

ROOT = Path(__file__).resolve().parent.parent.parent.parent
LAYOUTS_DIR = ROOT / "layouts"

LayoutType: TypeAlias = Literal["trivial", "dispersed", "clustered", "random"]

logger = logging.getLogger(__name__)


@dataclass
class ControllerLayout:
    """Class representing a controller layout for a specific hardware type."""

    hardware_label: str
    layout_type: LayoutType
    qpg: int
    group_qubits: dict[int, list[int]]
    method: str

    # Computed basic properties (set in __post_init__)
    num_qubits: int = field(init=False)
    num_groups: int = field(init=False)
    largest_group_size: int = field(init=False)
    group_sizes: dict[int, int] = field(init=False)

    # Metrics (calculated with graph)
    min_intra_dist: Optional[int] = field(default=None, init=False)
    avg_min_intra_dist: Optional[float] = field(default=None, init=False)
    avg_intra_dist: Optional[float] = field(default=None, init=False)
    min_compactness: float = field(default=None, init=False)
    avg_compactness: float = field(default=None, init=False)
    num_fully_connected_groups: int = field(default=None, init=False)

    def __post_init__(self):
        # make sure that keys and values are regular Python ints
        self.group_qubits = {
            int(k): [int(qubit) for qubit in v] for k, v in self.group_qubits.items()
        }

        # Calculate basic properties
        self.num_qubits = sum(len(v) for v in self.group_qubits.values())
        self.num_groups = len(self.group_qubits)
        self.largest_group_size = max(len(group) for group in self.group_qubits.values())
        self.group_sizes = {
            group_id: len(qubits) for group_id, qubits in self.group_qubits.items()
        }

        # Validate that every group has less than or equal to qpg qubits
        self.validate_qpg(self.qpg)

    @classmethod
    def create_with_graph(
        cls,
        hardware_label: str,
        layout_type: LayoutType,
        qpg: int,
        group_qubits: dict[int, list[int]],
        method: str,
        graph: nx.Graph,
    ):
        """Create instance and store all metrics that require the graph."""

        instance = cls(
            hardware_label=hardware_label,
            layout_type=layout_type,
            qpg=qpg,
            group_qubits=group_qubits,
            method=method,
        )
        instance.calculate_metrics(graph)
        return instance

    def calculate_metrics(self, graph: nx.Graph) -> None:
        # Calculate the distance matrix
        distance_matrix = nx.floyd_warshall_numpy(graph).astype(int)

        # Calculate distance metrics
        all_min_distances = self.all_minimum_distance_in_groups(distance_matrix)
        self.min_intra_dist = int(min(all_min_distances.values()))
        self.avg_min_intra_dist = float(sum(all_min_distances.values()) / len(all_min_distances))
        self.avg_intra_dist = self.average_distance_in_groups(distance_matrix)

        # Calculate compactness metrics
        all_compactness = self.all_groups_compactness(graph)
        self.avg_compactness = (
            float(sum(all_compactness.values()) / len(all_compactness)) if all_compactness else 0.0
        )
        self.min_compactness = min(all_compactness.values()) if all_compactness else 0.0
        self.num_fully_connected_groups = self.get_num_fully_connected_groups(distance_matrix)

    @cached_property
    def qubit_to_group(self):
        """Convert group_qubits dict to qubit to group mapping."""
        qubit_to_group = {}
        for group_id, qubits in self.group_qubits.items():
            for qubit in qubits:
                if qubit in qubit_to_group:
                    raise ValueError(
                        f"Qubit {qubit} is assigned to multiple groups: "
                        f"{group_id} and {qubit_to_group[qubit]}"
                    )
                qubit_to_group[qubit] = group_id
        return qubit_to_group

    def all_minimum_distance_in_groups(self, distance_matrix) -> dict[int, int]:
        """Calculate the minimum distance between qubits in each group."""
        min_distances = {}
        for group_id, qubits in self.group_qubits.items():
            if len(qubits) < 2:
                min_distances[group_id] = 0
                continue
            min_distance = float("inf")
            for u in qubits:
                for v in qubits:
                    if u != v:
                        min_distance = min(min_distance, distance_matrix[u][v])
            min_distances[group_id] = int(min_distance)  # Convert to Python int
        return min_distances

    def average_distance_in_groups(self, distance_matrix):
        """Calculate the average distance between qubits in each group."""
        avg_distances = {}
        total_distances_sum = 0
        total_count = 0

        for group_id, qubits in self.group_qubits.items():
            if len(qubits) < 2:
                avg_distances[group_id] = 0.0
                continue
            total_distance = 0
            count = 0
            for u in qubits:
                for v in qubits:
                    if u != v:
                        total_distance += distance_matrix[u][v]
                        count += 1
            group_avg = float(total_distance / count) if count > 0 else 0.0
            avg_distances[group_id] = group_avg
            total_distances_sum += total_distance
            total_count += count

        # Return overall average across all groups
        return float(total_distances_sum / total_count) if total_count > 0 else 0.0

    def get_num_fully_connected_groups(self, distance_matrix):
        """Count the number of fully connected groups."""
        count = 0
        for group_id, qubits in self.group_qubits.items():
            if len(qubits) < 2:
                continue
            fully_connected = True
            for u in qubits:
                for v in qubits:
                    if u != v and distance_matrix[u][v] == float("inf"):
                        fully_connected = False
                        break
                if not fully_connected:
                    break
            if fully_connected:
                count += 1
        return count

    def all_groups_compactness(self, graph):
        """Calculate the compactness of all groups based on edge density."""
        compactness = {}
        for group_id, qubits in self.group_qubits.items():
            if len(qubits) < 2:
                compactness[group_id] = 0.0
                continue
            subgraph = graph.subgraph(qubits)
            num_edges = subgraph.number_of_edges()
            num_nodes = len(qubits)
            if num_nodes < 2:
                compactness[group_id] = 0.0
            else:
                compactness[group_id] = num_edges / (num_nodes * (num_nodes - 1) / 2)
        return compactness

    @property
    def metrics_calculated(self):
        return self.min_intra_dist is not None

    def to_dict(self):
        """Convert to JSON-serializable dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ControllerLayout":
        """Create instance from dictionary and restore all metrics."""
        # Create instance normally (calls __post_init__)
        instance = cls(
            hardware_label=data["hardware_label"],
            layout_type=data["layout_type"],
            qpg=data["qpg"],
            group_qubits=data["group_qubits"],
            method=data["method"],
        )

        # Restore metrics if they exist in the data
        metrics_to_restore = [
            "min_intra_dist",
            "avg_min_intra_dist",
            "avg_intra_dist",
            "avg_compactness",
            "min_compactness",
            "num_fully_connected_groups",
        ]

        for metric in metrics_to_restore:
            if metric in data and data[metric] is not None:
                setattr(instance, metric, data[metric])

        return instance

    def to_json(self):
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_json(cls, json_str):
        return cls.from_dict(json.loads(json_str))

    def validate_qpg(self, qpg):
        """Validate that the qpg is a positive integer and does not exceed the number of qubits."""
        if not isinstance(qpg, int) or qpg <= 0:
            raise ValueError(f"qpg must be a positive integer, got {qpg}.")
        if qpg > self.num_qubits:
            raise ValueError(f"qpg {qpg} cannot exceed total qubits {self.num_qubits}.")

    def validate_k_distance(self, graph, k):
        """Verify coloring for larger graphs"""

        distances = nx.floyd_warshall_numpy(graph).astype(int)

        # Check distance constraints
        for node1 in graph.nodes():
            for node2 in graph.nodes():
                if node1 != node2:
                    dist = distances[node1][node2]

                    assert dist >= k or self.qubit_to_group[node1] != self.qubit_to_group[node2], (
                        f"Nodes {node1} and {node2} have same group but distance {dist} < {k}"
                    )

    def __repr__(self) -> str:
        """Custom representation."""
        metrics_status = "metrics" if self.metrics_calculated else "no metrics"
        return f"ControllerLayout(hardware={self.hardware_label}, layout_type={self.layout_type}, groups={self.num_groups}, {metrics_status})"


def rustworkx_to_networkx(rg):
    """Convert RustWorkX graph to NetworkX graph"""
    ng = nx.Graph()
    ng.add_nodes_from(rg.node_indices())
    ng.add_edges_from(rg.edge_list())
    return ng


def get_balanced_group_sizes(num_qubits: int, qpg: int) -> list[int]:
    """Distributes qubits into balanced groups.

    Args:
        num_qubits: Total number of qubits.
        qpg: Maximum qubits per group.

    Returns:
        List of group sizes, balanced so groups are close to same size.
    """
    if qpg <= 0:
        raise ValueError("qpg must be a positive integer.")

    # Calculate minimum number of groups needed
    num_groups = (num_qubits + qpg - 1) // qpg  # Ceiling division

    # Calculate ideal balanced size
    ideal_size = num_qubits // num_groups
    remainder = num_qubits % num_groups

    # Create groups with balanced sizes
    group_sizes = []
    for i in range(num_groups):
        group_size = ideal_size + (1 if i < remainder else 0)
        group_sizes.append(group_size)

    return group_sizes


def generate_layout(
    target: Target,
    layout_type: LayoutType,
    qpg: int,
    hardware_label: str = "custom",
    random_seed: int = None,
    max_refinement_steps: int = 100,
    calculate_metrics: bool = True,
) -> ControllerLayout:
    """Generate a single controller layout for a given target and strategy.

    This is the simplified API for generating layouts on-the-fly without
    needing to pre-compute and store all layouts for all qpg values.

    Args:
        target: Qiskit Target with coupling map information.
        layout_type: One of "trivial", "dispersed", "clustered", "random".
        qpg: Maximum qubits per group (qubits per controller).
        hardware_label: Label for the hardware (default: "custom").
        random_seed: Seed for random layout generation (optional).
        max_refinement_steps: Max refinement steps for clustered layout.
        calculate_metrics: Whether to calculate layout metrics (default: True).

    Returns:
        ControllerLayout instance with the generated layout.

    Example:
        >>> from qiskit_ibm_runtime import QiskitRuntimeService
        >>> service = QiskitRuntimeService()
        >>> backend = service.backend("ibm_brisbane")
        >>> target = backend.target
        >>> layout = generate_layout(target, "clustered", qpg=8)
        >>> print(layout.group_qubits)
    """
    if layout_type not in get_args(LayoutType):
        raise ValueError(
            f"Invalid layout_type: {layout_type}. Must be one of {get_args(LayoutType)}."
        )

    # Extract graph from target
    num_qubits = target.num_qubits
    coupling_map = target.build_coupling_map()
    rw_graph = coupling_map.graph.to_undirected()
    nx_graph = rustworkx_to_networkx(rw_graph)

    # Get balanced group sizes
    group_sizes = get_balanced_group_sizes(num_qubits, qpg)

    # Generate layout based on type
    if layout_type == "trivial":
        group_qubits = _generate_trivial_layout(num_qubits, group_sizes)
        method = "trivial"

    elif layout_type == "dispersed":
        group_qubits, method = _generate_dispersed_layout(nx_graph, group_sizes)

    elif layout_type == "clustered":
        group_qubits = _generate_clustered_layout(nx_graph, group_sizes, max_refinement_steps)
        method = "bfs"

    elif layout_type == "random":
        group_qubits = _generate_random_layout(num_qubits, group_sizes, random_seed)
        method = "random_shuffle"

    # Create ControllerLayout instance
    layout = ControllerLayout(
        hardware_label=hardware_label,
        layout_type=layout_type,
        qpg=qpg,
        group_qubits=group_qubits,
        method=method,
    )

    # Calculate metrics if requested
    if calculate_metrics:
        layout.calculate_metrics(nx_graph)

    return layout


def _generate_trivial_layout(num_qubits: int, group_sizes: list[int]) -> dict[int, list[int]]:
    """Generate trivial layout where qubits are assigned in order."""
    group_qubits = {}
    qubit_idx = 0

    for group_id, group_size in enumerate(group_sizes):
        group_qubits[group_id] = list(range(qubit_idx, qubit_idx + group_size))
        qubit_idx += group_size

    return group_qubits


def _generate_dispersed_layout(
    nx_graph: nx.Graph, group_sizes: list[int]
) -> tuple[dict[int, list[int]], str]:
    """Generate dispersed layout maximizing distance between qubits in same group."""
    num_qubits = nx_graph.number_of_nodes()
    color_sizes = {i: size for i, size in enumerate(group_sizes)}

    # Handle edge case where qpg equals total qubits
    if len(group_sizes) == 1:
        return {0: list(range(num_qubits))}, "trivial"

    max_k = nx.diameter(nx_graph)
    best_coloring = None
    best_min_dist = 0
    best_avg_dist = 0.0
    found_exact = False

    # Try exact k-distance coloring from max_k down to 2
    for k in range(max_k + 1, 1, -1):
        exact_coloring, exact_min_dist, exact_avg_dist, exact_success = exact_k_distance_coloring(
            G=nx_graph,
            color_sizes=color_sizes,
            k=k,
        )

        if exact_success:
            if exact_min_dist > best_min_dist or (
                exact_min_dist == best_min_dist and exact_avg_dist > best_avg_dist
            ):
                best_coloring = exact_coloring
                best_min_dist = exact_min_dist
                best_avg_dist = exact_avg_dist
                found_exact = True
                break

    # If no exact solution, try heuristic
    if not found_exact:
        heuristic_coloring, heur_min_dist, heur_avg_dist, _ = heuristic_distance_coloring(
            G=nx_graph,
            color_sizes=color_sizes,
            num_attempts=10,
            use_parallel=False,
        )

        if heuristic_coloring is not None:
            best_coloring = heuristic_coloring
            best_min_dist = heur_min_dist
            best_avg_dist = heur_avg_dist

    if best_coloring is None:
        raise ValueError(f"Could not find valid dispersed layout for group_sizes={group_sizes}")

    # Convert coloring to groups
    groups = defaultdict(list)
    for node, group_id in best_coloring.items():
        groups[group_id].append(node)

    method = "exact" if found_exact else "heuristic"
    return dict(groups), method


def _generate_clustered_layout(
    nx_graph: nx.Graph, group_sizes: list[int], max_refinement_steps: int
) -> dict[int, list[int]]:
    """Generate clustered layout minimizing distance between qubits in same group."""
    qubit_to_cluster = cluster_qubits_bfs(
        nx_graph, group_sizes, refinement_steps=max_refinement_steps
    )

    # Convert to groups
    groups = defaultdict(list)
    for node, group_id in qubit_to_cluster.items():
        groups[group_id].append(node)

    return dict(groups)


def _generate_random_layout(
    num_qubits: int, group_sizes: list[int], seed: int = None
) -> dict[int, list[int]]:
    """Generate random layout with balanced group sizes."""
    if seed is not None:
        random.seed(seed)

    # Shuffle qubits randomly
    all_qubits = list(range(num_qubits))
    random.shuffle(all_qubits)

    # Assign to groups
    group_qubits = {}
    qubit_idx = 0

    for group_id, group_size in enumerate(group_sizes):
        group_qubits[group_id] = all_qubits[qubit_idx : qubit_idx + group_size]
        qubit_idx += group_size

    return group_qubits


class QubitControllerLayoutManager:
    """Manager for generating, saving, and loading controller layouts for hardware types.

    This class handles batch generation and persistence of layouts. For generating
    a single layout on-the-fly, use the `generate_layout()` function instead.
    """

    def __init__(self, hardware_label: str, target: Target = None):
        self.layout_dir = LAYOUTS_DIR
        self.hardware_label = hardware_label
        self.layouts_file = self.layout_dir / f"{hardware_label}.json.gz"

        if target is not None:
            self.target = target
            self.target_num_qubits = target.num_qubits
            self.coupling_map = target.build_coupling_map()
            self.graph = self.coupling_map.graph.to_undirected()
            self.nx_graph = rustworkx_to_networkx(self.graph)
            self.distance_matrix = self.coupling_map.distance_matrix.astype(int)

    def _get_unique_qpg_values(self, start_qpg: int = 1) -> list[tuple[int, int, list[int]]]:
        """Get unique (layout_key, qpg, group_sizes) tuples to avoid duplicate layouts.

        Returns list of (layout_key, qpg, group_sizes) for each unique layout configuration.
        """
        seen_layout_keys = set()
        unique_configs = []

        for qpg in range(start_qpg, self.target_num_qubits + 1):
            group_sizes = get_balanced_group_sizes(self.target_num_qubits, qpg)
            layout_key = max(group_sizes)

            if layout_key not in seen_layout_keys:
                seen_layout_keys.add(layout_key)
                unique_configs.append((layout_key, qpg, group_sizes))

        return unique_configs

    def _generate_all_layouts_for_type(
        self,
        layout_type: LayoutType,
        max_refinement_steps: int = 100,
        random_seed: int = None,
    ) -> dict[int, ControllerLayout]:
        """Generate all layouts for a specific layout type using the helper functions."""
        all_layouts = {}
        start_qpg = 2 if layout_type in ("dispersed", "clustered") else 1

        for layout_key, qpg, group_sizes in self._get_unique_qpg_values(start_qpg):
            logger.info(f"Generating {layout_type} layout for qpg={qpg}")

            if layout_type == "trivial":
                group_qubits = _generate_trivial_layout(self.target_num_qubits, group_sizes)
                method = "trivial"
            elif layout_type == "dispersed":
                group_qubits, method = _generate_dispersed_layout(self.nx_graph, group_sizes)
            elif layout_type == "clustered":
                group_qubits = _generate_clustered_layout(
                    self.nx_graph, group_sizes, max_refinement_steps
                )
                method = "bfs"
            elif layout_type == "random":
                group_qubits = _generate_random_layout(
                    self.target_num_qubits, group_sizes, random_seed
                )
                method = "random_shuffle"

            layout = ControllerLayout(
                hardware_label=self.hardware_label,
                layout_type=layout_type,
                qpg=qpg,
                group_qubits=group_qubits,
                method=method,
            )
            layout.calculate_metrics(self.nx_graph)

            all_layouts[layout_key] = layout

        return all_layouts

    def save_all_layouts(self, all_layouts_by_type: dict[LayoutType, dict[int, ControllerLayout]]):
        """Save all layout types to a single compressed JSON file."""
        layout_data = {
            "hardware": self.hardware_label,
            "total_qubits": self.target_num_qubits,
            "layout_types": {
                layout_type: {str(qpg): layout.to_dict() for qpg, layout in layouts.items()}
                for layout_type, layouts in all_layouts_by_type.items()
            },
        }

        with gzip.open(self.layouts_file, "wt") as f:
            json.dump(layout_data, f, indent=1)

        total_layouts = sum(len(layouts) for layouts in all_layouts_by_type.values())
        layout_type_counts = {lt: len(layouts) for lt, layouts in all_layouts_by_type.items()}
        logger.info(f"Saved {total_layouts} layouts ({layout_type_counts}) to {self.layouts_file}")

    def save_visualizations(
        self, layout_type: LayoutType, all_layouts: dict[int, ControllerLayout]
    ):
        """Visualize layouts with colored groups and save as PDFs."""
        for layout in all_layouts.values():
            preview_dir = self.layout_dir / "layout_previews" / layout.hardware_label
            preview_dir.mkdir(parents=True, exist_ok=True)
            filepath = preview_dir / f"{layout_type}_qpg_{layout.qpg}.pdf"

            colors = cm.tab20.colors
            if len(layout.group_qubits) > len(colors):
                logger.warning(
                    f"More groups ({len(layout.group_qubits)}) than available colors ({len(colors)})"
                )

            node_colors = {}
            for group_id, group in layout.group_qubits.items():
                color = colors[group_id % len(colors)]
                for node in group:
                    node_colors[node] = color

            node_colors = [
                node_colors.get(node, "lightgray") for node in sorted(self.nx_graph.nodes())
            ]

            plt.figure(figsize=(10, 8))
            pos = nx.nx_agraph.graphviz_layout(self.nx_graph, prog="neato")
            nx.draw(
                self.nx_graph,
                pos,
                node_color=node_colors,
                with_labels=True,
                node_size=500,
                font_size=10,
                font_weight="bold",
            )

            for i, color in enumerate(colors[: len(layout.group_qubits)]):
                plt.plot(
                    [],
                    [],
                    "o",
                    color=color,
                    label=f"Group {i} ({len(layout.group_qubits[i])} nodes)",
                )
            plt.legend(loc="best")
            plt.title(f"Graph Grouping (qpg={layout.qpg})")

            plt.savefig(
                filepath.with_suffix(".pdf"),
                bbox_inches="tight",
                facecolor="white",
                edgecolor="none",
                dpi=300,
            )
            plt.close()

    def generate_and_save_all_layouts(self, max_refinement_steps=100, random_seed=None):
        """Generate and save all layout types for this hardware."""
        logger.info(f"Generating all layouts for hardware '{self.hardware_label}'")

        all_layouts_by_type = {}
        for layout_type in get_args(LayoutType):
            logger.info(f"Generating {layout_type} layouts...")
            all_layouts_by_type[layout_type] = self._generate_all_layouts_for_type(
                layout_type, max_refinement_steps, random_seed
            )

        self.save_all_layouts(all_layouts_by_type)

        for layout_type, layouts in all_layouts_by_type.items():
            if layouts:
                self.save_visualizations(layout_type, layouts)

        logger.info("All layouts generated and saved successfully!")

    def list_layouts(
        self,
        hardware_label: str = None,
        layout_types: list[LayoutType] = ["trivial"],
        verbose: bool = True,
        lower_qpg_limit: int = 2,
    ):
        """List all available layouts, optionally filtered."""
        if hardware_label is None:
            hardware_label = self.hardware_label

        unified_filepath = self.layout_dir / f"{hardware_label}.json.gz"
        if not unified_filepath.exists():
            logger.warning(f"Layout file not found: {unified_filepath}")
            return []

        with gzip.open(unified_filepath, "rt") as f:
            file_content = json.load(f)

        if file_content["hardware"] != hardware_label:
            logger.warning(f"Hardware mismatch in {unified_filepath}")
            return []

        return_layouts = [
            layout_data
            for lt, layouts_for_type in file_content.get("layout_types", {}).items()
            if lt in layout_types
            for layout_data in layouts_for_type.values()
            if layout_data["qpg"] >= lower_qpg_limit
        ]

        logger.info(
            f"Found {len(return_layouts)} layouts for hardware '{hardware_label}', filtered by {layout_types}"
        )
        if verbose:
            for layout in return_layouts:
                logger.info(ControllerLayout.from_dict(layout))

        return return_layouts

    def load_layout(self, layout_type: LayoutType, qpg: int, hardware_label: str = None):
        """Load a layout configuration from file."""
        if hardware_label is None:
            hardware_label = self.hardware_label

        if layout_type not in get_args(LayoutType):
            raise ValueError(
                f"Invalid layout type: {layout_type}. Must be one of {get_args(LayoutType)}."
            )

        if not self.layouts_file.exists():
            raise FileNotFoundError(f"Layout file not found: {self.layouts_file}")

        with gzip.open(self.layouts_file, "rt") as f:
            file_content = json.load(f)

        if file_content["hardware"] != hardware_label:
            raise ValueError(
                f"Layout hardware mismatch: expected {hardware_label}, found {file_content['hardware']}"
            )

        if layout_type not in file_content["layout_types"]:
            raise ValueError(f"Layout type '{layout_type}' not found in {self.layouts_file}")

        available_qpgs = list(file_content["layout_types"][layout_type].keys())

        # Use the maximum qpg that is <= requested qpg
        if str(qpg) not in available_qpgs:
            valid_qpgs = [int(q) for q in available_qpgs if int(q) <= qpg]
            if not valid_qpgs:
                raise ValueError(f"No suitable layout found for qpg={qpg} in {self.layouts_file}")
            qpg = max(valid_qpgs)

        layout_data = file_content["layout_types"][layout_type].get(str(qpg))
        if layout_data is None:
            raise ValueError(f"No layout found for {layout_type} qpg={qpg}")

        controller_layout = ControllerLayout.from_dict(layout_data)

        if (
            hasattr(self, "target_num_qubits")
            and controller_layout.num_qubits != self.target_num_qubits
        ):
            raise ValueError(
                f"Total qubits mismatch: expected {self.target_num_qubits}, "
                f"found {controller_layout.num_qubits}"
            )

        return controller_layout

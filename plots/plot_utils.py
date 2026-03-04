import warnings

from qiskit.visualization import plot_coupling_map
from qiskit.transpiler import CouplingMap
from qiskit_ibm_runtime.fake_provider import FakeProviderForBackendV2
import numpy as np
import pygraphviz as pgv
import plot_settings

warnings.filterwarnings(
    "ignore",
    message="Properties of fake_nighthawk are not intended to represent typical nighthawk error values.",
)


def plot_grid121(ax):
    coupling_map = CouplingMap.from_grid(11, 11)

    qubit_coordinates = []
    for row in range(11):
        for col in range(11):
            qubit_coordinates.append([col, row])

    _ = plot_coupling_map(
        num_qubits=121,
        qubit_coordinates=qubit_coordinates,
        coupling_map=coupling_map.get_edges(),
        figsize=(1, 1),
        label_qubits=False,
        qubit_size=100,
        line_width=10,
        font_size=8,
        qubit_color=["black"] * 121,
        line_color=["black"] * len(coupling_map.get_edges()),
        ax=ax,
    )


def plot_grid25(ax):
    coupling_map = CouplingMap.from_grid(5, 5)

    qubit_coordinates = []

    # Generate coordinates for 11x11 grid
    for row in range(5):
        for col in range(5):
            qubit_coordinates.append([col, row])

    _ = plot_coupling_map(
        num_qubits=25,
        qubit_coordinates=qubit_coordinates,
        coupling_map=coupling_map.get_edges(),
        figsize=(1, 1),
        label_qubits=False,
        qubit_size=100,
        line_width=10,
        font_size=8,
        qubit_color=["black"] * 25,
        line_color=["black"] * len(coupling_map.get_edges()),
        ax=ax,
    )


def plot_brisbane(ax):
    def generate_neato_coordinates(coupling_map, num_qubits):
        """
        Generate coordinates using pygraphviz neato layout from a coupling map.

        Args:
            coupling_map: List of edges [(qubit1, qubit2), ...]
            num_qubits: Total number of qubits

        Returns:
            coordinates: List of [x, y] positions for each qubit
        """
        # Create a new graph
        G = pgv.AGraph(strict=False, directed=False)

        # Add nodes
        for i in range(num_qubits):
            G.add_node(i)

        # Add edges from coupling map
        for edge in coupling_map:
            G.add_edge(edge[0], edge[1])

        # Apply neato layout
        G.layout(prog="neato")

        # Extract coordinates
        coordinates = []
        for i in range(num_qubits):
            node = G.get_node(i)
            pos = node.attr["pos"]
            # Parse position string "x,y"
            x, y = map(float, pos.split(","))
            coordinates.append([x, y])

        return coordinates

    provider = FakeProviderForBackendV2()
    backend = provider.backend("fake_brisbane")

    # For IBM Eagle (127 qubits)
    target = backend.target
    coupling_map = target.build_coupling_map().get_edges()

    # Generate coordinates using neato layout
    neato_coordinates = generate_neato_coordinates(coupling_map, 127)

    # Optional: Scale and center coordinates for better visualization
    def normalize_coordinates(coords):
        """Normalize coordinates to a reasonable range"""
        coords = np.array(coords)

        # Center coordinates
        coords = coords - np.mean(coords, axis=0)

        # Scale to reasonable range (e.g., 0-20)
        scale = 20 / max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
        coords = coords * scale

        # Shift to positive values
        coords = coords - np.min(coords) + 1

        return coords.tolist()

    normalized_coords = normalize_coordinates(neato_coordinates)

    def rotate_coordinates(coords, angle_degrees):
        """Rotate coordinates by angle (in degrees)"""
        angle_rad = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        coords_array = np.array(coords)
        rotated = coords_array @ rotation_matrix.T
        return rotated.tolist()

    rotated_coords = rotate_coordinates(normalized_coords, 103)

    # Plot with generated coordinates
    _ = plot_coupling_map(
        num_qubits=127,
        qubit_coordinates=rotated_coords,
        coupling_map=coupling_map,
        figsize=(1, 1),
        label_qubits=False,
        qubit_size=200,
        line_width=10,
        qubit_color=["black"] * 127,
        line_color=["black"] * len(coupling_map),
        ax=ax,
    )


def plot_coupling_map_from_backend(backend, ax=None):
    def generate_neato_coordinates(coupling_map, num_qubits):
        """
        Generate coordinates using pygraphviz neato layout from a coupling map.

        Args:
            coupling_map: List of edges [(qubit1, qubit2), ...]
            num_qubits: Total number of qubits

        Returns:
            coordinates: List of [x, y] positions for each qubit
        """
        # Create a new graph
        G = pgv.AGraph(strict=False, directed=False)

        # Add nodes
        for i in range(num_qubits):
            G.add_node(i)

        # Add edges from coupling map
        for edge in coupling_map:
            G.add_edge(edge[0], edge[1])

        # Apply neato layout
        G.layout(prog="neato")

        # Extract coordinates
        coordinates = []
        for i in range(num_qubits):
            node = G.get_node(i)
            pos = node.attr["pos"]
            # Parse position string "x,y"
            x, y = map(float, pos.split(","))
            coordinates.append([x, y])

        return coordinates

    target = backend.target
    coupling_map = target.build_coupling_map().get_edges()

    # Generate coordinates using neato layout
    neato_coordinates = generate_neato_coordinates(coupling_map, 127)

    # Optional: Scale and center coordinates for better visualization
    def normalize_coordinates(coords):
        """Normalize coordinates to a reasonable range"""
        coords = np.array(coords)

        # Center coordinates
        coords = coords - np.mean(coords, axis=0)

        # Scale to reasonable range (e.g., 0-20)
        scale = 20 / max(np.ptp(coords[:, 0]), np.ptp(coords[:, 1]))
        coords = coords * scale

        # Shift to positive values
        coords = coords - np.min(coords) + 1

        return coords.tolist()

    normalized_coords = normalize_coordinates(neato_coordinates)

    def rotate_coordinates(coords, angle_degrees):
        """Rotate coordinates by angle (in degrees)"""
        angle_rad = np.radians(angle_degrees)
        cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

        coords_array = np.array(coords)
        rotated = coords_array @ rotation_matrix.T
        return rotated.tolist()

    rotated_coords = rotate_coordinates(normalized_coords, 103)

    # Plot with generated coordinates
    _ = plot_coupling_map(
        num_qubits=127,
        qubit_coordinates=rotated_coords,
        coupling_map=coupling_map,
        figsize=(1, 1),
        label_qubits=False,
        qubit_size=200,
        line_width=10,
        qubit_color=["black"] * 127,
        line_color=["black"] * len(coupling_map),
        ax=ax,
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    provider = FakeProviderForBackendV2()
    backend = provider.backend("fake_brisbane")
    fig, ax = plt.subplots(
        figsize=(plot_settings.SINGLE_COLUMN_WIDTH, plot_settings.SINGLE_COLUMN_WIDTH)
    )
    plot_coupling_map_from_backend(backend, ax=ax)

    fig, ax = plt.subplots(
        figsize=(plot_settings.SINGLE_COLUMN_WIDTH, plot_settings.SINGLE_COLUMN_WIDTH)
    )
    plot_grid25(ax)

    fig, ax = plt.subplots(
        figsize=(plot_settings.SINGLE_COLUMN_WIDTH, plot_settings.SINGLE_COLUMN_WIDTH)
    )
    plot_grid121(ax)

    fig, ax = plt.subplots(
        figsize=(plot_settings.SINGLE_COLUMN_WIDTH, plot_settings.SINGLE_COLUMN_WIDTH)
    )
    plot_brisbane(ax)

    plt.show()

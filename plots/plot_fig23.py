from collections import defaultdict
import itertools

import matplotlib.pyplot as plt
import numpy as np
from plot_settings import SINGLE_COLUMN_WIDTH, wong_colors
from qiskit import QuantumCircuit
from tqdm import tqdm


def layerize_greedy(circ, remove_meas=True):
    """Greedy ASAP layerizer: earliest layer where all qubits are free."""
    n_qubits = circ.num_qubits
    next_free = [0] * n_qubits
    layers = defaultdict(list)
    for item in circ.data:
        inst = item.operation
        qargs = item.qubits
        if remove_meas and inst.name in ("measure", "reset", "barrier"):
            continue
        qidxs = [q._index for q in qargs]
        if not qidxs:
            continue
        layer = max(next_free[q] for q in qidxs)
        layers[layer].append((inst, qidxs))
        for q in qidxs:
            next_free[q] = layer + 1
    return layers


def serialization_overhead_from_layers(layers, num_qubits, m, gamma=4):
    """Compute overhead factor for one circuit given precomputed layers."""
    L = max(layers.keys()) + 1 if layers else 0
    if L == 0:
        return 1.0

    num_switches = num_qubits // m
    total_serial = 0.0
    total_ideal = 0.0

    for s in range(num_switches):
        grp = list(range(s * m, (s + 1) * m))
        for ell in range(L):
            oneq = 0
            twoq = 0
            for inst, qidxs in layers.get(ell, []):
                if all(q in grp for q in qidxs):
                    if getattr(inst, "num_qubits", None) == 1:
                        oneq += 1
                    elif getattr(inst, "num_qubits", None) == 2:
                        twoq += 1

            T_ideal = 0.0
            if oneq > 0:
                T_ideal = max(T_ideal, 1.0)
            if twoq > 0:
                T_ideal = max(T_ideal, float(gamma))

            if twoq > 0:
                T_serial = max(float(gamma), float(oneq) * 1.0)
            else:
                T_serial = float(oneq) * 1.0

            total_serial += T_serial
            total_ideal += T_ideal

    return (total_serial / total_ideal) if total_ideal > 0 else 1.0


def serialization_overhead(qc, m, gamma=4):
    """Convenience wrapper: layerize + compute overhead."""
    layers = layerize_greedy(qc)
    return serialization_overhead_from_layers(layers, qc.num_qubits, m, gamma=gamma)


def synthetic_binomial_circuit(n_qubits, depth, p1, p2, rng=None):
    """
    Generate a synthetic circuit with gates chosen via binomial sampling.

    p1: probability of a 1q gate per qubit per layer
    p2: probability of a 2q gate per nearest-neighbor pair per layer
    Connectivity: square grid (nearest-neighbors)
    """
    if rng is None:
        rng = np.random.default_rng()

    L = int(np.sqrt(n_qubits))
    if L * L != n_qubits:
        raise ValueError("n_qubits must be a perfect square for a square grid.")

    qc = QuantumCircuit(n_qubits)

    edges = []
    for i in range(L):
        for j in range(L):
            idx = i * L + j
            if j + 1 < L:
                edges.append((idx, idx + 1))
            if i + 1 < L:
                edges.append((idx, idx + L))

    for _ in range(depth):
        for q in range(n_qubits):
            if rng.random() < p1:
                qc.rx(np.pi / 4, q)

        for i, j in edges:
            if rng.random() < p2:
                qc.cz(i, j)

    return qc


def run_experiment(m_values, num_qubits, depth, gamma, n_samples, rng, p1, p2):
    mean_overheads = []
    for m in tqdm(m_values):
        vals = []
        for _ in range(n_samples):
            qc_syn = synthetic_binomial_circuit(num_qubits, depth, p1, p2, rng=rng)
            vals.append(serialization_overhead(qc_syn, m, gamma=gamma))
        mean_overheads.append(np.mean(vals))
    return mean_overheads


def plot_figure() -> plt.Figure:
    num_qubits = 25
    depth = num_qubits
    m_values = np.arange(1, num_qubits, 2)
    n_samples = 1000
    rng = np.random.default_rng(428)

    params_list = [
        {"p1": 0.2, "p2": 0.01, "gamma": 5, "ref": "log"},
        {"p1": 0.2, "p2": 0.00, "gamma": 5, "ref": "linear"},
    ]

    colors = iter(wong_colors[1:])
    markers = itertools.cycle(["o", "s", "^", "D", "v", "P", "*", "X"])

    fig = plt.figure(figsize=(SINGLE_COLUMN_WIDTH, 2.5))
    ax = plt.gca()
    ref_handles = []
    ref_labels = []

    for params in params_list:
        color = next(colors)
        marker = next(markers)
        mean_overheads = run_experiment(
            m_values,
            num_qubits,
            depth,
            params["gamma"],
            n_samples,
            rng,
            params["p1"],
            params["p2"],
        )
        param_label = (
            f"$p_2={params['p2']:.2f}, t_\\text{{2q}}={params['gamma']}\\times t_\\text{{1q}}$"
        )
        ax.plot(m_values, mean_overheads, marker + ":", label=param_label, color=color)

        m_arr = np.array(m_values, dtype=float)
        if params["ref"] == "log":
            (ref_handle,) = ax.plot(
                m_arr,
                1.0 + np.log(m_arr) / np.log(m_arr[-1]) * (mean_overheads[-1] - 1.0),
                "-",
                color=color,
            )
            ref_handles.append(ref_handle)
            ref_labels.append("Logarithmic ref.")
        elif params["ref"] == "linear":
            dy = mean_overheads[-1] - mean_overheads[0]
            dx = m_arr[-1] - m_arr[0]
            k = dy / dx
            (ref_handle,) = ax.plot(m_arr, m_arr * k + mean_overheads[0] - k, "-", color=color)
            ref_handles.append(ref_handle)
            ref_labels.append("Linear ref.")

    ax.set_xlabel("Qubits per switch $k$")
    ax.set_ylabel("Relative overhead")
    ax.grid(True, which="both")

    all_handles, all_labels = ax.get_legend_handles_labels()
    interleaved_handles = []
    interleaved_labels = []
    for data_handle, data_label, ref_handle, ref_label in zip(
        all_handles, all_labels, ref_handles, ref_labels
    ):
        interleaved_handles.extend([data_handle, ref_handle])
        interleaved_labels.extend([data_label, ref_label])
    ax.legend(interleaved_handles, interleaved_labels)
    fig.tight_layout()

    return fig


if __name__ == "__main__":
    fig = plot_figure()
    plt.show()

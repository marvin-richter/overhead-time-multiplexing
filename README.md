# Overhead in Quantum Circuits with Time-Multiplexed Qubit Control

Code repository accompanying the publication:

> **Overhead in quantum circuits with time-multiplexed qubit control**
> Marvin Richter, Ingrid Strandberg, Simone Gasparinetti, and Anton Frisk Kockum
>
> PRX Quantum (2026) — Accepted 24 February 2026
> DOI: [10.1103/82cj-lfzy](https://doi.org/10.1103/82cj-lfzy)

## Abstract

When scaling up quantum processors in a cryogenic environment, it is desirable to limit the number of qubit drive lines going into the cryostat, since fewer lines makes cooling of the system more manageable and the need for complicated electronics setups is reduced. However, although time multiplexing of qubit control enables using just a few drive lines to steer many qubits, it comes with a trade-off: fewer drive lines means fewer qubits can be controlled in parallel, which leads to an overhead in the execution time for quantum algorithms. In this article, we quantify this trade-off through numerical and analytical investigations. For standard quantum processor layouts and typical gate times, we show that the trade-off is favorable for many common quantum algorithms—the number of drive lines can be significantly reduced without introducing much overhead. Specifically, we show that couplers for two-qubit gates can be grouped on common drive lines without any overhead up to a limit set by the connectivity of the qubits. For single-qubit gates, we find that the serialization overhead generally scales only logarithmically in the number of qubits sharing a drive line, and the serialization overhead relative to total quantum circuit duration tends to grow only sublinearly or stay nearly constant with the total number of qubits on the quantum processor. These results are promising for continued progress towards large-scale quantum computers.

## Repository structure

```
project/
├── README.md
├── pyproject.toml
├── src/                # Core library
├── scripts/            # Entry point for data generation (run.py)
├── plots/              # Code to reproduce all figures in the publication
├── configs/            # YAML config files specifying experiments
├── layouts/            # Layouts generated with layout manager can be stored here (optional)
└── results/            # Generated data for reproducing plots
```

### Key components

**`src/overhead_time_multiplexing/overhead_pass.py`** — The main contribution: a Qiskit transpiler [TransformationPass](https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.transpiler.TransformationPass) that inserts dummy gates to model the overhead from single-qubit gate serialization.

**`src/overhead_time_multiplexing/layouts/`** — Code to generate `ControllerLayout` objects, i.e., the mapping of qubits to controllers (switches) that determines which qubits can be operated concurrently.

**`src/overhead_time_multiplexing/targets/`** — Utilities to construct device targets:
  - Square-grid devices inspired by hardware at Chalmers University of Technology (see, e.g., [Kosen *et al.*, PRX Quantum **5**, 030350 (2024)](http://dx.doi.org/10.1103/PRXQuantum.5.030350))
  - IBM-based hardware, with a focus on the Eagle-generation device `ibm_brisbane`

**`src/overhead_time_multiplexing/experiments/`** — Experiment runners for the numerical simulations, configured via YAML files.

## Installation

This package is managed with [`uv`](https://github.com/astral-sh/uv). All dependencies are specified in `pyproject.toml`.

```bash
uv sync
```

### Note on `pygraphviz` (macOS)

The optional `pygraphviz` dependency (used only for plotting) requires extra steps on macOS when installed via Homebrew:

```bash
brew install graphviz
CFLAGS="-I/opt/homebrew/include" LDFLAGS="-L/opt/homebrew/lib" uv sync --group plot
```

## Running experiments

`scripts/run.py` is the main entry point for generating data. Example usage:

```bash
uv run scripts/run.py --parallel --n-workers 8 configs/demo_random.yaml
```

Generated data is saved to `results/`.

| Argument         | Description                                                                 |
| ---------------- | --------------------------------------------------------------------------- |
| `config_path`    | Path to an experiment config YAML file (e.g., `configs/paper/example.yaml`) |
| `--parallel`     | Run experiments in parallel using multiple processes                        |
| `--n-workers N`  | Number of parallel workers (default: auto-detect)                           |
| `--save-every N` | Save intermediate results every *N* experiments (default: 100)              |

## Reproducing the figures

`plots/plots.ipynb` provides an overview of all figures in the publication, including comments on which configs were used.

Most data can be regenerated with the configs in `configs/paper/`. A few figures (8, 9, 10, and 24) rely on data generated with an earlier version of this code due to the high runtime of the underlying simulations. The corresponding data is provided in `results/`, with configs to reproduce it also included.

## Citation

```bibtex
@article{Richter2026overhead,
  title   = {Overhead in quantum circuits with time-multiplexed qubit control},
  author  = {Richter, Marvin and Strandberg, Ingrid and Gasparinetti, Simone and Kockum, Anton Frisk},
  journal = {PRX Quantum},
  year    = {2026},
  month   = feb,
  doi     = {10.1103/82cj-lfzy},
  url     = {https://link.aps.org/doi/10.1103/82cj-lfzy}
}
```

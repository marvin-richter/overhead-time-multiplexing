import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import textalloc as ta
from matplotlib import cm
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit

from plot_settings import DOUBLE_COLUMN_WIDTH, ROOT, SINGLE_COLUMN_WIDTH, wong_colors


def _load_df(root) -> pl.DataFrame:
    """Load and process the main circuit benchmark dataframe."""
    df_grid121 = pl.read_parquet(root / "results" / "20250812-172502_grid121_mqt_N43200.parquet")
    df_ibm_eagle = pl.read_parquet(
        root / "results" / "20250812-142452_ibm_eagle_127_mqt_N45360.parquet"
    )
    df_grid25 = pl.read_parquet(
        root / "results" / "20250807-111625_grid25_mqt_N26880.parquet"
    ).drop(["circuit_gen", "routing"])

    return (
        pl.concat([df_grid121, df_ibm_eagle, df_grid25], how="vertical")
        .filter(pl.col("circuit_id") != "qwalk")
        .with_columns(
            n1q=(pl.col("qc_num_gates") * (2 - pl.col("average_operands"))).cast(pl.Int32)
        )
    )


def _load_circuit_stats(root) -> pl.DataFrame:
    """Load T2 dataset and return per-circuit median single-qubit gate counts."""
    return (
        pl.read_parquet(root / "results" / "20250822-150625_grid121_mqt_t2_N39690.parquet")
        .filter(pl.col("duration_two_qubit_gate") == 200e-9)
        .group_by("circuit_id")
        .agg(qc_routed_n1=pl.col("qc_routed_n1").median())
    )


def plot_figure(specifier: str) -> plt.Figure:
    """Create figure 8 (selected circuits) or figure 24 (all circuits with fits).

    Parameters
    ----------
    specifier:
        ``"fig08"`` – selected circuits on grid121, single-column.
        ``"fig24"`` – all circuits across all hardware configs, double-column.
    """
    df = _load_df(ROOT)
    circuit_stats = _load_circuit_stats(ROOT)

    if specifier == "fig08":
        return _plot_fig08(df, circuit_stats)
    elif specifier == "fig24":
        return _plot_fig24(df, circuit_stats)
    else:
        raise ValueError(f"Unknown specifier {specifier!r}. Use 'fig08' or 'fig24'.")


def _plot_fig08(df: pl.DataFrame, circuit_stats: pl.DataFrame) -> plt.Figure:
    circuits = ["shor", "qaoa", "qpeexact", "vqe_su2", "bmw_quark_cardinality"]
    hardware = "grid121"

    df_plot = (
        df.filter(pl.col("layout_type") == "trivial", pl.col("hardware_label") == hardware)
        .group_by("circuit_id", "qpg")
        .agg(
            overhead_routed_serial=(pl.col("duration_serial") - pl.col("duration_routing")).mean(),
        )
        .sort("circuit_id", "qpg")
    )

    fig, ax = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.6),
        layout="constrained",
    )

    x = np.linspace(2, 121, 1000)
    fitted_params = {}
    annotate_data = []

    for i, circuit_id in enumerate(circuits):
        circuit_data = df_plot.filter(pl.col("circuit_id") == circuit_id)
        x_data = circuit_data["qpg"].to_numpy()
        y_data = circuit_data["overhead_routed_serial"].to_numpy()

        qc_routed_n1q = circuit_stats.filter(pl.col("circuit_id") == circuit_id)[
            "qc_routed_n1"
        ].item()

        color = wong_colors[(i + 1) % len(wong_colors)]

        ax.plot(
            x_data,
            y_data,
            label=circuit_id.replace("shor", "shor_58q"),
            color=color,
            marker="o",
            markerfacecolor="white",
            markersize=3,
            linestyle="",
        )

        def fit_func(x, A, n1q=qc_routed_n1q):
            return A * np.log(x) * n1q * 20e-9

        popt, pcov = curve_fit(fit_func, x_data, y_data, p0=[1.0])
        fitted_A = popt[0]
        fit_error = np.sqrt(np.diag(pcov))[0]

        fitted_params[circuit_id] = {"A": fitted_A, "fit_error": fit_error}

        ax.plot(x, fit_func(x, fitted_A), linestyle="-", color=color)
        print(f"Circuit {circuit_id}: p = {fitted_A:.1e} ± {fit_error:.1e}")

        annotate_data.append(
            {
                "circuit_id": circuit_id,
                "fitted_A": fitted_A,
                "color": color,
                "x": x_data,
                "y": y_data,
            }
        )

    fit_legend = Line2D(
        [0],
        [0],
        linestyle="-",
        color="gray",
        label="Fit: $p N_{1}^\\text{routed} t_\\text{1q} \\log(k)$",
    )
    handles, _ = ax.get_legend_handles_labels()
    handles.append(fit_legend)
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.4), ncol=2)

    ax.loglog()
    ax.set_xlim(left=1)
    ax.set_xlabel("Qubits per switch $k$")
    ax.set_ylabel("Overhead (s)")

    print("\nFitted Parameters Summary:")
    print("Circuit ID | A (fitted) | Fit Error")
    print("-" * 55)
    for circuit_id, params in fitted_params.items():
        print(f"{circuit_id:9} | {params['A']:.3e} | {params['fit_error']:.3e}")

    for data in annotate_data:
        y_norm = (np.log10(data["y"][-1]) - np.log10(ax.get_ylim()[0])) / (
            np.log10(ax.get_ylim()[1]) - np.log10(ax.get_ylim()[0])
        )
        ax.annotate(
            f"p = {data['fitted_A']:.2f}",
            xy=(data["x"][-1], data["y"][-1]),
            xytext=(1.01, y_norm),
            textcoords="axes fraction",
            color=data["color"],
            fontsize=8,
        )

    return fig


def _plot_fig24(df: pl.DataFrame, circuit_stats: pl.DataFrame) -> plt.Figure:
    hardware_configs = ["grid25", "grid121", "ibm_eagle_127"]

    fig, axes = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH),
        nrows=1,
        ncols=3,
        layout="constrained",
        gridspec_kw={"wspace": 0.1, "hspace": 0.1},
    )

    for h, hardware in enumerate(hardware_configs):
        ax_current = axes[h]
        ax_current.set_ylabel("Overhead (s)")

        df_plot = (
            df.filter(pl.col("hardware_label") == hardware, pl.col("layout_type") == "trivial")
            .group_by("circuit_id", "qpg")
            .agg(
                overhead_routed_serial=(
                    pl.col("duration_serial") - pl.col("duration_routing")
                ).mean(),
                qc_routed_n1=pl.col("n1q").mean(),
            )
            .sort("circuit_id", "qpg")
        )

        x_theory = np.linspace(df_plot["qpg"].min(), df_plot["qpg"].max(), 100)

        circuits_by_overhead = (
            df_plot.group_by("circuit_id")
            .agg(largest_overhead=pl.col("overhead_routed_serial").max())
            .sort("largest_overhead", descending=True)
        )

        annotations = []

        for i, circuit_id in enumerate(circuits_by_overhead["circuit_id"]):
            circuit_data = df_plot.filter(pl.col("circuit_id") == circuit_id)
            x_data = circuit_data["qpg"].to_numpy()
            y_data = circuit_data["overhead_routed_serial"].to_numpy()

            n1q_routed = circuit_stats.filter(pl.col("circuit_id") == circuit_id)[
                "qc_routed_n1"
            ].item()

            color = cm.tab20(i % 20)
            ax_current.plot(
                x_data,
                y_data,
                linestyle="",
                marker="o",
                color=color,
                markerfacecolor="white",
                markersize=3,
                label=circuit_id,
            )

            def model(k, A, n1q=n1q_routed):
                return A * np.log(k) * n1q * 20e-9

            try:
                popt, pcov = curve_fit(model, x_data, y_data, p0=[1.0])
                fitted_A = popt[0]
                ax_current.plot(x_theory, model(x_theory, fitted_A), "-", color=color)
                annotations.append(
                    {
                        "circuit_id": circuit_id,
                        "A": fitted_A,
                        "color": color,
                        "x_line": x_theory[-1],
                        "y_line": model(x_theory, fitted_A)[-1],
                        "i": i,
                    }
                )
            except Exception as e:
                print(f"Failed to fit {circuit_id} on {hardware}: {e}")

        ax_current.loglog()
        ax_current.set_xlim(left=2.5)
        ax_current.set_xlabel("Qubits per switch $k$")

        if annotations:
            x_coords = [ann["x_line"] for ann in annotations]
            y_coords = [ann["y_line"] for ann in annotations]
            ann_labels = [str(ann["i"]) for ann in annotations]
            ann_colors = [ann["color"] for ann in annotations]

            xlims = ax_current.get_xlim()
            ax_current.set_xlim(xlims[0], xlims[1] * 2.5)

            ta.allocate(
                ax=ax_current,
                x=x_coords,
                y=y_coords,
                text_list=ann_labels,
                x_scatter=x_coords,
                y_scatter=y_coords,
                textsize=9,
                textcolor=ann_colors,
                draw_lines=True,
                linecolor=ann_colors,
                linewidth=0.8,
                min_distance=0.05,
                max_distance=0.2,
                margin=0.01,
                draw_all=True,
                nbr_candidates=1000,
                direction="east",
                avoid_label_lines_overlap=False,
            )

    caption_pos = (-0.05, 1.0)
    for ax_sub, label in zip(axes, ["(a)", "(b)", "(c)"]):
        ax_sub.annotate(label, xy=caption_pos, xycoords="axes fraction", ha="right", va="center")

    return fig


if __name__ == "__main__":
    for specifier in ("fig08", "fig24"):
        fig = plot_figure(specifier)
        plt.show()

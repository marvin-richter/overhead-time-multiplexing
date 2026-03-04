import matplotlib.pyplot as plt
import polars as pl
from matplotlib.lines import Line2D
from plot_settings import SINGLE_COLUMN_WIDTH, wong_colors, RESULTS_DIR, save_fig


def plot_figure() -> plt.Figure:
    hardware = "11x11"
    colors = wong_colors
    markers = {"trivial": "o", "dispersed": "s", "clustered": "^", "random": "D"}
    circuit_ids = [
        "shor",
        "randomcircuit",
        "qaoa",
        "qpeexact",
    ]

    df = pl.read_parquet(RESULTS_DIR / "20260303-224851_mqt_layouts_N336.parquet")

    df_plot = (
        df.filter(pl.col("hardware.id") == hardware)
        .filter(pl.col("hardware.layout.k") >= 2)
        .group_by("circuit.id", "hardware.layout.k", "hardware.layout.strategy")
        .agg(
            overhead_routed_serial=(pl.col("T_routed_serial") - pl.col("T_routed")).mean(),
        )
        .sort("circuit.id", "hardware.layout.k")
    )

    fig, axes = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.85),
        layout="constrained",
    )

    for i, circuit_id in enumerate(circuit_ids):
        for layout in df_plot["hardware.layout.strategy"].unique().sort():
            circuit_data = df_plot.filter(pl.col("circuit.id") == circuit_id).filter(
                pl.col("hardware.layout.strategy") == layout
            )
            x_data = circuit_data["hardware.layout.k"].to_numpy()
            y_data = circuit_data["overhead_routed_serial"].to_numpy()

            axes.plot(
                x_data,
                y_data,
                label=f"{circuit_id}",
                color=colors[i + 1 % len(colors)],
                marker=markers[layout],
                markersize=4,
                linestyle="-",
            )

    circuit_ids_legend = [c.replace("shor", "shor_58q") for c in circuit_ids]

    color_legend = [
        Line2D([0], [0], color=colors[i + 1 % len(colors)], linewidth=2, label=circuit_id)
        for i, circuit_id in enumerate(circuit_ids_legend)
    ]

    layout_types = df_plot["hardware.layout.strategy"].unique().sort().to_list()
    marker_legend = [
        Line2D(
            [0],
            [0],
            marker=markers[layout_type],
            color="black",
            markersize=6,
            linestyle="",
            label=layout_type,
        )
        for layout_type in layout_types
    ]

    legend1 = axes.legend(
        handles=color_legend, title="Algorithm", loc="lower right", bbox_to_anchor=(0.67, 0.0)
    )
    _ = axes.legend(
        handles=marker_legend, title="Layout type", loc="lower right", bbox_to_anchor=(1.0, 0.0)
    )
    axes.add_artist(legend1)

    axes.loglog()
    axes.set_xlim(left=1)
    axes.set_xlabel("Qubits per switch")
    axes.set_ylabel("Serialization overhead (s)")
    axes.grid(True, which="major", linestyle="--", linewidth=0.5, color="lightgray")
    axes.set_ylim(bottom=1e-6)

    return fig


if __name__ == "__main__":
    fig = plot_figure()
    save_fig(fig, "layout_comparison_11x11")
    plt.show()

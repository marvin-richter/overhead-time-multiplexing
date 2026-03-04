from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from plot_settings import SINGLE_COLUMN_WIDTH, wong_colors, RESULTS_DIR, save_fig


def plot_figure() -> plt.Figure:
    hardware = "11x11"
    layout = "trivial"

    df = pl.read_parquet(RESULTS_DIR / "20260302-222157_random_gates_11x11_N4200.parquet")
    df = df.with_columns(
        (pl.col("T_routed_serial") - pl.col("T_routed")).alias("overhead_dur_serial"),
    )

    fig, axes = plt.subplots(
        figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.6), sharex=True, layout="constrained"
    )

    for i, qpg in enumerate([121, 31, 13, 8, 2]):
        df_plot = df.filter(
            (pl.col("hardware.id") == hardware)
            & (pl.col("hardware.layout.strategy") == layout)
            & (pl.col("hardware.layout.k") == qpg)
        )

        axes.plot(
            df_plot["qc_routed_n1"],
            df_plot["overhead_dur_serial"] * 1e3,
            marker=".",
            linestyle="",
            alpha=0.5,
            markersize=1,
            color=wong_colors[i % len(wong_colors) + 1],
            label=f"{qpg}",
            markerfacecolor=wong_colors[(i % len(wong_colors)) + 1],
        )

        x = np.linspace(0, df_plot["qc_routed_n1"].max(), 100)
        y_theory = x * 20e-9 * np.log(qpg) * 1e3

        axes.plot(
            x,
            y_theory,
            marker="",
            linestyle="-",
            alpha=0.5,
            markersize=1,
            color=wong_colors[i % len(wong_colors) + 1],
        )

    axes.set_xlabel("Single qubit gates $N_{1}^\\text{routed}$")
    axes.set_ylabel("Overhead (ms)")

    legend1 = axes.legend(
        loc="best",
        framealpha=0.9,
        markerscale=6,
        alignment="center",
    )

    legend_elements = [
        Line2D(
            [0],
            [0],
            linestyle="-",
            color="gray",
            label="Prediction:\n$N_{1}^\\text{routed} t_\\text{1q} \\log(k) $",
        )
    ]
    _ = axes.legend(
        handles=legend_elements, loc="upper center", framealpha=0.9, bbox_to_anchor=(0.5, 1.0)
    )
    axes.add_artist(legend1)

    axes.set_yticks(np.arange(0, 9, step=2))
    axes.set_yticks(np.arange(0, 9, step=1), minor=True)
    axes.set_xticks(np.arange(0, 70_001, step=10_000), minor=True)

    return fig


if __name__ == "__main__":
    fig = plot_figure()
    save_fig(fig, "random_gates_scaling_11x11")
    plt.show()

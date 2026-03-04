from plot_settings import (
    SINGLE_COLUMN_WIDTH,
    wong_colors,
    ROOT,
)

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from plot_utils import plot_grid25, plot_grid121, plot_brisbane

_PARQUET_FILES = {
    "5x5": ROOT / "results" / "20260302-221600_random_gates_5x5_N4800.parquet",
    "11x11": ROOT / "results" / "20260302-222157_random_gates_11x11_N4200.parquet",
    "brisbane": ROOT / "results" / "20260303-124631_random_gates_brisbane_N840.parquet",
}

_QPG_VALUES = {
    "5x5": {"line": [2, 3, 5, 9, 25], "bar": [2, 7, 25]},
    "11x11": {"line": [2, 3, 8, 31, 121], "bar": [2, 13, 121]},
    "brisbane": {"line": [2, 3, 8, 26, 127], "bar": [2, 12, 127]},
}

_NUM_GATES_VALUES = {
    "5x5": [100, 1000, 2500],
    "11x11": [500, 4000, 10000],
    "brisbane": [500, 4000, 10000],
}

_INSET_PLOTTERS = {
    "5x5": plot_grid121,
    "11x11": plot_grid25,
    "brisbane": plot_brisbane,
}


def plot_figure(specifier: str) -> plt.Figure:
    """Create the overhead-vs-gates figure for the given hardware specifier.

    Parameters
    ----------
    specifier:
        One of ``"5x5"``, ``"11x11"``, or ``"brisbane"``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if specifier not in _PARQUET_FILES:
        raise ValueError(f"specifier must be one of {list(_PARQUET_FILES)}; got {specifier!r}")

    hardware = specifier
    layout = "trivial"

    df = pl.read_parquet(_PARQUET_FILES[specifier])
    df = df.with_columns(
        (pl.col("T_routed") - pl.col("T_trans")).alias("overhead_dur_routing"),
        (pl.col("T_routed_serial") - pl.col("T_routed")).alias("overhead_dur_serial"),
    )

    colors = wong_colors
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]
    line_styles = ["-", "--", "-.", ":", "-", "--", "-.", ":", "-", "--"]

    fig = plt.figure(
        figsize=(SINGLE_COLUMN_WIDTH, 1.7 * SINGLE_COLUMN_WIDTH),
        dpi=200,
        layout="constrained",
    )
    axd = fig.subplot_mosaic(
        [["abs"], ["rel"], ["bars"]],
        height_ratios=[1, 1, 1],
    )
    axd["rel"].sharex(axd["abs"])

    # ── First row: Absolute overhead ──────────────────────────────────────────
    ax_abs = axd["abs"]
    for j, qpg in enumerate(reversed(_QPG_VALUES[specifier]["line"])):
        df_plot = (
            df.filter(
                (pl.col("hardware.layout.k") == qpg)
                & (pl.col("hardware.layout.strategy") == layout)
                & (pl.col("hardware.id") == hardware)
            )
            .group_by(["circuit.num_gates"])
            .agg(
                [
                    pl.col("overhead_dur_serial").median().alias("overhead_serial_dur_median"),
                    pl.col("overhead_dur_serial").quantile(0.75).alias("overhead_serial_abs_q75"),
                    pl.col("overhead_dur_serial").quantile(0.25).alias("overhead_serial_abs_q25"),
                ]
            )
            .sort("circuit.num_gates")
        )
        ax_abs.plot(
            df_plot["circuit.num_gates"],
            df_plot["overhead_serial_dur_median"] * 1e3,
            label=qpg,
            color=colors[j + 1],
            marker=markers[j],
            linestyle=line_styles[j],
            markerfacecolor="white",
            alpha=1.0,
        )
        ax_abs.fill_between(
            df_plot["circuit.num_gates"],
            df_plot["overhead_serial_abs_q25"] * 1e3,
            df_plot["overhead_serial_abs_q75"] * 1e3,
            alpha=0.15,
            color=colors[j + 1],
        )

    ax_abs.set_ylabel("Serialization overhead (ms)")
    ax_abs.set_ylim(bottom=0)
    ylims = ax_abs.get_ylim()

    if specifier == "5x5":
        ax_abs.set_yticks(np.arange(0, ylims[1], step=0.1))
        ax_abs.set_yticks(np.arange(0, ylims[1], step=0.05), minor=True)
        ax_abs.set_yticklabels(["0" if t == 0.0 else f"{t:.1f}" for t in ax_abs.get_yticks()])
    elif specifier == "11x11":
        ax_abs.set_yticks(np.arange(0, ylims[1], step=2))
        ax_abs.set_yticks(np.arange(0, ylims[1], step=1), minor=True)
    elif specifier == "brisbane":
        ax_abs.set_yticks(np.arange(0, ylims[1], step=10))
        ax_abs.set_yticks(np.arange(0, ylims[1], step=5), minor=True)

    ax_abs.tick_params(which="minor", length=2, color="gray")
    ax_abs.legend(title="Qubits per switch $k$", loc="best", framealpha=0.9)

    # ── Second row: Relative overhead ─────────────────────────────────────────
    ax_rel = axd["rel"]
    for j, qpg in enumerate(reversed(_QPG_VALUES[specifier]["line"])):
        df_plot = (
            df.filter(
                (pl.col("hardware.layout.k") == qpg)
                & (pl.col("hardware.layout.strategy") == layout)
                & (pl.col("hardware.id") == hardware)
            )
            .group_by(["circuit.num_gates"])
            .agg(
                [
                    (pl.col("T_routed_serial") / pl.col("T_routed"))
                    .median()
                    .alias("dur_serial_rel"),
                    (pl.col("T_routed_serial") / pl.col("T_routed"))
                    .quantile(0.75)
                    .alias("dur_serial_rel_q75"),
                    (pl.col("T_routed_serial") / pl.col("T_routed"))
                    .quantile(0.25)
                    .alias("dur_serial_rel_q25"),
                ]
            )
            .sort("circuit.num_gates")
        )
        ax_rel.plot(
            df_plot["circuit.num_gates"],
            df_plot["dur_serial_rel"],
            label=qpg,
            color=colors[j + 1],
            marker=markers[j],
            linestyle=line_styles[j],
            markerfacecolor="white",
            alpha=1.0,
        )
        ax_rel.fill_between(
            df_plot["circuit.num_gates"],
            df_plot["dur_serial_rel_q25"],
            df_plot["dur_serial_rel_q75"],
            alpha=0.15,
            color=colors[j + 1],
        )

    ax_rel.axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax_rel.set_xlabel("Number of gates")
    ax_rel.set_ylabel("Relative overhead")
    ax_rel.set_ylim(bottom=0)
    ylims = ax_rel.get_ylim()

    if specifier == "5x5":
        ax_rel.set_yticks(np.arange(0, ylims[1], step=1))
        ax_rel.set_yticks(np.arange(0, ylims[1], step=0.5), minor=True)
    elif specifier in ("11x11", "brisbane"):
        ax_rel.set_yticks(np.arange(0, ylims[1], step=2))
        ax_rel.set_yticks(np.arange(0, ylims[1], step=1), minor=True)

    ax_rel.tick_params(which="minor", length=2, color="gray")
    ax_rel.tick_params(axis="both", which="major")

    # ── Third row: Bar plots ───────────────────────────────────────────────────
    x_values = _NUM_GATES_VALUES[specifier]
    qpg_values_ = _QPG_VALUES[specifier]["bar"]
    ax_bar = axd["bars"]

    main_width = 0.6
    branch_width = main_width / len(qpg_values_)
    x_positions = range(len(x_values))

    for k, num_gates in enumerate(x_values):
        base_stack_data = []
        routing_data = []
        serial_data = []

        for qpg in qpg_values_:
            df_inset = df.filter(
                (pl.col("hardware.layout.k") == qpg)
                & (pl.col("hardware.layout.strategy") == layout)
                & (pl.col("circuit.num_gates") == num_gates)
                & (pl.col("hardware.id") == hardware)
            )
            base_stack_data.append(df_inset["T_trans"].median() * 1e3)
            routing_data.append(df_inset["overhead_dur_routing"].median() * 1e3)
            serial_data.append(df_inset["overhead_dur_serial"].median() * 1e3)

        base_duration = base_stack_data[0]
        routing_overhead = routing_data[0]

        ax_bar.bar(
            k,
            base_duration,
            width=main_width,
            color=wong_colors[7],
            edgecolor="black",
            linewidth=0.3,
            label="Translated circuit" if k == 0 else "",
        )
        ax_bar.bar(
            k,
            routing_overhead,
            bottom=base_duration,
            width=main_width,
            color=wong_colors[2],
            edgecolor="black",
            linewidth=0.3,
            label="Routing" if k == 0 else "",
        )

        for j, (qpg, serial_overhead) in enumerate(zip(qpg_values_, serial_data)):
            branch_pos = k + (j - 1) * branch_width
            bars = ax_bar.bar(
                branch_pos,
                serial_overhead,
                width=branch_width,
                bottom=base_duration + routing_overhead,
                color=[wong_colors[5], wong_colors[3], wong_colors[1]][j],
                alpha=1.0,
                hatch=["/", "--", "||", "\\", "-"][j],
                edgecolor="black",
                label=f"Serialization, $k={qpg}$" if k == 0 else "",
                hatch_linewidth=0.5,
            )
            for bar in bars:
                bar._hatch_color = (0.0, 0.0, 0.0, 1.0)

            offset_label = {"5x5": 0.015, "11x11": 0.2, "brisbane": 1.5}
            ax_bar.text(
                branch_pos,
                (base_duration + routing_overhead + serial_overhead) + offset_label[specifier],
                f"{qpg}",
                ha="center",
                va="center",
                fontsize=8,
            )

    ax_bar.set_facecolor("none")
    ax_bar.set_xticks(x_positions)
    ax_bar.set_xticklabels(x_values)
    ax_bar.set_xlabel("Number of gates")
    ax_bar.set_ylabel("Circuit duration (ms)")
    ax_bar.set_ylim(top=1.1 * (base_duration + routing_overhead + max(serial_data)))
    ylims = ax_bar.get_ylim()

    if specifier == "5x5":
        ax_bar.set_yticks(np.arange(0, ylims[1], step=0.2))
        ax_bar.set_yticks(np.arange(0, ylims[1], step=0.1), minor=True)
        ax_bar.set_yticklabels(["0" if t == 0.0 else f"{t:.1f}" for t in ax_bar.get_yticks()])
    elif specifier == "11x11":
        ax_bar.set_yticks(np.arange(0, ylims[1], step=2))
        ax_bar.set_yticks(np.arange(0, ylims[1], step=1), minor=True)
    elif specifier == "brisbane":
        ax_bar.set_yticks(np.arange(0, ylims[1], step=10))
        ax_bar.set_yticks(np.arange(0, ylims[1], step=5), minor=True)

    ax_bar.tick_params(which="minor", length=2, color="gray")
    ax_bar.legend()

    # Inset topology diagram
    inset_ax = ax_abs.inset_axes([0.37, 0.75, 0.2, 0.2])
    _INSET_PLOTTERS[specifier](inset_ax)

    # Subplot captions
    captions = ["(a)", "(b)", "(c)"]
    for ax_i, ax in enumerate([ax_abs, ax_rel, ax_bar]):
        ax.text(
            0,
            1,
            captions[ax_i],
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            va="bottom",
            ha="right",
        )

    return fig


if __name__ == "__main__":
    for specifier in ("5x5", "11x11", "brisbane"):
        fig = plot_figure(specifier)
        plt.show()

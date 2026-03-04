from plot_settings import SINGLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH, wong_colors, RESULTS_DIR

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import matplotlib.ticker as ticker
from plot_utils import plot_grid25, plot_grid121, plot_brisbane

# ============================================================================
# Helper functions
# ============================================================================


def format_ticks(x, pos):
    """Formatter for gate-density twin axis: show positives with one decimal."""
    if x == 0:
        return "0"
    return f"{x:.1f}" if x >= 0 else ""


# ============================================================================
# Figures
# ============================================================================


def plot_figure(specifier: str) -> plt.Figure:
    fig_id, hardware = specifier.split("_", 1)

    circuit_id_order = [
        "bv",
        "graphstate",
        "dj",
        "ghz",
        "wstate",
        "qnn",
        "vqe_real_amp",
        "vqe_su2",
        "cdkm_ripple_carry_adder",
        "full_adder",
        "vbe_ripple_carry_adder",
        "bmw_quark_cardinality",
        "half_adder",
        "qft",
        "qftentangled",
        "qpeexact",
        "qpeinexact",
        "modular_adder",
        "draper_qft_adder",
        "qaoa",
        "bmw_quark_copula",
        "vqe_two_local",
        "hhl",
        "randomcircuit",
        "hrs_cumulative_multiplier",
        "rg_qft_multiplier",
        "shor",
    ]
    order_mapping = {circuit_id: i for i, circuit_id in enumerate(circuit_id_order)}

    df = pl.concat(
        [
            # 11x11 grid data
            pl.read_parquet(RESULTS_DIR / "20260302-224810_mqt_11x11_N2700.parquet"),
            # addition 11x11 grid data for k=[14]
            pl.read_parquet(RESULTS_DIR / "20260303-215921_mqt_11x11_N540.parquet"),
            # brisbane data
            pl.read_parquet(RESULTS_DIR / "20260303-151441_mqt_brisbane_N2700.parquet"),
            # additional brisbane mqt data for k=[13]
            pl.read_parquet(RESULTS_DIR / "20260303-204352_mqt_brisbane_N540.parquet"),
            # 5x5 grid data
            pl.read_parquet(RESULTS_DIR / "20260302-224033_mqt_5x5_N2160.parquet"),
            # additional 5x5 mqt data for k=[3, 5, 9]
            pl.read_parquet(RESULTS_DIR / "20260303-212759_mqt_5x5_N1620.parquet"),
        ]
    ).with_columns(
        (pl.col("T_routed_serial") / pl.col("T_routed")).alias("overhead_serial_rel_routing"),
    )

    if fig_id == "line":
        qpg_values = {
            "brisbane": [2, 3, 8, 26, 127],
            "11x11": [2, 3, 8, 31, 121],
            "5x5": [2, 4, 13, 25],
        }

        colors = wong_colors[1:]
        markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

        df_hardware = df.filter(pl.col("hardware.id") == hardware)
        print("hardware:", hardware)

        fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 4.5), dpi=200, layout="constrained")
        axd = fig.subplot_mosaic(
            [["overhead"], ["rel_overhead"]],
            sharex=True,
            height_ratios=[4, 4],
        )

        abs_overhead_lines = []
        df_plot_agg = None

        for i, qpg in enumerate(reversed(qpg_values[hardware])):
            df_plot_agg = (
                df_hardware.filter(pl.col("hardware.layout.strategy") == "trivial")
                .filter(pl.col("hardware.layout.k") == qpg)
                .group_by("circuit.id")
                .agg(
                    [
                        pl.col("qc_trans_num_gates").min().alias("min_qc_num_gates"),
                        pl.col("T_routed_serial").min().alias("min_serial"),
                        pl.col("T_routed_serial").median().alias("median_serial"),
                        pl.col("T_routed").min().alias("min_routing"),
                        pl.col("T_routed").median().alias("median_routing"),
                        ((pl.col("T_routed_serial")) / pl.col("T_routed"))
                        .median()
                        .alias("rel_overhead_median"),
                        pl.col("qc_trans_rho_1").mean().alias("mean_rho_1"),
                        pl.col("qc_trans_rho_2").mean().alias("mean_rho_2"),
                        pl.col("qc_trans_rho_total").mean().alias("mean_rho"),
                    ]
                )
                .with_columns(
                    pl.col("circuit.id")
                    .map_elements(
                        lambda x: order_mapping.get(x, len(circuit_id_order)),
                        return_dtype=pl.Int32,
                    )
                    .alias("sort_order")
                )
                .sort("sort_order")
            )

            x = df_plot_agg["sort_order"].to_numpy()

            if i == 0:
                axd_twin = axd["rel_overhead"].twinx()
                axd_twin.fill_between(
                    x,
                    df_plot_agg["mean_rho"],
                    alpha=0.2,
                    color="green",
                    label=r"$\rho_\text{total}$",
                    zorder=1,
                )
                axd_twin.fill_between(
                    x,
                    df_plot_agg["mean_rho_2"],
                    alpha=0.2,
                    color="blue",
                    label=r"$\rho_{2}$",
                    zorder=2,
                )
                axd_twin.fill_between(
                    x,
                    df_plot_agg["mean_rho_1"],
                    alpha=0.2,
                    color="red",
                    label=r"$\rho_{1}$",
                    zorder=3,
                )

            axd["rel_overhead"].plot(
                x,
                df_plot_agg["rel_overhead_median"],
                marker=markers[i],
                linestyle="",
                color=colors[i],
                alpha=1.0,
                label=f"{qpg} qubits per switch",
                zorder=10,
                markerfacecolor="None",
            )

            (line,) = axd["overhead"].plot(
                x,
                df_plot_agg["median_serial"],
                marker=markers[i],
                linestyle="",
                color=colors[i],
                alpha=0.9,
                label=f"{qpg} qubits per switch",
                markerfacecolor="None",
                zorder=5,
            )
            abs_overhead_lines.append(line)

        # Add routing baseline (use last df_plot_agg, same routing for all qpg)
        (routing_baseline,) = axd["overhead"].plot(
            x,
            df_plot_agg["min_routing"],
            marker="x",
            linestyle="--",
            color="black",
            label="Routed",
        )

        # --- Axes configuration ---
        axd["rel_overhead"].set_ylim(bottom=0)

        axd_twin.set_ylabel("Gate densities", color="black")
        axd_twin.tick_params(axis="y", labelcolor="black")
        axd_twin.set_zorder(0)
        axd["rel_overhead"].set_zorder(10)
        axd["rel_overhead"].patch.set_visible(False)

        # Align twin y-axis so that y=1 on rel_overhead matches y=0 on twin
        y1_min, y1_max = axd["rel_overhead"].get_ylim()
        target_position = (1.0 - y1_min) / (y1_max - y1_min)
        desired_y2_max = (
            df.filter(pl.col("hardware.id") == hardware)[
                "qc_trans_rho_1", "qc_trans_rho_2", "qc_trans_rho_total"
            ]
            .max()
            .to_numpy()
            .max()
            * 1.1
        )
        new_y2_min = -target_position * desired_y2_max / (1 - target_position)
        axd_twin.set_ylim(new_y2_min, desired_y2_max)

        axd_twin.set_yticks(np.arange(0, desired_y2_max, step=0.2))
        axd_twin.set_yticks(np.arange(0, desired_y2_max, step=0.1), minor=True)
        axd_twin.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

        # --- Legends ---
        leg1 = axd["overhead"].legend(handles=abs_overhead_lines, title="Serialized and routed")
        fig.canvas.draw()
        bbox_leg1 = leg1.get_tightbbox(fig.canvas.get_renderer())
        bbox_leg1_fig = bbox_leg1.transformed(fig.transFigure.inverted())
        _ = axd["overhead"].legend(
            handles=[routing_baseline],
            title="",
            loc="upper left",
            bbox_to_anchor=(bbox_leg1_fig.x1 + 0.01, 1.0),
        )
        axd["overhead"].add_artist(leg1)

        axd["overhead"].semilogy()
        axd["overhead"].set_ylabel("Circuit duration (s)")

        axd["rel_overhead"].set_ylabel("Relative overhead")
        axd["rel_overhead"].axhline(y=1, color="black", linestyle="--", linewidth=1, alpha=0.7)

        leg_rel_1 = axd["rel_overhead"].legend(
            handles=axd["rel_overhead"].get_legend_handles_labels()[0],
            labels=axd["rel_overhead"].get_legend_handles_labels()[1],
            loc="upper left",
            ncols=2,
            bbox_to_anchor=(0.085, 1.0),
        )
        leg_rel_1.set_zorder(110)
        _ = axd["rel_overhead"].legend(
            handles=axd_twin.get_legend_handles_labels()[0],
            labels=axd_twin.get_legend_handles_labels()[1],
            loc="upper right",
        )
        axd["rel_overhead"].add_artist(leg_rel_1)

        ylims = axd["rel_overhead"].get_ylim()
        axd["rel_overhead"].set_yticks(np.arange(1, ylims[1], step=1 if hardware == "5x5" else 2))
        axd["rel_overhead"].set_yticks(
            np.arange(0, ylims[1], step=0.5 if hardware == "5x5" else 1), minor=True
        )
        axd["rel_overhead"].set_yticklabels(
            [int(tick) for tick in axd["rel_overhead"].get_yticks()]
        )
        axd["rel_overhead"].tick_params(which="minor", length=2, color="gray")

        axd["rel_overhead"].set_xticks(range(len(circuit_id_order)))
        axd["rel_overhead"].set_xticklabels(circuit_id_order, rotation=45, ha="right")

        axd["overhead"].xaxis.grid(True)
        axd["rel_overhead"].xaxis.grid(True)

        if hardware == "11x11":
            inset_ax = axd["overhead"].inset_axes([0.80, 0.05, 0.3, 0.3])
            inset_ax._transparent_bg = True

            plot_grid121(inset_ax)

        elif hardware == "brisbane":
            inset_ax = axd["overhead"].inset_axes([0.78, 0.05, 0.3, 0.3])
            plot_brisbane(inset_ax)
        elif hardware == "5x5":
            inset_ax = axd["overhead"].inset_axes([0.80, 0.05, 0.3, 0.3])
            plot_grid25(inset_ax)

        caption_pos = (-0.073, 1.0)
        axd["overhead"].text(
            *caption_pos,
            "(a)",
            transform=axd["overhead"].transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )
        axd["rel_overhead"].text(
            *caption_pos,
            "(b)",
            transform=axd["rel_overhead"].transAxes,
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

        return fig

    elif fig_id == "hist":
        qpg_values = {
            "11x11": [2, 3, 8, 14, 31, 121],
            "brisbane": [2, 3, 8, 13, 26, 127],
            "5x5": [2, 3, 4, 5, 9, 25],
        }
        cutoff = {"11x11": 9, "brisbane": 9, "5x5": 4}

        fig, axes = plt.subplots(
            figsize=(SINGLE_COLUMN_WIDTH, 2.5),
            nrows=3,
            ncols=2,
            dpi=150,
            layout="constrained",
            sharey=True,
            sharex=True,
        )

        filtered_df = df.filter(
            (pl.col("hardware.layout.strategy") == "trivial") & (pl.col("hardware.id") == hardware)
        )
        print("hardware:", hardware)
        print("total counts:", len(filtered_df))

        counts_over_cutoff = (
            df.filter(
                (pl.col("overhead_serial_rel_routing") > cutoff[hardware])
                & (pl.col("hardware.layout.strategy") == "trivial")
                & (pl.col("hardware.id") == hardware)
            )
            .group_by("hardware.layout.k")
            .agg(pl.len())
            .sort("hardware.layout.k")
        )
        counts_over_cutoff_dict = dict(
            counts_over_cutoff.select("hardware.layout.k", "len").iter_rows()
        )

        bin_range = (1, cutoff[hardware])
        num_bins = 6 if hardware == "5x5" else bin_range[1] - bin_range[0]

        for i, qpg in enumerate(qpg_values[hardware]):
            plot_df = filtered_df.filter(pl.col("hardware.layout.k") == qpg)

            ax = axes.flatten()[i]
            ax.hist(
                plot_df["overhead_serial_rel_routing"],
                bins=num_bins,
                range=bin_range,
                alpha=1.0,
                color=wong_colors[i + 1],
                label=f"Qubits per group: {qpg}",
            )
            ax.set_xlim(bin_range)
            ylims = ax.get_ylim()

            if hardware == "5x5":
                ax.set_xticks([1, 2, 3])
                ax.set_xticks([1.5, 2.5], minor=True)
                ax.set_yticks(np.arange(0, ylims[1], step=250))
                ax.set_yticks(np.arange(0, ylims[1], step=125), minor=True)
                ax.set_yticklabels(
                    [int(tick) if i % 1 == 0 else "" for i, tick in enumerate(ax.get_yticks())]
                )
            else:
                ax.set_xticks(np.arange(bin_range[0], bin_range[1], step=2))
                ax.set_xticks(np.arange(bin_range[0], bin_range[1], step=1), minor=True)
                ax.set_yticks(np.arange(0, ylims[1], step=200))
                ax.set_yticks(np.arange(0, ylims[1], step=100), minor=True)
                ax.set_yticklabels(
                    [int(tick) if i % 1 == 0 else "" for i, tick in enumerate(ax.get_yticks())]
                )

            if i % 2 == 0:
                ax.set_ylabel("Counts")
            if i >= 4:
                ax.set_xlabel("Relative overhead")

            ax.annotate(
                f"{qpg} qubits per switch",
                xy=(0.5, 0.9),
                xycoords="axes fraction",
                ha="center",
                va="top",
            )

            num_over_cutoff = counts_over_cutoff_dict.get(qpg)
            str_over_cutoff = "+" + str(num_over_cutoff) if num_over_cutoff is not None else ""
            ax.annotate(
                f"{str_over_cutoff}",
                xy=(0.99, 0.08),
                xycoords="axes fraction",
                ha="right",
            )
            ax.text(
                -0.0,
                1.0,
                "(" + chr(97 + i) + ")",
                transform=ax.transAxes,
                fontsize=10,
                fontweight="bold",
                va="bottom",
                ha="right",
            )

        return fig

    else:
        raise ValueError(f"Unknown figure id: {fig_id!r}. Expected 'line' or 'hist'.")


# ============================================================================
# Detail figures
# ============================================================================

_DETAIL_CIRCUIT_ID_ORDER = [
    "bv",
    "graphstate",
    "dj",
    "ghz",
    "wstate",
    "qnn",
    "vqe_real_amp",
    "vqe_su2",
    "cdkm_ripple_carry_adder",
    "full_adder",
    "vbe_ripple_carry_adder",
    "bmw_quark_cardinality",
    "half_adder",
    "qft",
    "qftentangled",
    "qpeexact",
    "qpeinexact",
    "modular_adder",
    "draper_qft_adder",
    "qaoa",
    "bmw_quark_copula",
    "vqe_two_local",
    "hhl",
    "randomcircuit",
    "hrs_cumulative_multiplier",
    "rg_qft_multiplier",
    "shor",
]
_DETAIL_ORDER_MAPPING = {cid: i for i, cid in enumerate(_DETAIL_CIRCUIT_ID_ORDER)}


def _load_detail_df() -> pl.DataFrame:
    return pl.concat(
        [
            pl.read_parquet(RESULTS_DIR / "20260302-224810_mqt_11x11_N2700.parquet"),
            pl.read_parquet(RESULTS_DIR / "20260303-215921_mqt_11x11_N540.parquet"),
            pl.read_parquet(RESULTS_DIR / "20260304-122936_mqt_11x11_N540.parquet"),
            pl.read_parquet(RESULTS_DIR / "20260303-151441_mqt_brisbane_N2700.parquet"),
            pl.read_parquet(RESULTS_DIR / "20260303-204352_mqt_brisbane_N540.parquet"),
            pl.read_parquet(RESULTS_DIR / "20260304-125224_mqt_brisbane_N540.parquet"),
            pl.read_parquet(RESULTS_DIR / "20260302-224033_mqt_5x5_N2160.parquet"),
            pl.read_parquet(RESULTS_DIR / "20260303-212759_mqt_5x5_N1620.parquet"),
        ]
    )


def _plot_detail_ax(
    ax: plt.Axes,
    df: pl.DataFrame,
    hardware: str,
    qpg_values: list[int],
    caption: str,
) -> None:
    """Populate one subplot of a detail figure."""
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    df_hardware = df.filter(
        (pl.col("hardware.layout.strategy") == "trivial") & (pl.col("hardware.id") == hardware)
    )

    df_plot = (
        df_hardware.group_by("circuit.id", "hardware.layout.k").agg(
            min_qc_num_gates=pl.col("qc_trans_num_gates").min(),
            serial_overhead=(pl.col("T_routed_serial") - pl.col("T_routed")).median(),
        )
    ).sort("min_qc_num_gates", "hardware.layout.k")

    for i, qpg in enumerate(qpg_values):
        df_qpg = (
            df_plot.filter(pl.col("hardware.layout.k") == qpg)
            .with_columns(
                pl.col("circuit.id")
                .map_elements(
                    lambda x: _DETAIL_ORDER_MAPPING.get(x, len(_DETAIL_CIRCUIT_ID_ORDER)),
                    return_dtype=pl.Int32,
                )
                .alias("sort_order")
            )
            .sort("sort_order")
        )
        ax.semilogy(
            df_qpg["sort_order"].to_numpy(),
            df_qpg["serial_overhead"],
            color=wong_colors[i + 1],
            marker=markers[i],
            markersize=3,
            linestyle="",
            label=f"{qpg} qubits per switch",
            alpha=0.9,
            zorder=-i,
        )

    ax.set_xticks(range(len(_DETAIL_CIRCUIT_ID_ORDER)))
    ax.set_xticklabels(
        [k if k != "shor" else "shor_{58q,18q}" for k in _DETAIL_CIRCUIT_ID_ORDER],
        rotation=30,
        ha="right",
        va="top",
    )

    ax.set_ylabel("Overhead (s)")
    ax.legend(loc="lower right")
    ax.grid(True, which="both", axis="x")

    ax.text(
        0,
        1,
        caption,
        transform=ax.transAxes,
        fontweight="bold",
        fontsize=10,
        ha="right",
        va="bottom",
    )

    if hardware == "11x11":
        ax.set_ylim(top=0.3)
        inset_ax = ax.inset_axes([-0.07, 0.55, 0.4, 0.4])
        plot_grid121(inset_ax)
    elif hardware == "5x5":
        inset_ax = ax.inset_axes([-0.07, 0.55, 0.4, 0.4])
        plot_grid25(inset_ax)
    elif hardware == "brisbane":
        inset_ax = ax.inset_axes([-0.07, 0.55, 0.4, 0.4])
        plot_brisbane(inset_ax)


def plot_detail_grid() -> plt.Figure:
    """Detail figure: 11x11 grid (top) and 5x5 grid (bottom)."""
    df = _load_detail_df()

    fig, axes = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 1.01),
        dpi=200,
        layout="constrained",
        nrows=2,
        sharex=True,
    )
    axes = axes.flatten()

    for hx, (hardware, qpg_values) in enumerate([("11x11", [2, 4]), ("5x5", [2, 4])]):
        _plot_detail_ax(
            axes[hx],
            df,
            hardware,
            qpg_values,
            caption=f"({'ab'[hx]})",
        )

    return fig


def plot_detail_brisbane() -> plt.Figure:
    """Detail figure: Brisbane hardware only."""
    df = _load_detail_df()

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH),
        dpi=200,
        layout="constrained",
    )

    _plot_detail_ax(ax, df, "brisbane", [2, 4], caption="(a)")

    return fig


if __name__ == "__main__":
    for specifier in (
        "line_11x11",
        "line_brisbane",
        "line_5x5",
        "hist_11x11",
        "hist_brisbane",
        "hist_5x5",
    ):
        fig = plot_figure(specifier)

    fig = plot_detail_grid()

    fig = plot_detail_brisbane()
    plt.show()

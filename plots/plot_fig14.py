from plot_settings import (
    DOUBLE_COLUMN_WIDTH,
    wong_colors,
    ROOT as base_path,
)

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import matplotlib.ticker as ticker

df = pl.read_parquet(base_path / "results" / "20260302-224033_mqt_5x5_N2160.parquet")

df = df.with_columns(
    (pl.col("T_routed_serial") / pl.col("T_routed")).alias("overhead_serial_rel_routing"),
)

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

qpg_values = {
    "ibm_eagle_127": [2, 3, 8, 26, 127],
    "11x11": [2, 3, 8, 31, 121],
    "5x5": [2, 4, 13, 25],
}

hardwares = df["hardware.id"].unique().sort().to_list()
for hardware in hardwares:
    print("hardware:", hardware)
    df_hardware = df.filter(pl.col("hardware.id") == hardware)

    colors = wong_colors[1:]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "h", "*"]

    fig = plt.figure(figsize=(DOUBLE_COLUMN_WIDTH, 4.5), dpi=200, layout="constrained")
    axd = fig.subplot_mosaic(
        [["overhead"], ["rel_overhead"]],
        sharex=True,
        height_ratios=[4, 4],
    )

    abs_overhead_lines = []

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
                    lambda x: order_mapping.get(x, len(circuit_id_order)), return_dtype=pl.Int32
                )
                .alias("sort_order")
            )
            .sort("sort_order")
        )

        if i == 0:
            axd_twin = axd["rel_overhead"].twinx()
            axd_twin.fill_between(
                df_plot_agg["circuit.id"],
                df_plot_agg["mean_rho"],
                alpha=0.2,
                color="green",
                label=r"$\rho_\text{total}$",
                zorder=1,
            )
            axd_twin.fill_between(
                df_plot_agg["circuit.id"],
                df_plot_agg["mean_rho_2"],
                alpha=0.2,
                color="blue",
                label=r"$\rho_{2}$",
                zorder=2,
            )
            axd_twin.fill_between(
                df_plot_agg["circuit.id"],
                df_plot_agg["mean_rho_1"],
                alpha=0.2,
                color="red",
                label=r"$\rho_{1}$",
                zorder=3,
            )

        axd["rel_overhead"].plot(
            df_plot_agg["circuit.id"],
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
            df_plot_agg["circuit.id"],
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
        df_plot_agg["circuit.id"],
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

    def format_ticks(x, pos):
        if x == 0:
            return "0"
        return f"{x:.1f}" if x >= 0 else ""

    axd_twin.set_yticks(np.arange(0, desired_y2_max, step=0.2))
    axd_twin.set_yticks(np.arange(0, desired_y2_max, step=0.1), minor=True)
    axd_twin.yaxis.set_major_formatter(ticker.FuncFormatter(format_ticks))

    # --- Legends ---
    leg1 = axd["overhead"].legend(handles=abs_overhead_lines, title="Serialized and routed")
    fig.canvas.draw()
    bbox_leg1 = leg1.get_tightbbox(fig.canvas.get_renderer())
    bbox_leg1_fig = bbox_leg1.transformed(fig.transFigure.inverted())
    leg2 = axd["overhead"].legend(
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
    leg_twin = axd["rel_overhead"].legend(
        handles=axd_twin.get_legend_handles_labels()[0],
        labels=axd_twin.get_legend_handles_labels()[1],
        loc="upper right",
    )
    axd["rel_overhead"].add_artist(leg_rel_1)

    ylims = axd["rel_overhead"].get_ylim()
    axd["rel_overhead"].set_yticks(np.arange(1, ylims[1], step=1 if hardware == "grid25" else 2))
    axd["rel_overhead"].set_yticks(
        np.arange(0, ylims[1], step=0.5 if hardware == "grid25" else 1), minor=True
    )
    axd["rel_overhead"].set_yticklabels([int(tick) for tick in axd["rel_overhead"].get_yticks()])
    axd["rel_overhead"].tick_params(which="minor", length=2, color="gray")

    axd["rel_overhead"].set_xticks(range(len(circuit_id_order)))
    axd["rel_overhead"].set_xticklabels(circuit_id_order, rotation=45, ha="right")

    axd["overhead"].xaxis.grid(True)
    axd["rel_overhead"].xaxis.grid(True)

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

    plt.show()

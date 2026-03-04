import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from plot_settings import SINGLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH, wong_colors, ROOT


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(
        ROOT / "results" / "20250903-115331_grid121_mqt_tsw_select_N39690.parquet"
    )
    return df.rename(
        {
            "duration_two_qubit_gate": "T2",
            "duration_single_qubit_gate": "T1",
            "duration_switch": "Tsw",
        }
    )


def plot_figure() -> plt.Figure:
    benches_to_viz = ["shor", "qaoa", "graphstate"]
    Tsw_selected = [1e-07, 1.6e-08, 2e-09]
    line_styles = ["-", "--", ":", "-."]
    captions = ["(a)", "(b)", "(c)"]
    caption_pos = (-0.2, 1.0)

    df = load_data()

    fig, ax = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.7),
        dpi=200,
        layout="constrained",
        nrows=1,
        ncols=3,
        gridspec_kw={"wspace": 0.1},
    )
    axes = ax.flatten()

    for i, bench in enumerate(benches_to_viz):
        for Tsw in Tsw_selected:
            df_plot = (
                df.filter(pl.col("circuit_id") == bench)
                .filter(pl.col("Tsw") == Tsw)
                .group_by("qpg")
                .agg(
                    overhead_median=((pl.col("T_routed_serial")) / pl.col("T_routed")).median(),
                    overhead_q25=((pl.col("T_routed_serial")) / pl.col("T_routed")).quantile(0.25),
                    overhead_q75=((pl.col("T_routed_serial")) / pl.col("T_routed")).quantile(0.75),
                )
            ).sort("qpg")

            ls = line_styles[Tsw_selected.index(Tsw) % len(line_styles)]
            axes[i].semilogx(
                df_plot["qpg"],
                df_plot["overhead_median"],
                marker=".",
                linestyle=ls,
                label=f"{Tsw * 1e9:.0f}",
                markerfacecolor="black",
                color="black",
            )
            axes[i].fill_between(
                df_plot["qpg"],
                df_plot["overhead_q25"],
                df_plot["overhead_q75"],
                alpha=0.2,
                color="gray",
                linestyle=ls,
            )

        axes[i].set_xlabel("Qubits per switch")
        axes[i].set_ylabel("Relative overhead")
        axes[i].legend(bbox_to_anchor=(0.03, 1), loc="upper left", title=r"Switch time (ns)")

        if bench == "shor":
            axes[i].set_ylim(bottom=0.8)
            ylims = axes[i].get_ylim()
            axes[i].set_yticks(np.arange(1, ylims[1], step=0.5))
            axes[i].set_yticks(np.arange(1, ylims[1], step=0.25), minor=True)
            axes[i].set_yticklabels(
                [f"{tick:.1f}" if tick != 0 else "0" for tick in axes[i].get_yticks()]
            )
        elif bench == "qaoa":
            axes[i].set_ylim(bottom=-0.2)
            ylims = axes[i].get_ylim()
            axes[i].set_yticks(np.arange(1, ylims[1], step=2))
            axes[i].set_yticks(np.arange(1, ylims[1], step=1), minor=True)
            axes[i].set_yticklabels([f"{tick:.0f}" for tick in axes[i].get_yticks()])
        elif bench == "graphstate":
            axes[i].set_ylim(bottom=-1)
            ylims = axes[i].get_ylim()
            axes[i].set_yticks(np.arange(1, ylims[1], step=4))
            axes[i].set_yticks(np.arange(1, ylims[1], step=1), minor=True)
            axes[i].set_yticklabels([f"{tick:.0f}" for tick in axes[i].get_yticks()])

        inset_ax = axes[i].inset_axes([0.59, 0.15, 0.3, 0.18])

        for qx, qpg in enumerate([2, 10, 121]):
            color = wong_colors[(qx + 1) % len(wong_colors)]
            axes[i].axvline(x=qpg, color=color, linestyle="--", alpha=0.8)

            df_inset = (
                df.filter(pl.col("circuit_id") == bench)
                .filter(pl.col("qpg") == qpg)
                .group_by("Tsw")
                .agg(
                    overhead_median=((pl.col("T_routed_serial")) / pl.col("T_routed")).median(),
                    overhead_q25=((pl.col("T_routed_serial")) / pl.col("T_routed")).quantile(0.25),
                    overhead_q75=((pl.col("T_routed_serial")) / pl.col("T_routed")).quantile(0.75),
                )
            ).sort("Tsw")
            inset_ax.plot(
                df_inset["Tsw"] * 1e9,
                df_inset["overhead_median"],
                marker=".",
                linestyle="-",
                label=f"{qpg}",
                color=color,
                markerfacecolor=color,
                markeredgecolor="None",
            )
            inset_ax.fill_between(
                df_inset["Tsw"] * 1e9,
                df_inset["overhead_q25"],
                df_inset["overhead_q75"],
                alpha=0.2,
                color=color,
            )

        inset_ax.set_xlabel("Switch time (ns)", labelpad=0, fontsize=8)
        inset_ax.set_xticks([0, 100, 200])
        if bench == "shor":
            inset_ax.set_ylim(bottom=0.7, top=3.5)
            inset_ax.set_yticks([1, 2, 3])
        elif bench == "qaoa":
            inset_ax.set_ylim(bottom=0, top=12)
            inset_ax.set_yticks([1, 6, 11])
        elif bench == "graphstate":
            inset_ax.set_ylim(bottom=-0.5, top=20)
            inset_ax.set_yticks([1, 9, 17])

        axes[i].text(
            *caption_pos,
            captions[i],
            transform=axes[i].transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
            ha="left",
        )

    return fig


if __name__ == "__main__":
    fig = plot_figure()
    plt.show()

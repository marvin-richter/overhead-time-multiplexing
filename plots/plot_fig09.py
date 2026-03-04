import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from plot_settings import SINGLE_COLUMN_WIDTH, DOUBLE_COLUMN_WIDTH, wong_colors, ROOT


def label_number(T2: float) -> str:
    if T2 == 10e-9:
        return f"{T2 / 20e-9:.1f}"
    if T2 == 20e-9:
        return ""
    return f"{T2 / 20e-9:.0f}"


def load_data() -> pl.DataFrame:
    df = pl.read_parquet(ROOT / "results" / "20250827-104539_grid121_mqt_t2_select_N39690.parquet")
    df = df.rename({"duration_two_qubit_gate": "T2", "duration_single_qubit_gate": "T1"})
    df = df.with_columns(
        qc_routed_rho_1=pl.col("qc_routed_n1")
        / pl.col("qc_routed_depth")
        / pl.col("qc_routed_num_qubits"),
        qc_routed_rho_2=2
        * pl.col("qc_routed_n2")
        / pl.col("qc_routed_depth")
        / pl.col("qc_routed_num_qubits"),
    )
    return df


def plot_figure() -> plt.Figure:
    benches_to_viz = ["shor", "qaoa", "graphstate"]
    T2_selected = [10e-9, 20e-9, 80e-9, 200e-9, 400e-9]
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
    insets = []

    for i, bench in enumerate(benches_to_viz):
        for T2 in T2_selected:
            df_plot = (
                df.filter(pl.col("circuit_id") == bench)
                .filter(pl.col("T2") == T2)
                .group_by("qpg")
                .agg(
                    overhead_median=((pl.col("T_routed_serial")) / pl.col("T_routed")).median(),
                    overhead_q25=((pl.col("T_routed_serial")) / pl.col("T_routed")).quantile(0.25),
                    overhead_q75=((pl.col("T_routed_serial")) / pl.col("T_routed")).quantile(0.75),
                )
            ).sort("qpg")

            axes[i].plot(
                df_plot["qpg"],
                df_plot["overhead_median"],
                marker=".",
                linestyle="-",
                label=f"$t_\\text{{2q}}$={label_number(T2)}$t_\\text{{1q}}$",
                color=wong_colors[T2_selected.index(T2) + 1 % len(wong_colors)],
            )

        axes[i].set_xlabel("Qubits per switch")
        ylabel_obj = axes[i].set_ylabel("Relative overhead")
        axes[i].legend(bbox_to_anchor=(0, 1), loc="upper left")

        ylims = axes[i].get_ylim()
        if bench == "shor":
            axes[i].set_yticks(np.arange(1, ylims[1], step=0.5))
            axes[i].set_yticks(np.arange(1, ylims[1], step=0.25), minor=True)
            axes[i].set_yticklabels(
                [f"{tick:.1f}" if tick != 0 else "0" for tick in axes[i].get_yticks()]
            )
        elif bench == "qaoa":
            axes[i].set_yticks(np.arange(1, ylims[1], step=2))
            axes[i].set_yticks(np.arange(1, ylims[1], step=1), minor=True)
            axes[i].set_yticklabels([f"{tick:.0f}" for tick in axes[i].get_yticks()])
        elif bench == "graphstate":
            axes[i].set_yticks([1] + list(np.arange(5, ylims[1], step=5)))
            axes[i].set_yticks(np.arange(1, ylims[1], step=1), minor=True)
            axes[i].set_yticklabels([f"{tick:.0f}" for tick in axes[i].get_yticks()])

        inset_ax = axes[i].inset_axes([0.7, 0.1, 0.2, 0.2])
        insets.append(inset_ax)
        inset_ax.bar(
            1, df.filter(pl.col("circuit_id") == bench)["qc_routed_rho_1"].mean(), color="darkgray"
        )
        inset_ax.bar(
            2, df.filter(pl.col("circuit_id") == bench)["qc_routed_rho_2"].mean(), color="darkgray"
        )
        inset_ax.set_xticks([1, 2])
        inset_ax.set_xticklabels([r"$\rho_1$", r"$\rho_2$"])

        axes[i].semilogx()

        fig.canvas.draw()
        ylabel_bbox = ylabel_obj.get_window_extent(renderer=fig.canvas.get_renderer())
        _ = fig.transFigure.inverted().transform(ylabel_bbox)[0, 0]

        axes[i].text(
            *caption_pos,
            captions[i],
            transform=axes[i].transAxes,
            fontweight="bold",
            fontsize=10,
            va="top",
            ha="left",
        )

    for inset_ax in insets[:-1]:
        inset_ax.sharey(insets[2])
        ylims = inset_ax.get_ylim()
        inset_ax.set_yticks(np.arange(0, ylims[1], step=0.05))
        inset_ax.set_yticks(np.arange(0, ylims[1], step=0.025), minor=True)
        inset_ax.set_yticklabels(
            [f"{tick:.2f}" if tick != 0 else "0" for tick in inset_ax.get_yticks()]
        )

    return fig


if __name__ == "__main__":
    fig = plot_figure()
    plt.show()

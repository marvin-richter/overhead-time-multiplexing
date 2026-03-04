import matplotlib.pyplot as plt
import polars as pl
from polars import col
from plot_settings import SINGLE_COLUMN_WIDTH, wong_colors, RESULTS_DIR


def make_box_plot(ax, df, qpg, num_gates):
    df_plot = df.filter(
        (pl.col("hardware.id") == "11x11")
        & (pl.col("hardware.layout.strategy") == "trivial")
        & (pl.col("hardware.layout.k") == qpg)
        & (pl.col("circuit.num_gates") == num_gates)
    ).with_columns(overhead_serial=(col("T_routed_serial") - col("T_routed")) * 1e3)

    data = [
        df_plot.filter(
            (pl.col("serialization.delay_check") == dc)
            & (pl.col("serialization.topord_method") == tm)
        )["overhead_serial"]
        for tm in ["default", "prio_two"]
        for dc in [False, True]
    ]

    box_plot = ax.boxplot(data, patch_artist=True)

    for patch, color in zip(box_plot["boxes"], [wong_colors[i] for i in range(1, 5)]):
        patch.set_facecolor(color)
    for median in box_plot["medians"]:
        median.set_color("black")


def plot_figure() -> plt.Figure:
    df = pl.read_parquet(RESULTS_DIR / "20260303-222726_opt_compare_11x11_N1600.parquet")

    qpgA = 13
    qpgB = 121
    num_gatesA = 1000
    num_gatesB = 10000

    fig, axes = plt.subplot_mosaic(
        mosaic=[["box1", "box2"], ["box3", "box4"]],
        figsize=(SINGLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.8),
        layout="constrained",
        dpi=200,
        sharex=True,
        gridspec_kw={"hspace": 0.1, "wspace": 0.1},
    )

    for ax_key, qpg, num_gates in [
        ("box1", qpgA, num_gatesA),
        ("box2", qpgA, num_gatesB),
        ("box3", qpgB, num_gatesA),
        ("box4", qpgB, num_gatesB),
    ]:
        make_box_plot(axes[ax_key], df, qpg, num_gates)

    axes["box3"].set_yticks([0.4, 0.5, 0.6, 0.7])
    axes["box3"].set_yticks([0.45, 0.55, 0.65], minor=True)

    handles = [plt.Line2D([0], [0], color=wong_colors[i], lw=4) for i in range(1, 5)]
    labels = [
        "1 Default order",
        "2 Default order + delay removal",
        "3 Adj. order",
        "4 Adj. order + delay removal",
    ]
    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        fontsize=7,
    )
    fig.text(
        -0.03,
        0.5,
        "Serialization overhead (\u03bcs)",
        rotation="vertical",
        va="center",
        ha="center",
    )

    axes["box1"].set_ylim((axes["box1"].get_ylim()[0], axes["box1"].get_ylim()[1] * 1.135))
    axes["box3"].set_ylim((axes["box3"].get_ylim()[0], axes["box3"].get_ylim()[1] * 1.1))

    pos_gates = (0.95, 0.86)
    axes["box1"].annotate(
        "$10^3$ gates", xy=pos_gates, xycoords="axes fraction", va="top", ha="right"
    )
    axes["box2"].annotate(
        "$10^4$ gates", xy=pos_gates, xycoords="axes fraction", va="top", ha="right"
    )
    axes["box3"].annotate(
        "$10^3$ gates", xy=pos_gates, xycoords="axes fraction", va="top", ha="right"
    )
    axes["box4"].annotate(
        "$10^4$ gates", xy=pos_gates, xycoords="axes fraction", va="top", ha="right"
    )

    pos_qpg = (0.95, 0.95)
    axes["box1"].annotate(
        f"{qpgA} q/s", xy=pos_qpg, xycoords="axes fraction", va="top", ha="right"
    )
    axes["box2"].annotate(
        f"{qpgA} q/s", xy=pos_qpg, xycoords="axes fraction", va="top", ha="right"
    )
    axes["box3"].annotate(
        f"{qpgB} q/s", xy=pos_qpg, xycoords="axes fraction", va="top", ha="right"
    )
    axes["box4"].annotate(
        f"{qpgB} q/s", xy=pos_qpg, xycoords="axes fraction", va="top", ha="right"
    )

    caption_pos = (-0.33, 0.95)
    axes["box1"].text(
        *caption_pos, "(a)", transform=axes["box1"].transAxes, fontweight="bold", fontsize=10
    )
    axes["box2"].text(
        *caption_pos, "(b)", transform=axes["box2"].transAxes, fontweight="bold", fontsize=10
    )
    axes["box3"].text(
        *caption_pos, "(c)", transform=axes["box3"].transAxes, fontweight="bold", fontsize=10
    )
    axes["box4"].text(
        *caption_pos, "(d)", transform=axes["box4"].transAxes, fontweight="bold", fontsize=10
    )

    return fig


if __name__ == "__main__":
    fig = plot_figure()
    plt.show()

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.legend_handler import HandlerTuple

from plot_settings import DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH, wong_colors, ROOT as base_path


# ============================================================================
# Helper functions
# ============================================================================


def fit_power_law(x, y, y_lower, y_upper, x_min=None, n_samples=None):
    """Fit y = a * x^b in log-space with IQR-based weighting."""
    mask = (x > 0) & (y > 0) & (y_lower > 0) & (y_upper > 0)
    if x_min is not None:
        mask &= x >= x_min

    x, y = x[mask], y[mask]
    y_lower, y_upper = y_lower[mask], y_upper[mask]
    n = np.array(n_samples)[mask] if n_samples is not None else np.full(len(x), 30)

    iqr = y_upper - y_lower
    log_y_err = np.maximum((iqr / 1.35) / (np.sqrt(n) * y * np.log(10)), 1e-10)
    weights = np.clip(1 / log_y_err**2, 1e-10, 1e10)

    try:
        coeffs, cov = np.polyfit(np.log10(x), np.log10(y), 1, w=weights, cov=True)
    except np.linalg.LinAlgError:
        coeffs, cov = np.polyfit(np.log10(x), np.log10(y), 1, cov=True)

    b, log_a = coeffs
    b_err, log_a_err = np.sqrt(np.diag(cov))
    a = 10**log_a
    a_err = a * np.log(10) * log_a_err

    x_fit = np.logspace(np.log10(x.min()), np.log10(x.max()), 100)
    return a, b, a_err, b_err, x_fit, a * x_fit**b


def plot_with_fit(ax, df_k, metric, color, marker, label, x_min=25):
    """Plot median ± IQR with power-law fit; return handle tuple and (a, b)."""
    x = df_k["hardware.num_qubits"].to_numpy()
    y = df_k[f"{metric}_median"].to_numpy()
    y_lo = df_k[f"{metric}_q25"].to_numpy()
    y_hi = df_k[f"{metric}_q75"].to_numpy()
    n = df_k["count"].to_numpy()

    h1 = ax.errorbar(
        x,
        y,
        yerr=[y - y_lo, y_hi - y],
        color=color,
        marker=marker,
        linestyle="",
        capsize=2,
        capthick=1,
        elinewidth=1,
    )

    a, b, a_err, b_err, x_fit, y_fit = fit_power_law(x, y, y_lo, y_hi, x_min=x_min, n_samples=n)
    print(f"{label}: a={a:.2e}±{a_err:.2e}, b={b:.3f}±{b_err:.3f}")

    (h2,) = ax.plot(x_fit, y_fit, "--", color=color, alpha=0.7)
    return (h1, h2), (a, b)


def add_grid_markers(ax, y_pos=-0.05):
    """Add vertical lines with below-axis annotations for 5×5 and 11×11 grid sizes."""
    ax.annotate(
        r"    5$\times$5",
        xy=(25, 1.0),
        xytext=(25, y_pos),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )
    ax.annotate(
        r"11$\times$11",
        xy=(121, 1.0),
        xytext=(121, y_pos - 0.04),
        xycoords=("data", "axes fraction"),
        textcoords=("data", "axes fraction"),
        ha="center",
        va="top",
        fontsize=8,
        arrowprops=dict(arrowstyle="-", color="black", lw=0.8),
    )


# ============================================================================
# Figure
# ============================================================================


def plot_figure() -> plt.Figure:
    # Parameters
    fit_x_min = 121
    k_values = [8, 4, 2]
    colors = [wong_colors[1], wong_colors[2], wong_colors[3]]
    markers = ["^", "D", "v"]

    # Data loading
    df = pl.read_parquet(base_path / "results" / "20260122-175216_grids_qft_N600.parquet")
    df = df.filter(pl.col("hardware.num_qubits") > 9)

    df = df.with_columns(
        DT_serial=(pl.col("T_routed_serial") - pl.col("T_routed")),
        DT_routed=(pl.col("T_routed") - pl.col("T_trans")),
        R=pl.col("T_routed_serial") / pl.col("T_routed"),
    )

    df_grouped = (
        df.group_by(["hardware.num_qubits", "hardware.layout.k"])
        .agg(
            pl.len().alias("count"),
            pl.col("T_trans").median().alias("T_trans_median"),
            pl.col("T_trans").quantile(0.25).alias("T_trans_q25"),
            pl.col("T_trans").quantile(0.75).alias("T_trans_q75"),
            pl.col("DT_routed").median().alias("DT_routed_median"),
            pl.col("DT_routed").quantile(0.25).alias("DT_routed_q25"),
            pl.col("DT_routed").quantile(0.75).alias("DT_routed_q75"),
            pl.col("DT_serial").median().alias("DT_serial_median"),
            pl.col("DT_serial").quantile(0.25).alias("DT_serial_q25"),
            pl.col("DT_serial").quantile(0.75).alias("DT_serial_q75"),
            pl.col("R").median().alias("R_median"),
            pl.col("R").quantile(0.25).alias("R_q25"),
            pl.col("R").quantile(0.75).alias("R_q75"),
        )
        .sort(["hardware.num_qubits", "hardware.layout.k"])
    )

    fig, axes = plt.subplots(
        figsize=(DOUBLE_COLUMN_WIDTH, SINGLE_COLUMN_WIDTH * 0.8),
        dpi=200,
        layout="constrained",
        ncols=2,
    )

    # --- Panel (a): Absolute overhead ---
    ax = axes[0]
    handles_dict = {}

    # Routing overhead
    df_k2 = df_grouped.filter(pl.col("hardware.layout.k") == 2)
    handles, (a_r, b_r) = plot_with_fit(
        ax, df_k2, "DT_routed", wong_colors[0], "s", "Routing overhead", fit_x_min
    )
    handles_dict[
        f"Routing: ${a_r * 1e6:.1f}$" + r"$\,\mu$s $\cdot\, n^{" + f"{b_r:.2f}" + r"}$"
    ] = handles

    # Serialisation overhead for each k
    serial_fits = {}
    for k, color, marker in zip(k_values, colors, markers):
        df_k = df_grouped.filter(pl.col("hardware.layout.k") == k)
        handles, (a, b) = plot_with_fit(
            ax, df_k, "DT_serial", color, marker, f"Serial k={k}", fit_x_min
        )
        handles_dict[
            f"Serial. $k={k}$: ${a * 1e6:.1f}$" + r"$\,\mu$s $\cdot\, n^{" + f"{b:.2f}" + r"}$"
        ] = handles
        serial_fits[k] = (a, b)

    # QFT native circuit duration (k=2)
    handles, (a_trans, b_trans) = plot_with_fit(
        ax, df_k2, "T_trans", wong_colors[-2], "x", "QFT (native)", fit_x_min
    )
    handles_dict[
        f"QFT (native): ${a_trans * 1e6:.1f}$"
        + r"$\,\mu$s $\cdot\, n^{"
        + f"{b_trans:.2f}"
        + r"}$"
    ] = handles
    print(f"Target circuits: a={a_trans:.2e}, b={b_trans:.3f}")

    ax.set_xlabel("Number of qubits $n$")
    ax.set_ylabel("Overhead (ms)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1e3:g}"))
    ax.legend(
        handles_dict.values(),
        handles_dict.keys(),
        handler_map={tuple: HandlerTuple(ndivide=None)},
        loc="upper left",
    )
    add_grid_markers(ax)
    ax.text(
        -0.10,
        1.0,
        "(a)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="right",
    )

    # --- Panel (b): Relative overhead ---
    ax = axes[1]

    x_theory = np.logspace(
        np.log10(fit_x_min), np.log10(df_grouped["hardware.num_qubits"].max()), 100
    )
    T_base = a_trans * x_theory**b_trans
    denom = T_base + a_r * x_theory**b_r

    handles_dict_b = {}
    for k, color, marker in zip(k_values, colors, markers):
        df_k = df_grouped.filter(pl.col("hardware.layout.k") == k)
        x = df_k["hardware.num_qubits"].to_numpy()
        y = df_k["R_median"].to_numpy()
        y_lo = df_k["R_q25"].to_numpy()
        y_hi = df_k["R_q75"].to_numpy()

        h1 = ax.errorbar(
            x,
            y,
            yerr=[y - y_lo, y_hi - y],
            color=color,
            marker=marker,
            linestyle="",
            capsize=2,
            capthick=1,
            elinewidth=1,
        )

        a_s, b_s = serial_fits[k]
        y_theory = (denom + a_s * x_theory**b_s) / denom
        h2 = ax.plot(x_theory, y_theory, "--", color=color, alpha=0.7)[0]
        handles_dict_b[k] = (h1, h2)

    ax.set_xlabel("Number of qubits")
    ax.set_ylabel("Relative overhead")
    ax.legend(
        [handles_dict_b[k] for k in k_values],
        [str(k) for k in k_values],
        title="Qubits per switch $k$",
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    add_grid_markers(ax)
    ax.text(
        -0.10,
        1.0,
        "(b)",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

    return fig


if __name__ == "__main__":
    fig = plot_figure()
    plt.show()

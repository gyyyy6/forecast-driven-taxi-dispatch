"""Generate publication-ready metric comparison figures for the ride-hailing project."""

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script also works on headless machines.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Global style
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="paper")
plt.rcParams.update(
    {
        # SimHei is the primary font on Windows; the remaining fonts are
        # graceful fallbacks for other machines.
        "font.family": "sans-serif",
        "font.sans-serif": [
            "SimHei",
            "Microsoft YaHei",
            "Noto Sans SC",
            "Source Han Sans SC",
            "DejaVu Sans",
        ],
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.titleweight": "bold",
        "axes.titlesize": 15.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
    }
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = PROJECT_ROOT / "docs" / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _clean_axes(ax: plt.Axes, *, show_y_grid: bool = True) -> None:
    """Apply a minimal academic style by removing chart junk."""
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="both", which="both", length=0, pad=7, colors="#374151")
    ax.grid(axis="x", visible=False)
    if show_y_grid:
        ax.grid(axis="y", color="#DDE3EA", linewidth=0.8, alpha=0.9)
        ax.set_axisbelow(True)
    else:
        ax.grid(False)


def _save(fig: plt.Figure, filename: str) -> Path:
    output_path = FIGURE_DIR / filename
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


# ---------------------------------------------------------------------------
# Figure 1: STGCN graph-structure evolution / ablation study
# ---------------------------------------------------------------------------
def plot_stgcn_ablation() -> Path:
    models = [
        "单区域时间模型\n(空间失效)",
        "质心图模型\n(H32)",
        "混合图模型\n(H64 Hybrid)",
    ]
    mae = np.array([4.7604, 4.5423, 4.4314])
    wape = np.array([18.59, 17.74, 17.31])

    x = np.arange(len(models))
    bar_colors = ["#B8D5EC", "#6FA2CC", "#1F4E79"]
    wape_color = "#D97745"

    fig, ax = plt.subplots(figsize=(10.6, 7.0))
    fig.subplots_adjust(left=0.085, right=0.905, top=0.795, bottom=0.13)

    bars = ax.bar(
        x,
        mae,
        width=0.58,
        color=bar_colors,
        edgecolor="white",
        linewidth=1.6,
        zorder=3,
    )

    # MAE labels: exactly four decimal places.
    for bar, value in zip(bars, mae):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.09,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#1F2937",
            zorder=6,
        )

    ax.set_xticks(x, models)
    ax.set_xlim(-0.62, len(models) - 0.38)
    ax.set_ylim(0, 5.65)
    ax.set_yticks(np.arange(0, 5.6, 1.0))
    ax.set_xlabel("模型版本", fontsize=11.5, labelpad=10, color="#374151")
    ax.set_ylabel("MAE（左轴，越低越好）", fontsize=11.5, labelpad=10, color="#1F4E79")
    ax.axhline(0, color="#94A3B8", linewidth=0.8, zorder=4)
    _clean_axes(ax)

    # Secondary axis: WAPE trend.
    ax2 = ax.twinx()
    ax2.plot(
        x,
        wape,
        color=wape_color,
        linewidth=2.5,
        marker="o",
        markersize=9,
        markerfacecolor=wape_color,
        markeredgecolor="white",
        markeredgewidth=1.6,
        zorder=7,
    )
    for xi, value in zip(x, wape):
        ax2.annotate(
            f"{value:.2f}%",
            xy=(xi, value),
            xytext=(0, 13),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9.8,
            fontweight="semibold",
            color="#A24D23",
            bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="none", alpha=0.92),
            zorder=8,
        )

    ax2.set_ylim(16.85, 19.10)
    ax2.set_yticks(np.arange(17.0, 19.1, 0.5))
    ax2.set_ylabel("WAPE（右轴，%）", fontsize=11.5, labelpad=12, color=wape_color)
    ax2.tick_params(axis="y", length=0, pad=7, colors="#A24D23")
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.grid(False)

    legend_handles = [
        Patch(facecolor=bar_colors[1], edgecolor="none", label="MAE（柱）"),
        Line2D(
            [0],
            [0],
            color=wape_color,
            marker="o",
            markersize=6,
            markeredgecolor="white",
            linewidth=2.2,
            label="WAPE（折线）",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        ncol=2,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.8,
        borderaxespad=0.2,
    )

    fig.suptitle("STGCN 空间图结构演进与误差对比", y=0.97, fontsize=17, fontweight="bold")
    fig.text(
        0.5,
        0.902,
        "通过引入质心图与OD融合，MAE从4.76稳步降至4.43，验证了空间依赖的重要性。",
        ha="center",
        va="center",
        fontsize=10.4,
        color="#2B4C6F",
        bbox=dict(
            boxstyle="round,pad=0.48",
            facecolor="#F3F7FB",
            edgecolor="#C7D7E6",
            linewidth=0.9,
        ),
    )

    return _save(fig, "fig1_stgcn_ablation.png")


# ---------------------------------------------------------------------------
# Figure 2: LightGBM baseline versus the STGCN model
# ---------------------------------------------------------------------------
def plot_baseline_vs_stgcn() -> Path:
    models = ["LightGBM\n(强手工特征)", "STGCN\n(Hybrid图)"]
    mae = np.array([4.0595, 4.4314])
    colors = ["#E9B48D", "#1F4E79"]

    fig, ax = plt.subplots(figsize=(10.6, 7.2))
    fig.subplots_adjust(left=0.09, right=0.94, top=0.80, bottom=0.14)

    x = np.arange(len(models))
    bars = ax.bar(
        x,
        mae,
        width=0.48,
        color=colors,
        edgecolor="white",
        linewidth=1.7,
        zorder=3,
    )

    for bar, value in zip(bars, mae):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.11,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color="#1F2937",
            zorder=6,
        )

    ax.set_xticks(x, models)
    ax.set_xlim(-0.58, len(models) - 0.42)
    ax.set_ylim(0, 5.80)
    ax.set_yticks(np.arange(0, 6.0, 1.0))
    ax.set_xlabel("模型", fontsize=11.5, labelpad=10, color="#374151")
    ax.set_ylabel("MAE（越低越好）", fontsize=11.5, labelpad=10, color="#374151")
    ax.axhline(0, color="#94A3B8", linewidth=0.8, zorder=4)
    _clean_axes(ax)

    # Model-characteristic callout boxes in the upper blank area.
    ax.text(
        0.015,
        0.975,
        "LightGBM 特性\n依赖手工滞后特征，逐区域独立预测",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.8,
        color="#7C3E17",
        linespacing=1.55,
        bbox=dict(
            boxstyle="round,pad=0.52",
            facecolor="#FDF4EC",
            edgecolor="#E9B48D",
            linewidth=1.0,
        ),
        zorder=7,
    )
    ax.text(
        0.985,
        0.975,
        "STGCN 特性\n端到端学习，捕捉全局空间溢出，参数量仅 33K",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.8,
        color="#173B5C",
        linespacing=1.55,
        multialignment="left",
        bbox=dict(
            boxstyle="round,pad=0.52",
            facecolor="#EDF4FA",
            edgecolor="#1F4E79",
            linewidth=1.0,
        ),
        zorder=7,
    )

    fig.suptitle(
        "基线模型与高级时空模型预测精度对比",
        y=0.97,
        fontsize=17,
        fontweight="bold",
    )

    return _save(fig, "fig2_baseline_vs_stgcn.png")


def main() -> None:
    figure_paths = [
        plot_stgcn_ablation(),
        plot_baseline_vs_stgcn(),
    ]

    print("图表绘制完成，图片已按 DPI=300 保存至：")
    for index, figure_path in enumerate(figure_paths, start=1):
        print(f"  图 {index}: {figure_path}")


if __name__ == "__main__":
    main()

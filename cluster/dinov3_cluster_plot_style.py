"""Paper figure style shared with the manual/automatic diameter figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager


STYLE_REFERENCE = "Multi-Task-Learning/check/automatic_figures_um/compare_measurements.py"
SCATTER_SIZE = 12
SCATTER_EDGE_WIDTH = 0.4
SCATTER_ALPHA = 0.75
# The first three colors exactly match SHAPE_COLORS in the reference figure.
CLUSTER_COLORS = (
    "#2474a6", "#c66b20", "#418652", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#d62728",
    "#332288", "#882255",
)
PAPER_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 9,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.linewidth": 0.8,
    "axes.spines.left": True,
    "axes.spines.right": True,
    "axes.spines.top": True,
    "axes.spines.bottom": True,
    "axes.unicode_minus": True,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "grid.color": "#b0b0b0",
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "grid.alpha": 0.2,
    "legend.frameon": True,
    "legend.framealpha": 0.8,
    "legend.edgecolor": "0.8",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def style_axes(ax, grid: str | None = "both") -> None:
    """Restore reference axes even for libraries that override Matplotlib rc."""
    ax.set_title("")
    ax.set_facecolor("white")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.8)
    ax.tick_params(axis="both", labelsize=11, direction="out", width=0.8, length=3.5)
    ax.xaxis.label.set_fontfamily("DejaVu Sans")
    ax.yaxis.label.set_fontfamily("DejaVu Sans")
    ax.xaxis.label.set_fontsize(11)
    ax.yaxis.label.set_fontsize(11)
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily("DejaVu Sans")
        label.set_fontsize(11)
    ax.grid(False)
    if grid is not None:
        ax.grid(axis=grid, alpha=0.2, color="#b0b0b0", linewidth=0.8, linestyle="-")


def save_paper_figure(fig, stem: Path, tight: bool = False) -> None:
    """Match reference PNG/TIFF export and also retain vector PDF output."""
    # Keep all exports title-free, including third-party plot helpers.
    for ax in fig.axes:
        for loc in ("left", "center", "right"):
            ax.set_title("", loc=loc)
    if fig._suptitle is not None:
        fig._suptitle.set_text("")
    options = {"bbox_inches": "tight"} if tight else {}
    stem = Path(stem)
    fig.savefig(stem.with_suffix(".png"), dpi=300, facecolor="white", **options)
    fig.savefig(stem.with_suffix(".pdf"), facecolor="white", **options)
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, facecolor="white",
                pil_kwargs={"compression": "tiff_lzw"}, **options)
    plt.close(fig)


def plot_style_metadata() -> dict:
    return {
        "reference_source": STYLE_REFERENCE,
        "font": "DejaVu Sans",
        "font_file": font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans")),
        "axis_label_pt": 11, "tick_label_pt": 11, "legend_pt": 9,
        "titles": False, "all_four_spines": True, "spine_linewidth_pt": 0.8,
        "scatter_size_pt_squared": SCATTER_SIZE, "scatter_alpha": SCATTER_ALPHA,
        "scatter_edge_linewidth_pt": SCATTER_EDGE_WIDTH,
        "boxplot_outlier_marker_size_pt": 1.5,
        "grid_alpha": 0.2,
        "cluster_colors": {str(i): color for i, color in enumerate(CLUSTER_COLORS)},
        "png_dpi": 300, "tiff_dpi": 600, "tiff_compression": "tiff_lzw", "pdf_vector": True,
    }

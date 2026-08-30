"""Regenerate images/speedup-by-shape.png from the measured appendix results.

Numbers are the final full-settings figures reported in TechnicalReport.md section 8.
Run:  python scripts/plot_speedups.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.path import Path
from matplotlib.patches import PathPatch

# shape number, short description, speedup vs baseline
SHAPES = [
    (1, "base", 4.41),
    (2, "batch 1", 39.67),
    (3, "batch 4", 23.37),
    (4, "batch 16", 9.86),
    (5, "batch 128", 4.35),
    (6, "batch 10000", 6.53),
    (7, "d_model 32", 13.63),
    (8, "d_model 1024", 1.41),
    (9, "1 head", 2.36),
    (10, "2 heads", 3.14),
    (11, "16 heads", 8.93),
    (12, "seq 32", 11.00),
    (13, "seq 1024", 15.14),
    (14, "seq 100000", 22.84),
]

GEOMEAN = 8.13

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"

DPI = 200
X_MAX = 44.0


def rounded_end_bar(ax, y, width, height, radius, **kw):
    """A bar square at the baseline and rounded at the data end."""
    r = min(radius, width, height / 2.0)
    y0, y1 = y - height / 2.0, y + height / 2.0
    verts = [
        (0.0, y0),
        (width - r, y0),
        (width, y0),
        (width, y0 + r),
        (width, y1 - r),
        (width, y1),
        (width - r, y1),
        (0.0, y1),
        (0.0, y0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    ax.add_patch(PathPatch(Path(verts, codes), **kw))


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )

    fig, ax = plt.subplots(figsize=(8.4, 5.6), dpi=DPI)
    fig.subplots_adjust(left=0.215, right=0.985, top=0.855, bottom=0.115)

    ys = list(range(len(SHAPES)))

    # recessive vertical gridlines, behind the data
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.yaxis.grid(False)

    # 1x = parity with the baseline
    ax.axvline(1.0, color=BASELINE, linewidth=1.0, zorder=2)
    ax.text(
        1.35,
        len(SHAPES) - 0.35,
        "1× = baseline",
        color=MUTED,
        fontsize=7.5,
        va="center",
        ha="left",
    )

    # bar thickness capped, in data units: 14 rows over ~4.35in of plot height
    bar_h = 0.6
    radius = 0.55  # ~4px at this scale

    for i, (num, label, speed) in enumerate(SHAPES):
        y = len(SHAPES) - 1 - i
        rounded_end_bar(
            ax, y, speed, bar_h, radius, facecolor=SERIES, edgecolor="none", zorder=3
        )
        ax.text(
            speed + 0.7,
            y,
            f"{speed:.2f}×",
            color=SECONDARY,
            fontsize=8.5,
            va="center",
            ha="left",
        )

    # Two label columns instead of one ragged tick label: shape numbers right-aligned
    # in their own gutter, descriptions right-aligned flush against the axis.
    ax.set_yticks(ys)
    ax.set_yticklabels([])
    ax.set_ylim(-0.7, len(SHAPES) - 0.3)

    label_tf = transforms.blended_transform_factory(ax.transAxes, ax.transData)
    for i, (num, label, _) in enumerate(SHAPES):
        y = len(SHAPES) - 1 - i
        ax.text(
            -0.215,
            y,
            str(num),
            transform=label_tf,
            color=SECONDARY,
            fontsize=8.5,
            weight="semibold",
            va="center",
            ha="right",
            clip_on=False,
        )
        ax.text(
            -0.022,
            y,
            label,
            transform=label_tf,
            color=MUTED,
            fontsize=8.5,
            va="center",
            ha="right",
            clip_on=False,
        )

    ax.set_xlim(0, X_MAX)
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30, 35, 40])
    ax.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=0)
    ax.tick_params(axis="y", length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_color(SECONDARY)
    ax.set_xlabel(
        "Speedup over the PyTorch baseline (×, higher is better)",
        fontsize=8.5,
        color=MUTED,
        labelpad=8,
    )

    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["left"].set_linewidth(1.0)

    fig.text(
        0.02,
        0.955,
        "Speedup on each of the 14 official test shapes",
        fontsize=12.5,
        color=INK,
        weight="semibold",
        ha="left",
        va="center",
    )
    fig.text(
        0.02,
        0.902,
        f"RTX 3070 · every shape passes the accuracy check · geometric mean {GEOMEAN:.2f}×",
        fontsize=8.5,
        color=MUTED,
        ha="left",
        va="center",
    )

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "images",
        "speedup-by-shape.png",
    )
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()

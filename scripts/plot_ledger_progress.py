"""Regenerate images/geomean-by-iteration.png from docs/OPTIMIZATION_LEDGER.md.

Numbers are the `Geomean progression` table's absolute column. The sweep re-times
`BaselineTransformer` on every run, so that arm moves with machine state as well as
with our changes; the ledger calls each stable stretch an "epoch" and only numbers
within one epoch are comparable. The line is therefore drawn broken at each epoch
boundary rather than joined across it, and the epochs are shaded as alternating bands.

Cycles 0-18 are plotted. Cycle 0 is the start state rather than a cycle, so it gets its
own recessive marker, but the line is drawn from it into cycle 1 so the first cycle's
jump is visible. Cycle 19 is alone in its own epoch and is left off.

Run:  python scripts/plot_ledger_progress.py
"""

import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# cycle, geomean vs BaselineTransformer, epoch, kept a gain?
# Cycles 5 and 12 were accepted but behaviour-neutral; 11, 15, 16, 17 and 18 were
# rejected and edited nothing. Both count as "kept nothing" here.
ROWS = [
    (0, 6.831, "A", False),   # start state, not a cycle
    (1, 7.853, "A", True),
    (2, 8.191, "A", True),
    (3, 8.278, "A", True),
    (4, 7.875, "B", True),
    (5, 7.875, "B", False),
    (6, 8.685, "B", True),
    (7, 8.867, "B", True),
    (8, 9.023, "B", True),
    (9, 9.467, "B", True),
    (10, 9.473, "B", True),
    (11, 9.473, "B", False),
    (12, 9.333, "C", False),
    (13, 9.4417, "C", True),
    (14, 9.5914, "C", True),
    (15, 9.5914, "C", False),
    (16, 9.5914, "C", False),
    (17, 9.5914, "C", False),
    (18, 9.5914, "C", False),
]

# cycle -> (text, x offset, y offset, ha, va)
MILESTONES = {
    1: ("projections onto the\ncustom matrix kernel", 0.28, -0.16, "left", "top"),
    6: ("fp16 through\nthe multiplies", -0.28, 0.10, "right", "bottom"),
    9: ("overlapped key/value\nloading", 0.0, 0.16, "center", "bottom"),
}

SURFACE = "#fcfcfb"
BAND = "#f3f2ed"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
SERIES = "#2a78d6"

DPI = 200
X_MIN, X_MAX = -0.7, 18.85
Y_MIN, Y_MAX = 6.4, 10.3


def main():
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
        }
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=DPI)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.795, bottom=0.135)

    # group consecutive cycles into their epoch
    epochs = []
    for it, val, ep, _ in ROWS:
        if not epochs or epochs[-1][0] != ep:
            epochs.append((ep, [], []))
        epochs[-1][1].append(it)
        epochs[-1][2].append(val)

    # alternating bands, so a re-timing of the reference reads as a change of
    # background rather than as a rule competing with the data
    edges = [X_MIN]
    for i in range(1, len(epochs)):
        edges.append((epochs[i][1][0] + epochs[i - 1][1][-1]) / 2.0)
    edges.append(X_MAX)
    for i in range(len(epochs)):
        if i % 2 == 1:
            ax.axvspan(edges[i], edges[i + 1], color=BAND, linewidth=0, zorder=0)

    ax.set_axisbelow(True)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, linestyle="-")
    ax.xaxis.grid(False)

    for _, xs, ys in epochs:
        if len(xs) > 1:
            ax.plot(
                xs, ys, color=SERIES, linewidth=2.0, zorder=3, solid_capstyle="round"
            )

    # filled = the cycle kept a gain, hollow = it kept nothing. Cycle 0 is the
    # start state rather than a cycle, so it gets its own recessive marker.
    gain = [(it, v) for it, v, _, g in ROWS[1:] if g]
    flat = [(it, v) for it, v, _, g in ROWS[1:] if not g]
    ax.scatter(*zip(*gain), s=30, color=SERIES, zorder=4, edgecolors="none")
    ax.scatter(
        *zip(*flat), s=30, facecolors=SURFACE, edgecolors=SERIES, linewidths=1.4, zorder=4
    )
    ax.scatter([0], [ROWS[0][1]], s=34, color=BASELINE, zorder=4, edgecolors="none")

    ax.text(
        0.34, ROWS[0][1] + 0.03, "6.83×", color=SECONDARY, fontsize=9.0,
        weight="semibold", ha="left", va="center", zorder=5,
    )
    ax.text(
        0.34, ROWS[0][1] - 0.23, "where the loop started", color=MUTED, fontsize=8.0,
        ha="left", va="center", zorder=5,
    )

    by_cycle = {it: v for it, v, _, _ in ROWS}
    for cyc, (text, dx, dy, ha, va) in MILESTONES.items():
        ax.text(
            cyc + dx,
            by_cycle[cyc] + dy,
            text,
            color=SECONDARY,
            fontsize=8.0,
            linespacing=1.45,
            ha=ha,
            va=va,
            zorder=5,
        )

    ax.text(
        14, 9.5914 + 0.14, "9.59×", color=SECONDARY, fontsize=9.0,
        weight="semibold", ha="center", va="bottom", zorder=5,
    )

    # the flat run: four consecutive cycles that changed nothing
    ax.annotate(
        "",
        xy=(14.85, 9.50), xytext=(18.15, 9.50),
        arrowprops=dict(arrowstyle="-", color=BASELINE, linewidth=1.0),
    )
    ax.text(
        16.5, 9.44, "four cycles, nothing kept", color=MUTED, fontsize=8.0,
        ha="center", va="top", zorder=5,
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_xticks(range(0, 19))
    ticks = [6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{v:.1f}×" for v in ticks])
    ax.tick_params(axis="both", labelsize=8.5, colors=MUTED, length=0)
    ax.set_xlabel("Cycle of the loop", fontsize=8.5, color=MUTED, labelpad=8)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
        ax.spines[side].set_linewidth(1.0)

    fig.text(
        0.02, 0.950, "Speed after each cycle of the loop",
        fontsize=13.0, color=INK, weight="semibold", ha="left", va="center",
    )
    fig.text(
        0.02, 0.898,
        "Geometric mean over the 13 shapes the loop runs. Filled dots kept a gain, hollow "
        "kept nothing.",
        fontsize=8.5, color=MUTED, ha="left", va="center",
    )
    fig.text(
        0.02, 0.856,
        "Shaded blocks separate runs that re-timed the PyTorch baseline — only points inside "
        "one block compare with each other.",
        fontsize=8.5, color=MUTED, ha="left", va="center",
    )

    out = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "images",
        "geomean-by-iteration.png",
    )
    fig.savefig(out, dpi=DPI, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()

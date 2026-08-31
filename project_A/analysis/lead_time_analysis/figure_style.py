"""One house style for the paper's three data figures (Figs. 2-4).

All three are single-column blocks, so all three are authored at the same width.
That matters more than it looks: matplotlib scales type with the figure, so a
figure drawn 3.2in wide and one drawn 3.4in wide end up with different effective
point sizes once both are placed in the same 8cm column.

Strokes sit well under matplotlib's defaults. At 3.4in reduced to 8cm a default
0.8pt spine prints about as dark as the data it is supposed to frame, and these
figures are mostly frame -- stacked panels, gridlines, reference rules.

Authoring dpi and export dpi are the same number here. They were not before: two
of the three built the figure at 400 and then wrote it at 300.

Usage
    import figure_style as fs
    fs.apply()
    fig, axes = plt.subplots(2, 1, figsize=(fs.COL_W, 3.7))
    ...
    fs.style_axes(ax)
"""

COL_W = 3.4          # single-column figure width (in)
FULL_W = 7.0         # figure spanning both columns, at the same author-to-page
                     # ratio as COL_W -- so a 7pt label prints the same size in
                     # a full-width figure as in a single-column one
DPI = 400            # authoring and export

# Type. Four steps only -- more than that reads as inconsistency, not hierarchy.
FS_TITLE = 8.0       # panel title
FS_LABEL = 7.0       # axis label
FS_TICK = 8.0        # tick label
FS_NOTE = 5.8        # in-plot annotation, legend

# Strokes.
LW_SPINE = 0.5
LW_GRID = 0.45
LW_EDGE = 0.35       # bar / marker outline
LW_DATA = 0.8        # data curve
LW_RULE = 0.7        # reference line: threshold, median, event marker
LW_HATCH = 0.5       # hatch stroke; matplotlib's default 1.0 prints heavier
                     # than the spines the hatched patch sits next to

# Ink. Nothing is pure black; the darkest element should be the data.
INK = "#333333"      # titles, axis labels
INK_TICK = "#555555"
SPINE = "#6f6f6f"
GRID = "#8c8c8c"
GRID_ALPHA = 0.25

# Label placement. A shared y label sits in the tick-label column rather than
# out at the figure edge, where tight_layout leaves a visible gutter.
TITLE_PAD = 3        # gap between a panel title and its axes
SUPY_X = 0.038
LABELPAD = 2.5

# Shared marks.
BAR_ALPHA = 0.72
BAR_EDGE = "#444444"   # bar outline; never pure black -- the data is the darkest ink
FILL_ALPHA = 0.15
ACCENT = "#c0392b"   # annotation text tied to a red reference line


def apply():
    """Set the rcParams every figure in the set shares."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="ticks", palette="muted")
    plt.rcParams.update({
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        # Computer Modern for math, Arial for everything else. Every symbol in
        # these figures also appears in the manuscript's equations, so it is set
        # the way the manuscript sets it -- an italic H in the figure and an
        # italic H in the text should not be two different letters. Full
        # text.usetex would route the Korean labels through LaTeX too, which the
        # default engine cannot typeset.
        "mathtext.fontset": "cm",
        "hatch.linewidth": LW_HATCH,
        "axes.edgecolor": SPINE,
        "axes.linewidth": LW_SPINE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "text.color": INK,
        "xtick.color": INK_TICK,
        "ytick.color": INK_TICK,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "xtick.major.width": 0.4,
        "ytick.major.width": 0.4,
        "xtick.major.size": 2.0,
        "ytick.major.size": 2.0,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "legend.fontsize": FS_NOTE,
        "legend.frameon": False,
    })


def style_axes(ax, despine=True):
    """Horizontal gridlines only, and drop the top/right spines."""
    import seaborn as sns

    ax.grid(True, axis="y", linestyle=":", alpha=GRID_ALPHA,
            color=GRID, linewidth=LW_GRID)
    ax.grid(False, axis="x")
    if despine:
        sns.despine(ax=ax, top=True, right=True)

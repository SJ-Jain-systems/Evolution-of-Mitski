"""A small matplotlib theme built from the validated data-viz palette.

Colours and roles follow the reference palette: a fixed-order categorical set
(assigned by entity, never cycled), a single-hue blue ramp for magnitude, and
recessive grey chrome so the marks carry the ink. Figures render on the light
chart surface, which is what Quarto's default HTML/PDF output shows.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

# Chart chrome & ink (light surface)
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GOOD = "#006300"

# Categorical palette, fixed order (blue, green, magenta, yellow, aqua, orange,
# violet, red). Assign by entity, in this order, and never rotate.
CATEGORICAL = [
    "#2a78d6",  # 1 blue
    "#008300",  # 2 green
    "#e87ba4",  # 3 magenta
    "#eda100",  # 4 yellow
    "#1baf7a",  # 5 aqua
    "#eb6834",  # 6 orange
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

# Single-hue sequential ramp (blue, light -> dark), for ordered magnitude.
SEQUENTIAL = [
    "#cde2fb",
    "#9ec5f4",
    "#6da7ec",
    "#3987e5",
    "#2a78d6",
    "#1c5cab",
    "#184f95",
    "#104281",
]

# Semantic roles used repeatedly in this report.
BLUE = CATEGORICAL[0]
GREEN = CATEGORICAL[1]
ORANGE = CATEGORICAL[5]
VIOLET = CATEGORICAL[6]
RED = CATEGORICAL[7]

# The visual "trilogy" colour story the source video describes.
TRILOGY = {
    "Be the Cowboy": "#c62828",  # explosion of red
    "Laurel Hell": "#111111",  # drenched in black
    "The Land Is Inhospitable and So Are We": "#7d8794",  # pulled-back, muted
}


def apply() -> None:
    """Install the theme globally for the current session/render."""
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.dpi": 140,
            "savefig.dpi": 140,
            "savefig.bbox": "tight",
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
            "text.color": INK,
            "axes.edgecolor": BASELINE,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": False,
            "legend.fontsize": 10,
            "lines.linewidth": 2.0,
            "lines.markersize": 7,
        }
    )


def style_axes(ax) -> None:
    """Per-axes touches the rcParams can't express (horizontal-only grid)."""
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(BASELINE)


def new_fig(width=8.2, height=4.8):
    apply()
    fig, ax = plt.subplots(figsize=(width, height))
    style_axes(ax)
    return fig, ax


# --- Shared colour ramp ----------------------------------------------------
def sequential_cmap():
    """The house single-hue blue ramp as a matplotlib colormap (light -> dark).

    Centralised here so the 2D motif heatmap and the 3D figures draw ordered
    magnitude from exactly the same ramp. Time -- the axis the whole project is
    about -- is carried consistently by this ramp, early (light) to late (dark).
    """
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("mitski_blues", SEQUENTIAL)


# --- 3D helpers ------------------------------------------------------------
# The matplotlib 3D figures share the 2D theme so they sit in the report as one
# visual system: same palette, same recessive grey chrome, marks carry the ink.
def new_fig3d(width=8.4, height=6.6):
    """A themed 3D figure/axes pair (mirrors ``new_fig`` for the 3D toolkit)."""
    apply()
    fig = plt.figure(figsize=(width, height))
    ax = fig.add_subplot(projection="3d")
    style_axes3d(ax)
    return fig, ax


def style_axes3d(ax) -> None:
    """Recessive chrome for a 3D axes: faint panes, grey grid and tick/label
    colours, so the data marks (not the box) carry the ink."""
    pane = (1.0, 1.0, 1.0, 0.0)  # transparent panes -> the page shows through
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color(pane)
        axis.pane.set_edgecolor(GRID)
        axis.pane.set_alpha(1.0)
        # Grey, thin gridlines matching the 2D grid.
        axis._axinfo["grid"].update(color=GRID, linewidth=0.7)
        axis.line.set_color(BASELINE)
        axis.label.set_color(INK_2)
        axis.set_tick_params(colors=MUTED, labelsize=8)
    ax.set_facecolor(SURFACE)

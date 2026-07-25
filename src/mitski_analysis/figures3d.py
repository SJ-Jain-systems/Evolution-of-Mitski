"""Three-dimensional figures for the report, built with matplotlib's mplot3d.

These were originally Plotly figures, which render as an inlined ``plotly.js``
runtime -- i.e. as JavaScript in the page. They are re-implemented here in pure
Python (matplotlib), so no HTML/JS rendering layer is involved: every ``fig3d_*``
function takes the album (and sometimes song) table and returns a
``matplotlib.figure.Figure``, exactly mirroring the ``figures.py`` contract, so a
script and the Quarto document render identical charts.

A single frozen projection tends to mislead -- you cannot read depth from one
camera angle -- which is why the interactive version let the reader rotate. To
recover that here without JavaScript, each figure is composed from a chosen,
readable angle *and* ``scripts/make_3d.py`` renders a 360-degree rotation
animation (GIF) via :func:`spin`, so the shape still reads. Direct on-figure
text labels stand in for the old hover tooltips.

Colour and chrome follow the same house palette as the 2D figures (see
``theme.py``): the fixed-order categorical set assigned by entity, the single-hue
blue ramp for ordered magnitude, and recessive grey chrome so the marks carry the
ink. Time -- the axis the whole project is about -- is carried consistently by
the blue sequential ramp, early (light) to late (dark).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FuncFormatter

from . import theme as TH

# Short album labels, single line (kept compact for 3D text annotations).
SHORT = {
    "Lush": "Lush",
    "Retired from Sad, New Career in Business": "Retired from Sad",
    "Bury Me at Makeout Creek": "Bury Me at Makeout Creek",
    "Puberty 2": "Puberty 2",
    "Be the Cowboy": "Be the Cowboy",
    "Laurel Hell": "Laurel Hell",
    "The Land Is Inhospitable and So Are We": "The Land Is Inhospitable…",
}


def _year_colors(years) -> tuple[np.ndarray, Normalize]:
    """Map release years onto the house blue ramp (light=early, dark=late)."""
    years = np.asarray(years, dtype=float)
    norm = Normalize(vmin=years.min(), vmax=years.max())
    cmap = TH.sequential_cmap()
    return cmap(norm(years)), norm


def _year_colorbar(fig, ax, norm) -> None:
    sm = ScalarMappable(norm=norm, cmap=TH.sequential_cmap())
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cbar.set_label("Release year", color=TH.INK_2, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8, colors=TH.MUTED)
    # Integer year ticks (no ".0").
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{int(round(v))}"))


def _title(ax, text: str) -> None:
    ax.set_title(text, fontsize=13, fontweight="bold", color=TH.INK, pad=6)


# --------------------------------------------------------------------------- #
# 1. The evolution trajectory: a path through "lyric space".
#    X = word density, Y = lexical diversity, Z = repetition, time by colour.
# --------------------------------------------------------------------------- #
def fig3d_trajectory(albums: pd.DataFrame):
    order = albums.sort_values("release_date").reset_index(drop=True)
    x = order["words_per_minute"].to_numpy()
    y = order["mattr"].to_numpy()
    z = order["repetition_index"].to_numpy()
    colors, norm = _year_colors(order["release_year"])

    fig, ax = TH.new_fig3d()

    # The path itself: an ordered line through time, recessive so the album
    # markers carry the ink.
    ax.plot(x, y, z, color=TH.MUTED, lw=1.8, zorder=1)

    # Faint drop-lines to the floor give the eye a depth cue without a second
    # camera angle.
    zfloor = float(z.min()) - 0.15
    for xi, yi, zi in zip(x, y, z):
        ax.plot([xi, xi], [yi, yi], [zi, zfloor], color=TH.GRID, lw=1.0, zorder=1)

    # Album markers, coloured by release year on the house blue ramp so travel
    # direction (early = light, late = dark) is legible at a glance.
    ax.scatter(
        x, y, z, s=70, c=colors, edgecolor=TH.SURFACE, linewidth=1.2, depthshade=False, zorder=3
    )
    for xi, yi, zi, name in zip(x, y, z, order["album"]):
        ax.text(
            xi, yi, zi + 0.12, SHORT[name], fontsize=7.5, color=TH.INK_2, ha="center", va="bottom"
        )

    ax.set_xlabel("Words per minute  →  denser")
    ax.set_ylabel("Lexical diversity (MATTR)  →  wider")
    ax.set_zlabel("Repetition index  →  more repetitive")
    _title(ax, "The path through lyric space")
    ax.view_init(elev=20, azim=-58)
    _year_colorbar(fig, ax, norm)
    return fig


# --------------------------------------------------------------------------- #
# 2. Motif terrain: the motif heatmap as a 3D surface.
#    Height = imagery rate per 1,000 words; ridges rise on the late albums.
# --------------------------------------------------------------------------- #
def fig3d_motif_terrain(albums: pd.DataFrame):
    motifs = ["body", "water", "fire_light", "home_domestic", "death", "animals"]
    nice = ["Body", "Water", "Fire /\nlight", "Home", "Death", "Animals"]
    order = albums.sort_values("release_date").reset_index(drop=True)
    labels = [SHORT[a] for a in order["album"]]

    # Rows = albums (time), cols = motifs; Z = rate per 1k words.
    Z = np.array([[row[f"motif_{m}_per_1k"] for m in motifs] for _, row in order.iterrows()])
    ncol, nrow = len(motifs), len(order)
    X, Y = np.meshgrid(np.arange(ncol), np.arange(nrow))

    fig, ax = TH.new_fig3d(width=8.8, height=6.8)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=TH.sequential_cmap(),
        rcount=nrow,
        ccount=ncol,
        linewidth=0.3,
        edgecolor=TH.SURFACE,
        antialiased=True,
        shade=True,
    )

    ax.set_xticks(np.arange(ncol))
    ax.set_xticklabels(nice, fontsize=8)
    ax.set_yticks(np.arange(nrow))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_zlabel("Rate per 1,000 words")
    ax.set_ylabel("Album  (early → late)", labelpad=18)
    _title(ax, "Recurring imagery as terrain")
    ax.view_init(elev=28, azim=-62)

    cbar = fig.colorbar(surf, ax=ax, fraction=0.03, pad=0.02, shrink=0.6)
    cbar.set_label("Per 1,000 words", color=TH.INK_2, fontsize=9)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=8, colors=TH.MUTED)
    return fig


# --------------------------------------------------------------------------- #
# 3. Per-song cloud: every song in density × diversity × word-length space,
#    coloured by album, to show how the spread moves era to era.
# --------------------------------------------------------------------------- #
def fig3d_song_cloud(songs: pd.DataFrame, albums: pd.DataFrame):
    order = albums.sort_values("release_date")["album"].tolist()
    color_by_album = {a: TH.CATEGORICAL[i % len(TH.CATEGORICAL)] for i, a in enumerate(order)}

    fig, ax = TH.new_fig3d(width=8.8, height=6.8)
    for album in order:
        sub = songs[songs["album"] == album]
        ax.scatter(
            sub["word_count"],
            sub["mattr"],
            sub["mean_word_length"],
            s=26,
            color=color_by_album[album],
            edgecolor=TH.SURFACE,
            linewidth=0.4,
            alpha=0.9,
            depthshade=False,
            label=SHORT[album],
        )

    ax.set_xlabel("Words per song")
    ax.set_ylabel("Lexical diversity (MATTR)")
    ax.set_zlabel("Mean word length")
    _title(ax, "Every song in three dimensions")
    ax.view_init(elev=18, azim=-60)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.0, 0.95),
        fontsize=7.5,
        frameon=False,
        handletextpad=0.2,
        labelcolor=TH.INK_2,
    )
    return fig


# --------------------------------------------------------------------------- #
# 4. Pronoun trajectory: the compositional I / you / we path over time.
#    The three shares sum to 1, so the points live on a plane; the 3D view
#    keeps that constraint visible while time bends the path.
# --------------------------------------------------------------------------- #
def fig3d_pronoun_trajectory(albums: pd.DataFrame):
    order = albums.sort_values("release_date").reset_index(drop=True)
    xi = order["pron_first_singular_share"].to_numpy()
    yi = order["pron_second_share"].to_numpy()
    zi = order["pron_first_plural_share"].to_numpy()
    colors, norm = _year_colors(order["release_year"])

    fig, ax = TH.new_fig3d()
    ax.plot(xi, yi, zi, color=TH.MUTED, lw=1.8, zorder=1)
    ax.scatter(
        xi, yi, zi, s=70, c=colors, edgecolor=TH.SURFACE, linewidth=1.2, depthshade=False, zorder=3
    )
    for a, b, c, name in zip(xi, yi, zi, order["album"]):
        ax.text(a, b, c + 0.01, SHORT[name], fontsize=7.5, color=TH.INK_2, ha="center", va="bottom")

    pct = FuncFormatter(lambda v, _p: f"{v:.0%}")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.zaxis.set_major_formatter(pct)
    ax.set_xlabel("Share: I / me / my")
    ax.set_ylabel("Share: you / your")
    ax.set_zlabel("Share: we / us / our")
    _title(ax, "Who the songs address, over time")
    ax.view_init(elev=22, azim=-52)
    _year_colorbar(fig, ax, norm)
    return fig


# --------------------------------------------------------------------------- #
# Rotation animation: spin a 3D figure a full turn and save it as a GIF, so the
# depth the old drag-to-rotate gave is recoverable with no JavaScript.
# --------------------------------------------------------------------------- #
def spin(fig, out_path, frames: int = 72, elev: float | None = None, fps: int = 18, dpi: int = 90):
    """Write a 360-degree azimuthal rotation of ``fig``'s 3D axes to ``out_path``.

    Uses matplotlib's ``FuncAnimation`` + ``PillowWriter`` (a GIF; Pillow ships
    as a matplotlib dependency), so it needs no ffmpeg and no browser. ``fig`` is
    expected to contain exactly one 3D axes (every ``fig3d_*`` builder here does).
    """
    from matplotlib.animation import FuncAnimation, PillowWriter

    ax = next(a for a in fig.axes if getattr(a, "name", "") == "3d")
    base_elev = ax.elev if elev is None else elev
    base_azim = ax.azim

    def update(i):
        ax.view_init(elev=base_elev, azim=base_azim + (360.0 * i / frames))
        return ()

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out_path), writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return out_path

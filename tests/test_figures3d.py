"""Smoke tests for the 3D figures (pure matplotlib, no Plotly / JavaScript).

These mirror the contract the report and scripts/make_3d.py rely on: every
``fig3d_*`` builder returns a matplotlib Figure whose axes are 3D, built from the
real album/song tables (so a schema drift in data.py fails loudly here), and
``spin`` writes a non-empty rotation GIF.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis import figures3d as F3  # noqa: E402
from mitski_analysis.data import build_album_table, build_song_table  # noqa: E402


@pytest.fixture(scope="module")
def tables():
    return build_album_table(), build_song_table()


def _sole_3d_axes(fig):
    """Return the figure's single 3D axes, asserting there is exactly one."""
    threed = [a for a in fig.axes if getattr(a, "name", "") == "3d"]
    assert len(threed) == 1
    return threed[0]


def test_trajectory_builds(tables):
    albums, _ = tables
    fig = F3.fig3d_trajectory(albums)
    assert isinstance(fig, matplotlib.figure.Figure)
    _sole_3d_axes(fig)


def test_motif_terrain_is_a_surface(tables):
    albums, _ = tables
    fig = F3.fig3d_motif_terrain(albums)
    ax = _sole_3d_axes(fig)
    # A surface collection is present (7 albums x 6 motifs underneath).
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    assert any(isinstance(c, Poly3DCollection) for c in ax.collections)


def test_song_cloud_has_one_scatter_per_album(tables):
    albums, songs = tables
    fig = F3.fig3d_song_cloud(songs, albums)
    ax = _sole_3d_axes(fig)
    # One legend entry (one scatter trace) per album.
    assert len(ax.get_legend().get_texts()) == len(albums)


def test_pronoun_trajectory_builds(tables):
    albums, _ = tables
    fig = F3.fig3d_pronoun_trajectory(albums)
    assert isinstance(fig, matplotlib.figure.Figure)
    _sole_3d_axes(fig)


def test_spin_writes_a_gif(tables, tmp_path):
    albums, _ = tables
    out = tmp_path / "spin.gif"
    # A short animation keeps the test fast while still exercising the writer.
    F3.spin(F3.fig3d_trajectory(albums), out, frames=4, fps=8, dpi=60)
    assert out.exists() and out.stat().st_size > 0

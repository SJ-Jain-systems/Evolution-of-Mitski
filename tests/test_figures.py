"""Smoke tests for the 2D figure builders.

Each ``fig_*`` builder must return a matplotlib Figure from the real album/song
tables, so a schema drift in data.py or a plotting error fails loudly here rather
than only at report-render time.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis import figures as F  # noqa: E402
from mitski_analysis.data import (  # noqa: E402
    build_album_table,
    build_song_table,
    build_vocab_growth,
)


@pytest.fixture(scope="module")
def tables():
    return build_album_table(), build_song_table(), build_vocab_growth()


def test_album_only_figures_build(tables):
    albums, _, _ = tables
    for builder in (
        F.fig_wpm_over_time,
        F.fig_wpm_vs_diversity_panels,
        F.fig_inverse_scatter,
        F.fig_pronoun_mix,
        F.fig_motif_heatmap,
        F.fig_valence_over_time,
        F.fig_trilogy,
        F.fig_refrain_over_time,
        F.fig_rhyme_over_time,
        F.fig_structure_over_time,
    ):
        fig = builder(albums)
        assert isinstance(fig, matplotlib.figure.Figure)
        assert fig.axes  # at least one axes drawn


def test_words_per_song_builds(tables):
    albums, songs, _ = tables
    fig = F.fig_words_per_song(songs, albums)
    assert isinstance(fig, matplotlib.figure.Figure)


def test_vocab_growth_builds(tables):
    albums, _, growth = tables
    fig = F.fig_vocab_growth(growth, albums)
    assert isinstance(fig, matplotlib.figure.Figure)


if __name__ == "__main__":
    import traceback

    albums, songs, growth = build_album_table(), build_song_table(), build_vocab_growth()

    def _run():
        test_album_only_figures_build((albums, songs, growth))
        test_words_per_song_builds((albums, songs, growth))
        test_vocab_growth_builds((albums, songs, growth))

    try:
        _run()
        print("PASS all figure smoke tests")
    except Exception:
        traceback.print_exc()
        sys.exit(1)

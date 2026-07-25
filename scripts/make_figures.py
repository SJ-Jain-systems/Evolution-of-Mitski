#!/usr/bin/env python3
"""Render every report figure to figures/*.png.

The Quarto report generates these same charts inline at render time (it imports
the same functions), so this script is for previewing and for verifying the
plotting code runs without a Quarto install.

Run from the repo root:  python scripts/make_figures.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis import figures as F  # noqa: E402
from mitski_analysis.data import (  # noqa: E402
    build_album_table,
    build_song_table,
    build_vocab_growth,
    repo_root,
)

RENDERERS = {
    "01_wpm_over_time": lambda a, s, g: F.fig_wpm_over_time(a),
    "02_wpm_vs_diversity": lambda a, s, g: F.fig_wpm_vs_diversity_panels(a),
    "03_inverse_scatter": lambda a, s, g: F.fig_inverse_scatter(a),
    "04_words_per_song": lambda a, s, g: F.fig_words_per_song(s, a),
    "05_pronoun_mix": lambda a, s, g: F.fig_pronoun_mix(a),
    "06_motif_heatmap": lambda a, s, g: F.fig_motif_heatmap(a),
    "07_valence_over_time": lambda a, s, g: F.fig_valence_over_time(a),
    "08_trilogy": lambda a, s, g: F.fig_trilogy(a),
    "09_refrain_over_time": lambda a, s, g: F.fig_refrain_over_time(a),
    "10_vocab_growth": lambda a, s, g: F.fig_vocab_growth(g, a),
    "11_rhyme_over_time": lambda a, s, g: F.fig_rhyme_over_time(a),
    "12_structure_over_time": lambda a, s, g: F.fig_structure_over_time(a),
}


def main() -> None:
    root = repo_root()
    out = root / "figures"
    out.mkdir(exist_ok=True)
    albums = build_album_table(root)
    songs = build_song_table(root)
    growth = build_vocab_growth(root)

    for name, fn in RENDERERS.items():
        fig = fn(albums, songs, growth)
        path = out / f"{name}.png"
        fig.savefig(path)
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()

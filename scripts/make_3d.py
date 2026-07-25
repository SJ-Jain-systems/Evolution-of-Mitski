#!/usr/bin/env python3
"""Render the 3D figures to static PNGs and rotation-animation GIFs under
``figures/3d/``.

These figures are pure-Python matplotlib (no Plotly / JavaScript). A static 3D
render is a single camera angle, which can be hard to read for depth, so each
chart is also written as a 360-degree rotation GIF (see ``figures3d.spin``). The
Quarto report imports the same ``figures3d`` functions, so the report and these
files render identical charts.

Run from the repo root:  python scripts/make_3d.py
Pass ``--no-gif`` to skip the (slower) rotation animations and write PNGs only.
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis import figures3d as F3  # noqa: E402
from mitski_analysis.data import build_album_table, build_song_table, repo_root  # noqa: E402

RENDERERS = {
    "01_evolution_trajectory": lambda a, s: F3.fig3d_trajectory(a),
    "02_motif_terrain": lambda a, s: F3.fig3d_motif_terrain(a),
    "03_song_cloud": lambda a, s: F3.fig3d_song_cloud(s, a),
    "04_pronoun_trajectory": lambda a, s: F3.fig3d_pronoun_trajectory(a),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-gif", action="store_true", help="write PNGs only, skip the rotation animations"
    )
    args = parser.parse_args()

    root = repo_root()
    out = root / "figures" / "3d"
    out.mkdir(parents=True, exist_ok=True)
    albums = build_album_table(root)
    songs = build_song_table(root)

    for name, fn in RENDERERS.items():
        png = out / f"{name}.png"
        fn(albums, songs).savefig(png)
        print(f"Wrote {png}")
        if not args.no_gif:
            # Rebuild the figure for the animation (savefig above may finalise it).
            gif = out / f"{name}.gif"
            F3.spin(fn(albums, songs), gif)
            print(f"Wrote {gif}")


if __name__ == "__main__":
    main()

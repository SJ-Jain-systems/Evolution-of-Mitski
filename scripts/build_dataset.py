#!/usr/bin/env python3
"""Join the cleaned lyrics with album metadata and write the tidy analysis
tables the report is built on:

    data/processed/album_stats.csv   one row per studio album
    data/processed/song_stats.csv    one row per canonical album track

Run from the repo root:  python scripts/build_dataset.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis.data import build_album_table, build_song_table, repo_root  # noqa: E402


def main() -> None:
    root = repo_root()
    out = root / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)

    albums = build_album_table(root)
    songs = build_song_table(root)

    albums.to_csv(out / "album_stats.csv", index=False)
    songs.to_csv(out / "song_stats.csv", index=False)

    print(f"Wrote {out / 'album_stats.csv'}  ({len(albums)} albums)")
    print(f"Wrote {out / 'song_stats.csv'}  ({len(songs)} canonical tracks)")
    print()
    cols = [
        "album",
        "release_year",
        "total_words",
        "duration_minutes",
        "words_per_minute",
        "mattr",
        "repetition_index",
    ]
    print(albums[cols].round(2).to_string(index=False))


if __name__ == "__main__":
    main()

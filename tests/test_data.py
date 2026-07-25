"""Integration tests for the lyrics <-> album-metadata join."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis.data import (
    build_album_table,
    build_distinctive_table,
    build_song_table,
    load_album_metadata,
)


def test_all_canonical_tracks_resolve():
    # build_song_table raises if any canonical track is missing; a clean
    # return means every album track matched a corpus entry.
    songs = build_song_table()
    meta = load_album_metadata()
    expected = sum(len(a["tracks"]) for a in meta["albums"])
    assert len(songs) == expected


def test_seven_albums_in_order():
    albums = build_album_table()
    assert len(albums) == 7
    years = albums["release_year"].tolist()
    assert years == sorted(years)  # sorted by release date


def test_words_per_minute_is_positive_and_reasonable():
    albums = build_album_table()
    wpm = albums["words_per_minute"]
    assert (wpm > 20).all() and (wpm < 120).all()


def test_the_land_is_the_sparsest():
    albums = build_album_table()
    sparsest = albums.loc[albums["words_per_minute"].idxmin(), "album"]
    assert sparsest.startswith("The Land")


def test_repetition_index_matches_definition():
    albums = build_album_table()
    row = albums.iloc[0]
    assert abs(row["repetition_index"] - row["total_words"] / row["unique_words"]) < 1e-9


def test_mean_valence_column_present_and_bounded():
    albums = build_album_table()
    assert "mean_valence" in albums.columns
    # AFINN scores span -5..+5; album means must sit well inside that range.
    assert (albums["mean_valence"].abs() < 5).all()


def test_rhyme_columns_present_and_bounded():
    albums = build_album_table()
    for col in ("rhyme_density", "rhyme_coverage", "end_rhyme_pairs_per_line"):
        assert col in albums.columns
    # Densities and coverage are shares in [0, 1].
    assert (albums["rhyme_density"].between(0, 1)).all()
    assert (albums["rhyme_coverage"].between(0, 1)).all()
    # The bundled pronunciation subset should know almost every line-end word.
    assert (albums["rhyme_coverage"] > 0.8).all()


def test_song_table_has_rhyme_columns():
    songs = build_song_table()
    for col in ("rhyme_density", "rhyme_coverage", "end_rhyme_pairs_per_line"):
        assert col in songs.columns


def test_section_columns_present_and_sane():
    albums = build_album_table()
    for col in (
        "sec_section_count",
        "sec_chorus_share",
        "sec_has_bridge",
        "sec_section_repetition",
    ):
        assert col in albums.columns
    # Shares are in [0, 1]; every album has at least one section per song.
    assert (albums["sec_chorus_share"].between(0, 1)).all()
    assert (albums["sec_has_bridge"].between(0, 1)).all()
    assert (albums["sec_section_count"] >= 1).all()


def test_song_table_has_section_columns():
    songs = build_song_table()
    for col in ("sec_section_count", "sec_chorus_share", "sec_has_bridge"):
        assert col in songs.columns


def test_distinctive_table_one_row_per_album_with_words():
    d = build_distinctive_table(n=6)
    assert len(d) == 7
    # Each album gets a non-empty, comma-joined word list of the requested size.
    for words in d["words"]:
        assert words and len(words.split(", ")) == 6


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)

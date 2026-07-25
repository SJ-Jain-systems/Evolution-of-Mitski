"""Load the cleaned lyrics corpus and album metadata, and join them into the
tidy per-song and per-album tables that every figure and the report consume.

The corpus (``data/processed/mitski_lyrics_clean.json``) is a flat list of 153
song entries with no album, date, or duration fields, and it mixes canonical
album tracks with live versions, demos, translations, covers, and non-album
singles. ``data/metadata/albums.json`` supplies the canonical studio-album
tracklists plus release dates and durations. Joining the two is what makes the
words-per-minute-over-time analysis possible.
"""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from . import text as T


def repo_root(start: Path | None = None) -> Path:
    """Walk upward from ``start`` (default: this file) until we find the repo
    root, identified by the presence of the ``data`` directory."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "data" / "metadata" / "albums.json").exists():
            return parent
    raise FileNotFoundError("Could not locate repo root (data/metadata/albums.json).")


def _match_key(title: str) -> str:
    """Normalize a song title to a punctuation-insensitive match key so the
    corpus's curly apostrophes / slashes line up with the metadata's ASCII
    spellings. 'First Love / Late Spring' -> 'first love late spring'."""
    title = unicodedata.normalize("NFKC", title)
    title = title.replace("’", "'").replace("‘", "'")
    title = title.lower()
    title = re.sub(r"[^a-z0-9 ]+", " ", title)  # drop apostrophes, slashes, punctuation
    return re.sub(r"\s+", " ", title).strip()


def load_corpus(root: Path | None = None) -> dict:
    root = root or repo_root()
    path = root / "data" / "processed" / "mitski_lyrics_clean.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_album_metadata(root: Path | None = None) -> dict:
    root = root or repo_root()
    path = root / "data" / "metadata" / "albums.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_song_table(root: Path | None = None) -> pd.DataFrame:
    """One row per canonical studio-album track, with lyric text, metrics, and
    its album/date. Raises if a listed canonical track is missing from the
    corpus, so metadata drift can never silently drop a song."""
    root = root or repo_root()
    corpus = load_corpus(root)
    meta = load_album_metadata(root)

    by_key: dict[str, dict] = {}
    for song in corpus["songs"]:
        by_key.setdefault(_match_key(song["title"]), song)

    pron_lexicon = T.load_pronunciation_lexicon(root)

    rows = []
    missing = []
    for album in meta["albums"]:
        for position, track in enumerate(album["tracks"], start=1):
            song = by_key.get(_match_key(track))
            if song is None:
                missing.append(f"{album['album']} :: {track}")
                continue
            lyrics = song["lyrics"]
            summary = T.lexical_summary(lyrics)
            row = {
                "album": album["album"],
                "album_no": album["album_no"],
                "release_date": pd.Timestamp(album["release_date"]),
                "release_year": pd.Timestamp(album["release_date"]).year,
                "track_no": position,
                "title": track,
                "corpus_title": song["title"],
                "word_count": summary["tokens"],
                **{k: v for k, v in summary.items() if k != "tokens"},
            }
            row.update({f"pron_{k}": v for k, v in T.pronoun_counts(lyrics).items()})
            row.update({f"motif_{k}": v for k, v in T.motif_counts(lyrics).items()})
            row.update(T.rhyme_summary(lyrics, pron_lexicon))
            row.update({f"sec_{k}": v for k, v in T.section_summary(lyrics).items()})
            rows.append(row)

    if missing:
        raise ValueError("Canonical tracks missing from the corpus:\n  " + "\n  ".join(missing))

    return pd.DataFrame(rows)


def build_album_table(root: Path | None = None) -> pd.DataFrame:
    """One row per studio album with aggregate lyric statistics.

    Word counts and lexical metrics are computed on the *concatenation* of the
    album's canonical lyrics (so diversity is measured across the record as a
    whole, matching the video's album-level framing), while durations come from
    the metadata file.
    """
    root = root or repo_root()
    meta = load_album_metadata(root)
    corpus = load_corpus(root)

    by_key: dict[str, dict] = {}
    for song in corpus["songs"]:
        by_key.setdefault(_match_key(song["title"]), song)

    valence_lexicon = T.load_valence_lexicon(root)
    pron_lexicon = T.load_pronunciation_lexicon(root)

    rows = []
    for album in meta["albums"]:
        texts = [by_key[_match_key(t)]["lyrics"] for t in album["tracks"]]
        blob = "\n".join(texts)
        summary = T.lexical_summary(blob)
        duration_min = album["duration_seconds"] / 60.0
        total_words = summary["tokens"]
        n_tracks = len(album["tracks"])
        pron = T.pronoun_counts(blob)
        motif = T.motif_counts(blob)
        vstats = T.valence_stats(blob, valence_lexicon)
        rsum = T.rhyme_summary(blob, pron_lexicon)
        # Section structure is a per-song property, so average each track's
        # section metrics rather than parsing the concatenated blob (which would
        # merge songs and conflate their section counts / bridges).
        sec_per_track = [T.section_summary(t) for t in texts]
        sec_avg = {
            k: sum(s[k] for s in sec_per_track) / len(sec_per_track) for k in sec_per_track[0]
        }
        row = {
            "album": album["album"],
            "album_no": album["album_no"],
            "release_date": pd.Timestamp(album["release_date"]),
            "release_year": pd.Timestamp(album["release_date"]).year,
            "n_tracks": n_tracks,
            "duration_seconds": album["duration_seconds"],
            "duration_minutes": duration_min,
            "duration_display": album["duration_display"],
            "total_words": total_words,
            "unique_words": summary["types"],
            "words_per_minute": total_words / duration_min,
            "words_per_track": total_words / n_tracks,
            "ttr": summary["ttr"],
            "mattr": summary["mattr"],
            "guiraud_r": summary["guiraud_r"],
            # Length-robust diversity checks that corroborate MATTR from the
            # opposite direction (MTLD up / Yule's K down = more varied).
            "mtld": summary["mtld"],
            "yules_k": summary["yules_k"],
            "hapax_ratio": summary["hapax_ratio"],
            "mean_word_length": summary["mean_word_length"],
            # Structural (line-level) measures of compression.
            "line_count": summary["line_count"],
            "lines_per_song": summary["line_count"] / n_tracks,
            "mean_line_length": summary["mean_line_length"],
            "refrain_ratio": summary["refrain_ratio"],
            # Video's literal definition: total / unique (mean repetition of a word).
            "repetition_index": total_words / summary["types"],
            # Mean emotional valence of the album's words (AFINN, -5..+5), plus
            # its spread/range so the report can read emotional dispersion, not
            # just the (misleading) average.
            "mean_valence": vstats["valence_mean"],
            "valence_std": vstats["valence_std"],
            "valence_range": vstats["valence_range"],
            "valence_coverage": vstats["valence_coverage"],
            # End-rhyme density of the album's line-final words (from the bundled
            # CMU pronunciation subset); rhyme_coverage caveats how many line-end
            # words the dictionary knows, mirroring valence_coverage.
            "rhyme_density": rsum["rhyme_density"],
            "end_rhyme_pairs_per_line": rsum["end_rhyme_pairs_per_line"],
            "rhyme_coverage": rsum["rhyme_coverage"],
            # Song structure, averaged over the album's tracks: mean sections per
            # song, mean chorus share, share of songs with a bridge, and mean
            # whole-section repetition. (sec_section_count is a per-song mean.)
            "sec_section_count": sec_avg["section_count"],
            "sec_distinct_section_types": sec_avg["distinct_section_types"],
            "sec_chorus_share": sec_avg["chorus_share"],
            "sec_has_bridge": sec_avg["has_bridge"],
            "sec_section_repetition": sec_avg["section_repetition"],
        }
        total_pron = sum(pron.values()) or 1
        for k, v in pron.items():
            row[f"pron_{k}"] = v
            row[f"pron_{k}_share"] = v / total_pron
        for k, v in motif.items():
            row[f"motif_{k}"] = v
            row[f"motif_{k}_per_1k"] = 1000 * v / total_words
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("release_date").reset_index(drop=True)
    return df


def build_distinctive_table(root: Path | None = None, n: int = 8) -> pd.DataFrame:
    """One row per studio album listing the ``n`` words most distinctive of that
    album relative to the pooled rest of the discography.

    Each album's lyrics are scored against every other album's lyrics combined,
    using ``text.distinctive_words`` (weighted log-odds with a Dirichlet prior),
    so the result surfaces what each record is *about* rather than which words
    are simply frequent. The ``words`` column is a ready-to-print comma-joined
    string; ``scored`` keeps the underlying (word, score) pairs.
    """
    root = root or repo_root()
    meta = load_album_metadata(root)
    corpus = load_corpus(root)

    by_key: dict[str, dict] = {}
    for song in corpus["songs"]:
        by_key.setdefault(_match_key(song["title"]), song)

    # Per-album token frequency maps, in release order.
    albums_meta = sorted(meta["albums"], key=lambda a: a["release_date"])
    counts_by_album: list[tuple[dict, object]] = []
    for album in albums_meta:
        blob = "\n".join(by_key[_match_key(t)]["lyrics"] for t in album["tracks"])
        counts_by_album.append((album, T.word_counts(blob)))

    rows = []
    for i, (album, target) in enumerate(counts_by_album):
        background = Counter()
        for j, (_, other) in enumerate(counts_by_album):
            if j != i:
                background.update(other)
        scored = T.distinctive_words(target, background, n=n)
        rows.append(
            {
                "album": album["album"],
                "album_no": album["album_no"],
                "release_year": pd.Timestamp(album["release_date"]).year,
                "words": ", ".join(w for w, _ in scored),
                "scored": scored,
            }
        )
    return pd.DataFrame(rows)


def build_vocab_growth(root: Path | None = None) -> pd.DataFrame:
    """Type-accumulation curve across the whole career, one row per canonical
    track in release order.

    Walking the discography chronologically and tokenizing each song in turn,
    this tracks the cumulative total words and the cumulative count of *distinct*
    words seen so far. The gap between the two curves -- how fast new vocabulary
    keeps arriving -- is the "reaching for a wider vocabulary" claim made visible
    at the level of the whole catalogue rather than per album.
    """
    root = root or repo_root()
    meta = load_album_metadata(root)
    corpus = load_corpus(root)

    by_key: dict[str, dict] = {}
    for song in corpus["songs"]:
        by_key.setdefault(_match_key(song["title"]), song)

    seen: set[str] = set()
    cum_words = 0
    rows = []
    for album in sorted(meta["albums"], key=lambda a: a["release_date"]):
        for track in album["tracks"]:
            tokens = T.tokenize(by_key[_match_key(track)]["lyrics"])
            cum_words += len(tokens)
            seen.update(tokens)
            rows.append(
                {
                    "album": album["album"],
                    "album_no": album["album_no"],
                    "release_year": pd.Timestamp(album["release_date"]).year,
                    "title": track,
                    "cumulative_words": cum_words,
                    "cumulative_types": len(seen),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":  # quick smoke check
    albums = build_album_table()
    cols = [
        "album",
        "release_year",
        "total_words",
        "duration_minutes",
        "words_per_minute",
        "mattr",
        "repetition_index",
    ]
    print(albums[cols].to_string(index=False))

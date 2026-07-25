# Data

The analysis reads three committed inputs and derives everything else. The
generated tables under `processed/` are git-ignored build artifacts — rebuild
them with `python scripts/build_dataset.py`.

## Inputs (committed)

| Path                                  | What it is                                                                 |
| ------------------------------------- | -------------------------------------------------------------------------- |
| `raw/mitski_lyrics.json`              | Original Genius scrape, 153 entries (with scraper header noise).           |
| `processed/mitski_lyrics_clean.json`  | Cleaned lyric text, same shape, produced by `scripts/clean_lyrics.py`.     |
| `metadata/albums.json`                | Curated: 7 studio albums with `album_no`, `release_date`, `duration_seconds`, `duration_display`, `duration_source`, and canonical `tracks[]`. |
| `lexicons/valence_afinn.txt`          | Bundled AFINN valence lexicon (`word<TAB>score`, −5..+5), ODbL.            |
| `lexicons/pronunciations_cmudict.txt` | CMU Pronouncing Dictionary subset (`word<TAB>ARPABET`), corpus vocabulary only, public domain. Regenerate with `scripts/build_pron_lexicon.py`. |

## Derived tables

Built by `src/mitski_analysis/data.py`. All metrics are defined and unit-tested
in `src/mitski_analysis/text.py`.

### `build_album_table()` — one row per studio album
Identity: `album`, `album_no`, `release_date`, `release_year`, `n_tracks`,
`duration_seconds`, `duration_minutes`, `duration_display`.

Density: `total_words`, `unique_words`, `words_per_minute`, `words_per_track`,
`repetition_index` (= total ÷ unique, the source video's definition).

Diversity (length-robust): `ttr`, `mattr`, `guiraud_r`, `mtld`, `yules_k`,
`hapax_ratio`, `mean_word_length`.

Structure / compression: `line_count`, `lines_per_song`, `mean_line_length`,
`refrain_ratio` (share of lines that repeat an earlier line).

Sentiment (AFINN): `mean_valence`, `valence_std`, `valence_range`,
`valence_coverage` (share of tokens the lexicon scored).

Rhyme (CMU pronunciations): `rhyme_density` (share of line-ends that rhyme with a
neighbour), `end_rhyme_pairs_per_line`, `rhyme_coverage` (share of line-final
words the pronunciation dictionary knows).

Structure (from section tags): `sec_section_count` (mean sections per song),
`sec_distinct_section_types`, `sec_chorus_share` (share of lines in a chorus),
`sec_has_bridge` (share of songs with a bridge), `sec_section_repetition`
(share of whole sections that repeat). Album values are per-song averages.

Pronouns: `pron_{first_singular,second,first_plural}` and their `_share`.

Motifs: `motif_{body,water,fire_light,home_domestic,death,animals}` and their
`_per_1k` (rate per 1,000 words).

### `build_song_table()` — one row per canonical track
The same lexical / structure / pronoun / motif metrics computed per song, plus
`track_no`, `title`, `corpus_title`, `word_count`, `types`. Raises if any
canonical track from the metadata is missing from the corpus, so album totals
can never silently drop a song.

### `build_vocab_growth()` — one row per track, in release order
`album`, `album_no`, `release_year`, `title`, `cumulative_words`,
`cumulative_types` (the career-long type-accumulation curve).

## Notes

Word-level sentiment uses the AFINN valence lexicon (Finn Årup Nielsen, 2011),
released under the Open Database License. Rhyme detection uses a subset of the
CMU Pronouncing Dictionary (Carnegie Mellon University; public domain), trimmed
to the corpus vocabulary so the bundled file stays small and auditable. Album
runtimes were compiled from Discogs, Apple Music, and Wikipedia, with the source
recorded per album in `metadata/albums.json`.

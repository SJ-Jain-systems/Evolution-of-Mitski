"""Re-test the report's headline correlations with robust_stats.

Usage (from the repo root, after `python scripts/build_dataset.py`):
    python scripts/run_robust_stats.py \
        data/processed/album_stats.csv data/processed/song_stats.csv > robust_stats_results.md
"""
import sys
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from mitski_analysis import robust_stats as S

albums = pd.read_csv(sys.argv[1])
songs = pd.read_csv(sys.argv[2])
songs = songs.merge(albums[["album", "words_per_minute", "repetition_index"]], on="album", how="left")

# (label, album-level x, y, optional subset predicate)
TESTS = [
    ("Density falls over time: words/min vs year", "release_year", "words_per_minute", None),
    ("Sparser = less repetitive: words/min vs repetition", "words_per_minute", "repetition_index", None),
    ("Sparser = wider vocabulary: words/min vs MATTR", "words_per_minute", "mattr", None),
    ("Mature era (album 4+): words/min vs repetition", "words_per_minute", "repetition_index", albums.album_no >= 4),
    ("'We' rises with year", "release_year", "pron_first_plural_share", None),
    ("Death imagery rises with year", "release_year", "motif_death_per_1k", None),
    ("Valence vs year", "release_year", "mean_valence", None),
]

rows, reps = [], []
for name, xc, yc, mask in TESTS:
    d = albums if mask is None else albums[mask]
    rep = S.correlation_report(d[xc], d[yc], labels=d["album"])
    reps.append((name, rep, len(d)))
    rows.append(rep.p_perm)

holm_p = S.holm(rows)
bh_p = S.benjamini_hochberg(rows)

print("# Rigorous re-test of headline correlations\n")
print("Album-level tests use exact permutation (all n! orderings), Fisher CIs, rank")
print("statistics and leave-one-out. p-values are adjusted over all tests below.\n")
print("| Test | n | r | Fisher 95% CI | exact perm p | Holm p | BH q | Spearman | LOO range | sign stable |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (name, rep, n), h, b in zip(reps, holm_p, bh_p):
    print(f"| {name} | {n} | {rep.r:.2f} | {rep.fisher_lo:.2f} to {rep.fisher_hi:.2f} | "
          f"{rep.p_perm:.3f} | {h:.3f} | {b:.3f} | {rep.spearman:.2f} | "
          f"{rep.loo_min:.2f} to {rep.loo_max:.2f} | {'yes' if rep.loo_sign_stable else 'NO'} |")

print("\n## Song-level check (about 75 songs, clustered by album)\n")
print("Song words-per-minute is not available, so song word count stands in for density.")
for name, xc, yc in [("Song word count vs year", "release_year", "word_count"),
                     ("Song MATTR vs year", "release_year", "mattr_x" if "mattr_x" in songs else "mattr"),
                     ("Song refrain ratio vs year", "release_year", "refrain_ratio")]:
    r, lo, hi = S.cluster_bootstrap_r(songs[xc], songs[yc], songs["album"])
    p = S.cluster_permutation_p(songs[xc], songs[yc], songs["album"])
    print(f"- {name}: r = {r:.2f}, cluster-bootstrap 95% CI {lo:.2f} to {hi:.2f}, album-block permutation p = {p:.3f}")

print("\n## Theil-Sen slopes (robust trend per year)\n")
for name, rep, n in reps:
    if "year" in name.lower():
        print(f"- {name}: {rep.slope:+.3f} per year (95% CI {rep.slope_lo:+.3f} to {rep.slope_hi:+.3f})")

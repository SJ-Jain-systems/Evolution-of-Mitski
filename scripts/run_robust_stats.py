#!/usr/bin/env python3
"""Re-test the report's headline correlations with ``robust_stats``.

Run from the repo root, after ``python scripts/build_dataset.py``:

    python scripts/run_robust_stats.py > robust_stats_results.md
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis import robust_stats as RS  # noqa: E402
from mitski_analysis.data import repo_root  # noqa: E402


def main() -> None:
    out = repo_root() / "data" / "processed"
    albums = pd.read_csv(out / "album_stats.csv")
    songs = pd.read_csv(out / "song_stats.csv")

    # (label, x column, y column, optional album subset)
    tests = [
        ("Density falls over time: words/min vs year", "release_year", "words_per_minute", None),
        ("Sparser = less repetitive", "words_per_minute", "repetition_index", None),
        ("Sparser = wider vocabulary (MATTR)", "words_per_minute", "mattr", None),
        (
            "Mature era (album 4+): words/min vs repetition",
            "words_per_minute",
            "repetition_index",
            albums["album_no"] >= 4,
        ),
        ("'We' rises with year", "release_year", "pron_first_plural_share", None),
        ("Death imagery rises with year", "release_year", "motif_death_per_1k", None),
        ("Valence vs year", "release_year", "mean_valence", None),
    ]

    reps = []
    for name, xc, yc, mask in tests:
        d = albums if mask is None else albums[mask]
        reps.append((name, len(d), RS.correlation_report(d[xc], d[yc], labels=d["album"])))

    raw_p = [rep.p_perm for _, _, rep in reps]
    holm_p = RS.holm(raw_p)
    bh_p = RS.benjamini_hochberg(raw_p)

    print("# Rigorous re-test of headline correlations\n")
    print("Album-level tests use exact permutation (all n! orderings), Fisher CIs, rank")
    print("statistics and leave-one-out. p-values are adjusted over all tests below.\n")
    print(
        "| Test | n | r | Fisher 95% CI | exact perm p | Holm p | BH q "
        "| Spearman | LOO range | sign stable |"
    )
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (name, n, rep), h, b in zip(reps, holm_p, bh_p):
        print(
            f"| {name} | {n} | {rep.r:.2f} | {rep.fisher_lo:.2f} to {rep.fisher_hi:.2f} | "
            f"{rep.p_perm:.3f} | {h:.3f} | {b:.3f} | {rep.spearman:.2f} | "
            f"{rep.loo_min:.2f} to {rep.loo_max:.2f} | "
            f"{'yes' if rep.loo_sign_stable else 'NO'} |"
        )

    print("\n## Song-level check (about 75 songs, clustered by album)\n")
    print("Song words-per-minute is not available, so song word count stands in for density.")
    for name, yc in [
        ("Song word count vs year", "word_count"),
        ("Song MATTR vs year", "mattr"),
        ("Song refrain ratio vs year", "refrain_ratio"),
    ]:
        x, y, g = songs["release_year"], songs[yc], songs["album"]
        r, lo, hi = RS.cluster_bootstrap_r(x, y, g)
        p = RS.cluster_permutation_p(x, y, g)
        print(
            f"- {name}: r = {r:.2f}, cluster-bootstrap 95% CI {lo:.2f} to {hi:.2f}, "
            f"album-block permutation p = {p:.3f}"
        )

    print("\n## Theil-Sen slopes (robust trend per year)\n")
    for name, _, rep in reps:
        if "year" in name.lower():
            print(
                f"- {name}: {rep.slope:+.3f} per year "
                f"(95% CI {rep.slope_lo:+.3f} to {rep.slope_hi:+.3f})"
            )


if __name__ == "__main__":
    main()

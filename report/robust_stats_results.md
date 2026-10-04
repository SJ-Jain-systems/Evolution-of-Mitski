# Rigorous re-test of headline correlations

Album-level tests use exact permutation (all n! orderings), Fisher CIs, rank
statistics and leave-one-out. p-values are adjusted over all tests below.

| Test | n | r | Fisher 95% CI | exact perm p | Holm p | BH q | Spearman | LOO range | sign stable |
|---|---|---|---|---|---|---|---|---|---|
| Density falls over time: words/min vs year | 7 | -0.81 | -0.97 to -0.16 | 0.026 | 0.185 | 0.185 | -0.75 | -0.90 to -0.71 | yes |
| Sparser = less repetitive: words/min vs repetition | 7 | 0.53 | -0.37 to 0.92 | 0.214 | 1.000 | 0.446 | 0.61 | 0.20 to 0.66 | yes |
| Sparser = wider vocabulary: words/min vs MATTR | 7 | -0.07 | -0.78 to 0.72 | 0.872 | 1.000 | 0.872 | -0.36 | -0.49 to 0.20 | NO |
| Mature era (album 4+): words/min vs repetition | 4 | 0.92 | -0.36 to 1.00 | 0.125 | 0.750 | 0.438 | 1.00 | 0.88 to 1.00 | yes |
| 'We' rises with year | 7 | 0.43 | -0.48 to 0.89 | 0.328 | 1.000 | 0.446 | 0.54 | 0.20 to 0.69 | yes |
| Death imagery rises with year | 7 | 0.41 | -0.50 to 0.89 | 0.379 | 1.000 | 0.446 | 0.18 | -0.38 to 0.74 | NO |
| Valence vs year | 7 | 0.39 | -0.51 to 0.88 | 0.382 | 1.000 | 0.446 | 0.18 | 0.15 to 0.56 | yes |

## Song-level check (about 75 songs, clustered by album)

Song words-per-minute is not available, so song word count stands in for density.
- Song word count vs year: r = -0.30, cluster-bootstrap 95% CI -0.51 to 0.00, album-block permutation p = 0.075
- Song MATTR vs year: r = 0.04, cluster-bootstrap 95% CI -0.32 to 0.33, album-block permutation p = 0.795
- Song refrain ratio vs year: r = -0.03, cluster-bootstrap 95% CI -0.34 to 0.32, album-block permutation p = 0.827

## Theil-Sen slopes (robust trend per year)

- Density falls over time: words/min vs year: -1.645 per year (95% CI -2.904 to +0.056)
- 'We' rises with year: +0.003 per year (95% CI -0.003 to +0.012)
- Death imagery rises with year: +0.121 per year (95% CI -0.508 to +0.992)
- Valence vs year: +0.023 per year (95% CI -0.054 to +0.209)

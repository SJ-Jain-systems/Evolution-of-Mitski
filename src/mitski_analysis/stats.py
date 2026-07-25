"""Small statistical helpers for the report.

The findings lean on a handful of Pearson correlations across only seven albums.
A bare ``r`` on n = 7 hides how much the estimate could move, so this module adds
a bootstrap confidence interval: resample the album pairs with replacement, refit
r each time, and report the 2.5th / 97.5th percentiles. It keeps the report
honest about a small sample without pulling in SciPy -- everything here is NumPy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def pearson_r(x, y) -> float:
    """Pearson correlation coefficient, returning 0.0 for a degenerate input."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


@dataclass(frozen=True)
class CorrelationCI:
    """A correlation with a bootstrap confidence interval."""

    r: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:  # e.g. "r = -0.81 (95% CI -0.95 to -0.42)"
        return f"r = {self.r:.2f} (95% CI {self.lo:.2f} to {self.hi:.2f})"


def bootstrap_r(x, y, n_boot: int = 10000, ci: float = 0.95, seed: int = 0) -> CorrelationCI:
    """Pearson r for ``x`` vs ``y`` with a percentile bootstrap CI.

    Pairs ``(x_i, y_i)`` are resampled with replacement ``n_boot`` times; the CI
    is the central ``ci`` percentile band of the resampled correlations. The
    point estimate is the plain correlation on the full sample. Deterministic for
    a given ``seed`` so the report renders identically every time.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of the same length")
    n = x.size
    point = pearson_r(x, y)
    if n < 3:
        return CorrelationCI(point, point, point, n)

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    xs = x[idx]
    ys = y[idx]
    # Vectorised per-row Pearson r.
    xm = xs - xs.mean(axis=1, keepdims=True)
    ym = ys - ys.mean(axis=1, keepdims=True)
    num = (xm * ym).sum(axis=1)
    den = np.sqrt((xm**2).sum(axis=1) * (ym**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        rs = np.where(den > 0, num / den, np.nan)
    rs = rs[np.isfinite(rs)]
    if rs.size == 0:
        return CorrelationCI(point, point, point, n)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.percentile(rs, [100 * alpha, 100 * (1 - alpha)])
    return CorrelationCI(point, float(lo), float(hi), n)

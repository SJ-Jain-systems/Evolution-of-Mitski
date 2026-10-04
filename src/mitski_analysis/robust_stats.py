"""Rigorous small-sample statistics for the Mitski report (NumPy only).

Why this replaces a bare percentile bootstrap on n = 7 albums:
a percentile bootstrap of r on seven points is anti-conservative (too narrow,
and it ignores that duplicates collapse r). This module instead offers

* exact_permutation_p    exact p-value; with n <= 9 every permutation is enumerated
* fisher_ci              Fisher-z CI (analytic, valid for small n if used honestly)
* spearman / kendall     rank statistics robust to one extreme album
* leave_one_out          does one album drive the result?
* theil_sen              robust slope with a bootstrap CI
* cluster_bootstrap_r    song-level resampling clustered by album (n ~ 75, not 7)
* holm / benjamini_hochberg   multiple-comparison control across the many tests
* correlation_report     one call that returns all of the above

Everything is deterministic for a given seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
from math import atanh, erf, sqrt, tanh

import numpy as np


# ---------------------------------------------------------------- basics
def pearson_r(x, y) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _rank(a) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(a.size)
    ranks[order] = np.arange(1, a.size + 1)
    for v in np.unique(a):
        m = a == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return ranks


def spearman_rho(x, y) -> float:
    return pearson_r(_rank(x), _rank(y))


def kendall_tau(x, y) -> float:
    """Kendall tau-b."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    if n < 2:
        return 0.0
    dx = np.sign(x[:, None] - x[None, :])
    dy = np.sign(y[:, None] - y[None, :])
    iu = np.triu_indices(n, 1)
    conc = (dx * dy)[iu].sum()
    tx = (dx[iu] != 0).sum()
    ty = (dy[iu] != 0).sum()
    if tx == 0 or ty == 0:
        return 0.0
    return float(conc / sqrt(tx * ty))


# ---------------------------------------------------- exact / MC permutation
def permutation_p(x, y, stat=pearson_r, n_perm: int = 20000, seed: int = 0,
                  exact_max_n: int = 9) -> float:
    """Two-sided permutation p-value for ``stat(x, y)``.

    Exact (all n! orderings) when n <= ``exact_max_n``; otherwise Monte Carlo
    with the +1 correction so p is never reported as exactly 0.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size
    obs = abs(stat(x, y))
    tol = 1e-12
    if n <= exact_max_n:
        count = total = 0
        for p in permutations(range(n)):
            total += 1
            if abs(stat(x, y[list(p)])) >= obs - tol:
                count += 1
        return count / total
    rng = np.random.default_rng(seed)
    hits = sum(abs(stat(x, rng.permutation(y))) >= obs - tol for _ in range(n_perm))
    return (hits + 1) / (n_perm + 1)


# ------------------------------------------------------------- analytic CI
def _norm_ppf(p: float) -> float:
    # Acklam-free: bisection on the normal CDF is plenty accurate here.
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fisher_ci(r: float, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Fisher z interval. Undefined (returns -1, 1) for n <= 3."""
    if n <= 3:
        return -1.0, 1.0
    r = max(min(r, 0.999999), -0.999999)
    z = atanh(r)
    se = 1.0 / sqrt(n - 3)
    q = _norm_ppf(1 - (1 - ci) / 2)
    return tanh(z - q * se), tanh(z + q * se)


# --------------------------------------------------------- robustness checks
@dataclass(frozen=True)
class LeaveOneOut:
    estimates: dict[str, float]
    min: float
    max: float
    sign_stable: bool


def leave_one_out(x, y, labels=None, stat=pearson_r) -> LeaveOneOut:
    """Refit the statistic dropping each unit once."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    labels = list(labels) if labels is not None else [str(i) for i in range(x.size)]
    est = {}
    for i, lab in enumerate(labels):
        keep = np.arange(x.size) != i
        est[lab] = stat(x[keep], y[keep])
    vals = list(est.values())
    full = stat(x, y)
    return LeaveOneOut(est, min(vals), max(vals), all(np.sign(v) == np.sign(full) for v in vals))


def theil_sen(x, y, n_boot: int = 5000, ci: float = 0.95, seed: int = 0):
    """Median-of-pairwise-slopes estimator with percentile bootstrap CI."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)

    def slope(xx, yy):
        i, j = np.triu_indices(xx.size, 1)
        dx = xx[j] - xx[i]
        ok = dx != 0
        return float(np.median((yy[j] - yy[i])[ok] / dx[ok])) if ok.any() else 0.0

    point = slope(x, y)
    rng = np.random.default_rng(seed)
    bs = []
    for _ in range(n_boot):
        idx = rng.integers(0, x.size, x.size)
        bs.append(slope(x[idx], y[idx]))
    a = (1 - ci) / 2
    lo, hi = np.percentile(bs, [100 * a, 100 * (1 - a)])
    return point, float(lo), float(hi)


# ------------------------------------------------- song-level / clustered
def cluster_bootstrap_r(x, y, groups, n_boot: int = 5000, ci: float = 0.95, seed: int = 0):
    """Two-stage bootstrap of r on song-level data.

    Resample albums (clusters) with replacement, then resample songs within each
    chosen album. This respects that songs on one album are not independent, while
    using the ~75 songs rather than 7 album means. Returns (r, lo, hi).
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    ids = np.unique(groups)
    members = {g: np.flatnonzero(groups == g) for g in ids}
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n_boot):
        picked = rng.choice(ids, size=ids.size, replace=True)
        idx = np.concatenate([rng.choice(members[g], size=members[g].size) for g in picked])
        r = pearson_r(x[idx], y[idx])
        if np.isfinite(r):
            rs.append(r)
    a = (1 - ci) / 2
    lo, hi = np.percentile(rs, [100 * a, 100 * (1 - a)])
    return pearson_r(x, y), float(lo), float(hi)


def cluster_permutation_p(x, y, groups, n_perm: int = 10000, seed: int = 0) -> float:
    """Permute the x-values *between albums* (whole-album blocks), so the null
    respects the clustering. Needs song-level x constant within an album, e.g.
    release year; use only for album-level predictors."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    groups = np.asarray(groups)
    ids = np.unique(groups)
    album_x = np.array([x[groups == g][0] for g in ids])
    obs = abs(pearson_r(x, y))
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_perm):
        shuffled = dict(zip(ids, rng.permutation(album_x)))
        xp = np.array([shuffled[g] for g in groups])
        hits += abs(pearson_r(xp, y)) >= obs - 1e-12
    return (hits + 1) / (n_perm + 1)


# ----------------------------------------------- multiple-comparison control
def holm(pvals) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (family-wise error control)."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    m = p.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def benjamini_hochberg(pvals) -> np.ndarray:
    """BH adjusted p-values (false-discovery-rate control)."""
    p = np.asarray(pvals, float)
    m = p.size
    order = np.argsort(p)[::-1]
    adj = np.empty(m)
    prev = 1.0
    for k, i in enumerate(order):
        rank = m - k
        prev = min(prev, p[i] * m / rank)
        adj[i] = prev
    return adj


# ------------------------------------------------------------- one-call API
@dataclass
class CorrelationReport:
    n: int
    r: float
    fisher_lo: float
    fisher_hi: float
    p_perm: float
    spearman: float
    p_spearman: float
    kendall: float
    loo_min: float
    loo_max: float
    loo_sign_stable: bool
    slope: float
    slope_lo: float
    slope_hi: float
    extras: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return (f"r = {self.r:.2f} (Fisher 95% CI {self.fisher_lo:.2f} to {self.fisher_hi:.2f}), "
                f"exact permutation p = {self.p_perm:.3f}; Spearman rho = {self.spearman:.2f}; "
                f"leave-one-out r in [{self.loo_min:.2f}, {self.loo_max:.2f}]")


def correlation_report(x, y, labels=None, seed: int = 0) -> CorrelationReport:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    r = pearson_r(x, y)
    lo, hi = fisher_ci(r, x.size)
    loo = leave_one_out(x, y, labels)
    s, sl, sh = theil_sen(x, y, seed=seed)
    return CorrelationReport(
        n=x.size, r=r, fisher_lo=lo, fisher_hi=hi,
        p_perm=permutation_p(x, y, pearson_r, seed=seed),
        spearman=spearman_rho(x, y),
        p_spearman=permutation_p(x, y, spearman_rho, seed=seed),
        kendall=kendall_tau(x, y),
        loo_min=loo.min, loo_max=loo.max, loo_sign_stable=loo.sign_stable,
        slope=s, slope_lo=sl, slope_hi=sh,
    )

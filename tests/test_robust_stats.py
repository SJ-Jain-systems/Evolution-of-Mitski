import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis import robust_stats as S


def test_exact_perm_perfect_correlation_n5():
    x = [1, 2, 3, 4, 5]
    assert abs(S.permutation_p(x, x) - 2 / 120) < 1e-12  # r=1 and r=-1 orderings


def test_exact_perm_null_is_large():
    rng = np.random.default_rng(3)
    assert S.permutation_p(rng.normal(size=7), rng.normal(size=7)) > 0.05


def test_spearman_monotone_nonlinear():
    x = np.arange(1, 8)
    assert abs(S.spearman_rho(x, x**3) - 1) < 1e-12


def test_kendall_reversed():
    assert abs(S.kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) + 1) < 1e-12


def test_fisher_ci_contains_r_and_widens_small_n():
    lo1, hi1 = S.fisher_ci(0.8, 7)
    lo2, hi2 = S.fisher_ci(0.8, 70)
    assert lo1 < 0.8 < hi1 and (hi1 - lo1) > (hi2 - lo2)


def test_holm_and_bh_monotone():
    p = [0.01, 0.04, 0.03, 0.5]
    h = S.holm(p)
    b = S.benjamini_hochberg(p)
    assert all(h >= np.asarray(p)) and all(b <= h + 1e-12)
    assert abs(h[0] - 0.04) < 1e-12


def test_loo_detects_driver():
    x = np.arange(7.0)
    y = np.array([0, 0, 0, 0, 0, 0, 10.0])
    loo = S.leave_one_out(x, y)
    assert loo.min < 0.2 < loo.max


def test_cluster_bootstrap_brackets_point():
    rng = np.random.default_rng(1)
    g = np.repeat(np.arange(6), 10)
    x = g + rng.normal(size=60) * 0.1
    y = x + rng.normal(size=60)
    r, lo, hi = S.cluster_bootstrap_r(x, y, g, n_boot=500)
    assert lo <= r <= hi


def test_theil_sen_recovers_slope():
    x = np.arange(10.0)
    p, lo, hi = S.theil_sen(x, 2 * x + 1, n_boot=200)
    assert abs(p - 2) < 1e-9

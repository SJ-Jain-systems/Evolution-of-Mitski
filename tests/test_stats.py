"""Unit tests for the bootstrap correlation helper."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mitski_analysis import stats as ST


def test_pearson_r_matches_numpy():
    x = [1, 2, 3, 4, 5]
    y = [2, 1, 4, 3, 6]
    assert abs(ST.pearson_r(x, y) - np.corrcoef(x, y)[0, 1]) < 1e-12


def test_pearson_r_degenerate_is_zero():
    assert ST.pearson_r([1, 1, 1], [1, 2, 3]) == 0.0
    assert ST.pearson_r([1.0], [2.0]) == 0.0


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(1)
    x = rng.normal(size=40)
    y = 0.8 * x + rng.normal(scale=0.5, size=40)
    ci = ST.bootstrap_r(x, y, n_boot=2000, seed=0)
    assert ci.lo <= ci.r <= ci.hi
    assert -1.0 <= ci.lo <= ci.hi <= 1.0
    assert ci.n == 40


def test_bootstrap_is_deterministic_for_a_seed():
    x = [1, 2, 3, 4, 5, 6, 7]
    y = [2, 1, 3, 5, 4, 7, 6]
    a = ST.bootstrap_r(x, y, n_boot=1000, seed=42)
    b = ST.bootstrap_r(x, y, n_boot=1000, seed=42)
    assert (a.r, a.lo, a.hi) == (b.r, b.lo, b.hi)


def test_str_is_human_readable():
    ci = ST.bootstrap_r([1, 2, 3, 4], [1, 2, 3, 4], n_boot=200, seed=0)
    s = str(ci)
    assert s.startswith("r = ") and "95% CI" in s


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

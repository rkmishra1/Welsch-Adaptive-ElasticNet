"""Runnable self-checks: exact sparsity, support recovery, RBIC selection.

    python -m pytest tests/      # or just: python tests/test_estimator.py
"""

import numpy as np

from welsch_adenet import welsch_adenet, welsch_loss, welsch_psi, fit_rbic
from simulation.dgp import make_dataset
from simulation.metrics import selection_metrics


def test_welsch_derivative():
    # finite-difference check of W_c' against W_c
    c, u, h = 1.7, np.array([0.3, 1.5, -4.0]), 1e-6
    num = (welsch_loss(u + h, c) - welsch_loss(u - h, c)) / (2 * h)
    assert np.allclose(num, welsch_psi(u, c), atol=1e-4)


def test_exact_zeros_and_recovery():
    rng = np.random.default_rng(1)
    d = make_dataset(n=300, p=40, rho=0.5, error="clean", rng=rng)
    res = fit_rbic(d["X"], d["y"])
    beta = res["beta"]
    # produces exact zeros (soft-threshold prox works)
    assert np.sum(np.abs(beta) <= 1e-8) > 0
    # recovers most of the true support
    tz, fz = selection_metrics(beta, d["beta_star"])
    true_active = set(d["active"])
    est_active = set(np.flatnonzero(np.abs(beta) > 1e-8))
    recovered = len(true_active & est_active) / len(true_active)
    assert recovered >= 0.6, f"only recovered {recovered:.0%} of support"


def test_robust_under_outliers():
    # with 10% vertical outliers W-AdEnet should still keep MSPE bounded
    rng = np.random.default_rng(2)
    d = make_dataset(n=300, p=40, rho=0.5, error="vertical", rng=rng)
    res = fit_rbic(d["X"], d["y"])
    resid = d["y_test"] - d["X_test"] @ res["beta"]
    assert np.median(resid ** 2) < 10.0


def test_competitors():
    # test that we can fit each competitor and get coefficients
    rng = np.random.default_rng(42)
    d = make_dataset(n=100, p=10, rho=0.5, error="clean", rng=rng)
    X, y = d["X"], d["y"]
    
    from welsch_adenet import fit_competitor, tune_competitor
    for loss in ["squared", "absolute", "huber", "tukey", "welsch"]:
        res = tune_competitor(X, y, loss_type=loss, n_l1=5, n_l2=2)
        assert res["beta"] is not None
        assert len(res["beta"]) == 10


if __name__ == "__main__":
    test_welsch_derivative()
    test_exact_zeros_and_recovery()
    test_robust_under_outliers()
    test_competitors()
    print("all checks passed")

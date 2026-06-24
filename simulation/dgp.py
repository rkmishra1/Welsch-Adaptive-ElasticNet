"""Data-generating process for the simulation study (Section 6, eq 6.1).

    y_i = x_i' beta* + e_i,   x_i ~ N(0, Sigma),   |A| = floor(p/3).

Active positions are random per replication, active coefficients are
heterogeneous with mixed signs, and the rest are exactly zero. Three error
regimes (clean / vertical outliers / leverage points) share one outlier
magnitude and contamination fraction delta, so they differ only in *where* the
corruption enters.
"""

import numpy as np


def _cov(p, rho, kind):
    """Compound-symmetry or AR(1) covariance Sigma (unit variances)."""
    if kind == "cs":
        S = np.full((p, p), rho)
        np.fill_diagonal(S, 1.0)
        return S
    if kind == "ar1":
        idx = np.arange(p)
        return rho ** np.abs(idx[:, None] - idx[None, :])
    raise ValueError(f"unknown covariance kind: {kind}")


def _true_beta(p, rng):
    """Sparse beta* with |A|=floor(p/3) heterogeneous, mixed-sign signals."""
    beta = np.zeros(p)
    k = p // 3
    active = rng.choice(p, size=k, replace=False)
    mags = rng.uniform(0.5, 2.5, size=k)
    signs = rng.choice([-1.0, 1.0], size=k)
    beta[active] = mags * signs
    return beta, np.sort(active)


def make_dataset(n, p, rho, *, cov="ar1", error="clean", delta=0.10,
                 sigma=1.0, outlier_mag=20.0, n_test=1000, rng=None):
    """Generate one (train, test) replication.

    error in {"clean", "vertical", "leverage"}; contamination hits the training
    set only, so test MSPE is measured on uncorrupted data.

    Returns a dict with X, y, X_test, y_test, beta_star, active.
    """
    rng = np.random.default_rng() if rng is None else rng
    beta, active = _true_beta(p, rng)
    L = np.linalg.cholesky(_cov(p, rho, cov) + 1e-10 * np.eye(p))

    def draw(m):
        X = rng.standard_normal((m, p)) @ L.T
        y = X @ beta + sigma * rng.standard_normal(m)
        return X, y

    X, y = draw(n)
    X_test, y_test = draw(n_test)

    if error != "clean":
        m = int(np.floor(delta * n))
        bad = rng.choice(n, size=m, replace=False)
        # vertical outliers: shift the responses by a common large magnitude
        y[bad] += outlier_mag * sigma * rng.choice([-1.0, 1.0], size=m)
        if error == "leverage":
            # also corrupt the design rows -> high-leverage bad points
            X[bad] = outlier_mag + rng.standard_normal((m, p))

    return {"X": X, "y": y, "X_test": X_test, "y_test": y_test,
            "beta_star": beta, "active": active}

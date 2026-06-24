"""RBIC tuning for W-AdEnet over the (lambda_1, lambda_2) grid -- eq (5.8).

    RBIC(l1, l2) = T_d(beta_hat, sigma_hat) + log(n) * |A|

T_d is the robust deviance (sum of the model's bounded Welsch loss over residuals)
and |A| is the number of non-zero coefficients. We minimize over a 2-D grid,
warm-starting each fit from the previous solution to keep the path smooth.

Per the manuscript design, RBIC is the tuning rule for W-AdEnet only; competing
methods are tuned by their own (cross-validation) protocol.
"""

import numpy as np

from .estimator import welsch_adenet, welsch_loss
from .init_scale import robust_init, adaptive_weights


def rbic_score(X, y, beta, sigma, c):
    """RBIC objective -- eq (5.8). Lower is better."""
    n = X.shape[0]
    r = y - X @ beta
    deviance = welsch_loss(r / sigma, c).sum()  # T_d: robust fit term
    active = int(np.sum(np.abs(beta) > 1e-8))
    return deviance + np.log(n) * active


def default_grid(X, y, sigma, n_l1=20, n_l2=5):
    """Log-spaced lambda_1 grid (data-driven max) crossed with a small lambda_2 grid."""
    n = X.shape[0]
    # lambda_max: smallest l1 that zeroes all coefficients at beta=0 (KKT-style)
    l1_max = np.max(np.abs(X.T @ (y / sigma))) / n
    l1_grid = np.logspace(np.log10(l1_max), np.log10(l1_max * 1e-3), n_l1)
    l2_grid = np.concatenate([[0.0], np.logspace(-3, 0, n_l2 - 1)])
    return l1_grid, l2_grid


def fit_rbic(X, y, c=2.11, gamma=1.0, l1_grid=None, l2_grid=None, **adam_kw):
    """Fit W-AdEnet selecting (lambda_1, lambda_2) by RBIC.

    Returns dict with beta, lambda1, lambda2, sigma, weights, rbic, and the
    full path of (l1, l2, rbic) scores.
    """
    beta0, sigma = robust_init(X, y)
    w = adaptive_weights(beta0, gamma=gamma)

    if l1_grid is None or l2_grid is None:
        l1_grid, l2_grid = default_grid(X, y, sigma)

    best = {"rbic": np.inf}
    path = []
    for l2 in l2_grid:
        warm = beta0.copy()  # warm-start each l1 sweep from the robust init
        for l1 in l1_grid:
            beta = welsch_adenet(X, y, l1, l2, w, sigma, c=c, beta0=warm, **adam_kw)
            warm = beta
            score = rbic_score(X, y, beta, sigma, c)
            path.append((l1, l2, score))
            if score < best["rbic"]:
                best = {"beta": beta, "lambda1": l1, "lambda2": l2,
                        "sigma": sigma, "weights": w, "rbic": score}

    best["path"] = np.array(path)
    return best

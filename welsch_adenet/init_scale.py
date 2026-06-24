"""Robust initialization, scale, and adaptive weights (Section 5: Initialization).

The manuscript uses an MM-estimator when p < n and MM-Ridge when p >> n to land
in a good basin of attraction. We approximate that with a high-breakdown robust
fit: HuberRegressor when p < n, ridge otherwise. The init supplies (i) the warm
start beta0, (ii) the robust scale hat_sigma via normalized MAD of its residuals,
and (iii) the adaptive weights hat_w_j = 1/|beta0_j|^gamma (eq 5.1).
"""

import numpy as np
from sklearn.linear_model import HuberRegressor, Ridge

MAD_CONST = 1.4826  # MAD -> sigma for Gaussian


def robust_init(X, y, ridge_alpha=1.0):
    """Return (beta0, sigma): robust warm start and scale estimate."""
    n, p = X.shape
    if p < n:
        # MM-like: Huber M-estimate, no intercept (DGP is centered)
        fit = HuberRegressor(fit_intercept=False, alpha=0.0, max_iter=500)
        try:
            fit.fit(X, y)
            beta0 = fit.coef_
        except Exception:
            beta0 = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(X, y).coef_
    else:
        # MM-Ridge surrogate for the wide regime
        beta0 = Ridge(alpha=ridge_alpha, fit_intercept=False).fit(X, y).coef_

    r = y - X @ beta0
    sigma = MAD_CONST * np.median(np.abs(r - np.median(r)))
    sigma = max(sigma, 1e-3)  # guard against a degenerate (near-zero) scale
    return beta0, sigma


def adaptive_weights(beta0, gamma=1.0, eps=1e-4):
    """hat_w_j = 1 / (|beta0_j| + eps)^gamma  -- adaptive (Zou 2006) weights."""
    return 1.0 / (np.abs(beta0) + eps) ** gamma

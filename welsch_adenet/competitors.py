"""Unified competitor solver and robust BIC tuning.

Supports various loss functions (squared, absolute/LAD, Huber, Tukey, Welsch)
with adaptive elastic-net penalty, solved via proximal Adam and tuned by Robust BIC.
"""

import numpy as np
from .init_scale import robust_init, adaptive_weights

# 1. Loss Functions and Derivatives (Psi)

def loss_squared(u):
    return 0.5 * u * u

def psi_squared(u):
    return u

def loss_absolute(u):
    return np.abs(u)

def psi_absolute(u):
    return np.sign(u)

def loss_huber(u, k=1.345):
    abs_u = np.abs(u)
    return np.where(abs_u <= k, 0.5 * u * u, k * abs_u - 0.5 * k * k)

def psi_huber(u, k=1.345):
    return np.where(np.abs(u) <= k, u, k * np.sign(u))

def loss_tukey(u, d=4.685):
    abs_u = np.abs(u)
    inside = (d**2 / 6.0) * (1.0 - (1.0 - (u / d) ** 2) ** 3)
    outside = d**2 / 6.0
    return np.where(abs_u <= d, inside, outside)

def psi_tukey(u, d=4.685):
    return np.where(np.abs(u) <= d, u * (1.0 - (u / d) ** 2) ** 2, 0.0)

def loss_welsch(u, c=2.11):
    return c * c * (1.0 - np.exp(-(u * u) / (2.0 * c * c)))

def psi_welsch(u, c=2.11):
    return u * np.exp(-(u * u) / (2.0 * c * c))


LOSS_REGISTRY = {
    "squared": (loss_squared, psi_squared),
    "absolute": (loss_absolute, psi_absolute),
    "huber": (loss_huber, psi_huber),
    "tukey": (loss_tukey, psi_tukey),
    "welsch": (loss_welsch, psi_welsch),
}

# 2. Unified Solver

def _soft_ridge_prox(v, step, l1, l2, w):
    thresh = step * l1 * w
    shrunk = np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)
    return shrunk / (1.0 + step * l2)

def fit_competitor(
    X, y, l1, l2, weights, sigma, loss_type="squared",
    eta=1e-2, omega1=0.9, omega2=0.999, eps=1e-8,
    beta0=None, max_iter=2000, tol=1e-6, loss_param=None
):
    """Fit a model with a given loss function and adaptive elastic-net penalty."""
    n, p = X.shape
    beta = np.zeros(p) if beta0 is None else beta0.astype(float).copy()
    m = np.zeros(p)
    v = np.zeros(p)

    loss_fn, psi_fn = LOSS_REGISTRY[loss_type]

    # Setup loss parameter (e.g. k, d, c)
    kwargs = {}
    if loss_param is not None:
        if loss_type == "huber":
            kwargs["k"] = loss_param
        elif loss_type == "tukey":
            kwargs["d"] = loss_param
        elif loss_type == "welsch":
            kwargs["c"] = loss_param

    r = y - X @ beta
    prev_loss = loss_fn(r / sigma, **kwargs)

    for t in range(1, max_iter + 1):
        # gradient of the smooth loss term (subgradient for absolute loss)
        g = -(X.T @ psi_fn(r / sigma, **kwargs)) / sigma

        # Adam moments + bias correction
        m = omega1 * m + (1.0 - omega1) * g
        v = omega2 * v + (1.0 - omega2) * (g * g)
        m_hat = m / (1.0 - omega1 ** t)
        v_hat = v / (1.0 - omega2 ** t)

        # per-coordinate step, descent point, prox
        step = eta / (np.sqrt(v_hat) + eps)
        vtent = beta - step * m_hat
        beta = _soft_ridge_prox(vtent, step, l1, l2, weights)

        # stop on average change in the loss
        r = y - X @ beta
        loss = loss_fn(r / sigma, **kwargs)
        if np.linalg.norm(loss - prev_loss) / n < tol:
            break
        prev_loss = loss

    return (1.0 + l2 / n) * beta

# 3. Tuning

def rbic_score(X, y, beta, sigma, loss_type, loss_param=None):
    n = X.shape[0]
    r = y - X @ beta
    loss_fn, _ = LOSS_REGISTRY[loss_type]
    
    kwargs = {}
    if loss_param is not None:
        if loss_type == "huber":
            kwargs["k"] = loss_param
        elif loss_type == "tukey":
            kwargs["d"] = loss_param
        elif loss_type == "welsch":
            kwargs["c"] = loss_param
            
    deviance = loss_fn(r / sigma, **kwargs).sum()
    active = int(np.sum(np.abs(beta) > 1e-8))
    return deviance + np.log(n) * active

def tune_competitor(
    X, y, loss_type="squared", loss_param=None, gamma=1.0, 
    n_l1=20, n_l2=5, l1_grid=None, l2_grid=None, **adam_kw
):
    """Grid search over (l1, l2) minimizing robust BIC."""
    n, p = X.shape
    beta0, sigma = robust_init(X, y)
    
    # LAD-Lasso is usually non-adaptive (we can set weights to 1s)
    # But for other methods we use adaptive weights
    if loss_type == "absolute":
        w = np.ones(p)  # non-adaptive L1 penalty for standard LAD-Lasso
    else:
        w = adaptive_weights(beta0, gamma=gamma)

    if l1_grid is None or l2_grid is None:
        # data-driven lambda_1 max
        l1_max = np.max(np.abs(X.T @ (y / sigma))) / n
        l1_grid = np.logspace(np.log10(l1_max), np.log10(l1_max * 1e-3), n_l1)
        l2_grid = np.concatenate([[0.0], np.logspace(-3, 0, n_l2 - 1)])

    best = {"rbic": np.inf}
    for l2 in l2_grid:
        warm = beta0.copy()
        for l1 in l1_grid:
            beta = fit_competitor(
                X, y, l1, l2, w, sigma, loss_type=loss_type, 
                loss_param=loss_param, beta0=warm, **adam_kw
            )
            warm = beta
            score = rbic_score(X, y, beta, sigma, loss_type, loss_param)
            if score < best["rbic"]:
                best = {
                    "beta": beta, "lambda1": l1, "lambda2": l2,
                    "sigma": sigma, "weights": w, "rbic": score
                }
    return best

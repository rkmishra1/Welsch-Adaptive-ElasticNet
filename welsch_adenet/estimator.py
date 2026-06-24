"""Welsch Adaptive Elastic-Net via proximal Adam (Algorithm 5.1).

Implements equations (5.1)-(5.7) of the manuscript:
    F(beta) = sum_i W_c((y_i - x_i'beta)/sigma)        # smooth, non-convex
              + sum_j ( w_j*l1*|beta_j| + (l2/2)*beta_j^2 )   # convex, non-smooth
solved by a gradient step on the Welsch data term followed by the closed-form
soft-threshold/ridge-shrink prox of the elastic-net penalty, accelerated by Adam.
"""

import numpy as np


def welsch_loss(u, c):
    """Welsch loss W_c(u) = c^2 (1 - exp(-u^2/(2c^2))), elementwise -- eq (5.2).

    Note: eq (5.2) prints the prefactor as c^2/2, but with the exponent -u^2/(2c^2)
    that does not match the stated derivative W_c'(u)=u*exp(-u^2/(2c^2)) used by the
    gradient (5.3) and Algorithm 5.1. We anchor on the operative derivative and use
    c^2 here so that welsch_loss is exactly the antiderivative of welsch_psi.
    """
    return c * c * (1.0 - np.exp(-(u * u) / (2.0 * c * c)))


def welsch_psi(u, c):
    """W_c'(u) = u * exp(-u^2/(2c^2))  -- eq (5.2), elementwise."""
    return u * np.exp(-(u * u) / (2.0 * c * c))


def _soft_ridge_prox(v, step, l1, l2, w):
    """Coordinatewise prox of the elastic-net penalty -- eq (5.6)."""
    thresh = step * l1 * w
    shrunk = np.sign(v) * np.maximum(np.abs(v) - thresh, 0.0)
    return shrunk / (1.0 + step * l2)


def welsch_adenet(
    X, y, l1, l2, weights, sigma, c=2.11,
    eta=1e-2, omega1=0.9, omega2=0.999, eps=1e-8,
    beta0=None, max_iter=2000, tol=1e-6,
):
    """Fit the Welsch adaptive elastic-net for one (l1, l2) pair.

    Parameters
    ----------
    X, y : design and response.
    l1, l2 : sparsity (lambda_1) and ridge (lambda_2) penalties.
    weights : adaptive weights hat_w_j (eq 5.1), shape (p,).
    sigma : robust scale estimate hat_sigma.
    c : Welsch tuning constant (default 2.11 ~ 95% Gaussian efficiency).
    eta, omega1, omega2, eps : Adam hyper-parameters.
    beta0 : warm start (defaults to 0).

    Returns the rescaled estimate hat_beta = (1 + l2/n) * beta* (eq 5.1).
    """
    n, p = X.shape
    beta = np.zeros(p) if beta0 is None else beta0.astype(float).copy()
    m = np.zeros(p)
    v = np.zeros(p)

    r = y - X @ beta
    prev_loss = welsch_loss(r / sigma, c)

    for t in range(1, max_iter + 1):
        # gradient of the smooth Welsch term -- eq (5.3)
        g = -(X.T @ welsch_psi(r / sigma, c)) / sigma

        # Adam moments + bias correction -- lines 6-9
        m = omega1 * m + (1.0 - omega1) * g
        v = omega2 * v + (1.0 - omega2) * (g * g)
        m_hat = m / (1.0 - omega1 ** t)
        v_hat = v / (1.0 - omega2 ** t)

        # per-coordinate step, descent point, prox -- lines 11-13
        step = eta / (np.sqrt(v_hat) + eps)
        vtent = beta - step * m_hat
        beta = _soft_ridge_prox(vtent, step, l1, l2, weights)

        # stop on average change in the Welsch loss -- line 2
        r = y - X @ beta
        loss = welsch_loss(r / sigma, c)
        if np.linalg.norm(loss - prev_loss) / n < tol:
            break
        prev_loss = loss

    return (1.0 + l2 / n) * beta  # undo elastic-net double shrinkage

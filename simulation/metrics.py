"""Selection and prediction metrics (Section 6).

TZ   correct exclusions: # true zeros estimated as zero (larger better, ceil p-|A|)
FZ   false inclusions:    # true zeros incorrectly retained (smaller better)
MSPE mean squared prediction error on the test set (median reported over reps)
"""

import numpy as np


def selection_metrics(beta_hat, beta_star, tol=1e-8):
    true_zero = np.abs(beta_star) <= tol
    est_zero = np.abs(beta_hat) <= tol
    tz = int(np.sum(true_zero & est_zero))
    fz = int(np.sum(true_zero & ~est_zero))
    return tz, fz


def mspe(beta_hat, X_test, y_test):
    resid = y_test - X_test @ beta_hat
    return float(np.mean(resid ** 2))

MAD_CONST <- 1.4826 # MAD -> sigma for Gaussian

#' Robust warm start and scale estimate
#'
#' Approximates the manuscript's MM-estimator initialization: a Huber
#' M-estimate (via [MASS::rlm()]) when `p < n`, and a ridge fit otherwise, to
#' land in a good basin of attraction for [welsch_adenet()]. The residuals of
#' that fit give a robust scale estimate via the normalized median absolute
#' deviation (MAD).
#'
#' @param X Numeric design matrix, `n` by `p`.
#' @param y Numeric response vector, length `n`.
#' @param ridge_alpha Non-negative ridge penalty used for the wide-`p`
#'   surrogate fit, and as a fallback if the Huber fit fails to converge.
#' @return A list with elements `beta0` (numeric vector of length `p`, the
#'   warm start) and `sigma` (a positive scalar scale estimate).
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(50 * 4), 50, 4)
#' y <- as.numeric(X %*% c(1, 0, -1, 0)) + rnorm(50)
#' robust_init(X, y)
#' @export
robust_init <- function(X, y, ridge_alpha = 1.0) {
  n <- nrow(X)
  p <- ncol(X)

  ridge_fit <- function() {
    as.numeric(solve(crossprod(X) + ridge_alpha * diag(p), crossprod(X, y)))
  }

  if (p < n) {
    beta0 <- tryCatch({
      fit <- MASS::rlm(X, y, method = "M", maxit = 500)
      as.numeric(stats::coef(fit))
    }, error = function(e) ridge_fit(), warning = function(w) ridge_fit())
  } else {
    beta0 <- ridge_fit()
  }

  r <- y - as.numeric(X %*% beta0)
  sigma <- MAD_CONST * stats::median(abs(r - stats::median(r)))
  sigma <- max(sigma, 1e-3) # guard against a degenerate (near-zero) scale
  list(beta0 = beta0, sigma = sigma)
}

#' Adaptive elastic-net weights
#'
#' \eqn{\hat w_j = 1 / (|\tilde\beta_j| + \epsilon)^\gamma}, the adaptive
#' (Zou, 2006) weights that place a heavier lasso penalty on small (likely
#' noise) coefficients from a preliminary fit `beta0`, promoting sparser
#' solutions without over-shrinking large (true signal) coefficients.
#'
#' @param beta0 Numeric vector, a preliminary coefficient estimate (e.g. from
#'   [robust_init()]).
#' @param gamma Positive exponent controlling how aggressively small
#'   coefficients are up-weighted.
#' @param eps Small positive constant preventing division by zero.
#' @return Numeric vector, the same length as `beta0`.
#' @examples
#' adaptive_weights(c(2, 0.01, -1.5, 0))
#' @export
adaptive_weights <- function(beta0, gamma = 1.0, eps = 1e-4) {
  1 / (abs(beta0) + eps)^gamma
}

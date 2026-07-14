#' Welsch loss
#'
#' Welsch loss \eqn{W_c(u) = c^2 (1 - \exp(-u^2 / (2 c^2)))}, elementwise.
#'
#' The loss uses prefactor \eqn{c^2} (rather than \eqn{c^2/2}) so that it is
#' exactly the antiderivative of the gradient [welsch_psi()] used by the
#' proximal Adam iterations in [welsch_adenet()].
#'
#' @param u Numeric vector of (standardized) residuals.
#' @param c Positive numeric scalar, the Welsch tuning constant.
#' @return Numeric vector, the same length as `u`.
#' @examples
#' welsch_loss(c(-1, 0, 1, 5), c = 2.11)
#' @export
welsch_loss <- function(u, c) {
  c * c * (1 - exp(-(u * u) / (2 * c * c)))
}

#' Welsch score (influence) function
#'
#' \eqn{W_c'(u) = u \exp(-u^2 / (2 c^2))}, the derivative of [welsch_loss()].
#' Its redescending shape -- rising, then decaying back to zero for large
#' `u` -- gives the Welsch adaptive elastic-net its robustness to outliers.
#'
#' @inheritParams welsch_loss
#' @return Numeric vector, the same length as `u`.
#' @examples
#' welsch_psi(c(-5, -1, 0, 1, 5), c = 2.11)
#' @export
welsch_psi <- function(u, c) {
  u * exp(-(u * u) / (2 * c * c))
}

#' Coordinatewise proximal map of the adaptive elastic-net penalty
#'
#' Closed-form soft-threshold (lasso) followed by ridge shrinkage, used
#' internally by [welsch_adenet()] and [fit_competitor()].
#'
#' @param v Numeric vector, the gradient-descent point.
#' @param step Numeric vector (or scalar), the per-coordinate step size.
#' @param l1,l2 Non-negative scalars, the lasso and ridge penalties.
#' @param w Numeric vector, the adaptive per-coordinate lasso weights.
#' @return Numeric vector, the same length as `v`.
#' @keywords internal
#' @noRd
soft_ridge_prox <- function(v, step, l1, l2, w) {
  thresh <- step * l1 * w
  shrunk <- sign(v) * pmax(abs(v) - thresh, 0)
  shrunk / (1 + step * l2)
}

#' Fit the Welsch adaptive elastic-net at one (lambda1, lambda2) pair
#'
#' Minimizes the (non-convex) Welsch data-fit term plus a convex adaptive
#' elastic-net penalty by a proximal Adam scheme: an Adam gradient step on
#' the smooth Welsch loss, followed by the closed-form soft-threshold /
#' ridge-shrink proximal map of the penalty. This is the core solver
#' (Algorithm 5.1 in the accompanying manuscript); see [fit_rbic()] to also
#' select `(l1, l2)` automatically.
#'
#' @param X Numeric design matrix, `n` by `p`.
#' @param y Numeric response vector, length `n`.
#' @param l1 Non-negative scalar, the adaptive lasso penalty \eqn{\lambda_1}.
#' @param l2 Non-negative scalar, the ridge penalty \eqn{\lambda_2}.
#' @param weights Numeric vector of length `p`, the adaptive weights
#'   \eqn{\hat w_j} (see [adaptive_weights()]).
#' @param sigma Positive scalar, the robust scale estimate \eqn{\hat\sigma}
#'   (see [robust_init()]).
#' @param c Positive scalar, the Welsch tuning constant (default `2.11`,
#'   approximately 95\% Gaussian efficiency).
#' @param eta Numeric scalar, the Adam base step size.
#' @param omega1,omega2 Numeric scalars in `[0, 1)`, the Adam first- and
#'   second-moment decay rates.
#' @param eps Small positive constant preventing division by zero in the
#'   Adam step.
#' @param beta0 Optional numeric warm start of length `p` (defaults to a
#'   zero vector).
#' @param max_iter Maximum number of proximal Adam iterations.
#' @param tol Relative loss-change convergence tolerance.
#' @return Numeric vector of length `p`, the rescaled coefficient estimate
#'   \eqn{(1 + \lambda_2/n) \beta^\star} that undoes the elastic-net's
#'   double shrinkage.
#' @seealso [fit_rbic()] for automatic penalty selection by robust BIC,
#'   [robust_init()] and [adaptive_weights()] for the required `sigma` and
#'   `weights` inputs.
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(100 * 5), 100, 5)
#' beta_star <- c(2, -1.5, 0, 0, 1)
#' y <- as.numeric(X %*% beta_star) + rnorm(100)
#'
#' init <- robust_init(X, y)
#' w <- adaptive_weights(init$beta0)
#' beta_hat <- welsch_adenet(X, y, l1 = 0.05, l2 = 0.01, weights = w,
#'                            sigma = init$sigma)
#' round(beta_hat, 2)
#' @export
welsch_adenet <- function(X, y, l1, l2, weights, sigma, c = 2.11,
                           eta = 1e-2, omega1 = 0.9, omega2 = 0.999, eps = 1e-8,
                           beta0 = NULL, max_iter = 2000L, tol = 1e-6) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- if (is.null(beta0)) numeric(p) else as.numeric(beta0)
  m <- numeric(p)
  v <- numeric(p)

  r <- y - as.numeric(X %*% beta)
  prev_loss <- welsch_loss(r / sigma, c)

  for (t in seq_len(max_iter)) {
    # gradient of the smooth Welsch term -- eq (5.3)
    g <- -as.numeric(crossprod(X, welsch_psi(r / sigma, c))) / sigma

    # Adam moments + bias correction
    m <- omega1 * m + (1 - omega1) * g
    v <- omega2 * v + (1 - omega2) * (g * g)
    m_hat <- m / (1 - omega1^t)
    v_hat <- v / (1 - omega2^t)

    # per-coordinate step, descent point, prox
    step <- eta / (sqrt(v_hat) + eps)
    vtent <- beta - step * m_hat
    beta <- soft_ridge_prox(vtent, step, l1, l2, weights)

    # stop on the (Euclidean) norm of the average change in the Welsch loss
    r <- y - as.numeric(X %*% beta)
    loss <- welsch_loss(r / sigma, c)
    if (sqrt(sum((loss - prev_loss)^2)) / n < tol) break
    prev_loss <- loss
  }

  (1 + l2 / n) * beta
}

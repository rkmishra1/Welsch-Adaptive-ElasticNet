#' Robust BIC (RBIC) score
#'
#' Robust BIC objective: the summed Welsch deviance of the standardized
#' residuals plus a BIC-style penalty on the number of active (non-zero)
#' coefficients. Lower is better; see [fit_rbic()], which minimizes this
#' score over a `(lambda1, lambda2)` grid.
#'
#' @param X Numeric design matrix, `n` by `p`.
#' @param y Numeric response vector, length `n`.
#' @param beta Numeric coefficient vector, length `p`.
#' @param sigma Positive scalar, the robust scale estimate.
#' @param c Positive scalar, the Welsch tuning constant.
#' @return A single numeric RBIC score.
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 3), 60, 3)
#' y <- as.numeric(X %*% c(1, 0, -1)) + rnorm(60)
#' rbic_score(X, y, beta = c(1, 0, -1), sigma = 1, c = 2.11)
#' @export
rbic_score <- function(X, y, beta, sigma, c) {
  n <- nrow(X)
  r <- y - as.numeric(X %*% beta)
  deviance <- sum(welsch_loss(r / sigma, c)) # T_d: robust fit term
  active <- sum(abs(beta) > 1e-8)
  deviance + log(n) * active
}

#' Default (lambda1, lambda2) penalty grid
#'
#' A data-driven, log-spaced \eqn{\lambda_1} grid -- from the smallest
#' penalty that zeroes every coefficient at `beta = 0` (a KKT-style
#' \eqn{\lambda_{\max}}) down to 0.1\% of that value -- crossed with a small
#' \eqn{\lambda_2} grid.
#'
#' @param X Numeric design matrix, `n` by `p`.
#' @param y Numeric response vector, length `n`.
#' @param sigma Positive scalar, the robust scale estimate.
#' @param n_l1,n_l2 Sizes of the \eqn{\lambda_1} and \eqn{\lambda_2} grids.
#' @return A list with numeric elements `l1_grid` and `l2_grid`.
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(60 * 3), 60, 3)
#' y <- as.numeric(X %*% c(1, 0, -1)) + rnorm(60)
#' default_grid(X, y, sigma = 1, n_l1 = 5, n_l2 = 3)
#' @export
default_grid <- function(X, y, sigma, n_l1 = 20L, n_l2 = 5L) {
  n <- nrow(X)
  l1_max <- max(abs(crossprod(X, y / sigma))) / n
  l1_grid <- 10^seq(log10(l1_max), log10(l1_max * 1e-3), length.out = n_l1)
  l2_grid <- c(0, 10^seq(-3, 0, length.out = n_l2 - 1))
  list(l1_grid = l1_grid, l2_grid = l2_grid)
}

#' Fit W-AdEnet, selecting penalties by robust BIC
#'
#' Runs [robust_init()] and [adaptive_weights()] to obtain a warm start,
#' scale estimate, and adaptive weights, then performs a warm-started 2-D
#' grid search over `(lambda1, lambda2)`, refitting [welsch_adenet()] at
#' each grid point and keeping the fit with the lowest [rbic_score()]. Per
#' the manuscript design, RBIC tunes the Welsch adaptive elastic-net only;
#' see [tune_competitor()] for the analogous procedure applied to other
#' robust losses.
#'
#' @param X Numeric design matrix, `n` by `p`.
#' @param y Numeric response vector, length `n`.
#' @param c Positive scalar, the Welsch tuning constant (default `2.11`).
#' @param gamma Adaptive-weight exponent, passed to [adaptive_weights()].
#' @param l1_grid,l2_grid Optional numeric penalty grids. If either is
#'   `NULL`, both default to [default_grid()] with its default sizes (20 and
#'   5 respectively).
#' @param ... Additional arguments passed to [welsch_adenet()] (e.g. `eta`,
#'   `max_iter`, `tol`).
#' @return A list with elements `beta` (the selected coefficient vector),
#'   `lambda1`, `lambda2`, `sigma`, `weights`, `rbic` (the best score), and
#'   `path` (a matrix with one row per grid point visited, columns `l1`,
#'   `l2`, `rbic`).
#' @examples
#' \donttest{
#' set.seed(1)
#' n <- 150
#' p <- 20
#' beta_star <- c(2, -1.5, 1, rep(0, p - 3))
#' X <- matrix(rnorm(n * p), n, p)
#' y <- as.numeric(X %*% beta_star) + rnorm(n)
#'
#' res <- fit_rbic(X, y, l1_grid = c(0.2, 0.05, 0.01), l2_grid = c(0, 0.1))
#' round(res$beta, 2)
#' res$lambda1
#' res$lambda2
#' }
#' @export
fit_rbic <- function(X, y, c = 2.11, gamma = 1.0, l1_grid = NULL, l2_grid = NULL, ...) {
  init <- robust_init(X, y)
  beta0 <- init$beta0
  sigma <- init$sigma
  w <- adaptive_weights(beta0, gamma = gamma)

  if (is.null(l1_grid) || is.null(l2_grid)) {
    grid <- default_grid(X, y, sigma)
    l1_grid <- grid$l1_grid
    l2_grid <- grid$l2_grid
  }

  best <- list(rbic = Inf)
  path <- matrix(numeric(0), ncol = 3, dimnames = list(NULL, c("l1", "l2", "rbic")))
  for (l2 in l2_grid) {
    warm <- beta0 # warm-start each l1 sweep from the robust init
    for (l1 in l1_grid) {
      beta <- welsch_adenet(X, y, l1, l2, w, sigma, c = c, beta0 = warm, ...)
      warm <- beta
      score <- rbic_score(X, y, beta, sigma, c)
      path <- rbind(path, c(l1 = l1, l2 = l2, rbic = score))
      if (score < best$rbic) {
        best <- list(beta = beta, lambda1 = l1, lambda2 = l2,
                      sigma = sigma, weights = w, rbic = score)
      }
    }
  }

  best$path <- path
  best
}

cov_matrix <- function(p, rho, kind) {
  if (kind == "cs") {
    S <- matrix(rho, p, p)
    diag(S) <- 1
    return(S)
  }
  if (kind == "ar1") {
    idx <- seq_len(p) - 1
    return(rho^abs(outer(idx, idx, "-")))
  }
  stop("unknown covariance kind: ", kind)
}

true_beta <- function(p) {
  beta <- numeric(p)
  k <- p %/% 3
  active <- sample.int(p, k)
  mags <- stats::runif(k, 0.5, 2.5)
  signs <- sample(c(-1, 1), k, replace = TRUE)
  beta[active] <- mags * signs
  list(beta = beta, active = sort(active))
}

#' Simulate one W-AdEnet benchmark replication
#'
#' Generates a sparse-signal train/test regression dataset following the
#' manuscript's data-generating process (Section 6, eq. 6.1): predictors are
#' multivariate normal with a compound-symmetry or AR(1) covariance, a
#' random third of coefficients are active with heterogeneous, mixed-sign
#' magnitudes, and the rest are exactly zero. The training set (only) may be
#' contaminated with vertical outliers or high-leverage points, so that test
#' MSPE is always measured on uncorrupted data.
#'
#' @param n Training sample size.
#' @param p Number of predictors.
#' @param rho Correlation parameter of the predictor covariance.
#' @param cov Covariance structure, `"ar1"` or `"cs"` (compound symmetry).
#' @param error Contamination regime: `"clean"`, `"vertical"` (response
#'   outliers), or `"leverage"` (response and design outliers).
#' @param delta Fraction of training rows contaminated (ignored when
#'   `error = "clean"`).
#' @param sigma Error standard deviation.
#' @param outlier_mag Magnitude of the outlier contamination.
#' @param n_test Test-set sample size (always generated clean).
#' @return A list with elements `X`, `y` (training design and response),
#'   `X_test`, `y_test` (held-out design and response), `beta_star` (the
#'   true coefficient vector), and `active` (the sorted 1-based indices of
#'   its non-zero entries).
#' @examples
#' set.seed(1)
#' d <- make_dataset(n = 100, p = 12, rho = 0.5, error = "vertical")
#' dim(d$X)
#' d$active
#' @export
make_dataset <- function(n, p, rho, cov = "ar1", error = "clean", delta = 0.10,
                          sigma = 1.0, outlier_mag = 20.0, n_test = 1000L) {
  tb <- true_beta(p)
  beta <- tb$beta
  active <- tb$active
  R <- chol(cov_matrix(p, rho, cov) + 1e-10 * diag(p))

  draw <- function(m) {
    Z <- matrix(stats::rnorm(m * p), m, p)
    X <- Z %*% R
    y <- as.numeric(X %*% beta) + sigma * stats::rnorm(m)
    list(X = X, y = y)
  }

  tr <- draw(n)
  te <- draw(n_test)
  X <- tr$X
  y <- tr$y

  if (error != "clean") {
    m <- floor(delta * n)
    bad <- sample.int(n, m)
    # vertical outliers: shift the responses by a common large magnitude
    y[bad] <- y[bad] + outlier_mag * sigma * sample(c(-1, 1), m, replace = TRUE)
    if (error == "leverage") {
      # also corrupt the design rows -> high-leverage bad points
      X[bad, ] <- outlier_mag + matrix(stats::rnorm(m * p), m, p)
    }
  }

  list(X = X, y = y, X_test = te$X, y_test = te$y, beta_star = beta, active = active)
}

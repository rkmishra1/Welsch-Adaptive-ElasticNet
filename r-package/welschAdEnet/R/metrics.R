#' Variable-selection metrics
#'
#' Counts true zeros correctly excluded (TZ) and true zeros incorrectly kept
#' (FZ, false inclusions), comparing an estimate against the true
#' coefficient vector (Section 6).
#'
#' @param beta_hat Numeric coefficient estimate.
#' @param beta_star Numeric true coefficient vector, same length as
#'   `beta_hat`.
#' @param tol Threshold below which a coefficient is treated as zero.
#' @return A list with integer elements `tz` (true zeros, higher is better)
#'   and `fz` (false inclusions, lower is better).
#' @examples
#' selection_metrics(c(0, 1.2, 0, 0.01), c(0, 1, 0, 0))
#' @export
selection_metrics <- function(beta_hat, beta_star, tol = 1e-8) {
  true_zero <- abs(beta_star) <= tol
  est_zero <- abs(beta_hat) <= tol
  list(tz = sum(true_zero & est_zero), fz = sum(true_zero & !est_zero))
}

#' Mean squared prediction error
#'
#' @param beta_hat Numeric coefficient vector.
#' @param X_test Numeric held-out design matrix.
#' @param y_test Numeric held-out response vector.
#' @return A single numeric value, the mean squared residual on the test
#'   set.
#' @examples
#' mspe(c(1, 0), matrix(c(1, 2, 3, 4), 2, 2), c(1, 2))
#' @export
mspe <- function(beta_hat, X_test, y_test) {
  resid <- y_test - as.numeric(X_test %*% beta_hat)
  mean(resid^2)
}

#' Run one design-grid cell of the simulation study
#'
#' Repeatedly draws a dataset via [make_dataset()], fits [fit_rbic()], and
#' aggregates selection and prediction metrics -- the inner loop of the
#' manuscript's simulation study (Section 6). Competing estimators are out
#' of scope here (RBIC tunes W-AdEnet only); benchmark another `fit(X, y)`
#' callable by looping it alongside this function.
#'
#' @param n Training sample size.
#' @param p Number of predictors.
#' @param rho Correlation parameter of the predictor covariance.
#' @param cov Covariance structure, `"ar1"` or `"cs"`, passed to
#'   [make_dataset()].
#' @param error Contamination regime, passed to [make_dataset()].
#' @param reps Number of independent replications.
#' @param ... Additional arguments passed to [fit_rbic()] (e.g. `l1_grid`,
#'   `l2_grid`, `max_iter`).
#' @return A one-row data frame with columns `TZ`, `FZ` (means over `reps`
#'   replications), `median_MSPE`, and `MSPE_SE` (standard error of the
#'   median, \eqn{1.2533 \times s / \sqrt{reps}}).
#' @examples
#' \donttest{
#' set.seed(1)
#' run_simulation_cell(n = 60, p = 8, rho = 0.5, error = "clean", reps = 2,
#'                      l1_grid = c(0.1, 0.01), l2_grid = c(0, 0.1),
#'                      max_iter = 200)
#' }
#' @export
run_simulation_cell <- function(n, p, rho, cov = "ar1", error = "clean", reps = 10L, ...) {
  tz <- integer(reps)
  fz <- integer(reps)
  mspes <- numeric(reps)
  for (i in seq_len(reps)) {
    d <- make_dataset(n, p, rho, cov = cov, error = error)
    res <- fit_rbic(d$X, d$y, ...)
    sel <- selection_metrics(res$beta, d$beta_star)
    tz[i] <- sel$tz
    fz[i] <- sel$fz
    mspes[i] <- mspe(res$beta, d$X_test, d$y_test)
  }
  # standard error of the median ~ 1.2533 * sd / sqrt(reps)
  se <- if (reps > 1) 1.2533 * stats::sd(mspes) / sqrt(reps) else 0
  data.frame(TZ = mean(tz), FZ = mean(fz), median_MSPE = stats::median(mspes), MSPE_SE = se)
}

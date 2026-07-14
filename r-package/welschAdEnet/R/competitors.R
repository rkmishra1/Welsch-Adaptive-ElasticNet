loss_squared <- function(u) 0.5 * u * u
psi_squared <- function(u) u

loss_absolute <- function(u) abs(u)
psi_absolute <- function(u) sign(u)

loss_huber <- function(u, k = 1.345) {
  abs_u <- abs(u)
  ifelse(abs_u <= k, 0.5 * u * u, k * abs_u - 0.5 * k * k)
}
psi_huber <- function(u, k = 1.345) {
  ifelse(abs(u) <= k, u, k * sign(u))
}

loss_tukey <- function(u, d = 4.685) {
  abs_u <- abs(u)
  inside <- (d^2 / 6) * (1 - (1 - (u / d)^2)^3)
  outside <- d^2 / 6
  ifelse(abs_u <= d, inside, outside)
}
psi_tukey <- function(u, d = 4.685) {
  ifelse(abs(u) <= d, u * (1 - (u / d)^2)^2, 0)
}

loss_welsch_ <- function(u, c = 2.11) welsch_loss(u, c)
psi_welsch_ <- function(u, c = 2.11) welsch_psi(u, c)

.loss_registry <- list(
  squared  = list(loss = loss_squared,  psi = psi_squared),
  absolute = list(loss = loss_absolute, psi = psi_absolute),
  huber    = list(loss = loss_huber,    psi = psi_huber),
  tukey    = list(loss = loss_tukey,    psi = psi_tukey),
  welsch   = list(loss = loss_welsch_,  psi = psi_welsch_)
)

loss_param_args <- function(loss_type, loss_param) {
  if (is.null(loss_param)) return(list())
  switch(loss_type,
    huber  = list(k = loss_param),
    tukey  = list(d = loss_param),
    welsch = list(c = loss_param),
    list()
  )
}

#' Fit a penalized robust-regression competitor
#'
#' A single proximal Adam solve generalizing [welsch_adenet()] to a
#' selectable robust loss, so that alternative estimators can be benchmarked
#' against the Welsch adaptive elastic-net with the same solver and adaptive
#' elastic-net penalty.
#'
#' @param X Numeric design matrix, `n` by `p`.
#' @param y Numeric response vector, length `n`.
#' @param l1 Non-negative scalar, the adaptive lasso penalty \eqn{\lambda_1}.
#' @param l2 Non-negative scalar, the ridge penalty \eqn{\lambda_2}.
#' @param weights Numeric vector of length `p`, the adaptive lasso weights.
#' @param sigma Positive scalar, the robust scale estimate.
#' @param loss_type One of `"squared"`, `"absolute"`, `"huber"`, `"tukey"`,
#'   or `"welsch"`.
#' @param eta Numeric scalar, the Adam base step size.
#' @param omega1,omega2 Numeric scalars in `[0, 1)`, the Adam first- and
#'   second-moment decay rates.
#' @param eps Small positive constant preventing division by zero in the
#'   Adam step.
#' @param beta0 Optional numeric warm start of length `p` (defaults to a
#'   zero vector).
#' @param max_iter Maximum number of proximal Adam iterations.
#' @param tol Relative loss-change convergence tolerance.
#' @param loss_param Optional tuning constant for the chosen loss (`k` for
#'   Huber, `d` for Tukey, `c` for Welsch); `NULL` uses each loss's usual
#'   default constant.
#' @return Numeric vector of length `p`, the rescaled coefficient estimate.
#' @seealso [tune_competitor()] to also select `(l1, l2)` by robust BIC,
#'   [welsch_adenet()] for the Welsch-specific solver this generalizes.
#' @examples
#' set.seed(1)
#' X <- matrix(rnorm(80 * 4), 80, 4)
#' y <- as.numeric(X %*% c(1, -1, 0, 0)) + rnorm(80)
#' init <- robust_init(X, y)
#' w <- adaptive_weights(init$beta0)
#' beta_hat <- fit_competitor(X, y, l1 = 0.05, l2 = 0.01, weights = w,
#'                             sigma = init$sigma, loss_type = "huber")
#' round(beta_hat, 2)
#' @export
fit_competitor <- function(X, y, l1, l2, weights, sigma, loss_type = "squared",
                            eta = 1e-2, omega1 = 0.9, omega2 = 0.999, eps = 1e-8,
                            beta0 = NULL, max_iter = 2000L, tol = 1e-6,
                            loss_param = NULL) {
  n <- nrow(X)
  p <- ncol(X)
  beta <- if (is.null(beta0)) numeric(p) else as.numeric(beta0)
  m <- numeric(p)
  v <- numeric(p)

  reg <- .loss_registry[[loss_type]]
  if (is.null(reg)) stop("unknown loss_type: ", loss_type)
  loss_args <- loss_param_args(loss_type, loss_param)

  r <- y - as.numeric(X %*% beta)
  prev_loss <- do.call(reg$loss, c(list(r / sigma), loss_args))

  for (t in seq_len(max_iter)) {
    # gradient of the smooth loss term (subgradient for the absolute loss)
    psi_val <- do.call(reg$psi, c(list(r / sigma), loss_args))
    g <- -as.numeric(crossprod(X, psi_val)) / sigma

    # Adam moments + bias correction
    m <- omega1 * m + (1 - omega1) * g
    v <- omega2 * v + (1 - omega2) * (g * g)
    m_hat <- m / (1 - omega1^t)
    v_hat <- v / (1 - omega2^t)

    # per-coordinate step, descent point, prox
    step <- eta / (sqrt(v_hat) + eps)
    vtent <- beta - step * m_hat
    beta <- soft_ridge_prox(vtent, step, l1, l2, weights)

    # stop on average change in the loss
    r <- y - as.numeric(X %*% beta)
    loss <- do.call(reg$loss, c(list(r / sigma), loss_args))
    if (sqrt(sum((loss - prev_loss)^2)) / n < tol) break
    prev_loss <- loss
  }

  (1 + l2 / n) * beta
}

competitor_rbic_score <- function(X, y, beta, sigma, loss_type, loss_param = NULL) {
  n <- nrow(X)
  r <- y - as.numeric(X %*% beta)
  reg <- .loss_registry[[loss_type]]
  loss_args <- loss_param_args(loss_type, loss_param)
  deviance <- sum(do.call(reg$loss, c(list(r / sigma), loss_args)))
  active <- sum(abs(beta) > 1e-8)
  deviance + log(n) * active
}

#' Tune a competitor loss by grid search over robust BIC
#'
#' Grid search over `(lambda1, lambda2)` for a given `loss_type`, minimizing
#' a robust-BIC-style score (the summed loss over standardized residuals
#' plus a BIC complexity penalty on the active set), analogous to
#' [fit_rbic()] but for the competing losses in [fit_competitor()].
#'
#' @param X Numeric design matrix, `n` by `p`.
#' @param y Numeric response vector, length `n`.
#' @param loss_type One of `"squared"`, `"absolute"`, `"huber"`, `"tukey"`,
#'   or `"welsch"`.
#' @param loss_param Optional tuning constant for the chosen loss; see
#'   [fit_competitor()].
#' @param gamma Adaptive-weight exponent. Ignored for
#'   `loss_type = "absolute"`, which uses uniform weights (matching the
#'   usual non-adaptive LAD-lasso).
#' @param n_l1,n_l2 Sizes of the default `(lambda1, lambda2)` grids, used
#'   only when `l1_grid`/`l2_grid` are `NULL`.
#' @param l1_grid,l2_grid Optional numeric penalty grids.
#' @param ... Additional arguments passed to [fit_competitor()].
#' @return A list with elements `beta`, `lambda1`, `lambda2`, `sigma`,
#'   `weights`, and `rbic` (the best score found).
#' @examples
#' \donttest{
#' set.seed(1)
#' X <- matrix(rnorm(80 * 4), 80, 4)
#' y <- as.numeric(X %*% c(1, -1, 0, 0)) + rnorm(80)
#' res <- tune_competitor(X, y, loss_type = "huber",
#'                         l1_grid = c(0.1, 0.02), l2_grid = c(0, 0.1))
#' round(res$beta, 2)
#' }
#' @export
tune_competitor <- function(X, y, loss_type = "squared", loss_param = NULL,
                             gamma = 1.0, n_l1 = 20L, n_l2 = 5L,
                             l1_grid = NULL, l2_grid = NULL, ...) {
  n <- nrow(X)
  p <- ncol(X)
  init <- robust_init(X, y)
  beta0 <- init$beta0
  sigma <- init$sigma

  # LAD-lasso is usually non-adaptive (uniform weights); other losses use
  # the adaptive weights.
  w <- if (identical(loss_type, "absolute")) rep(1, p) else adaptive_weights(beta0, gamma = gamma)

  if (is.null(l1_grid) || is.null(l2_grid)) {
    l1_max <- max(abs(crossprod(X, y / sigma))) / n
    l1_grid <- 10^seq(log10(l1_max), log10(l1_max * 1e-3), length.out = n_l1)
    l2_grid <- c(0, 10^seq(-3, 0, length.out = n_l2 - 1))
  }

  best <- list(rbic = Inf)
  for (l2 in l2_grid) {
    warm <- beta0
    for (l1 in l1_grid) {
      beta <- fit_competitor(X, y, l1, l2, w, sigma, loss_type = loss_type,
                              loss_param = loss_param, beta0 = warm, ...)
      warm <- beta
      score <- competitor_rbic_score(X, y, beta, sigma, loss_type, loss_param)
      if (score < best$rbic) {
        best <- list(beta = beta, lambda1 = l1, lambda2 = l2,
                      sigma = sigma, weights = w, rbic = score)
      }
    }
  }
  best
}

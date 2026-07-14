test_that("welsch_psi matches a finite-difference derivative of welsch_loss", {
  c <- 1.7
  u <- c(0.3, 1.5, -4.0)
  h <- 1e-6
  num <- (welsch_loss(u + h, c) - welsch_loss(u - h, c)) / (2 * h)
  expect_equal(num, welsch_psi(u, c), tolerance = 1e-4)
})

test_that("fit_rbic produces exact zeros and recovers most of the true support", {
  set.seed(1)
  d <- make_dataset(n = 300, p = 40, rho = 0.5, error = "clean")
  res <- fit_rbic(d$X, d$y)
  beta <- res$beta

  expect_gt(sum(abs(beta) <= 1e-8), 0)

  true_active <- d$active
  est_active <- which(abs(beta) > 1e-8)
  recovered <- length(intersect(true_active, est_active)) / length(true_active)
  expect_gte(recovered, 0.6)
})

test_that("fit_rbic keeps MSPE bounded under 10% vertical outliers", {
  set.seed(2)
  d <- make_dataset(n = 300, p = 40, rho = 0.5, error = "vertical")
  res <- fit_rbic(d$X, d$y)
  resid <- d$y_test - as.numeric(d$X_test %*% res$beta)
  expect_lt(stats::median(resid^2), 10.0)
})

test_that("tune_competitor fits every registered loss and returns full-length coefficients", {
  set.seed(42)
  d <- make_dataset(n = 100, p = 10, rho = 0.5, error = "clean")
  X <- d$X
  y <- d$y

  for (loss in c("squared", "absolute", "huber", "tukey", "welsch")) {
    res <- tune_competitor(X, y, loss_type = loss, n_l1 = 5, n_l2 = 2)
    expect_false(is.null(res$beta))
    expect_length(res$beta, 10)
  }
})

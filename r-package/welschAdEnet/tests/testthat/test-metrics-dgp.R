test_that("selection_metrics counts true zeros and false inclusions correctly", {
  # beta_star has true zeros at positions 1, 3, 4; beta_hat estimates zeros
  # at 1, 3 (correct) and a small non-zero at 4 (a false inclusion).
  sel <- selection_metrics(c(0, 1.2, 0, 0.01), c(0, 1, 0, 0))
  expect_equal(sel$tz, 2)
  expect_equal(sel$fz, 1)
})

test_that("mspe computes the mean squared residual", {
  val <- mspe(c(1, 0), matrix(c(1, 2, 3, 4), 2, 2), c(1, 2))
  X_test <- matrix(c(1, 2, 3, 4), 2, 2)
  expected <- mean((c(1, 2) - as.numeric(X_test %*% c(1, 0)))^2)
  expect_equal(val, expected)
})

test_that("make_dataset returns consistently shaped train/test data", {
  set.seed(1)
  d <- make_dataset(n = 50, p = 9, rho = 0.4, error = "leverage", n_test = 20)
  expect_equal(dim(d$X), c(50, 9))
  expect_equal(dim(d$X_test), c(20, 9))
  expect_length(d$y, 50)
  expect_length(d$y_test, 20)
  expect_length(d$beta_star, 9)
  expect_equal(sum(d$beta_star != 0), length(d$active))
})

test_that("make_dataset error='clean' leaves the training data uncontaminated relative to signal", {
  set.seed(1)
  d <- make_dataset(n = 200, p = 15, rho = 0.3, error = "clean")
  resid <- d$y - as.numeric(d$X %*% d$beta_star)
  expect_lt(max(abs(resid)), 6) # no injected outliers, just N(0, 1) noise
})

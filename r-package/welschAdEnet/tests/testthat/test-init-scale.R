test_that("robust_init returns a beta0/sigma pair of the right shape", {
  set.seed(1)
  X <- matrix(rnorm(50 * 4), 50, 4)
  y <- as.numeric(X %*% c(1, 0, -1, 0)) + rnorm(50)

  init <- robust_init(X, y)
  expect_length(init$beta0, 4)
  expect_gt(init$sigma, 0)
})

test_that("robust_init falls back to ridge in the wide (p >= n) regime", {
  set.seed(1)
  X <- matrix(rnorm(20 * 30), 20, 30)
  y <- as.numeric(X %*% c(rep(1, 5), rep(0, 25))) + rnorm(20)

  init <- robust_init(X, y)
  expect_length(init$beta0, 30)
  expect_gt(init$sigma, 0)
})

test_that("adaptive_weights up-weights small coefficients", {
  w <- adaptive_weights(c(2, 0.01, -1.5, 0))
  expect_gt(w[2], w[1])
  expect_gt(w[4], w[3])
})

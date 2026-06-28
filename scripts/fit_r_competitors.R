#!/usr/bin/env Rscript
# scripts/fit_r_competitors.R

args <- commandArgs(trailingOnly = TRUE)
train_x_path <- args[1]
train_y_path <- args[2]
test_x_path <- args[3]
output_path <- args[4]

suppressPackageStartupMessages({
  library(robustHD)
})

# Load data
X_train <- as.matrix(read.csv(train_x_path, header = FALSE))
y_train <- as.numeric(read.csv(train_y_path, header = FALSE)[, 1])
X_test <- as.matrix(read.csv(test_x_path, header = FALSE))

n <- nrow(X_train)
p <- ncol(X_train)

# Fit S-LTS
fit_s_lts <- tryCatch({
  lambda_grid <- seq(0.20, 0.05, by = -0.05)
  fit <- robustHD::sparseLTS(x = X_train, y = y_train, lambda = lambda_grid, mode = "fraction", intercept = FALSE, crit = "BIC")
  coefs <- as.numeric(stats::coef(fit, fit = "reweighted", zeros = TRUE))
  if (length(coefs) == p + 1) coefs <- coefs[-1]
  coefs
}, error = function(e) {
  rep(0, p)
})

# Fit R-LARS
fit_r_lars <- tryCatch({
  s_max <- min(10, p)
  fit <- robustHD::rlars(x = X_train, y = y_train, sMax = s_max, crit = "BIC")
  coefs <- as.numeric(stats::coef(fit, zeros = TRUE))
  if (length(coefs) == p + 1) coefs <- coefs[-1]
  coefs
}, error = function(e) {
  rep(0, p)
})

# Write outputs
write.csv(data.frame(S_LTS = fit_s_lts, RLARS = fit_r_lars), output_path, row.names = FALSE)

# welschAdEnet

**Robust sparse regression via proximal Adam, tuned by robust BIC.**

An R port of the reference Python implementation in this repository. Fits
the **Welsch Adaptive Elastic-Net (W-AdEnet)**, a high-breakdown estimator
for sparse linear regression that couples a redescending Welsch loss with
an adaptive elastic-net penalty, solved by a first-order proximal Adam
scheme and tuned by a robust Bayesian information criterion (RBIC).

## Installation

```r
# from a local clone of this repository
install.packages("r-package/welschAdEnet", repos = NULL, type = "source")
```

## Quickstart

```r
library(welschAdEnet)

# X : n x p design matrix, y : length-n response
res <- fit_rbic(X, y)          # robust init -> adaptive weights -> RBIC grid search

beta_hat <- res$beta           # rescaled (1 + lambda2/n) estimate
res$lambda1; res$lambda2; res$rbic
```

Single fit at fixed penalties (the core solver directly):

```r
init <- robust_init(X, y)               # MM-like warm start + MAD scale
w <- adaptive_weights(init$beta0)       # what_j = 1 / |beta0_j|
beta <- welsch_adenet(X, y, l1 = 0.1, l2 = 0.01, weights = w, sigma = init$sigma)
```

## Reproducing a simulation cell

```r
run_simulation_cell(n = 300, p = 40, rho = 0.5, error = "leverage", reps = 100)
```

## Package layout

```
R/
├── estimator.R       welsch_loss, welsch_psi, welsch_adenet (proximal Adam core)
├── init_scale.R      robust_init, adaptive_weights
├── tuning.R          rbic_score, default_grid, fit_rbic
├── competitors.R      fit_competitor, tune_competitor (squared/absolute/huber/tukey/welsch)
├── dgp.R             make_dataset: simulation data-generating process
└── metrics.R         selection_metrics, mspe, run_simulation_cell
tests/testthat/       testthat unit tests mirroring tests/test_estimator.py
```

## Testing

```r
devtools::test()
# or
R CMD check --as-cran .
```

## License

MIT -- see [LICENSE.md](LICENSE.md).

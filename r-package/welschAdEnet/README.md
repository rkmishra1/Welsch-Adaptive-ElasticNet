<div align="center">

# welschAdEnet

**Robust sparse regression via proximal Adam, tuned by robust BIC**

[![R](https://img.shields.io/badge/R-%3E%3D3.6-276DC3.svg?logo=r&logoColor=white)](https://www.r-project.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE.md)
[![Version](https://img.shields.io/badge/version-0.1.0-blue.svg)](NEWS.md)

*R port of the reference Python implementation in this repository.*

</div>

---

## Overview

**welschAdEnet** fits the **Welsch Adaptive Elastic-Net (W-AdEnet)**, a
high-breakdown estimator for sparse linear regression that couples a
redescending **Welsch loss** with an **adaptive elastic-net penalty**. The
estimator is solved by a first-order **proximal Adam** scheme and its
penalties are tuned by a **robust Bayesian information criterion (RBIC)**:

$$
\hat\beta = \arg\min_{\beta}\ \sum_{i=1}^{n}\mathcal{W}_c\!\left(\frac{y_i-\mathbf{x}_i^{\top}\beta}{\hat\sigma}\right)\ +\ \sum_{j=1}^{p}\left(\hat w_j\lambda_1|\beta_j|+\tfrac{\lambda_2}{2}\beta_j^{2}\right)
$$

The Welsch term downweights large residuals to zero, giving resistance to
both vertical outliers and high-leverage points, while the adaptive
elastic-net penalty yields exact variable selection and grouped shrinkage
of correlated predictors.

<p align="center">
  <img src="../../docs/figures/pipeline.png" width="780" alt="W-AdEnet fitting pipeline"/>
  <br><em>Four-stage pipeline: robust initialisation → adaptive weights → RBIC grid search → proximal Adam iterations.</em>
</p>

---

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

<details>
<summary><b>Single fit at fixed penalties (the core solver directly)</b></summary>

```r
init <- robust_init(X, y)               # MM-like warm start + MAD scale
w <- adaptive_weights(init$beta0)       # what_j = 1 / |beta0_j|
beta <- welsch_adenet(X, y, l1 = 0.1, l2 = 0.01, weights = w, sigma = init$sigma)
```
</details>

<details>
<summary><b>Reproducing a simulation cell</b></summary>

```r
run_simulation_cell(n = 300, p = 40, rho = 0.5, error = "leverage", reps = 100)
```
</details>

---

## Function Reference

| Function | Purpose |
|:---------|:--------|
| `welsch_adenet()` | Proximal Adam solver for the W-AdEnet objective at fixed `(l1, l2)` |
| `welsch_loss()`, `welsch_psi()` | Welsch loss and its derivative (influence function) |
| `robust_init()` | MM-like robust warm start + MAD scale estimate |
| `adaptive_weights()` | Adaptive elastic-net weights `what_j = 1 / |beta0_j|` |
| `fit_rbic()`, `rbic_score()`, `default_grid()` | RBIC-tuned fit over a `(lambda1, lambda2)` grid |
| `fit_competitor()`, `tune_competitor()` | Same solver generalized to squared / absolute / Huber / Tukey losses |
| `make_dataset()` | Simulation data-generating process (clean / vertical / leverage regimes) |
| `selection_metrics()`, `mspe()`, `run_simulation_cell()` | Variable-selection and prediction-error evaluation |

---

## Package Layout

```
R/
├── estimator.R       welsch_loss, welsch_psi, welsch_adenet (proximal Adam core)
├── init_scale.R      robust_init, adaptive_weights
├── tuning.R          rbic_score, default_grid, fit_rbic
├── competitors.R     fit_competitor, tune_competitor (squared/absolute/huber/tukey/welsch)
├── dgp.R             make_dataset: simulation data-generating process
└── metrics.R         selection_metrics, mspe, run_simulation_cell
tests/testthat/       testthat unit tests mirroring tests/test_estimator.py
```

---

## Testing

```r
devtools::test()
# or
R CMD check --as-cran .
```

---

## Citation

```bibtex
@article{welsch_adenet,
  title  = {Welsch Adaptive Elastic-Net for Robust High-Dimensional Regression},
  author = {Mishra, R. K. and others},
  year   = {2026},
  note   = {Manuscript}
}
```

## License

Released under the [MIT License](LICENSE.md).

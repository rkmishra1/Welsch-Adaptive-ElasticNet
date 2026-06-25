<div align="center">

# Welsch Adaptive Elastic-Net

**Robust sparse regression via proximal Adam, tuned by robust BIC**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/numpy-≥1.23-013243.svg?logo=numpy)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-≥1.1-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

*Reference implementation and simulation study accompanying the manuscript.*

</div>

---

## Overview

The **Welsch Adaptive Elastic-Net (W-AdEnet)** is a high-breakdown estimator for
sparse linear regression that stays accurate when the data contain outliers,
heavy-tailed errors, or high-leverage points. It couples a **redescending Welsch
loss** with an **adaptive elastic-net penalty** and is fit by a first-order
**proximal Adam** scheme that scales linearly in the data, `O(np)` per iteration.

$$
\hat\beta = \arg\min_{\beta}\ \underbrace{\sum_{i=1}^{n}\mathcal{W}_c\left(\frac{y_i-\mathbf{x}_i^{\top}\beta}{\hat\sigma}\right)}_{\text{robust, non-convex data term}}\ +\ \underbrace{\sum_{j=1}^{p}\left(\hat w_j\,\lambda_1|\beta_j|+\tfrac{\lambda_2}{2}\beta_j^{2}\right)}_{\text{convex elastic-net penalty}}
$$

Each iteration takes an Adam gradient step on the smooth Welsch term and applies
the closed-form **soft-threshold → ridge-shrink** proximal map of the penalty,
yielding exact zeros (variable selection) and grouped shrinkage of correlated
predictors. Penalties $(\lambda_1,\lambda_2)$ are selected by a **robust BIC
(RBIC)** criterion that resists the prediction-error inflation that derails
ordinary cross-validation under contamination.

> [!NOTE]
> **Why Adam?** The redescending Welsch weight $e^{-r_i^2/(2c^2\hat\sigma^2)}$
> makes the loss curvature wildly uneven and non-stationary across coordinates.
> Adam's second-moment rescaling acts as a diagonal preconditioner, its momentum
> carries through the flat regions of the non-convex loss, and it never forms or
> inverts a Hessian.

## Features

- ⚡ **Linear-time fits** — `O(np)` per iteration, no `p × p` factorization.
- 🛡️ **High breakdown** — bounded, redescending influence; stable under vertical
  outliers *and* leverage points.
- 🎯 **Exact sparsity** — closed-form elastic-net prox produces true zeros.
- 🔧 **Outlier-resistant tuning** — RBIC over a 2-D `(λ₁, λ₂)` grid with warm starts.
- 🧪 **Full simulation study** — reproduces the manuscript design grid end-to-end.

## Installation

```bash
git clone https://github.com/rkmishra1/welsch-adenet-python.git
cd welsch-adenet-python
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Quickstart

```python
from welsch_adenet import fit_rbic

# X : (n, p) design,  y : (n,) response
res = fit_rbic(X, y)          # robust init → adaptive weights → RBIC grid search

beta_hat = res["beta"]                       # rescaled (1 + λ₂/n) estimate
print(res["lambda1"], res["lambda2"], res["rbic"])
```

<details>
<summary><b>Single fit at fixed penalties (Algorithm 5.1 directly)</b></summary>

```python
from welsch_adenet import welsch_adenet, robust_init, adaptive_weights

beta0, sigma = robust_init(X, y)             # MM-like warm start + MAD scale
w = adaptive_weights(beta0)                  # ŵ_j = 1 / |β̃_j|
beta = welsch_adenet(X, y, l1=0.1, l2=0.01, weights=w, sigma=sigma)
```
</details>

## Reproducing the simulation study

```bash
python -m simulation.run_simulation --smoke                  # fast sanity check
python -m simulation.run_simulation --regime low      --reps 300
python -m simulation.run_simulation --regime moderate --reps 300
python -m simulation.run_simulation --regime high     --reps 300 --cov ar1
```

Each run sweeps the design grid for one dimension regime and writes mean **TZ**,
mean **FZ**, and **median MSPE** (± standard error) to `results_<regime>.csv`.

| Regime    | Growth          | `p` at `n = 800, 1600, 2400` |
|:----------|:----------------|:-----------------------------|
| Low       | `p ∝ √n`        | 108, 155, 190                |
| Moderate  | `p ∝ n^(2/3)`   | 347, 555, 730                |
| High      | `p ∝ n`         | 960, 1896, 2760              |

Crossed with correlation `ρ ∈ {0.35, 0.65, 0.85}` and error regime
`∈ {clean, vertical, leverage}`, at contamination fraction `δ = 0.10`.

**Reported quantities**

| Metric | Meaning | Direction |
|:-------|:--------|:----------|
| **TZ** | True zeros correctly excluded | higher ↑ (ceiling `p − |A|`) |
| **FZ** | True zeros wrongly retained (false inclusions) | lower ↓ |
| **MSPE** | Median squared prediction error on a clean test set | lower ↓ |

## Project layout

```
welsch_adenet/
├── estimator.py      Algorithm 5.1 — proximal Adam, Welsch loss, elastic-net prox
├── init_scale.py     robust warm start, MAD scale, adaptive weights
└── tuning.py         RBIC criterion + 2-D grid search   (W-AdEnet only)
simulation/
├── dgp.py            data-generating process + 3 contamination regimes  (eq 6.1)
├── metrics.py        TZ, FZ, MSPE
└── run_simulation.py design-grid driver
tests/
└── test_estimator.py runnable self-checks
```

## Testing

```bash
python -m pytest tests/ -q          # or:  python tests/test_estimator.py
```

The suite checks the Welsch gradient against finite differences, exact-zero
support recovery on clean data, and bounded prediction error under 10% vertical
outliers.

## Notes & scope

- **RBIC tunes W-AdEnet only.** The competing estimators in the manuscript (AdL,
  AdEnet, H-AdL, T-AdL, S-LTS, R-LARS) follow their own tuning protocols and are
  not bundled here. Add one as a `fit(X, y) → beta_hat` callable and loop it
  alongside W-AdEnet in `run_simulation.py`.
- **Initialization.** A Huber M-fit (`p < n`) or ridge (`p ≥ n`) serves as a
  practical surrogate for the MM / MM-Ridge starts described in the manuscript.
- **Loss prefactor.** The Welsch loss uses prefactor `c²` (not the `c²/2` printed
  in eq. 5.2) so that it is the exact antiderivative of the gradient used in
  Algorithm 5.1 — see the note in [`estimator.py`](welsch_adenet/estimator.py).

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

Released under the [MIT License](LICENSE).

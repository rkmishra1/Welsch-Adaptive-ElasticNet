# Welsch Adaptive Elastic-Net (W-AdEnet)

Reference implementation and simulation study for the **Welsch Adaptive
Elastic-Net** estimator fit by **proximal Adam** (Algorithm 5.1) and tuned by the
**robust BIC (RBIC)** criterion (eq. 5.8). Code accompanying the manuscript.

The estimator solves

```
min_beta  sum_i W_c((y_i - x_i'beta)/sigma)  +  sum_j ( w_j*l1*|beta_j| + (l2/2)*beta_j^2 )
```

with the redescending Welsch loss `W_c` as the data term (non-convex, smooth) and
the adaptive elastic-net penalty (convex, non-smooth). Each iteration is an Adam
gradient step on the Welsch term followed by the closed-form soft-threshold /
ridge-shrink proximal map of the penalty.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```python
from welsch_adenet import fit_rbic

res = fit_rbic(X, y)          # robust init -> adaptive weights -> RBIC grid search
beta_hat = res["beta"]        # rescaled (1 + l2/n) estimate
res["lambda1"], res["lambda2"], res["rbic"]
```

A single `(lambda_1, lambda_2)` fit (Algorithm 5.1 directly):

```python
from welsch_adenet import welsch_adenet, robust_init, adaptive_weights
beta0, sigma = robust_init(X, y)
w = adaptive_weights(beta0)
beta = welsch_adenet(X, y, l1=0.1, l2=0.01, weights=w, sigma=sigma)
```

## Reproducing the simulation study

```bash
python -m simulation.run_simulation --smoke              # quick sanity check
python -m simulation.run_simulation --regime low  --reps 300
python -m simulation.run_simulation --regime moderate --reps 300
python -m simulation.run_simulation --regime high --reps 300 --cov ar1
```

Each run walks the design grid (sample size × correlation × error regime) for one
dimension regime and writes mean **TZ** (correct exclusions), mean **FZ** (false
inclusions), and **median MSPE** (with standard error) to `results_<regime>.csv`.

Design grid (Section 6):

| regime   | growth        | p at n=800,1600,2400 |
|----------|---------------|----------------------|
| low      | p ~ sqrt(n)   | 108, 155, 190        |
| moderate | p ~ n^(2/3)   | 347, 555, 730        |
| high     | p ~ n         | 960, 1896, 2760      |

crossed with rho ∈ {0.35, 0.65, 0.85} and error ∈ {clean, vertical, leverage},
contamination fraction δ = 0.10.

## Layout

```
welsch_adenet/
  estimator.py    Algorithm 5.1: proximal Adam, Welsch loss, elastic-net prox
  init_scale.py   robust warm start, MAD scale, adaptive weights
  tuning.py       RBIC criterion + 2-D grid search (W-AdEnet only)
simulation/
  dgp.py          data-generating process + 3 contamination regimes (eq 6.1)
  metrics.py      TZ, FZ, MSPE
  run_simulation.py   design-grid driver
tests/            runnable self-checks (pytest or `python tests/test_estimator.py`)
```

## Scope notes

- **RBIC tunes W-AdEnet only.** The six competing estimators in the manuscript
  (AdL, AdEnet, H-AdL, T-AdL, S-LTS, R-LARS) are tuned by their own protocols and
  are not bundled here; add one as a `fit(X, y) -> beta_hat` callable and loop it
  alongside W-AdEnet in `run_simulation.py`.
- Robust initialization uses a Huber M-fit (`p < n`) or ridge (`p >= n`) as a
  practical stand-in for the MM / MM-Ridge starts described in the manuscript.
- The Welsch loss prefactor is taken as `c^2` (not the `c^2/2` printed in eq. 5.2)
  so that the loss is the exact antiderivative of the gradient used in Algorithm
  5.1; see the note in `estimator.py`.

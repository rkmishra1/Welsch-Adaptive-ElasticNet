# welschAdEnet 0.1.0

* Initial release.
* `welsch_adenet()`, `robust_init()`, `adaptive_weights()`: the Welsch
  adaptive elastic-net proximal Adam solver and its robust initialization.
* `fit_rbic()`, `rbic_score()`, `default_grid()`: robust-BIC penalty
  selection over a `(lambda1, lambda2)` grid.
* `fit_competitor()`, `tune_competitor()`: the same solver generalized to
  squared, absolute, Huber, and Tukey losses for benchmarking.
* `make_dataset()`, `selection_metrics()`, `mspe()`, `run_simulation_cell()`:
  the simulation-study data-generating process and evaluation metrics.

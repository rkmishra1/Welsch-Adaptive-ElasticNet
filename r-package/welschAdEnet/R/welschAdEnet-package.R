#' welschAdEnet: Welsch Adaptive Elastic-Net for Robust Sparse Regression
#'
#' @description
#' Fits the Welsch Adaptive Elastic-Net (W-AdEnet), a high-breakdown
#' estimator for sparse linear regression that couples a redescending
#' Welsch loss with an adaptive elastic-net penalty, solved by a
#' first-order proximal Adam scheme and tuned by a robust Bayesian
#' information criterion (RBIC). Also provides several competing penalized
#' robust-regression losses (squared, absolute, Huber, Tukey) fit with the
#' same solver, and tools ([make_dataset()], [selection_metrics()],
#' [mspe()], [run_simulation_cell()]) for reproducing the accompanying
#' manuscript's simulation study under vertical-outlier and leverage
#' contamination.
#'
#' @section Main functions:
#' \itemize{
#'   \item [fit_rbic()] -- fit W-AdEnet with automatic RBIC-based penalty
#'     selection (the typical entry point).
#'   \item [welsch_adenet()] -- fit W-AdEnet at a single, fixed
#'     `(lambda1, lambda2)` pair.
#'   \item [robust_init()], [adaptive_weights()] -- the robust warm start,
#'     scale estimate, and adaptive weights required by the solver.
#'   \item [fit_competitor()], [tune_competitor()] -- the same solver
#'     generalized to other robust losses, for benchmarking.
#'   \item [make_dataset()], [selection_metrics()], [mspe()],
#'     [run_simulation_cell()] -- the simulation study data-generating
#'     process and metrics.
#' }
#'
#' @keywords internal
"_PACKAGE"

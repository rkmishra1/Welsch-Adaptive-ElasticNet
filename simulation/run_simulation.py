"""Run the W-AdEnet simulation study (Section 6).

Walks the design grid -- dimension regime x sample size x correlation x error
regime -- fits W-AdEnet with RBIC tuning over `reps` replications, and writes
mean TZ, mean FZ, and median MSPE (with its standard error) to CSV.

    python -m simulation.run_simulation --regime low --reps 300
    python -m simulation.run_simulation --smoke          # tiny, fast sanity run

Competing estimators are out of scope here (RBIC tunes W-AdEnet only); add them
by writing a fit(X, y) -> beta_hat callable and looping it alongside W-AdEnet.
"""

import argparse
import csv
import time

import numpy as np

from welsch_adenet import fit_rbic
from .dgp import make_dataset
from .metrics import selection_metrics, mspe

# dimension regimes: sample size -> p  (Section 6 growth regimes)
DIM_REGIMES = {
    "low":      {800: 108, 1600: 155, 2400: 190},    # p ~ sqrt(n)
    "moderate": {800: 347, 1600: 555, 2400: 730},    # p ~ n^(2/3)
    "high":     {800: 960, 1600: 1896, 2400: 2760},  # p ~ n
}
RHOS = [0.35, 0.65, 0.85]
ERRORS = ["clean", "vertical", "leverage"]


def run_cell(n, p, rho, error, reps, cov, rng):
    tz, fz, mspes = [], [], []
    for _ in range(reps):
        d = make_dataset(n, p, rho, cov=cov, error=error, rng=rng)
        res = fit_rbic(d["X"], d["y"])
        beta_hat = res["beta"]
        a, b = selection_metrics(beta_hat, d["beta_star"])
        tz.append(a)
        fz.append(b)
        mspes.append(mspe(beta_hat, d["X_test"], d["y_test"]))
    mspes = np.array(mspes)
    med = np.median(mspes)
    # standard error of the median ~ 1.253 * sd / sqrt(reps)
    se = 1.2533 * np.std(mspes, ddof=1) / np.sqrt(len(mspes)) if reps > 1 else 0.0
    return np.mean(tz), np.mean(fz), med, se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=list(DIM_REGIMES), default="low")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--cov", choices=["ar1", "cs"], default="ar1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny grid + few reps for a quick sanity check")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = args.out or f"results_{args.regime}.csv"

    if args.smoke:
        grid = {"regime": "smoke", "sizes": {200: 40}, "rhos": [0.65],
                "errors": ["clean", "vertical"], "reps": 3}
        sizes, rhos, errors, reps = grid["sizes"], grid["rhos"], grid["errors"], grid["reps"]
    else:
        sizes, rhos, errors, reps = DIM_REGIMES[args.regime], RHOS, ERRORS, args.reps

    rows = []
    for error in errors:
        for n, p in sizes.items():
            for rho in rhos:
                t0 = time.time()
                tz, fz, med, se = run_cell(n, p, rho, error, reps, args.cov, rng)
                rows.append({"error": error, "n": n, "p": p, "rho": rho,
                             "TZ": round(tz, 2), "FZ": round(fz, 2),
                             "median_MSPE": round(med, 4), "MSPE_SE": round(se, 4)})
                print(f"[{error:8s}] n={n} p={p} rho={rho}: "
                      f"TZ={tz:.1f} FZ={fz:.1f} MSPE={med:.3f}({se:.3f}) "
                      f"[{time.time()-t0:.1f}s]")

    with open(out, "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wtr.writeheader()
        wtr.writerows(rows)
    print(f"\nwrote {len(rows)} cells -> {out}")


if __name__ == "__main__":
    main()

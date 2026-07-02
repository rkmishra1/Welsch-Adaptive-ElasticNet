#!/usr/bin/env python
# scripts/sensitivity_analysis_combined.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from welsch_adenet.competitors import tune_competitor

# Set random seed for reproducibility
np.random.seed(42)

# Output directories
fig_dir = "/Users/ramakrushnamishra/welsch-adenet/docs/figures"
os.makedirs(fig_dir, exist_ok=True)

# Helper to split, scale, and center data (80/20)
def prepare_train_test(X, y, train_prop=0.80):
    n = X.shape[0]
    train_idx = np.random.choice(n, size=int(train_prop * n), replace=False)
    test_idx = np.delete(np.arange(n), train_idx)
    
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    
    mean_x = X_train.mean(axis=0)
    sd_x = X_train.std(axis=0)
    sd_x[sd_x < 1e-10] = 1.0
    
    X_train_scaled = (X_train - mean_x) / sd_x
    X_test_scaled = (X_test - mean_x) / sd_x
    
    mean_y = y_train.mean()
    y_train_centered = y_train - mean_y
    y_test_centered = y_test - mean_y
    
    return {
        "X_train": X_train_scaled,
        "y_train": y_train_centered,
        "X_test": X_test_scaled,
        "y_test": y_test_centered,
        "mean_y": mean_y
    }

# ============================================================
# 1. Load Datasets and Add Noise
# ============================================================

rho_noise = 0.8
corr_orig_noise = 0.7
c_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
B = 10  # Replications

# --- A. Boston Housing ---
boston_df = pd.read_csv("/Users/ramakrushnamishra/welsch-adenet/data/Boston.csv")
X_bos = boston_df.drop("medv", axis=1).values
y_bos = boston_df["medv"].values
p_orig_bos = X_bos.shape[1]
n_bos = X_bos.shape[0]
p_noise_bos = 30

Z_bos = np.random.randn(n_bos, p_noise_bos)
X_noise_bos = Z_bos.copy()
innov_scale = np.sqrt(1.0 - rho_noise**2)
for j in range(1, p_noise_bos):
    X_noise_bos[:, j] = rho_noise * X_noise_bos[:, j - 1] + innov_scale * Z_bos[:, j]
for j in range(min(5, p_orig_bos, p_noise_bos)):
    X_noise_bos[:, j] = corr_orig_noise * X_bos[:, j] + np.sqrt(1.0 - corr_orig_noise**2) * X_noise_bos[:, j]
X_full_bos = np.hstack([X_bos, X_noise_bos])


# --- B. HBK Data ---
hbk_df = pd.read_csv("/Users/ramakrushnamishra/welsch-adenet/data/hbk.csv")
X_hbk = hbk_df.drop("Y", axis=1).values
y_hbk = hbk_df["Y"].values
p_orig_hbk = X_hbk.shape[1]
n_hbk = X_hbk.shape[0]
p_noise_hbk = 15

Z_hbk = np.random.randn(n_hbk, p_noise_hbk)
X_noise_hbk = Z_hbk.copy()
for j in range(1, p_noise_hbk):
    X_noise_hbk[:, j] = rho_noise * X_noise_hbk[:, j - 1] + innov_scale * Z_hbk[:, j]
for j in range(min(p_orig_hbk, p_noise_hbk)):
    X_noise_hbk[:, j] = corr_orig_noise * X_hbk[:, j] + np.sqrt(1.0 - corr_orig_noise**2) * X_noise_hbk[:, j]
X_full_hbk = np.hstack([X_hbk, X_noise_hbk])


# --- C. NCI60 Data ---
nci60_df = pd.read_csv("/Users/ramakrushnamishra/welsch-adenet/data/nci60_protein.csv")
X_nci = nci60_df.drop("DoublingTime", axis=1).values
y_nci = nci60_df["DoublingTime"].values
p_orig_nci = X_nci.shape[1]
n_nci = X_nci.shape[0]
p_noise_nci = 30

Z_nci = np.random.randn(n_nci, p_noise_nci)
X_noise_nci = Z_nci.copy()
for j in range(1, p_noise_nci):
    X_noise_nci[:, j] = rho_noise * X_noise_nci[:, j - 1] + innov_scale * Z_nci[:, j]
for j in range(min(5, p_orig_nci, p_noise_nci)):
    X_noise_nci[:, j] = corr_orig_noise * X_nci[:, j] + np.sqrt(1.0 - corr_orig_noise**2) * X_noise_nci[:, j]
X_full_nci = np.hstack([X_nci, X_noise_nci])


# ============================================================
# 2. Run Sensitivity Analyses
# ============================================================

def run_analysis(X_full, y_target, p_orig, label):
    results = []
    print(f"Running Sensitivity Analysis for {label}...")
    for c in c_values:
        print(f"  {label}: Evaluating c = {c}...")
        for b in range(1, B + 1):
            split_data = prepare_train_test(X_full, y_target, train_prop=0.80)
            res = tune_competitor(
                split_data["X_train"], split_data["y_train"],
                loss_type="welsch", loss_param=c,
                n_l1=12, n_l2=5, max_iter=800
            )
            beta_hat = res["beta"]
            
            preds = split_data["X_test"] @ beta_hat
            residuals = split_data["y_test"] - preds
            med_spe = np.median(residuals**2)
            
            tp = np.sum(np.abs(beta_hat[:p_orig]) > 1e-8)
            fp = np.sum(np.abs(beta_hat[p_orig:]) > 1e-8)
            
            results.append({"c": c, "Rep": b, "MedSPE": med_spe, "TP": tp, "FP": fp})
            
    df = pd.DataFrame(results)
    summary = df.groupby("c").agg(
        MedSPE_median=("MedSPE", "median"),
        MedSPE_se=("MedSPE", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        TP_mean=("TP", "mean"),
        FP_mean=("FP", "mean")
    ).reset_index()
    return summary

summary_bos = run_analysis(X_full_bos, y_bos, p_orig_bos, "Boston Housing")
summary_hbk = run_analysis(X_full_hbk, y_hbk, p_orig_hbk, "HBK")
summary_nci = run_analysis(X_full_nci, y_nci, p_orig_nci, "NCI60")


# ============================================================
# 3. Plotting Combined Results (3 x 2 Grid) matching user style
# ============================================================
fig, axes = plt.subplots(3, 2, figsize=(12, 13.5))

def plot_dataset(row_idx, summary, label):
    # Left side (Prediction Error) is blue, Right side (Variable Selection) matches reference
    c_ours = "#0984E3"  # Blue for left side plot
    c_tp = "#2ecc71"    # Green/teal for True Positives
    c_fp = "#e74c3c"    # Orange/red for False Positives
    c_default = "red"   # Red for default parameter indicator

    # Left Column: MedSPE vs c
    axes[row_idx, 0].errorbar(
        summary["c"], summary["MedSPE_median"], yerr=summary["MedSPE_se"],
        fmt="-o", color=c_ours, linewidth=2, elinewidth=1.5, capsize=4, label="Median MedSPE (Ours)"
    )
    axes[row_idx, 0].axvline(2.11, color=c_default, linestyle="--", label="Default $c = 2.11$")
    axes[row_idx, 0].set_title(f"{label}: Prediction Error vs. Tuning Constant $c$", fontsize=11, fontweight="bold")
    axes[row_idx, 0].set_xlabel("Welsch Tuning Constant $c$", fontweight="bold")
    axes[row_idx, 0].set_ylabel("Median Test MedSPE", fontweight="bold")
    axes[row_idx, 0].grid(True, which='both', linestyle='--', color='#dcdde1', alpha=0.8)
    axes[row_idx, 0].legend()

    # Right Column: TP & FP vs c
    axes[row_idx, 1].plot(summary["c"], summary["TP_mean"], "-o", color=c_tp, linewidth=2, label="True Positives (TP)")
    axes[row_idx, 1].plot(summary["c"], summary["FP_mean"], "-o", color=c_fp, linewidth=2, label="False Positives (FP)")
    axes[row_idx, 1].axvline(2.11, color=c_default, linestyle="--", label="Default $c = 2.11$")
    axes[row_idx, 1].set_title(f"{label}: Variable Selection vs. Tuning Constant $c$", fontsize=11, fontweight="bold")
    axes[row_idx, 1].set_xlabel("Welsch Tuning Constant $c$", fontweight="bold")
    axes[row_idx, 1].set_ylabel("Average Selected Variables", fontweight="bold")
    axes[row_idx, 1].grid(True, which='both', linestyle='--', color='#dcdde1', alpha=0.8)
    axes[row_idx, 1].legend()

plot_dataset(0, summary_bos, "Boston Housing")
plot_dataset(1, summary_hbk, "HBK")
plot_dataset(2, summary_nci, "NCI60")

plt.tight_layout()
fig_path = os.path.join(fig_dir, "sensitivity_analysis_combined.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved combined sensitivity analysis plot to {fig_path}")

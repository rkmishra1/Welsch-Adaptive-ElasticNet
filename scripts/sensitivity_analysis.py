#!/usr/bin/env python
# scripts/sensitivity_analysis.py

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

# Load Boston Housing dataset
boston_path = "/Users/ramakrushnamishra/welsch-adenet/data/Boston.csv"
boston_df = pd.read_csv(boston_path)
X_boston = boston_df.drop("medv", axis=1).values
y_boston = boston_df["medv"].values

# Generate and append noise columns to evaluate variable selection
p_noise = 30
p_orig = X_boston.shape[1]
n_samples = X_boston.shape[0]

# Generate correlated noise
rho_noise = 0.8
corr_orig_noise = 0.7
Z = np.random.randn(n_samples, p_noise)
X_noise = Z.copy()
innov_scale = np.sqrt(1.0 - rho_noise**2)
for j in range(1, p_noise):
    X_noise[:, j] = rho_noise * X_noise[:, j - 1] + innov_scale * Z[:, j]
for j in range(min(5, p_orig, p_noise)):
    X_noise[:, j] = corr_orig_noise * X_boston[:, j] + np.sqrt(1.0 - corr_orig_noise**2) * X_noise[:, j]

X_full = np.hstack([X_boston, X_noise])

# Define grid of Welsch tuning constant c values
c_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
B = 10  # number of replications for sensitivity analysis

results = []

print("Running Sensitivity Analysis (varying Welsch tuning constant c)...")
for c in c_values:
    print(f"  Evaluating c = {c}...")
    for b in range(1, B + 1):
        # Split and preprocess
        split_data = prepare_train_test(X_full, y_boston, train_prop=0.80)
        
        # Fit Welsch Adaptive Elastic-Net
        res = tune_competitor(
            split_data["X_train"], split_data["y_train"],
            loss_type="welsch", loss_param=c,
            n_l1=12, n_l2=5, max_iter=800
        )
        beta_hat = res["beta"]
        
        # Compute MedSPE
        preds = split_data["X_test"] @ beta_hat
        residuals = split_data["y_test"] - preds
        med_spe = np.median(residuals**2)
        
        # Variable selection counts
        tp = np.sum(np.abs(beta_hat[:p_orig]) > 1e-8)
        fp = np.sum(np.abs(beta_hat[p_orig:]) > 1e-8)
        
        results.append({
            "c": c,
            "Rep": b,
            "MedSPE": med_spe,
            "TP": tp,
            "FP": fp
        })

df = pd.DataFrame(results)

# Aggregate results
summary = df.groupby("c").agg(
    MedSPE_median=("MedSPE", "median"),
    MedSPE_se=("MedSPE", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
    TP_mean=("TP", "mean"),
    FP_mean=("FP", "mean")
).reset_index()

print("\nSensitivity Analysis Summary:")
print(summary.to_string(index=False))

# Plot sensitivity analysis
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 1. MedSPE vs c
axes[0].errorbar(
    summary["c"], summary["MedSPE_median"], yerr=summary["MedSPE_se"],
    fmt="-o", color="#0984E3", linewidth=2, elinewidth=1.5, capsize=4, label="Median MedSPE"
)
axes[0].set_title("Prediction Error vs. Tuning Constant $c$", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Welsch Tuning Constant $c$", fontweight="bold")
axes[0].set_ylabel("Median Test MedSPE", fontweight="bold")
axes[0].grid(linestyle="--", alpha=0.7)
axes[0].axvline(2.11, color="red", linestyle="--", label="Default $c = 2.11$")
axes[0].legend()

# 2. TP & FP vs c
axes[1].plot(summary["c"], summary["TP_mean"], "-o", color="#2ecc71", linewidth=2, label="True Positives (TP)")
axes[1].plot(summary["c"], summary["FP_mean"], "-o", color="#e74c3c", linewidth=2, label="False Positives (FP)")
axes[1].set_title("Variable Selection vs. Tuning Constant $c$", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Welsch Tuning Constant $c$", fontweight="bold")
axes[1].set_ylabel("Average Selected Variables", fontweight="bold")
axes[1].grid(linestyle="--", alpha=0.7)
axes[1].axvline(2.11, color="red", linestyle="--", label="Default $c = 2.11$")
axes[1].legend()

plt.tight_layout()
fig_path = os.path.join(fig_dir, "sensitivity_analysis.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved sensitivity analysis plot to {fig_path}")

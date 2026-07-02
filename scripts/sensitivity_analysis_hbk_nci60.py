#!/usr/bin/env python
# scripts/sensitivity_analysis_hbk_nci60.py

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
# 1. Load Datasets
# ============================================================

# --- HBK Data ---
hbk_df = pd.read_csv("/Users/ramakrushnamishra/welsch-adenet/data/hbk.csv")
X_hbk = hbk_df.drop("Y", axis=1).values
y_hbk = hbk_df["Y"].values

# Generate & Append correlated noise variables for HBK
p_noise_hbk = 15
p_orig_hbk = X_hbk.shape[1]
n_hbk = X_hbk.shape[0]
Z_hbk = np.random.randn(n_hbk, p_noise_hbk)
X_noise_hbk = Z_hbk.copy()
rho_noise = 0.8
corr_orig_noise = 0.7
innov_scale = np.sqrt(1.0 - rho_noise**2)
for j in range(1, p_noise_hbk):
    X_noise_hbk[:, j] = rho_noise * X_noise_hbk[:, j - 1] + innov_scale * Z_hbk[:, j]
for j in range(min(p_orig_hbk, p_noise_hbk)):
    X_noise_hbk[:, j] = corr_orig_noise * X_hbk[:, j] + np.sqrt(1.0 - corr_orig_noise**2) * X_noise_hbk[:, j]
X_full_hbk = np.hstack([X_hbk, X_noise_hbk])


# --- NCI60 Data ---
nci60_df = pd.read_csv("/Users/ramakrushnamishra/welsch-adenet/data/nci60_protein.csv")
X_nci = nci60_df.drop("DoublingTime", axis=1).values
y_nci = nci60_df["DoublingTime"].values

# Generate & Append correlated noise variables for NCI60
p_noise_nci = 30
p_orig_nci = X_nci.shape[1]
n_nci = X_nci.shape[0]
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
c_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.5, 10.0]
B = 10  # Replications

# --- HBK Run ---
hbk_results = []
print("Running HBK Sensitivity Analysis...")
for c in c_values:
    print(f"  Evaluating c = {c}...")
    for b in range(1, B + 1):
        split_data = prepare_train_test(X_full_hbk, y_hbk, train_prop=0.80)
        res = tune_competitor(
            split_data["X_train"], split_data["y_train"],
            loss_type="welsch", loss_param=c,
            n_l1=12, n_l2=5, max_iter=800
        )
        beta_hat = res["beta"]
        
        preds = split_data["X_test"] @ beta_hat
        residuals = split_data["y_test"] - preds
        med_spe = np.median(residuals**2)
        
        tp = np.sum(np.abs(beta_hat[:p_orig_hbk]) > 1e-8)
        fp = np.sum(np.abs(beta_hat[p_orig_hbk:]) > 1e-8)
        
        hbk_results.append({"c": c, "Rep": b, "MedSPE": med_spe, "TP": tp, "FP": fp})

df_hbk = pd.DataFrame(hbk_results)
summary_hbk = df_hbk.groupby("c").agg(
    MedSPE_median=("MedSPE", "median"),
    MedSPE_se=("MedSPE", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
    TP_mean=("TP", "mean"),
    FP_mean=("FP", "mean")
).reset_index()


# --- NCI60 Run ---
nci_results = []
print("Running NCI60 Sensitivity Analysis...")
for c in c_values:
    print(f"  Evaluating c = {c}...")
    for b in range(1, B + 1):
        split_data = prepare_train_test(X_full_nci, y_nci, train_prop=0.80)
        res = tune_competitor(
            split_data["X_train"], split_data["y_train"],
            loss_type="welsch", loss_param=c,
            n_l1=12, n_l2=5, max_iter=800
        )
        beta_hat = res["beta"]
        
        preds = split_data["X_test"] @ beta_hat
        residuals = split_data["y_test"] - preds
        med_spe = np.median(residuals**2)
        
        tp = np.sum(np.abs(beta_hat[:p_orig_nci]) > 1e-8)
        fp = np.sum(np.abs(beta_hat[p_orig_nci:]) > 1e-8)
        
        nci_results.append({"c": c, "Rep": b, "MedSPE": med_spe, "TP": tp, "FP": fp})

df_nci = pd.DataFrame(nci_results)
summary_nci = df_nci.groupby("c").agg(
    MedSPE_median=("MedSPE", "median"),
    MedSPE_se=("MedSPE", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
    TP_mean=("TP", "mean"),
    FP_mean=("FP", "mean")
).reset_index()


# ============================================================
# 3. Plotting Results
# ============================================================
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

# --- Row 1: HBK Dataset ---
# 1a. MedSPE vs c
axes[0, 0].errorbar(
    summary_hbk["c"], summary_hbk["MedSPE_median"], yerr=summary_hbk["MedSPE_se"],
    fmt="-o", color="#0984E3", linewidth=2, elinewidth=1.5, capsize=4, label="Median MedSPE"
)
axes[0, 0].set_title("HBK: Prediction Error vs. Tuning Constant $c$", fontsize=11, fontweight="bold")
axes[0, 0].set_xlabel("Welsch Tuning Constant $c$", fontweight="bold")
axes[0, 0].set_ylabel("Median Test MedSPE", fontweight="bold")
axes[0, 0].grid(linestyle="--", alpha=0.7)
axes[0, 0].axvline(2.11, color="red", linestyle="--", label="Default $c = 2.11$")
axes[0, 0].legend()

# 1b. TP & FP vs c
axes[0, 1].plot(summary_hbk["c"], summary_hbk["TP_mean"], "-o", color="#2ecc71", linewidth=2, label="True Positives (TP)")
axes[0, 1].plot(summary_hbk["c"], summary_hbk["FP_mean"], "-o", color="#e74c3c", linewidth=2, label="False Positives (FP)")
axes[0, 1].set_title("HBK: Variable Selection vs. Tuning Constant $c$", fontsize=11, fontweight="bold")
axes[0, 1].set_xlabel("Welsch Tuning Constant $c$", fontweight="bold")
axes[0, 1].set_ylabel("Average Selected Variables", fontweight="bold")
axes[0, 1].grid(linestyle="--", alpha=0.7)
axes[0, 1].axvline(2.11, color="red", linestyle="--", label="Default $c = 2.11$")
axes[0, 1].legend()


# --- Row 2: NCI60 Dataset ---
# 2a. MedSPE vs c
axes[1, 0].errorbar(
    summary_nci["c"], summary_nci["MedSPE_median"], yerr=summary_nci["MedSPE_se"],
    fmt="-o", color="#0984E3", linewidth=2, elinewidth=1.5, capsize=4, label="Median MedSPE"
)
axes[1, 0].set_title("NCI60: Prediction Error vs. Tuning Constant $c$", fontsize=11, fontweight="bold")
axes[1, 0].set_xlabel("Welsch Tuning Constant $c$", fontweight="bold")
axes[1, 0].set_ylabel("Median Test MedSPE", fontweight="bold")
axes[1, 0].grid(linestyle="--", alpha=0.7)
axes[1, 0].axvline(2.11, color="red", linestyle="--", label="Default $c = 2.11$")
axes[1, 0].legend()

# 2b. TP & FP vs c
axes[1, 1].plot(summary_nci["c"], summary_nci["TP_mean"], "-o", color="#2ecc71", linewidth=2, label="True Positives (TP)")
axes[1, 1].plot(summary_nci["c"], summary_nci["FP_mean"], "-o", color="#e74c3c", linewidth=2, label="False Positives (FP)")
axes[1, 1].set_title("NCI60: Variable Selection vs. Tuning Constant $c$", fontsize=11, fontweight="bold")
axes[1, 1].set_xlabel("Welsch Tuning Constant $c$", fontweight="bold")
axes[1, 1].set_ylabel("Average Selected Variables", fontweight="bold")
axes[1, 1].grid(linestyle="--", alpha=0.7)
axes[1, 1].axvline(2.11, color="red", linestyle="--", label="Default $c = 2.11$")
axes[1, 1].legend()


plt.tight_layout()
fig_path = os.path.join(fig_dir, "sensitivity_analysis_hbk_nci60.png")
plt.savefig(fig_path, dpi=150)
plt.close()
print(f"\nSaved sensitivity analysis plots for HBK and NCI60 to {fig_path}")

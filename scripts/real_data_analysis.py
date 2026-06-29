#!/usr/bin/env python
# scripts/real_data_analysis.py

import os
import subprocess
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare

from welsch_adenet.competitors import tune_competitor, robust_init

# Set random seed for reproducibility
np.random.seed(42)

# Output directories
fig_dir = "/Users/ramakrushnamishra/welsch-adenet/docs/figures"
res_dir = "/Users/ramakrushnamishra/welsch-adenet/results"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

# Helper function to generate correlated noise
def generate_correlated_noise(n_samples, p_noise, X_orig, rho_noise=0.8, corr_orig_noise=0.7):
    Z = np.random.randn(n_samples, p_noise)
    X_noise = Z.copy()
    if p_noise > 1:
        innov_scale = np.sqrt(1.0 - rho_noise**2)
        for j in range(1, p_noise):
            X_noise[:, j] = rho_noise * X_noise[:, j - 1] + innov_scale * Z[:, j]
            
    # Correlation between first 5 noise variables and first 5 original features (or all if p_orig < 5)
    p_orig = X_orig.shape[1]
    limit = min(5, p_orig, p_noise)
    if limit > 0:
        for j in range(limit):
            X_noise[:, j] = corr_orig_noise * X_orig[:, j] + np.sqrt(1.0 - corr_orig_noise**2) * X_noise[:, j]
            
    return X_noise

# Helper to split, scale, and center data (now default to 80/20)
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

# Dataset Benchmark Runner
def run_dataset_benchmark(X_orig, y_orig, p_noise, B=20, dataset_name=""):
    print(f"\n==================================================")
    print(f"Benchmarking Dataset: {dataset_name} (n = {X_orig.shape[0]}, p_orig = {X_orig.shape[1]}, p_noise = {p_noise})")
    print(f"==================================================")
    
    methods = ["AdL", "AdEnet", "HAdL", "T-AdL", "S-LTS", "RLARS", "Welsch-AdEnet"]
    results_list = []
    
    p_orig = X_orig.shape[1]
    p_total = p_orig + p_noise
    n_samples = X_orig.shape[0]
    
    for b in range(1, B + 1):
        print(f"  Replication {b}/{B}...")
        
        # Generate and append noise columns
        X_noise = generate_correlated_noise(n_samples, p_noise, X_orig)
        X_full = np.hstack([X_orig, X_noise])
        
        # Split and preprocess (80% training, 20% testing)
        split_data = prepare_train_test(X_full, y_orig, train_prop=0.80)
        
        # Run S-LTS and R-LARS in R via helper script
        beta_s_lts = np.zeros(p_total)
        beta_rlars = np.zeros(p_total)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            train_x_file = os.path.join(temp_dir, "train_x.csv")
            train_y_file = os.path.join(temp_dir, "train_y.csv")
            test_x_file = os.path.join(temp_dir, "test_x.csv")
            out_file = os.path.join(temp_dir, "out.csv")
            
            np.savetxt(train_x_file, split_data["X_train"], delimiter=",")
            np.savetxt(train_y_file, split_data["y_train"], delimiter=",")
            np.savetxt(test_x_file, split_data["X_test"], delimiter=",")
            
            cmd = [
                "Rscript", "/Users/ramakrushnamishra/welsch-adenet/scripts/fit_r_competitors.R",
                train_x_file, train_y_file, test_x_file, out_file
            ]
            try:
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                r_coefs = pd.read_csv(out_file)
                beta_s_lts = r_coefs["S_LTS"].values
                beta_rlars = r_coefs["RLARS"].values
            except Exception as re:
                print(f"    R helper failed in rep {b}: {str(re)}")
        
        for m in methods:
            try:
                if m == "S-LTS":
                    beta_hat = beta_s_lts.copy()
                elif m == "RLARS":
                    beta_hat = beta_rlars.copy()
                else:
                    loss_type = "squared"
                    loss_param = None
                    n_l2 = 5
                    
                    if m == "AdL":
                        loss_type = "squared"
                        n_l2 = 1
                    elif m == "AdEnet":
                        loss_type = "squared"
                    elif m == "HAdL":
                        loss_type = "huber"
                        loss_param = 1.345
                        n_l2 = 1
                    elif m == "T-AdL":
                        loss_type = "tukey"
                        loss_param = 4.685
                        n_l2 = 1
                    elif m == "Welsch-AdEnet":
                        loss_type = "welsch"
                        loss_param = 2.11
                        
                    res = tune_competitor(
                        split_data["X_train"], split_data["y_train"],
                        loss_type=loss_type, loss_param=loss_param,
                        n_l1=12, n_l2=n_l2, max_iter=800
                    )
                    beta_hat = res["beta"]
                
                # Predict and compute errors
                preds = split_data["X_test"] @ beta_hat
                residuals = split_data["y_test"] - preds
                
                mspe = np.mean(residuals**2)
                med_spe = np.median(residuals**2)
                
                # Variable selection metrics
                active_vars = np.sum(np.abs(beta_hat) > 1e-8)
                signal_sel = np.sum(np.abs(beta_hat[:p_orig]) > 1e-8)
                noise_sel = np.sum(np.abs(beta_hat[p_orig:]) > 1e-8)
                
                # Precision, Recall, F1
                precision = signal_sel / active_vars if active_vars > 0 else 0.0
                recall = signal_sel / p_orig
                f1 = 2.0 * precision * recall / (precision + recall) if (precision + recall) > 0.0 else 0.0
                
                results_list.append({
                    "Dataset": dataset_name,
                    "Rep": b,
                    "Method": m,
                    "MSPE": mspe,
                    "MedSPE": med_spe,
                    "Active": active_vars,
                    "SignalSelected": signal_sel,
                    "NoiseSelected": noise_sel,
                    "Precision": precision,
                    "Recall": recall,
                    "F1": f1
                })
            except Exception as e:
                print(f"    Method {m} failed in rep {b}: {str(e)}")
                
    df = pd.DataFrame(results_list)
    
    # Compile summary statistics
    summary = df.groupby("Method").agg(
        MedSPE_median=("MedSPE", "median"),
        MedSPE_se=("MedSPE", lambda x: np.std(x, ddof=1) / np.sqrt(len(x))),
        TP_mean=("SignalSelected", "mean"),
        FP_mean=("NoiseSelected", "mean"),
        ModelSize_median=("Active", "median"),
        Precision_median=("Precision", "median"),
        Recall_median=("Recall", "median"),
        F1_median=("F1", "median")
    ).reset_index()
    
    print("\nSummary Results:")
    print(summary.to_string(index=False))
    
    return {"raw": df, "summary": summary}

# Load datasets
# 1. Boston Housing
boston_path = "/Users/ramakrushnamishra/welsch-adenet/data/Boston.csv"
boston_df = pd.read_csv(boston_path)
X_boston = boston_df.drop("medv", axis=1).values
y_boston = boston_df["medv"].values
boston_res = run_dataset_benchmark(X_boston, y_boston, p_noise=30, B=20, dataset_name="Boston")

# 2. hbk
hbk_path = "/Users/ramakrushnamishra/welsch-adenet/data/hbk.csv"
hbk_df = pd.read_csv(hbk_path)
X_hbk = hbk_df.drop("Y", axis=1).values
y_hbk = hbk_df["Y"].values
hbk_res = run_dataset_benchmark(X_hbk, y_hbk, p_noise=20, B=20, dataset_name="hbk")

# 3. NCI60 protein doubling time (High-Dimensional)
nci_path = "/Users/ramakrushnamishra/welsch-adenet/data/nci60_protein.csv"
nci_df = pd.read_csv(nci_path)
X_nci = nci_df.drop("DoublingTime", axis=1).values
y_nci = nci_df["DoublingTime"].values
nci_res = run_dataset_benchmark(X_nci, y_nci, p_noise=50, B=20, dataset_name="nci60")

# Save outputs
all_summaries = pd.concat([
    boston_res["summary"].assign(Dataset="Boston"),
    hbk_res["summary"].assign(Dataset="hbk"),
    nci_res["summary"].assign(Dataset="nci60")
])
all_summaries.to_csv(os.path.join(res_dir, "real_data_summary_results.csv"), index=False)

df_all_raw = pd.concat([boston_res["raw"], hbk_res["raw"], nci_res["raw"]])
df_all_raw.to_csv(os.path.join(res_dir, "real_data_raw_results.csv"), index=False)

# Plotting Function with Violin Plots and Precision/Recall/F1
def plot_dataset_results(dataset_name, res_dict, file_name):
    raw_df = res_dict["raw"]
    summary = res_dict["summary"]
    methods = ["AdL", "AdEnet", "HAdL", "T-AdL", "S-LTS", "RLARS", "Welsch-AdEnet"]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 1. MedSPE Violin Plots
    medspe_data = [raw_df[raw_df["Method"] == m]["MedSPE"].values for m in methods]
    parts = axes[0].violinplot(medspe_data, showmedians=True, showextrema=True)
    
    # Style violins
    for pc in parts['bodies']:
        pc.set_facecolor('#0984E3')
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)
    for partname in ['cmaxes', 'cmins', 'cbars', 'cmedians']:
        vp = parts[partname]
        vp.set_edgecolor('black')
        vp.set_linewidth(1.2)
        
    axes[0].set_title(f"{dataset_name}: MedSPE Distribution", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Median Squared Prediction Error (MedSPE)", fontweight="bold")
    axes[0].set_xticks(range(1, len(methods) + 1))
    axes[0].set_xticklabels(methods, rotation=35, ha="right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    
    # 2. Grouped bar chart of Precision, Recall, and F1-score
    summary["Method"] = pd.Categorical(summary["Method"], categories=methods, ordered=True)
    summary_sorted = summary.sort_values("Method")
    
    x_indices = np.arange(len(methods))
    width = 0.25
    
    axes[1].bar(x_indices - width, summary_sorted["Precision_median"], width, label="Precision", color="#3498db", edgecolor="black", alpha=0.9)
    axes[1].bar(x_indices, summary_sorted["Recall_median"], width, label="Recall", color="#2ecc71", edgecolor="black", alpha=0.9)
    axes[1].bar(x_indices + width, summary_sorted["F1_median"], width, label="F1-score", color="#e74c3c", edgecolor="black", alpha=0.9)
    
    axes[1].set_title(f"{dataset_name}: Variable Selection Metrics", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Metric Value (Median)", fontweight="bold")
    axes[1].set_xticks(x_indices)
    axes[1].set_xticklabels(methods, rotation=35, ha="right")
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(loc="lower left")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, file_name), dpi=150)
    plt.close()
    print(f"Saved figure {file_name}")

plot_dataset_results("Boston Housing", boston_res, "real_data_boston.png")
plot_dataset_results("hbk", hbk_res, "real_data_hbk.png")
plot_dataset_results("NCI60 Cancer Cell Lines", nci_res, "real_data_nci60.png")

# Friedman and Nemenyi post-hoc test & CD Diagram
def generate_cd_diagram(df_all, cd_file_name="cd_diagram.png"):
    df_all = df_all.copy()
    grouped = df_all.groupby(["Dataset", "Rep"])
    for name, group in grouped:
        group_sorted = group.sort_values("MedSPE")
        for rank, idx in enumerate(group_sorted.index, 1):
            df_all.loc[idx, "Rank"] = rank
            
    avg_ranks = df_all.groupby("Method")["Rank"].mean().reset_index()
    avg_ranks = avg_ranks.sort_values("Rank")
    print("\nAverage Ranks of Methods:")
    print(avg_ranks.to_string(index=False))
    
    medspe_pivot = df_all.pivot(index=["Dataset", "Rep"], columns="Method", values="MedSPE")
    methods = ["AdL", "AdEnet", "HAdL", "T-AdL", "S-LTS", "RLARS", "Welsch-AdEnet"]
    medspe_matrix = medspe_pivot[methods].dropna().values
    
    stat, p_val = friedmanchisquare(*[medspe_matrix[:, i] for i in range(len(methods))])
    print(f"\nFriedman test: stat = {stat:.4f}, p-value = {p_val:.4e}")
    
    plt.figure(figsize=(10, 3.5))
    ax = plt.gca()
    
    ax.hlines(0, 1, 7, colors="black", linewidths=1.5)
    for tick in range(1, 8):
        ax.plot([tick, tick], [-0.05, 0.05], color="black", linewidth=1.5)
        ax.text(tick, -0.15, str(tick), ha="center", fontsize=10, fontweight="bold")
        
    heights = [0.1, 0.22, 0.34, 0.46, 0.58, 0.70, 0.82]
    for i, row in enumerate(avg_ranks.itertuples()):
        method_name = row.Method
        rank_val = row.Rank
        
        ax.plot(rank_val, 0, 'ro', markersize=6)
        h = heights[i % len(heights)]
        ax.plot([rank_val, rank_val], [0, h], 'r--', linewidth=0.8)
        ax.text(rank_val, h + 0.02, f"{method_name} ({rank_val:.2f})", 
                ha="center", va="bottom", fontsize=9, fontweight="bold")
        
    cd_val = 3.031 * np.sqrt((7 * 8) / (6 * 60))
    ax.plot([1, 1 + cd_val], [-0.3, -0.3], color="blue", linewidth=3)
    ax.text(1 + cd_val/2, -0.28, f"CD = {cd_val:.3f}", ha="center", va="bottom", color="blue", fontsize=9, fontweight="bold")
    
    sorted_methods = avg_ranks["Method"].tolist()
    sorted_ranks = avg_ranks["Rank"].tolist()
    clique_y = -0.45
    for i in range(len(sorted_methods)):
        for j in range(i + 1, len(sorted_methods)):
            if sorted_ranks[j] - sorted_ranks[i] <= cd_val:
                ax.plot([sorted_ranks[i], sorted_ranks[j]], [clique_y, clique_y], color="gray", linewidth=2.5, solid_capstyle="round")
                clique_y -= 0.08
                
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")
    plt.title(f"Critical Difference Diagram (Friedman p = {p_val:.2e})", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, cd_file_name), dpi=150)
    plt.close()
    print(f"Saved CD diagram to {cd_file_name}")

generate_cd_diagram(df_all_raw, "cd_diagram.png")

print("\nAll analyses completed successfully!")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

# ---------------------------------------
#  تنظیمات اصلی: نرخ یادگیری و تکرارها
# ---------------------------------------
ALPHA      = 0.01   # learning rate α
ITERATIONS = 400    # number of gradient descent iterations

# -------------------------------
# 1) FEATURE NORMALIZATION
# -------------------------------
def feature_normalize(X):
    """
    Normalize features:
      $X_{\text{norm}} = \frac{X - \mu}{\sigma}$
    Returns:
      X_norm, mu, sigma
    """
    mu    = np.mean(X, axis=0)
    sigma = np.std(X,  axis=0, ddof=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

# -------------------------------
# 2) COMPUTE COST
# -------------------------------
def compute_cost(X, y, theta):
    """
    Compute cost function:
      $J(\theta) = \frac{1}{2m}\,(X\theta - y)^T (X\theta - y)$
    """
    m   = len(y)
    err = X.dot(theta) - y
    return (1.0 / (2*m)) * np.dot(err, err)

# -------------------------------
# 3) GRADIENT DESCENT
# -------------------------------
def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Performs gradient descent to learn θ:
      $\theta := \theta - \alpha\,\frac{1}{m}\,X^T (X\theta - y)$
    Returns:
      final theta, history of J(theta)
    """
    m = len(y)
    J_history = np.zeros(num_iters)

    for i in range(num_iters):
        error = X.dot(theta) - y
        grad  = (1.0/m) * X.T.dot(error)
        theta = theta - alpha * grad
        J_history[i] = compute_cost(X, y, theta)

    return theta, J_history

# -------------------------------
# 4) RUN ONE EXPERIMENT
# -------------------------------
def run_experiment(X_raw, y, alpha=ALPHA, num_iters=ITERATIONS, title=""):
    """
    1) Normalize features
    2) Add bias term
    3) Run gradient descent
    4) Print results
    Returns:
      theta, mu, sigma, J_history
    """
    X_norm, mu, sigma = feature_normalize(X_raw)
    m = len(y)

    # design matrix با ستون بایاس
    X = np.concatenate([np.ones((m,1)), X_norm], axis=1)

    theta_init = np.zeros(X.shape[1])
    theta, J_history = gradient_descent(X, y, theta_init, alpha, num_iters)

    # چاپ نتایج
    print(f"--- Results {title} ---")
    print("θ =", theta)
    print(f"Final cost = {J_history[-1]:.4f}")
    print("mu =", mu)
    print("sigma =", sigma, "\n")

    return theta, mu, sigma, J_history

# -------------------------------
# 5) HELPER: اضافه کردن باکس متن زیر نمودار
# -------------------------------
def add_text_box(ax, theta, J_history, mu, sigma, alpha, num_iters):
    txt = (
        f"θ = {np.array2string(theta, precision=4, separator=', ')}\n"
        f"$J_{{final}}$ = {J_history[-1]:.4f}\n"
        f"μ = {np.array2string(mu, precision=4, separator=', ')}\n"
        f"σ = {np.array2string(sigma, precision=4, separator=', ')}\n"
        f"α = {alpha}, iter = {num_iters}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # قرار دادن باکس در زیر هر subplot
    ax.text(0.5, -0.35, txt,
            transform=ax.transAxes,
            fontsize=8,
            va='top', ha='center',
            bbox=props)

# -------------------------------
# 6) MAIN FUNCTION
# -------------------------------
def main():
    # یافتن مسیر فایل housing.csv در کنار این اسکریپت
    script_dir = os.path.dirname(os.path.realpath(__file__))
    csv_path   = os.path.join(script_dir, "housing.csv")

    # خواندن داده‌ها (whitespace-delimited، بدون هدر)
    df   = pd.read_csv(csv_path, delim_whitespace=True, header=None)
    data = df.values

    # جداکردن X و y
    X_all   = data[:, :-1]       # همه ویژگی‌ها
    X_lstat = data[:, -2:-1]     # فقط ویژگی lstat
    y       = data[:, -1]        # متغیر هدف medv

    # --- Experiment 1: only lstat ---
    theta_lstat, mu_lstat, sigma_lstat, Jhist_lstat = run_experiment(
        X_lstat, y,
        alpha=ALPHA,
        num_iters=ITERATIONS,
        title="(Only lstat)"
    )

    # --- Experiment 2: all features ---
    theta_all, mu_all, sigma_all, Jhist_all = run_experiment(
        X_all, y,
        alpha=ALPHA,
        num_iters=ITERATIONS,
        title="(All features)"
    )

    # ===============================
    # 7) رسم هم‌زمان 4 نمودار (2×2)
    # ===============================
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # 7.1) Convergence (Only lstat)
    ax = axs[0, 0]
    ax.plot(np.arange(1, ITERATIONS+1), Jhist_lstat, 'b-', lw=2)
    ax.set_title("Convergence (Only lstat)")
    # برچسب y به صورت افقی و خواناتر
    ax.set_ylabel("Cost $J(\\theta)$", rotation=0, labelpad=30)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_xlabel("Iteration")
    ax.grid(True)
    add_text_box(ax, theta_lstat, Jhist_lstat, mu_lstat, sigma_lstat, ALPHA, ITERATIONS)

    # 7.2) Convergence (All features)
    ax = axs[0, 1]
    ax.plot(np.arange(1, ITERATIONS+1), Jhist_all, 'g-', lw=2)
    ax.set_title("Convergence (All features)")
    ax.set_ylabel("Cost $J(\\theta)$", rotation=0, labelpad=30)
    ax.yaxis.set_label_coords(-0.15, 0.5)
    ax.set_xlabel("Iteration")
    ax.grid(True)
    add_text_box(ax, theta_all, Jhist_all, mu_all, sigma_all, ALPHA, ITERATIONS)

    # 7.3) Scatter + Fit (lstat)
    ax = axs[1, 0]
    ax.scatter(X_lstat, y, c='blue', marker='x', label='Training data')
    x_vals = np.linspace(X_lstat.min(), X_lstat.max(), 100)
    x_norm = (x_vals - mu_lstat[0]) / sigma_lstat[0]
    X_line = np.column_stack([np.ones_like(x_norm), x_norm])
    y_line = X_line.dot(theta_lstat)
    ax.plot(x_vals, y_line, 'r-', lw=2, label='Linear fit')
    ax.set_title('Fit on Single Feature (lstat)')
    ax.set_xlabel('lstat')
    ax.set_ylabel('medv')
    ax.legend()
    ax.grid(True) 

    # 7.4) Actual vs Predicted (All features)
    ax = axs[1, 1]
    X_all_norm = (X_all - mu_all) / sigma_all
    X_design   = np.concatenate([np.ones((len(y),1)), X_all_norm], axis=1)
    y_pred     = X_design.dot(theta_all)
    ax.scatter(y, y_pred, c='purple', alpha=0.6, label='Predicted vs Actual')
    lims = [min(y.min(), y_pred.min()), max(y.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', lw=2, label='y = x')
    ax.set_title('Actual vs Predicted (All features)')
    ax.set_xlabel('Actual medv')
    ax.set_ylabel('Predicted medv')
    ax.legend()
    ax.grid(True) 

    # تنظیم فاصله‌ها و نمایش
    fig.tight_layout(pad=2.0)
    plt.show()

if __name__ == "__main__":
    main()
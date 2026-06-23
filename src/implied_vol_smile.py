"""
Inverts Monte Carlo option prices to Black-Scholes implied vol and plots
the 3-model smile comparison at T=0.25 (Output 1, the flagship figure).
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from option_pricing import bs_call, S0, R


def implied_vol(price, S, K, T, r, lo=1e-4, hi=5.0):
    """Brent's method root-find on the BS price function. Returns NaN if no sign change
    in [lo, hi] (e.g. price is below intrinsic value due to MC noise)."""
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if price <= intrinsic:
        return np.nan
    f = lambda sigma: bs_call(sigma, S, K, T, r) - price
    try:
        return brentq(f, lo, hi)
    except ValueError:
        return np.nan


def add_implied_vols(df):
    df = df.copy()
    df["implied_vol"] = df.apply(
        lambda row: implied_vol(row["price"], S0, row["K"], row["T"], R), axis=1
    )
    return df


def plot_smile(df, T_target=0.25, out_path="plots/implied_vol_smile.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    sub = df[np.isclose(df["T"], T_target)]
    labels = {
        "classical_heston": "Classical Heston",
        "rough_heston": "Rough Heston",
        "our_model": "Our Model (Rough + NIG)",
    }
    for model_name, label in labels.items():
        model_data = sub[sub["model"] == model_name].sort_values("K_over_S")
        plt.plot(model_data["K_over_S"], model_data["implied_vol"], marker="o", label=label)

    plt.xlabel("K / S")
    plt.ylabel("Implied volatility")
    plt.title(f"Implied volatility smile at T={T_target}")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig(out_path, dpi=150)
    print(f"Saved smile plot to {out_path}")


if __name__ == "__main__":
    # sanity check: invert a known BS price and recover the input sigma
    known_sigma = 0.25
    known_price = bs_call(known_sigma, S=100, K=100, T=0.5, r=0.03)
    recovered = implied_vol(known_price, S=100, K=100, T=0.5, r=0.03)
    print(f"Inversion sanity check: input sigma={known_sigma}, recovered={recovered:.4f}")

    csv_path = "data/csv/option_prices.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found — run option_pricing.py first.")

    df = pd.read_csv(csv_path)
    df = add_implied_vols(df)
    df.to_csv("data/csv/option_prices_with_iv.csv", index=False)
    plot_smile(df)

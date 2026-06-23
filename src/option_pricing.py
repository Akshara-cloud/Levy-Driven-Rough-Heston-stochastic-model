"""
Monte Carlo option pricing grid across 3 models: Classical Heston,
Rough Heston, and Our Model (Rough Heston + NIG Lévy jumps).

NOTE: the path simulators below for "rough_heston" and "our_model" are
PLACEHOLDERS using the deterministic Volterra forward-variance curve as
a proxy for the stochastic vt process. Swap simulate_rough_heston_paths
and simulate_our_model_paths for Priyesh's actual hybrid-scheme + NIG
Monte Carlo paths (Day 8-11 work) once ready. The grid/CSV/BS machinery
below does not need to change when you do that swap.
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import norm

from volterra.graded_mesh import solve_graded_mesh

# ---- Global config ----
S0 = 100.0
R = 0.03          # risk-free rate
MU = R            # drift under pricing measure
KAPPA, THETA, V0 = 1.0, 0.2, 0.04
OMEGA_ROUGH = 0.7   # H=0.2
N_PATHS = 1000
N_STEPS_PER_YEAR = 252

MONEYNESS_GRID = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
MATURITIES = [0.25, 0.5, 1.0]


def bs_call(sigma, S, K, T, r):
    """Black-Scholes call price — used both as a sanity check and inside the implied-vol inversion."""
    if sigma <= 0 or T <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def _euler_terminal_prices(v_path_fn, T, n_paths, seed):
    """Shared Euler-Maruyama driver: takes a function returning the vt path, returns S_T array."""
    rng = np.random.default_rng(seed)
    n_steps = max(int(N_STEPS_PER_YEAR * T), 10)
    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)
    v_path = v_path_fn(t_grid)

    S = np.full(n_paths, S0)
    for n in range(n_steps):
        vn = max(v_path[n], 0.0)
        Z = rng.standard_normal(n_paths)
        S = S * np.exp((MU - vn / 2) * dt + np.sqrt(vn * dt) * Z)
    return S


def simulate_classical_heston_paths(T, n_paths=N_PATHS, seed=1):
    """omega=1 special case: v(t) = theta + (v0-theta)*exp(-kappa*t), deterministic but the
    real model's long-run baseline — used here as a clean Classical Heston comparator."""
    v_fn = lambda t: THETA + (V0 - THETA) * np.exp(-KAPPA * t)
    return _euler_terminal_prices(v_fn, T, n_paths, seed)


def simulate_rough_heston_paths(T, n_paths=N_PATHS, seed=2):
    """PLACEHOLDER: deterministic graded-mesh v_bar(t) as vt proxy. Replace with stochastic
    Volterra+Brownian Monte Carlo paths for the real Rough Heston comparator."""
    t_ref, v_ref = solve_graded_mesh(400, T, OMEGA_ROUGH, KAPPA, THETA, V0)
    v_fn = lambda t: np.interp(t, t_ref, v_ref)
    return _euler_terminal_prices(v_fn, T, n_paths, seed)


def simulate_our_model_paths(T, n_paths=N_PATHS, seed=3):
    """PLACEHOLDER: same deterministic vt proxy as rough_heston, plus a crude left-skewed
    jump term bolted onto log-returns to mimic NIG downside asymmetry. Replace entirely with
    Priyesh's hybrid-scheme + NIG simulator output (Day 8-11) once available."""
    base = simulate_rough_heston_paths(T, n_paths, seed)
    rng = np.random.default_rng(seed + 100)
    jump_factor = np.exp(rng.normal(loc=-0.01, scale=0.05, size=n_paths))  # crude left-skew stand-in
    return base * jump_factor


MODEL_SIMULATORS = {
    "classical_heston": simulate_classical_heston_paths,
    "rough_heston": simulate_rough_heston_paths,
    "our_model": simulate_our_model_paths,
}


def price_grid_for_model(model_name):
    sim_fn = MODEL_SIMULATORS[model_name]
    rows = []
    for T in MATURITIES:
        S_T = sim_fn(T)
        for m in MONEYNESS_GRID:
            K = m * S0
            price = np.exp(-R * T) * np.mean(np.maximum(S_T - K, 0.0))
            rows.append({"model": model_name, "T": T, "K_over_S": m, "K": K, "price": price})
    return rows


def build_full_grid(out_csv="data/csv/option_prices.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    all_rows = []
    for model_name in MODEL_SIMULATORS:
        print(f"Pricing grid: {model_name}")
        all_rows.extend(price_grid_for_model(model_name))
        # save incrementally so a crash mid-run doesn't lose earlier models
        pd.DataFrame(all_rows).to_csv(out_csv, index=False)
    print(f"Saved {len(all_rows)} rows to {out_csv}")
    return pd.DataFrame(all_rows)


if __name__ == "__main__":
    # sanity check bs_call against a known case before trusting anything downstream
    test_price = bs_call(sigma=0.2, S=100, K=100, T=1.0, r=0.03)
    print(f"BS sanity check (ATM, sigma=0.2): {test_price:.4f}  (expect ~9.5-9.6)")

    df = build_full_grid()
    print(df.head(10))
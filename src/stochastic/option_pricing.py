import os
import csv
import math
import numpy as np
from scipy.stats import norm

from src.stochastic.stochastic_volterra import stochastic_volterra
from src.stochastic.nig_levy import nig_increments

# global configurations matching other files
S_0 = 100.0
r = 0.03          # risk-free rate
mu = r            # drift
k = 1.0           # mean reversion speed
theta = 0.2       # long-term mean of variance
v_0 = 0.04        # initial variance
sigma = 0.3       # vol-of-vol
a = 3.0           # NIG parameter a
b = -1.0          # NIG parameter b
w = 0.7           # Volterra parameter (w = H + 0.5 for H=0.2)
n_paths = 1000
n_steps_per_year = 252

moneyness_grid = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
maturities = [0.25, 0.5, 1.0]

# Black-Scholes call price
def bs_call(sigma_val, S, K, T, risk_free):
    if sigma_val <= 0 or T <= 0:
        return max(S - K * np.exp(-risk_free * T), 0.0)
    d1 = (np.log(S / K) + (risk_free + 0.5 * sigma_val**2) * T) / (sigma_val * np.sqrt(T))
    d2 = d1 - sigma_val * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-risk_free * T) * norm.cdf(d2)

# Custom graded mesh constructor for option pricing to avoid hardcoded T
def t_gm_pricing(N, w_for_grading, T_val):
    r_val = 1 + math.ceil(1 / w_for_grading)
    return ((np.arange(N + 1) / N) ** r_val) * T_val

# Vectorized simulation of stochastic Volterra paths to run fast
def stochastic_volterra_vectorized(t, w, v0, theta, k, sigma, a, b, lam, num_paths):
    N = len(t) - 1
    dt = np.diff(t)
    
    v_t = np.zeros((num_paths, N + 1))
    v_t[:, 0] = v0
    
    dL = np.zeros((num_paths, N))
    for p in range(num_paths):
        dL[p, :] = nig_increments(N, t, a, b)
        
    inv_gamma = 1.0 / math.gamma(w)
    for i in range(1, N + 1):
        j_indices = np.arange(i)
        kernel_weight = ((t[i] - t[j_indices]) ** w - (t[i] - t[j_indices + 1]) ** w) / w
        
        v_d = np.sum(kernel_weight * k * (theta - v_t[:, :i]), axis=1)
        sqrt_v = np.sqrt(np.maximum(v_t[:, :i], 0.0))
        v_s_terms = sigma * sqrt_v * (lam * dL[:, :i]) * (kernel_weight / dt[:i])
        v_s = np.sum(v_s_terms, axis=1)
        
        v_t[:, i] = np.maximum(v0 + inv_gamma * (v_d + v_s), 0.0)
        
    return v_t

# Euler-Maruyama simulator using vectorized variance paths
def _euler_terminal_prices_vectorized(v_path, T_val, num_paths, seed):
    rng = np.random.default_rng(seed)
    n_steps = v_path.shape[1] - 1
    dt = T_val / n_steps
    
    S = np.full(num_paths, S_0)
    for n in range(n_steps):
        vn = np.maximum(v_path[:, n], 0.0)
        Z = rng.standard_normal(num_paths)
        S = S * np.exp((mu - vn / 2.0) * dt + np.sqrt(vn * dt) * Z)
    return S

# Classical Heston paths (omega = 1, deterministic mean-reverting variance path)
def simulate_classical_heston_paths(T_val, num_paths=n_paths, seed=1):
    n_steps = max(int(n_steps_per_year * T_val), 10)
    t = t_gm_pricing(n_steps, w, T_val)
    v_single = theta + (v_0 - theta) * np.exp(-k * t)
    v_path = np.tile(v_single, (num_paths, 1))
    return _euler_terminal_prices_vectorized(v_path, T_val, num_paths, seed)

# Rough Heston paths (omega = 0.7, deterministic Volterra variance path)
def simulate_rough_heston_paths(T_val, num_paths=n_paths, seed=2):
    n_steps = max(int(n_steps_per_year * T_val), 10)
    t = t_gm_pricing(n_steps, w, T_val)
    v_single = stochastic_volterra(t, w, v_0, theta, k, sigma, a, b, lam=0.0)
    v_path = np.tile(v_single, (num_paths, 1))
    return _euler_terminal_prices_vectorized(v_path, T_val, num_paths, seed)

# Our Model paths (stochastic Volterra variance paths driven by NIG increments)
def simulate_our_model_paths(T_val, num_paths=n_paths, seed=3):
    n_steps = max(int(n_steps_per_year * T_val), 10)
    t = t_gm_pricing(n_steps, w, T_val)
    v_path = stochastic_volterra_vectorized(t, w, v_0, theta, k, sigma, a, b, lam=1.0, num_paths=num_paths)
    return _euler_terminal_prices_vectorized(v_path, T_val, num_paths, seed)

model_simulators = {
    "classical_heston": simulate_classical_heston_paths,
    "rough_heston": simulate_rough_heston_paths,
    "our_model": simulate_our_model_paths,
}

# price call options on the grid for a model
def price_grid_for_model(model_name):
    sim_fn = model_simulators[model_name]
    rows = []
    for T_val in maturities:
        S_T = sim_fn(T_val)
        for m in moneyness_grid:
            K = m * S_0
            price = np.exp(-r * T_val) * np.mean(np.maximum(S_T - K, 0.0))
            rows.append({"model": model_name, "T": T_val, "K_over_S": m, "K": K, "price": price})
    return rows

# build the pricing grid and print/save to CSV if needed
def build_full_grid(out_csv="data/csv/option_prices.csv"):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    all_rows = []
    for model_name in model_simulators:
        print(f"Pricing grid: {model_name}")
        all_rows.extend(price_grid_for_model(model_name))
        
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["model", "T", "K_over_S", "K", "price"])
            writer.writeheader()
            writer.writerows(all_rows)
            
    print(f"Saved {len(all_rows)} rows to {out_csv}")
    return all_rows

if __name__ == "__main__":
    test_price = bs_call(sigma_val=0.2, S=100.0, K=100.0, T=1.0, risk_free=0.03)
    print(f"BS sanity check (ATM, sigma=0.2): {test_price:.4f} (expect ~9.5-9.6)")

    rows = build_full_grid()
    print("Preview of grid:")
    for row in rows[:10]:
        print(row)

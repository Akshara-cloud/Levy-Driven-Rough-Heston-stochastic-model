import math
import numpy as np
import matplotlib.pyplot as plt

from src.volterra.volterra_solver import solve, t_gm
from src.stochastic.nig_levy import nig_increments

# Parameters
S_0 = 100.0
r = 0.03
mu = r
theta = 0.2
v_0 = 0.04
sigma = 0.3
a = 3.0
b = -1.0
w = 0.7  # H=0.2 
N_steps = 200
num_paths = 2000

def stochastic_volterra_vectorized(t, w, v0, theta, sigma, a, b, k=1, lam=1, num_paths=200, noise_type="nig", seed=3):
    N = len(t) - 1
    dt = np.diff(t)
    
    v_t = np.zeros((num_paths, N + 1))
    v_t[:, 0] = v0
    
    dL = np.zeros((num_paths, N))
    if noise_type == "brownian":
        rng = np.random.default_rng(seed)
        dL = rng.standard_normal((num_paths, N)) * np.sqrt(dt)
    else:
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

def simulate_asset_paths(v_path, T_val, seed=42):
    rng = np.random.default_rng(seed)
    num_paths = v_path.shape[0]
    n_steps = v_path.shape[1] - 1
    dt = T_val / n_steps
    
    S = np.zeros((num_paths, n_steps + 1))
    S[:, 0] = S_0
    
    for n in range(n_steps):
        vn = np.maximum(v_path[:, n], 0.0)
        Z = rng.standard_normal(num_paths)
        S[:, n+1] = S[:, n] * np.exp((mu - vn / 2.0) * dt + np.sqrt(vn * dt) * Z)
        
    return S

def run_validation():
    # getting graded mesh
    t = t_gm(N_steps, w)
    T_val = t[-1] # Should be 2.0 based on volterra_solver
    
    # calculate deterministic mean field volterra solution 
    print("Computing deterministic mean-field Volterra solution...")
    v_t_det = np.zeros(N_steps + 1)
    v_t_det[0] = v_0
    v_t_det = solve(v_t_det, t, w, v_0, theta, N_steps)
    
    # simulating stochastic volterra paths 
    print(f"Simulating {num_paths} stochastic Volterra paths...")
    v_paths = stochastic_volterra_vectorized(t, w, v_0, theta, sigma, a, b, k=1, lam=1, num_paths=num_paths)
    
    # mean across all paths
    v_mc_mean = np.mean(v_paths, axis=0)
    
    # plotting variance validation
    plt.figure(figsize=(10, 6))
    plt.plot(t, v_t_det, label=r"Deterministic v_t", color='black', linewidth=3)
    plt.plot(t, v_mc_mean, label=f"Monte Carlo Average (N={num_paths})", color='red', linestyle='--')
    
    # plotting a few sample paths in background
    for p in range(min(5, num_paths)):
        plt.plot(t, v_paths[p, :], color='gray', alpha=0.2)
        
    plt.title("Monte Carlo Validation of Variance Process v_t")
    plt.xlabel("Time")
    plt.ylabel("Variance")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig("plots/variance_validation.png", dpi=150)
    print("Saved plots/variance_validation.png")
    
    # simulating asset paths 
    print("Simulating asset price paths S(t)...")
    S_paths = simulate_asset_paths(v_paths, T_val)
    
    plt.figure(figsize=(10, 6))
    for p in range(min(20, num_paths)):
        plt.plot(t, S_paths[p, :], alpha=0.6)
        
    # expected value
    expected_S = S_0 * np.exp(mu * t)
    plt.plot(t, expected_S, color='black', linewidth=3, linestyle='--', label=r"$E[S_t] = S_0 e^{\mu t}$")
    
    plt.title("Simulated Asset Price Paths $S_t$ (Our Model)")
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.savefig("plots/asset_paths.png", dpi=150)
    print("Saved plots/asset_paths.png")

if __name__ == "__main__":
    run_validation()

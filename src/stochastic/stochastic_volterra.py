import numpy as np
try:
    from src.stochastic.nig_levy import nig_increments
except ModuleNotFoundError:
    from nig_levy import nig_increments
import matplotlib.pyplot as plt
import math
from src.volterra.volterra_solver import t_gm

# Volatility process:
# vt = v0 + (1/Gamma(w)) ∫_0^t (t-s)^(w-1) k(theta-vs) ds + (1/Gamma(w)) ∫_0^t (t-s)^(w-1) sigma*sqrt(vs) dLs
# We have already implemented the deterministic Volterra term
# We now add the stochastic Lévy-driven term using simulated NIG increments

# lambda is for jump intensity ( plots are drawn later by varying this parameter )
def stochastic_volterra(t, w, v0, theta, k, sigma, a, b, lam=1.0):
    N = len(t) - 1
    dt = np.diff(t)

    v_t = np.zeros(N+1)
    v_t[0] = v0
    dL = nig_increments(N, t, a, b)

    inv_gamma = 1/math.gamma(w)
    for i in range(1, N+1):
        v_d = 0 # deterministic term 
        v_s = 0 # stochastic term 
        for j in range(i):
            kernel_weight = ((t[i]-t[j])**(w)-(t[i]-t[j+1])**(w))/w # taken from the volterra solver 
            v_d += kernel_weight * k * (theta-v_t[j])
            v_s += sigma * np.sqrt(max(v_t[j], 0.0)) * (lam * dL[j]) * (kernel_weight)/(t[j+1]-t[j])
        v_t[i] = max(v0 + inv_gamma * (v_d + v_s), 0.0)
    
    return v_t

# test or plot — only runs when this file is executed directly
if __name__ == "__main__":
    T = 2
    N = 1000
    w = 0.6
    v0 = 0.04
    k = 1.0
    theta = 0.2
    sigma = 0.3
    a = 3.0
    b = 1.0
    t = t_gm(N, w)

    plt.title("Levy driven Rough Heston Variance Paths")
    plt.xlabel("Time")
    plt.ylabel("Variance")
    plt.plot(t, stochastic_volterra(t, w, v0, theta, k, sigma, a, b))
    plt.show()

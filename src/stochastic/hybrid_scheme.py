import numpy as np
import matplotlib.pyplot as plt
from src.volterra.volterra_solver import t_gm

# Simplified hybrid-scheme approximation of the stochastic term X(t) = ∫_0^t (t-s)^(omega-1) dW(s)
# We approximate the integral by X(t_n) ≈ Σ_j (t_n-t_j)^(omega-1) ΔW_j 
# This allows us to study how fractional kernel weights create past shocks and create memory
# In the full model we replace the gaussian increments ( brownian motion ) with NIG levy increments

def rought_stochastic_integral(t, w):
    N = len(t)-1
    dt = np.diff(t)
    # brownian increments
    dW = np.random.normal(loc = 0, scale = np.sqrt(dt))

    X = np.zeros(N+1)
    for i in range(1, N+1):
        total = 0
        for j in range(i):
            total += ((t[i]-t[j])**(w-1))*dW[j]
        X[i] = total
    return X

# plot
T = 2
N = 500
w = 0.6
t = t_gm(N, w)

for i in range(5):
    X = rought_stochastic_integral(t, w)
    plt.plot(t, X)

plt.title("Simplified Hybrid Scheme")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.show()



import numpy as np
import matplotlib.pyplot as plt

from src.volterra.volterra_solver import t_gm
from src.stochastic.stochastic_volterra import stochastic_volterra

# common parameters
N = 500
v_0 = 0.04
theta = 0.2
k = 1.0
sigma = 0.3
a = 3.0
b = -1.0

# ==========================================================
# Vary the Hurst parameter (roughness index)
# ==========================================================

H_values = [0.1, 0.15, 0.2, 0.3]

fig, axs = plt.subplots(
    2,
    2,
    figsize=(12, 8),
    sharex=True,
    sharey=True
)

fig.suptitle("Volatility Paths $v_t$ under Varying Roughness $H$")

for idx, H in enumerate(H_values):
    w = H + 0.5

    t = t_gm(N, w)

    v = stochastic_volterra(
        t,
        w,
        v_0,
        theta,
        k,
        sigma,
        a,
        b
    )

    row = idx // 2
    col = idx % 2

    axs[row, col].plot(t, v)
    axs[row, col].set_title(f"H = {H}")
    axs[row, col].set_xlabel("Time")
    axs[row, col].set_ylabel("Variance")

plt.tight_layout()

plt.savefig(
    "plots/vary_H.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# ==========================================================
# Vary jump intensity λ
# ==========================================================

w = 0.7  # H = 0.2

t = t_gm(N, w)

np.random.seed(42)
v_no_jumps = stochastic_volterra(
    t,
    w,
    v_0,
    theta,
    k,
    sigma,
    a,
    b,
    lam=0.0
)

np.random.seed(42)
v_small = stochastic_volterra(
    t,
    w,
    v_0,
    theta,
    k,
    sigma,
    a,
    b,
    lam=0.1
)

np.random.seed(42)
v_large = stochastic_volterra(
    t,
    w,
    v_0,
    theta,
    k,
    sigma,
    a,
    b,
    lam=0.5
)

plt.figure(figsize=(8, 5))

plt.plot(
    t,
    v_no_jumps,
    label="λ = 0.0 (No Jumps)",
    color="black",
    linestyle="--"
)

plt.plot(
    t,
    v_small,
    label="λ = 0.1 (Small Jumps)",
    color="green"
)

plt.plot(
    t,
    v_large,
    label="λ = 0.5 (Large Jumps)",
    color="red"
)

plt.title("Volatility Paths $v_t$ under Varying Jump Intensity (H = 0.2)")
plt.xlabel("Time")
plt.ylabel("Variance")
plt.legend()

plt.savefig(
    "plots/vary_lambda.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
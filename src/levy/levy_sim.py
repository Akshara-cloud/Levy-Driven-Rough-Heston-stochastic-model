import numpy as np
from scipy.stats import norminvgauss
import matplotlib.pyplot as plt

# NIG parameters
a = 5
b = -2
scale = 0.01

# Number of time steps
n_steps = 1000

# Theoretical mean of one NIG increment
mean_increment = norminvgauss.mean(a, b, scale=scale)

# ----------------------------
# Plot 5 sample Lévy paths
# ----------------------------

plt.figure(figsize=(10, 6))

for i in range(5):

    increments = norminvgauss.rvs(
        a,
        b,
        scale=scale,
        size=n_steps
    )

    # Center using theoretical mean
    increments -= mean_increment

    path = np.cumsum(increments)

    plt.plot(path, label=f"Path {i+1}")

plt.title("5 NIG Lévy Sample Paths")
plt.xlabel("Time Step")
plt.ylabel("L(t)")
plt.legend()

plt.savefig("plots/nig_levy_5paths.png")

print("Plot saved to plots/nig_levy_5paths.png")

# ----------------------------
# Empirical mean check
# ----------------------------

num_paths = 1000
final_values = []

for _ in range(num_paths):

    increments = norminvgauss.rvs(
        a,
        b,
        scale=scale,
        size=n_steps
    )

    increments -= mean_increment

    path = np.cumsum(increments)

    final_values.append(path[-1])

empirical_mean = np.mean(final_values)

print(f"Empirical mean at final time over {num_paths} paths: {empirical_mean:.6f}")
import matplotlib.pyplot as plt
import numpy as np
from src.stochastic.nig_levy import levy_paths, nig_increments

T = 2
N = 1000
t = np.linspace(0, T, N+1)

# verify centering — mean of increments should be close to 0
dL = nig_increments(N, t, 2.0, -1.0)
print(np.mean(dL))

for i in range(5):
    L = levy_paths(N, t, a=3.0, b=-1.0)
    plt.plot(t, L)

plt.title("Sample NIG Lévy Paths")
plt.xlabel("Time")
plt.ylabel("L(t)")
plt.show()
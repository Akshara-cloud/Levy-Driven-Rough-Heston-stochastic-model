import matplotlib.pyplot as plt
import numpy as np
from nig_levy import levy_paths
from nig_levy import nig_increments

T = 2
N = 1000
t = np.linspace(0, T, N+1)

# verify centering 
dL = nig_increments(10000, 2, 2.0, 0.5)
print(np.mean(dL))

for i in range(5):
    L = levy_paths(N, T, a = 3.0, b = 1.0)
    plt.plot(t, L)

plt.title("Sample NIG Lévy Paths")
plt.xlabel("Time")
plt.ylabel("L(t)")
plt.show()
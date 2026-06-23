import matplotlib.pyplot as plt
import numpy as np
import math
from volterra_solver import solve 
from volterra_solver import t_um 
from volterra_solver import t_gm 
from volterra_solver import classical_heston 

# This code compares Uniform mesh with Graded mesh
# The graded mesh should cluster points near t = 0, because v(t) is changing rapidly in there 
T = 2
N = 10
k = 1
theta = 0.2
v_0 = 0.04 
w = 0.6
t1 = t_gm(N, w)
t11 = t_gm(N, 0.7)
t12 = t_um(N)

t2 = t_um(N)

v_t1 = np.zeros(N+1)
v_t1[0] = v_0
v_t2 = v_t1.copy()
v_new = v_t1.copy() # for another value of w

v_t1 = solve(v_t1, t1, w, v_0, theta, N)
v_t2 = solve(v_t2, t2, w, v_0, theta, N)

# compare two graded mesh plots with w = 0.6 and w = 0.7 and w = 1 ( classical heston )
v_new = solve(v_new, t11, 0.7, v_0, theta, N)
v_classical_heston = classical_heston(t12, v_0, theta, 1)

fig, axs = plt.subplots(2)
fig.suptitle('Forward Variance Comparision plots')

axs[0].set_xlabel("Time")
axs[0].set_ylabel("Variance")
axs[0].plot(t1, v_t1, color='r', label="Rough Heston (w = 0.6)")
axs[0].plot(t11, v_new, color='b', label="Rough Heston (w = 0.7)")
axs[0].plot(t12, v_classical_heston, color='g', label="Classical Heston (w = 1)")
axs[0].legend()

axs[1].set_xlabel("Time")
axs[1].set_ylabel("Variance")
axs[1].plot(t1, v_t1, color='r', label="graded_mesh_plot")
axs[1].plot(t2, v_t2, color='b', label="uniform_mesh_plot")
axs[1].legend()

print(v_t1[-1]) # outputs the volatility at T = 2
print(v_t2[-1])

plt.show()



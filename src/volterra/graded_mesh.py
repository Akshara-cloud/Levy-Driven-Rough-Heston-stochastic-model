import matplotlib.pyplot as plt
import numpy as np
import math
from volterra_solver import solve

# This code compares Uniform mesh with Graded mesh
# The graded mesh should cluster points near t = 0, because v(t) is changing rapidl in there 
# volterra equation ( discretized ) : v(tn) = v0 + sigma i=0 to n-1 (tn - ti)^(w-1).k(theta-v(ti))delta_t
T = 2
N = 10
k = 1
theta = 0.2
v_0 = 0.04 
w = 0.6
t1 = np.zeros(N+1) 

for i in range(0, N+1):
    t1[i] = ((i/N)**(1 + math.ceil(1/w)))*T

t2 = np.linspace(0, T, N+1)

v_t1 = np.zeros(N+1)
v_t1[0] = v_0
v_t2 = v_t1.copy()
v_new = v_t1.copy() # for another value of w

v_t1 = solve(v_t1, t1, w, v_0, theta, N)
v_t2 = solve(v_t2, t2, w, v_0, theta, N)

# compare two graded mesh plots with w = 0.6 and w = 0.7
v_new = solve(v_new, t1, 0.7, v_0, theta, N)

fig, axs = plt.subplots(2)
fig.suptitle('Graded Mesh Plots')

axs[0].plot(t1, v_t1, color='r', label="w = 0.6")
axs[0].plot(t1, v_new, color='b', label="w = 0.7")
axs[0].legend()

axs[1].plot(t1, v_t1, color='r', label="graded_mesh_plot")
axs[1].plot(t2, v_t2, color='b', label="uniform_mesh_plot")
axs[1].legend()

print(v_t1[-1]) # outputs the volatility at T = 2
print(v_t2[-1])

plt.show()



import matplotlib.pyplot as plt
import numpy as np
import math
from volterra_solver import solve
from volterra_solver import t_um

# volterra equation ( discretized ) : v(tn) = v0 + sigma i=0 to n-1 k(theta-v(ti)).intergral ti to ti+1 (tn - s)^(w-1) ds
# since kernel is changing rapidly we have to integrate instead of assuming ds = tj+1 - tj ( would cause errors )
T = 2
N = 100
k = 1
theta = 0.2
v_0 = 0.04 
w = 0.6
t = t_um(N) # arange is used when step size is N+1, linspace is used if no. of points are N+1
v_t = np.zeros(N+1)
v_t[0] = v_0

v_t = solve(v_t, t, w, v_0, theta, N)

plt.title("Uniform Mesh Solution")
plt.xlabel("time")
plt.ylabel("Average Volatility")
plt.plot(t, v_t)
print(v_t[-1])
plt.show()



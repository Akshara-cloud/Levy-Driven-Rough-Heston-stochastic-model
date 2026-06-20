import matplotlib.pyplot as plt
import numpy as np
import math

# This code compares Uniform mesh with Graded mesh
# The graded mesh should cluster points near t = 0, because v(t) is changing rapidl in there 
# volterra equation ( discretized ) : v(tn) = v0 + sigma i=0 to n-1 (tn - ti)^(w-1).k(theta-v(ti))delta_t
T = 2
N = 1000
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

for i in range(1, N+1):
    total1 = v_0
    total2 = v_0
    for j in range(0, i):
        # ignore k as it is 1 
        kernel_weight1 = ((t1[i]-t1[j])**(w)-(t1[i]-t1[j+1])**(w))/w
        kernel_weight2 = ((t2[i]-t2[j])**(w)-(t2[i]-t2[j+1])**(w))/w
        total1 +=  (1/math.gamma(w))*(theta-v_t1[j])*kernel_weight1
        total2 +=  (1/math.gamma(w))*(theta-v_t2[j])*kernel_weight2

    v_t1[i] = total1
    v_t2[i] = total2


plt.plot(t1, v_t1, color='r', label="graded_mesh_plot")
plt.plot(t2, v_t2, color='b', label="uniform_mesh_plot")
print(v_t1[-1]) # outputs the volatility at T = 2
print(v_t2[-1]) 
plt.legend()
plt.show()



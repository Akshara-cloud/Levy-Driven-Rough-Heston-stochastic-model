import matplotlib.pyplot as plt
import numpy as np
import math

# volterra equation ( discretized ) : v(tn) = v0 + sigma i=0 to n-1 k(theta-v(ti)).intergral ti to ti+1 (tn - s)^(w-1) ds
# since kernel is changing rapidly we have to integrate instead of assuming ds = tj+1 - tj ( would cause errors )
T = 2
N = 100
k = 1
theta = 0.2
v_0 = 0.04 
w = 0.6
t = np.linspace(0, T, N+1) # arange is used when step size is N+1, linspace is used if no. of points are N+1
v_t = np.zeros(N+1)
v_t[0] = v_0

for i in range(1, N+1):
    total = v_0
    for j in range(0, i):
        # ignore k as it is 1 
        kernel_weight = ((t[i]-t[j])**(w)-(t[i]-t[j+1])**(w))/w
        total +=  (1/math.gamma(w))*(theta-v_t[j])*kernel_weight
    v_t[i] = total

plt.title("Uniform Mesh Solution")
plt.xlabel("time")
plt.ylabel("Average Volatility")
plt.plot(t, v_t)
print(v_t)
plt.show()



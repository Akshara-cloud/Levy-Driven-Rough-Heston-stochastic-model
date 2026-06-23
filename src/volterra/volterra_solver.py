import math
import numpy as np

T = 2

def solve(v_t, t, w, v_0, theta, N):
    for i in range(1, N+1):
        total = v_0
        for j in range(0, i):
            # ignore k as it is 1 
            kernel_weight = ((t[i]-t[j])**(w)-(t[i]-t[j+1])**(w))/w
            total +=  (1/math.gamma(w))*(theta-v_t[j])*kernel_weight
        v_t[i] = total
    return v_t

# Uniform time mesh : equally spaced points on [0, T]
def t_um(N):
   return np.linspace(0, T, N+1)

# Graded time mesh: t_i = (i/N)^r T. Concentrates nodes near t = 0 to
# improve accuracy for rough Volterra equations with singular kernel
def t_gm(N, w_for_grading):
    r = 1 + math.ceil(1/w_for_grading)
    return ((np.arange(N+1)/N)**r) * T

# classical heston equation
def classical_heston(t, v_0, theta, k=1):
    return theta + (v_0 - theta) * np.exp(-k * t)
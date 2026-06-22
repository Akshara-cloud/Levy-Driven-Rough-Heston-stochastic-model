import matplotlib.pyplot as plt
import numpy as np
import math
from volterra_solver import solve

T = 2
Ref_N = 6400 
v_0 = 0.04
w = 0.6
theta = 0.2

def t_um(N):
   return np.linspace(0, T, N+1)

def t_gm(N): 
    t_gm = np.zeros(N+1) 

    for i in range(0, N+1):
        t_gm[i] = ((i/N)**(1 + math.ceil(1/w)))*T
    return t_gm

v_um = np.zeros(Ref_N+1)
v_um[0] = v_0
v_gm = v_um.copy()

v_um = solve(v_um, t_um(Ref_N), w, v_0, theta, Ref_N)
v_gm = solve(v_gm, t_gm(Ref_N), w, v_0, theta, Ref_N)

# sample quantities
ns = [10, 20, 40, 80, 160, 320]
errors_um = []
errors_gm = []

for i in range(len(ns)):
    vs1 = np.zeros(ns[i]+1)
    vs1[0] = v_0
    vs2 = np.zeros(ns[i]+1)
    vs2[0] = v_0 

    N = ns[i]

    # finding error using scaling
    vs1 = solve(vs1, t_um(N), w, v_0, theta, N)
    step = Ref_N // N
    err = np.max(np.abs(vs1 - v_um[::step]))
    errors_um.append(err)

    # finding error using interpolation
    tg = t_gm(N)
    vs2 = solve(vs2, tg, w, v_0, theta, N)
    v_ref_interp = np.interp(tg, t_gm(Ref_N), v_gm)
    err_gm = np.max(np.abs(vs2 - v_ref_interp))
    errors_gm.append(err_gm)

plt.title("Convergence Analysis")
plt.xlabel("log(N)")
plt.ylabel("log(error)")
plt.loglog(ns, errors_um, 'o-', label='Uniform')
plt.loglog(ns, errors_gm, 's-', label='Graded')
plt.legend()
plt.show()


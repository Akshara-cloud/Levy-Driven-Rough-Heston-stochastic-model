import math

def solve(v_t, t, w, v_0, theta, N):
    for i in range(1, N+1):
        total = v_0
        for j in range(0, i):
            # ignore k as it is 1 
            kernel_weight = ((t[i]-t[j])**(w)-(t[i]-t[j+1])**(w))/w
            total +=  (1/math.gamma(w))*(theta-v_t[j])*kernel_weight
        v_t[i] = total
    return v_t
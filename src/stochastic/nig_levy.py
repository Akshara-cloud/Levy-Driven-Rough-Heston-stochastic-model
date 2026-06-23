from scipy.stats import norminvgauss
import numpy as np 

# NIG sampler generates random samples from the Normal Inverse Gaussian Distribution 
# We use this NIG distribution to create the Levy Noise 
# L(tn) = sigma i = 1 to n (dLi)

# a controls tail heavyness, b controls skewness, a should be greater than |b|
def nig_increments(N, T, a, b):
    dt = np.diff(T) # use this instead of direct T/N since when kernel comes into consideration we need to use graded mesh and dt would be different for each interval
    a_dt, b_dt = a * dt, b * dt
    dL = norminvgauss.rvs(a_dt, b_dt, loc = 0, scale = dt, size=N, random_state = None) # at each step we draw a nig increment using random variate 
    mean = norminvgauss.mean(a_dt, b_dt, loc = 0, scale = dt) # for centering 
    return dL - mean 

# now turn these increments into levy paths 
def levy_paths(N, T, a, b):
    dL = nig_increments(N, T, a, b)
    # cumulative sum of these increments 
    L = np.concatenate([[0], np.cumsum(dL)])
    return L
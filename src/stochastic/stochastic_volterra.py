import numpy as np
from nig_levy import nig_increments
import math

# Volatility process:
# vt = v0 + (1/Gamma(w)) ∫_0^t (t-s)^(w-1) k(theta-vs) ds + (1/Gamma(w)) ∫_0^t (t-s)^(w-1) sigma*sqrt(vs) dLs
# We have already implemented the deterministic Volterra term
# We now add the stochastic Lévy-driven term using simulated NIG increments
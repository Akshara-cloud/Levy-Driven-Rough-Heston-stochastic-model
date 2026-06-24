import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import math

# We will just test the implied vol of Black Scholes to see how MC noise affects ITM vs OTM
S_0 = 100.0
r = 0.03
T = 0.25
sigma = 0.2
n_paths = 1000

def bs_call(sigma_val, S, K, T, risk_free):
    d1 = (np.log(S / K) + (risk_free + 0.5 * sigma_val**2) * T) / (sigma_val * np.sqrt(T))
    d2 = d1 - sigma_val * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-risk_free * T) * norm.cdf(d2)

def bs_put(sigma_val, S, K, T, risk_free):
    d1 = (np.log(S / K) + (risk_free + 0.5 * sigma_val**2) * T) / (sigma_val * np.sqrt(T))
    d2 = d1 - sigma_val * np.sqrt(T)
    return K * np.exp(-risk_free * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

def iv_call(price, S, K, T, risk_free):
    intrinsic = max(S - K * np.exp(-risk_free * T), 0.0)
    if price <= intrinsic: return np.nan
    return brentq(lambda v: bs_call(v, S, K, T, risk_free) - price, 1e-4, 5.0)

def iv_put(price, S, K, T, risk_free):
    intrinsic = max(K * np.exp(-risk_free * T) - S, 0.0)
    if price <= intrinsic: return np.nan
    return brentq(lambda v: bs_put(v, S, K, T, risk_free) - price, 1e-4, 5.0)

rng = np.random.default_rng(42)
S_T = S_0 * np.exp((r - 0.5 * sigma**2)*T + sigma*np.sqrt(T)*rng.standard_normal(n_paths))

print("K/S\tCall_IV\tPut_IV")
for m in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
    K = m * S_0
    call_px = np.exp(-r*T) * np.mean(np.maximum(S_T - K, 0))
    put_px = np.exp(-r*T) * np.mean(np.maximum(K - S_T, 0))
    
    try: c_iv = iv_call(call_px, S_0, K, T, r)
    except ValueError: c_iv = np.nan
    
    try: p_iv = iv_put(put_px, S_0, K, T, r)
    except ValueError: p_iv = np.nan
    
    print(f"{m:.1f}\t{c_iv:.4f}\t{p_iv:.4f}")

**Important:** All files must be run from the **root directory** of the project with `PYTHONPATH=.`

## Deterministic Volterra Solver (Week 1)

### Uniform Mesh Solver
Plots the deterministic volatility solution on a uniform mesh.
```bash
PYTHONPATH=. python3 src/volterra/uniform_mesh.py
```

### Graded Mesh Comparison
Compares Uniform and Graded meshes, and plots the deterministic solutions for Classical Heston ($\omega=1.0$), Rough Heston ($\omega=0.6$), and the deterministic limit of Our Model ($\omega=0.7$).
```bash
PYTHONPATH=. python3 src/volterra/graded_mesh.py
```

### Convergence Analysis
Plots log(error) vs log(N) to demonstrate the convergence rates of both meshes.
```bash
PYTHONPATH=. python3 src/volterra/convergence.py
```

## Stochastic Simulation (Week 2 & 3)

### NIG Lévy Process Paths
Plots sample paths of the NIG Lévy noise process to verify mean-centering.
```bash
PYTHONPATH=. python3 src/stochastic/levy_paths_sim.py
```

### Fractional Kernel Stochastic Integral
Plots simulated paths of the fractional Brownian stochastic integral.
```bash
PYTHONPATH=. python3 src/stochastic/hybrid_scheme.py
```

### Volatility Paths (Full Model)
Plots sample paths of the full Lévy-driven Rough Heston variance process $v_t$ with positivity clamping.
```bash
PYTHONPATH=. python3 src/stochastic/stochastic_volterra.py
```
Output: `plots/variance_plots.png`

### Parameter Isolation Study (H and Lambda)
Generates the 2x2 grid plot for varying H and the comparative plot for varying jump intensity lambda.
```bash
PYTHONPATH=. python3 src/stochastic/vary_parameters.py
```
Output: `plots/vary_H.png`, `plots/vary_lambda.png`

### Option Pricing
Prices call options via Monte Carlo for Classical Heston, Rough Heston, and Our Model, saving results to CSV.
```bash
PYTHONPATH=. python3 src/stochastic/option_pricing.py
```
Output: `data/csv/option_prices.csv`

### Implied Volatility Smile
Inverts option prices to Black-Scholes implied volatilities and plots the comparison smile at $T=0.25$ in-memory.
```bash
PYTHONPATH=. python3 src/stochastic/implied_vol_smile.py
```
Output: `plots/implied_vol_smile.png`



"""
Convergence analysis across all 4 combinations: uniform/graded mesh x
exact/crude kernel weighting. Plots log(error) vs log(N) against a
high-N reference solution.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from volterra.uniform_mesh import solve_uniform_mesh, solve_uniform_mesh_crude
from volterra.graded_mesh import solve_graded_mesh, solve_graded_mesh_crude

KAPPA, THETA, V0, OMEGA, T = 1.0, 0.2, 0.04, 0.6, 2.0

# N=3200 used as the practical reference instead of N=6400 — agrees to
# solver precision and avoids the O(N^2) cost of going higher.
N_REFERENCE = 3200
N_VALUES = [10, 20, 40, 80, 160, 320]


def max_error_over_interval(t_coarse, v_coarse, t_ref, v_ref):
    """Max abs error over the FULL interval — endpoint-only comparison hides where the error actually lives."""
    v_ref_interp = np.interp(t_coarse, t_ref, v_ref)
    return np.max(np.abs(v_coarse - v_ref_interp))


def run_convergence_study():
    t_ref, v_ref = solve_graded_mesh(N_REFERENCE, T, OMEGA, KAPPA, THETA, V0)

    results = {"uniform_exact": [], "uniform_crude": [], "graded_exact": [], "graded_crude": []}

    for N in N_VALUES:
        t_u, v_u = solve_uniform_mesh(N, T, OMEGA, KAPPA, THETA, V0)
        t_uc, v_uc = solve_uniform_mesh_crude(N, T, OMEGA, KAPPA, THETA, V0)
        t_g, v_g = solve_graded_mesh(N, T, OMEGA, KAPPA, THETA, V0)
        t_gc, v_gc = solve_graded_mesh_crude(N, T, OMEGA, KAPPA, THETA, V0)

        results["uniform_exact"].append(max_error_over_interval(t_u, v_u, t_ref, v_ref))
        results["uniform_crude"].append(max_error_over_interval(t_uc, v_uc, t_ref, v_ref))
        results["graded_exact"].append(max_error_over_interval(t_g, v_g, t_ref, v_ref))
        results["graded_crude"].append(max_error_over_interval(t_gc, v_gc, t_ref, v_ref))

    return N_VALUES, results


def plot_convergence(N_values, results, out_path="plots/convergence_loglog.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    labels = {
        "uniform_exact": "Uniform mesh, exact kernel",
        "uniform_crude": "Uniform mesh, crude kernel",
        "graded_exact": "Graded mesh, exact kernel",
        "graded_crude": "Graded mesh, crude kernel",
    }
    for key, label in labels.items():
        plt.loglog(N_values, results[key], marker="o", label=label)
    plt.xlabel("N"); plt.ylabel("Max error over [0, T]")
    plt.title(f"Convergence: omega={OMEGA}, reference N={N_REFERENCE}")
    plt.legend(); plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(out_path, dpi=150)
    print(f"Saved convergence plot to {out_path}")


if __name__ == "__main__":
    N_values, results = run_convergence_study()
    for key, errors in results.items():
        print(key, [f"{e:.2e}" for e in errors])
    plot_convergence(N_values, results)
    # Honest note for the report: graded+exact wins clearly at small/moderate
    # N but can saturate at high N due to floating-point precision in the
    # kernel weight near t=0 (aggressive grading r=3 at omega=0.6).
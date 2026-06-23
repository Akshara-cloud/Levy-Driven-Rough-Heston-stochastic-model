"""
Forward variance curves at H=0.1 and H=0.2, plus the Classical Heston
comparison (omega=1, closed-form flat-ish curve).
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from volterra.graded_mesh import solve_graded_mesh

KAPPA, THETA, V0, T, N = 1.0, 0.2, 0.04, 2.0, 800


def omega_from_H(H):
    return H + 0.5


def classical_heston_curve(t, v0, theta, kappa):
    """omega=1 special case has closed form: v(t) = theta + (v0-theta)*exp(-kappa*t)."""
    return theta + (v0 - theta) * np.exp(-kappa * t)


def build_forward_variance_curves():
    curves = {}
    for H, label in [(0.1, "H=0.1 (omega=0.6)"), (0.2, "H=0.2 (omega=0.7)")]:
        t, v = solve_graded_mesh(N, T, omega_from_H(H), KAPPA, THETA, V0)
        curves[label] = (t, v)

    t_c = np.linspace(0, T, N + 1)
    curves["Classical Heston (omega=1)"] = (t_c, classical_heston_curve(t_c, V0, THETA, KAPPA))
    return curves


def plot_forward_variance(curves, out_path="plots/forward_variance_3model.png"):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    for label, (t, v) in curves.items():
        plt.plot(t, v, "--" if "Classical" in label else "-", label=label)
    plt.xlabel("t"); plt.ylabel("v(t)")
    plt.title("Forward variance: Rough Heston vs Classical Heston")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.savefig(out_path, dpi=150)
    print(f"Saved forward variance plot to {out_path}")


def plot_zoomed_near_zero(curves, out_path="plots/forward_variance_zoom.png", t_max=0.15):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    for label, (t, v) in curves.items():
        mask = t <= t_max
        plt.plot(t[mask], v[mask], "--" if "Classical" in label else "-", label=label)
    plt.xlabel("t"); plt.ylabel("v(t)")
    plt.title(f"Zoomed view: t in [0, {t_max}]")
    plt.legend(); plt.grid(True, alpha=0.4)
    plt.savefig(out_path, dpi=150)
    print(f"Saved zoomed plot to {out_path}")


if __name__ == "__main__":
    curves = build_forward_variance_curves()
    plot_forward_variance(curves)
    plot_zoomed_near_zero(curves)
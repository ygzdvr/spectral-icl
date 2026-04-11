"""Reusable SGD sweep runners for isotropic and RMT scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from dynamics import (
    simple_sgd_isotropic_dynamics,
    simple_sgd_isotropic_theory,
    simple_sgd_rmt_isotropic_dynamics,
    simple_sgd_rmt_isotropic_theory,
)

from .output_dir import OutputDir


def run_experiment(
    d: int,
    alpha: float,
    taus: list[float],
    eta: float,
    t_steps: int,
    out: OutputDir,
    device: str,
    dtype: torch.dtype,
) -> None:
    """Run one isotropic SGD experiment for a fixed alpha."""
    print(f"\n=== alpha = {alpha} ===")

    losses_emp = []
    losses_th = []
    for tau in taus:
        b = int(tau * d)
        p = int(alpha * d)
        print(f"  tau={tau:.2f}, B={b}, P={p}")
        loss_e = simple_sgd_isotropic_dynamics(d, b, p, eta, t_steps, device=device, dtype=dtype)
        loss_t = simple_sgd_isotropic_theory(tau, alpha, eta, t_steps, device=device, dtype=dtype)
        losses_emp.append(loss_e.cpu().numpy())
        losses_th.append(loss_t.cpu().numpy())

    plt.figure()
    sns.set_palette("rocket", n_colors=len(losses_emp))
    plt.title(rf"$\alpha = {alpha:.1f}$")
    for i, loss in enumerate(losses_emp):
        plt.semilogy(loss, label=rf"$\tau = {taus[i]:.2f}$")
    for loss in losses_th:
        plt.semilogy(loss, "--", color="black")
    plt.legend()

    tag = f"alpha_{alpha:.1f}".replace(".", "p")
    fig_name = f"sgd_isotropic_{tag}"
    plt.savefig(out.png(fig_name), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf(fig_name), dpi=200, bbox_inches="tight")
    print(f"  Saved plot to: {out.png(fig_name)}")

    payload: dict[str, np.ndarray] = {"taus": np.asarray(taus, dtype=np.float64)}
    for i, tau in enumerate(taus):
        payload[f"emp_tau_{tau:.2f}"] = losses_emp[i]
        payload[f"th_tau_{tau:.2f}"] = losses_th[i]
    np.savez(out.numpy(fig_name), **payload)
    print(f"  Saved losses to: {out.numpy(fig_name)}")


def run_sweep(
    name: str,
    sweep_var: str,
    sweep_vals: list[float],
    d: int,
    tau_fixed: float,
    alpha_fixed: float,
    kappa_fixed: float,
    eta: float,
    t_steps: int,
    out: OutputDir,
    device: str,
    dtype: torch.dtype,
    use_semilogy: bool = False,
) -> None:
    """Evaluate one parameter sweep for RMT isotropic SGD dynamics."""
    print(f"\n=== {name} ===")
    losses_emp: list[np.ndarray] = []
    losses_th: list[np.ndarray] = []

    for val in sweep_vals:
        tau = tau_fixed
        alpha = alpha_fixed
        kappa = kappa_fixed

        if sweep_var == "tau":
            tau = val
        elif sweep_var == "alpha":
            alpha = val
        elif sweep_var == "kappa":
            kappa = val

        b = int(tau * d)
        k = int(kappa * d)
        p = int(alpha * d)
        print(f"  {sweep_var}={val:.2f}, B={b}, K={k}, P={p}")

        loss_e = simple_sgd_rmt_isotropic_dynamics(d, b, k, p, eta, t_steps, device=device, dtype=dtype)
        loss_t = simple_sgd_rmt_isotropic_theory(tau, alpha, kappa, eta, t_steps, device=device, dtype=dtype)
        losses_emp.append(loss_e.cpu().numpy())
        losses_th.append(loss_t.cpu().numpy())

    plt.figure()
    sns.set_palette("rocket", n_colors=len(sweep_vals))
    plot_fn = plt.semilogy if use_semilogy else plt.plot
    greek = {"tau": r"\tau", "alpha": r"\alpha", "kappa": r"\kappa"}[sweep_var]

    for i, val in enumerate(sweep_vals):
        plot_fn(losses_emp[i], label=rf"${greek} = {val:.1f}$")
    for loss in losses_th:
        plot_fn(loss, "--", color="black")
    plot_fn([], [], "--", color="black", label="Theory")
    plt.legend()
    plt.xlabel(r"$t$", fontsize=20)
    plt.ylabel(r"$\mathcal{L}(t)$", fontsize=20)

    safe_name = name.replace(" ", "_").replace("=", "").replace(",", "")
    fig_name = f"sgd_rmt_{safe_name}"
    plt.savefig(out.png(fig_name), dpi=200, bbox_inches="tight")
    plt.savefig(out.pdf(fig_name), dpi=200, bbox_inches="tight")
    print(f"  Saved plot to: {out.png(fig_name)}")

    payload: dict[str, np.ndarray] = {"sweep_vals": np.asarray(sweep_vals, dtype=np.float64)}
    for i, val in enumerate(sweep_vals):
        payload[f"emp_{val:.2f}"] = losses_emp[i]
        payload[f"th_{val:.2f}"] = losses_th[i]
    np.savez(out.numpy(fig_name), **payload)
    print(f"  Saved losses to: {out.numpy(fig_name)}")

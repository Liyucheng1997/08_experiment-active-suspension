from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.controllers.risk_weights import (
    RiskWeightParams,
    q_c,
    q_p,
    q_p_local,
    sigma_rho,
)
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def run_phase_4_2(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    config_path = Path(config_path)
    params = RiskWeightParams(
        rho_th=0.75,
        k_rho=25.0,
        kappa_rho=5.0,
        q_c_min=0.1,
        q_c_max=1.0,
        q_p_min=0.0,
        q_p_max=2.0,
    )
    logger = RunLogger.create(
        results_root, phase="phase-4.2", step="risk-weights", descriptor="sigma-qc-qp"
    )
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    # --- σ_ρ curve and q_c / q_p schedules ---
    rho_grid = np.linspace(0.0, 1.5, 401)
    sigma = sigma_rho(rho_grid, params.rho_th, params.k_rho)
    q_c_vals = q_c(sigma, params.q_c_min, params.q_c_max)
    q_p_vals = q_p(sigma, params.q_p_min, params.q_p_max)

    fig, ax = plot_timeseries(
        rho_grid,
        sigma,
        labels=[rf"$\sigma(\rho)$, $k_\rho$={params.k_rho:.0f}"],
        ylabel=r"$\sigma$",
        title=rf"Risk scheduler $\sigma(\rho_{{max}})$ — threshold $\rho_{{th}}$={params.rho_th}",
    )
    ax.set_xlabel(r"$\rho_{max}$")
    ax.axvline(params.rho_th, color="0.3", linestyle="--", linewidth=1.0,
               label=rf"$\rho_{{th}}$={params.rho_th}")
    ax.axhline(0.5, color="0.6", linestyle=":", linewidth=1.0, label=r"$\sigma$=0.5")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "sigma_vs_rho.png")

    fig, ax = plot_timeseries(
        rho_grid,
        np.column_stack([q_c_vals, q_p_vals]),
        labels=[r"$q_c(\sigma)$ — comfort", r"$q_p(\sigma)$ — protection"],
        ylabel="weight",
        title="Global weight schedules: comfort fades, protection rises with σ",
    )
    ax.set_xlabel(r"$\rho_{max}$")
    ax.axvline(params.rho_th, color="0.3", linestyle="--", linewidth=1.0,
               label=rf"$\rho_{{th}}$={params.rho_th}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "qc_qp_vs_rho.png")

    # --- q_p_local: per-wheel amplification at a chosen worst-wheel value ---
    # Sweep ρ_ij of one wheel from 0 to 1.3 while keeping σ tied to ρ_max=that wheel
    # (worst wheel). Show how κ_ρ controls the local sharpening.
    rho_ij_grid = np.linspace(0.0, 1.3, 261)
    sigma_grid = sigma_rho(rho_ij_grid, params.rho_th, params.k_rho)
    kappa_family = [0.0, 2.0, 5.0, 10.0]
    curves = np.column_stack(
        [
            q_p_local(
                sigma_grid, rho_ij_grid, params.rho_th, k,
                params.q_p_min, params.q_p_max,
            )
            for k in kappa_family
        ]
    )
    fig, ax = plot_timeseries(
        rho_ij_grid,
        curves,
        labels=[rf"$\kappa_\rho$={k:g}" for k in kappa_family],
        ylabel=r"$q_{p, ij}$",
        title=(
            r"Per-wheel protection weight (κ_ρ=0 ⇒ uniform global q_p; "
            r"κ_ρ>0 sharpens on overshoot wheel)"
        ),
    )
    ax.set_xlabel(r"$\rho_{ij}$ (one wheel; σ tracks the worst)")
    ax.axvline(params.rho_th, color="0.3", linestyle="--", linewidth=1.0)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "qp_local_vs_rho_ij.png")

    # Spot checks logged as metrics
    sigma_at_th = sigma_rho(params.rho_th, params.rho_th, params.k_rho)
    logger.log_kv("sigma_at_rho_th", float(sigma_at_th))
    logger.log_kv("q_c_at_sigma_0", float(q_c(0.0, params.q_c_min, params.q_c_max)))
    logger.log_kv("q_c_at_sigma_1", float(q_c(1.0, params.q_c_min, params.q_c_max)))
    logger.log_kv("q_p_at_sigma_0", float(q_p(0.0, params.q_p_min, params.q_p_max)))
    logger.log_kv("q_p_at_sigma_1", float(q_p(1.0, params.q_p_min, params.q_p_max)))
    logger.log_kv("kappa_rho", params.kappa_rho)
    # Verify ablation A3 hook: κ_ρ=0 collapses local to uniform global.
    test_rho = np.array([0.4, 0.85, 1.05, 0.6])
    uniform = q_p_local(0.7, test_rho, params.rho_th, kappa_rho=0.0,
                        q_p_min=params.q_p_min, q_p_max=params.q_p_max)
    logger.log_kv("kappa_zero_uniform_std", float(np.std(uniform)))

    np.savez(
        logger.run_dir / "raw.npz",
        rho_grid=rho_grid,
        sigma=sigma,
        q_c=q_c_vals,
        q_p=q_p_vals,
        rho_ij_grid=rho_ij_grid,
        qp_local_curves=curves,
        kappa_family=np.array(kappa_family),
    )
    logger.log("Phase 4.2 risk-weight demo completed.")
    return logger.run_dir


def main() -> None:
    print(run_phase_4_2())


if __name__ == "__main__":
    main()

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.metrics.tire import (
    F_Z_MIN_DEFAULT,
    f_z_required,
    rho,
)
from risk_aware_active_suspension.utils.config import from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def run_phase_4_1(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
    mu: float = 0.9,
    rho_safe: float = 0.85,
) -> Path:
    config_path = Path(config_path)
    logger = RunLogger.create(
        results_root, phase="phase-4.1", step="tire-metrics", descriptor="rho-and-required-fz"
    )
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    # Sanity numbers
    f_c_zero_required = float(f_z_required(0.0, mu, rho_safe))
    rho_at_inverse = float(rho(800.0, 0.0, f_z_required(800.0, mu, rho_safe), mu))
    logger.log_kv("f_z_required_at_zero_demand_n", f_c_zero_required)
    logger.log_kv("rho_at_inverse_check", rho_at_inverse)
    logger.log_kv("f_z_min_floor_n", F_Z_MIN_DEFAULT)

    # ρ vs F_z curves for several horizontal-demand levels. Start the grid at
    # 500 N (well below realistic per-corner F_z ~ 3.7 kN) — going to F_z=0
    # makes ρ blow up against the F_z_min floor and crushes the useful range.
    f_z_grid = np.linspace(500.0, 8000.0, 401)
    demand_levels = [500.0, 1500.0, 3000.0, 5000.0]
    curves = np.column_stack(
        [rho(np.full_like(f_z_grid, fc), 0.0, f_z_grid, mu) for fc in demand_levels]
    )
    fig, ax = plot_timeseries(
        f_z_grid,
        curves,
        labels=[f"F_c={fc:.0f} N" for fc in demand_levels],
        ylabel=r"$\rho$ (friction utilization)",
        title=rf"$\rho$ vs F_z at mu={mu} — each curve is a fixed horizontal demand",
    )
    ax.set_xlabel("F_z (per-corner tire normal load) [N]")
    ax.set_ylim(0.0, 2.0)
    ax.axhline(1.0, color="0.3", linestyle="--", linewidth=1.0, label=r"$\rho$=1 (slip)")
    ax.axhline(rho_safe, color="0.5", linestyle=":", linewidth=1.0,
               label=rf"$\rho_{{safe}}$={rho_safe}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "rho_vs_fz.png")

    # F_z_required vs horizontal demand for several mu values, at the chosen
    # safety margin. Lower mu (wet/snow/ice) demands a steeper F_z to stay
    # below rho_safe — when the per-corner static load can't supply it,
    # the active controller has to redistribute load via the suspension.
    f_c_grid = np.linspace(0.0, 6000.0, 301)
    mu_family = [0.9, 0.7, 0.5, 0.3]
    fz_required_curves = np.column_stack(
        [f_z_required(f_c_grid, m, rho_safe) for m in mu_family]
    )
    static_corner_load = from_yaml(config_path).m * 9.81 / 4.0  # N
    fig, ax = plot_timeseries(
        f_c_grid,
        fz_required_curves,
        labels=[rf"$\mu$={m}" for m in mu_family],
        ylabel="F_z_required [N]",
        title=rf"Required normal load vs horizontal demand ($\rho_{{safe}}$={rho_safe})",
    )
    ax.set_xlabel("F_c (combined horizontal demand) [N]")
    ax.set_ylim(0.0, 20000.0)
    ax.axhline(
        static_corner_load,
        color="0.3",
        linestyle="--",
        linewidth=1.0,
        label=f"per-corner static load ({static_corner_load:.0f} N)",
    )
    ax.axhline(
        2.0 * static_corner_load,
        color="0.5",
        linestyle=":",
        linewidth=1.0,
        label="2x static (load transfer ceiling)",
    )
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "fz_required_vs_fc.png")
    # Keep the single-mu curve in raw.npz for downstream consumers.
    fz_req = fz_required_curves[:, mu_family.index(0.9)] if 0.9 in mu_family else fz_required_curves[:, 0]

    np.savez(
        logger.run_dir / "raw.npz",
        f_z_grid=f_z_grid,
        rho_curves=curves,
        f_c_grid=f_c_grid,
        f_z_required=fz_req,
        demand_levels=np.array(demand_levels),
        mu=mu,
        rho_safe=rho_safe,
    )

    assert f_c_zero_required == 0.0, "F_z_required at zero demand must be exactly zero."
    assert abs(rho_at_inverse - rho_safe) < 1e-9, "rho-vs-F_z_required inverse identity broken."
    logger.log("Phase 4.1 tire metrics demo completed; acceptance passed.")
    return logger.run_dir


def main() -> None:
    print(run_phase_4_1())


if __name__ == "__main__":
    main()

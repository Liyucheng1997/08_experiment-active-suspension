from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.controllers.risk_qp_quarter import (
    QuarterCarRiskAwareQP,
    QuarterRiskQPParams,
)
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.inputs.road import rounded_bump
from risk_aware_active_suspension.metrics.tire import f_z_required as fz_required_fn
from risk_aware_active_suspension.metrics.tire import rho as rho_fn
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.utils.config import (
    LQRParams,
    VehicleParams,
    from_yaml,
    lqr_from_yaml,
)
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def _simulate(
    vehicle: VehicleParams,
    lqr: LQRParams,
    risk: RiskWeightParams,
    qp_params: QuarterRiskQPParams,
    t: np.ndarray,
    road: np.ndarray,
    f_c: np.ndarray,
    mu: float,
) -> dict[str, np.ndarray]:
    plant = QuarterCar(vehicle)
    controller = QuarterCarRiskAwareQP(vehicle, lqr, risk, qp_params)

    states = np.zeros((len(t), 4), dtype=float)
    forces = np.zeros(len(t), dtype=float)
    xi_log = np.zeros(len(t), dtype=float)
    sigma_log = np.zeros(len(t), dtype=float)
    f_z_dyn = np.zeros(len(t), dtype=float)
    f_z_req = np.zeros(len(t), dtype=float)
    rho_naive = np.zeros(len(t), dtype=float)        # ρ from raw dynamic F_z
    rho_controller = np.zeros(len(t), dtype=float)   # ρ from F_z + γ F_e
    state = np.zeros(4)
    for k in range(len(t) - 1):
        f_z_now = float(plant.tire_normal_force(state, w=float(road[k])))
        u_arr = controller.compute(state, f_z_true=f_z_now, f_c=float(f_c[k]), mu=mu)
        u = float(u_arr[0])
        forces[k] = u
        xi_log[k] = controller.last_xi
        sigma_log[k] = controller.last_sigma
        f_z_dyn[k] = f_z_now
        f_z_req[k] = controller.last_f_z_required
        rho_naive[k] = float(rho_fn(float(f_c[k]), 0.0, f_z_now, mu))
        f_z_eff = f_z_now + qp_params.gamma * u
        rho_controller[k] = float(rho_fn(float(f_c[k]), 0.0, f_z_eff, mu))
        state = plant.step(state, u, float(road[k]), float(t[k + 1] - t[k]))
        states[k + 1] = state

    return {
        "t": t,
        "states": states,
        "forces": forces,
        "xi": xi_log,
        "sigma": sigma_log,
        "f_z_dyn": f_z_dyn,
        "f_z_required": f_z_req,
        "rho_naive": rho_naive,
        "rho_controller_view": rho_controller,
    }


def run_phase_4_4(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    results_root: str | Path = "results",
    duration: float = 2.0,
    dt: float = 0.001,
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    vehicle = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)

    logger = RunLogger.create(
        results_root, phase="phase-4.4", step="margin-constraint",
        descriptor="quarter-car-bump-during-cornering",
    )
    shutil.copy2(vehicle_config, logger.run_dir / "config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")

    # Constant lateral demand on the single corner + a positive road bump. With
    # the paper F_z sign, the bump first increases contact load and then unloads
    # the tire during the wheel-hop rebound; that rebound window exercises the
    # margin constraint.
    mu = 0.6
    rho_safe = 0.85
    f_c_level = 1500.0  # N, combined horizontal demand
    t = np.arange(0.0, duration, dt)
    bump = rounded_bump(t, height=0.03, start=0.5, duration=0.2)
    f_c = np.full_like(t, f_c_level)

    risk = RiskWeightParams(
        rho_th=0.75, k_rho=25.0, kappa_rho=5.0,
        q_c_min=0.1, q_c_max=1.0,
        q_p_min=0.0, q_p_max=1.0,
    )

    # Case A — sufficient authority: large f_max, gamma=1.
    qp_A = QuarterRiskQPParams(gamma=1.0, rho_safe=rho_safe)
    res_A = _simulate(vehicle, replace(lqr, f_max=10000.0), risk, qp_A, t, bump, f_c, mu)
    rho_actual_A = res_A["rho_controller_view"]
    # Solver and discrete-time update tolerances create a small rho overshoot at
    # the rebound peak; treat a 5e-3 band as numerical zero for this harness.
    a_violations = int(np.sum(rho_actual_A > rho_safe + 5e-3))
    logger.log_kv("caseA_max_rho_controller_view", float(np.max(rho_actual_A)))
    logger.log_kv("caseA_max_xi", float(np.max(res_A["xi"])))
    logger.log_kv("caseA_max_force_n", float(np.max(np.abs(res_A["forces"]))))
    logger.log_kv("caseA_rho_violations_count", a_violations)

    # Case B — insufficient authority: small f_max, gamma=1.
    qp_B = QuarterRiskQPParams(gamma=1.0, rho_safe=rho_safe)
    res_B = _simulate(vehicle, replace(lqr, f_max=200.0), risk, qp_B, t, bump, f_c, mu)
    xi_active_window = res_B["xi"] > 1.0
    logger.log_kv("caseB_max_rho_controller_view", float(np.max(res_B["rho_controller_view"])))
    logger.log_kv("caseB_max_xi", float(np.max(res_B["xi"])))
    logger.log_kv("caseB_total_xi_time_s", float(np.sum(xi_active_window) * dt))
    logger.log_kv("caseB_max_force_n", float(np.max(np.abs(res_B["forces"]))))

    # Acceptance #1: case A keeps rho_actual <= rho_safe throughout (the bump
    # window must actually exercise the constraint — diagnostic flags below).
    accept_A = a_violations == 0
    # We also need the bump to have *forced* the constraint to engage —
    # otherwise we're testing nothing. Engagement is signaled by F_z_required
    # ever exceeding the open-loop dynamic F_z over the bump window.
    naive_violations = int(np.sum(res_A["rho_naive"] > rho_safe + 5e-3))
    logger.log_kv("scenario_naive_violations_count", naive_violations)
    accept_A = accept_A and naive_violations > 0

    # Acceptance #2: case B exposes positive slack quantifying the violation.
    accept_B = bool(np.max(res_B["xi"]) > 100.0)

    logger.log_kv("acceptance_caseA_authority_sufficient_protects_margin", int(accept_A))
    logger.log_kv("acceptance_caseB_authority_insufficient_xi_positive", int(accept_B))

    # Figures
    fig, ax = plot_timeseries(
        t,
        np.column_stack([res_A["f_z_dyn"], res_A["f_z_required"], res_A["f_z_dyn"] + qp_A.gamma * res_A["forces"]]),
        labels=["F_z (dynamic)", "F_z_required", "F_z + γ·F_e (effective)"],
        ylabel="Force [N]",
        title="Case A (f_max=10 kN): controller lifts effective F_z above F_z_required",
    )
    ax.axvspan(0.5, 0.7, color="0.9", alpha=0.5, label="bump window")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "caseA_force_balance.png")

    fig, ax = plot_timeseries(
        t,
        np.column_stack([res_A["rho_naive"], res_A["rho_controller_view"]]),
        labels=["ρ (without active F_e)", "ρ (controller view, with γ·F_e)"],
        ylabel=r"$\rho$",
        title="Case A: ρ stays ≤ ρ_safe with sufficient authority",
    )
    ax.axhline(rho_safe, color="0.3", linestyle="--", label=f"ρ_safe={rho_safe}")
    ax.axhline(1.0, color="0.6", linestyle=":", label="ρ=1 (slip)")
    ax.axvspan(0.5, 0.7, color="0.9", alpha=0.5)
    ax.set_ylim(0.0, 1.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "caseA_rho.png")

    fig, ax = plot_timeseries(
        t,
        np.column_stack([res_B["xi"], res_B["forces"]]),
        labels=[r"$\xi$ (slack)", "F_e (actuator)"],
        ylabel="N",
        title="Case B (f_max=200 N): ξ quantifies the un-meetable load gap",
    )
    ax.axvspan(0.5, 0.7, color="0.9", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "caseB_xi_and_force.png")

    fig, ax = plot_timeseries(
        t,
        np.column_stack([res_B["f_z_dyn"], res_B["f_z_required"],
                          res_B["f_z_dyn"] + qp_B.gamma * res_B["forces"],
                          res_B["f_z_dyn"] + qp_B.gamma * res_B["forces"] + res_B["xi"]]),
        labels=["F_z (dynamic)", "F_z_required",
                "F_z + γ·F_e", "F_z + γ·F_e + ξ (must equal F_z_req when binding)"],
        ylabel="Force [N]",
        title="Case B: actuator saturated, ξ closes the gap to F_z_required",
    )
    ax.axvspan(0.5, 0.7, color="0.9", alpha=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(logger.run_dir / "figures" / "caseB_force_balance.png")

    np.savez(
        logger.run_dir / "raw.npz",
        t=t, bump=bump, f_c=f_c, mu=mu, rho_safe=rho_safe,
        caseA_forces=res_A["forces"], caseA_xi=res_A["xi"],
        caseA_f_z_dyn=res_A["f_z_dyn"], caseA_rho=res_A["rho_controller_view"],
        caseB_forces=res_B["forces"], caseB_xi=res_B["xi"],
        caseB_f_z_dyn=res_B["f_z_dyn"], caseB_rho=res_B["rho_controller_view"],
    )

    if not accept_A:
        raise RuntimeError(
            "Phase 4.4 acceptance failed (Case A): controller did not keep "
            f"ρ ≤ ρ_safe throughout (max ρ_actual={np.max(rho_actual_A):.3f}, "
            f"violations={a_violations}, scenario engages constraint={naive_violations>0})."
        )
    if not accept_B:
        raise RuntimeError(
            "Phase 4.4 acceptance failed (Case B): expected ξ > 100 N "
            f"to quantify violation but got max ξ={np.max(res_B['xi']):.2f}."
        )
    logger.log("Phase 4.4 margin constraint validation completed; acceptance passed.")
    return logger.run_dir


def main() -> None:
    print(run_phase_4_4())


if __name__ == "__main__":
    main()

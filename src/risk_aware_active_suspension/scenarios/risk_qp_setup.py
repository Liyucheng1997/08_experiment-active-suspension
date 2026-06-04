from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_qp import (
    FullCarRiskAwareQP,
    RiskQPParams,
)
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.inputs.road import iso8608
from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.scenarios.baseline_comparison import lane_change_a_y
from risk_aware_active_suspension.scenarios.rough_road_validation import (
    _straight_two_track_road,
)
from risk_aware_active_suspension.utils.config import (
    LQRParams,
    VehicleParams,
    from_yaml,
    lqr_from_yaml,
)
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def compare_against_comfort_qp(
    vehicle: VehicleParams, lqr: LQRParams, n_samples: int = 200, seed: int = 0
) -> dict[str, float]:
    """Acceptance #1: q_p ≡ 0 with no margin constraint -> matches Phase 3.4."""
    risk = RiskWeightParams(
        rho_th=0.75, k_rho=25.0, kappa_rho=0.0,
        q_c_min=1.0, q_c_max=1.0,
        q_p_min=0.0, q_p_max=0.0,
    )
    rqp = FullCarRiskAwareQP(
        vehicle, lqr, risk,
        RiskQPParams(enforce_rate=False, r_u_factor=0.0, r_du_factor=0.0),
    )
    cqp = FullCarComfortQP(vehicle, lqr)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_samples):
        x = rng.standard_normal(14) * np.array(
            [0.05, 0.5, 0.01, 0.1, 0.01, 0.1, *([0.005, 0.5] * 4)]
        )
        diffs.append(float(np.max(np.abs(rqp.compute(x) - cqp.compute(x)))))
    return {
        "max_abs_diff_N": float(max(diffs)),
        "median_abs_diff_N": float(np.median(diffs)),
    }


def check_xi_zero_without_margin(
    vehicle: VehicleParams, lqr: LQRParams, n_samples: int = 200, seed: int = 1
) -> dict[str, float]:
    """Acceptance #2: xi* = 0 when no margin constraint is active."""
    risk = RiskWeightParams(
        rho_th=0.75, k_rho=25.0, kappa_rho=5.0,
        q_c_min=0.1, q_c_max=1.0,
        q_p_min=0.0, q_p_max=100.0,
    )
    rqp = FullCarRiskAwareQP(vehicle, lqr, risk)
    rng = np.random.default_rng(seed)
    max_xi = 0.0
    max_sigma = 0.0
    for _ in range(n_samples):
        x = rng.standard_normal(14) * 0.1
        rho = rng.uniform(0.0, 1.2, 4)
        rqp.compute(x, rho_ij=rho)
        max_xi = max(max_xi, float(np.max(np.abs(rqp.last_xi))))
        max_sigma = max(max_sigma, rqp.last_sigma)
    return {"max_xi": max_xi, "max_sigma": max_sigma}


def run_phase_4_3(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    results_root: str | Path = "results",
    duration: float = 12.0,
    dt: float = 0.002,
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    params = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)
    plant = FullCar(params)

    logger = RunLogger.create(
        results_root, phase="phase-4.3", step="risk-qp",
        descriptor="cost-reparam-with-slack",
    )
    shutil.copy2(vehicle_config, logger.run_dir / "config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")

    # ---- Acceptance #1 ----
    match = compare_against_comfort_qp(params, lqr)
    logger.log_kv("acceptance_match_to_comfort_max_abs_diff_n", match["max_abs_diff_N"])
    logger.log_kv("acceptance_match_to_comfort_median_abs_diff_n", match["median_abs_diff_N"])
    match_ok = match["max_abs_diff_N"] < 1e-2
    logger.log_kv("acceptance_qp_eq_comfort_qp_when_qp_zero", int(match_ok))

    # ---- Acceptance #2 ----
    xi_chk = check_xi_zero_without_margin(params, lqr)
    logger.log_kv("acceptance_max_xi_without_margin_constraint", xi_chk["max_xi"])
    logger.log_kv("max_sigma_observed", xi_chk["max_sigma"])
    xi_ok = xi_chk["max_xi"] < 1e-6
    logger.log_kv("acceptance_xi_zero_without_margin", int(xi_ok))

    # ---- Closed-loop demo: lane-change on class B, σ stays low (proxy ρ_ij from a_y) ----
    t = np.arange(0.0, duration, dt)
    roads = _straight_two_track_road(plant, t=t, dt=dt, v_x=80.0 / 3.6, road_class="B")
    a_y = lane_change_a_y(t, peak_ay=3.0, start=2.0, duration=4.0)
    body_acc = np.column_stack([np.zeros_like(t), a_y])

    # Synthetic ρ_ij proxy: scale a_y to populate the outer wheels' utilization
    # (rough placeholder until 4.4/4.5 wire up the real F_z and F_c).
    rho_history = np.zeros((len(t), 4), dtype=float)
    # Left wheels (FL=0, RL=2) load up under positive a_y; rho roughly tracks |a_y|.
    rho_proxy = 0.85 / 3.0 * np.abs(a_y)  # peaks ~0.85 at peak a_y
    rho_history[:, 0] = rho_proxy
    rho_history[:, 2] = rho_proxy
    rho_history[:, 1] = 0.3 * rho_proxy
    rho_history[:, 3] = 0.3 * rho_proxy

    risk = RiskWeightParams(
        rho_th=0.75, k_rho=25.0, kappa_rho=5.0,
        q_c_min=0.2, q_c_max=1.0,
        q_p_min=0.0, q_p_max=50.0,
    )
    # The default df_max=20 kN/s rate limit binds heavily at lqr.T_s=5 ms
    # (only 100 N change per call). For this scaffolding demo we relax the rate
    # to keep the trajectory comparable to LQR/comfort-QP — Phase 4.7 retunes
    # the realistic hardware rate jointly with the observer-error tightening.
    rqp = FullCarRiskAwareQP(
        params, lqr, risk, RiskQPParams(df_max=1.0e6, enforce_rate=True)
    )
    states = np.zeros((len(t), 14), dtype=float)
    forces = np.zeros((len(t), 4), dtype=float)
    sigma_log = np.zeros(len(t), dtype=float)
    qc_log = np.zeros(len(t), dtype=float)
    xi_log = np.zeros((len(t), 4), dtype=float)
    timings = np.zeros(len(t), dtype=float)
    state = np.zeros(14)
    for k in range(len(t) - 1):
        u = rqp.compute(state, rho_ij=rho_history[k])
        forces[k] = u
        sigma_log[k] = rqp.last_sigma
        qc_log[k] = rqp.last_q_c
        xi_log[k] = rqp.last_xi
        timings[k] = rqp.last_solve_time_s
        state = plant.step(state, roads[k], float(t[k + 1] - t[k]), u=u, body_acc=body_acc[k])
        states[k + 1] = state

    body_accel = np.array(
        [
            plant.body_accelerations(s, w, u=f, body_acc=ba)
            for s, w, f, ba in zip(states, roads, forces, body_acc)
        ]
    )
    logger.log_kv("closed_loop_heave_rms_m_s2", float(rms(body_accel[:, 0])))
    logger.log_kv("closed_loop_roll_rms_rad_s2", float(rms(body_accel[:, 1])))
    logger.log_kv("closed_loop_pitch_rms_rad_s2", float(rms(body_accel[:, 2])))
    logger.log_kv("closed_loop_max_force_n", float(np.max(np.abs(forces))))
    logger.log_kv("closed_loop_max_xi_observed", float(np.max(np.abs(xi_log))))
    logger.log_kv("closed_loop_solve_p95_ms", float(np.percentile(timings[:-1], 95) * 1000.0))

    # Figures
    win = slice(0, int(min(duration, 8.0) / dt))
    plot_timeseries(
        t[win],
        np.column_stack([sigma_log[win], qc_log[win]]),
        labels=[r"$\sigma$", r"$q_c$"],
        ylabel="weights",
        title="Risk-QP weights along the lane-change trajectory",
        save_path=logger.run_dir / "figures" / "sigma_qc_trajectory.png",
    )
    plot_timeseries(
        t[win],
        rho_history[win],
        labels=list(CORNER_NAMES),
        ylabel=r"$\rho_{ij}$ (proxy)",
        title=r"Per-wheel $\rho$ proxy fed to the risk-QP (placeholder until Phase 4.4)",
        save_path=logger.run_dir / "figures" / "rho_proxy.png",
    )
    plot_timeseries(
        t[win],
        forces[win],
        labels=list(CORNER_NAMES),
        ylabel="Actuator force [N]",
        title="Risk-QP actuator forces under class-B + lane change",
        save_path=logger.run_dir / "figures" / "actuator_forces.png",
    )
    plot_timeseries(
        t[win],
        xi_log[win],
        labels=list(CORNER_NAMES),
        ylabel=r"$\xi_{ij}$",
        title=r"Slack $\xi_{ij}$ (must be zero without margin constraint)",
        save_path=logger.run_dir / "figures" / "xi_trajectory.png",
    )

    np.savez(
        logger.run_dir / "raw.npz",
        t=t, states=states, forces=forces, sigma=sigma_log, q_c=qc_log,
        xi=xi_log, rho_history=rho_history, body_accel=body_accel, solve_times_s=timings,
    )

    if not match_ok:
        raise RuntimeError(
            f"Phase 4.3 acceptance failed: risk-QP does not collapse to comfort-QP "
            f"with q_p=0 (max diff {match['max_abs_diff_N']:.3e} N)."
        )
    if not xi_ok:
        raise RuntimeError(
            f"Phase 4.3 acceptance failed: xi* not zero without margin constraint "
            f"(max |xi| = {xi_chk['max_xi']:.3e})."
        )
    logger.log("Phase 4.3 risk-aware QP scaffolding completed; acceptance passed.")
    return logger.run_dir


def main() -> None:
    print(run_phase_4_3())


if __name__ == "__main__":
    main()

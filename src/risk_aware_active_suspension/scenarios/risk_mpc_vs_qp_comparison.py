from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import (
    RESIDUAL_DECAY,
    FullCarRiskMPC,
)
from risk_aware_active_suspension.controllers.risk_qp import FullCarRiskAwareQP, RiskQPParams
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.metrics.signals import rmse
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.scenarios.risk_mpc_residual_prediction import (
    _moving_average,
    run_mpc_residual_closed_loop,
)
from risk_aware_active_suspension.scenarios.risk_qp_fullcar_ground_truth import (
    _build_scenario,
    _closed_loop,
)
from risk_aware_active_suspension.scenarios.risk_qp_sto_closed_loop import closed_loop_with_sto
from risk_aware_active_suspension.utils.config import from_yaml, lqr_from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


def run_phase_5_5(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    observer_config: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
    duration: float = 3.5,
    dt: float = 0.001,
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    observer_config = Path(observer_config)
    vehicle = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)
    observer_params = observer_from_yaml(observer_config)
    plant = FullCar(vehicle)

    logger = RunLogger.create(
        results_root,
        phase="phase-5.5",
        step="risk-mpc",
        descriptor="mpc-vs-one-step-qp",
    )
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")
    shutil.copy2(observer_config, logger.run_dir / "observer_config.yaml")

    mu = 0.7
    rho_safe = 0.85
    rho_th = 0.75
    scenario = _build_scenario(
        plant,
        duration=duration,
        dt=dt,
        peak_ay=4.0,
        bump_corner=1,
        bump_height=0.02,
        bump_start=1.8,
        bump_duration=0.03,
        mu=mu,
    )
    risk = RiskWeightParams(
        rho_th=rho_th,
        k_rho=25.0,
        kappa_rho=0.0,
        q_c_min=1.0,
        q_c_max=1.0,
        q_p_min=0.0,
        q_p_max=0.003,
    )
    qp_params = RiskQPParams(
        rho_safe=rho_safe,
        gamma_n_horizon=1,
        enforce_rate=False,
        r_du_factor=1.0e-5,
        osqp_eps_abs=1.0e-5,
        osqp_eps_rel=1.0e-5,
        osqp_max_iter=100000,
    )

    comfort = _closed_loop(FullCarComfortQP(vehicle, lqr), plant, scenario, is_risk=False)
    one_step_qp = closed_loop_with_sto(
        FullCarRiskAwareQP(vehicle, lqr, risk, qp_params),
        plant,
        scenario,
        observer_params,
        fz_tightening_n=observer_params.delta_fz_err,
    )
    mpc_controller = FullCarRiskMPC(
        vehicle,
        lqr,
        risk,
        qp_params,
        horizon=10,
        warm_start=False,
        reuse_solver=True,
    )
    mpc = run_mpc_residual_closed_loop(
        mpc_controller,
        plant,
        scenario["t"],
        scenario["roads"],
        scenario["a_y"],
        scenario["f_c"],
        mu,
        observer_params,
        prediction_mode=RESIDUAL_DECAY,
        alpha=0.95,
        fz_tightening_n=observer_params.delta_fz_err,
    )

    t = scenario["t"]
    window = (t >= 1.7) & (t <= 2.1)
    metrics = _compute_metrics(window, comfort, one_step_qp, mpc)
    mpc_extra_reduction = (
        metrics["qp_peak_rho"] - metrics["mpc_peak_rho"]
    ) / max(metrics["qp_peak_rho"], 1.0e-9)
    mpc_vs_comfort_reduction = (
        metrics["comfort_peak_rho"] - metrics["mpc_peak_rho"]
    ) / max(metrics["comfort_peak_rho"], 1.0e-9)
    mpc_heave_degradation = (
        metrics["mpc_heave_rms"] - metrics["comfort_heave_rms"]
    ) / max(metrics["comfort_heave_rms"], 1.0e-9)

    logger.log_kv("mu", mu)
    logger.log_kv("rho_safe", rho_safe)
    logger.log_kv("rho_th", rho_th)
    logger.log_kv("mpc_horizon", 10)
    logger.log_kv("mpc_warm_start", 0)
    logger.log_kv("q_p_max", risk.q_p_max)
    logger.log_kv("comfort_peak_rho_window", metrics["comfort_peak_rho"])
    logger.log_kv("qp_peak_rho_window", metrics["qp_peak_rho"])
    logger.log_kv("mpc_peak_rho_window", metrics["mpc_peak_rho"])
    logger.log_kv("mpc_extra_rho_reduction_vs_qp_ratio", mpc_extra_reduction)
    logger.log_kv("mpc_rho_reduction_vs_comfort_ratio", mpc_vs_comfort_reduction)
    logger.log_kv("comfort_heave_rms_window_m_s2", metrics["comfort_heave_rms"])
    logger.log_kv("qp_heave_rms_window_m_s2", metrics["qp_heave_rms"])
    logger.log_kv("mpc_heave_rms_window_m_s2", metrics["mpc_heave_rms"])
    logger.log_kv("mpc_heave_degradation_vs_comfort_ratio", mpc_heave_degradation)
    logger.log_kv("qp_max_abs_force_window_n", metrics["qp_max_force"])
    logger.log_kv("mpc_max_abs_force_window_n", metrics["mpc_max_force"])
    logger.log_kv("mpc_p95_solve_time_ms", float(np.percentile(mpc["solve_time"][mpc["solve_time"] > 0.0], 95) * 1000.0))
    logger.log_kv("acceptance_mpc_extra_rho_reduction_ge_5pct", int(mpc_extra_reduction >= 0.05))
    logger.log_kv("acceptance_mpc_no_force_saturation", int(metrics["mpc_max_force"] < 0.95 * lqr.f_max))
    logger.log_kv("acceptance_mpc_stable_no_nan", int(np.all(np.isfinite(mpc["states"]))))

    _plot_rho_comparison(
        logger.run_dir / "figures" / "rho_max_mpc_vs_qp.png",
        t,
        window,
        comfort,
        one_step_qp,
        mpc,
        rho_safe,
    )
    _plot_outer_front(
        logger.run_dir / "figures" / "outer_front_rho_mpc_vs_qp.png",
        t,
        window,
        comfort,
        one_step_qp,
        mpc,
        rho_safe,
    )
    _plot_heave(
        logger.run_dir / "figures" / "heave_mpc_vs_qp.png",
        t,
        window,
        comfort,
        one_step_qp,
        mpc,
    )
    _plot_forces(logger.run_dir / "figures" / "forces_mpc_vs_qp.png", t, window, one_step_qp, mpc)

    np.savez(
        logger.run_dir / "raw.npz",
        t=t,
        roads=scenario["roads"],
        a_y=scenario["a_y"],
        f_c=scenario["f_c"],
        comfort_rho=comfort["rho_contact"],
        qp_rho=one_step_qp["rho_contact"],
        mpc_rho=mpc["rho_contact"],
        comfort_body_accel=comfort["body_accel"],
        qp_body_accel=one_step_qp["body_accel"],
        mpc_body_accel=mpc["body_accel"],
        qp_forces=one_step_qp["forces"],
        mpc_forces=mpc["forces"],
        qp_fz_true=one_step_qp["fz_true"],
        mpc_fz_true=mpc["fz_true"],
        mpc_solve_time=mpc["solve_time"],
    )
    logger.log("Phase 5.5 MPC vs one-step QP comparison completed.")
    return logger.run_dir


def _compute_metrics(window: np.ndarray, comfort: dict, qp: dict, mpc: dict) -> dict[str, float]:
    return {
        "comfort_peak_rho": float(np.max(comfort["rho_contact"][window])),
        "qp_peak_rho": float(np.max(qp["rho_contact"][window])),
        "mpc_peak_rho": float(np.max(mpc["rho_contact"][window])),
        "comfort_heave_rms": float(rmse(comfort["body_accel"][window, 0])),
        "qp_heave_rms": float(rmse(qp["body_accel"][window, 0])),
        "mpc_heave_rms": float(rmse(mpc["body_accel"][window, 0])),
        "qp_max_force": float(np.max(np.abs(qp["forces"][window]))),
        "mpc_max_force": float(np.max(np.abs(mpc["forces"][window]))),
    }


def _plot_rho_comparison(path: Path, t: np.ndarray, window: np.ndarray, comfort: dict, qp: dict, mpc: dict, rho_safe: float) -> None:
    idx = np.where(window)[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t[idx], np.max(comfort["rho_contact"][idx], axis=1), label="comfort-QP", color="tab:red")
    ax.plot(t[idx], np.max(qp["rho_contact"][idx], axis=1), label="one-step risk-QP", color="tab:gray")
    ax.plot(t[idx], np.max(mpc["rho_contact"][idx], axis=1), label="risk-MPC Np=10", color="tab:blue")
    ax.axhline(rho_safe, color="0.2", linestyle="--", linewidth=1.0, label="rho_safe")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rho_max (physical)")
    ax.set_title("Phase 5.5: MPC vs one-step QP")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_outer_front(path: Path, t: np.ndarray, window: np.ndarray, comfort: dict, qp: dict, mpc: dict, rho_safe: float) -> None:
    idx = np.where(window)[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t[idx], comfort["rho_contact"][idx, 1], label="comfort-QP FR", color="tab:red")
    ax.plot(t[idx], qp["rho_contact"][idx, 1], label="one-step risk-QP FR", color="tab:gray")
    ax.plot(t[idx], mpc["rho_contact"][idx, 1], label="risk-MPC FR", color="tab:blue")
    ax.axhline(rho_safe, color="0.2", linestyle="--", linewidth=1.0, label="rho_safe")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rho_FR")
    ax.set_title("Outer-front wheel hit by the bump")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_heave(path: Path, t: np.ndarray, window: np.ndarray, comfort: dict, qp: dict, mpc: dict) -> None:
    idx = np.where(window)[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t[idx], _moving_average(comfort["body_accel"][idx][:, :1], 21)[:, 0], label="comfort-QP", color="tab:red")
    ax.plot(t[idx], _moving_average(qp["body_accel"][idx][:, :1], 21)[:, 0], label="one-step risk-QP", color="tab:gray")
    ax.plot(t[idx], _moving_average(mpc["body_accel"][idx][:, :1], 21)[:, 0], label="risk-MPC Np=10", color="tab:blue")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Heave acceleration [m/s^2]")
    ax.set_title("Comfort trade-off (20 ms moving average)")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_forces(path: Path, t: np.ndarray, window: np.ndarray, qp: dict, mpc: dict) -> None:
    idx = np.where(window)[0]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    for ax, label, run in zip(axes, ("one-step risk-QP", "risk-MPC Np=10"), (qp, mpc)):
        forces = _moving_average(run["forces"][idx], 21)
        for corner_idx, corner in enumerate(CORNER_NAMES):
            ax.plot(t[idx], forces[:, corner_idx], label=corner)
        ax.set_ylabel("Actuator force [N]")
        ax.set_title(label)
        ax.legend(loc="best", fontsize=8)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    print(run_phase_5_5())


if __name__ == "__main__":
    main()

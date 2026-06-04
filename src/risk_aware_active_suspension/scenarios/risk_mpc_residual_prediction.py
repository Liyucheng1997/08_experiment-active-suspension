from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import (
    RESIDUAL_DECAY,
    RESIDUAL_FREEZE,
    FullCarRiskMPC,
    predict_fz_horizon,
)
from risk_aware_active_suspension.controllers.risk_qp import RiskQPParams
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.inputs.road import sine
from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.metrics.tire import rho
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.scenarios.risk_qp_fullcar_ground_truth import _smooth_j_turn
from risk_aware_active_suspension.utils.config import (
    ObserverParams,
    from_yaml,
    lqr_from_yaml,
    observer_from_yaml,
)
from risk_aware_active_suspension.utils.logger import RunLogger


def run_mpc_residual_closed_loop(
    controller: FullCarRiskMPC,
    plant: FullCar,
    t: np.ndarray,
    roads: np.ndarray,
    a_y: np.ndarray,
    f_c: np.ndarray,
    mu: float,
    observer_params: ObserverParams,
    prediction_mode: str,
    alpha: float,
    fz_tightening_n: float = 0.0,
    a_x: np.ndarray | None = None,
    preview_exogenous: bool = True,
) -> dict[str, np.ndarray]:
    n = len(t)
    dt = float(t[1] - t[0])
    a_x_seq = np.zeros(n, dtype=float) if a_x is None else np.asarray(a_x, dtype=float)
    if a_x_seq.shape != (n,):
        raise ValueError(f"a_x must have shape ({n},), got {a_x_seq.shape}.")
    steps_per_update = max(1, int(round(controller.lqr.T_s / dt)))
    states = np.zeros((n, 14), dtype=float)
    forces = np.zeros((n, 4), dtype=float)
    fz_true = np.zeros((n, 4), dtype=float)
    fz_hat = np.zeros((n, 4), dtype=float)
    fz_safe_first = np.zeros((n, 4), dtype=float)
    rho_contact = np.zeros((n, 4), dtype=float)
    xi = np.zeros((n, 4), dtype=float)
    solve_time = np.zeros(n, dtype=float)
    observers = [
        STO(observer_params, unsprung_mass=plant.params.m_u, use_saturation=True)
        for _ in range(4)
    ]

    state = np.zeros(14)
    force = np.zeros(4)
    for corner_idx, observer in enumerate(observers):
        observer.reset(v_u_hat=state[7 + 2 * corner_idx], chi_hat=0.0)

    for idx in range(n - 1):
        fz_true[idx] = plant.tire_normal_forces(state, roads[idx])
        rho_contact[idx] = rho(f_c[idx], 0.0, fz_true[idx], mu)
        # The STO residual estimates k_t * (z_r - z_u), the same dynamic
        # normal-load contribution used by plant.tire_normal_forces.
        fz_bar_now = plant.static_loads()
        for corner_idx, observer in enumerate(observers):
            phi_known = _full_car_corner_phi_known(plant, state, force, corner_idx)
            additive_force_hat, _ = observer.step(
                v_u_meas=state[7 + 2 * corner_idx],
                phi_known=phi_known,
                dt=dt,
            )
            fz_hat[idx, corner_idx] = fz_bar_now[corner_idx] + additive_force_hat

        if idx % steps_per_update == 0:
            horizon_indices = np.minimum(idx + steps_per_update * np.arange(controller.horizon), n - 1)
            fz_bar_h = np.tile(plant.static_loads(), (controller.horizon, 1))
            residual = fz_hat[idx] - fz_bar_now
            fz_h = predict_fz_horizon(
                fz_bar_h,
                residual,
                mode=prediction_mode,
                alpha=alpha,
            ) - fz_tightening_n
            if preview_exogenous:
                fc_h = f_c[horizon_indices]
                road_h = roads[horizon_indices]
                body_acc_h = np.column_stack([
                    a_x_seq[horizon_indices],
                    a_y[horizon_indices],
                ])
            else:
                fc_h = np.tile(f_c[idx], (controller.horizon, 1))
                road_h = np.tile(roads[idx], (controller.horizon, 1))
                body_acc_h = np.tile(
                    np.array([a_x_seq[idx], a_y[idx]], dtype=float),
                    (controller.horizon, 1),
                )
            force = controller.compute(
                state,
                f_z_hat=fz_h,
                f_c=fc_h,
                mu=mu,
                road_horizon=road_h,
                body_acc_horizon=body_acc_h,
                f_z_residual=fz_h - fz_bar_h,
            )
            fz_safe_first[idx] = fz_h[0]
        else:
            fz_safe_first[idx] = fz_safe_first[idx - 1]

        forces[idx] = force
        xi[idx] = controller.last_xi
        solve_time[idx] = controller.last_solve_time_s
        state = plant.step(
            state,
            roads[idx],
            dt,
            u=force,
            body_acc=np.array([a_x_seq[idx], a_y[idx]]),
        )
        states[idx + 1] = state

    fz_true[-1] = plant.tire_normal_forces(states[-1], roads[-1])
    fz_hat[-1] = fz_hat[-2]
    fz_safe_first[-1] = fz_safe_first[-2]
    rho_contact[-1] = rho(f_c[-1], 0.0, fz_true[-1], mu)
    body_accel = np.array(
        [
            plant.body_accelerations(state_i, road_i, u=force_i, body_acc=np.array([ax_i, ay_i]))
            for state_i, road_i, force_i, ax_i, ay_i in zip(states, roads, forces, a_x_seq, a_y)
        ]
    )
    return {
        "states": states,
        "forces": forces,
        "fz_true": fz_true,
        "fz_hat": fz_hat,
        "fz_safe_first": fz_safe_first,
        "rho_contact": rho_contact,
        "xi": xi,
        "solve_time": solve_time,
        "body_accel": body_accel,
    }


def run_phase_5_3(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    observer_config: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
    duration: float = 4.0,
    dt: float = 0.001,
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    observer_config = Path(observer_config)
    vehicle = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)
    base_observer = observer_from_yaml(observer_config)
    observer_params = ObserverParams(
        lambda_1=800.0,
        lambda_2=30000.0,
        epsilon=0.01,
        delta_fz_err=base_observer.delta_fz_err,
    )
    plant = FullCar(vehicle)
    logger = RunLogger.create(
        results_root,
        phase="phase-5.3",
        step="risk-mpc",
        descriptor="residual-prediction",
    )
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")
    shutil.copy2(observer_config, logger.run_dir / "observer_config.yaml")

    mu = 0.8
    rho_safe = 0.85
    t = np.arange(0.0, duration, dt)
    a_y = _smooth_j_turn(t, peak=6.0, rise_time=0.5)
    roads = np.zeros((len(t), 4), dtype=float)
    roads[:, 0] = sine(t, amplitude=0.015, freq_hz=1.0)
    roads[:, 2] = sine(t, amplitude=0.015, freq_hz=1.0, phase_rad=-0.7)
    f_c = np.tile(np.array([2300.0, 800.0, 900.0, 700.0]), (len(t), 1))
    window = (t >= 1.0) & (t <= 3.5)

    risk_params = RiskWeightParams(
        rho_th=0.75,
        k_rho=25.0,
        kappa_rho=5.0,
        q_c_min=1.0,
        q_c_max=1.0,
        q_p_min=0.0,
        q_p_max=1.0e-5,
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

    modes = {
        RESIDUAL_FREEZE: {"mode": RESIDUAL_FREEZE, "alpha": 1.0},
        "decay_alpha_0.95": {"mode": RESIDUAL_DECAY, "alpha": 0.95},
    }
    runs = {}
    for label, cfg in modes.items():
        mpc = FullCarRiskMPC(vehicle, lqr, risk_params, qp_params, horizon=10)
        runs[label] = run_mpc_residual_closed_loop(
            mpc,
            plant,
            t,
            roads,
            a_y,
            f_c,
            mu,
            observer_params,
            prediction_mode=cfg["mode"],
            alpha=cfg["alpha"],
            fz_tightening_n=base_observer.delta_fz_err,
        )

    comfort = _run_comfort_baseline(FullCarComfortQP(vehicle, lqr), plant, t, roads, a_y, f_c, mu)
    comfort_peak = float(np.max(comfort["rho_contact"][window]))
    logger.log_kv("comfort_peak_rho_contact_window", comfort_peak)
    logger.log_kv("rho_safe", rho_safe)
    for label, run in runs.items():
        peak = float(np.max(run["rho_contact"][window]))
        heave_rms = float(rms(run["body_accel"][window, 0]))
        max_force = float(np.max(np.abs(run["forces"][window])))
        p95_solve_ms = float(np.percentile(run["solve_time"][run["solve_time"] > 0.0], 95) * 1000.0)
        logger.log_kv(f"{label}_peak_rho_contact_window", peak)
        logger.log_kv(f"{label}_rho_reduction_vs_comfort_ratio", (comfort_peak - peak) / max(comfort_peak, 1.0e-9))
        logger.log_kv(f"{label}_heave_rms_window_m_s2", heave_rms)
        logger.log_kv(f"{label}_max_abs_force_n", max_force)
        logger.log_kv(f"{label}_p95_solve_time_ms", p95_solve_ms)
        logger.log_kv(f"{label}_stable_no_nan", int(np.all(np.isfinite(run["states"]))))

    _plot_rho(logger.run_dir / "figures" / "rho_max_residual_modes.png", t, window, comfort, runs, rho_safe)
    _plot_heave(logger.run_dir / "figures" / "heave_residual_modes.png", t, window, comfort, runs)
    _plot_forces(logger.run_dir / "figures" / "forces_residual_modes.png", t, window, runs)
    _plot_fz_prediction(
        logger.run_dir / "figures" / "outer_front_fz_prediction_modes.png",
        t,
        window,
        runs,
    )

    np.savez(
        logger.run_dir / "raw.npz",
        t=t,
        roads=roads,
        a_y=a_y,
        f_c=f_c,
        comfort_rho=comfort["rho_contact"],
        freeze_rho=runs[RESIDUAL_FREEZE]["rho_contact"],
        decay_rho=runs["decay_alpha_0.95"]["rho_contact"],
        freeze_forces=runs[RESIDUAL_FREEZE]["forces"],
        decay_forces=runs["decay_alpha_0.95"]["forces"],
        freeze_fz_hat=runs[RESIDUAL_FREEZE]["fz_hat"],
        decay_fz_hat=runs["decay_alpha_0.95"]["fz_hat"],
        freeze_fz_true=runs[RESIDUAL_FREEZE]["fz_true"],
        decay_fz_true=runs["decay_alpha_0.95"]["fz_true"],
        freeze_fz_safe=runs[RESIDUAL_FREEZE]["fz_safe_first"],
        decay_fz_safe=runs["decay_alpha_0.95"]["fz_safe_first"],
    )
    logger.log("Phase 5.3 residual prediction comparison completed.")
    return logger.run_dir


def _run_comfort_baseline(controller, plant, t, roads, a_y, f_c, mu) -> dict[str, np.ndarray]:
    states = np.zeros((len(t), 14), dtype=float)
    forces = np.zeros((len(t), 4), dtype=float)
    rho_contact = np.zeros((len(t), 4), dtype=float)
    state = np.zeros(14)
    for idx in range(len(t) - 1):
        force = controller.compute(state)
        forces[idx] = force
        fz = plant.tire_normal_forces(state, roads[idx])
        rho_contact[idx] = rho(f_c[idx], 0.0, fz, mu)
        state = plant.step(state, roads[idx], float(t[idx + 1] - t[idx]), u=force, body_acc=np.array([0.0, a_y[idx]]))
        states[idx + 1] = state
    rho_contact[-1] = rho(f_c[-1], 0.0, plant.tire_normal_forces(states[-1], roads[-1]), mu)
    body_accel = np.array(
        [plant.body_accelerations(s, r, u=u, body_acc=np.array([0.0, ay])) for s, r, u, ay in zip(states, roads, forces, a_y)]
    )
    return {"states": states, "forces": forces, "rho_contact": rho_contact, "body_accel": body_accel}


def _full_car_corner_phi_known(
    plant: FullCar,
    state: np.ndarray,
    force: np.ndarray,
    corner_idx: int,
) -> float:
    x_pos, y_pos = plant.corner_xy[corner_idx]
    k = plant.corner_k[corner_idx]
    c = plant.corner_c[corner_idx]
    z_u_idx = 6 + 2 * corner_idx
    v_u_idx = z_u_idx + 1
    body_corner_z = state[0] + y_pos * state[2] + x_pos * state[4]
    body_corner_v = state[1] + y_pos * state[3] + x_pos * state[5]
    suspension_accel = (
        k * (body_corner_z - state[z_u_idx]) + c * (body_corner_v - state[v_u_idx])
    ) / plant.params.m_u
    actuator_accel = -float(force[corner_idx]) / plant.params.m_u
    return float(suspension_accel + actuator_accel)


def _moving_average(values: np.ndarray, window_samples: int) -> np.ndarray:
    if window_samples <= 1:
        return values
    if window_samples % 2 == 0:
        window_samples += 1
    kernel = np.ones(window_samples, dtype=float) / window_samples
    padded = np.pad(values, ((window_samples // 2, window_samples // 2), (0, 0)), mode="edge")
    out = np.zeros_like(values, dtype=float)
    for col in range(values.shape[1]):
        out[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return out


def _plot_rho(path: Path, t: np.ndarray, window: np.ndarray, comfort: dict, runs: dict, rho_safe: float) -> None:
    idx = np.where(window)[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t[idx], np.max(comfort["rho_contact"][idx], axis=1), label="comfort-QP", color="tab:red")
    ax.plot(t[idx], np.max(runs[RESIDUAL_FREEZE]["rho_contact"][idx], axis=1), label="MPC freeze", color="tab:blue")
    ax.plot(t[idx], np.max(runs["decay_alpha_0.95"]["rho_contact"][idx], axis=1), label="MPC decay α=0.95", color="tab:orange")
    ax.axhline(rho_safe, color="0.2", linestyle="--", linewidth=1.0, label="rho_safe")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rho_max (physical)")
    ax.set_title("Phase 5.3 residual prediction modes")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_heave(path: Path, t: np.ndarray, window: np.ndarray, comfort: dict, runs: dict) -> None:
    idx = np.where(window)[0]
    comfort_heave = _moving_average(comfort["body_accel"][idx][:, :1], window_samples=21)[:, 0]
    freeze_heave = _moving_average(runs[RESIDUAL_FREEZE]["body_accel"][idx][:, :1], window_samples=21)[:, 0]
    decay_heave = _moving_average(runs["decay_alpha_0.95"]["body_accel"][idx][:, :1], window_samples=21)[:, 0]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.plot(t[idx], comfort_heave, label="comfort-QP", color="tab:red")
    ax.plot(t[idx], freeze_heave, label="MPC freeze", color="tab:blue")
    ax.plot(t[idx], decay_heave, label="MPC decay α=0.95", color="tab:orange")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Heave acceleration [m/s^2]")
    ax.set_title("Comfort comparison (20 ms moving average)")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_forces(path: Path, t: np.ndarray, window: np.ndarray, runs: dict) -> None:
    idx = np.where(window)[0]
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.0), sharex=True)
    for ax, label in zip(axes, (RESIDUAL_FREEZE, "decay_alpha_0.95")):
        forces = _moving_average(runs[label]["forces"][idx], window_samples=51)
        for corner_idx, corner in enumerate(CORNER_NAMES):
            ax.plot(t[idx], forces[:, corner_idx], label=corner)
        ax.set_ylabel("Actuator force [N]")
        ax.set_title(label)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_fz_prediction(path: Path, t: np.ndarray, window: np.ndarray, runs: dict) -> None:
    idx = np.where(window)[0]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    fz_hat = _moving_average(runs[RESIDUAL_FREEZE]["fz_hat"][idx], window_samples=51)[:, 0]
    fz_safe_freeze = _moving_average(runs[RESIDUAL_FREEZE]["fz_safe_first"][idx], window_samples=51)[:, 0]
    fz_safe_decay = _moving_average(runs["decay_alpha_0.95"]["fz_safe_first"][idx], window_samples=51)[:, 0]
    fz_true = _moving_average(runs[RESIDUAL_FREEZE]["fz_true"][idx], window_samples=51)[:, 0]
    ax.plot(t[idx], fz_hat, label="Fz_hat FL freeze run", color="tab:blue", alpha=0.75)
    ax.plot(t[idx], fz_safe_freeze, label="Fz_safe FL freeze", color="tab:blue", linestyle="--")
    ax.plot(t[idx], fz_safe_decay, label="Fz_safe FL decay", color="tab:orange", linestyle="--")
    ax.plot(t[idx], fz_true, label="Fz true FL", color="0.2", linewidth=1.0)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Force [N]")
    ax.set_title("First-step F_z prediction diagnostic (50 ms moving average)")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    print(run_phase_5_3())


if __name__ == "__main__":
    main()

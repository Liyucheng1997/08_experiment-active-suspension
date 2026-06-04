from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.utils.config import ObserverParams, from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def constant_disturbance_validation(
    unsprung_mass: float,
    params: ObserverParams,
    force_n: float = 200.0,
    duration: float = 1.0,
    dt: float = 0.0005,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    truth = np.full_like(t, force_n)
    return _run_synthetic_case(unsprung_mass, params, t, truth)


def sinusoidal_disturbance_validation(
    unsprung_mass: float,
    params: ObserverParams,
    amplitude_n: float = 200.0,
    freq_hz: float = 3.0,
    duration: float = 2.0,
    dt: float = 0.0005,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    truth = amplitude_n * np.sin(2.0 * np.pi * freq_hz * t)
    return _run_synthetic_case(unsprung_mass, params, t, truth)


def settling_time(t: np.ndarray, error: np.ndarray, tolerance: float) -> float:
    outside = np.flatnonzero(np.abs(error) > tolerance)
    if len(outside) == 0:
        return 0.0
    last_outside = outside[-1]
    if last_outside >= len(t) - 1:
        return float("inf")
    return float(t[last_outside + 1])


def _run_synthetic_case(
    unsprung_mass: float,
    params: ObserverParams,
    t: np.ndarray,
    truth_force: np.ndarray,
) -> dict[str, float | np.ndarray]:
    dt = float(t[1] - t[0])
    v_meas = np.zeros_like(t)
    truth_accel = truth_force / unsprung_mass
    for idx in range(len(t) - 1):
        v_meas[idx + 1] = v_meas[idx] + dt * truth_accel[idx]

    sto = STO(params=params, unsprung_mass=unsprung_mass)
    sto.reset(v_u_hat=v_meas[0], chi_hat=0.0)
    d_z_hat = np.zeros_like(t)
    chi_hat = np.zeros_like(t)
    v_u_hat = np.zeros_like(t)
    for idx in range(len(t)):
        d_z_hat[idx], _ = sto.step(v_u_meas=v_meas[idx], phi_known=0.0, dt=dt)
        chi_hat[idx] = sto.chi_hat
        v_u_hat[idx] = sto.v_u_hat

    error = d_z_hat - truth_force
    post = t >= 0.2
    return {
        "t": t,
        "truth_force": truth_force,
        "truth_accel": truth_accel,
        "v_meas": v_meas,
        "v_u_hat": v_u_hat,
        "chi_hat": chi_hat,
        "d_z_hat": d_z_hat,
        "force_error": error,
        "settling_time_s": settling_time(t, error, tolerance=0.01 * max(1.0, np.max(np.abs(truth_force)))),
        "steady_state_rmse_n": rms(error[post]),
        "steady_state_rmse_ratio": rms(error[post]) / max(1.0, np.max(np.abs(truth_force))),
    }


def run_phase_2_2(
    vehicle_config_path: str | Path = "configs/vehicle_default.yaml",
    observer_config_path: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    vehicle_config_path = Path(vehicle_config_path)
    observer_config_path = Path(observer_config_path)
    vehicle = from_yaml(vehicle_config_path)
    observer_params = observer_from_yaml(observer_config_path)
    logger = RunLogger.create(results_root, phase="phase-2.2", step="sto", descriptor="synthetic-disturbance")
    shutil.copy2(vehicle_config_path, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(observer_config_path, logger.run_dir / "observer_config.yaml")

    constant = constant_disturbance_validation(vehicle.m_u, observer_params)
    sinusoidal = sinusoidal_disturbance_validation(vehicle.m_u, observer_params)
    metrics = {
        "constant_settling_time_s": constant["settling_time_s"],
        "constant_steady_state_rmse_n": constant["steady_state_rmse_n"],
        "constant_steady_state_rmse_ratio": constant["steady_state_rmse_ratio"],
        "sinusoidal_steady_state_rmse_n": sinusoidal["steady_state_rmse_n"],
        "sinusoidal_steady_state_rmse_ratio": sinusoidal["steady_state_rmse_ratio"],
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    plot_timeseries(
        constant["t"],
        np.column_stack([constant["truth_force"], constant["d_z_hat"]]),
        labels=("truth", "STO estimate"),
        ylabel="Residual force [N]",
        title="STO constant synthetic disturbance",
        save_path=logger.run_dir / "figures" / "constant_disturbance.png",
    )
    plot_timeseries(
        sinusoidal["t"],
        np.column_stack([sinusoidal["truth_force"], sinusoidal["d_z_hat"]]),
        labels=("truth", "STO estimate"),
        ylabel="Residual force [N]",
        title="STO sinusoidal synthetic disturbance",
        save_path=logger.run_dir / "figures" / "sinusoidal_disturbance.png",
    )
    plot_timeseries(
        sinusoidal["t"],
        sinusoidal["force_error"],
        ylabel="Force error [N]",
        title="STO sinusoidal synthetic disturbance error",
        save_path=logger.run_dir / "figures" / "sinusoidal_error.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        constant_t=constant["t"],
        constant_truth_force=constant["truth_force"],
        constant_d_z_hat=constant["d_z_hat"],
        constant_force_error=constant["force_error"],
        sinusoidal_t=sinusoidal["t"],
        sinusoidal_truth_force=sinusoidal["truth_force"],
        sinusoidal_d_z_hat=sinusoidal["d_z_hat"],
        sinusoidal_force_error=sinusoidal["force_error"],
    )
    logger.log("Phase 2.2 STO synthetic-disturbance validation completed.")
    return logger.run_dir


def main() -> None:
    run_dir = run_phase_2_2()
    print(run_dir)


if __name__ == "__main__":
    main()

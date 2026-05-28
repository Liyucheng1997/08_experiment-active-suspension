from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.utils.config import from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def sto_zero_disturbance_validation(
    sto: STO,
    duration: float = 2.0,
    dt: float = 0.001,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    phi_known = 1.5 * np.sin(2.0 * np.pi * 1.2 * t)
    v_meas = np.zeros_like(t)
    for idx in range(len(t) - 1):
        v_meas[idx + 1] = v_meas[idx] + dt * phi_known[idx]

    sto.reset(v_u_hat=v_meas[0], chi_hat=0.0)
    d_z_hat = np.zeros_like(t)
    f_z_hat = np.zeros_like(t)
    v_u_hat = np.zeros_like(t)
    chi_hat = np.zeros_like(t)
    for idx in range(len(t)):
        d_z_hat[idx], f_z_hat[idx] = sto.step(v_meas[idx], phi_known[idx], dt, f_z_bar=0.0)
        v_u_hat[idx] = sto.v_u_hat
        chi_hat[idx] = sto.chi_hat

    transient = t >= 0.2
    v_error = v_u_hat - v_meas
    return {
        "t": t,
        "phi_known": phi_known,
        "v_meas": v_meas,
        "v_u_hat": v_u_hat,
        "v_error": v_error,
        "chi_hat": chi_hat,
        "d_z_hat": d_z_hat,
        "f_z_hat": f_z_hat,
        "max_abs_chi_after_transient": float(np.max(np.abs(chi_hat[transient]))),
        "max_abs_dz_after_transient": float(np.max(np.abs(d_z_hat[transient]))),
        "max_abs_velocity_error_after_transient": float(np.max(np.abs(v_error[transient]))),
    }


def run_phase_2_1(
    vehicle_config_path: str | Path = "configs/vehicle_default.yaml",
    observer_config_path: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    vehicle_config_path = Path(vehicle_config_path)
    observer_config_path = Path(observer_config_path)
    vehicle = from_yaml(vehicle_config_path)
    observer_params = observer_from_yaml(observer_config_path)
    sto = STO(observer_params, unsprung_mass=vehicle.m_u)
    logger = RunLogger.create(results_root, phase="phase-2.1", step="sto", descriptor="zero-disturbance")
    shutil.copy2(vehicle_config_path, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(observer_config_path, logger.run_dir / "observer_config.yaml")

    result = sto_zero_disturbance_validation(sto)
    metrics = {
        "max_abs_chi_after_transient": result["max_abs_chi_after_transient"],
        "max_abs_dz_after_transient_n": result["max_abs_dz_after_transient"],
        "max_abs_velocity_error_after_transient_m_s": result["max_abs_velocity_error_after_transient"],
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    plot_timeseries(
        result["t"],
        np.column_stack([result["v_meas"], result["v_u_hat"]]),
        labels=("v_u measured", "v_u hat"),
        ylabel="Unsprung velocity [m/s]",
        title="STO zero-disturbance velocity tracking",
        save_path=logger.run_dir / "figures" / "velocity_tracking.png",
    )
    plot_timeseries(
        result["t"],
        result["chi_hat"],
        ylabel="chi_hat [m/s^2]",
        title="STO zero-disturbance residual acceleration",
        save_path=logger.run_dir / "figures" / "chi_hat.png",
    )
    plot_timeseries(
        result["t"],
        result["d_z_hat"],
        ylabel="d_z_hat [N]",
        title="STO zero-disturbance residual force",
        save_path=logger.run_dir / "figures" / "d_z_hat.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        t=result["t"],
        phi_known=result["phi_known"],
        v_meas=result["v_meas"],
        v_u_hat=result["v_u_hat"],
        v_error=result["v_error"],
        chi_hat=result["chi_hat"],
        d_z_hat=result["d_z_hat"],
        f_z_hat=result["f_z_hat"],
    )
    logger.log("Phase 2.1 STO zero-disturbance validation completed.")
    return logger.run_dir


def main() -> None:
    run_dir = run_phase_2_1()
    print(run_dir)


if __name__ == "__main__":
    main()

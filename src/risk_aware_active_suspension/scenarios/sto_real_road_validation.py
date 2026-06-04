from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.inputs.road import iso8608
from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.utils.config import ObserverParams, from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_side_by_side, plot_timeseries


def sto_real_road_validation(
    plant: QuarterCar,
    observer_params,
    duration: float = 12.0,
    dt: float = 0.0005,
    v_x: float = 80.0 / 3.6,
    road_class: str = "B",
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    road = iso8608(road_class, v_x=v_x, t=t, seed=41)
    states = plant.simulate(t, u_seq=0.0, w_seq=road)
    phi_known = np.array([_quarter_car_phi_known(plant, state) for state in states])
    truth_tire_dynamic_force = plant.params.k_t * (states[:, 2] - road)
    truth_additive_force = -truth_tire_dynamic_force
    truth_fz = plant.params.m * 9.81 / 4.0 + truth_tire_dynamic_force

    sto = STO(observer_params, unsprung_mass=plant.params.m_u)
    sto.reset(v_u_hat=states[0, 3], chi_hat=0.0)
    additive_force_hat = np.zeros_like(t)
    tire_dynamic_force_hat = np.zeros_like(t)
    f_z_hat = np.zeros_like(t)
    v_u_hat = np.zeros_like(t)
    chi_hat = np.zeros_like(t)
    for idx in range(len(t)):
        additive_force_hat[idx], _ = sto.step(
            v_u_meas=states[idx, 3],
            phi_known=phi_known[idx],
            dt=dt,
        )
        tire_dynamic_force_hat[idx] = -additive_force_hat[idx]
        f_z_hat[idx] = plant.params.m * 9.81 / 4.0 + tire_dynamic_force_hat[idx]
        v_u_hat[idx] = sto.v_u_hat
        chi_hat[idx] = sto.chi_hat

    post = t >= 0.5
    truth_post = truth_tire_dynamic_force[post]
    estimate_post = tire_dynamic_force_hat[post]
    corr = float(np.corrcoef(truth_post, estimate_post)[0, 1])
    error = tire_dynamic_force_hat - truth_tire_dynamic_force
    return {
        "t": t,
        "road": road,
        "states": states,
        "phi_known": phi_known,
        "truth_additive_force": truth_additive_force,
        "truth_tire_dynamic_force": truth_tire_dynamic_force,
        "truth_fz": truth_fz,
        "additive_force_hat": additive_force_hat,
        "tire_dynamic_force_hat": tire_dynamic_force_hat,
        "f_z_hat": f_z_hat,
        "v_u_hat": v_u_hat,
        "chi_hat": chi_hat,
        "force_error": error,
        "correlation_after_transient": corr,
        "rmse_after_transient_n": rms(error[post]),
        "rmse_ratio_after_transient": rms(error[post]) / max(1.0, np.max(np.abs(truth_post))),
        "truth_tire_dynamic_peak_abs_n": float(np.max(np.abs(truth_post))),
    }


def run_phase_2_4(
    vehicle_config_path: str | Path = "configs/vehicle_default.yaml",
    observer_config_path: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    vehicle_config_path = Path(vehicle_config_path)
    observer_config_path = Path(observer_config_path)
    vehicle = from_yaml(vehicle_config_path)
    base_observer_params = observer_from_yaml(observer_config_path)
    observer_params = ObserverParams(
        lambda_1=100.0,
        lambda_2=2000.0,
        epsilon=base_observer_params.epsilon,
        delta_fz_err=base_observer_params.delta_fz_err,
    )
    plant = QuarterCar(vehicle)
    logger = RunLogger.create(results_root, phase="phase-2.4", step="sto", descriptor="real-road-quarter-car")
    shutil.copy2(vehicle_config_path, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(observer_config_path, logger.run_dir / "observer_config.yaml")

    result = sto_real_road_validation(plant, observer_params)
    metrics = {
        "correlation_after_transient": result["correlation_after_transient"],
        "rmse_after_transient_n": result["rmse_after_transient_n"],
        "rmse_ratio_after_transient": result["rmse_ratio_after_transient"],
        "truth_tire_dynamic_peak_abs_n": result["truth_tire_dynamic_peak_abs_n"],
        "real_road_lambda_1": observer_params.lambda_1,
        "real_road_lambda_2": observer_params.lambda_2,
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    plot_timeseries(
        result["t"],
        np.column_stack([result["truth_tire_dynamic_force"], result["tire_dynamic_force_hat"]]),
        labels=("truth", "STO estimate"),
        ylabel="Tire dynamic force [N]",
        title="Quarter-car STO under ISO-B road excitation",
        save_path=logger.run_dir / "figures" / "tire_dynamic_force_tracking.png",
    )
    plot_timeseries(
        result["t"],
        result["force_error"],
        ylabel="Force error [N]",
        title="Quarter-car STO real-road residual error",
        save_path=logger.run_dir / "figures" / "tire_dynamic_force_error.png",
    )
    plot_side_by_side(
        result["t"],
        result["road"],
        result["states"][:, 3],
        left_label="Road height [m]",
        right_label="Unsprung velocity [m/s]",
        title="Quarter-car real-road STO inputs",
        save_path=logger.run_dir / "figures" / "road_and_velocity.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        t=result["t"],
        road=result["road"],
        states=result["states"],
        phi_known=result["phi_known"],
        truth_additive_force=result["truth_additive_force"],
        truth_tire_dynamic_force=result["truth_tire_dynamic_force"],
        truth_fz=result["truth_fz"],
        additive_force_hat=result["additive_force_hat"],
        tire_dynamic_force_hat=result["tire_dynamic_force_hat"],
        f_z_hat=result["f_z_hat"],
        force_error=result["force_error"],
    )
    logger.log("Phase 2.4 STO real-road validation completed.")
    return logger.run_dir


def _quarter_car_phi_known(plant: QuarterCar, state: np.ndarray) -> float:
    p = plant.params
    z_s, v_s, z_u, v_u = state
    return (p.k_s * (z_s - z_u) + p.c_s * (v_s - v_u)) / p.m_u


def main() -> None:
    run_dir = run_phase_2_4()
    print(run_dir)


if __name__ == "__main__":
    main()

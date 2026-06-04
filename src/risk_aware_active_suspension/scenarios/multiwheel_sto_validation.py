from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.inputs.road import rounded_bump
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.utils.config import ObserverParams, from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def single_wheel_bump_validation(
    plant: FullCar,
    observer_params: ObserverParams,
    target_wheel: int = 0,
    duration: float = 2.0,
    dt: float = 0.0005,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    roads = np.zeros((len(t), 4))
    roads[:, target_wheel] = rounded_bump(t, height=0.01, start=0.4, duration=0.08)
    states = plant.simulate(t, roads)
    truth = plant.params.k_t * (roads - states[:, 6::2])
    estimates = _run_multiwheel_sto(plant, observer_params, states, roads, dt)

    post = t >= 0.2
    estimate_peaks = np.max(np.abs(estimates[post]), axis=0)
    truth_peaks = np.max(np.abs(truth[post]), axis=0)
    off_indices = [idx for idx in range(4) if idx != target_wheel]
    off_to_on_ratio = float(np.max(estimate_peaks[off_indices]) / estimate_peaks[target_wheel])
    return {
        "t": t,
        "roads": roads,
        "states": states,
        "truth_tire_dynamic_force": truth,
        "tire_dynamic_force_hat": estimates,
        "estimate_peaks": estimate_peaks,
        "truth_peaks": truth_peaks,
        "off_to_on_ratio": off_to_on_ratio,
    }


def lateral_transfer_validation(
    plant: FullCar,
    observer_params: ObserverParams,
    a_y: float = 5.0,
    duration: float = 2.0,
    dt: float = 0.0005,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    roads = np.zeros((len(t), 4))
    states = plant.simulate(t, roads)
    estimates = _run_multiwheel_sto(plant, observer_params, states, roads, dt)
    fz_bar = np.array([plant.normal_loads_quasi_static(a_y=_smooth_step(a_y, time)) for time in t])
    fz_hat = fz_bar + estimates
    post = t >= 0.5
    mean_estimate = np.mean(estimates[post], axis=0)
    peak_lateral_transfer = float(np.max(np.abs(fz_bar[post] - plant.static_loads())))
    return {
        "t": t,
        "roads": roads,
        "states": states,
        "tire_dynamic_force_hat": estimates,
        "fz_bar": fz_bar,
        "fz_hat": fz_hat,
        "mean_dynamic_estimate": mean_estimate,
        "max_abs_mean_dynamic_estimate": float(np.max(np.abs(mean_estimate))),
        "peak_lateral_transfer_n": peak_lateral_transfer,
        "mean_to_transfer_ratio": float(np.max(np.abs(mean_estimate)) / peak_lateral_transfer),
    }


def run_phase_2_7(
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
    plant = FullCar(vehicle)
    logger = RunLogger.create(results_root, phase="phase-2.7", step="sto", descriptor="multiwheel")
    shutil.copy2(vehicle_config_path, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(observer_config_path, logger.run_dir / "observer_config.yaml")

    bump = single_wheel_bump_validation(plant, observer_params)
    lateral = lateral_transfer_validation(plant, observer_params)
    metrics = {
        "single_wheel_off_to_on_ratio": bump["off_to_on_ratio"],
        "single_wheel_on_estimate_peak_n": bump["estimate_peaks"][0],
        "single_wheel_max_off_estimate_peak_n": float(np.max(bump["estimate_peaks"][1:])),
        "lateral_max_abs_mean_dynamic_estimate_n": lateral["max_abs_mean_dynamic_estimate"],
        "lateral_peak_transfer_n": lateral["peak_lateral_transfer_n"],
        "lateral_mean_to_transfer_ratio": lateral["mean_to_transfer_ratio"],
        "real_road_lambda_1": observer_params.lambda_1,
        "real_road_lambda_2": observer_params.lambda_2,
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    plot_timeseries(
        bump["t"],
        bump["tire_dynamic_force_hat"],
        labels=CORNER_NAMES,
        ylabel="Tire dynamic force estimate [N]",
        title="Multi-wheel STO: single-wheel bump response",
        save_path=logger.run_dir / "figures" / "single_wheel_bump_estimates.png",
    )
    plot_timeseries(
        bump["t"],
        bump["truth_tire_dynamic_force"],
        labels=CORNER_NAMES,
        ylabel="Truth tire dynamic force [N]",
        title="Single-wheel bump truth",
        save_path=logger.run_dir / "figures" / "single_wheel_bump_truth.png",
    )
    plot_timeseries(
        lateral["t"],
        lateral["tire_dynamic_force_hat"],
        labels=CORNER_NAMES,
        ylabel="Tire dynamic force estimate [N]",
        title="Multi-wheel STO under quasi-static lateral transfer",
        save_path=logger.run_dir / "figures" / "lateral_transfer_dynamic_estimates.png",
    )
    plot_timeseries(
        lateral["t"],
        lateral["fz_bar"],
        labels=CORNER_NAMES,
        ylabel="F_z bar [N]",
        title="Known quasi-static lateral load transfer",
        save_path=logger.run_dir / "figures" / "lateral_transfer_fz_bar.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        bump_t=bump["t"],
        bump_roads=bump["roads"],
        bump_truth=bump["truth_tire_dynamic_force"],
        bump_estimates=bump["tire_dynamic_force_hat"],
        lateral_t=lateral["t"],
        lateral_estimates=lateral["tire_dynamic_force_hat"],
        lateral_fz_bar=lateral["fz_bar"],
        lateral_fz_hat=lateral["fz_hat"],
    )
    logger.log("Phase 2.7 multi-wheel STO validation completed.")
    return logger.run_dir


def _run_multiwheel_sto(
    plant: FullCar,
    observer_params: ObserverParams,
    states: np.ndarray,
    roads: np.ndarray,
    dt: float,
) -> np.ndarray:
    observers = [
        STO(observer_params, unsprung_mass=plant.params.m_u, use_saturation=True)
        for _ in range(4)
    ]
    for corner_idx, observer in enumerate(observers):
        observer.reset(v_u_hat=states[0, 7 + 2 * corner_idx], chi_hat=0.0)

    estimates = np.zeros((len(states), 4))
    for time_idx in range(len(states)):
        for corner_idx, observer in enumerate(observers):
            phi_known = _full_car_corner_phi_known(plant, states[time_idx], corner_idx)
            additive_force_hat, _ = observer.step(
                v_u_meas=states[time_idx, 7 + 2 * corner_idx],
                phi_known=phi_known,
                dt=dt,
            )
            estimates[time_idx, corner_idx] = additive_force_hat
    return estimates


def _full_car_corner_phi_known(plant: FullCar, state: np.ndarray, corner_idx: int) -> float:
    x_pos, y_pos = plant.corner_xy[corner_idx]
    k = plant.corner_k[corner_idx]
    c = plant.corner_c[corner_idx]
    z_u_idx = 6 + 2 * corner_idx
    v_u_idx = z_u_idx + 1
    body_corner_z = state[0] + y_pos * state[2] + x_pos * state[4]
    body_corner_v = state[1] + y_pos * state[3] + x_pos * state[5]
    return (k * (body_corner_z - state[z_u_idx]) + c * (body_corner_v - state[v_u_idx])) / plant.params.m_u


def _smooth_step(target: float, t: float, start: float = 0.2, rise_time: float = 0.4) -> float:
    tau = np.clip((t - start) / rise_time, 0.0, 1.0)
    smooth = tau * tau * (3.0 - 2.0 * tau)
    return float(target * smooth)


def main() -> None:
    run_dir = run_phase_2_7()
    print(run_dir)


if __name__ == "__main__":
    main()

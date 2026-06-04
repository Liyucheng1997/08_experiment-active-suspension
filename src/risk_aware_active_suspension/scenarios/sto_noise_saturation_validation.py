from __future__ import annotations

import shutil
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.inputs.road import iso8608
from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.scenarios.sto_real_road_validation import _quarter_car_phi_known
from risk_aware_active_suspension.utils.config import ObserverParams, from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def add_noise_for_snr(signal: np.ndarray, snr_db: float, seed: int = 101) -> tuple[np.ndarray, float]:
    signal = np.asarray(signal, dtype=float)
    signal_power = float(np.mean(signal**2))
    noise_std = np.sqrt(signal_power / (10.0 ** (snr_db / 10.0)))
    rng = np.random.default_rng(seed)
    return signal + rng.normal(0.0, noise_std, size=signal.shape), float(noise_std)


def sto_noise_saturation_validation(
    plant: QuarterCar,
    observer_params: ObserverParams,
    duration: float = 8.0,
    dt: float = 0.0005,
    v_x: float = 80.0 / 3.6,
    snr_db: float = 20.0,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    road = iso8608("B", v_x=v_x, t=t, seed=41)
    states = plant.simulate(t, u_seq=0.0, w_seq=road)
    phi_known = np.array([_quarter_car_phi_known(plant, state) for state in states])
    truth_tire_dynamic_force = plant.params.k_t * (states[:, 2] - road)
    noisy_velocity, noise_std = add_noise_for_snr(states[:, 3], snr_db=snr_db)

    ideal = _run_noisy_observer(plant, observer_params, noisy_velocity, phi_known, dt, use_saturation=False)
    saturated = _run_noisy_observer(plant, observer_params, noisy_velocity, phi_known, dt, use_saturation=True)
    post = t >= 0.5
    ideal_error = ideal["tire_dynamic_force_hat"] - truth_tire_dynamic_force
    saturated_error = saturated["tire_dynamic_force_hat"] - truth_tire_dynamic_force
    ideal_chatter = _chatter_metric(ideal["tire_dynamic_force_hat"][post], dt)
    saturated_chatter = _chatter_metric(saturated["tire_dynamic_force_hat"][post], dt)

    return {
        "t": t,
        "road": road,
        "states": states,
        "phi_known": phi_known,
        "truth_tire_dynamic_force": truth_tire_dynamic_force,
        "noisy_velocity": noisy_velocity,
        "velocity_noise_std": noise_std,
        "ideal_tire_dynamic_force_hat": ideal["tire_dynamic_force_hat"],
        "saturated_tire_dynamic_force_hat": saturated["tire_dynamic_force_hat"],
        "ideal_force_error": ideal_error,
        "saturated_force_error": saturated_error,
        "ideal_switching": ideal["switching"],
        "saturated_switching": saturated["switching"],
        "ideal_chatter_n_per_s": ideal_chatter,
        "saturated_chatter_n_per_s": saturated_chatter,
        "chatter_reduction_ratio": saturated_chatter / ideal_chatter,
        "ideal_rmse_after_transient_n": rms(ideal_error[post]),
        "saturated_rmse_after_transient_n": rms(saturated_error[post]),
        "saturated_rmse_ratio_after_transient": rms(saturated_error[post])
        / max(1.0, np.max(np.abs(truth_tire_dynamic_force[post]))),
        "saturated_correlation_after_transient": float(
            np.corrcoef(truth_tire_dynamic_force[post], saturated["tire_dynamic_force_hat"][post])[0, 1]
        ),
    }


def run_phase_2_5(
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
    logger = RunLogger.create(results_root, phase="phase-2.5", step="sto", descriptor="noise-saturation")
    shutil.copy2(vehicle_config_path, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(observer_config_path, logger.run_dir / "observer_config.yaml")

    result = sto_noise_saturation_validation(plant, observer_params)
    epsilon_sweep = saturation_epsilon_sweep(plant, observer_params)
    metrics = {
        "velocity_noise_std_m_s": result["velocity_noise_std"],
        "ideal_chatter_n_per_s": result["ideal_chatter_n_per_s"],
        "saturated_chatter_n_per_s": result["saturated_chatter_n_per_s"],
        "chatter_reduction_ratio": result["chatter_reduction_ratio"],
        "ideal_rmse_after_transient_n": result["ideal_rmse_after_transient_n"],
        "saturated_rmse_after_transient_n": result["saturated_rmse_after_transient_n"],
        "saturated_rmse_ratio_after_transient": result["saturated_rmse_ratio_after_transient"],
        "saturated_correlation_after_transient": result["saturated_correlation_after_transient"],
        "saturation_epsilon": observer_params.epsilon,
        "real_road_lambda_1": observer_params.lambda_1,
        "real_road_lambda_2": observer_params.lambda_2,
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    _write_epsilon_sweep_csv(logger.run_dir / "epsilon_sweep.csv", epsilon_sweep)
    _plot_epsilon_sweep(logger.run_dir / "figures" / "epsilon_sweep.png", epsilon_sweep)
    plot_timeseries(
        result["t"],
        np.column_stack(
            [
                result["truth_tire_dynamic_force"],
                result["ideal_tire_dynamic_force_hat"],
                result["saturated_tire_dynamic_force_hat"],
            ]
        ),
        labels=("truth", "ideal sign", "saturation"),
        ylabel="Tire dynamic force [N]",
        title="STO noisy measurement: ideal sign vs saturation",
        save_path=logger.run_dir / "figures" / "force_tracking_noise_saturation.png",
    )
    plot_timeseries(
        result["t"],
        np.column_stack([result["ideal_force_error"], result["saturated_force_error"]]),
        labels=("ideal sign", "saturation"),
        ylabel="Force error [N]",
        title="STO noisy measurement force error",
        save_path=logger.run_dir / "figures" / "force_error_noise_saturation.png",
    )
    plot_timeseries(
        result["t"],
        np.column_stack([result["ideal_switching"], result["saturated_switching"]]),
        labels=("ideal sign", "saturation"),
        ylabel="Switching signal",
        title="STO switching signal under 20 dB velocity noise",
        save_path=logger.run_dir / "figures" / "switching_signal.png",
    )
    plot_timeseries(
        result["t"],
        np.column_stack([result["states"][:, 3], result["noisy_velocity"]]),
        labels=("clean", "20 dB noisy"),
        ylabel="Unsprung velocity [m/s]",
        title="Unsprung velocity measurement noise",
        save_path=logger.run_dir / "figures" / "velocity_noise.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        t=result["t"],
        truth_tire_dynamic_force=result["truth_tire_dynamic_force"],
        noisy_velocity=result["noisy_velocity"],
        ideal_tire_dynamic_force_hat=result["ideal_tire_dynamic_force_hat"],
        saturated_tire_dynamic_force_hat=result["saturated_tire_dynamic_force_hat"],
        ideal_force_error=result["ideal_force_error"],
        saturated_force_error=result["saturated_force_error"],
        ideal_switching=result["ideal_switching"],
        saturated_switching=result["saturated_switching"],
        epsilon_values=epsilon_sweep["epsilon_values"],
        epsilon_chatter_reduction_ratio=epsilon_sweep["chatter_reduction_ratio"],
        epsilon_rmse_ratio=epsilon_sweep["rmse_ratio"],
        epsilon_correlation=epsilon_sweep["correlation"],
    )
    logger.log("Phase 2.5 STO noise and saturation validation completed.")
    return logger.run_dir


def _run_noisy_observer(
    plant: QuarterCar,
    observer_params: ObserverParams,
    measured_velocity: np.ndarray,
    phi_known: np.ndarray,
    dt: float,
    use_saturation: bool,
) -> dict[str, np.ndarray]:
    sto = STO(observer_params, unsprung_mass=plant.params.m_u, use_saturation=use_saturation)
    sto.reset(v_u_hat=measured_velocity[0], chi_hat=0.0)
    additive_force_hat = np.zeros_like(measured_velocity)
    tire_dynamic_force_hat = np.zeros_like(measured_velocity)
    switching = np.zeros_like(measured_velocity)
    for idx in range(len(measured_velocity)):
        error = sto.v_u_hat - measured_velocity[idx]
        switching[idx] = sto._switching(error)
        additive_force_hat[idx], _ = sto.step(measured_velocity[idx], phi_known[idx], dt)
        tire_dynamic_force_hat[idx] = -additive_force_hat[idx]
    return {
        "additive_force_hat": additive_force_hat,
        "tire_dynamic_force_hat": tire_dynamic_force_hat,
        "switching": switching,
    }


def _chatter_metric(signal: np.ndarray, dt: float) -> float:
    return float(np.mean(np.abs(np.diff(signal))) / dt)


def saturation_epsilon_sweep(
    plant: QuarterCar,
    observer_params: ObserverParams,
    epsilon_values: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    if epsilon_values is None:
        epsilon_values = np.array([0.002, 0.005, 0.01, 0.02, 0.04, 0.08, 0.12])
    chatter_reduction = np.zeros_like(epsilon_values, dtype=float)
    rmse_ratio = np.zeros_like(epsilon_values, dtype=float)
    correlation = np.zeros_like(epsilon_values, dtype=float)
    for idx, epsilon in enumerate(epsilon_values):
        params = ObserverParams(
            lambda_1=observer_params.lambda_1,
            lambda_2=observer_params.lambda_2,
            epsilon=float(epsilon),
            delta_fz_err=observer_params.delta_fz_err,
        )
        result = sto_noise_saturation_validation(plant, params, duration=4.0)
        chatter_reduction[idx] = result["chatter_reduction_ratio"]
        rmse_ratio[idx] = result["saturated_rmse_ratio_after_transient"]
        correlation[idx] = result["saturated_correlation_after_transient"]
    return {
        "epsilon_values": epsilon_values,
        "chatter_reduction_ratio": chatter_reduction,
        "rmse_ratio": rmse_ratio,
        "correlation": correlation,
    }


def _write_epsilon_sweep_csv(path: Path, result: dict[str, np.ndarray]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epsilon", "chatter_reduction_ratio", "rmse_ratio", "correlation"])
        for epsilon, chatter, error, corr in zip(
            result["epsilon_values"],
            result["chatter_reduction_ratio"],
            result["rmse_ratio"],
            result["correlation"],
        ):
            writer.writerow([epsilon, chatter, error, corr])


def _plot_epsilon_sweep(path: Path, result: dict[str, np.ndarray]) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    eps = result["epsilon_values"]
    ax.plot(eps, result["chatter_reduction_ratio"], marker="o", label="chatter ratio")
    ax.plot(eps, result["rmse_ratio"], marker="o", label="RMSE ratio")
    ax.plot(eps, result["correlation"], marker="o", label="correlation")
    ax.set_xscale("log")
    ax.set_xlabel("Saturation epsilon")
    ax.set_ylabel("Metric")
    ax.set_title("STO saturation epsilon sweep")
    ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    run_dir = run_phase_2_5()
    print(run_dir)


if __name__ == "__main__":
    main()

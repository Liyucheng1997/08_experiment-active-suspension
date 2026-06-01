from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.inputs.road import iso8608
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.plants.carsim_fmu import CarSimFmuPlant
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.scenarios.carsim_passthrough import DEFAULT_FMU_PATH
from risk_aware_active_suspension.utils.config import ObserverParams, from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


def run_phase_7_3_sto_validation(
    fmu_path: str | Path = DEFAULT_FMU_PATH,
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    observer_config: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
    duration: float = 6.0,
    sample_time: float = 0.005,
    speed_kmh: float = 80.0,
    road_class: str = "B",
    road_scale: float = 0.5,
    mu: float = 0.9,
) -> Path:
    """Run Phase 7.3 STO validation against the CarSim FMU."""

    vehicle_config = Path(vehicle_config)
    observer_config = Path(observer_config)
    vehicle = from_yaml(vehicle_config)
    full_car = FullCar(vehicle)
    base_observer = observer_from_yaml(observer_config)
    observer = ObserverParams(
        lambda_1=50.0,
        lambda_2=500.0,
        epsilon=base_observer.epsilon,
        delta_fz_err=base_observer.delta_fz_err,
    )
    carsim = CarSimFmuPlant(Path(fmu_path))
    logger = RunLogger.create(results_root, phase="phase-7.3", step="carsim", descriptor="sto-validation")
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(observer_config, logger.run_dir / "observer_config.yaml")

    t = np.arange(0.0, duration + 0.5 * sample_time, sample_time)
    roads = _build_four_wheel_road(t, vehicle.wheelbase, speed_kmh / 3.6, road_class, road_scale)
    sim = _run_carsim(carsim, t, roads, speed_kmh, mu, sample_time)
    sto = _run_sto(full_car, observer, sim, sample_time)
    valid = t >= 1.0
    truth_dynamic = sim["fz_n"] - np.mean(sim["fz_n"][valid], axis=0)
    estimate_dynamic = sto["fz_dynamic_hat_n"] - np.mean(sto["fz_dynamic_hat_n"][valid], axis=0)
    fz_hat = np.mean(sim["fz_n"][valid], axis=0)[None, :] + estimate_dynamic

    corr = np.array(
        [
            np.corrcoef(truth_dynamic[valid, idx], estimate_dynamic[valid, idx])[0, 1]
            for idx in range(4)
        ]
    )
    rmse = np.sqrt(np.mean((truth_dynamic[valid] - estimate_dynamic[valid]) ** 2, axis=0))
    truth_peak = np.max(np.abs(truth_dynamic[valid]), axis=0)
    rmse_ratio = rmse / np.maximum(truth_peak, 1.0)

    metrics = {
        "road_class": road_class,
        "road_scale": road_scale,
        "speed_mean_kmh": float(np.mean(sim["vx_kmh"][valid])),
        "min_correlation": float(np.min(corr)),
        "mean_correlation": float(np.mean(corr)),
        "max_rmse_n": float(np.max(rmse)),
        "max_rmse_ratio": float(np.max(rmse_ratio)),
        "max_truth_dynamic_peak_n": float(np.max(truth_peak)),
        "lambda_1": observer.lambda_1,
        "lambda_2": observer.lambda_2,
        "acceptance_min_corr_gt_0p85": int(np.min(corr) > 0.85),
        "acceptance_no_divergence": int(np.all(np.isfinite(estimate_dynamic)) and np.max(np.abs(estimate_dynamic[valid])) < 5000.0),
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)
    for idx, corner in enumerate(CORNER_NAMES):
        logger.log_kv(f"{corner}_correlation", float(corr[idx]))
        logger.log_kv(f"{corner}_rmse_n", float(rmse[idx]))
        logger.log_kv(f"{corner}_rmse_ratio", float(rmse_ratio[idx]))

    np.savez(
        logger.run_dir / "raw.npz",
        t=t,
        roads=roads,
        fz_true_n=sim["fz_n"],
        fz_dynamic_truth_n=truth_dynamic,
        fz_dynamic_hat_n=estimate_dynamic,
        fz_hat_n=fz_hat,
        jounce_m=sim["jounce_m"],
        jounce_rate_m_s=sim["jounce_rate_m_s"],
        wheel_vz_m_s=sim["wheel_vz_m_s"],
        vx_kmh=sim["vx_kmh"],
        corr=corr,
        rmse=rmse,
    )
    _plot_tracking(logger.run_dir / "figures" / "sto_dynamic_fz_tracking.png", t, truth_dynamic, estimate_dynamic)
    _plot_error(logger.run_dir / "figures" / "sto_dynamic_fz_error.png", t, truth_dynamic - estimate_dynamic)
    _plot_inputs(logger.run_dir / "figures" / "sto_inputs.png", t, roads, sim["wheel_vz_m_s"])
    logger.log("Phase 7.3 CarSim STO validation completed.")
    return logger.run_dir


def _build_four_wheel_road(
    t: np.ndarray,
    wheelbase: float,
    speed_m_s: float,
    road_class: str,
    road_scale: float,
) -> np.ndarray:
    front = road_scale * iso8608(road_class, v_x=speed_m_s, t=t, seed=73)
    delay = int(round((wheelbase / speed_m_s) / float(np.median(np.diff(t)))))
    rear = np.roll(front, delay)
    rear[:delay] = 0.0
    return np.column_stack([front, front, rear, rear])


def _run_carsim(
    plant: CarSimFmuPlant,
    t: np.ndarray,
    roads: np.ndarray,
    speed_kmh: float,
    mu: float,
    sample_time: float,
) -> dict[str, np.ndarray]:
    outputs = (
        "Vx",
        "Fz_L1",
        "Fz_R1",
        "Fz_L2",
        "Fz_R2",
        "Jnc_L1",
        "Jnc_R1",
        "Jnc_L2",
        "Jnc_R2",
        "JncR_L1",
        "JncR_R1",
        "JncR_L2",
        "JncR_R2",
        "Vz_WC_L1",
        "Vz_WC_R1",
        "Vz_WC_L2",
        "Vz_WC_R2",
    )
    result = plant.simulate(
        t,
        speed=speed_kmh,
        road_height=roads,
        active_force=0.0,
        brake_pressure=0.0,
        steer=0.0,
        mu_x=mu,
        mu_y=mu,
        output_interval=sample_time,
        outputs=outputs,
    )
    return {
        "time": result["time"],
        "vx_kmh": result["Vx"],
        "fz_n": plant.collect_corners(result, plant.signal_map.fz),
        "jounce_m": plant.collect_corners(result, plant.signal_map.jounce) / 1000.0,
        "jounce_rate_m_s": plant.collect_corners(result, plant.signal_map.jounce_rate) / 1000.0,
        "wheel_vz_m_s": plant.collect_corners(result, plant.signal_map.wheel_vz) / 3.6,
    }


def _run_sto(
    plant: FullCar,
    observer_params: ObserverParams,
    sim: dict[str, np.ndarray],
    dt: float,
) -> dict[str, np.ndarray]:
    observers = [
        STO(observer_params, unsprung_mass=plant.params.m_u, use_saturation=True)
        for _ in range(4)
    ]
    for idx, observer in enumerate(observers):
        observer.reset(v_u_hat=sim["wheel_vz_m_s"][0, idx], chi_hat=0.0)

    estimate = np.zeros_like(sim["fz_n"])
    for time_idx in range(len(sim["time"])):
        for corner_idx, observer in enumerate(observers):
            # CarSim jounce is positive in compression. The STO residual sign
            # below is calibrated against CarSim's Fz convention for this FMU.
            phi_known = -(
                plant.corner_k[corner_idx] * sim["jounce_m"][time_idx, corner_idx]
                + plant.corner_c[corner_idx] * sim["jounce_rate_m_s"][time_idx, corner_idx]
            ) / plant.params.m_u
            additive_force_hat, _ = observer.step(
                v_u_meas=sim["wheel_vz_m_s"][time_idx, corner_idx],
                phi_known=phi_known,
                dt=dt,
            )
            estimate[time_idx, corner_idx] = additive_force_hat
    return {"fz_dynamic_hat_n": estimate}


def _plot_tracking(path: Path, t: np.ndarray, truth: np.ndarray, estimate: np.ndarray) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.0), sharex=True)
    for idx, ax in enumerate(axes):
        ax.plot(t, truth[:, idx], label="CarSim truth")
        ax.plot(t, estimate[:, idx], linestyle="--", label="STO estimate")
        ax.set_ylabel(f"{CORNER_NAMES[idx]} [N]")
        ax.legend(fontsize=7)
    axes[-1].set_xlabel("Time [s]")
    fig.suptitle("CarSim STO dynamic Fz tracking")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_error(path: Path, t: np.ndarray, error: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(t, error)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Dynamic Fz error [N]")
    ax.set_title("CarSim STO dynamic Fz error")
    ax.legend(CORNER_NAMES, ncol=4, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_inputs(path: Path, t: np.ndarray, roads: np.ndarray, wheel_vz: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.0), sharex=True)
    axes[0].plot(t, roads)
    axes[0].set_ylabel("Road [m]")
    axes[0].legend(CORNER_NAMES, ncol=4, fontsize=8)
    axes[1].plot(t, wheel_vz)
    axes[1].set_ylabel("Wheel Vz [m/s]")
    axes[1].set_xlabel("Time [s]")
    axes[1].legend(CORNER_NAMES, ncol=4, fontsize=8)
    fig.suptitle("CarSim STO inputs")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    print(run_phase_7_3_sto_validation())


if __name__ == "__main__":
    main()

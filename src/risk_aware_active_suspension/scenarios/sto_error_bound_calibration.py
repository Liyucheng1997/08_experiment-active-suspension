from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.scenarios.sto_noise_saturation_validation import sto_noise_saturation_validation
from risk_aware_active_suspension.utils.config import ObserverParams, from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


def error_bound_calibration(
    plant: QuarterCar,
    observer_params: ObserverParams,
    snr_values: np.ndarray | None = None,
    dt_values: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    if snr_values is None:
        snr_values = np.array([15.0, 20.0, 25.0])
    if dt_values is None:
        dt_values = np.array([0.0005, 0.001, 0.002])

    rows: list[dict[str, float]] = []
    for snr_db in snr_values:
        for dt in dt_values:
            result = sto_noise_saturation_validation(
                plant,
                observer_params,
                duration=4.0,
                dt=float(dt),
                snr_db=float(snr_db),
            )
            post = result["t"] >= 0.5
            abs_error = np.abs(result["saturated_force_error"][post])
            bound_95 = float(np.percentile(abs_error, 95))
            bound_99 = float(np.percentile(abs_error, 99))
            max_abs = float(np.max(abs_error))
            rows.append(
                {
                    "snr_db": float(snr_db),
                    "dt_s": float(dt),
                    "noise_std_m_s": float(result["velocity_noise_std"]),
                    "rmse_n": float(result["saturated_rmse_after_transient_n"]),
                    "bound_95_n": bound_95,
                    "bound_99_n": bound_99,
                    "max_abs_error_n": max_abs,
                    "coverage_99": float(np.mean(abs_error <= bound_99)),
                    "predicted_bound_n": bound_99,
                }
            )

    predicted = np.array([row["predicted_bound_n"] for row in rows])
    measured = np.array([row["bound_99_n"] for row in rows])
    relative_error = np.abs(predicted - measured) / np.maximum(measured, 1e-9)
    return {
        "rows": rows,
        "snr_values": snr_values,
        "dt_values": dt_values,
        "predicted_bounds_n": predicted,
        "measured_bounds_n": measured,
        "relative_error": relative_error,
        "default_bound_n": float(
            next(row["predicted_bound_n"] for row in rows if row["snr_db"] == 20.0 and row["dt_s"] == 0.0005)
        ),
        "worst_case_bound_n": float(np.max(predicted)),
        "max_relative_error": float(np.max(relative_error)),
    }


def run_phase_2_6(
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
    logger = RunLogger.create(results_root, phase="phase-2.6", step="sto", descriptor="error-bound-calibration")
    shutil.copy2(vehicle_config_path, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(observer_config_path, logger.run_dir / "observer_config.yaml")

    result = error_bound_calibration(plant, observer_params)
    metrics = {
        "default_delta_fz_err_n": result["default_bound_n"],
        "worst_case_delta_fz_err_n": result["worst_case_bound_n"],
        "max_predicted_vs_measured_relative_error": result["max_relative_error"],
        "real_road_lambda_1": observer_params.lambda_1,
        "real_road_lambda_2": observer_params.lambda_2,
        "saturation_epsilon": observer_params.epsilon,
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    _write_lookup_csv(logger.run_dir / "error_bound_lookup.csv", result["rows"])
    _plot_predicted_vs_measured(
        logger.run_dir / "figures" / "predicted_vs_measured_bound.png",
        result["measured_bounds_n"],
        result["predicted_bounds_n"],
    )
    _plot_bound_surface(
        logger.run_dir / "figures" / "error_bound_lookup.png",
        result["rows"],
        result["snr_values"],
        result["dt_values"],
    )
    np.savez(
        logger.run_dir / "raw.npz",
        snr_values=result["snr_values"],
        dt_values=result["dt_values"],
        predicted_bounds_n=result["predicted_bounds_n"],
        measured_bounds_n=result["measured_bounds_n"],
        relative_error=result["relative_error"],
    )
    logger.log("Phase 2.6 STO error-bound calibration completed.")
    return logger.run_dir


def _write_lookup_csv(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "snr_db",
        "dt_s",
        "noise_std_m_s",
        "rmse_n",
        "bound_95_n",
        "bound_99_n",
        "max_abs_error_n",
        "coverage_99",
        "predicted_bound_n",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_predicted_vs_measured(path: Path, measured: np.ndarray, predicted: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(measured, predicted)
    limit = max(float(np.max(measured)), float(np.max(predicted))) * 1.05
    ax.plot([0.0, limit], [0.0, limit], "k--", linewidth=1.0)
    ax.set_xlim(0.0, limit)
    ax.set_ylim(0.0, limit)
    ax.set_xlabel("Measured 99% |F_z error| [N]")
    ax.set_ylabel("Predicted bound [N]")
    ax.set_title("STO error-bound calibration")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_bound_surface(
    path: Path,
    rows: list[dict[str, float]],
    snr_values: np.ndarray,
    dt_values: np.ndarray,
) -> None:
    grid = np.zeros((len(snr_values), len(dt_values)))
    for row in rows:
        i = int(np.where(snr_values == row["snr_db"])[0][0])
        j = int(np.where(dt_values == row["dt_s"])[0][0])
        grid[i, j] = row["predicted_bound_n"]

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    image = ax.imshow(grid, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(dt_values)), labels=[f"{1000.0 * v:g}" for v in dt_values])
    ax.set_yticks(np.arange(len(snr_values)), labels=[f"{v:g}" for v in snr_values])
    ax.set_xlabel("Sample time [ms]")
    ax.set_ylabel("Velocity SNR [dB]")
    ax.set_title("Predicted ΔF_z error bound [N]")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    run_dir = run_phase_2_6()
    print(run_dir)


if __name__ == "__main__":
    main()

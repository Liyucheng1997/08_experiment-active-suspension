from __future__ import annotations

import csv
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.scenarios.sto_synthetic_disturbance_validation import (
    constant_disturbance_validation,
    sinusoidal_disturbance_validation,
)
from risk_aware_active_suspension.utils.config import ObserverParams, from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


def gain_sweep_validation(
    unsprung_mass: float,
    lambda_1_values: np.ndarray | None = None,
    lambda_2_values: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    if lambda_1_values is None:
        lambda_1_values = np.array([5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0])
    if lambda_2_values is None:
        lambda_2_values = np.array([25.0, 50.0, 75.0, 100.0, 150.0, 200.0, 300.0, 500.0])

    sinusoidal_rmse_ratio = np.full((len(lambda_1_values), len(lambda_2_values)), np.nan)
    constant_settling = np.full_like(sinusoidal_rmse_ratio, np.nan)
    constant_rmse_ratio = np.full_like(sinusoidal_rmse_ratio, np.nan)
    feasible = np.zeros_like(sinusoidal_rmse_ratio, dtype=bool)

    for i, lambda_1 in enumerate(lambda_1_values):
        for j, lambda_2 in enumerate(lambda_2_values):
            params = ObserverParams(lambda_1=float(lambda_1), lambda_2=float(lambda_2))
            constant = constant_disturbance_validation(unsprung_mass, params)
            sinusoidal = sinusoidal_disturbance_validation(unsprung_mass, params)
            constant_settling[i, j] = constant["settling_time_s"]
            constant_rmse_ratio[i, j] = constant["steady_state_rmse_ratio"]
            sinusoidal_rmse_ratio[i, j] = sinusoidal["steady_state_rmse_ratio"]
            feasible[i, j] = (
                constant_settling[i, j] < 0.1
                and constant_rmse_ratio[i, j] < 0.01
                and sinusoidal_rmse_ratio[i, j] < 0.05
            )

    objective = sinusoidal_rmse_ratio + 0.1 * np.nan_to_num(constant_settling, nan=10.0, posinf=10.0)
    masked_objective = np.where(feasible, objective, np.inf)
    best_flat = int(np.argmin(masked_objective))
    best_i, best_j = np.unravel_index(best_flat, masked_objective.shape)
    if not np.isfinite(masked_objective[best_i, best_j]):
        best_i, best_j = np.unravel_index(int(np.nanargmin(objective)), objective.shape)

    return {
        "lambda_1_values": lambda_1_values,
        "lambda_2_values": lambda_2_values,
        "sinusoidal_rmse_ratio": sinusoidal_rmse_ratio,
        "constant_settling_s": constant_settling,
        "constant_rmse_ratio": constant_rmse_ratio,
        "feasible": feasible,
        "best_lambda_1": float(lambda_1_values[best_i]),
        "best_lambda_2": float(lambda_2_values[best_j]),
        "best_sinusoidal_rmse_ratio": float(sinusoidal_rmse_ratio[best_i, best_j]),
        "best_constant_settling_s": float(constant_settling[best_i, best_j]),
        "best_constant_rmse_ratio": float(constant_rmse_ratio[best_i, best_j]),
        "feasible_count": int(np.count_nonzero(feasible)),
    }


def run_phase_2_3(
    vehicle_config_path: str | Path = "configs/vehicle_default.yaml",
    observer_config_path: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    vehicle_config_path = Path(vehicle_config_path)
    observer_config_path = Path(observer_config_path)
    vehicle = from_yaml(vehicle_config_path)
    default_observer = observer_from_yaml(observer_config_path)
    logger = RunLogger.create(results_root, phase="phase-2.3", step="sto", descriptor="gain-sweep")
    shutil.copy2(vehicle_config_path, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(observer_config_path, logger.run_dir / "observer_config.yaml")

    result = gain_sweep_validation(vehicle.m_u)
    lambda_1_values = result["lambda_1_values"]
    lambda_2_values = result["lambda_2_values"]

    default_i = int(np.where(lambda_1_values == default_observer.lambda_1)[0][0])
    default_j = int(np.where(lambda_2_values == default_observer.lambda_2)[0][0])
    metrics = {
        "best_lambda_1": result["best_lambda_1"],
        "best_lambda_2": result["best_lambda_2"],
        "best_sinusoidal_rmse_ratio": result["best_sinusoidal_rmse_ratio"],
        "best_constant_settling_s": result["best_constant_settling_s"],
        "best_constant_rmse_ratio": result["best_constant_rmse_ratio"],
        "feasible_count": result["feasible_count"],
        "default_lambda_1": default_observer.lambda_1,
        "default_lambda_2": default_observer.lambda_2,
        "default_sinusoidal_rmse_ratio": result["sinusoidal_rmse_ratio"][default_i, default_j],
        "default_constant_settling_s": result["constant_settling_s"][default_i, default_j],
        "default_constant_rmse_ratio": result["constant_rmse_ratio"][default_i, default_j],
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    _write_sweep_csv(logger.run_dir / "gain_sweep.csv", result)
    _plot_heatmap(
        logger.run_dir / "figures" / "sinusoidal_rmse_heatmap.png",
        lambda_1_values,
        lambda_2_values,
        result["sinusoidal_rmse_ratio"],
        "Sinusoidal RMSE ratio",
    )
    _plot_heatmap(
        logger.run_dir / "figures" / "constant_settling_heatmap.png",
        lambda_1_values,
        lambda_2_values,
        result["constant_settling_s"],
        "Constant settling time [s]",
    )
    _plot_heatmap(
        logger.run_dir / "figures" / "feasible_region.png",
        lambda_1_values,
        lambda_2_values,
        result["feasible"].astype(float),
        "Feasible region",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        lambda_1_values=lambda_1_values,
        lambda_2_values=lambda_2_values,
        sinusoidal_rmse_ratio=result["sinusoidal_rmse_ratio"],
        constant_settling_s=result["constant_settling_s"],
        constant_rmse_ratio=result["constant_rmse_ratio"],
        feasible=result["feasible"],
    )
    logger.log("Phase 2.3 STO gain-sweep validation completed.")
    return logger.run_dir


def _write_sweep_csv(path: Path, result: dict[str, np.ndarray | float]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "lambda_1",
                "lambda_2",
                "constant_settling_s",
                "constant_rmse_ratio",
                "sinusoidal_rmse_ratio",
                "feasible",
            ]
        )
        for i, lambda_1 in enumerate(result["lambda_1_values"]):
            for j, lambda_2 in enumerate(result["lambda_2_values"]):
                writer.writerow(
                    [
                        lambda_1,
                        lambda_2,
                        result["constant_settling_s"][i, j],
                        result["constant_rmse_ratio"][i, j],
                        result["sinusoidal_rmse_ratio"][i, j],
                        bool(result["feasible"][i, j]),
                    ]
                )


def _plot_heatmap(
    path: Path,
    lambda_1_values: np.ndarray,
    lambda_2_values: np.ndarray,
    values: np.ndarray,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    image = ax.imshow(values, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(lambda_2_values)), labels=[f"{v:g}" for v in lambda_2_values])
    ax.set_yticks(np.arange(len(lambda_1_values)), labels=[f"{v:g}" for v in lambda_1_values])
    ax.set_xlabel("lambda_2")
    ax.set_ylabel("lambda_1")
    ax.set_title(title)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    run_dir = run_phase_2_3()
    print(run_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.utils.config import from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_side_by_side


def load_transfer_validation(plant: FullCar, a_x: float = 5.0, a_y: float = 5.0) -> dict[str, float | np.ndarray]:
    static = plant.static_loads()
    longitudinal = plant.normal_loads_quasi_static(a_x=a_x, a_y=0.0)
    lateral = plant.normal_loads_quasi_static(a_x=0.0, a_y=a_y)
    p = plant.params
    wheelbase = p.l_f + p.l_r

    front_delta = float(np.sum(longitudinal[0:2]) - np.sum(static[0:2]))
    rear_delta = float(np.sum(longitudinal[2:4]) - np.sum(static[2:4]))
    expected_front_delta = -p.m * p.h_g * a_x / wheelbase
    left_delta = float(np.sum(lateral[[0, 2]]) - np.sum(static[[0, 2]]))
    right_delta = float(np.sum(lateral[[1, 3]]) - np.sum(static[[1, 3]]))
    expected_left_delta = p.m * p.h_g * a_y / p.t

    return {
        "static_loads": static,
        "longitudinal_loads": longitudinal,
        "lateral_loads": lateral,
        "front_delta": front_delta,
        "rear_delta": rear_delta,
        "expected_front_delta": float(expected_front_delta),
        "front_delta_rel_error": abs(front_delta - expected_front_delta) / abs(expected_front_delta),
        "left_delta": left_delta,
        "right_delta": right_delta,
        "expected_left_delta": float(expected_left_delta),
        "left_delta_rel_error": abs(left_delta - expected_left_delta) / abs(expected_left_delta),
    }


def run_phase_1_5(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    config_path = Path(config_path)
    plant = FullCar(from_yaml(config_path))
    logger = RunLogger.create(results_root, phase="phase-1.5", step="full-car", descriptor="load-transfer")
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    result = load_transfer_validation(plant)
    metrics = {
        "front_delta_n": result["front_delta"],
        "front_expected_delta_n": result["expected_front_delta"],
        "front_delta_rel_error": result["front_delta_rel_error"],
        "rear_delta_n": result["rear_delta"],
        "left_delta_n": result["left_delta"],
        "left_expected_delta_n": result["expected_left_delta"],
        "left_delta_rel_error": result["left_delta_rel_error"],
        "right_delta_n": result["right_delta"],
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    corners = np.arange(4)
    plot_side_by_side(
        corners,
        result["static_loads"],
        result["longitudinal_loads"],
        left_label="Static F_z [N]",
        right_label="a_x=5 m/s^2 F_z [N]",
        title="Longitudinal load transfer",
        save_path=logger.run_dir / "figures" / "longitudinal_load_transfer.png",
    )
    plot_side_by_side(
        corners,
        result["static_loads"],
        result["lateral_loads"],
        left_label="Static F_z [N]",
        right_label="a_y=5 m/s^2 F_z [N]",
        title="Lateral load transfer",
        save_path=logger.run_dir / "figures" / "lateral_load_transfer.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        static_loads=result["static_loads"],
        longitudinal_loads=result["longitudinal_loads"],
        lateral_loads=result["lateral_loads"],
    )
    logger.log("Phase 1.5 load-transfer validation completed.")
    return logger.run_dir


def main() -> None:
    run_dir = run_phase_1_5()
    print(run_dir)


if __name__ == "__main__":
    main()

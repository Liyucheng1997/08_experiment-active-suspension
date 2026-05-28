from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.inputs.road import rounded_bump
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.utils.config import from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_side_by_side


def static_load_validation(plant: FullCar, g: float = 9.81) -> dict[str, float | np.ndarray]:
    loads = plant.static_loads(g=g)
    p = plant.params
    wheelbase = p.l_f + p.l_r
    expected_front = p.m * g * p.l_r / wheelbase
    expected_rear = p.m * g * p.l_f / wheelbase
    return {
        "loads": loads,
        "total_load": float(np.sum(loads)),
        "expected_total_load": float(p.m * g),
        "front_load": float(loads[0] + loads[1]),
        "expected_front_load": float(expected_front),
        "rear_load": float(loads[2] + loads[3]),
        "expected_rear_load": float(expected_rear),
        "left_right_imbalance": float(abs((loads[0] + loads[2]) - (loads[1] + loads[3]))),
    }


def decoupling_validation(
    plant: FullCar,
    duration: float = 4.0,
    dt: float = 0.001,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    bump = rounded_bump(t, height=0.01, start=0.5, duration=1.0)

    heave_road = np.column_stack([bump, bump, bump, bump])
    roll_road = np.column_stack([bump, -bump, bump, -bump])
    rear_pitch_scale = -plant.params.l_r / plant.params.l_f
    pitch_road = np.column_stack([bump, bump, rear_pitch_scale * bump, rear_pitch_scale * bump])

    heave = plant.simulate(t, heave_road)
    roll = plant.simulate(t, roll_road)
    pitch = plant.simulate(t, pitch_road)

    return {
        "t": t,
        "bump": bump,
        "heave_states": heave,
        "roll_states": roll,
        "pitch_states": pitch,
        "heave_peak": float(np.max(np.abs(heave[:, 0]))),
        "heave_roll_leakage": float(np.max(np.abs(heave[:, 2]))),
        "heave_pitch_leakage": float(np.max(np.abs(heave[:, 4]))),
        "roll_peak": float(np.max(np.abs(roll[:, 2]))),
        "roll_heave_leakage": float(np.max(np.abs(roll[:, 0]))),
        "roll_pitch_leakage": float(np.max(np.abs(roll[:, 4]))),
        "pitch_peak": float(np.max(np.abs(pitch[:, 4]))),
        "pitch_heave_leakage": float(np.max(np.abs(pitch[:, 0]))),
        "pitch_roll_leakage": float(np.max(np.abs(pitch[:, 2]))),
    }


def run_phase_1_4(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    config_path = Path(config_path)
    plant = FullCar(from_yaml(config_path))
    logger = RunLogger.create(results_root, phase="phase-1.4", step="full-car", descriptor="static-decoupling")
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    static = static_load_validation(plant)
    decoupling = decoupling_validation(plant)
    metrics = {
        "static_total_load_n": static["total_load"],
        "static_expected_total_load_n": static["expected_total_load"],
        "static_total_error_n": abs(static["total_load"] - static["expected_total_load"]),
        "static_front_load_n": static["front_load"],
        "static_expected_front_load_n": static["expected_front_load"],
        "static_rear_load_n": static["rear_load"],
        "static_expected_rear_load_n": static["expected_rear_load"],
        "static_left_right_imbalance_n": static["left_right_imbalance"],
        "heave_peak_m": decoupling["heave_peak"],
        "heave_roll_leakage_rad": decoupling["heave_roll_leakage"],
        "heave_pitch_leakage_rad": decoupling["heave_pitch_leakage"],
        "heave_pitch_leakage_ratio": decoupling["heave_pitch_leakage"] / decoupling["heave_peak"],
        "roll_peak_rad": decoupling["roll_peak"],
        "roll_heave_leakage_m": decoupling["roll_heave_leakage"],
        "roll_pitch_leakage_rad": decoupling["roll_pitch_leakage"],
        "pitch_peak_rad": decoupling["pitch_peak"],
        "pitch_heave_leakage_m": decoupling["pitch_heave_leakage"],
        "pitch_heave_leakage_ratio": decoupling["pitch_heave_leakage"] / decoupling["pitch_peak"],
        "pitch_roll_leakage_rad": decoupling["pitch_roll_leakage"],
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    plot_side_by_side(
        decoupling["t"],
        decoupling["heave_states"][:, 0],
        decoupling["heave_states"][:, 2],
        left_label="Heave input z_s [m]",
        right_label="Roll leakage phi [rad]",
        title="Full-car heave decoupling",
        save_path=logger.run_dir / "figures" / "heave_decoupling.png",
    )
    plot_side_by_side(
        decoupling["t"],
        decoupling["roll_states"][:, 2],
        decoupling["roll_states"][:, 0],
        left_label="Roll input phi [rad]",
        right_label="Heave leakage z_s [m]",
        title="Full-car roll decoupling",
        save_path=logger.run_dir / "figures" / "roll_decoupling.png",
    )
    plot_side_by_side(
        decoupling["t"],
        decoupling["pitch_states"][:, 4],
        decoupling["pitch_states"][:, 0],
        left_label="Pitch input theta [rad]",
        right_label="Heave leakage z_s [m]",
        title="Full-car pitch decoupling",
        save_path=logger.run_dir / "figures" / "pitch_decoupling.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        static_loads=static["loads"],
        t=decoupling["t"],
        bump=decoupling["bump"],
        heave_states=decoupling["heave_states"],
        roll_states=decoupling["roll_states"],
        pitch_states=decoupling["pitch_states"],
    )
    logger.log("Phase 1.4 full-car static-load and decoupling validation completed.")
    return logger.run_dir


def main() -> None:
    run_dir = run_phase_1_4()
    print(run_dir)


if __name__ == "__main__":
    main()

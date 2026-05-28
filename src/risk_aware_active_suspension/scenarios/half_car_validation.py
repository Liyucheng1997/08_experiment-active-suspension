from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.inputs.road import rounded_bump
from risk_aware_active_suspension.plants.half_car import HalfCar
from risk_aware_active_suspension.utils.config import from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_side_by_side


def decoupling_validation(
    plant: HalfCar,
    duration: float = 2.0,
    dt: float = 0.001,
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    road = rounded_bump(t, height=0.01, start=0.2, duration=0.08)

    symmetric = plant.simulate(t, np.column_stack([road, road]))
    anti_symmetric = plant.simulate(t, np.column_stack([road, -road]))
    eigvals = np.linalg.eigvals(plant.A)
    target = plant.analytical_roll_natural_frequency()
    modal_freqs = np.array(sorted(abs(ev) for ev in eigvals if ev.imag > 0.0))
    roll_mode = float(modal_freqs[np.argmin(np.abs(modal_freqs - target))])

    return {
        "t": t,
        "road": road,
        "symmetric_states": symmetric,
        "anti_symmetric_states": anti_symmetric,
        "symmetric_roll_leakage": float(np.max(np.abs(symmetric[:, 2]))),
        "symmetric_heave_peak": float(np.max(np.abs(symmetric[:, 0]))),
        "anti_symmetric_heave_leakage": float(np.max(np.abs(anti_symmetric[:, 0]))),
        "anti_symmetric_roll_peak": float(np.max(np.abs(anti_symmetric[:, 2]))),
        "roll_mode_rad_s": roll_mode,
        "roll_mode_expected_rad_s": target,
        "roll_mode_rel_error": abs(roll_mode - target) / target,
    }


def run_phase_1_3(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    config_path = Path(config_path)
    plant = HalfCar(from_yaml(config_path))
    logger = RunLogger.create(results_root, phase="phase-1.3", step="half-car", descriptor="decoupling")
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    result = decoupling_validation(plant)
    metrics = {
        "symmetric_roll_leakage_rad": result["symmetric_roll_leakage"],
        "symmetric_heave_peak_m": result["symmetric_heave_peak"],
        "anti_symmetric_heave_leakage_m": result["anti_symmetric_heave_leakage"],
        "anti_symmetric_roll_peak_rad": result["anti_symmetric_roll_peak"],
        "roll_mode_rad_s": result["roll_mode_rad_s"],
        "roll_mode_expected_rad_s": result["roll_mode_expected_rad_s"],
        "roll_mode_rel_error": result["roll_mode_rel_error"],
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    plot_side_by_side(
        result["t"],
        result["symmetric_states"][:, 0],
        result["symmetric_states"][:, 2],
        left_label="Symmetric heave z_s [m]",
        right_label="Symmetric roll phi [rad]",
        title="Half-car symmetric road decoupling",
        save_path=logger.run_dir / "figures" / "symmetric_decoupling.pdf",
    )
    plot_side_by_side(
        result["t"],
        result["anti_symmetric_states"][:, 0],
        result["anti_symmetric_states"][:, 2],
        left_label="Anti-symmetric heave z_s [m]",
        right_label="Anti-symmetric roll phi [rad]",
        title="Half-car anti-symmetric road decoupling",
        save_path=logger.run_dir / "figures" / "anti_symmetric_decoupling.pdf",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        t=result["t"],
        road=result["road"],
        symmetric_states=result["symmetric_states"],
        anti_symmetric_states=result["anti_symmetric_states"],
    )
    logger.log("Phase 1.3 half-car decoupling validation completed.")
    return logger.run_dir


def main() -> None:
    run_dir = run_phase_1_3()
    print(run_dir)


if __name__ == "__main__":
    main()

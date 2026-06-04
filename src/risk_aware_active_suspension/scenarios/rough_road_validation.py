from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.inputs.road import iso8608
from risk_aware_active_suspension.metrics.signals import peak_abs, percentile95, rms
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.utils.config import from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def rough_road_validation(
    plant: FullCar,
    duration: float = 60.0,
    dt: float = 0.002,
    v_x: float = 80.0 / 3.6,
    road_class: str = "B",
) -> dict[str, float | np.ndarray]:
    t = np.arange(0.0, duration, dt)
    roads = _straight_two_track_road(plant, t=t, dt=dt, v_x=v_x, road_class=road_class)
    states = plant.simulate(t, roads)
    body_accel = plant.batch_body_accelerations(states, roads)
    strokes = plant.batch_suspension_strokes(states)
    tire_forces = plant.batch_tire_normal_forces(states, roads)
    vertical_accel = body_accel[:, 0]

    return {
        "t": t,
        "roads": roads,
        "states": states,
        "body_accel": body_accel,
        "strokes": strokes,
        "tire_forces": tire_forces,
        "vertical_accel_rms": rms(vertical_accel),
        "vertical_accel_peak_abs": peak_abs(vertical_accel),
        "vertical_accel_p95_abs": percentile95(vertical_accel),
        "stroke_peak_abs": float(np.max(np.abs(strokes))),
        "stroke_p95_abs": float(np.percentile(np.abs(strokes), 95)),
        "min_tire_force": float(np.min(tire_forces)),
        "tire_force_peak_abs_dynamic": float(np.max(np.abs(tire_forces - plant.static_loads()))),
    }


def _straight_two_track_road(
    plant: FullCar,
    t: np.ndarray,
    dt: float,
    v_x: float,
    road_class: str,
) -> np.ndarray:
    delay = (plant.params.l_f + plant.params.l_r) / v_x
    delay_samples = int(round(delay / dt))
    extended_t = np.arange(0.0, t[-1] + delay + (delay_samples + 2) * dt, dt)
    left_track = iso8608(road_class, v_x=v_x, t=extended_t, seed=11)
    right_track = iso8608(road_class, v_x=v_x, t=extended_t, seed=17)
    n = len(t)
    return np.column_stack(
        [
            left_track[delay_samples : delay_samples + n],
            right_track[delay_samples : delay_samples + n],
            left_track[:n],
            right_track[:n],
        ]
    )


def run_phase_1_6(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    config_path = Path(config_path)
    plant = FullCar(from_yaml(config_path))
    logger = RunLogger.create(results_root, phase="phase-1.6", step="full-car", descriptor="class-b-rough-road")
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    result = rough_road_validation(plant)
    metrics = {
        "vertical_accel_rms_m_s2": result["vertical_accel_rms"],
        "vertical_accel_p95_abs_m_s2": result["vertical_accel_p95_abs"],
        "vertical_accel_peak_abs_m_s2": result["vertical_accel_peak_abs"],
        "stroke_p95_abs_m": result["stroke_p95_abs"],
        "stroke_peak_abs_m": result["stroke_peak_abs"],
        "min_tire_force_n": result["min_tire_force"],
        "tire_force_peak_abs_dynamic_n": result["tire_force_peak_abs_dynamic"],
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    plot_timeseries(
        result["t"],
        result["body_accel"][:, 0],
        ylabel="Vertical acceleration [m/s^2]",
        title="Full-car class-B rough-road vertical acceleration",
        save_path=logger.run_dir / "figures" / "vertical_acceleration.png",
    )
    plot_timeseries(
        result["t"],
        result["strokes"],
        labels=CORNER_NAMES,
        ylabel="Suspension stroke [m]",
        title="Full-car class-B rough-road suspension strokes",
        save_path=logger.run_dir / "figures" / "suspension_strokes.png",
    )
    plot_timeseries(
        result["t"],
        result["tire_forces"],
        labels=CORNER_NAMES,
        ylabel="Tire normal force [N]",
        title="Full-car class-B rough-road tire normal forces",
        save_path=logger.run_dir / "figures" / "tire_normal_forces.png",
    )
    plot_timeseries(
        result["t"],
        result["roads"],
        labels=CORNER_NAMES,
        ylabel="Road height [m]",
        title="ISO 8608 class-B road inputs",
        save_path=logger.run_dir / "figures" / "road_inputs.png",
    )
    np.savez(
        logger.run_dir / "raw.npz",
        t=result["t"],
        roads=result["roads"],
        states=result["states"],
        body_accel=result["body_accel"],
        strokes=result["strokes"],
        tire_forces=result["tire_forces"],
    )
    logger.log("Phase 1.6 full-car rough-road validation completed.")
    return logger.run_dir


def main() -> None:
    run_dir = run_phase_1_6()
    print(run_dir)


if __name__ == "__main__":
    main()

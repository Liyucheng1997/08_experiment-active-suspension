from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.controllers.passive import PassiveController
from risk_aware_active_suspension.metrics.signals import peak_abs, percentile95, rmse
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.scenarios.rough_road_validation import (
    _straight_two_track_road,
)
from risk_aware_active_suspension.utils.config import from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def passive_closed_loop(
    plant: FullCar,
    controller: PassiveController,
    duration: float = 60.0,
    dt: float = 0.002,
    v_x: float = 80.0 / 3.6,
    road_class: str = "B",
) -> dict[str, np.ndarray | float]:
    """Run the passive controller in closed loop over a class-B straight road.

    Verifies the controller interface end-to-end and produces the canonical
    passive-baseline metrics that later active controllers will be compared
    against in Phase 3.5.
    """
    t = np.arange(0.0, duration, dt)
    roads = _straight_two_track_road(plant, t=t, dt=dt, v_x=v_x, road_class=road_class)

    states = np.zeros((len(t), 14), dtype=float)
    forces = np.zeros((len(t), 4), dtype=float)
    state = np.zeros(14)
    for k in range(len(t) - 1):
        u = controller.compute(state, t=float(t[k]), road=roads[k])
        forces[k] = u
        state = plant.step(state, roads[k], float(t[k + 1] - t[k]))
        states[k + 1] = state

    open_loop_states = plant.simulate(t, roads)
    bit_identical = bool(np.array_equal(states, open_loop_states))

    body_accel = plant.batch_body_accelerations(states, roads)
    strokes = plant.batch_suspension_strokes(states)
    tire_forces = plant.batch_tire_normal_forces(states, roads)
    vertical_accel = body_accel[:, 0]

    return {
        "t": t,
        "roads": roads,
        "states": states,
        "forces": forces,
        "body_accel": body_accel,
        "strokes": strokes,
        "tire_forces": tire_forces,
        "vertical_accel_rms": rmse(vertical_accel),
        "vertical_accel_peak_abs": peak_abs(vertical_accel),
        "vertical_accel_p95_abs": percentile95(vertical_accel),
        "stroke_peak_abs": float(np.max(np.abs(strokes))),
        "stroke_p95_abs": float(np.percentile(np.abs(strokes), 95)),
        "min_tire_force": float(np.min(tire_forces)),
        "max_actuator_force": float(np.max(np.abs(forces))),
        "bit_identical_to_open_loop": bit_identical,
    }


def run_phase_3_1(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    config_path = Path(config_path)
    plant = FullCar(from_yaml(config_path))
    controller = PassiveController(n_actuators=4)

    logger = RunLogger.create(
        results_root, phase="phase-3.1", step="passive", descriptor="class-b-straight"
    )
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    result = passive_closed_loop(plant, controller)

    metrics = {
        "vertical_accel_rms_m_s2": result["vertical_accel_rms"],
        "vertical_accel_p95_abs_m_s2": result["vertical_accel_p95_abs"],
        "vertical_accel_peak_abs_m_s2": result["vertical_accel_peak_abs"],
        "stroke_p95_abs_m": result["stroke_p95_abs"],
        "stroke_peak_abs_m": result["stroke_peak_abs"],
        "min_tire_force_n": result["min_tire_force"],
        "max_actuator_force_n": result["max_actuator_force"],
        "bit_identical_to_open_loop": int(result["bit_identical_to_open_loop"]),
    }
    for key, value in metrics.items():
        logger.log_kv(key, value)

    if not result["bit_identical_to_open_loop"]:
        raise RuntimeError(
            "Phase 3.1 acceptance failed: closed-loop passive run differs from open-loop plant."
        )

    plot_timeseries(
        result["t"],
        result["body_accel"][:, 0],
        ylabel="Vertical acceleration [m/s^2]",
        title="Passive baseline: vertical acceleration (class-B, 80 km/h)",
        save_path=logger.run_dir / "figures" / "vertical_acceleration.png",
    )
    plot_timeseries(
        result["t"],
        result["forces"],
        labels=CORNER_NAMES,
        ylabel="Actuator force [N]",
        title="Passive baseline: actuator force (must be identically zero)",
        save_path=logger.run_dir / "figures" / "actuator_forces.png",
    )
    plot_timeseries(
        result["t"],
        result["tire_forces"],
        labels=CORNER_NAMES,
        ylabel="Tire normal force [N]",
        title="Passive baseline: tire normal forces",
        save_path=logger.run_dir / "figures" / "tire_normal_forces.png",
    )

    np.savez(
        logger.run_dir / "raw.npz",
        t=result["t"],
        roads=result["roads"],
        states=result["states"],
        forces=result["forces"],
        body_accel=result["body_accel"],
        strokes=result["strokes"],
        tire_forces=result["tire_forces"],
    )
    logger.log("Phase 3.1 passive baseline closed-loop run completed.")
    logger.log("Acceptance: closed-loop states bit-identical to plant-only Phase 1 run.")
    return logger.run_dir


def main() -> None:
    run_dir = run_phase_3_1()
    print(run_dir)


if __name__ == "__main__":
    main()

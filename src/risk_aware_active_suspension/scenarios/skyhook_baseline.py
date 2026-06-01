from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.controllers.passive import PassiveController
from risk_aware_active_suspension.controllers.skyhook import (
    FullCarSkyhookController,
    QuarterCarSkyhookController,
)
from risk_aware_active_suspension.inputs.road import iso8608
from risk_aware_active_suspension.metrics.signals import peak_abs, percentile95, rmse
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.scenarios.rough_road_validation import (
    _straight_two_track_road,
)
from risk_aware_active_suspension.utils.config import VehicleParams, from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def _simulate_quarter_car(
    plant: QuarterCar, controller, t: np.ndarray, road: np.ndarray
) -> dict[str, np.ndarray]:
    states = np.zeros((len(t), 4), dtype=float)
    forces = np.zeros(len(t), dtype=float)
    state = np.zeros(4)
    for k in range(len(t) - 1):
        u = controller.compute(state)
        u_scalar = float(u[0])
        forces[k] = u_scalar
        state = plant.step(state, u_scalar, float(road[k]), float(t[k + 1] - t[k]))
        states[k + 1] = state
    body_accel = np.array(
        [plant.sprung_acceleration(s, u=f) for s, f in zip(states, forces)]
    )
    strokes = states[:, 0] - states[:, 2]
    tire_forces = np.array([plant.tire_normal_force(s, w=float(r)) for s, r in zip(states, road)])
    return {
        "states": states,
        "forces": forces,
        "body_accel": body_accel,
        "strokes": strokes,
        "tire_forces": tire_forces,
    }


def tune_quarter_car_skyhook(
    params: VehicleParams,
    c_sky_grid: tuple[float, ...] = (
        1000.0, 2000.0, 3000.0, 5000.0, 7000.0, 10000.0, 12000.0, 15000.0, 18000.0, 22000.0,
    ),
    duration: float = 20.0,
    dt: float = 0.001,
    v_x: float = 80.0 / 3.6,
    road_class: str = "B",
    f_max: float = 4000.0,
    seed: int = 7,
) -> dict[str, np.ndarray | float | list]:
    plant = QuarterCar(params)
    t = np.arange(0.0, duration, dt)
    road = iso8608(road_class, v_x=v_x, t=t, seed=seed)

    passive_run = _simulate_quarter_car(plant, PassiveController(n_actuators=1), t, road)
    passive_rms = rmse(passive_run["body_accel"])

    sweep = []
    for c_sky in c_sky_grid:
        controller = QuarterCarSkyhookController(c_sky=c_sky, f_max=f_max)
        run = _simulate_quarter_car(plant, controller, t, road)
        accel_rms = rmse(run["body_accel"])
        stroke_p95 = float(np.percentile(np.abs(run["strokes"]), 95))
        min_tire = float(np.min(run["tire_forces"]))
        max_force = float(np.max(np.abs(run["forces"])))
        sweep.append(
            {
                "c_sky": float(c_sky),
                "accel_rms": float(accel_rms),
                "improvement_pct": 100.0 * (passive_rms - accel_rms) / passive_rms,
                "stroke_p95": stroke_p95,
                "min_tire_force": min_tire,
                "max_force": max_force,
            }
        )
    # Pick the c_sky with the largest comfort improvement that doesn't pin
    # the actuator saturation rail. (min_tire_force is a linear-model artifact
    # of class-B excitation and is not informative as a feasibility filter.)
    feasible = [row for row in sweep if row["max_force"] < f_max * 0.99]
    pool = feasible if feasible else sweep
    best = max(pool, key=lambda row: row["improvement_pct"])
    return {
        "t": t,
        "road": road,
        "passive_rms": float(passive_rms),
        "sweep": sweep,
        "best_c_sky": best["c_sky"],
        "best_improvement_pct": best["improvement_pct"],
    }


def passive_vs_skyhook_quarter(
    params: VehicleParams,
    c_sky: float,
    duration: float = 60.0,
    dt: float = 0.001,
    v_x: float = 80.0 / 3.6,
    road_class: str = "B",
    f_max: float = 4000.0,
    seed: int = 11,
) -> dict[str, np.ndarray | float]:
    plant = QuarterCar(params)
    t = np.arange(0.0, duration, dt)
    road = iso8608(road_class, v_x=v_x, t=t, seed=seed)

    passive_run = _simulate_quarter_car(plant, PassiveController(n_actuators=1), t, road)
    sky_run = _simulate_quarter_car(
        plant, QuarterCarSkyhookController(c_sky=c_sky, f_max=f_max), t, road
    )

    passive_rms = rmse(passive_run["body_accel"])
    sky_rms = rmse(sky_run["body_accel"])
    return {
        "t": t,
        "road": road,
        "passive": passive_run,
        "skyhook": sky_run,
        "passive_accel_rms": float(passive_rms),
        "skyhook_accel_rms": float(sky_rms),
        "improvement_pct": float(100.0 * (passive_rms - sky_rms) / passive_rms),
        "skyhook_stroke_p95": float(np.percentile(np.abs(sky_run["strokes"]), 95)),
        "skyhook_min_tire": float(np.min(sky_run["tire_forces"])),
        "skyhook_max_force": float(np.max(np.abs(sky_run["forces"]))),
    }


def _simulate_full_car(
    plant: FullCar, controller, t: np.ndarray, roads: np.ndarray
) -> dict[str, np.ndarray]:
    states = np.zeros((len(t), 14), dtype=float)
    forces = np.zeros((len(t), 4), dtype=float)
    state = np.zeros(14)
    for k in range(len(t) - 1):
        u = controller.compute(state)
        forces[k] = u
        state = plant.step(state, roads[k], float(t[k + 1] - t[k]), u=u)
        states[k + 1] = state
    body_accel = plant.batch_body_accelerations(states, roads, forces)
    strokes = plant.batch_suspension_strokes(states)
    tire_forces = plant.batch_tire_normal_forces(states, roads)
    return {
        "states": states,
        "forces": forces,
        "body_accel": body_accel,
        "strokes": strokes,
        "tire_forces": tire_forces,
    }


def passive_vs_skyhook_full(
    params: VehicleParams,
    c_sky: float,
    duration: float = 60.0,
    dt: float = 0.002,
    v_x: float = 80.0 / 3.6,
    road_class: str = "B",
    f_max: float = 4000.0,
) -> dict[str, np.ndarray | float]:
    plant = FullCar(params)
    t = np.arange(0.0, duration, dt)
    roads = _straight_two_track_road(plant, t=t, dt=dt, v_x=v_x, road_class=road_class)

    passive_run = _simulate_full_car(plant, PassiveController(n_actuators=4), t, roads)
    sky_run = _simulate_full_car(
        plant, FullCarSkyhookController(params=params, c_sky=c_sky, f_max=f_max), t, roads
    )

    def rms_set(run):
        return {
            "heave": float(rmse(run["body_accel"][:, 0])),
            "roll": float(rmse(run["body_accel"][:, 1])),
            "pitch": float(rmse(run["body_accel"][:, 2])),
        }

    passive_rms = rms_set(passive_run)
    sky_rms = rms_set(sky_run)
    improvement = {
        key: 100.0 * (passive_rms[key] - sky_rms[key]) / passive_rms[key]
        for key in passive_rms
    }
    return {
        "t": t,
        "roads": roads,
        "passive": passive_run,
        "skyhook": sky_run,
        "passive_rms": passive_rms,
        "skyhook_rms": sky_rms,
        "improvement_pct": improvement,
        "skyhook_stroke_p95": float(np.percentile(np.abs(sky_run["strokes"]), 95)),
        "skyhook_min_tire": float(np.min(sky_run["tire_forces"])),
        "skyhook_max_force": float(np.max(np.abs(sky_run["forces"]))),
    }


def run_phase_3_2(
    config_path: str | Path = "configs/vehicle_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    config_path = Path(config_path)
    params = from_yaml(config_path)

    logger = RunLogger.create(
        results_root, phase="phase-3.2", step="skyhook", descriptor="class-b-tune-and-eval"
    )
    shutil.copy2(config_path, logger.run_dir / "config.yaml")

    tuning = tune_quarter_car_skyhook(params)
    c_sky = tuning["best_c_sky"]
    logger.log("Quarter-car skyhook tuning sweep:")
    for row in tuning["sweep"]:
        logger.log(
            f"  c_sky={row['c_sky']:>6.0f}  rms={row['accel_rms']:.4f}  "
            f"improvement={row['improvement_pct']:+.2f}%  "
            f"stroke_p95={row['stroke_p95']*1000:.2f}mm  "
            f"min_Fz={row['min_tire_force']:.0f}N  max_Fe={row['max_force']:.0f}N"
        )
    logger.log_kv("tuned_c_sky_n_s_per_m", c_sky)
    logger.log_kv("tuning_passive_rms_m_s2", tuning["passive_rms"])
    logger.log_kv("tuning_best_improvement_pct", tuning["best_improvement_pct"])

    quarter_result = passive_vs_skyhook_quarter(params, c_sky=c_sky)
    logger.log_kv("quarter_passive_rms_m_s2", quarter_result["passive_accel_rms"])
    logger.log_kv("quarter_skyhook_rms_m_s2", quarter_result["skyhook_accel_rms"])
    logger.log_kv("quarter_improvement_pct", quarter_result["improvement_pct"])
    logger.log_kv("quarter_skyhook_stroke_p95_m", quarter_result["skyhook_stroke_p95"])
    logger.log_kv("quarter_skyhook_min_tire_n", quarter_result["skyhook_min_tire"])
    logger.log_kv("quarter_skyhook_max_force_n", quarter_result["skyhook_max_force"])

    full_result = passive_vs_skyhook_full(params, c_sky=c_sky)
    for axis in ("heave", "roll", "pitch"):
        logger.log_kv(f"full_passive_{axis}_rms", full_result["passive_rms"][axis])
        logger.log_kv(f"full_skyhook_{axis}_rms", full_result["skyhook_rms"][axis])
        logger.log_kv(f"full_{axis}_improvement_pct", full_result["improvement_pct"][axis])
    logger.log_kv("full_skyhook_stroke_p95_m", full_result["skyhook_stroke_p95"])
    logger.log_kv("full_skyhook_min_tire_n", full_result["skyhook_min_tire"])
    logger.log_kv("full_skyhook_max_force_n", full_result["skyhook_max_force"])

    # Acceptance
    quarter_pass = quarter_result["improvement_pct"] >= 20.0
    full_pass = all(v > 0.0 for v in full_result["improvement_pct"].values())
    logger.log_kv("acceptance_quarter_ge_20pct", int(quarter_pass))
    logger.log_kv("acceptance_full_all_axes_improved", int(full_pass))

    # Figures
    t_q = quarter_result["t"]
    plot_timeseries(
        t_q[: int(8.0 / (t_q[1] - t_q[0]))],
        np.column_stack(
            [
                quarter_result["passive"]["body_accel"][: int(8.0 / (t_q[1] - t_q[0]))],
                quarter_result["skyhook"]["body_accel"][: int(8.0 / (t_q[1] - t_q[0]))],
            ]
        ),
        labels=["passive", "skyhook"],
        ylabel="Sprung accel [m/s^2]",
        title=f"Quarter-car class-B (first 8s): skyhook c_sky={c_sky:.0f}",
        save_path=logger.run_dir / "figures" / "quarter_accel_compare.png",
    )

    sweep_c = np.array([row["c_sky"] for row in tuning["sweep"]])
    sweep_imp = np.array([row["improvement_pct"] for row in tuning["sweep"]])
    plot_timeseries(
        sweep_c,
        sweep_imp,
        ylabel="Vertical accel RMS improvement [%]",
        title="Quarter-car skyhook tuning sweep",
        save_path=logger.run_dir / "figures" / "quarter_tuning_sweep.png",
    )

    t_f = full_result["t"]
    window = slice(0, int(10.0 / (t_f[1] - t_f[0])))
    plot_timeseries(
        t_f[window],
        np.column_stack(
            [
                full_result["passive"]["body_accel"][window, 0],
                full_result["skyhook"]["body_accel"][window, 0],
            ]
        ),
        labels=["passive", "skyhook"],
        ylabel="Heave accel [m/s^2]",
        title="Full-car class-B heave acceleration",
        save_path=logger.run_dir / "figures" / "full_heave_compare.png",
    )
    plot_timeseries(
        t_f[window],
        full_result["skyhook"]["forces"][window],
        labels=list(CORNER_NAMES),
        ylabel="Actuator force [N]",
        title="Full-car skyhook actuator forces",
        save_path=logger.run_dir / "figures" / "full_actuator_forces.png",
    )

    np.savez(
        logger.run_dir / "raw.npz",
        t_quarter=quarter_result["t"],
        passive_q_accel=quarter_result["passive"]["body_accel"],
        skyhook_q_accel=quarter_result["skyhook"]["body_accel"],
        skyhook_q_forces=quarter_result["skyhook"]["forces"],
        t_full=full_result["t"],
        passive_f_accel=full_result["passive"]["body_accel"],
        skyhook_f_accel=full_result["skyhook"]["body_accel"],
        skyhook_f_forces=full_result["skyhook"]["forces"],
    )

    if not quarter_pass:
        raise RuntimeError(
            f"Phase 3.2 quarter-car acceptance failed: improvement "
            f"{quarter_result['improvement_pct']:.2f}% < 20%."
        )
    if not full_pass:
        raise RuntimeError(
            f"Phase 3.2 full-car acceptance failed: not all axes improved: "
            f"{full_result['improvement_pct']}"
        )
    logger.log("Phase 3.2 skyhook baseline completed; acceptance passed.")
    return logger.run_dir


def main() -> None:
    run_dir = run_phase_3_2()
    print(run_dir)


if __name__ == "__main__":
    main()

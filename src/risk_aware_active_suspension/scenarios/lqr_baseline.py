from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.controllers.lqr import (
    FullCarLQRController,
    QuarterCarLQRController,
)
from risk_aware_active_suspension.controllers.passive import PassiveController
from risk_aware_active_suspension.controllers.skyhook import (
    FullCarSkyhookController,
    QuarterCarSkyhookController,
)
from risk_aware_active_suspension.inputs.road import iso8608
from risk_aware_active_suspension.metrics.signals import rmse
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.scenarios.rough_road_validation import (
    _straight_two_track_road,
)
from risk_aware_active_suspension.scenarios.skyhook_baseline import (
    _simulate_full_car,
    _simulate_quarter_car,
    tune_quarter_car_skyhook,
)
from risk_aware_active_suspension.utils.config import (
    LQRParams,
    VehicleParams,
    from_yaml,
    lqr_from_yaml,
)
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def compare_quarter_car(
    params: VehicleParams,
    lqr: LQRParams,
    c_sky: float,
    duration: float = 60.0,
    dt: float = 0.001,
    v_x: float = 80.0 / 3.6,
    road_class: str = "B",
    seed: int = 11,
) -> dict[str, object]:
    plant = QuarterCar(params)
    t = np.arange(0.0, duration, dt)
    road = iso8608(road_class, v_x=v_x, t=t, seed=seed)

    passive_run = _simulate_quarter_car(plant, PassiveController(n_actuators=1), t, road)
    skyhook = QuarterCarSkyhookController(c_sky=c_sky, f_max=lqr.f_max)
    sky_run = _simulate_quarter_car(plant, skyhook, t, road)
    lqr_ctrl = QuarterCarLQRController(params, lqr)
    lqr_run = _simulate_quarter_car(plant, lqr_ctrl, t, road)

    return {
        "t": t,
        "passive_rms": float(rmse(passive_run["body_accel"])),
        "skyhook_rms": float(rmse(sky_run["body_accel"])),
        "lqr_rms": float(rmse(lqr_run["body_accel"])),
        "lqr_max_force": float(np.max(np.abs(lqr_run["forces"]))),
        "lqr_stroke_p95": float(np.percentile(np.abs(lqr_run["strokes"]), 95)),
        "passive_run": passive_run,
        "skyhook_run": sky_run,
        "lqr_run": lqr_run,
        "lqr_poles_max_abs": float(np.max(np.abs(lqr_ctrl.closed_loop_poles))),
    }


def compare_full_car(
    params: VehicleParams,
    lqr: LQRParams,
    c_sky: float,
    duration: float = 60.0,
    dt: float = 0.002,
    v_x: float = 80.0 / 3.6,
    road_class: str = "B",
) -> dict[str, object]:
    plant = FullCar(params)
    t = np.arange(0.0, duration, dt)
    roads = _straight_two_track_road(plant, t=t, dt=dt, v_x=v_x, road_class=road_class)

    passive_run = _simulate_full_car(plant, PassiveController(n_actuators=4), t, roads)
    sky_ctrl = FullCarSkyhookController(params=params, c_sky=c_sky, f_max=lqr.f_max)
    sky_run = _simulate_full_car(plant, sky_ctrl, t, roads)
    lqr_ctrl = FullCarLQRController(params, lqr)
    lqr_run = _simulate_full_car(plant, lqr_ctrl, t, roads)

    def rms_set(run):
        return {
            "heave": float(rmse(run["body_accel"][:, 0])),
            "roll": float(rmse(run["body_accel"][:, 1])),
            "pitch": float(rmse(run["body_accel"][:, 2])),
        }

    passive_rms = rms_set(passive_run)
    sky_rms = rms_set(sky_run)
    lqr_rms = rms_set(lqr_run)
    return {
        "t": t,
        "roads": roads,
        "passive_rms": passive_rms,
        "skyhook_rms": sky_rms,
        "lqr_rms": lqr_rms,
        "lqr_max_force": float(np.max(np.abs(lqr_run["forces"]))),
        "lqr_stroke_p95": float(np.percentile(np.abs(lqr_run["strokes"]), 95)),
        "passive_run": passive_run,
        "skyhook_run": sky_run,
        "lqr_run": lqr_run,
        "lqr_poles_max_abs": float(np.max(np.abs(lqr_ctrl.closed_loop_poles))),
    }


def run_phase_3_3(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    results_root: str | Path = "results",
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    params = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)

    logger = RunLogger.create(
        results_root, phase="phase-3.3", step="lqr", descriptor="class-b-compare-skyhook"
    )
    shutil.copy2(vehicle_config, logger.run_dir / "config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")

    # Reuse the Phase 3.2 tuned c_sky so the comparison is apples-to-apples.
    tuning = tune_quarter_car_skyhook(params)
    c_sky = float(tuning["best_c_sky"])
    logger.log_kv("skyhook_c_sky", c_sky)

    # Quarter-car comparison
    q = compare_quarter_car(params, lqr, c_sky=c_sky)
    logger.log_kv("quarter_passive_rms", q["passive_rms"])
    logger.log_kv("quarter_skyhook_rms", q["skyhook_rms"])
    logger.log_kv("quarter_lqr_rms", q["lqr_rms"])
    q_improve = 100.0 * (q["skyhook_rms"] - q["lqr_rms"]) / q["skyhook_rms"]
    logger.log_kv("quarter_lqr_vs_skyhook_pct", q_improve)
    logger.log_kv("quarter_lqr_max_force_n", q["lqr_max_force"])
    logger.log_kv("quarter_lqr_stroke_p95_m", q["lqr_stroke_p95"])
    logger.log_kv("quarter_lqr_poles_max_abs", q["lqr_poles_max_abs"])

    # Full-car comparison
    f = compare_full_car(params, lqr, c_sky=c_sky)
    for axis in ("heave", "roll", "pitch"):
        logger.log_kv(f"full_passive_{axis}_rms", f["passive_rms"][axis])
        logger.log_kv(f"full_skyhook_{axis}_rms", f["skyhook_rms"][axis])
        logger.log_kv(f"full_lqr_{axis}_rms", f["lqr_rms"][axis])
        logger.log_kv(
            f"full_lqr_vs_skyhook_{axis}_pct",
            100.0 * (f["skyhook_rms"][axis] - f["lqr_rms"][axis]) / f["skyhook_rms"][axis],
        )
    logger.log_kv("full_lqr_max_force_n", f["lqr_max_force"])
    logger.log_kv("full_lqr_stroke_p95_m", f["lqr_stroke_p95"])
    logger.log_kv("full_lqr_poles_max_abs", f["lqr_poles_max_abs"])

    # Acceptance
    poles_ok = q["lqr_poles_max_abs"] < 1.0 and f["lqr_poles_max_abs"] < 1.0
    heave_improve = (
        100.0 * (f["skyhook_rms"]["heave"] - f["lqr_rms"]["heave"]) / f["skyhook_rms"]["heave"]
    )
    quarter_improve = q_improve
    heave_pass = heave_improve >= 10.0
    quarter_pass = quarter_improve >= 10.0
    logger.log_kv("acceptance_poles_inside_unit_circle", int(poles_ok))
    logger.log_kv("acceptance_full_heave_vs_skyhook_ge_10pct", int(heave_pass))
    logger.log_kv("acceptance_quarter_vs_skyhook_ge_10pct", int(quarter_pass))

    # Figures
    t_q = q["t"]
    n_win = int(8.0 / (t_q[1] - t_q[0]))
    plot_timeseries(
        t_q[:n_win],
        np.column_stack(
            [
                q["passive_run"]["body_accel"][:n_win],
                q["skyhook_run"]["body_accel"][:n_win],
                q["lqr_run"]["body_accel"][:n_win],
            ]
        ),
        labels=["passive", "skyhook", "lqr"],
        ylabel="Sprung accel [m/s^2]",
        title="Quarter-car class-B sprung acceleration",
        save_path=logger.run_dir / "figures" / "quarter_accel_compare.png",
    )

    t_f = f["t"]
    win = slice(0, int(10.0 / (t_f[1] - t_f[0])))
    plot_timeseries(
        t_f[win],
        np.column_stack(
            [
                f["passive_run"]["body_accel"][win, 0],
                f["skyhook_run"]["body_accel"][win, 0],
                f["lqr_run"]["body_accel"][win, 0],
            ]
        ),
        labels=["passive", "skyhook", "lqr"],
        ylabel="Heave accel [m/s^2]",
        title="Full-car class-B heave acceleration",
        save_path=logger.run_dir / "figures" / "full_heave_compare.png",
    )
    plot_timeseries(
        t_f[win],
        f["lqr_run"]["forces"][win],
        labels=list(CORNER_NAMES),
        ylabel="Actuator force [N]",
        title="Full-car LQR actuator forces",
        save_path=logger.run_dir / "figures" / "full_lqr_actuator_forces.png",
    )

    np.savez(
        logger.run_dir / "raw.npz",
        t_quarter=q["t"],
        q_passive_accel=q["passive_run"]["body_accel"],
        q_skyhook_accel=q["skyhook_run"]["body_accel"],
        q_lqr_accel=q["lqr_run"]["body_accel"],
        q_lqr_forces=q["lqr_run"]["forces"],
        t_full=f["t"],
        f_passive_accel=f["passive_run"]["body_accel"],
        f_skyhook_accel=f["skyhook_run"]["body_accel"],
        f_lqr_accel=f["lqr_run"]["body_accel"],
        f_lqr_forces=f["lqr_run"]["forces"],
    )

    if not poles_ok:
        raise RuntimeError(
            f"Phase 3.3 acceptance failed: closed-loop poles outside unit circle "
            f"(quarter={q['lqr_poles_max_abs']:.4f}, full={f['lqr_poles_max_abs']:.4f})."
        )
    if not heave_pass:
        raise RuntimeError(
            f"Phase 3.3 acceptance failed: full-car heave improvement vs skyhook "
            f"= {heave_improve:.2f}% < 10%."
        )
    if not quarter_pass:
        raise RuntimeError(
            f"Phase 3.3 acceptance failed: quarter-car LQR improvement vs skyhook "
            f"= {quarter_improve:.2f}% < 10%."
        )
    logger.log("Phase 3.3 LQR comfort baseline completed; acceptance passed.")
    return logger.run_dir


def main() -> None:
    print(run_phase_3_3())


if __name__ == "__main__":
    main()

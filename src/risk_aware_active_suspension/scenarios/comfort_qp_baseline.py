from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import (
    FullCarComfortQP,
    QuarterCarComfortQP,
)
from risk_aware_active_suspension.controllers.lqr import (
    FullCarLQRController,
    QuarterCarLQRController,
)
from risk_aware_active_suspension.controllers.passive import PassiveController
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
)
from risk_aware_active_suspension.utils.config import (
    LQRParams,
    VehicleParams,
    from_yaml,
    lqr_from_yaml,
)
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


def _run_with_timing(plant, controller, t, road_seq):
    if isinstance(plant, QuarterCar):
        states = np.zeros((len(t), 4), dtype=float)
        forces = np.zeros(len(t), dtype=float)
        timings = np.zeros(len(t), dtype=float)
        state = np.zeros(4)
        for k in range(len(t) - 1):
            u = controller.compute(state)
            forces[k] = float(u[0])
            timings[k] = getattr(controller, "last_solve_time_s", 0.0)
            state = plant.step(state, float(u[0]), float(road_seq[k]), float(t[k + 1] - t[k]))
            states[k + 1] = state
        body_accel = np.array(
            [plant.sprung_acceleration(s, u=f) for s, f in zip(states, forces)]
        )
        strokes = states[:, 0] - states[:, 2]
        tire_forces = np.array(
            [plant.tire_normal_force(s, w=float(r)) for s, r in zip(states, road_seq)]
        )
        return {
            "states": states,
            "forces": forces,
            "body_accel": body_accel,
            "strokes": strokes,
            "tire_forces": tire_forces,
            "timings_s": timings,
        }
    states = np.zeros((len(t), 14), dtype=float)
    forces = np.zeros((len(t), 4), dtype=float)
    timings = np.zeros(len(t), dtype=float)
    state = np.zeros(14)
    for k in range(len(t) - 1):
        u = controller.compute(state)
        forces[k] = u
        timings[k] = getattr(controller, "last_solve_time_s", 0.0)
        state = plant.step(state, road_seq[k], float(t[k + 1] - t[k]), u=u)
        states[k + 1] = state
    body_accel = plant.batch_body_accelerations(states, road_seq, forces)
    strokes = plant.batch_suspension_strokes(states)
    tire_forces = plant.batch_tire_normal_forces(states, road_seq)
    return {
        "states": states,
        "forces": forces,
        "body_accel": body_accel,
        "strokes": strokes,
        "tire_forces": tire_forces,
        "timings_s": timings,
    }


def compare_unconstrained_match(
    params: VehicleParams, lqr: LQRParams, n_samples: int = 500, seed: int = 0
) -> dict[str, float]:
    """Verify that with f_max -> infinity the QP recovers the LQR gain."""
    huge = replace(lqr, f_max=1e9)
    rng = np.random.default_rng(seed)

    q_qp = QuarterCarComfortQP(params, huge)
    q_lqr = QuarterCarLQRController(params, huge)
    q_diff = []
    for _ in range(n_samples):
        x = rng.standard_normal(4) * np.array([0.05, 0.5, 0.01, 0.5])
        q_diff.append(float(np.max(np.abs(q_qp.compute(x) - q_lqr.compute(x)))))

    f_qp = FullCarComfortQP(params, huge)
    f_lqr = FullCarLQRController(params, huge)
    f_diff = []
    for _ in range(n_samples):
        x = rng.standard_normal(14) * np.array(
            [0.05, 0.5, 0.01, 0.1, 0.01, 0.1, *([0.005, 0.5] * 4)]
        )
        f_diff.append(float(np.max(np.abs(f_qp.compute(x) - f_lqr.compute(x)))))

    return {
        "quarter_max_abs_diff_N": float(max(q_diff)),
        "full_max_abs_diff_N": float(max(f_diff)),
    }


def run_phase_3_4(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    results_root: str | Path = "results",
    duration: float = 30.0,
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    params = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)

    logger = RunLogger.create(
        results_root, phase="phase-3.4", step="comfort-qp", descriptor="class-b-vs-lqr"
    )
    shutil.copy2(vehicle_config, logger.run_dir / "config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")

    # ---- Acceptance #1: unconstrained QP == LQR ----
    match = compare_unconstrained_match(params, lqr)
    logger.log_kv("unconstrained_quarter_max_abs_diff_n", match["quarter_max_abs_diff_N"])
    logger.log_kv("unconstrained_full_max_abs_diff_n", match["full_max_abs_diff_N"])
    match_pass = (
        match["quarter_max_abs_diff_N"] < 1e-3 and match["full_max_abs_diff_N"] < 1e-3
    )
    logger.log_kv("acceptance_unconstrained_matches_lqr", int(match_pass))

    # ---- Closed-loop runs with finite f_max ----
    dt_q = 0.001
    dt_f = 0.002
    t_q = np.arange(0.0, duration, dt_q)
    road_q = iso8608("B", v_x=80.0 / 3.6, t=t_q, seed=11)
    qc_plant = QuarterCar(params)
    qc_qp = QuarterCarComfortQP(params, lqr)
    qc_lqr_run = _simulate_quarter_car(qc_plant, QuarterCarLQRController(params, lqr), t_q, road_q)
    qc_qp_run = _run_with_timing(qc_plant, qc_qp, t_q, road_q)

    t_f = np.arange(0.0, duration, dt_f)
    fc_plant = FullCar(params)
    roads_f = _straight_two_track_road(fc_plant, t=t_f, dt=dt_f, v_x=80.0 / 3.6, road_class="B")
    fc_passive = _simulate_full_car(fc_plant, PassiveController(n_actuators=4), t_f, roads_f)
    fc_lqr_run = _simulate_full_car(fc_plant, FullCarLQRController(params, lqr), t_f, roads_f)
    fc_qp = FullCarComfortQP(params, lqr)
    fc_qp_run = _run_with_timing(fc_plant, fc_qp, t_f, roads_f)

    # Metrics
    def rms_axes(run):
        return {
            "heave": float(rmse(run["body_accel"][:, 0])),
            "roll": float(rmse(run["body_accel"][:, 1])),
            "pitch": float(rmse(run["body_accel"][:, 2])),
        }

    qc_lqr_rms = float(rmse(qc_lqr_run["body_accel"]))
    qc_qp_rms = float(rmse(qc_qp_run["body_accel"]))
    fc_lqr_rms = rms_axes(fc_lqr_run)
    fc_qp_rms = rms_axes(fc_qp_run)
    fc_passive_rms = rms_axes(fc_passive)
    logger.log_kv("quarter_lqr_rms", qc_lqr_rms)
    logger.log_kv("quarter_qp_rms", qc_qp_rms)
    for axis in ("heave", "roll", "pitch"):
        logger.log_kv(f"full_passive_{axis}_rms", fc_passive_rms[axis])
        logger.log_kv(f"full_lqr_{axis}_rms", fc_lqr_rms[axis])
        logger.log_kv(f"full_qp_{axis}_rms", fc_qp_rms[axis])

    # ---- Acceptance #2: solve time < 1 ms ----
    qp_times = fc_qp_run["timings_s"][:-1]  # last entry unset
    timing = {
        "median_ms": float(np.median(qp_times) * 1000.0),
        "p95_ms": float(np.percentile(qp_times, 95) * 1000.0),
        "p99_ms": float(np.percentile(qp_times, 99) * 1000.0),
        "max_ms": float(np.max(qp_times) * 1000.0),
        "n_solves": int(len(qp_times)),
    }
    for k, v in timing.items():
        logger.log_kv(f"full_qp_solve_{k}", v)
    timing_pass = timing["p95_ms"] < 1.0
    logger.log_kv("acceptance_solve_time_p95_lt_1ms", int(timing_pass))

    # Figures
    win_f = slice(0, int(8.0 / dt_f))
    plot_timeseries(
        t_f[win_f],
        np.column_stack(
            [
                fc_passive["body_accel"][win_f, 0],
                fc_lqr_run["body_accel"][win_f, 0],
                fc_qp_run["body_accel"][win_f, 0],
            ]
        ),
        labels=["passive", "lqr", "comfort_qp"],
        ylabel="Heave accel [m/s^2]",
        title="Full-car class-B heave: comfort-QP recovers LQR",
        save_path=logger.run_dir / "figures" / "full_heave_qp_vs_lqr.png",
    )
    plot_timeseries(
        t_f[win_f],
        fc_qp_run["forces"][win_f],
        labels=list(CORNER_NAMES),
        ylabel="Actuator force [N]",
        title="Full-car comfort-QP actuator forces",
        save_path=logger.run_dir / "figures" / "full_qp_actuator_forces.png",
    )
    plot_timeseries(
        np.arange(len(qp_times)),
        qp_times * 1000.0,
        ylabel="Solve time [ms]",
        title="Full-car comfort-QP per-step solve time",
        save_path=logger.run_dir / "figures" / "full_qp_solve_times.png",
    )

    np.savez(
        logger.run_dir / "raw.npz",
        t_quarter=t_q,
        q_lqr_accel=qc_lqr_run["body_accel"],
        q_qp_accel=qc_qp_run["body_accel"],
        q_qp_forces=qc_qp_run["forces"],
        t_full=t_f,
        f_passive_accel=fc_passive["body_accel"],
        f_lqr_accel=fc_lqr_run["body_accel"],
        f_qp_accel=fc_qp_run["body_accel"],
        f_qp_forces=fc_qp_run["forces"],
        f_qp_timings_s=qp_times,
    )

    if not match_pass:
        raise RuntimeError(
            f"Phase 3.4 acceptance failed: unconstrained QP does not match LQR "
            f"(quarter={match['quarter_max_abs_diff_N']:.3e} N, full={match['full_max_abs_diff_N']:.3e} N)."
        )
    if not timing_pass:
        raise RuntimeError(
            f"Phase 3.4 acceptance failed: QP solve time p95={timing['p95_ms']:.3f} ms >= 1 ms."
        )
    logger.log("Phase 3.4 comfort-QP baseline completed; acceptance passed.")
    return logger.run_dir


def main() -> None:
    print(run_phase_3_4())


if __name__ == "__main__":
    main()

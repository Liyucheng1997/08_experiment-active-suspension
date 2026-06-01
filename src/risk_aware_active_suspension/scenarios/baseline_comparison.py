from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.lqr import FullCarLQRController
from risk_aware_active_suspension.controllers.passive import PassiveController
from risk_aware_active_suspension.controllers.skyhook import FullCarSkyhookController
from risk_aware_active_suspension.inputs.road import iso8608
from risk_aware_active_suspension.metrics.signals import rmse
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.scenarios.rough_road_validation import (
    _straight_two_track_road,
)
from risk_aware_active_suspension.scenarios.skyhook_baseline import tune_quarter_car_skyhook
from risk_aware_active_suspension.utils.config import (
    LQRParams,
    VehicleParams,
    from_yaml,
    lqr_from_yaml,
)
from risk_aware_active_suspension.utils.logger import RunLogger
from risk_aware_active_suspension.utils.plot import plot_timeseries


@dataclass(frozen=True)
class ScenarioSpec:
    name: str
    road_class: str
    lane_change: bool
    duration: float


SCENARIOS: tuple[ScenarioSpec, ...] = (
    ScenarioSpec("classB_straight", "B", False, 30.0),
    ScenarioSpec("classD_straight", "D", False, 30.0),
    ScenarioSpec("classB_lane_change", "B", True, 12.0),
)


def lane_change_a_y(
    t: np.ndarray, peak_ay: float = 3.0, start: float = 2.0, duration: float = 4.0
) -> np.ndarray:
    """Windowed-sine a_y profile shaped like a single lane change."""
    out = np.zeros_like(t)
    mask = (t >= start) & (t <= start + duration)
    tau = (t[mask] - start) / duration
    envelope = np.sin(np.pi * tau) ** 2
    out[mask] = peak_ay * envelope * np.sin(2.0 * np.pi * tau)
    return out


def _build_scenario_inputs(
    plant: FullCar, spec: ScenarioSpec, dt: float, v_x: float = 80.0 / 3.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, spec.duration, dt)
    roads = _straight_two_track_road(plant, t=t, dt=dt, v_x=v_x, road_class=spec.road_class)
    if spec.lane_change:
        a_y = lane_change_a_y(t, peak_ay=3.0, start=2.0, duration=4.0)
    else:
        a_y = np.zeros_like(t)
    body_acc = np.column_stack([np.zeros_like(t), a_y])
    return t, roads, body_acc


def _simulate(
    plant: FullCar,
    controller,
    t: np.ndarray,
    roads: np.ndarray,
    body_acc: np.ndarray,
) -> dict[str, np.ndarray]:
    states = np.zeros((len(t), 14), dtype=float)
    forces = np.zeros((len(t), 4), dtype=float)
    state = np.zeros(14)
    for k in range(len(t) - 1):
        u = controller.compute(state)
        forces[k] = u
        state = plant.step(
            state, roads[k], float(t[k + 1] - t[k]), u=u, body_acc=body_acc[k]
        )
        states[k + 1] = state
    body_accel = np.array(
        [plant.body_accelerations(s, w, u=f, body_acc=ba)
         for s, w, f, ba in zip(states, roads, forces, body_acc)]
    )
    strokes = plant.batch_suspension_strokes(states)
    tire_forces = plant.batch_tire_normal_forces(states, roads)
    return {
        "states": states,
        "forces": forces,
        "body_accel": body_accel,
        "strokes": strokes,
        "tire_forces": tire_forces,
    }


def _summary(run: dict[str, np.ndarray]) -> dict[str, float]:
    a = run["body_accel"]
    return {
        "heave_rms": float(rmse(a[:, 0])),
        "roll_rms": float(rmse(a[:, 1])),
        "pitch_rms": float(rmse(a[:, 2])),
        "aggregate_rms": float(np.sqrt(rmse(a[:, 0]) ** 2 + rmse(a[:, 1]) ** 2 + rmse(a[:, 2]) ** 2)),
        "max_force_n": float(np.max(np.abs(run["forces"]))),
        "stroke_p95_m": float(np.percentile(np.abs(run["strokes"]), 95)),
        "min_tire_force_n": float(np.min(run["tire_forces"])),
    }


def _build_controllers(params: VehicleParams, lqr: LQRParams, c_sky: float) -> dict[str, object]:
    return {
        "passive": PassiveController(n_actuators=4),
        "skyhook": FullCarSkyhookController(params=params, c_sky=c_sky, f_max=lqr.f_max),
        "lqr": FullCarLQRController(params, lqr),
        "comfort_qp": FullCarComfortQP(params, lqr),
    }


def run_phase_3_5(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    results_root: str | Path = "results",
    dt: float = 0.002,
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    params = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)
    plant = FullCar(params)

    logger = RunLogger.create(
        results_root, phase="phase-3.5", step="baseline-comparison", descriptor="four-controllers"
    )
    shutil.copy2(vehicle_config, logger.run_dir / "config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")

    c_sky = float(tune_quarter_car_skyhook(params)["best_c_sky"])
    logger.log_kv("skyhook_c_sky", c_sky)

    controllers = _build_controllers(params, lqr, c_sky)
    controller_order = ("passive", "skyhook", "lqr", "comfort_qp")

    table_rows: list[dict[str, object]] = []
    runs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for spec in SCENARIOS:
        t, roads, body_acc = _build_scenario_inputs(plant, spec, dt=dt)
        runs[spec.name] = {}
        for name in controller_order:
            run = _simulate(plant, controllers[name], t, roads, body_acc)
            runs[spec.name][name] = run
            metrics = _summary(run)
            metrics.update({"scenario": spec.name, "controller": name})
            table_rows.append(metrics)
            logger.log(
                f"[{spec.name:>18}] {name:>11}  heave={metrics['heave_rms']:.3f} "
                f"roll={metrics['roll_rms']:.3f} pitch={metrics['pitch_rms']:.3f} "
                f"agg={metrics['aggregate_rms']:.3f} maxF={metrics['max_force_n']:.0f}N "
                f"strokeP95={metrics['stroke_p95_m']*1000:.1f}mm"
            )

    # Write the comparison CSV
    fieldnames = [
        "scenario", "controller",
        "heave_rms", "roll_rms", "pitch_rms", "aggregate_rms",
        "max_force_n", "stroke_p95_m", "min_tire_force_n",
    ]
    with (logger.run_dir / "comparison.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in table_rows:
            writer.writerow({k: row[k] for k in fieldnames})

    # Acceptance: ordering passive > skyhook > LQR ≈ comfort-QP on aggregate RMS
    # (worst -> best). LQR ≈ QP within 10% relative — they share the same
    # quadratic cost so they coincide when unconstrained (proven to 1e-7 N
    # in Phase 3.4); under heavy saturation they can legitimately diverge
    # because OSQP redistributes force across corners while LQR clips each
    # corner independently.
    ordering_pass = True
    ordering_details = []
    for spec in SCENARIOS:
        agg = {
            name: next(
                row["aggregate_rms"] for row in table_rows
                if row["scenario"] == spec.name and row["controller"] == name
            )
            for name in controller_order
        }
        passive_skyhook = agg["passive"] > agg["skyhook"]
        skyhook_lqr = agg["skyhook"] > agg["lqr"]
        lqr_qp_close = abs(agg["lqr"] - agg["comfort_qp"]) / agg["lqr"] < 0.10
        ok = passive_skyhook and skyhook_lqr and lqr_qp_close
        ordering_pass = ordering_pass and ok
        ordering_details.append(
            f"{spec.name}: passive>{passive_skyhook} skyhook>{skyhook_lqr} "
            f"lqr~qp={lqr_qp_close} "
            f"(p={agg['passive']:.3f},s={agg['skyhook']:.3f},"
            f"l={agg['lqr']:.3f},q={agg['comfort_qp']:.3f})"
        )
    for line in ordering_details:
        logger.log(line)
    logger.log_kv("acceptance_ordering_passive_gt_skyhook_gt_lqr_approx_qp", int(ordering_pass))

    # Figures
    for spec in SCENARIOS:
        t, roads, body_acc = _build_scenario_inputs(plant, spec, dt=dt)
        win = slice(0, min(len(t), int(8.0 / dt)))
        plot_timeseries(
            t[win],
            np.column_stack(
                [runs[spec.name][name]["body_accel"][win, 0] for name in controller_order]
            ),
            labels=list(controller_order),
            ylabel="Heave accel [m/s^2]",
            title=f"{spec.name}: heave acceleration",
            save_path=logger.run_dir / "figures" / f"{spec.name}_heave.png",
        )
        plot_timeseries(
            t[win],
            np.column_stack(
                [runs[spec.name][name]["body_accel"][win, 1] for name in controller_order]
            ),
            labels=list(controller_order),
            ylabel="Roll accel [rad/s^2]",
            title=f"{spec.name}: roll acceleration",
            save_path=logger.run_dir / "figures" / f"{spec.name}_roll.png",
        )

    if spec.lane_change:
        plot_timeseries(
            t,
            body_acc[:, 1],
            ylabel="a_y [m/s^2]",
            title="Lane-change lateral acceleration profile",
            save_path=logger.run_dir / "figures" / "lane_change_a_y.png",
        )

    if not ordering_pass:
        raise RuntimeError(
            "Phase 3.5 acceptance failed: ordering not satisfied. Details:\n"
            + "\n".join(ordering_details)
        )
    logger.log("Phase 3.5 baseline comparison completed; acceptance passed.")
    return logger.run_dir


def main() -> None:
    print(run_phase_3_5())


if __name__ == "__main__":
    main()

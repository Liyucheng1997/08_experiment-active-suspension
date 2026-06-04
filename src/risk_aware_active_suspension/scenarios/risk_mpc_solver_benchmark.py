from __future__ import annotations

import csv
import shutil
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import FullCarRiskMPC, predict_fz_horizon
from risk_aware_active_suspension.controllers.risk_qp import RiskQPParams
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.inputs.road import sine
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.risk_mpc_residual_prediction import (
    _full_car_corner_phi_known,
)
from risk_aware_active_suspension.scenarios.risk_qp_fullcar_ground_truth import _smooth_j_turn
from risk_aware_active_suspension.utils.config import (
    ObserverParams,
    from_yaml,
    lqr_from_yaml,
    observer_from_yaml,
)
from risk_aware_active_suspension.utils.logger import RunLogger


def run_phase_5_4(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    observer_config: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
    duration: float = 2.2,
    dt: float = 0.001,
    n_benchmark_solves: int = 120,
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    observer_config = Path(observer_config)
    vehicle = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)
    base_observer = observer_from_yaml(observer_config)
    observer_params = ObserverParams(
        lambda_1=800.0,
        lambda_2=30000.0,
        epsilon=0.01,
        delta_fz_err=base_observer.delta_fz_err,
    )
    plant = FullCar(vehicle)
    logger = RunLogger.create(
        results_root,
        phase="phase-5.4",
        step="risk-mpc",
        descriptor="warm-start-timing",
    )
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")
    shutil.copy2(observer_config, logger.run_dir / "observer_config.yaml")

    t, roads, a_y, f_c, states, fz_hats = _make_benchmark_trace(
        plant, lqr, observer_params, duration=duration, dt=dt
    )
    mu = 0.8
    rho_safe = 0.85
    risk_params = RiskWeightParams(
        rho_th=0.75,
        k_rho=25.0,
        kappa_rho=5.0,
        q_c_min=1.0,
        q_c_max=1.0,
        q_p_min=0.0,
        q_p_max=1.0e-5,
    )
    qp_params = RiskQPParams(
        rho_safe=rho_safe,
        enforce_rate=False,
        r_du_factor=1.0e-5,
        osqp_eps_abs=1.0e-5,
        osqp_eps_rel=1.0e-5,
        osqp_max_iter=100000,
    )

    rows: list[dict[str, float | int | str]] = []
    horizons = [1, 5, 10, 20]
    for horizon in horizons:
        for warm_start in (False, True):
            ctrl = FullCarRiskMPC(
                vehicle,
                lqr,
                risk_params,
                qp_params,
                horizon=horizon,
                warm_start=warm_start,
            )
            result = _benchmark_controller(
                ctrl,
                plant,
                t,
                roads,
                a_y,
                f_c,
                states,
                fz_hats,
                mu,
                n_benchmark_solves=n_benchmark_solves,
            )
            row = {
                "horizon": horizon,
                "warm_start": int(warm_start),
                **result,
            }
            rows.append(row)
            prefix = f"N{horizon}_{'warm' if warm_start else 'cold'}"
            for key, value in row.items():
                if key not in {"horizon", "warm_start"}:
                    logger.log_kv(f"{prefix}_{key}", value)

    _write_rows(logger.run_dir / "solver_benchmark.csv", rows)
    reductions = _log_acceptance(logger, rows, lqr.T_s)
    _plot_timing(logger.run_dir / "figures" / "solver_timing_by_horizon.png", rows, lqr.T_s)
    _plot_iterations(logger.run_dir / "figures" / "solver_iterations_by_horizon.png", rows, reductions)

    logger.log("Phase 5.4 warm-start timing benchmark completed.")
    return logger.run_dir


def _make_benchmark_trace(
    plant: FullCar,
    lqr,
    observer_params: ObserverParams,
    duration: float,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, duration, dt)
    a_y = _smooth_j_turn(t, peak=6.0, rise_time=0.5)
    roads = np.zeros((len(t), 4), dtype=float)
    roads[:, 0] = sine(t, amplitude=0.015, freq_hz=1.0)
    roads[:, 2] = sine(t, amplitude=0.015, freq_hz=1.0, phase_rad=-0.7)
    f_c = np.tile(np.array([2300.0, 800.0, 900.0, 700.0]), (len(t), 1))
    states = np.zeros((len(t), 14), dtype=float)
    fz_hat = np.zeros((len(t), 4), dtype=float)
    observers = [
        STO(observer_params, unsprung_mass=plant.params.m_u, use_saturation=True)
        for _ in range(4)
    ]
    controller = FullCarComfortQP(plant.params, lqr)
    state = np.zeros(14)
    force = np.zeros(4)
    static_loads = plant.static_loads()
    for idx in range(len(t) - 1):
        force = controller.compute(state)
        for corner_idx, observer in enumerate(observers):
            phi_known = _full_car_corner_phi_known(plant, state, force, corner_idx)
            additive_force_hat, _ = observer.step(
                v_u_meas=state[7 + 2 * corner_idx],
                phi_known=phi_known,
                dt=dt,
            )
            fz_hat[idx, corner_idx] = static_loads[corner_idx] + additive_force_hat
        state = plant.step(state, roads[idx], dt, u=force, body_acc=np.array([0.0, a_y[idx]]))
        states[idx + 1] = state
    fz_hat[-1] = fz_hat[-2]
    return t, roads, a_y, f_c, states, fz_hat


def _benchmark_controller(
    ctrl: FullCarRiskMPC,
    plant: FullCar,
    t: np.ndarray,
    roads: np.ndarray,
    a_y: np.ndarray,
    f_c: np.ndarray,
    states: np.ndarray,
    fz_hat: np.ndarray,
    mu: float,
    n_benchmark_solves: int,
) -> dict[str, float]:
    steps_per_update = max(1, int(round(ctrl.lqr.T_s / float(t[1] - t[0]))))
    valid = np.arange(int(0.7 / (t[1] - t[0])), len(t) - steps_per_update * ctrl.horizon - 1, steps_per_update)
    solve_indices = valid[:n_benchmark_solves]
    if len(solve_indices) == 0:
        raise RuntimeError("benchmark trace is too short for the requested horizon.")

    wall_times = []
    setup_solve_times = []
    solve_times = []
    iterations = []
    static_h = np.tile(plant.static_loads(), (ctrl.horizon, 1))
    for idx in solve_indices:
        horizon_indices = idx + steps_per_update * np.arange(ctrl.horizon)
        residual = fz_hat[idx] - plant.static_loads()
        fz_h = predict_fz_horizon(static_h, residual, mode="decay", alpha=0.95)
        body_acc_h = np.column_stack([
            np.zeros(ctrl.horizon, dtype=float),
            a_y[horizon_indices],
        ])
        tic = time.perf_counter()
        ctrl.compute(
            state=states[idx],
            f_z_hat=fz_h,
            f_c=f_c[horizon_indices],
            mu=mu,
            road_horizon=roads[horizon_indices],
            body_acc_horizon=body_acc_h,
            f_z_residual=fz_h - static_h,
        )
        wall_times.append(time.perf_counter() - tic)
        setup_solve_times.append(ctrl.last_setup_solve_time_s)
        solve_times.append(ctrl.last_solve_time_s)
        iterations.append(ctrl.last_iterations)

    wall = np.asarray(wall_times)
    setup_solve = np.asarray(setup_solve_times)
    solve = np.asarray(solve_times)
    iters = np.asarray(iterations, dtype=float)
    return {
        "mean_wall_ms": float(np.mean(wall) * 1000.0),
        "p95_wall_ms": float(np.percentile(wall, 95) * 1000.0),
        "mean_setup_solve_ms": float(np.mean(setup_solve) * 1000.0),
        "p95_setup_solve_ms": float(np.percentile(setup_solve, 95) * 1000.0),
        "mean_solve_ms": float(np.mean(solve) * 1000.0),
        "p95_solve_ms": float(np.percentile(solve, 95) * 1000.0),
        "mean_iterations": float(np.mean(iters)),
        "p95_iterations": float(np.percentile(iters, 95)),
    }


def _write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _log_acceptance(logger: RunLogger, rows: list[dict[str, float | int | str]], sample_time_s: float) -> dict[int, float]:
    reductions: dict[int, float] = {}
    largest_feasible = 0
    for horizon in sorted({int(r["horizon"]) for r in rows}):
        cold = next(r for r in rows if int(r["horizon"]) == horizon and int(r["warm_start"]) == 0)
        warm = next(r for r in rows if int(r["horizon"]) == horizon and int(r["warm_start"]) == 1)
        reduction = (
            (float(cold["mean_iterations"]) - float(warm["mean_iterations"]))
            / max(float(cold["mean_iterations"]), 1.0e-9)
        )
        reductions[horizon] = reduction
        logger.log_kv(f"N{horizon}_warm_iteration_reduction_ratio", reduction)
        if float(warm["p95_wall_ms"]) <= sample_time_s * 1000.0:
            largest_feasible = horizon
    n10_warm = next(r for r in rows if int(r["horizon"]) == 10 and int(r["warm_start"]) == 1)
    n10_reduction = reductions[10]
    logger.log_kv("acceptance_N10_p95_wall_within_Ts", int(float(n10_warm["p95_wall_ms"]) <= sample_time_s * 1000.0))
    logger.log_kv(
        "acceptance_N10_p95_setup_solve_within_Ts",
        int(float(n10_warm["p95_setup_solve_ms"]) <= sample_time_s * 1000.0),
    )
    logger.log_kv("acceptance_N10_iteration_reduction_ge_30pct", int(n10_reduction >= 0.30))
    logger.log_kv("largest_feasible_horizon_p95_wall_within_Ts", largest_feasible)
    return reductions


def _plot_timing(path: Path, rows: list[dict[str, float | int | str]], sample_time_s: float) -> None:
    horizons = sorted({int(r["horizon"]) for r in rows})
    x = np.arange(len(horizons))
    width = 0.36
    cold = [float(next(r for r in rows if int(r["horizon"]) == h and int(r["warm_start"]) == 0)["p95_wall_ms"]) for h in horizons]
    warm = [float(next(r for r in rows if int(r["horizon"]) == h and int(r["warm_start"]) == 1)["p95_wall_ms"]) for h in horizons]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - width / 2, cold, width, label="cold start", color="tab:gray")
    ax.bar(x + width / 2, warm, width, label="warm start", color="tab:blue")
    ax.axhline(sample_time_s * 1000.0, color="tab:red", linestyle="--", linewidth=1.2, label="T_s = 5 ms")
    ax.set_xticks(x, [str(h) for h in horizons])
    ax.set_xlabel("Prediction horizon N_p")
    ax.set_ylabel("p95 wall time [ms]")
    ax.set_title("Phase 5.4 MPC timing benchmark")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_iterations(path: Path, rows: list[dict[str, float | int | str]], reductions: dict[int, float]) -> None:
    horizons = sorted({int(r["horizon"]) for r in rows})
    x = np.arange(len(horizons))
    width = 0.36
    cold = [float(next(r for r in rows if int(r["horizon"]) == h and int(r["warm_start"]) == 0)["mean_iterations"]) for h in horizons]
    warm = [float(next(r for r in rows if int(r["horizon"]) == h and int(r["warm_start"]) == 1)["mean_iterations"]) for h in horizons]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(x - width / 2, cold, width, label="cold start", color="tab:gray")
    ax.bar(x + width / 2, warm, width, label="warm start", color="tab:blue")
    for idx, horizon in enumerate(horizons):
        ax.text(idx, max(cold[idx], warm[idx]) + 1.0, f"{100.0 * reductions[horizon]:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(h) for h in horizons])
    ax.set_xlabel("Prediction horizon N_p")
    ax.set_ylabel("Mean OSQP iterations")
    ax.set_title("Warm-start iteration reduction")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    print(run_phase_5_4())


if __name__ == "__main__":
    main()

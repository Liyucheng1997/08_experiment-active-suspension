from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.controllers.risk_mpc import RESIDUAL_DECAY, FullCarRiskMPC, predict_fz_horizon
from risk_aware_active_suspension.controllers.risk_qp import RiskQPParams
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.metrics.tire import rho
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.risk_mpc_residual_prediction import _full_car_corner_phi_known
from risk_aware_active_suspension.scenarios.risk_mpc_scenario_sweep import (
    PHASE6_SCENARIOS,
    Phase6Scenario,
    _build_inputs,
    _risk_params_for,
)
from risk_aware_active_suspension.utils.config import ObserverParams, from_yaml, lqr_from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


@dataclass(frozen=True)
class AblationSpec:
    name: str
    label: str
    disable_sto: bool = False
    disable_sigma: bool = False
    disable_local: bool = False
    disable_tightening: bool = False


ABLATIONS: tuple[AblationSpec, ...] = (
    AblationSpec("full", "Full risk-MPC"),
    AblationSpec("A1_no_sto", "A1: no STO residual", disable_sto=True),
    AblationSpec("A2_sigma_zero", "A2: sigma_rho = 0", disable_sigma=True),
    AblationSpec("A3_kappa_zero", "A3: kappa_rho = 0", disable_local=True),
    AblationSpec("A4_no_tightening", "Extra: no Fz tightening", disable_tightening=True),
)


def run_phase_6_6(
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    observer_config: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
    dt: float = 0.001,
) -> Path:
    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    observer_config = Path(observer_config)
    vehicle = from_yaml(vehicle_config)
    lqr = lqr_from_yaml(controller_config)
    observer_params = observer_from_yaml(observer_config)
    plant = FullCar(vehicle)
    logger = RunLogger.create(results_root, phase="phase-6.6", step="risk-mpc", descriptor="component-ablation")
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")
    shutil.copy2(observer_config, logger.run_dir / "observer_config.yaml")

    scenario_specs = [s for s in PHASE6_SCENARIOS if s.name in {"S8_worst_case", "S9_corner_bump"}]
    rows: list[dict[str, float | str | int]] = []
    runs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scenario in scenario_specs:
        t, roads, a_x, a_y, f_c = _build_inputs(plant, scenario, dt)
        runs[scenario.name] = {}
        for ablation in ABLATIONS:
            risk = _ablation_risk_params(_risk_params_for(scenario), ablation)
            qp = RiskQPParams(
                rho_safe=0.85,
                enforce_rate=False,
                r_du_factor=1.0e-5,
                osqp_eps_abs=1.0e-5,
                osqp_eps_rel=1.0e-5,
                osqp_max_iter=100000,
            )
            run = _run_ablation_closed_loop(
                FullCarRiskMPC(vehicle, lqr, risk, qp, horizon=10, warm_start=False, reuse_solver=True),
                plant,
                t,
                roads,
                a_x,
                a_y,
                f_c,
                scenario.mu,
                observer_params,
                disable_sto=ablation.disable_sto,
                fz_tightening_n=0.0 if ablation.disable_tightening else observer_params.delta_fz_err,
            )
            runs[scenario.name][ablation.name] = run
            row = _summarize_run(scenario, ablation, run)
            rows.append(row)
            logger.log(
                f"{scenario.name} {ablation.name}: peak_rho={row['peak_rho']:.3f}, "
                f"p95={row['p95_rho']:.3f}, heave={row['heave_rms']:.3f}, "
                f"maxF={row['max_force_n']:.0f}N"
            )

    _attach_degradation_metrics(rows)
    _write_rows(logger.run_dir / "phase6_6_ablation_summary.csv", rows)
    _plot_ablation_bars(logger.run_dir / "figures" / "phase6_6_ablation_peak_rho.png", rows, metric="peak_rho", ylabel="Peak rho")
    _plot_ablation_bars(logger.run_dir / "figures" / "phase6_6_ablation_p95_rho.png", rows, metric="p95_rho", ylabel="P95 rho_max")
    _plot_ablation_bars(logger.run_dir / "figures" / "phase6_6_ablation_heave.png", rows, metric="heave_rms", ylabel="Heave RMS [m/s^2]")
    for scenario in scenario_specs:
        t, *_ = _build_inputs(plant, scenario, dt)
        _plot_timeseries(
            logger.run_dir / "figures" / f"{scenario.name}_ablation_rho_timeseries.png",
            t,
            runs[scenario.name],
            scenario,
        )

    primary = [r for r in rows if r["ablation"] in {"A1_no_sto", "A2_sigma_zero", "A4_no_tightening"}]
    a3 = [r for r in rows if r["ablation"] == "A3_kappa_zero"]
    logger.log_kv("acceptance_primary_ablations_degrade_tradeoff", int(all(int(r["degrades_vs_full"]) for r in primary)))
    logger.log_kv("acceptance_A2_sigma_off_degrades_peak_rho", int(all(float(r["peak_rho_delta_vs_full"]) > 0.0 for r in rows if r["ablation"] == "A2_sigma_zero")))
    logger.log_kv("diagnostic_A3_kappa_has_observable_effect", int(any(abs(float(r["peak_rho_delta_vs_full"])) > 1.0e-6 for r in a3)))
    logger.log("Phase 6.6 ablation completed.")
    return logger.run_dir


def _ablation_risk_params(base: RiskWeightParams, ablation: AblationSpec) -> RiskWeightParams:
    return RiskWeightParams(
        rho_th=base.rho_th,
        k_rho=base.k_rho,
        kappa_rho=0.0 if ablation.disable_local else base.kappa_rho,
        q_c_min=base.q_c_min,
        q_c_max=base.q_c_max,
        q_p_min=0.0 if ablation.disable_sigma else base.q_p_min,
        q_p_max=0.0 if ablation.disable_sigma else base.q_p_max,
    )


def _run_ablation_closed_loop(
    controller: FullCarRiskMPC,
    plant: FullCar,
    t: np.ndarray,
    roads: np.ndarray,
    a_x: np.ndarray,
    a_y: np.ndarray,
    f_c: np.ndarray,
    mu: float,
    observer_params: ObserverParams,
    disable_sto: bool,
    fz_tightening_n: float,
) -> dict[str, np.ndarray]:
    n = len(t)
    dt = float(t[1] - t[0])
    steps_per_update = max(1, int(round(controller.lqr.T_s / dt)))
    states = np.zeros((n, 14), dtype=float)
    forces = np.zeros((n, 4), dtype=float)
    fz_true = np.zeros((n, 4), dtype=float)
    fz_hat = np.zeros((n, 4), dtype=float)
    rho_contact = np.zeros((n, 4), dtype=float)
    xi = np.zeros((n, 4), dtype=float)
    solve_time = np.zeros(n, dtype=float)
    observers = [STO(observer_params, unsprung_mass=plant.params.m_u, use_saturation=True) for _ in range(4)]
    state = np.zeros(14)
    force = np.zeros(4)
    for corner_idx, observer in enumerate(observers):
        observer.reset(v_u_hat=state[7 + 2 * corner_idx], chi_hat=0.0)

    for idx in range(n - 1):
        fz_true[idx] = plant.tire_normal_forces(state, roads[idx])
        rho_contact[idx] = rho(f_c[idx], 0.0, fz_true[idx], mu)
        fz_bar_now = plant.static_loads()
        if disable_sto:
            fz_hat[idx] = fz_bar_now
        else:
            for corner_idx, observer in enumerate(observers):
                additive_force_hat, _ = observer.step(
                    v_u_meas=state[7 + 2 * corner_idx],
                    phi_known=_full_car_corner_phi_known(plant, state, force, corner_idx),
                    dt=dt,
                )
                fz_hat[idx, corner_idx] = fz_bar_now[corner_idx] + additive_force_hat

        if idx % steps_per_update == 0:
            horizon_indices = np.minimum(idx + steps_per_update * np.arange(controller.horizon), n - 1)
            fz_bar_h = np.tile(plant.static_loads(), (controller.horizon, 1))
            residual = np.zeros(4) if disable_sto else fz_hat[idx] - fz_bar_now
            fz_h = predict_fz_horizon(fz_bar_h, residual, mode=RESIDUAL_DECAY, alpha=0.95) - fz_tightening_n
            force = controller.compute(
                state,
                f_z_hat=fz_h,
                f_c=f_c[horizon_indices],
                mu=mu,
                road_horizon=roads[horizon_indices],
                body_acc_horizon=np.column_stack([a_x[horizon_indices], a_y[horizon_indices]]),
                f_z_residual=fz_h - fz_bar_h,
            )
        forces[idx] = force
        xi[idx] = controller.last_xi
        solve_time[idx] = controller.last_solve_time_s
        state = plant.step(state, roads[idx], dt, u=force, body_acc=np.array([a_x[idx], a_y[idx]]))
        states[idx + 1] = state

    fz_true[-1] = plant.tire_normal_forces(states[-1], roads[-1])
    fz_hat[-1] = fz_hat[-2]
    rho_contact[-1] = rho(f_c[-1], 0.0, fz_true[-1], mu)
    body_accel = np.array(
        [
            plant.body_accelerations(state_i, road_i, u=force_i, body_acc=np.array([ax_i, ay_i]))
            for state_i, road_i, force_i, ax_i, ay_i in zip(states, roads, forces, a_x, a_y)
        ]
    )
    return {
        "states": states,
        "forces": forces,
        "fz_true": fz_true,
        "fz_hat": fz_hat,
        "rho_contact": rho_contact,
        "xi": xi,
        "solve_time": solve_time,
        "body_accel": body_accel,
    }


def _summarize_run(scenario: Phase6Scenario, ablation: AblationSpec, run: dict[str, np.ndarray]) -> dict[str, float | str | int]:
    rho_max = np.max(run["rho_contact"], axis=1)
    positive_solve = run["solve_time"][run["solve_time"] > 0.0]
    return {
        "scenario": scenario.name,
        "ablation": ablation.name,
        "label": ablation.label,
        "peak_rho": float(np.max(rho_max)),
        "p95_rho": float(np.percentile(rho_max, 95)),
        "heave_rms": float(rms(run["body_accel"][:, 0])),
        "max_force_n": float(np.max(np.abs(run["forces"]))),
        "min_fz_n": float(np.min(run["fz_true"])),
        "max_xi_n": float(np.max(np.abs(run["xi"]))),
        "p95_solve_time_ms": float(np.percentile(positive_solve, 95) * 1000.0) if positive_solve.size else 0.0,
        "stable": int(np.all(np.isfinite(run["states"]))),
    }


def _attach_degradation_metrics(rows: list[dict[str, float | str | int]]) -> None:
    full_by_scenario = {r["scenario"]: r for r in rows if r["ablation"] == "full"}
    for row in rows:
        full = full_by_scenario[row["scenario"]]
        row["peak_rho_delta_vs_full"] = float(row["peak_rho"]) - float(full["peak_rho"])
        row["p95_rho_delta_vs_full"] = float(row["p95_rho"]) - float(full["p95_rho"])
        row["heave_delta_vs_full"] = float(row["heave_rms"]) - float(full["heave_rms"])
        row["max_force_delta_vs_full_n"] = float(row["max_force_n"]) - float(full["max_force_n"])
        row["degrades_vs_full"] = int(
            float(row["peak_rho_delta_vs_full"]) > 1.0e-3
            or float(row["p95_rho_delta_vs_full"]) > 1.0e-3
            or float(row["heave_delta_vs_full"]) > 1.0e-3
            or float(row["max_force_delta_vs_full_n"]) > 1.0
        )


def _write_rows(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_ablation_bars(path: Path, rows: list[dict[str, float | str | int]], metric: str, ylabel: str) -> None:
    scenarios = ["S8_worst_case", "S9_corner_bump"]
    ablations = [a.name for a in ABLATIONS]
    labels = ["Full", "A1", "A2", "A3", "No tight."]
    x = np.arange(len(ablations))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for offset, scenario in [(-width / 2, scenarios[0]), (width / 2, scenarios[1])]:
        values = [
            float(next(r[metric] for r in rows if r["scenario"] == scenario and r["ablation"] == ablation))
            for ablation in ablations
        ]
        ax.bar(x + offset, values, width, label=scenario.split("_", 1)[0])
    ax.set_xticks(x, labels)
    ax.set_ylabel(ylabel)
    ax.set_title(f"Phase 6.6 ablation: {ylabel}")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_timeseries(path: Path, t: np.ndarray, runs: dict[str, dict[str, np.ndarray]], scenario: Phase6Scenario) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    colors = {
        "full": "tab:blue",
        "A1_no_sto": "tab:orange",
        "A2_sigma_zero": "tab:red",
        "A3_kappa_zero": "tab:green",
        "A4_no_tightening": "tab:purple",
    }
    for ablation in ABLATIONS:
        rho_max = np.max(runs[ablation.name]["rho_contact"], axis=1)
        ax.plot(t, rho_max, label=ablation.name, color=colors[ablation.name], linewidth=1.4)
    ax.axhline(0.85, color="0.2", linestyle="--", linewidth=1.0, label="rho_safe")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("rho_max")
    ax.set_title(f"{scenario.name}: ablation rho_max")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    print(run_phase_6_6())


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import FullCarRiskMPC
from risk_aware_active_suspension.controllers.risk_qp import RiskQPParams
from risk_aware_active_suspension.plants.carsim_fmu import CarSimFmuPlant
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.carsim_passthrough import DEFAULT_FMU_PATH
from risk_aware_active_suspension.scenarios.carsim_risk_mpc_validation import (
    _build_carsim_inputs,
    _carsim_risk_params_for,
    _plot_scenario,
    _run_closed_loop,
    _summarize,
)
from risk_aware_active_suspension.scenarios.risk_mpc_scenario_sweep import (
    PHASE6_SCENARIOS,
)
from risk_aware_active_suspension.utils.config import from_yaml, lqr_from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


@dataclass(frozen=True)
class RobustnessCase:
    name: str
    scenario_name: str
    plant_mu: float
    controller_mu: float
    note: str


ROBUSTNESS_CASES: tuple[RobustnessCase, ...] = (
    RobustnessCase(
        name="matched_mu_0p8",
        scenario_name="S8_worst_case",
        plant_mu=0.8,
        controller_mu=0.8,
        note="CarSim and controller both use mu=0.8.",
    ),
    RobustnessCase(
        name="mismatch_actual_0p6_assumed_0p8",
        scenario_name="S8_worst_case",
        plant_mu=0.6,
        controller_mu=0.8,
        note="CarSim uses mu=0.6 while the controller still assumes mu=0.8.",
    ),
)


def run_phase_7_5_robustness_validation(
    fmu_path: str | Path = DEFAULT_FMU_PATH,
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    observer_config: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
    sample_time: float = 0.005,
    speed_kmh: float = 80.0,
    carsim_force_limit_n: float = 800.0,
) -> Path:
    """Run Phase 7.5 CarSim robustness checks for scripted friction mismatch."""

    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    observer_config = Path(observer_config)
    vehicle = from_yaml(vehicle_config)
    lqr = replace(lqr_from_yaml(controller_config), f_max=float(carsim_force_limit_n))
    observer = observer_from_yaml(observer_config)
    full_car = FullCar(vehicle)
    carsim = CarSimFmuPlant(Path(fmu_path))
    logger = RunLogger.create(results_root, phase="phase-7.5", step="carsim", descriptor="robustness-validation")
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")
    shutil.copy2(observer_config, logger.run_dir / "observer_config.yaml")

    rows: list[dict[str, float | str | int]] = []
    runs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for case in ROBUSTNESS_CASES:
        spec = next(item for item in PHASE6_SCENARIOS if item.name == case.scenario_name)
        t, roads, a_x, a_y, f_c = _build_carsim_inputs(full_car, spec, sample_time, speed_kmh)
        risk = _carsim_risk_params_for(spec)
        qp = RiskQPParams(
            rho_safe=0.85,
            enforce_rate=False,
            r_du_factor=1.0e-5,
            osqp_eps_abs=1.0e-5,
            osqp_eps_rel=1.0e-5,
            osqp_max_iter=100000,
        )
        comfort = _run_closed_loop(
            carsim,
            full_car,
            FullCarComfortQP(vehicle, lqr),
            "comfort_qp",
            t,
            roads,
            a_x,
            a_y,
            f_c,
            case.plant_mu,
            case.controller_mu,
            speed_kmh,
            sample_time,
            observer.delta_fz_err,
            -1.0,
        )
        mpc = _run_closed_loop(
            carsim,
            full_car,
            FullCarRiskMPC(vehicle, lqr, risk, qp, horizon=10, warm_start=False, reuse_solver=True),
            "risk_mpc",
            t,
            roads,
            a_x,
            a_y,
            f_c,
            case.plant_mu,
            case.controller_mu,
            speed_kmh,
            sample_time,
            observer.delta_fz_err,
            -1.0,
        )
        row = _summarize(case.name, case.plant_mu, case.controller_mu, comfort, mpc, lqr.f_max)
        row["base_scenario"] = case.scenario_name
        row["note"] = case.note
        rows.append(row)
        runs[case.name] = {"comfort_qp": comfort, "risk_mpc": mpc}
        logger.log(
            f"{case.name}: comfort_rho={row['comfort_peak_rho']:.3f}, "
            f"mpc_rho={row['mpc_peak_rho']:.3f}, reduction={100.0 * row['rho_reduction_ratio']:.1f}%, "
            f"p95={row['comfort_rho_p95']:.3f}->{row['mpc_rho_p95']:.3f}, "
            f"heave={row['comfort_heave_rms_m_s2']:.3f}->{row['mpc_heave_rms_m_s2']:.3f}"
        )
        _plot_scenario(
            logger.run_dir / "figures" / f"{case.name}_closed_loop.png",
            t,
            comfort,
            mpc,
            f"{case.name}: {case.note}",
        )

    _attach_degradation(rows)
    _write_rows(logger.run_dir / "phase7_5_robustness_summary.csv", rows)
    _plot_robustness_bars(logger.run_dir / "figures" / "phase7_5_robustness_bars.png", rows)
    _save_raw(logger.run_dir / "raw.npz", runs)
    _log_acceptance(logger, rows)
    logger.log("Pacejka tire-model switching was not executed because the current FMU exposes mu inputs, not a tire-model selector.")
    logger.log("Phase 7.5 CarSim robustness validation completed.")
    return logger.run_dir


def _attach_degradation(rows: list[dict[str, float | str | int]]) -> None:
    matched = next(row for row in rows if row["scenario"] == "matched_mu_0p8")
    for row in rows:
        row["mpc_peak_rho_delta_vs_matched"] = float(row["mpc_peak_rho"]) - float(matched["mpc_peak_rho"])
        row["mpc_p95_rho_delta_vs_matched"] = float(row["mpc_rho_p95"]) - float(matched["mpc_rho_p95"])
        row["mpc_heave_delta_vs_matched"] = float(row["mpc_heave_rms_m_s2"]) - float(matched["mpc_heave_rms_m_s2"])


def _write_rows(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_robustness_bars(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    labels = [str(row["scenario"]).replace("_", "\n") for row in rows]
    x = np.arange(len(rows))
    width = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].bar(x - width / 2, [float(row["comfort_peak_rho"]) for row in rows], width, label="comfort-QP")
    axes[0].bar(x + width / 2, [float(row["mpc_peak_rho"]) for row in rows], width, label="risk-MPC")
    axes[0].axhline(0.85, color="0.2", linestyle="--", linewidth=1.0)
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Peak rho")
    axes[0].legend(fontsize=8)
    axes[1].bar(x - width / 2, [float(row["comfort_rho_p95"]) for row in rows], width, label="comfort-QP")
    axes[1].bar(x + width / 2, [float(row["mpc_rho_p95"]) for row in rows], width, label="risk-MPC")
    axes[1].axhline(0.85, color="0.2", linestyle="--", linewidth=1.0)
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("P95 rho_max")
    axes[1].legend(fontsize=8)
    fig.suptitle("Phase 7.5 CarSim mu-mismatch robustness")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_raw(path: Path, runs: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    payload = {}
    for case_name, case_runs in runs.items():
        for controller_name, data in case_runs.items():
            for key, value in data.items():
                payload[f"{case_name}_{controller_name}_{key}"] = value
    np.savez(path, **payload)


def _log_acceptance(logger: RunLogger, rows: list[dict[str, float | str | int]]) -> None:
    mismatch = next(row for row in rows if row["scenario"] == "mismatch_actual_0p6_assumed_0p8")
    logger.log_kv("acceptance_all_runs_stable", int(all(int(row["mpc_stable"]) for row in rows)))
    logger.log_kv("acceptance_mismatch_mpc_reduces_peak_rho", int(float(mismatch["rho_reduction_ratio"]) > 0.0))
    logger.log_kv("acceptance_mismatch_mpc_reduces_p95_rho", int(float(mismatch["mpc_rho_p95"]) < float(mismatch["comfort_rho_p95"])))
    logger.log_kv("diagnostic_pacejka_switch_exposed_in_fmu", 0)


def main() -> None:
    print(run_phase_7_5_robustness_validation())


if __name__ == "__main__":
    main()

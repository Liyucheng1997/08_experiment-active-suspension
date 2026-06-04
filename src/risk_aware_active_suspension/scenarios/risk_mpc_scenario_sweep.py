from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import RESIDUAL_DECAY, FullCarRiskMPC
from risk_aware_active_suspension.controllers.risk_qp import RiskQPParams
from risk_aware_active_suspension.controllers.risk_weights import RiskWeightParams
from risk_aware_active_suspension.inputs.road import rounded_bump
from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.metrics.tire import rho
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.baseline_comparison import lane_change_a_y
from risk_aware_active_suspension.scenarios.risk_mpc_residual_prediction import (
    _moving_average,
    run_mpc_residual_closed_loop,
)
from risk_aware_active_suspension.scenarios.risk_qp_fullcar_ground_truth import _coupled_f_c, _smooth_j_turn
from risk_aware_active_suspension.scenarios.rough_road_validation import _straight_two_track_road
from risk_aware_active_suspension.utils.config import from_yaml, lqr_from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


@dataclass(frozen=True)
class Phase6Scenario:
    name: str
    description: str
    duration: float
    road_class: str | None
    mu: float
    f_c: tuple[float, float, float, float]
    peak_ax: float = 0.0
    peak_ay: float = 0.0
    lane_change: bool = False
    bump_wheel: int | None = None
    bump_height: float = 0.0
    bump_start: float = 2.0
    bump_duration: float = 0.02
    road_scale: float = 1.0
    coupled_lateral_demand: bool = False


LOW_RISK_PREFIXES = ("S1", "S2", "S3", "S4", "S6")
MEDIUM_RISK_PREFIXES = ("S5",)
HIGH_RISK_PREFIXES = ("S7", "S8", "S9")
Q_P_MAX_BY_RISK = {
    "low": 1.0e-4,
    "medium": 3.0e-4,
    "high": 1.0e-3,
}


PHASE6_SCENARIOS: tuple[Phase6Scenario, ...] = (
    Phase6Scenario("S1_classB_straight", "B x0.5 + straight", 4.0, "B", 0.9, (700, 700, 600, 600), road_scale=0.5),
    Phase6Scenario("S2_smooth_lane_change", "flat + single lane change", 4.0, None, 0.9, (700, 700, 600, 600), lane_change=True, peak_ay=3.0),
    Phase6Scenario("S3_smooth_jturn", "flat + moderate J-turn", 4.0, None, 0.9, (850, 850, 700, 700), peak_ay=4.0),
    Phase6Scenario("S4_classC_cornering", "C x0.1 + steady cornering", 4.0, "C", 0.8, (1200, 900, 900, 750), peak_ay=4.0, road_scale=0.1),
    Phase6Scenario("S5_classC_lane_change", "C x0.25 + lane change", 4.0, "C", 0.8, (1200, 900, 900, 750), lane_change=True, peak_ay=4.0, road_scale=0.25),
    Phase6Scenario("S6_emergency_brake", "flat + emergency brake", 3.5, None, 0.8, (750, 750, 1200, 1200), peak_ax=-8.0),
    Phase6Scenario("S7_brake_cornering", "flat + brake during cornering", 3.5, None, 0.75, (1500, 900, 1200, 900), peak_ax=-6.0, peak_ay=4.0),
    Phase6Scenario("S8_worst_case", "D x0.02 + brake + cornering", 4.0, "D", 0.7, (900, 600, 800, 600), peak_ax=-6.0, peak_ay=5.0, road_scale=0.02),
    Phase6Scenario("S9_corner_bump", "flat + cornering + FR bump", 3.5, None, 0.7, (0, 0, 0, 0), peak_ay=4.0, bump_wheel=1, bump_height=0.02, bump_start=1.8, bump_duration=0.03, coupled_lateral_demand=True),
)


def run_phase_6_main_sweep(
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
    logger = RunLogger.create(
        results_root,
        phase="phase-6",
        step="risk-mpc",
        descriptor="scenario-sweep-main",
    )
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")
    shutil.copy2(observer_config, logger.run_dir / "observer_config.yaml")

    rows: list[dict[str, float | str | int]] = []
    runs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for spec in PHASE6_SCENARIOS:
        t, roads, a_x, a_y, f_c = _build_inputs(plant, spec, dt)
        risk = _risk_params_for(spec)
        qp = RiskQPParams(
            rho_safe=0.85,
            enforce_rate=False,
            r_du_factor=1.0e-5,
            osqp_eps_abs=1.0e-5,
            osqp_eps_rel=1.0e-5,
            osqp_max_iter=100000,
        )
        comfort = _run_comfort(FullCarComfortQP(vehicle, lqr), plant, t, roads, a_x, a_y, f_c, spec.mu)
        mpc = run_mpc_residual_closed_loop(
            FullCarRiskMPC(vehicle, lqr, risk, qp, horizon=10, warm_start=False, reuse_solver=True),
            plant,
            t,
            roads,
            a_y,
            f_c,
            spec.mu,
            observer_params,
            prediction_mode=RESIDUAL_DECAY,
            alpha=0.95,
            fz_tightening_n=observer_params.delta_fz_err,
            a_x=a_x,
        )
        runs[spec.name] = {"comfort": comfort, "risk_mpc": mpc}
        row = _summarize_scenario(spec, comfort, mpc, lqr.f_max)
        rows.append(row)
        logger.log(
            f"{spec.name}: comfort_rho={row['comfort_peak_rho']:.3f}, "
            f"mpc_rho={row['mpc_peak_rho']:.3f}, reduction={100.0*row['rho_reduction_ratio']:.1f}%, "
            f"heave={row['comfort_heave_rms']:.3f}->{row['mpc_heave_rms']:.3f}, "
            f"maxF={row['mpc_max_force_n']:.0f}N"
        )

    _write_summary(logger.run_dir / "phase6_summary.csv", rows)
    _plot_summary_bars(logger.run_dir / "figures" / "phase6_rho_summary.png", rows)
    _plot_heave_bars(logger.run_dir / "figures" / "phase6_heave_tradeoff.png", rows)
    for scenario_name in ("S7_brake_cornering", "S8_worst_case", "S9_corner_bump"):
        spec = next(s for s in PHASE6_SCENARIOS if s.name == scenario_name)
        t, *_ = _build_inputs(plant, spec, dt)
        _plot_scenario_timeseries(
            logger.run_dir / "figures" / f"{scenario_name}_timeseries.png",
            t,
            runs[scenario_name]["comfort"],
            runs[scenario_name]["risk_mpc"],
            spec,
        )

    low_risk = [r for r in rows if r["risk_level"] == "low"]
    medium_risk = [r for r in rows if r["risk_level"] == "medium"]
    high_risk = [r for r in rows if r["risk_level"] == "high"]
    logger.log_kv("acceptance_low_risk_mpc_rho_change_lt_10pct", int(all(abs(float(r["rho_reduction_ratio"])) < 0.10 for r in low_risk)))
    logger.log_kv("acceptance_medium_risk_peak_change_lt_10pct", int(all(abs(float(r["rho_reduction_ratio"])) < 0.10 for r in medium_risk)))
    logger.log_kv("acceptance_S9_rho_reduction_ge_15pct", int(next(float(r["rho_reduction_ratio"]) for r in rows if r["scenario"] == "S9_corner_bump") >= 0.15))
    logger.log_kv("acceptance_any_high_risk_reduction_ge_15pct", int(any(float(r["rho_reduction_ratio"]) >= 0.15 for r in high_risk)))
    logger.log("Phase 6 main scenario sweep completed.")
    return logger.run_dir


def _build_inputs(
    plant: FullCar,
    spec: Phase6Scenario,
    dt: float,
    v_x: float = 80.0 / 3.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t = np.arange(0.0, spec.duration, dt)
    roads = np.zeros((len(t), 4), dtype=float)
    if spec.road_class is not None:
        roads = _straight_two_track_road(plant, t=t, dt=dt, v_x=v_x, road_class=spec.road_class)
        roads *= spec.road_scale
    if spec.bump_wheel is not None:
        roads[:, spec.bump_wheel] += rounded_bump(
            t, height=spec.bump_height, start=spec.bump_start, duration=spec.bump_duration
        )
    a_x = np.zeros_like(t)
    if spec.peak_ax != 0.0:
        a_x = spec.peak_ax * _smooth_step(t, start=0.8, rise_time=0.35)
    if spec.lane_change:
        a_y = lane_change_a_y(t, peak_ay=spec.peak_ay, start=1.0, duration=2.2)
    else:
        a_y = spec.peak_ay * _smooth_step(t, start=0.8, rise_time=0.5)
    if spec.coupled_lateral_demand:
        f_c = np.array([_coupled_f_c(plant, float(a)) for a in a_y])
    else:
        f_c = np.tile(np.asarray(spec.f_c, dtype=float), (len(t), 1))
    return t, roads, a_x, a_y, f_c


def _smooth_step(t: np.ndarray, start: float, rise_time: float) -> np.ndarray:
    tau = np.clip((t - start) / rise_time, 0.0, 1.0)
    return tau * tau * (3.0 - 2.0 * tau)


def _risk_params_for(spec: Phase6Scenario) -> RiskWeightParams:
    q_p_max = Q_P_MAX_BY_RISK[_risk_level_for(spec)]
    return RiskWeightParams(
        rho_th=0.75,
        k_rho=25.0,
        kappa_rho=0.0,
        q_c_min=1.0,
        q_c_max=1.0,
        q_p_min=0.0,
        q_p_max=q_p_max,
    )


def _risk_level_for(spec: Phase6Scenario) -> str:
    if spec.name.startswith(LOW_RISK_PREFIXES):
        return "low"
    if spec.name.startswith(MEDIUM_RISK_PREFIXES):
        return "medium"
    if spec.name.startswith(HIGH_RISK_PREFIXES):
        return "high"
    raise ValueError(f"unknown scenario risk level for {spec.name!r}")


def _run_comfort(controller, plant: FullCar, t: np.ndarray, roads: np.ndarray, a_x: np.ndarray, a_y: np.ndarray, f_c: np.ndarray, mu: float) -> dict[str, np.ndarray]:
    states = np.zeros((len(t), 14), dtype=float)
    forces = np.zeros((len(t), 4), dtype=float)
    fz_true = np.zeros((len(t), 4), dtype=float)
    rho_contact = np.zeros((len(t), 4), dtype=float)
    state = np.zeros(14)
    # Update the comfort-QP at the same control period T_s as the Risk-MPC
    # (zero-order hold between updates) so the two controllers are compared at
    # an identical sampling rate. Recomputing the LQR feedback every plant
    # integration step would give the baseline an unfair 5x update advantage.
    dt = float(t[1] - t[0])
    steps_per_update = max(1, int(round(controller.lqr.T_s / dt)))
    force = np.zeros(4, dtype=float)
    for idx in range(len(t) - 1):
        if idx % steps_per_update == 0:
            force = controller.compute(state)
        forces[idx] = force
        fz_true[idx] = plant.tire_normal_forces(state, roads[idx])
        rho_contact[idx] = rho(f_c[idx], 0.0, fz_true[idx], mu)
        state = plant.step(state, roads[idx], float(t[idx + 1] - t[idx]), u=force, body_acc=np.array([a_x[idx], a_y[idx]]))
        states[idx + 1] = state
    fz_true[-1] = plant.tire_normal_forces(states[-1], roads[-1])
    rho_contact[-1] = rho(f_c[-1], 0.0, fz_true[-1], mu)
    body_accel = np.array([
        plant.body_accelerations(s, r, u=u, body_acc=np.array([ax, ay]))
        for s, r, u, ax, ay in zip(states, roads, forces, a_x, a_y)
    ])
    return {"states": states, "forces": forces, "fz_true": fz_true, "rho_contact": rho_contact, "body_accel": body_accel}


def _summarize_scenario(spec: Phase6Scenario, comfort: dict, mpc: dict, f_max: float) -> dict[str, float | str | int]:
    comfort_peak = float(np.max(comfort["rho_contact"]))
    mpc_peak = float(np.max(mpc["rho_contact"]))
    return {
        "scenario": spec.name,
        "description": spec.description,
        "risk_level": _risk_level_for(spec),
        "q_p_max": Q_P_MAX_BY_RISK[_risk_level_for(spec)],
        "mu": spec.mu,
        "comfort_peak_rho": comfort_peak,
        "mpc_peak_rho": mpc_peak,
        "rho_reduction_ratio": (comfort_peak - mpc_peak) / max(comfort_peak, 1.0e-9),
        "comfort_rho_p95": float(np.percentile(np.max(comfort["rho_contact"], axis=1), 95)),
        "mpc_rho_p95": float(np.percentile(np.max(mpc["rho_contact"], axis=1), 95)),
        "comfort_heave_rms": float(rms(comfort["body_accel"][:, 0])),
        "mpc_heave_rms": float(rms(mpc["body_accel"][:, 0])),
        "heave_degradation_ratio": (float(rms(mpc["body_accel"][:, 0])) - float(rms(comfort["body_accel"][:, 0]))) / max(float(rms(comfort["body_accel"][:, 0])), 1.0e-9),
        "mpc_max_force_n": float(np.max(np.abs(mpc["forces"]))),
        "mpc_force_saturation": int(float(np.max(np.abs(mpc["forces"]))) >= 0.95 * f_max),
        "comfort_min_fz_n": float(np.min(comfort["fz_true"])),
        "mpc_min_fz_n": float(np.min(mpc["fz_true"])),
        "mpc_p95_solve_time_ms": float(np.percentile(mpc["solve_time"][mpc["solve_time"] > 0.0], 95) * 1000.0),
        "mpc_stable": int(np.all(np.isfinite(mpc["states"]))),
    }


def _write_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_summary_bars(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    names = [str(r["scenario"]).split("_", 1)[0] for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(x - width / 2, [float(r["comfort_peak_rho"]) for r in rows], width, label="comfort-QP", color="tab:red")
    ax.bar(x + width / 2, [float(r["mpc_peak_rho"]) for r in rows], width, label="risk-MPC", color="tab:blue")
    ax.axhline(0.85, color="0.2", linestyle="--", linewidth=1.0, label="rho_safe")
    ax.set_xticks(x, names)
    ax.set_ylabel("Peak rho")
    ax.set_title("Phase 6 scenario sweep: peak tire-utilization")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_heave_bars(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    names = [str(r["scenario"]).split("_", 1)[0] for r in rows]
    x = np.arange(len(rows))
    width = 0.38
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(x - width / 2, [float(r["comfort_heave_rms"]) for r in rows], width, label="comfort-QP", color="tab:red")
    ax.bar(x + width / 2, [float(r["mpc_heave_rms"]) for r in rows], width, label="risk-MPC", color="tab:blue")
    ax.set_xticks(x, names)
    ax.set_ylabel("Heave RMS [m/s^2]")
    ax.set_title("Phase 6 scenario sweep: comfort trade-off")
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def _plot_scenario_timeseries(path: Path, t: np.ndarray, comfort: dict, mpc: dict, spec: Phase6Scenario) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8.0, 8.0), sharex=True)
    axes[0].plot(t, np.max(comfort["rho_contact"], axis=1), label="comfort-QP", color="tab:red")
    axes[0].plot(t, np.max(mpc["rho_contact"], axis=1), label="risk-MPC", color="tab:blue")
    axes[0].axhline(0.85, color="0.2", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("rho_max")
    axes[0].legend(loc="best")
    axes[1].plot(t, _moving_average(comfort["body_accel"][:, :1], 21)[:, 0], label="comfort-QP", color="tab:red")
    axes[1].plot(t, _moving_average(mpc["body_accel"][:, :1], 21)[:, 0], label="risk-MPC", color="tab:blue")
    axes[1].set_ylabel("Heave accel [m/s^2]")
    axes[2].plot(t, _moving_average(mpc["forces"], 21))
    axes[2].set_ylabel("MPC force [N]")
    axes[2].set_xlabel("Time [s]")
    fig.suptitle(f"{spec.name}: {spec.description}")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    print(run_phase_6_main_sweep())


if __name__ == "__main__":
    main()

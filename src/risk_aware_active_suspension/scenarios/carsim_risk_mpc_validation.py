from __future__ import annotations

import csv
import shutil
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import RESIDUAL_DECAY, FullCarRiskMPC, predict_fz_horizon
from risk_aware_active_suspension.controllers.risk_qp import RiskQPParams
from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.metrics.tire import rho
from risk_aware_active_suspension.plants.carsim_fmu import CarSimFmuPlant
from risk_aware_active_suspension.plants.full_car import CORNER_NAMES, FullCar
from risk_aware_active_suspension.scenarios.carsim_passthrough import DEFAULT_FMU_PATH
from risk_aware_active_suspension.scenarios.risk_mpc_scenario_sweep import (
    PHASE6_SCENARIOS,
    _build_inputs,
    _risk_params_for,
)
from risk_aware_active_suspension.utils.config import from_yaml, lqr_from_yaml, observer_from_yaml
from risk_aware_active_suspension.utils.logger import RunLogger


G = 9.81
CONTROL_WARMUP_S = 0.10
S8_COMBINED_DEMAND_SCALE = 0.45
OUTPUTS = (
    "Vx",
    "Ax",
    "Ay",
    "Az_SM",
    "Zcg_SM",
    "Vz_SM",
    "Roll",
    "Pitch",
    "AVx",
    "AVy",
    "Fz_L1",
    "Fz_R1",
    "Fz_L2",
    "Fz_R2",
    "Fx_L1",
    "Fx_R1",
    "Fx_L2",
    "Fx_R2",
    "Fy_L1",
    "Fy_R1",
    "Fy_L2",
    "Fy_R2",
    "Z_L1",
    "Z_R1",
    "Z_L2",
    "Z_R2",
    "Vz_WC_L1",
    "Vz_WC_R1",
    "Vz_WC_L2",
    "Vz_WC_R2",
    "Alpha_L1",
    "Alpha_R1",
    "Alpha_L2",
    "Alpha_R2",
    "Kappa_L1",
    "Kappa_R1",
    "Kappa_L2",
    "Kappa_R2",
)


def run_phase_7_4_risk_mpc_validation(
    fmu_path: str | Path = DEFAULT_FMU_PATH,
    vehicle_config: str | Path = "configs/vehicle_default.yaml",
    controller_config: str | Path = "configs/controller_default.yaml",
    observer_config: str | Path = "configs/observer_default.yaml",
    results_root: str | Path = "results",
    sample_time: float = 0.005,
    speed_kmh: float = 80.0,
    scenario_names: tuple[str, ...] = ("S4_classC_cornering", "S8_worst_case"),
    fmu_force_sign: float = -1.0,
    carsim_force_limit_n: float = 4000.0,
    carsim_enforce_rate: bool = True,
    carsim_df_max_n_per_s: float = 8000.0,
    carsim_risk_tuning: str = "analytical",
) -> Path:
    """Run Phase 7.4 risk-aware MPC closed-loop validation on the CarSim FMU."""

    vehicle_config = Path(vehicle_config)
    controller_config = Path(controller_config)
    observer_config = Path(observer_config)
    vehicle = from_yaml(vehicle_config)
    lqr = replace(lqr_from_yaml(controller_config), f_max=float(carsim_force_limit_n))
    observer = observer_from_yaml(observer_config)
    if carsim_risk_tuning not in {"carsim", "analytical"}:
        raise ValueError("carsim_risk_tuning must be 'carsim' or 'analytical'.")
    full_car = FullCar(vehicle)
    carsim = CarSimFmuPlant(Path(fmu_path))
    logger = RunLogger.create(results_root, phase="phase-7.4", step="carsim", descriptor="risk-mpc-validation")
    shutil.copy2(vehicle_config, logger.run_dir / "vehicle_config.yaml")
    shutil.copy2(controller_config, logger.run_dir / "controller_config.yaml")
    shutil.copy2(observer_config, logger.run_dir / "observer_config.yaml")

    rows: list[dict[str, float | str | int]] = []
    runs: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for scenario_name in scenario_names:
        spec = next(item for item in PHASE6_SCENARIOS if item.name == scenario_name)
        t, roads, a_x, a_y, f_c = _build_carsim_inputs(full_car, spec, sample_time, speed_kmh)
        risk = _risk_params_for(spec) if carsim_risk_tuning == "analytical" else _carsim_risk_params_for(spec)
        qp = RiskQPParams(
            rho_safe=0.85,
            enforce_rate=bool(carsim_enforce_rate),
            df_max=float(carsim_df_max_n_per_s),
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
            spec.mu,
            spec.mu,
            speed_kmh,
            sample_time,
            observer.delta_fz_err,
            fmu_force_sign,
            carsim_df_max_n_per_s if carsim_enforce_rate else None,
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
            spec.mu,
            spec.mu,
            speed_kmh,
            sample_time,
            observer.delta_fz_err,
            fmu_force_sign,
            carsim_df_max_n_per_s if carsim_enforce_rate else None,
        )
        runs[scenario_name] = {"comfort_qp": comfort, "risk_mpc": mpc}
        row = _summarize(spec.name, spec.mu, spec.mu, comfort, mpc, lqr.f_max)
        rows.append(row)
        logger.log(
            f"{spec.name}: comfort_rho={row['comfort_peak_rho']:.3f}, "
            f"mpc_rho={row['mpc_peak_rho']:.3f}, reduction={100.0 * row['rho_reduction_ratio']:.1f}%, "
            f"heave={row['comfort_heave_rms_m_s2']:.3f}->{row['mpc_heave_rms_m_s2']:.3f}, "
            f"maxF={row['mpc_max_force_n']:.0f}N"
        )
        _plot_scenario(logger.run_dir / "figures" / f"{spec.name}_closed_loop.png", t, comfort, mpc, spec.name)

    _write_summary(logger.run_dir / "phase7_4_summary.csv", rows)
    _save_raw(logger.run_dir / "raw.npz", runs)
    for key, value in _aggregate_metrics(rows).items():
        logger.log_kv(key, value)
    logger.log_kv("carsim_force_limit_n", carsim_force_limit_n)
    logger.log_kv("carsim_enforce_rate", int(carsim_enforce_rate))
    logger.log_kv("carsim_df_max_n_per_s", carsim_df_max_n_per_s)
    logger.log(f"CarSim risk tuning: {carsim_risk_tuning}")
    logger.log("Phase 7.4 CarSim risk-aware MPC validation completed.")
    return logger.run_dir


def _run_closed_loop(
    carsim: CarSimFmuPlant,
    full_car: FullCar,
    controller,
    controller_name: str,
    t: np.ndarray,
    roads: np.ndarray,
    a_x: np.ndarray,
    a_y: np.ndarray,
    f_c: np.ndarray,
    plant_mu: float,
    controller_mu: float,
    speed_kmh: float,
    sample_time: float,
    fz_tightening_n: float,
    fmu_force_sign: float,
    force_rate_limit_n_per_s: float | None = None,
) -> dict[str, np.ndarray]:
    n = len(t)
    states = np.zeros((n, 14), dtype=float)
    forces = np.zeros((n, 4), dtype=float)
    fmu_forces = np.zeros((n, 4), dtype=float)
    fz_true = np.zeros((n, 4), dtype=float)
    fx_true = np.zeros((n, 4), dtype=float)
    fy_true = np.zeros((n, 4), dtype=float)
    alpha = np.zeros((n, 4), dtype=float)
    kappa = np.zeros((n, 4), dtype=float)
    rho_contact = np.zeros((n, 4), dtype=float)
    az = np.zeros(n, dtype=float)
    vx = np.zeros(n, dtype=float)
    ax_g = np.zeros(n, dtype=float)
    ay_g = np.zeros(n, dtype=float)
    solve_time = np.zeros(n, dtype=float)
    iterations = np.zeros(n, dtype=int)
    sigma = np.zeros(n, dtype=float)
    force_prev = np.zeros(4, dtype=float)
    baseline: dict[str, float] | None = None

    initial_inputs = _input_values(carsim, speed_kmh, steer=0.0, force=np.zeros(4), brake=np.zeros(4), road=roads[0], mu=plant_mu)
    with carsim.session(start_time=float(t[0]), stop_time=float(t[-1]), initial_inputs=initial_inputs) as session:
        for idx in range(n - 1):
            out = session.get_reals(OUTPUTS)
            if baseline is None and out["Vx"] > 1.0:
                baseline = out.copy()
            state = _state_from_carsim(carsim, out, baseline)
            states[idx] = state
            fz_now = _corner_output(carsim, out, carsim.signal_map.fz)
            if np.all(fz_now <= 1.0):
                fz_now = full_car.static_loads()
            fz_true[idx] = fz_now
            fx_true[idx] = _corner_output(carsim, out, carsim.signal_map.fx)
            fy_true[idx] = _corner_output(carsim, out, carsim.signal_map.fy)
            alpha[idx] = _corner_output(carsim, out, carsim.signal_map.alpha)
            kappa[idx] = _corner_output(carsim, out, carsim.signal_map.kappa)
            vx[idx] = out["Vx"]
            ax_g[idx] = out["Ax"]
            ay_g[idx] = out["Ay"]
            az[idx] = out["Az_SM"] * G
            rho_contact[idx] = rho(f_c[idx], 0.0, np.maximum(fz_now, 1.0), plant_mu)

            force = np.zeros(4, dtype=float)
            if t[idx] >= CONTROL_WARMUP_S:
                if controller_name == "risk_mpc":
                    horizon = controller.horizon
                    fc_h = _horizon(f_c, idx, horizon)
                    fz_h = predict_fz_horizon(
                        np.tile(fz_now, (horizon, 1)),
                        np.zeros(4),
                        mode=RESIDUAL_DECAY,
                        alpha=0.95,
                    )
                    force = controller.compute(
                        state,
                        f_z_hat=fz_h,
                        f_c=fc_h,
                        mu=controller_mu,
                    )
                    solve_time[idx] = controller.last_solve_time_s
                    iterations[idx] = controller.last_iterations
                    sigma[idx] = controller.last_sigma
                else:
                    force = controller.compute(state)
            if force_rate_limit_n_per_s is not None:
                du_max = float(force_rate_limit_n_per_s) * float(sample_time)
                force = np.clip(force, force_prev - du_max, force_prev + du_max)
            forces[idx] = force
            force_prev = force.copy()
            fmu_force = float(fmu_force_sign) * force
            fmu_forces[idx] = fmu_force
            session.set_reals(
                _input_values(
                    carsim,
                    speed_kmh,
                    steer=float(_steer_deg_from_ay(a_y[idx])),
                    force=fmu_force,
                    brake=np.full(4, _brake_mpa_from_ax(a_x[idx])),
                    road=roads[idx],
                    mu=plant_mu,
                )
            )
            session.do_step(float(t[idx]), sample_time)

        out = session.get_reals(OUTPUTS)
        states[-1] = _state_from_carsim(carsim, out, baseline)
        fz_true[-1] = _corner_output(carsim, out, carsim.signal_map.fz)
        fx_true[-1] = _corner_output(carsim, out, carsim.signal_map.fx)
        fy_true[-1] = _corner_output(carsim, out, carsim.signal_map.fy)
        alpha[-1] = _corner_output(carsim, out, carsim.signal_map.alpha)
        kappa[-1] = _corner_output(carsim, out, carsim.signal_map.kappa)
        vx[-1] = out["Vx"]
        ax_g[-1] = out["Ax"]
        ay_g[-1] = out["Ay"]
        az[-1] = out["Az_SM"] * G
        rho_contact[-1] = rho(f_c[-1], 0.0, np.maximum(fz_true[-1], 1.0), plant_mu)

    return {
        "time": t,
        "states": states,
        "forces": forces,
        "fmu_forces": fmu_forces,
        "fz_true": fz_true,
        "fx_true": fx_true,
        "fy_true": fy_true,
        "alpha_deg": alpha,
        "kappa": kappa,
        "rho_contact": rho_contact,
        "az_m_s2": az,
        "vx_kmh": vx,
        "ax_g": ax_g,
        "ay_g": ay_g,
        "solve_time": solve_time,
        "iterations": iterations,
        "sigma": sigma,
        "roads": roads,
        "f_c": f_c,
    }


def _input_values(
    carsim: CarSimFmuPlant,
    speed: float,
    steer: float,
    force: np.ndarray,
    brake: np.ndarray,
    road: np.ndarray,
    mu: float,
) -> dict[str, float]:
    signals = carsim.signal_map
    values = {name: 0.0 for name in signals.input_names}
    values[signals.speed] = float(speed)
    values[signals.steer] = float(steer)
    for names, data in (
        (signals.active_force, force),
        (signals.brake_pressure, brake),
        (signals.road_height, road),
    ):
        for name, value in zip(names, np.asarray(data, dtype=float)):
            values[name] = float(value)
    for names in (signals.mu_x, signals.mu_y):
        for name in names:
            values[name] = float(mu)
    return values


def _state_from_carsim(carsim: CarSimFmuPlant, out: dict[str, float], baseline: dict[str, float] | None) -> np.ndarray:
    if baseline is None:
        return np.zeros(14, dtype=float)
    state = np.zeros(14, dtype=float)
    state[0] = out["Zcg_SM"] - baseline["Zcg_SM"]
    state[1] = out["Vz_SM"] / 3.6
    state[2] = np.deg2rad(out["Roll"] - baseline["Roll"])
    state[3] = np.deg2rad(out["AVx"])
    state[4] = np.deg2rad(out["Pitch"] - baseline["Pitch"])
    state[5] = np.deg2rad(out["AVy"])
    wheel_z = _corner_output(carsim, out, carsim.signal_map.wheel_z)
    wheel_z0 = _corner_output(carsim, baseline, carsim.signal_map.wheel_z)
    wheel_vz = _corner_output(carsim, out, carsim.signal_map.wheel_vz) / 3.6
    for corner_idx in range(4):
        state[6 + 2 * corner_idx] = wheel_z[corner_idx] - wheel_z0[corner_idx]
        state[7 + 2 * corner_idx] = wheel_vz[corner_idx]
    return state


def _corner_output(carsim: CarSimFmuPlant, out: dict[str, float], names: tuple[str, str, str, str]) -> np.ndarray:
    return np.array([out[name] for name in names], dtype=float)


def _horizon(values: np.ndarray, idx: int, horizon: int) -> np.ndarray:
    end = min(idx + horizon, len(values))
    out = np.asarray(values[idx:end], dtype=float)
    if len(out) < horizon:
        pad = np.repeat(out[-1:], horizon - len(out), axis=0)
        out = np.vstack([out, pad])
    return out


def _steer_deg_from_ay(a_y: float) -> float:
    return float(np.clip(1.5 * a_y, -10.0, 10.0))


def _brake_mpa_from_ax(a_x: float) -> float:
    return float(np.clip(-a_x / 10.0, 0.0, 1.0))


def _build_carsim_inputs(
    full_car: FullCar,
    spec,
    sample_time: float,
    speed_kmh: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, roads, a_x, a_y, f_c = _build_inputs(full_car, spec, sample_time, v_x=speed_kmh / 3.6)
    return t, roads, a_x, a_y, _adapt_carsim_friction_demand(full_car, spec.name, a_x, a_y, f_c)


def _adapt_carsim_friction_demand(
    full_car: FullCar,
    scenario_name: str,
    a_x: np.ndarray,
    a_y: np.ndarray,
    f_c: np.ndarray,
) -> np.ndarray:
    if scenario_name != "S8_worst_case":
        return f_c
    vehicle = full_car.params
    fx_front = 0.65 * vehicle.m * np.abs(a_x) / 2.0
    fx_rear = 0.35 * vehicle.m * np.abs(a_x) / 2.0
    fy_front = 0.55 * vehicle.m * np.abs(a_y) / 2.0
    fy_rear = 0.45 * vehicle.m * np.abs(a_y) / 2.0
    combined = S8_COMBINED_DEMAND_SCALE * np.column_stack(
        [
            np.hypot(fx_front, fy_front),
            np.hypot(fx_front, fy_front),
            np.hypot(fx_rear, fy_rear),
            np.hypot(fx_rear, fy_rear),
        ]
    )
    return np.maximum(f_c, combined)


def _carsim_risk_params_for(spec) -> object:
    base = _risk_params_for(spec)
    q_p_max = 0.05 if spec.name == "S8_worst_case" else 0.01
    return replace(base, rho_th=0.60, k_rho=35.0, q_p_max=q_p_max)


def _summarize(
    scenario: str,
    plant_mu: float,
    controller_mu: float,
    comfort: dict[str, np.ndarray],
    mpc: dict[str, np.ndarray],
    f_max: float,
) -> dict[str, float | str | int]:
    valid = _valid_window(comfort)
    comfort_peak = float(np.max(comfort["rho_contact"][valid]))
    mpc_peak = float(np.max(mpc["rho_contact"][valid]))
    solve_samples = mpc["solve_time"][mpc["solve_time"] > 0.0]
    return {
        "scenario": scenario,
        "plant_mu": plant_mu,
        "controller_mu": controller_mu,
        "comfort_peak_rho": comfort_peak,
        "mpc_peak_rho": mpc_peak,
        "rho_reduction_ratio": (comfort_peak - mpc_peak) / max(comfort_peak, 1.0e-9),
        "comfort_rho_p95": float(np.percentile(np.max(comfort["rho_contact"][valid], axis=1), 95)),
        "mpc_rho_p95": float(np.percentile(np.max(mpc["rho_contact"][valid], axis=1), 95)),
        "comfort_min_fz_n": float(np.min(comfort["fz_true"][valid])),
        "mpc_min_fz_n": float(np.min(mpc["fz_true"][valid])),
        "comfort_heave_rms_m_s2": float(rms(comfort["az_m_s2"][valid])),
        "mpc_heave_rms_m_s2": float(rms(mpc["az_m_s2"][valid])),
        "mpc_max_force_n": float(np.max(np.abs(mpc["forces"][valid]))),
        "mpc_force_saturation": int(float(np.max(np.abs(mpc["forces"][valid]))) >= 0.95 * f_max),
        "mpc_p95_solve_time_ms": float(np.percentile(solve_samples, 95) * 1000.0) if len(solve_samples) else 0.0,
        "mpc_stable": int(np.all(np.isfinite(mpc["states"])) and np.all(np.isfinite(mpc["fz_true"]))),
        "mean_vx_kmh": float(np.mean(mpc["vx_kmh"][valid])),
    }


def _write_summary(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_metrics(rows: list[dict[str, float | str | int]]) -> dict[str, float | int]:
    reductions = [float(row["rho_reduction_ratio"]) for row in rows]
    return {
        "min_rho_reduction_ratio": float(np.min(reductions)),
        "max_mpc_peak_rho": float(max(float(row["mpc_peak_rho"]) for row in rows)),
        "max_mpc_p95_solve_time_ms": float(max(float(row["mpc_p95_solve_time_ms"]) for row in rows)),
        "acceptance_all_mpc_stable": int(all(int(row["mpc_stable"]) for row in rows)),
        "acceptance_any_rho_reduction_positive": int(any(value > 0.0 for value in reductions)),
    }


def _valid_window(run: dict[str, np.ndarray]) -> np.ndarray:
    t = run["time"]
    return (t >= 0.5) & (run["vx_kmh"] > 1.0) & np.all(run["fz_true"] > 100.0, axis=1)


def _save_raw(path: Path, runs: dict[str, dict[str, dict[str, np.ndarray]]]) -> None:
    payload = {}
    for scenario_name, scenario_runs in runs.items():
        for controller_name, data in scenario_runs.items():
            for key, value in data.items():
                payload[f"{scenario_name}_{controller_name}_{key}"] = value
    np.savez(path, **payload)


def _plot_scenario(path: Path, t: np.ndarray, comfort: dict[str, np.ndarray], mpc: dict[str, np.ndarray], title: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 9.0), sharex=True)
    axes[0].plot(t, np.max(comfort["rho_contact"], axis=1), label="comfort-QP")
    axes[0].plot(t, np.max(mpc["rho_contact"], axis=1), label="risk-MPC")
    axes[0].axhline(0.85, color="0.2", linestyle="--", linewidth=1.0)
    axes[0].set_ylabel("rho_max")
    axes[0].legend(fontsize=8)
    axes[1].plot(t, comfort["az_m_s2"], label="comfort-QP")
    axes[1].plot(t, mpc["az_m_s2"], label="risk-MPC")
    axes[1].set_ylabel("Az [m/s^2]")
    axes[1].legend(fontsize=8)
    axes[2].plot(t, mpc["forces"])
    axes[2].set_ylabel("MPC force [N]")
    axes[2].legend(CORNER_NAMES, ncol=4, fontsize=7)
    axes[3].plot(t, mpc["vx_kmh"])
    axes[3].set_ylabel("Vx [km/h]")
    axes[3].set_xlabel("Time [s]")
    fig.suptitle(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main() -> None:
    print(run_phase_7_4_risk_mpc_validation())


if __name__ == "__main__":
    main()

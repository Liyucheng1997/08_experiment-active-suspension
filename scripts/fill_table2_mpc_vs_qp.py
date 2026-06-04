"""Fill Table II with a fair Risk-MPC horizon ablation.

Runs Comfort-QP, Risk-MPC (N_p=1), and Risk-MPC (N_p=10) on the three
high-risk scenarios from Phase 6. Computes peak rho_max and heave RMS
over the maneuver window and prints a paste-ready LaTeX table. Raw data
for Fig. 8 are written to results/paper_table2_horizon_ablation_current.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(r"F:/我的科研工作/05_Risk_Aware_Active_Suspension")
sys.path.insert(0, str(PROJECT / "src"))

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import RESIDUAL_DECAY, FullCarRiskMPC
from risk_aware_active_suspension.controllers.risk_qp import RiskQPParams
from risk_aware_active_suspension.metrics.signals import rms
from risk_aware_active_suspension.metrics.tire import rho as rho_fn
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.risk_mpc_residual_prediction import run_mpc_residual_closed_loop
from risk_aware_active_suspension.scenarios.risk_mpc_scenario_sweep import (
    PHASE6_SCENARIOS, _build_inputs, _risk_params_for,
)
from risk_aware_active_suspension.scenarios.risk_qp_fullcar_ground_truth import _ground_truth_fz
from risk_aware_active_suspension.utils.config import from_yaml, lqr_from_yaml, observer_from_yaml


OUT_DIR = PROJECT / "results" / "paper_table2_horizon_ablation_current"


def _run_comfort(plant: FullCar, t, roads, a_x, a_y, f_c, mu, controller):
    n = len(t)
    dt = float(t[1] - t[0])
    steps_per_update = max(1, int(round(controller.lqr.T_s / dt)))
    states = np.zeros((n, 14)); forces = np.zeros((n, 4))
    fz_true = np.zeros((n, 4)); rho_c = np.zeros((n, 4))
    state = np.zeros(14); u = np.zeros(4)
    for idx in range(n - 1):
        fz_true[idx] = _ground_truth_fz(plant, state, roads[idx])
        rho_c[idx] = rho_fn(f_c[idx], 0.0, fz_true[idx], mu)
        if idx % steps_per_update == 0:
            u = controller.compute(state)
        forces[idx] = u
        state = plant.step(state, roads[idx], dt, u=u,
                           body_acc=np.array([a_x[idx], a_y[idx]]))
        states[idx + 1] = state
    fz_true[-1] = _ground_truth_fz(plant, states[-1], roads[-1])
    rho_c[-1] = rho_fn(f_c[-1], 0.0, fz_true[-1], mu)
    body_accel = np.array([
        plant.body_accelerations(s, w, u=f, body_acc=np.array([ax, ay]))
        for s, w, f, ax, ay in zip(states, roads, forces, a_x, a_y)
    ])
    return {"states": states, "forces": forces, "rho_contact": rho_c, "body_accel": body_accel}


def _run_mpc_horizon(vehicle, lqr, risk, qp_params, plant, t, roads, a_x, a_y, f_c, mu, observer_params, horizon: int):
    return run_mpc_residual_closed_loop(
        FullCarRiskMPC(
            vehicle, lqr, risk, qp_params, horizon=horizon,
            warm_start=False, reuse_solver=True,
        ),
        plant, t, roads, a_y, f_c, mu, observer_params,
        prediction_mode=RESIDUAL_DECAY, alpha=0.95,
        fz_tightening_n=observer_params.delta_fz_err, a_x=a_x,
    )


def _metrics(run: dict[str, np.ndarray], win: np.ndarray) -> tuple[float, float]:
    peak = float(np.max(run["rho_contact"][win].max(axis=1)))
    heave = float(rms(run["body_accel"][win, 0]))
    return peak, heave


def main() -> None:
    vehicle = from_yaml(PROJECT / "configs/vehicle_default.yaml")
    lqr = lqr_from_yaml(PROJECT / "configs/controller_default.yaml")
    observer_params = observer_from_yaml(PROJECT / "configs/observer_default.yaml")
    plant = FullCar(vehicle)
    dt = 0.001
    target = {"S7_brake_cornering": "S7",
              "S8_worst_case":      "S8",
              "S9_corner_bump":     "S9"}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_rows: dict[str, dict[str, float | str | np.ndarray]] = {}
    csv_rows: list[dict[str, str | float]] = []
    for spec in PHASE6_SCENARIOS:
        if spec.name not in target: continue
        short = target[spec.name]
        print(f"\n=== {short} ({spec.name}) ===")
        t, roads, a_x, a_y, f_c = _build_inputs(plant, spec, dt)
        risk = _risk_params_for(spec)
        qp_params = RiskQPParams(
            rho_safe=0.85, enforce_rate=False, r_du_factor=1.0e-5,
            osqp_eps_abs=1.0e-5, osqp_eps_rel=1.0e-5, osqp_max_iter=100000,
        )

        # --- Comfort-QP baseline ---
        comfort = _run_comfort(plant, t, roads, a_x, a_y, f_c, spec.mu,
                               FullCarComfortQP(vehicle, lqr))

        # --- Same Risk-MPC implementation, only the horizon differs. ---
        mpc_n1 = _run_mpc_horizon(
            vehicle, lqr, risk, qp_params, plant, t, roads, a_x, a_y, f_c,
            spec.mu, observer_params, horizon=1,
        )
        mpc_n10 = _run_mpc_horizon(
            vehicle, lqr, risk, qp_params, plant, t, roads, a_x, a_y, f_c,
            spec.mu, observer_params, horizon=10,
        )

        # --- Metrics over the maneuver window (exclude first 0.5 s) ---
        win = (t >= 0.5) & (t <= t[-1])
        comfort_peak, comfort_heave = _metrics(comfort, win)
        n1_peak, n1_heave = _metrics(mpc_n1, win)
        n10_peak, n10_heave = _metrics(mpc_n10, win)

        print(f"  Comfort-QP: peak={comfort_peak:.3f}  heave={comfort_heave:.3f}")
        print(f"  Risk-MPC Np=1: peak={n1_peak:.3f}  heave={n1_heave:.3f}")
        print(f"  Risk-MPC Np=10: peak={n10_peak:.3f}  heave={n10_heave:.3f}")

        table_rows[short] = {
            "comfort_peak": f"{comfort_peak:.3f}", "comfort_heave": f"{comfort_heave:.3f}",
            "n1_peak": f"{n1_peak:.3f}", "n1_heave": f"{n1_heave:.3f}",
            "n10_peak": f"{n10_peak:.3f}", "n10_heave": f"{n10_heave:.3f}",
        }
        csv_rows.append({
            "scenario": short,
            "comfort_peak_rho": comfort_peak,
            "comfort_heave_rms": comfort_heave,
            "n1_peak_rho": n1_peak,
            "n1_heave_rms": n1_heave,
            "n10_peak_rho": n10_peak,
            "n10_heave_rms": n10_heave,
        })

        if short == "S9":
            np.savez(
                OUT_DIR / "raw_s9.npz",
                t=t,
                roads=roads,
                a_x=a_x,
                a_y=a_y,
                f_c=f_c,
                comfort_rho=comfort["rho_contact"],
                n1_rho=mpc_n1["rho_contact"],
                n10_rho=mpc_n10["rho_contact"],
                comfort_body_accel=comfort["body_accel"],
                n1_body_accel=mpc_n1["body_accel"],
                n10_body_accel=mpc_n10["body_accel"],
                n1_forces=mpc_n1["forces"],
                n10_forces=mpc_n10["forces"],
                n1_solve_time=mpc_n1["solve_time"],
                n10_solve_time=mpc_n10["solve_time"],
            )

    with (OUT_DIR / "table2_horizon_ablation.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    # Emit paste-ready table body
    print("\n=== LaTeX table body ===")
    for short in ("S7", "S8", "S9"):
        r = table_rows[short]
        print(
            f"{short} & {r['comfort_peak']} & {r['comfort_heave']} "
            f"& {r['n1_peak']} & {r['n1_heave']} "
            f"& {r['n10_peak']} & {r['n10_heave']} \\\\"
        )
    print(f"\nWrote {OUT_DIR}")


if __name__ == "__main__":
    main()

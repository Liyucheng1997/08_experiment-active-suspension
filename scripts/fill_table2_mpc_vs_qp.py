"""Fill Table II (Risk-MPC vs one-step Risk-QP) with S7/S8/S9 numbers.

Runs Comfort-QP, one-step Risk-QP (with STO + tightening), and Risk-MPC
(N_p=10) on the three high-risk scenarios from Phase 6. Computes the
peak ρ_max and heave RMS over the maneuver window and prints a paste-
ready LaTeX table.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT = Path(r"F:/我的科研工作/05_Risk_Aware_Active_Suspension")
sys.path.insert(0, str(PROJECT / "src"))

from risk_aware_active_suspension.controllers.comfort_qp import FullCarComfortQP
from risk_aware_active_suspension.controllers.risk_mpc import RESIDUAL_DECAY, FullCarRiskMPC
from risk_aware_active_suspension.controllers.risk_qp import FullCarRiskAwareQP, RiskQPParams
from risk_aware_active_suspension.metrics.signals import rmse
from risk_aware_active_suspension.metrics.tire import rho as rho_fn
from risk_aware_active_suspension.observers.sto import STO
from risk_aware_active_suspension.plants.full_car import FullCar
from risk_aware_active_suspension.scenarios.risk_mpc_residual_prediction import run_mpc_residual_closed_loop
from risk_aware_active_suspension.scenarios.risk_mpc_scenario_sweep import (
    PHASE6_SCENARIOS, _build_inputs, _risk_params_for,
)
from risk_aware_active_suspension.scenarios.risk_qp_sto_closed_loop import (
    STOPipeline, _phi_known_corner,
)
from risk_aware_active_suspension.scenarios.risk_qp_fullcar_ground_truth import _ground_truth_fz
from risk_aware_active_suspension.utils.config import from_yaml, lqr_from_yaml, observer_from_yaml


def _run_comfort(plant: FullCar, t, roads, a_x, a_y, f_c, mu, controller):
    n = len(t)
    dt = float(t[1] - t[0])
    states = np.zeros((n, 14)); forces = np.zeros((n, 4))
    fz_true = np.zeros((n, 4)); rho_c = np.zeros((n, 4))
    state = np.zeros(14)
    for idx in range(n - 1):
        fz_true[idx] = _ground_truth_fz(plant, state, roads[idx])
        rho_c[idx] = rho_fn(f_c[idx], 0.0, fz_true[idx], mu)
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
    return {"forces": forces, "rho_contact": rho_c, "body_accel": body_accel}


def _run_one_step_qp_with_sto(plant: FullCar, controller, t, roads, a_x, a_y, f_c,
                              mu, observer_params, fz_tightening_n: float,
                              controller_ts: float = 0.005):
    n = len(t)
    dt = float(t[1] - t[0])
    steps_per_update = max(1, int(round(controller_ts / dt)))
    sto = STOPipeline(observer_params=observer_params, plant=plant)
    states = np.zeros((n, 14)); forces = np.zeros((n, 4))
    fz_true = np.zeros((n, 4)); fz_hat = np.zeros((n, 4))
    rho_c = np.zeros((n, 4))
    state = np.zeros(14); u = np.zeros(4)
    for idx in range(n - 1):
        fz_true[idx] = _ground_truth_fz(plant, state, roads[idx])
        rho_c[idx] = rho_fn(f_c[idx], 0.0, fz_true[idx], mu)
        fz_hat[idx] = sto.step(state, u, dt)
        fz_safe = fz_hat[idx] - fz_tightening_n
        if idx % steps_per_update == 0:
            u = controller.compute(state, f_z_hat=fz_safe, f_c=f_c[idx], mu=mu)
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
    return {"forces": forces, "rho_contact": rho_c, "body_accel": body_accel,
            "states": states}


def main() -> None:
    vehicle = from_yaml(PROJECT / "configs/vehicle_default.yaml")
    lqr = lqr_from_yaml(PROJECT / "configs/controller_default.yaml")
    observer_params = observer_from_yaml(PROJECT / "configs/observer_default.yaml")
    plant = FullCar(vehicle)
    dt = 0.001
    target = {"S7_brake_cornering": "S7",
              "S8_worst_case":      "S8",
              "S9_corner_bump":     "S9"}

    table_rows: dict[str, dict[str, float | str]] = {}
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

        # --- One-step Risk-QP with STO ---
        try:
            one_step = _run_one_step_qp_with_sto(
                plant, FullCarRiskAwareQP(vehicle, lqr, risk, qp_params),
                t, roads, a_x, a_y, f_c, spec.mu, observer_params,
                fz_tightening_n=observer_params.delta_fz_err,
            )
            qp_finite = bool(np.all(np.isfinite(one_step["states"])))
        except Exception as exc:
            print(f"  Risk-QP raised: {exc}")
            one_step = None; qp_finite = False

        # --- Risk-MPC (N_p=10) ---
        mpc = run_mpc_residual_closed_loop(
            FullCarRiskMPC(vehicle, lqr, risk, qp_params, horizon=10,
                           warm_start=False, reuse_solver=True),
            plant, t, roads, a_y, f_c, spec.mu, observer_params,
            prediction_mode=RESIDUAL_DECAY, alpha=0.95,
            fz_tightening_n=observer_params.delta_fz_err, a_x=a_x,
        )

        # --- Metrics over the maneuver window (exclude first 0.5 s) ---
        win = (t >= 0.5) & (t <= t[-1])
        comfort_peak = float(np.max(comfort["rho_contact"][win].max(axis=1)))
        comfort_heave = float(rmse(comfort["body_accel"][win, 0]))
        if one_step is not None and qp_finite:
            qp_peak = float(np.max(one_step["rho_contact"][win].max(axis=1)))
            qp_heave = float(rmse(one_step["body_accel"][win, 0]))
            qp_peak_str = f"{qp_peak:.3f}"
            qp_heave_str = f"{qp_heave:.3f}"
        else:
            qp_peak_str = r"\textsc{Inf.}"
            qp_heave_str = r"\textsc{Inf.}"
        mpc_peak = float(np.max(mpc["rho_contact"][win].max(axis=1)))
        mpc_heave = float(rmse(mpc["body_accel"][win, 0]))

        print(f"  Comfort: peak={comfort_peak:.3f}  heave={comfort_heave:.3f}")
        print(f"  Risk-QP: peak={qp_peak_str}    heave={qp_heave_str}")
        print(f"  Risk-MPC: peak={mpc_peak:.3f}  heave={mpc_heave:.3f}")

        table_rows[short] = {
            "qp_peak": qp_peak_str, "qp_heave": qp_heave_str,
            "mpc_peak": f"{mpc_peak:.3f}", "mpc_heave": f"{mpc_heave:.3f}",
        }

    # Emit paste-ready table body
    print("\n=== LaTeX table body ===")
    for short in ("S7", "S8", "S9"):
        r = table_rows[short]
        print(f"{short} & {r['qp_peak']} & {r['qp_heave']} & {r['mpc_peak']} & {r['mpc_heave']} \\\\")


if __name__ == "__main__":
    main()

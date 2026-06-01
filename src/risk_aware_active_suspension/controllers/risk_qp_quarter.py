from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import osqp
import scipy.sparse as sp
from numpy.typing import ArrayLike
from scipy.linalg import solve_discrete_are

from risk_aware_active_suspension.controllers.lqr import _zoh_discretize
from risk_aware_active_suspension.controllers.risk_weights import (
    RiskWeightParams,
    compute_all as compute_risk_weights,
)
from risk_aware_active_suspension.metrics.tire import f_z_required, rho
from risk_aware_active_suspension.plants.quarter_car import QuarterCar
from risk_aware_active_suspension.utils.config import LQRParams, VehicleParams


@dataclass(frozen=True)
class QuarterRiskQPParams:
    """Knobs for the quarter-car risk-aware QP.

    gamma is the per-step actuator-to-normal-load transmission ratio used in
    the margin constraint  F_z_true + gamma * F_e + xi >= F_z_required.

    For a pure quarter car the dynamic (single-step) ∂F_z/∂F_e ~ k_t T_s^2 / (2 m_u)
    is small; gamma is therefore exposed as a configurable parameter. Phase 4.5
    derives an effective gamma from the full-car body-roll geometry; here the
    default 1.0 treats F_e as an idealized direct authority over F_z and lets
    us isolate the margin-protection logic.
    """

    gamma: float = 1.0
    df_max: float = 1.0e6  # effectively off by default
    r_u_factor: float = 0.0
    r_du_factor: float = 0.0
    enforce_rate: bool = False
    xi_regularization: float = 1.0e-6
    rho_safe: float = 0.85
    osqp_eps_abs: float = 1.0e-8
    osqp_eps_rel: float = 1.0e-8
    osqp_max_iter: int = 20000

    def __post_init__(self) -> None:
        if self.df_max <= 0.0:
            raise ValueError("df_max must be positive.")
        if self.r_u_factor < 0.0 or self.r_du_factor < 0.0:
            raise ValueError("r_u_factor and r_du_factor must be non-negative.")
        if self.xi_regularization <= 0.0:
            raise ValueError("xi_regularization must be positive.")
        if not (0.0 < self.rho_safe <= 1.0):
            raise ValueError("rho_safe must be in (0, 1].")


def _build_quarter_comfort_matrices(
    vehicle: VehicleParams, lqr: LQRParams
) -> dict[str, np.ndarray]:
    plant = QuarterCar(vehicle)
    a_d, b_d = _zoh_discretize(plant.A, plant.B, lqr.T_s)
    c = plant.A[1:2, :]
    d = plant.B[1:2, :]
    q_y = np.array([[lqr.q_accel]], dtype=float)
    r = np.array([[lqr.r_force]], dtype=float)
    s_cross = c.T @ q_y @ d
    r_eff = r + d.T @ q_y @ d
    p_dare = solve_discrete_are(a_d, b_d, c.T @ q_y @ c, r_eff, s=s_cross)
    return {
        "M": r_eff + b_d.T @ p_dare @ b_d,
        "L_x": s_cross.T + b_d.T @ p_dare @ a_d,
        "T_s": lqr.T_s,
    }


@dataclass
class QuarterCarRiskAwareQP:
    """Quarter-car risk-aware QP with margin constraint.

    Decision z = [u_s (1); xi (1)] ∈ R^2.

    Cost (identical structure to Phase 4.3, scaled scalars for one wheel):
        J = q_c(σ) * ||G_a u + a_0||^2_{Q_a} + q_{p}(σ, ρ) * xi^2
            + r_u_factor * u^2 + r_du_factor * (u - u_prev)^2

    Constraints:
        |u_s| ≤ f_max
        xi ≥ 0
        |u_s - u_prev| ≤ df_max * T_s        (optional)
        F_z_true + gamma * u_s + xi ≥ F_z_required  (margin — the new piece)
    """

    vehicle: VehicleParams
    lqr: LQRParams
    risk: RiskWeightParams
    qp: QuarterRiskQPParams = field(default_factory=QuarterRiskQPParams)
    name: str = "risk_qp_quarter"
    _M: np.ndarray = field(init=False, repr=False)
    _L_x: np.ndarray = field(init=False, repr=False)
    _prob: osqp.OSQP = field(init=False, repr=False)
    _u_prev: float = field(init=False, default=0.0)
    last_solve_time_s: float = field(init=False, default=0.0)
    last_sigma: float = field(init=False, default=0.0)
    last_q_c: float = field(init=False, default=1.0)
    last_q_p_local: float = field(init=False, default=0.0)
    last_xi: float = field(init=False, default=0.0)
    last_rho: float = field(init=False, default=0.0)
    last_f_z_required: float = field(init=False, default=0.0)
    last_status: str = field(init=False, default="not_solved")
    last_margin_active: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        comfort = _build_quarter_comfort_matrices(self.vehicle, self.lqr)
        object.__setattr__(self, "_M", comfort["M"])
        object.__setattr__(self, "_L_x", comfort["L_x"])

        H_uu0 = 2.0 * (
            self.risk.q_c_max * float(self._M[0, 0])
            + self.qp.r_u_factor
            + self.qp.r_du_factor
        )
        H_xixi0 = 2.0 * self.risk.q_p_min + self.qp.xi_regularization
        H_full = np.array([[H_uu0, 0.0], [0.0, H_xixi0]], dtype=float)
        P_template = sp.csc_matrix(np.triu(H_full))

        # A_cons rows: [u-box; xi-box; rate; margin]
        gamma = self.qp.gamma
        A_cons = sp.csc_matrix(
            np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [1.0, 0.0],
                    [gamma, 1.0],
                ],
                dtype=float,
            )
        )
        f_max = self.lqr.f_max
        rate_amp = self.qp.df_max * self.lqr.T_s if self.qp.enforce_rate else 1.0e12
        BIG = 1.0e12
        l_init = np.array([-f_max, 0.0, -rate_amp, -BIG], dtype=float)
        u_init = np.array([f_max, BIG, rate_amp, BIG], dtype=float)

        prob = osqp.OSQP()
        prob.setup(
            P=P_template,
            q=np.zeros(2),
            A=A_cons,
            l=l_init,
            u=u_init,
            verbose=False,
            warm_starting=True,
            polishing=False,
            eps_abs=self.qp.osqp_eps_abs,
            eps_rel=self.qp.osqp_eps_rel,
            max_iter=self.qp.osqp_max_iter,
        )
        object.__setattr__(self, "_prob", prob)

    def reset(self) -> None:
        self._u_prev = 0.0

    def compute(
        self,
        state: ArrayLike,
        f_z_true: float,
        f_c: float,
        mu: float,
        *_: Any,
        **__: Any,
    ) -> np.ndarray:
        x = np.asarray(state, dtype=float)
        if x.shape != (4,):
            raise ValueError(f"quarter-car state must have shape (4,), got {x.shape}.")
        if mu <= 0.0:
            raise ValueError("mu must be positive.")
        f_z_req = float(f_z_required(float(f_c), mu, self.qp.rho_safe))
        current_rho = float(rho(float(f_c), 0.0, float(f_z_true), mu))

        # Per-wheel risk weights (rho_ij collapses to one wheel).
        weights = compute_risk_weights(current_rho, np.array([current_rho]), self.risk)
        sigma_val = float(weights["sigma"])
        q_c_val = float(weights["q_c"])
        q_p_loc = float(np.asarray(weights["q_p_local"]).item())

        # Assemble Hessian (diagonal 2x2, upper triangular CSC).
        H_uu = 2.0 * (
            q_c_val * float(self._M[0, 0])
            + self.qp.r_u_factor
            + self.qp.r_du_factor
        )
        H_xixi = 2.0 * q_p_loc + self.qp.xi_regularization
        P_new = sp.csc_matrix(np.array([[H_uu, 0.0], [0.0, H_xixi]], dtype=float))

        # Linear term: q_u = 2 q_c L_x x − 2 r_du u_prev
        q_u = 2.0 * q_c_val * float(self._L_x[0, :] @ x) - 2.0 * self.qp.r_du_factor * self._u_prev
        q_lin = np.array([q_u, 0.0], dtype=float)

        # Bounds. Margin row: gamma * u + xi >= f_z_req - f_z_true.
        f_max = self.lqr.f_max
        rate_amp = self.qp.df_max * self.lqr.T_s if self.qp.enforce_rate else 1.0e12
        BIG = 1.0e12
        margin_lower = float(f_z_req - f_z_true)
        l_new = np.array(
            [-f_max, 0.0, self._u_prev - rate_amp, margin_lower], dtype=float
        )
        u_new = np.array(
            [f_max, BIG, self._u_prev + rate_amp, BIG], dtype=float
        )

        self._prob.update(Px=P_new.data, q=q_lin, l=l_new, u=u_new)
        t0 = time.perf_counter()
        res = self._prob.solve(raise_error=False)
        self.last_solve_time_s = time.perf_counter() - t0
        status = str(getattr(res.info, "status", "unknown"))
        self.last_status = status
        if status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"Quarter risk-QP OSQP failed: status={status}")

        z_sol = np.asarray(res.x, dtype=float)
        u_s = float(z_sol[0])
        xi = float(z_sol[1])
        self.last_sigma = sigma_val
        self.last_q_c = q_c_val
        self.last_q_p_local = q_p_loc
        self.last_xi = xi
        self.last_rho = current_rho
        self.last_f_z_required = f_z_req
        self.last_margin_active = margin_lower > -BIG / 2.0 and (
            self.qp.gamma * u_s + xi >= margin_lower - 1.0e-3
        )
        self._u_prev = u_s
        return np.array([u_s], dtype=float)
